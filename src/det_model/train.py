import os
import torch.nn as nn
from src.det_model.model import BasicModel
from src.det_model.loss import L1BalanceCELoss
from src.data.dataset import DETDataset, det_collate, DETValDataset, det_val_collate
from src.eval.seg_detector_representer import SegDetectorRepresenter
from src.eval.iou import DetectionIoUEvaluator

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from src.utils.utils import ctc_decoder, cer_counts, cleanup_checkpoints, get_latest_checkpoint
from tqdm import tqdm
from datetime import datetime

def place_batch_on_device(batch, device):
    for key, value in batch.items():
        batch[key] = value.to(device)

def validate(representer, evaluator, batch, out, results):
    boxes_batch, scores_batch = representer.represent(batch, out)


    for polygons, ignore_tags, boxes, scores in zip(batch["polygons"],batch["ignore_tags"],boxes_batch,scores_batch,):
        gt = [
            {
                "points": np.asarray(polygon.detach().cpu()).tolist(),
                "ignore": bool(ignore),
            }
            for polygon, ignore in zip(polygons, ignore_tags)
        ]

        predictions = [
            {"points": box.tolist()}
            for box, score in zip(boxes, scores)
            if score >= representer.box_thresh
        ]

        results.append(evaluator.evaluate_image(gt, predictions))
def run(args):
    os.environ["TRACKIO_DIR"] = args.log_dir
    import trackio
    run_name = f"{args.project_name}_{datetime.now():%Y%m%d_%H%M%S}"
    trackio.init(
        project=args.project_name,
        name=run_name,
        config={"steps": args.num_epochs, "learning_rate":args.learning_rate, "batch_size": args.batch_size}
    )
    accelerator  = Accelerator()
    train_dataset = DETDataset(os.path.join(args.data_path, 'processed_train'))
    val_dataset = DETValDataset(os.path.join(args.data_path, 'processed_val'))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, collate_fn=det_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=det_val_collate)

    representer = SegDetectorRepresenter()
    evaluator = DetectionIoUEvaluator()
    
    model = BasicModel().to(device)
    criterion = L1BalanceCELoss()  # ignores samples whose label is too long for T
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.num_epochs * len(train_dataloader))

    model, train_dataloader, val_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer, scheduler
    )
    latest_checkpoint, l_ep = get_latest_checkpoint(args.ckpt_dir, 'ckpt-')
    best_checkpoint, b_ep = get_latest_checkpoint(args.ckpt_dir, 'best-ckpt-')
    start_epoch=0
    if args.resume=='True':
        if best_checkpoint is not None:
            start_epoch = b_ep
            accelerator.load_state(best_checkpoint)
            accelerator.print(f"Resuming from: {best_checkpoint}")
        elif latest_checkpoint is not None:
            start_epoch = l_ep
            accelerator.load_state(latest_checkpoint)
            accelerator.print(f"Resuming from: {latest_checkpoint}")
        else:
            accelerator.print("No checkpoint found, starting from scratch")

    best_hmean = -1
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_bce_loss = 0.0
        total_train_l1_loss = 0.0
        total_train_dice_loss = 0.0
        for batch in tqdm(train_dataloader):
            place_batch_on_device(batch, device)
            x = batch['image']
            with accelerator.accumulate(model):
                out = model(x)
                loss, loss_metric = criterion(out, batch)
                optimizer.zero_grad()
                accelerator.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            total_train_loss += loss.item()
            total_train_bce_loss += loss_metric['bce_loss'].item()
            total_train_l1_loss +=loss_metric['l1_loss'].item()
            total_train_dice_loss +=loss_metric['thresh_loss'].item()

        model.eval()
        with torch.no_grad():
            results = []
            for batch in val_dataloader:
                x = batch['image']
                out = model(x.to(device)).to('cpu')
                validate(representer, evaluator, batch, out, results)
            val_metrics = evaluator.combine_results(results)
        metrics = {
                "train_loss": total_train_loss / len(train_dataloader),
                "train_bce_loss": total_train_bce_loss / len(train_dataloader),
                "train_l1_loss": total_train_l1_loss / len(train_dataloader),
                "train_thresh_loss": total_train_dice_loss / len(train_dataloader),
            }
        metrics.update(val_metrics)
        trackio.log(metrics, step=epoch+1)
        if accelerator.is_main_process:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f'ckpt-{epoch}')
            accelerator.save_state(ckpt_path, safe_serialization=False)
            cleanup_checkpoints(args.ckpt_dir, 'ckpt')

        accelerator.print(f"epoch {epoch + 1:02d}  train loss {total_train_loss / len(train_dataloader):.3f}  val precision {val_metrics['precision']:.3f} val recall {val_metrics['recall']:.3f} val hmean {val_metrics['hmean']:.3f}")
        if val_metrics["hmean"] > best_hmean:
            best_hmean  = val_metrics["hmean"]
            if accelerator.is_main_process:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt_dir, f'best-ckpt-{epoch}')
                accelerator.save_state(ckpt_path, safe_serialization=False)
                cleanup_checkpoints(args.ckpt_dir, 'best-ckpt')
        
def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default='det_data')
    parser.add_argument("--project_name", type=str, default='DBNet')
    parser.add_argument("--ckpt_dir", type=str, default='det_checkpoints')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--log_dir", type=str, default='det_train_logs')

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()