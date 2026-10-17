import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from src.rec_model.model import CRNN
from src.data.dataset import RECDataset, rec_collate
from src.utils.utils import ctc_decoder, cer_counts, cleanup_checkpoints, get_latest_checkpoint
from tqdm import tqdm
from datetime import datetime
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
    train_dataset = RECDataset(args.data_path, 'train.csv')
    val_dataset = RECDataset(args.data_path, 'val.csv')

    data = torch.load(f"{args.data_path}/vocab.pt", weights_only=False)

    vocab = data["vocab"]
    itoc = data["itoc"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, collate_fn=rec_collate, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=rec_collate, num_workers=8)

    
    model = CRNN(len(vocab)+1, args.height).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # ignores samples whose label is too long for T
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

    best_cer = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0.0
        for x, targets, target_lengths in tqdm(train_dataloader):
            x = x.to(device)
            with accelerator.accumulate(model):
                log_probs = model(x).log_softmax(-1)
                input_lengths = torch.tensor([log_probs.shape[0]] * log_probs.shape[1], dtype=torch.int64)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                optimizer.zero_grad()
                accelerator.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            total_train_loss += loss.item()

        model.eval()
        total_errors = total_chars = 0
        total_val_loss=0
        with torch.no_grad():
            for x, targets, target_lengths in val_dataloader:
                x = x.to(device)
                log_probs = model(x).log_softmax(-1)
                input_lengths = torch.tensor([log_probs.shape[0]] * log_probs.shape[1], dtype=torch.int64)
                val_loss = criterion(log_probs, targets, input_lengths, target_lengths)
                pred_texts, pred_labels = ctc_decoder(log_probs.permute(1, 0, 2), itoc)

                errors, chars = cer_counts(
                    pred_labels, targets, target_lengths
                )
                total_errors += errors
                total_chars += chars
                total_val_loss+=val_loss.item()
        metrics = {
                "train_loss": total_train_loss / len(train_dataloader),
                "val_loss": total_val_loss / len(val_dataloader),
                "val CER": total_errors / total_chars
            }
        trackio.log(metrics, step=epoch+1)
        if accelerator.is_main_process:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f'ckpt-{epoch}')
            accelerator.save_state(ckpt_path, safe_serialization=False)
            cleanup_checkpoints(args.ckpt_dir, 'ckpt')

        cer = total_errors / total_chars
        accelerator.print(f"epoch {epoch + 1:02d}  train loss {total_train_loss / len(train_dataloader):.3f}  val loss {total_val_loss / len(val_dataloader):.3f} val CER {cer:.3f}")
        if cer < best_cer :
            best_cer  = cer
            if accelerator.is_main_process:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt_dir, f'best-ckpt-{epoch}')
                accelerator.save_state(ckpt_path, safe_serialization=False)
                cleanup_checkpoints(args.ckpt_dir, 'best-ckpt')
        
def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default='rec_data/processed')
    parser.add_argument("--project_name", type=str, default='CRNN_CTC_1')
    parser.add_argument("--ckpt_dir", type=str, default='rec_checkpoints_1')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--log_dir", type=str, default='rec_train_logs_1')

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()