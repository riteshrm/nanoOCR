import os
import torch
import argparse
import numpy as np
from src.data.dataset import DETValDataset, det_val_collate
from src.det_model.model import BasicModel
from src.utils.utils import get_latest_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.eval.seg_detector_representer import SegDetectorRepresenter
from src.eval.iou import DetectionIoUEvaluator

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
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        path, _ = get_latest_checkpoint(args.ckpt_dir, "best-ckpt-")
        ckpt_path = f'{path}/pytorch_model.bin'
    if ckpt_path is None:
        print("No valid checkpoints found")
        return
    print("Loaded model from: ", ckpt_path)

    test_dataset = DETValDataset(os.path.join(args.data_path, 'processed_test'))


    device = "cuda" if torch.cuda.is_available() else "cpu"


    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=det_val_collate)

    representer = SegDetectorRepresenter()
    evaluator = DetectionIoUEvaluator()

    model = BasicModel().to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    totals = {
        "errors": 0,
        "matches": 0,
        "predicted_chars": 0,
        "target_chars": 0,
        "exact_words": 0,
        "words": 0,
    }
    with torch.no_grad():
        results = []
        for batch in tqdm(test_dataloader):
            x = batch['image']
            out = model(x.to(device)).to('cpu')
            validate(representer, evaluator, batch, out, results)
        test_metrics = evaluator.combine_results(results)

    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['precision']:.3f}")
    print(f"hmean {test_metrics['hmean']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, default='det_data')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default='det_checkpoints')

    args = parser.parse_args()
    
    run(args=args)

if __name__ == "__main__":
    main()