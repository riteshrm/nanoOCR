import os
import torch
import argparse
import time
import numpy as np
from src.data.dataset import DETValDataset, det_val_collate
from src.det_model.model import BasicModel
from src.utils.utils import get_latest_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.eval.seg_detector_representer import SegDetectorRepresenter
from src.eval.iou import DetectionIoUEvaluator
import onnxruntime as ort

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
    test_dataset = DETValDataset(os.path.join(args.data_path, 'processed_test'))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=det_val_collate)

    representer = SegDetectorRepresenter()
    evaluator = DetectionIoUEvaluator()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model = ort.InferenceSession(
                    f"{args.ckpt_dir}/det_model.onnx",
                    sess_options=options,
                    providers=["CPUExecutionProvider"])

    latencies = []
    for _ in range(20):
        model.run(["pred"], {"images": np.zeros((1, 3, 512, 512), dtype=np.float32)})

    results = []
    for batch in tqdm(test_dataloader):
        x = batch['image'].cpu().numpy()
        start = time.perf_counter()
        pred = model.run(["pred"],
                        {"images": x})[0]
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        out = torch.from_numpy(pred)
        validate(representer, evaluator, batch, out, results)
    test_metrics = evaluator.combine_results(results)

    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['recall']:.3f}")
    print(f"hmean {test_metrics['hmean']:.3f}")
    print(f"Mean latency:   {np.mean(latencies):.3f} ms")
    print(f"Median latency: {np.median(latencies):.3f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.3f} ms")

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, default='det_data')
    parser.add_argument("--ckpt_dir", type=str, default="inference_checkpoints")

    args = parser.parse_args()
    
    run(args=args)

if __name__ == "__main__":
    main()