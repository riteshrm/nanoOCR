import os
import argparse
import torch
from src.det_model.model import BasicModel
from src.utils.utils import get_latest_checkpoint
import torch
from torch.export import Dim



def run(args):
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path, _ = get_latest_checkpoint(args.ckpt_dir, "ckpt-")
    if ckpt_path is None:
        print("No valid checkpoints found")
        return
    state_dict = torch.load(
        f"{ckpt_path}/pytorch_model.bin",
        map_location="cpu",
        weights_only=True
    )

    model = BasicModel()
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512)
    print("Exporting model")

    torch.onnx.export(
    model,
    (dummy_input,),
    "det_model_fp32.onnx",
    input_names=["images"],
    output_names=["pred"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "images": {
            2: "height",
            3: "width"
        },
        "predictions": {
            2: "out_height",
            3: "out_width"
        }
    }
)

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default="det_checkpoints")

    args = parser.parse_args()

    run(args)

if __name__== "__main__":
    main()