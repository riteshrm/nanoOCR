import os
import argparse
import torch
from src.rec_model.model import CRNN
from src.utils.utils import get_latest_checkpoint
import json

def run(args):
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path, _ = get_latest_checkpoint(args.ckpt_dir, "best-ckpt-")
    if ckpt_path is None:
        print("No valid checkpoints found")
        return
    state_dict = torch.load(
        f"{ckpt_path}/pytorch_model.bin",
        map_location="cpu",
        weights_only=True
    )
    vocab = torch.load(os.path.join(args.save_path,'vocab.pt'))['vocab']
    itoc = torch.load(os.path.join(args.save_path,'vocab.pt'))['itoc']
    with open(f'{args.save_path}/itoc.json', 'w') as f:
        json.dump(itoc, f)

    model = CRNN(num_classes=len(vocab) + 1, height=32)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 1, 32, 128)

    torch.onnx.export(
        model,
        (dummy,),
        "rec_model_fp32.onnx",
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        verify=True,
    )

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default="rec_checkpoints")
    parser.add_argument('--save_path', type=str, default="rec_data/processed")

    args = parser.parse_args()

    run(args)

if __name__== "__main__":
    main()