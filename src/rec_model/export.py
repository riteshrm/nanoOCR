import os
import argparse
import torch
from src.rec_model.model import CRNN2
from src.utils.utils import get_latest_checkpoint
import json

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from src.data.dataset import RECDataset, rec_collate
from torch.utils.data import DataLoader

class RECDataReader(CalibrationDataReader):
    def __init__(self, data_path: str, model_path: str):
        self.enum_data = None

        # Use inference session to get the model input name.
        session = ort.InferenceSession(model_path, None)
        dataset = RECDataset(data_path, 'val.csv')
        val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=rec_collate)
        samples = []
        for x, _, _ in val_dataloader:
            x = x.cpu().numpy()
            samples.append(x)
            break

        self.nchw_data_list = samples
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nchw_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                {self.input_name: sample} for sample in self.nchw_data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

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
    vocab = torch.load(os.path.join(args.data_path,'vocab.pt'))['vocab']
    itoc = torch.load(os.path.join(args.data_path,'vocab.pt'))['itoc']
    with open(f'{args.data_path}/itoc.json', 'w') as f:
        json.dump(itoc, f)

    model = CRNN2(num_classes=len(vocab) + 1)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 1, 32, 128)

    torch.onnx.export(
        model,
        (dummy,),
        f"{args.save_dir}/rec_model_fp32.onnx",
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        verify=True,
    )

    quant_pre_process(f"{args.save_dir}/rec_model_fp32.onnx", f"{args.save_dir}/rec_model_opt_fp32.onnx", skip_symbolic_shape=True,)
    dr = RECDataReader(
            args.data_path, f"{args.save_dir}/rec_model_opt_fp32.onnx"
    )
    
    quantize_static(
        f"{args.save_dir}/rec_model_opt_fp32.onnx",
        f"{args.save_dir}/rec_model_int8.onnx",
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = (
        f"{args.save_dir}/rec_model_int8_cpu_optimized.onnx"
    )

    session = ort.InferenceSession(
        f"{args.save_dir}/rec_model_int8.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    
def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default="rec_checkpoints")
    parser.add_argument('--data_path', type=str, default="rec_data/processed")
    parser.add_argument('--save_dir', type=str, default="inference_checkpoints")

    args = parser.parse_args()

    run(args)

if __name__== "__main__":
    main()