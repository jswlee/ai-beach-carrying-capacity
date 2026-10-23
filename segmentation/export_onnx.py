"""Export a segmentation model dir to ONNX (static square input).

Works for both backends:
  - HF SegFormer dir  (config.json)         -> default imgsz 512
  - DinoSeg dir       (dinoseg_meta.json)   -> default imgsz = train size

Output: <model_dir>/model.onnx  (picked up automatically by
test_viz_segformer.py when running on CPU).

Usage:
    python export_onnx.py trained_models/segformer-waterline-v11i/best_model
    python export_onnx.py trained_models/dinov3-waterline-v11i
"""
import argparse
import os

import torch
import torch.nn as nn

from dinoseg import is_dinoseg_dir, load_dinoseg


class _LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Model dir (SegFormer HF dir or DinoSeg dir)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Square input size (default: dinoseg train size, else 512)")
    ap.add_argument("--out", default=None, help="default <model_dir>/model.onnx")
    args = ap.parse_args()

    if is_dinoseg_dir(args.model):
        model, train_imgsz = load_dinoseg(args.model, "cpu")
        imgsz = args.imgsz or train_imgsz
    else:
        from transformers import AutoModelForSemanticSegmentation
        model = AutoModelForSemanticSegmentation.from_pretrained(args.model).eval()
        imgsz = args.imgsz or 512

    out = args.out or os.path.join(args.model, "model.onnx")
    dummy = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        _LogitsOnly(model), (dummy,), out,
        input_names=["pixel_values"], output_names=["logits"],
        opset_version=18, do_constant_folding=True,
    )
    print(f"wrote {out}  (input {imgsz}x{imgsz})")


if __name__ == "__main__":
    main()
