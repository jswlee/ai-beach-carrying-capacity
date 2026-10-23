#!/usr/bin/env python3
"""Export a YOLO .pt model to .onnx (e.g. to compare p2 .pt vs .onnx latency).

Usage:
    python export_onnx.py path/to/best.pt
    python export_onnx.py path/to/best.pt --imgsz 640 --simplify --half
"""

import argparse

from ultralytics import YOLO

from utils import patch_torch_load


def main() -> None:
    patch_torch_load()

    p = argparse.ArgumentParser(description="Export a YOLO .pt model to .onnx")
    p.add_argument("model", help="Path to .pt model")
    p.add_argument("--imgsz", type=int, default=640, help="Export image size (match tile size)")
    p.add_argument("--half", action="store_true", help="Export FP16 weights")
    p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph (onnxslim)")
    p.add_argument("--static", action="store_true", help="Fixed batch=1 (default is dynamic batch, needed for tiled inference)")
    args = p.parse_args()

    out = YOLO(args.model).export(
        format="onnx", imgsz=args.imgsz, half=args.half,
        simplify=args.simplify, dynamic=not args.static,
    )
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
