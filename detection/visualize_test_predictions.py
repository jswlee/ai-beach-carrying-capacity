#!/usr/bin/env python3
"""Visualize YOLO predictions on the test set.

This script:
  - Loads a trained YOLO model
  - Reads the dataset YAML to locate the test images
  - Runs `model.predict` on the test images
  - Saves annotated images and shows internal visualizations (if supported)

Usage example:

    python visualize_test_predictions.py \
        --model training_results/models/beach_detection_20260104_101723_best.pt \
        --data processed_data/processed_images_v13/data.yaml \
        --img-size 1920 \
        --conf 0.20
"""

import argparse
from pathlib import Path
from typing import List

import yaml
from ultralytics import YOLO


def load_test_image_dir(data_yaml: Path) -> Path:
    with data_yaml.open("r") as f:
        data = yaml.safe_load(f)

    test_entry = data.get("test")
    base_path = data.get("path", None)

    if test_entry is None:
        raise ValueError(f"No 'test' key found in data.yaml: {data_yaml}")

    test_path = Path(test_entry)
    if not test_path.is_absolute():
        if base_path is not None:
            test_path = Path(base_path) / test_path
        else:
            test_path = data_yaml.parent / test_path

    images_dir = test_path
    # If user pointed 'test' at the split root, assume 'images' subdir
    if images_dir.is_dir() and images_dir.name != "images":
        candidate = images_dir / "images"
        if candidate.exists():
            images_dir = candidate

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")

    return images_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize YOLO test predictions.")
    parser.add_argument("--model", type=str, default="training_results/models/beach_detection_20260104_101723_best.pt", help="Path to trained YOLO .pt model")
    parser.add_argument("--data", type=str, default="processed_data/processed_images_v13/data.yaml", help="Path to dataset YAML (must define test split)")
    parser.add_argument("--img-size", type=int, default=1920, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve()

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    if not data_yaml.exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Reading test images from: {data_yaml}")
    test_images_dir = load_test_image_dir(data_yaml)
    print(f"Test images directory: {test_images_dir}")

    print("Running visualize predictions on test images...")
    print("-" * 80)

    # Ultralytics will create a runs/predict directory under the current cwd
    model.predict(
        source=str(test_images_dir),
        imgsz=args.img_size,
        conf=args.conf,
        save=True,        # save annotated images
        visualize=True,   # internal visualizations
        verbose=True,
    )

    print("\nVisualization run complete. Check the 'runs/predict*' directory for outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
