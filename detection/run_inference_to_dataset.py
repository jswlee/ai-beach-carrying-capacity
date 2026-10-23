#!/usr/bin/env python3
"""
Run YOLO inference on a directory of images and export predictions
as a YOLOv8-formatted dataset for further training or analysis.

- Loads a trained .pt model
- Runs inference on every image in a source directory (these should already be ROI-masked)
- Writes outputs under:

  inference/
    <timestamp>/
      visualized/           # images with color-coded labels (Ultralytics predict output)
      data_for_yolo/
        images/             # copies of the source images
        labels/             # YOLOv8 txt labels from predictions
        data.yaml           # minimal dataset config for YOLOv8

Usage example:

  python run_inference_to_dataset.py \
    --model training_results/models/beach_detection_20251125_133123.pt \
    --source processed_images_ready_for_training_person/test/images \
    --img-size 1920 --conf 0.25
"""

from pathlib import Path
import argparse
from datetime import datetime
import shutil
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Fix for PyTorch 2.6+ weights_only=True default (same as training script)
print("Setting torch.load to use weights_only=False for YOLO model loading")
_original_load = torch.load

def patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = patched_load


# Add preprocessing directory to path so we can reuse the ROI mask logic
ROOT_DIR = Path(__file__).parent
PREPROC_DIR = ROOT_DIR / "preprocessing"
if PREPROC_DIR.exists():
    sys.path.append(str(PREPROC_DIR))

from apply_roi_mask import create_roi_mask, apply_mask_to_image


def select_device() -> str:
    """Return best available device string for YOLO (cuda, mps, cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_image_map(source: Path) -> dict[str, Path]:
    """Map image stem -> image path for all images in source directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    mapping: dict[str, Path] = {}
    for p in source.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.stem] = p
    return mapping


def write_data_yaml(out_dir: Path, model: YOLO) -> None:
    """Write a minimal YOLOv8 data.yaml for the predicted dataset."""
    names = getattr(model, "names", None)
    if not names:
        # Fallback: no class names known
        return

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: images\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for cls_id, cls_name in names.items():
            f.write(f"  {cls_id}: {cls_name}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on a directory and export predictions as a YOLOv8 dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="training_results/models/beach_detection_20251125_133123.pt",
        help="Path to trained YOLO .pt model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="C:/Users/JR/Documents/GitHub/beach-monitor/training_data/train_object_detection/images",
        help="Directory of RAW source images (mask will be applied before inference)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1920,
        help="Image size for inference",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    source_dir = Path(args.source).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"❌ Source directory not found or is not a directory: {source_dir}")
        return 1

    device = select_device()
    print("YOLO Inference -> Dataset Export")
    print("=" * 80)
    print(f"Model:       {model_path}")
    print(f"Source dir:  {source_dir}")
    print(f"Image size:  {args.img_size}")
    print(f"Conf thresh: {args.conf}")
    print(f"Batch size:  {args.batch}")
    print(f"Device:      {device}")
    print()

    # Root inference directory and timestamped run dir
    inference_root = Path("inference")
    inference_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = inference_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    visualized_dir = run_dir / "visualized"
    data_for_yolo_dir = run_dir / "data_for_yolo"
    images_out = data_for_yolo_dir / "images"
    labels_out = data_for_yolo_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Directory where masked images will be written before inference
    masked_input_dir = run_dir / "masked_input"
    masked_input_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Apply ROI mask to all source images
    # ------------------------------------------------------------------
    print("Applying ROI mask to source images...")
    print("-" * 80)
    img_map_raw = build_image_map(source_dir)
    num_masked = 0
    for stem, img_path in img_map_raw.items():
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        mask = create_roi_mask(img.shape)
        masked_img = apply_mask_to_image(img, mask)

        out_path = masked_input_dir / img_path.name
        cv2.imwrite(str(out_path), masked_img)
        num_masked += 1

    print(f"✓ Masked {num_masked} images into: {masked_input_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Run prediction on MASKED images
    # ------------------------------------------------------------------
    print("Loading model...")
    model = YOLO(str(model_path))
    print()

    print("Running inference on masked images...")
    print("-" * 80)
    _ = model.predict(
        source=str(masked_input_dir),
        imgsz=args.img_size,
        conf=args.conf,
        device=device,
        batch=args.batch,
        project=str(run_dir),
        name="visualized",
        save=True,      # annotated images with colored labels
        save_txt=True,  # YOLO-format prediction txts
        verbose=True,
    )

    print()
    print("Collecting predictions into YOLOv8 dataset structure...")
    print("-" * 80)

    labels_src = visualized_dir / "labels"
    if not labels_src.exists():
        print(f"❌ Expected labels directory not found: {labels_src}")
        return 1

    # Map image stems to MASKED image paths (these are what we trained on / want to export)
    img_map = build_image_map(masked_input_dir)

    num_labels = 0
    for label_file in sorted(labels_src.glob("*.txt")):
        stem = label_file.stem
        src_img = img_map.get(stem)
        if src_img is None:
            # No matching source image; skip
            continue

        # Copy source image and label file into data_for_yolo
        shutil.copy2(src_img, images_out / src_img.name)
        shutil.copy2(label_file, labels_out / label_file.name)
        num_labels += 1

    # Write data.yaml using model's class names
    write_data_yaml(data_for_yolo_dir, model)

    print(f"✓ Copied {num_labels} labeled images into: {data_for_yolo_dir}")
    print()
    print("Directory layout:")
    print(f"  Visualized predictions: {visualized_dir}")
    print(f"  YOLO dataset images:    {images_out}")
    print(f"  YOLO dataset labels:    {labels_out}")
    print(f"  YOLO data.yaml:         {data_for_yolo_dir / 'data.yaml'}")
    print()
    print("=" * 80)
    print("✓ Inference-to-dataset export complete!")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
