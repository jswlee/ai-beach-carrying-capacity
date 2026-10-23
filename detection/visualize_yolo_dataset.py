#!/usr/bin/env python3
r"""Visualize YOLO labels on top of images in a dataset directory.

Reads images + labels from a YOLOv8-style dataset layout:
  <input_dir>/{train,valid,test}/images/*
  <input_dir>/{train,valid,test}/labels/*

Confidence scores are read from a consolidated <input-dir>/detections.csv
(produced by tiled_inference_to_dataset.py). If the CSV is not found, falls
back to confidence=1.0 for all boxes.

Draws each label box (class + conf) on the image and writes the annotated
image to a mirrored structure under  <input_dir>_viz/.

Usage:
  python visualize_yolo_dataset.py
  python visualize_yolo_dataset.py --input-dir "C:\Users\JR\Downloads\Kaanapali_yolo"
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

# Reuse the drawing helper from run_tiled_inference
ROOT_DIR = Path(__file__).parent
TRAINING_DIR = ROOT_DIR / "training"
if TRAINING_DIR.exists() and str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from run_tiled_inference import draw_dets, Det  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_confidence_csv(csv_path: Path) -> dict[str, list[tuple[int, float, float, float, float, float]]]:
    """Load detections.csv into a map: image_filename -> list of (cls, cx, cy, bw, bh, conf)."""
    if not csv_path.exists():
        return {}
    result: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image"]
            try:
                entry = (
                    int(row["class_id"]),
                    float(row["cx"]), float(row["cy"]),
                    float(row["bw"]), float(row["bh"]),
                    float(row["conf"]),
                )
            except (ValueError, KeyError):
                continue
            result.setdefault(img_name, []).append(entry)
    return result


def load_label_dets(
    label_path: Path,
    img_w: int,
    img_h: int,
    conf_lookup: list[tuple[int, float, float, float, float, float]] | None = None,
) -> list[Det]:
    """Parse a YOLO-format label file into Det objects in pixel coords.

    If conf_lookup is provided (from detections.csv), uses the confidence
    values from there instead of defaulting to 1.0. Boxes are matched by
    order (the CSV and .txt are written in the same iteration order).
    """
    if not label_path.exists():
        return []
    raw = label_path.read_text().strip()
    if not raw:
        return []
    dets: list[Det] = []
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        # Confidence: prefer CSV lookup by index, then 6th column, else 1.0
        conf = 1.0
        if conf_lookup and i < len(conf_lookup):
            conf = conf_lookup[i][5]
        elif len(parts) >= 6:
            try:
                conf = float(parts[5])
            except ValueError:
                pass
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        dets.append(Det(cls_id=cls_id, conf=conf,
                        x1=max(0.0, x1), y1=max(0.0, y1),
                        x2=min(float(img_w), x2), y2=min(float(img_h), y2)))
    return dets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw YOLO labels on images and save visualizations.",
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=r"C:\Users\JR\Downloads\Kaanapali_yolo",
        help="YOLOv8 dataset root (train/valid/test/{images,labels}).",
    )
    parser.add_argument(
        "--names", type=str, default=None,
        help="Optional comma-separated class names, e.g. 'person'. "
             "If omitted, reads from <input-dir>/data.yaml.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] Input dir not found: {input_dir}")
        return 1

    out_root = input_dir.parent / f"{input_dir.name}_viz"
    out_root.mkdir(parents=True, exist_ok=True)

    # Class names
    names: dict[int, str] | None = None
    if args.names:
        names = {i: n for i, n in enumerate(args.names.split(","))}
    else:
        yaml_path = input_dir / "data.yaml"
        if yaml_path.exists():
            import yaml
            with yaml_path.open("r") as f:
                data = yaml.safe_load(f)
            raw_names = data.get("names")
            if isinstance(raw_names, list):
                names = {i: n for i, n in enumerate(raw_names)}
            elif isinstance(raw_names, dict):
                names = {int(k): v for k, v in raw_names.items()}

    # Load confidence CSV (consolidated across all splits)
    csv_path = input_dir / "detections.csv"
    conf_map = load_confidence_csv(csv_path)
    if conf_map:
        print(f"Loaded {sum(len(v) for v in conf_map.values())} detections "
              f"from {csv_path.name} ({len(conf_map)} images)")
    else:
        print(f"[WARN] No detections.csv found at {csv_path} — using conf=1.0")

    print(f"Input:  {input_dir}")
    print(f"Output: {out_root}")
    if names:
        print(f"Classes: {names}")
    print("-" * 80)

    total = 0
    for split in ("train", "valid", "test"):
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"
        if not img_dir.is_dir():
            continue
        out_img_dir = out_root / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            conf_lookup = conf_map.get(img_path.name)
            dets = load_label_dets(
                lbl_dir / f"{img_path.stem}.txt", w, h, conf_lookup=conf_lookup,
            )
            vis = draw_dets(img, dets, names=names)
            cv2.imwrite(str(out_img_dir / img_path.name), vis)
            n += 1
        print(f"  [{split}] {n} images")
        total += n

    print()
    print(f"[OK] Visualized {total} images -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
