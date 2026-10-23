#!/usr/bin/env python3
"""Export YOLO predictions on the test set to a single CSV and compute metrics.

This script:
  - Loads a trained YOLO model and dataset YAML
  - Finds all test images from the YAML
  - Runs inference on each test image
  - Joins predictions with ground-truth labels
  - Writes one CSV row per prediction with best IoU to any GT box
  - Computes mAP50, mAP50-95 (single-class) and prints a short summary

Each CSV row corresponds to one predicted bounding box:

    image_path, class_id, confidence, cx, cy, w, h, best_iou

where (cx, cy, w, h) are normalized [0,1] YOLO-format coordinates and
best_iou is the maximum IoU with any ground-truth box for that image.

Usage example:

    python export_test_predictions_to_csv.py \
        --model training_results/models/beach_detection_20260104_101723_best.pt \
        --data processed_data/processed_images_v13/data.yaml \
        --output test_predictions_v13.csv \
        --img-size 1920 \
        --conf 0.20
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from ultralytics import YOLO


def load_test_images_from_yaml(data_yaml: Path) -> List[Path]:
    with data_yaml.open("r") as f:
        data = yaml.safe_load(f)

    test_entry = data.get("test")
    base_path = data.get("path", None)

    if test_entry is None:
        raise ValueError(f"No 'test' key found in data.yaml: {data_yaml}")

    # Resolve test image directory
    test_path = Path(test_entry)
    if not test_path.is_absolute():
        if base_path is not None:
            test_path = Path(base_path) / test_path
        else:
            test_path = data_yaml.parent / test_path

    if not test_path.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_path}")

    image_files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        image_files.extend(test_path.glob(ext))

    image_files = sorted(image_files)
    return image_files


def load_ground_truth_boxes(test_images: List[Path]) -> Dict[Path, List[Tuple[float, float, float, float]]]:
    """Load YOLO-format GT boxes for each test image.

    Returns a mapping from image_path -> list of (cx, cy, w, h) normalized.
    If a label file is missing, that image gets an empty list.
    """
    gt_boxes: Dict[Path, List[Tuple[float, float, float, float]]] = {}

    for img_path in test_images:
        # Assume standard YOLO structure: .../images/*.jpg -> .../labels/*.txt
        parent = img_path.parent
        if parent.name != "images":
            # Fallback: try sibling 'labels' directory
            labels_dir = parent.parent / "labels"
        else:
            labels_dir = parent.parent / "labels"

        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes: List[Tuple[float, float, float, float]] = []

        if label_path.exists():
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        # class_id = int(parts[0])  # single-class, we don't need it here
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cx, cy, w, h))
                    except ValueError:
                        continue

        gt_boxes[img_path] = boxes

    return gt_boxes


def xywhn_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert normalized cx,cy,w,h to absolute x1,y1,x2,y2 in pixels."""
    abs_cx = cx * img_w
    abs_cy = cy * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    x1 = abs_cx - abs_w / 2.0
    y1 = abs_cy - abs_h / 2.0
    x2 = abs_cx + abs_w / 2.0
    y2 = abs_cy + abs_h / 2.0
    return x1, y1, x2, y2


def box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in x1,y1,x2,y2 format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute AP using a simple interpolation over the precision-recall curve."""
    if not recalls:
        return 0.0

    # Sort by recall
    pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    r = [p[0] for p in pairs]
    p = [p[1] for p in pairs]

    # Make precision non-increasing when going from right to left
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    # Integrate using trapezoidal rule
    ap = 0.0
    prev_r = 0.0
    prev_p = p[0]
    for ri, pi in zip(r, p):
        ap += (ri - prev_r) * prev_p
        prev_r, prev_p = ri, pi

    return ap


def export_predictions_to_csv(model_path: Path, data_yaml: Path, output_csv: Path, img_size: int, conf: float) -> None:
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Reading test images from: {data_yaml}")
    test_images = load_test_images_from_yaml(data_yaml)
    if not test_images:
        print("No test images found.")
        return

    print(f"Found {len(test_images)} test images.")

    # Load GT boxes for IoU and mAP computation
    gt_boxes = load_ground_truth_boxes(test_images)
    num_gt_boxes = sum(len(v) for v in gt_boxes.values())
    print(f"Total GT boxes: {num_gt_boxes}")

    print(f"Writing predictions to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Store predictions for mAP computation
    # Each entry: (image_index, score, box_xyxy)
    pred_records: List[Tuple[int, float, Tuple[float, float, float, float]]] = []

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_id", "confidence", "cx", "cy", "w", "h", "best_iou"])  # header

        for img_idx, img_path in enumerate(test_images):
            # Run inference on a single image
            results = model(
                str(img_path),
                imgsz=img_size,
                conf=conf,
                verbose=False,
            )[0]

            img_h, img_w = results.orig_shape[:2]
            gts_norm = gt_boxes.get(img_path, [])
            gts_xyxy = [xywhn_to_xyxy(cx, cy, w, h, img_w, img_h) for (cx, cy, w, h) in gts_norm]

            if results.boxes is None or len(results.boxes) == 0:
                print(f"Processed {img_path.name}: 0 detections")
                continue

            boxes = results.boxes
            cls_ids = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xywhn = boxes.xywhn.cpu().tolist()  # normalized
            xyxy = boxes.xyxy.cpu().tolist()    # absolute

            for cls_id, score, (cx, cy, w, h), box_xyxy in zip(cls_ids, confs, xywhn, xyxy):
                # Compute best IoU with any GT box for this image
                best_iou = 0.0
                for gt_box in gts_xyxy:
                    iou_val = box_iou(tuple(box_xyxy), gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val

                writer.writerow([
                    str(img_path),
                    int(cls_id),
                    float(score),
                    float(cx),
                    float(cy),
                    float(w),
                    float(h),
                    best_iou,
                ])

                pred_records.append((img_idx, float(score), tuple(box_xyxy)))

            print(f"Processed {img_path.name}: {len(cls_ids)} detections")

    # ------------------------------------------------------------------
    # Compute mAP50 and mAP50-95 (single-class) from pred_records + GT
    # ------------------------------------------------------------------
    if num_gt_boxes == 0 or not pred_records:
        print("No GT boxes or predictions available to compute mAP.")
        print("Done.")
        return

    # Prepare GT boxes per image index in xyxy format for reuse
    gt_xyxy_per_img: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for idx, img_path in enumerate(test_images):
        gts_norm = gt_boxes.get(img_path, [])
        if not gts_norm:
            gt_xyxy_per_img[idx] = []
            continue
        # Need image size; use the model again in a lightweight way
        # (we assume orig_shape is constant per image)
        # To keep it simple, recompute using stored normalized boxes and one more read
        # This is cheap compared to training/inference.
        sample_results = model(str(img_path), imgsz=img_size, conf=conf, verbose=False)[0]
        img_h, img_w = sample_results.orig_shape[:2]
        gt_xyxy_per_img[idx] = [
            xywhn_to_xyxy(cx, cy, w, h, img_w, img_h) for (cx, cy, w, h) in gts_norm
        ]

    def evaluate_map_at_iou(threshold: float) -> float:
        # For each image, track which GT boxes are already matched
        matched_gt: Dict[int, List[bool]] = {
            img_idx: [False] * len(gt_xyxy_per_img[img_idx]) for img_idx in gt_xyxy_per_img
        }

        # Sort predictions by score descending
        preds_sorted = sorted(pred_records, key=lambda x: x[1], reverse=True)

        tps: List[int] = []
        fps: List[int] = []

        for img_idx, score, box_xyxy in preds_sorted:
            gts = gt_xyxy_per_img.get(img_idx, [])
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt_box in enumerate(gts):
                if matched_gt[img_idx][gi]:
                    continue
                iou_val = box_iou(box_xyxy, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gi

            if best_iou >= threshold and best_gt_idx >= 0:
                matched_gt[img_idx][best_gt_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        # Accumulate
        cum_tp = 0
        cum_fp = 0
        recalls: List[float] = []
        precisions: List[float] = []

        for tp, fp in zip(tps, fps):
            cum_tp += tp
            cum_fp += fp
            recall = cum_tp / num_gt_boxes
            precision = cum_tp / max(cum_tp + cum_fp, 1e-9)
            recalls.append(recall)
            precisions.append(precision)

        return compute_ap(recalls, precisions)

    # mAP50 and mAP50-95
    iou_thresholds = [0.50 + 0.05 * i for i in range(10)]  # 0.50, 0.55, ..., 0.95
    ap_values = []
    for thr in iou_thresholds:
        ap = evaluate_map_at_iou(thr)
        ap_values.append(ap)

    map50 = ap_values[0]
    map5095 = sum(ap_values) / len(ap_values)

    print("\nEvaluation metrics (single 'person' class):")
    print(f"  mAP50:     {map50:.4f}")
    print(f"  mAP50-95:  {map5095:.4f}")
    print("Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO test-set predictions to CSV.")
    parser.add_argument("--model", type=str, default="training_results/models/beach_detection_20260104_101723_best.pt", help="Path to trained YOLO .pt model")
    parser.add_argument("--data", type=str, default="processed_data/processed_images_v13/data.yaml", help="Path to dataset YAML (must define test split)")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--img-size", type=int, default=1920, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve()
    output_csv = Path(args.output).resolve()

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    if not data_yaml.exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        return 1

    export_predictions_to_csv(
        model_path=model_path,
        data_yaml=data_yaml,
        output_csv=output_csv,
        img_size=args.img_size,
        conf=args.conf,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
