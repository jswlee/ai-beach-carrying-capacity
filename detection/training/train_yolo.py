#!/usr/bin/env python3
"""
YOLO training script with optimized hyperparameters.
"""

from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from utils import patch_torch_load, select_device, save_training_summary

# Apply PyTorch compatibility patch
patch_torch_load()

# Training results directory
RESULTS_PATH = Path("training_results")

def train_yolo_model(
    dataset_yaml: str = "processed_images_ready_for_training/data.yaml",
    model_size: str = "yolov8x.pt",
    epochs: int = 150,
    batch_size: int = 4,
    img_size: int = 1920,
    output_dir: str = None,
    run_name: str = "beach_detection",
    freeze: int = 10,
    lr0: float = 0.0001,
    patience: int = 50,
    warmup_epochs: int = 10,
    optimizer: str = "AdamW",
    resume_from: str | None = None,
    p2: bool = False,
):
    """
    Train YOLO model with optimized hyperparameters
    
    Args:
        dataset_yaml: Path to dataset YAML file
        model_size: YOLO model size (e.g., 'yolov8x.pt')
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        output_dir: Output directory for training results
        run_name: Name for this training run
        freeze: Number of layers to freeze
        lr0: Initial learning rate
        patience: Patience for early stopping
        warmup_epochs: Number of warmup epochs
        optimizer: Optimizer to use for training
        p2: If True, use the P2 (stride-4) architecture variant for small-object detection
    Returns:
        Tuple of (path to the best model weights, Ultralytics run directory)
    """
    start_time = datetime.now()
    
    print("Beach Detection - YOLO Training")
    print("=" * 70)
    print()
    
    # Set output directory. Must be absolute: Ultralytics >=8.4 nests a relative
    # `project` under SETTINGS['runs_dir']/{task} (e.g. ./detect/training_results/...).
    output_dir = str(Path(output_dir).resolve()) if output_dir else str(RESULTS_PATH.resolve())
    
    # Create results subdirectories
    (RESULTS_PATH / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS_PATH / "models").mkdir(parents=True, exist_ok=True)
    
    # Check dataset
    dspath = Path(dataset_yaml)
    if not dspath.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    # Device selection
    device = select_device()
    print(f"Using device: {device}")
    print()
    
    # Build or resume model
    if resume_from:
        print(f"Loading model from checkpoint for resume: {resume_from}")
        model = YOLO(resume_from)
    elif p2:
        # Derive P2 architecture YAML from the model name (e.g. yolov8n.pt -> yolov8n-p2.yaml)
        base_name = Path(model_size).stem  # e.g. 'yolov8n'
        p2_yaml = f"{base_name}-p2.yaml"
        print(f"Loading P2 architecture: {p2_yaml} with weights from {model_size}")
        model = YOLO(p2_yaml).load(model_size)
    else:
        print("Loading model...")
        model = YOLO(model_size)
    print()
    
    # Train with optimized hyperparameters
    print("Starting training...")
    print("-" * 70)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name=run_name,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        verbose=True,
        resume=bool(resume_from),
        # Hyperparameters
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=9.0,
        cls=0.5,
        dfl=1.5,
        freeze=freeze,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        # Validation settings
        patience=patience,
        close_mosaic=10,
    )
    
    end_time = datetime.now()
    
    # Determine run dir and best weights from the trainer itself (authoritative,
    # accounts for Ultralytics path resolution and `increment_path` suffixes)
    run_dir = Path(getattr(model.trainer, "save_dir", None) or (Path(output_dir) / run_name))
    best_auto = run_dir / "weights" / "best.pt"
    best_model_path = RESULTS_PATH / "models" / f"{run_name}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_auto.exists():
        best_model_path.write_bytes(best_auto.read_bytes())
    else:
        model.save(str(best_model_path))
    
    print()
    print("-" * 70)
    print(f"✓ Training complete!")
    print(f"  Best model saved to: {best_model_path}")
    print(f"  Training results: {run_dir}")
    print()
    
    # Save comprehensive training summary (reads actual config from args.yaml)
    save_training_summary(
        run_dir,
        best_model_path,
        start_time,
        end_time
    )
    
    return str(best_model_path), run_dir

def run_test_inference(
    model_path: str,
    dataset_yaml: str,
    img_size: int,
    run_dir: Path,
    conf: float = 0.15,
):
    """Run inference on the test split and save predictions/metrics in the run directory.
    
    Args:
        model_path: Path to the trained model weights (.pt)
        dataset_yaml: Path to the dataset YAML file
        img_size: Inference image size
        run_dir: Training run directory where outputs should be saved
    """
    print("Running test-set inference...")
    print("=" * 70)
    print()

    # Device selection
    device = select_device()
    print(f"Using device for test inference: {device}")
    print()

    # Load trained model
    model = YOLO(model_path)

    # Run validation on the test split
    # This will save:
    # - Annotated images with colored labels
    # - Metrics CSV with per-class mAP50 and mAP50-95
    # - PR/F1/confusion-matrix plots
    results = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=img_size,
        device=device,
        batch=1,
        project=str(run_dir),
        name="test_inference",
        save=True,      # save images with predictions
        save_txt=True,
        plots=True,     # save metric plots
        verbose=True,
        conf=0.001,
    )

    print()
    print("Test inference completed.")
    print(f"  Results directory: {results.save_dir}")
    print()

def run_merged_test_inference(
    model_path: str,
    raw_data_dir: str,
    img_size: int,
    run_dir: Path,
    conf: float = 0.20,
    iou_merge: float = 0.50,
    iou_match: float = 0.50,
    person_only: bool = False,
    person_class_id: int = 3,
    tile_size: int | None = None,
):
    """Run tiled inference on raw test images with NMS merge and compute image-level metrics.

    For youtube_snapshot images: uses 3 bottom tiles + ROI masking.
    For other images: uses full-resolution grid tiling (no masking).
    Detections from overlapping tiles are merged via NMS to avoid double-counting.

    Args:
        model_path: Path to the trained model weights (.pt)
        raw_data_dir: Path to the raw (un-tiled) dataset directory
        img_size: Model inference input size (imgsz passed to predict)
        run_dir: Training run directory where outputs should be saved
        conf: Confidence threshold for inference
        iou_merge: IoU threshold for cross-tile NMS merge
        iou_match: IoU threshold for matching predictions to ground truth
        person_only: If True, only evaluate the person class
        person_class_id: Class ID for person in the raw dataset
        tile_size: Crop/tile size in pixels (default: same as img_size).
            Set smaller than img_size for zoomed inference, e.g.
            tile_size=640 with img_size=1280 upscales tiles 2x.
    """
    tile_size = tile_size or img_size
    import sys
    import json
    import cv2
    import numpy as np

    # Ensure preprocessing module is importable
    preproc_dir = str(Path(__file__).parent.parent / "preprocessing")
    if preproc_dir not in sys.path:
        sys.path.append(preproc_dir)

    from slice_and_preprocess import (
        is_youtube_snapshot, should_apply_mask,
        get_bottom_band_tiles, bottom_band_top, get_tiles,
        get_exclusion_polygon, OVERLAP_FULL,
    )
    from run_tiled_inference import nms_merge

    print("Running merged tiled test inference...")
    print("=" * 70)

    device = select_device()
    model = YOLO(model_path)

    test_img_dir = Path(raw_data_dir) / "test" / "images"
    test_lbl_dir = Path(raw_data_dir) / "test" / "labels"

    if not test_img_dir.exists():
        print(f"No test images directory found: {test_img_dir}")
        return

    image_files = sorted(
        list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    )
    if not image_files:
        print("No test images found.")
        return

    print(f"Device: {device}")
    print(f"Test images: {len(image_files)}")
    print(f"Confidence threshold: {conf}")
    print(f"NMS merge IoU: {iou_merge}")
    print(f"Match IoU threshold: {iou_match}")
    print()

    # Output directory
    out_dir = run_dir / "merged_test_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # Aggregate metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_image_results = []
    all_detections = []  # For AP computation: (conf, is_tp, class_id)
    total_gt_count = 0

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Determine tiling strategy
        yt_snapshot = is_youtube_snapshot(img_file.name)
        if yt_snapshot:
            tiles = get_bottom_band_tiles(w, h, tile_size)
        else:
            tiles = get_tiles(w, h, tile_size, OVERLAP_FULL)

        # Apply masking only to qualifying youtube_snapshot images
        apply_mask = should_apply_mask(img_file.name) if yt_snapshot else False
        mask = None
        if apply_mask:
            mask = np.ones((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [get_exclusion_polygon(img_file.name)], 0)
            img_for_inference = img.copy()
            img_for_inference[mask == 0] = 0
        else:
            img_for_inference = img

        # Extract tiles
        tile_imgs = []
        tile_offsets = []
        for tx, ty in tiles:
            tile = img_for_inference[ty:ty + tile_size, tx:tx + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tile_imgs.append(tile)
                tile_offsets.append((tx, ty))

        if not tile_imgs:
            continue

        # Batch inference
        results = model.predict(
            source=tile_imgs, imgsz=img_size, conf=conf,
            device=device, verbose=False,
            save=False
        )

        # Map detections back to full-image coords
        all_boxes_list, all_scores_list, all_cls_list = [], [], []
        for res, (tx, ty) in zip(results, tile_offsets):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy().copy()
            scores = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(np.int32)
            xyxy[:, 0] += tx
            xyxy[:, 1] += ty
            xyxy[:, 2] += tx
            xyxy[:, 3] += ty
            all_boxes_list.append(xyxy)
            all_scores_list.append(scores)
            all_cls_list.append(cls_ids)

        # NMS merge
        if all_boxes_list:
            boxes_np = np.concatenate(all_boxes_list).astype(np.float32)
            scores_np = np.concatenate(all_scores_list).astype(np.float32)
            cls_np = np.concatenate(all_cls_list)
            keep = nms_merge(boxes_np, scores_np, cls_np, iou_merge)
            pred_boxes = boxes_np[keep]
            pred_scores = scores_np[keep]
            pred_cls = cls_np[keep]
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty(0, dtype=np.float32)
            pred_cls = np.empty(0, dtype=np.int32)

        # Save predictions as YOLO-format txt
        pred_lines = []
        for i in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[i]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw_n = (x2 - x1) / w
            bh_n = (y2 - y1) / h
            pred_lines.append(
                f"{int(pred_cls[i])} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f} {pred_scores[i]:.4f}"
            )
        (pred_dir / f"{img_file.stem}.txt").write_text("\n".join(pred_lines))

        # --- Load ground truth (from raw labels) ---
        # Determine the spatial region the model can actually see so we only
        # evaluate GT boxes that fall inside the tiled area.
        if yt_snapshot:
            # youtube_snapshot: only bottom band is tiled
            y_band_top = bottom_band_top(h, tile_size)
        else:
            # full-res grid: entire image is tiled
            y_band_top = 0

        gt_boxes_xyxy = []
        gt_cls_list = []
        lbl_file = test_lbl_dir / f"{img_file.stem}.txt"
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = (
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                    )
                except (ValueError, IndexError):
                    continue

                if person_only and cls_id != person_class_id:
                    continue

                # Filter GT by exclusion zone for masked images
                if apply_mask and mask is not None:
                    cx_px = max(0, min(int(cx * w), w - 1))
                    cy_px = max(0, min(int(cy * h), h - 1))
                    if mask[cy_px, cx_px] == 0:
                        continue

                # Filter GT to the tiled region
                cy_px_abs = cy * h
                if cy_px_abs < y_band_top:
                    continue

                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                gt_boxes_xyxy.append([x1, y1, x2, y2])
                gt_cls_list.append(0 if person_only else cls_id)

        gt_boxes_np = (
            np.array(gt_boxes_xyxy, dtype=np.float32)
            if gt_boxes_xyxy
            else np.empty((0, 4), dtype=np.float32)
        )
        gt_cls_np = (
            np.array(gt_cls_list, dtype=np.int32)
            if gt_cls_list
            else np.empty(0, dtype=np.int32)
        )

        # --- Match predictions to GT (greedy IoU matching, conf-descending) ---
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes_np)
        total_gt_count += n_gt

        gt_matched = np.zeros(n_gt, dtype=bool)
        img_tp = 0
        img_fp = 0

        if n_pred > 0:
            sort_idx = np.argsort(-pred_scores)
            for idx in sort_idx:
                p_box = pred_boxes[idx]
                p_cls = pred_cls[idx]
                p_conf = float(pred_scores[idx])

                best_iou = 0.0
                best_gt = -1

                for g in range(n_gt):
                    if gt_matched[g]:
                        continue
                    if gt_cls_np[g] != p_cls:
                        continue

                    ix1 = max(p_box[0], gt_boxes_np[g, 0])
                    iy1 = max(p_box[1], gt_boxes_np[g, 1])
                    ix2 = min(p_box[2], gt_boxes_np[g, 2])
                    iy2 = min(p_box[3], gt_boxes_np[g, 3])

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_p = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    area_g = (
                        (gt_boxes_np[g, 2] - gt_boxes_np[g, 0])
                        * (gt_boxes_np[g, 3] - gt_boxes_np[g, 1])
                    )
                    iou = inter / (area_p + area_g - inter)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g

                if best_iou >= iou_match and best_gt >= 0:
                    gt_matched[best_gt] = True
                    img_tp += 1
                    all_detections.append((p_conf, True, int(p_cls)))
                else:
                    img_fp += 1
                    all_detections.append((p_conf, False, int(p_cls)))

        img_fn = n_gt - img_tp
        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn

        per_image_results.append({
            "image": img_file.name,
            "tiles": len(tile_offsets),
            "tiling": "3-bottom" if yt_snapshot else "full-grid",
            "masked": apply_mask,
            "gt_count": n_gt,
            "pred_count_raw": n_pred,
            "pred_count_merged": len(pred_boxes),
            "tp": img_tp,
            "fp": img_fp,
            "fn": img_fn,
        })

        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
        img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0.0
        print(
            f"  {img_file.name}: tiles={len(tile_offsets)} gt={n_gt} "
            f"pred={len(pred_boxes)} TP={img_tp} FP={img_fp} FN={img_fn} "
            f"P={img_prec:.2f} R={img_rec:.2f}"
        )

    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    ap50 = _compute_ap(all_detections, total_gt_count)

    summary = {
        "model": str(model_path),
        "raw_data_dir": str(raw_data_dir),
        "conf": conf,
        "iou_merge": iou_merge,
        "iou_match": iou_match,
        "person_only": person_only,
        "images": len(per_image_results),
        "total_gt": total_gt_count,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ap50": round(ap50, 4),
    }

    print()
    print("=" * 70)
    print("Merged Test Inference Results:")
    print(f"  Images:    {len(per_image_results)}")
    print(f"  GT total:  {total_gt_count}")
    print(f"  TP:        {total_tp}")
    print(f"  FP:        {total_fp}")
    print(f"  FN:        {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AP@50:     {ap50:.4f}")
    print("=" * 70)

    # Save results
    results_path = out_dir / "merged_metrics.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)

    print(f"Results saved to: {results_path}")


def _compute_ap(all_detections, total_gt):
    """Compute Average Precision (101-point COCO interpolation).

    Args:
        all_detections: list of (confidence, is_tp, class_id) tuples
        total_gt: total number of ground-truth boxes
    Returns:
        AP value (float)
    """
    if not all_detections or total_gt == 0:
        return 0.0

    import numpy as np

    sorted_dets = sorted(all_detections, key=lambda x: -x[0])

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for _conf_val, is_tp, _cls_id in sorted_dets:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)

    precisions_np = np.array(precisions)
    recalls_np = np.array(recalls)

    # Make precision monotonically decreasing (right-to-left max)
    for i in range(len(precisions_np) - 2, -1, -1):
        precisions_np[i] = max(precisions_np[i], precisions_np[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        mask = recalls_np >= t
        if mask.any():
            ap += precisions_np[mask].max()
    ap /= 101.0

    return float(ap)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model with optimized hyperparameters")
    parser.add_argument("--dataset", type=str, default="processed_images_ready_for_training/data.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=1920, help="Image size for training")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Model to use for training")
    parser.add_argument("--run-name", type=str, default="beach_detection", help="Name for this training run")
    parser.add_argument("--freeze", type=int, default=5, help="Number of layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    try:
        model_path, _run_dir = train_yolo_model(
            dataset_yaml=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            run_name=args.run_name,
            freeze=args.freeze,
            lr0=args.lr0,
            patience=args.patience,
        )
        print(f"\n✓ Training completed successfully!")
        print(f"  Trained model: {model_path}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
