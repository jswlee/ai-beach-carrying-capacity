#!/usr/bin/env python3
"""Run tiled YOLO inference (3 bottom 640x640 tiles) on full images and merge back.

Pipeline:
  original image -> crop bottom band -> mask tiles -> YOLO inference (batched)
  -> map boxes to full-image coords -> NMS merge -> save outputs.

This is intended for apples-to-apples evaluation/visualization when the model
was trained on 640x640 tiles.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import patch_torch_load, select_device

# Add preprocessing directory to path so we can reuse the ROI mask + tiling logic
ROOT_DIR = Path(__file__).parent.parent
PREPROC_DIR = ROOT_DIR / "preprocessing"
if PREPROC_DIR.exists():
    sys.path.append(str(PREPROC_DIR))

from apply_roi_mask import create_roi_mask
from slice_and_preprocess import (
    get_bottom_band_tiles,
    get_tiles,
    is_youtube_snapshot,
    should_apply_mask,
    get_exclusion_polygon,
    EXCLUSION_POLYGON,
    EXCLUSION_POLYGON_KAANAPALI_AFTER,
    OVERLAP_FULL,
    MIN_OVERLAP_RATIO,
)


patch_torch_load()


# ---------------------------------------------------------------------------
# Waterline-seg masking (--seg-mask)
#
# Replicates the exact frame preprocessing used by the segmentation pipeline
# (waterline-detection/test_viz_segformer.py + preprocess_masks.py) so the
# detection model sees pixel-identical inputs to the SegFormer model:
#   1. Night gate: mean HSV saturation < NIGHT_SAT_MAX -> IR night frame, skip.
#   2. Inclusion ROI polygon: pixels OUTSIDE the polygon are blacked out; the
#      pre-/post-camera-move polygon is picked by filename vs. the cutoff
#      (same rules and constants as preprocess_masks.apply_roi_mask).
#   3. Exclusion triangle: for filenames sorting after SEG_TRI_AFTER, the
#      small EXCLUDE triangle in the top-right corner is blacked out too.
# ---------------------------------------------------------------------------

# Inclusion ROI polygon, pre-camera-move (config.POLYGON_PTS_LIST)
SEG_ROI_POLYGON_BEFORE = np.array([
    [2, 693], [1655, 380], [1861, 485], [1850, 617], [1819, 614],
    [1790, 628], [1794, 644], [1782, 655], [1757, 673], [1753, 681],
    [1783, 693], [1642, 749], [1621, 723], [1444, 777], [1206, 823],
    [1169, 820], [904, 868], [804, 907], [745, 937], [676, 943],
    [621, 960], [595, 954], [580, 967], [551, 968], [472, 972],
    [249, 1012], [214, 1021], [159, 1023], [49, 1050], [4, 1073]
], dtype=np.int32)

# Inclusion ROI polygon, post-camera-move (config.POLYGON_PTS_AFTER_LIST)
SEG_ROI_POLYGON_AFTER = np.array([
    [1, 1060], [62, 1043], [171, 1019], [203, 1019], [236, 1017],
    [300, 1002], [330, 999], [398, 987], [421, 987], [430, 978],
    [501, 972], [509, 975], [521, 966], [589, 966], [592, 955],
    [621, 955], [630, 958], [654, 949], [677, 946], [707, 937],
    [780, 934], [792, 940], [825, 961], [866, 952], [819, 908],
    [836, 890], [866, 881], [880, 881], [913, 863], [951, 852],
    [975, 846], [1019, 837], [1048, 828], [1095, 825], [1125, 816],
    [1172, 810], [1210, 816], [1237, 810], [1260, 810], [1284, 805],
    [1305, 805], [1349, 799], [1387, 787], [1413, 787], [1443, 799],
    [1496, 784], [1461, 775], [1478, 763], [1499, 757], [1508, 754],
    [1540, 746], [1558, 746], [1575, 743], [1590, 734], [1620, 728],
    [1637, 725], [1681, 734], [1814, 687], [1779, 672], [1785, 663],
    [1802, 654], [1823, 648], [1820, 634], [1817, 622], [1835, 613],
    [1876, 607], [1875, 465], [1875, 440], [1875, 431], [1875, 431],
    [1875, 406], [1, 668], [1, 969], [1, 1016]
], dtype=np.int32)

# Same cutoffs/constants as the seg pipeline
SEG_CAMERA_MOVE_CUTOFF = "youtube_snapshot_Kaanapali_20260127_222935.jpg"
SEG_TRI_AFTER          = "youtube_snapshot_Kaanapali_20260127_221431"
SEG_EXCLUDE_TRI        = np.array([[1919, 543], [1800, 393], [1917, 396]],
                                  dtype=np.int32)
NIGHT_SAT_MAX          = 5.0   # IR night frames have mean saturation ~0
LOCAL_UTC_OFFSET_H     = -10   # camera is in Hawaii (HST = UTC-10, no DST)
DAY_START_H, DAY_END_H = 5.0, 20.0   # local-hour "day" window for --save-day-only

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def mean_saturation(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())


def is_night_frame(img_bgr: np.ndarray, sat_max: float = NIGHT_SAT_MAX) -> bool:
    """Same saturation-based night gate as test_viz_segformer.py."""
    return mean_saturation(img_bgr) < sat_max


def apply_seg_mask(img_bgr: np.ndarray, filename: str) -> np.ndarray:
    """Apply the segmentation-pipeline image masking (inclusion ROI + triangle).

    Mirrors preprocess_masks.apply_roi_mask: if the image already looks
    masked (>30% pure-black pixels) it is left unchanged.  Post-move frames
    additionally get the small exclusion triangle blacked out.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if (gray <= 10).mean() > 0.3:
        out = img_bgr  # already polygon-masked
    else:
        poly = (SEG_ROI_POLYGON_AFTER
                if filename >= SEG_CAMERA_MOVE_CUTOFF
                else SEG_ROI_POLYGON_BEFORE)
        roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi, [poly], 255)
        out = cv2.bitwise_and(img_bgr, img_bgr, mask=roi)
    if filename > SEG_TRI_AFTER:
        tri = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(tri, [SEG_EXCLUDE_TRI], 1)
        out[tri.astype(bool)] = 0
    return out


def local_hour(filename: str) -> float | None:
    """Local (HST) hour parsed from the embedded UTC timestamp, or None."""
    m = _TS_RE.search(filename)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    ldt = dt + timedelta(hours=LOCAL_UTC_OFFSET_H)
    return ldt.hour + ldt.minute / 60.0


def is_day_photo(filename: str) -> bool:
    """True when the filename's timestamp falls in the local day window.

    Filenames without a parseable timestamp are treated as day (the night
    gate already removes dark frames)."""
    h = local_hour(filename)
    return h is None or (DAY_START_H <= h < DAY_END_H)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Det:
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def get_tiles_for_image(
    img_filename: str, img_w: int, img_h: int, tile_size: int,
) -> list[tuple[int, int]]:
    """Return tile (x, y) offsets matching what slice_and_preprocess uses.

    YouTube snapshots -> 3 bottom tiles.
    Everything else   -> full-resolution overlapping grid.
    """
    if is_youtube_snapshot(img_filename):
        return get_bottom_band_tiles(img_w, img_h, tile_size)
    else:
        return get_tiles(img_w, img_h, tile_size, OVERLAP_FULL)


@lru_cache(maxsize=16)
def _get_roi_mask_u8(img_h: int, img_w: int, kaanapali_after: bool) -> np.ndarray:
    """Cached full-resolution ROI mask (0/255 uint8)."""
    polygon = EXCLUSION_POLYGON_KAANAPALI_AFTER if kaanapali_after else EXCLUSION_POLYGON
    mask01 = create_roi_mask((img_h, img_w, 3), polygon=polygon)
    return (mask01 * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# NMS (pure numpy, optimized for CPU with small detection counts)
# ---------------------------------------------------------------------------

def nms_merge(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
             iou_thr: float) -> np.ndarray:
    """Class-wise NMS using class-offset trick + vectorized numpy.

    Args:
        boxes: (N, 4) float32 xyxy
        scores: (N,) float32
        cls_ids: (N,) int
        iou_thr: IoU threshold
    Returns:
        keep: 1-D int array of indices into the original arrays
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.intp)

    # Offset boxes by class so cross-class boxes never overlap
    max_coord = boxes.max() + 1.0
    offsets = cls_ids.astype(np.float32) * max_coord
    x1 = boxes[:, 0] + offsets
    y1 = boxes[:, 1] + offsets
    x2 = boxes[:, 2] + offsets
    y2 = boxes[:, 3] + offsets

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]

    return np.array(keep, dtype=np.intp)


# ---------------------------------------------------------------------------
# Formatting / drawing helpers
# ---------------------------------------------------------------------------

def to_yolo_line(det: Det, w: int, h: int) -> str:
    cx = ((det.x1 + det.x2) / 2.0) / w
    cy = ((det.y1 + det.y2) / 2.0) / h
    bw = (det.x2 - det.x1) / w
    bh = (det.y2 - det.y1) / h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{det.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw_dets(
    img_bgr: np.ndarray,
    dets: list[Det],
    names: dict[int, str] | None,
    gt_boxes: list[tuple[int, float, float, float, float]] | None = None,
) -> np.ndarray:
    out = img_bgr.copy()

    if gt_boxes:
        h, w = out.shape[:2]
        for cls_id, cx, cy, bw, bh in gt_boxes:
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1] - 1, x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0] - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = str(d.cls_id)
        if names is not None and d.cls_id in names:
            label = names[d.cls_id]
        text = f"{label} {d.conf:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes.  (N,4) x (M,4) -> (N,M)"""
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def calculate_map_metrics(
    all_predictions: List[List[Tuple[float, float, float, float, float, int]]],
    all_ground_truth: List[List[Tuple[int, float, float, float, float]]],
    img_sizes: List[Tuple[int, int]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Calculate COCO-style mAP metrics (mAP@0.5, mAP@0.5:0.95, P, R, F1).

    Performs greedy IoU matching at 10 thresholds (0.50 .. 0.95) and computes
    per-class AP with all-point interpolation.
    """
    _zero = {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not all_predictions or not all_ground_truth:
        return _zero

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    n_iou = len(iou_thresholds)

    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_target_cls: list[np.ndarray] = []

    for preds, gts, (w, h) in zip(all_predictions, all_ground_truth, img_sizes):
        # --- predictions -------------------------------------------------------
        if preds:
            pred_arr = np.array(preds, dtype=np.float32)
            mask = pred_arr[:, 4] >= conf_threshold
            pred_arr = pred_arr[mask]
        else:
            pred_arr = np.zeros((0, 6), dtype=np.float32)

        # --- ground truth (YOLO xywh-norm -> xyxy-pixel) ----------------------
        if gts:
            gt_cls_l, gt_box_l = [], []
            for cls, cx, cy, bw, bh in gts:
                gt_cls_l.append(int(cls))
                gt_box_l.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h])
            gt_cls = np.array(gt_cls_l, dtype=np.int32)
            gt_boxes = np.array(gt_box_l, dtype=np.float32)
        else:
            gt_cls = np.zeros(0, dtype=np.int32)
            gt_boxes = np.zeros((0, 4), dtype=np.float32)

        npr, ngt = pred_arr.shape[0], gt_cls.shape[0]
        all_target_cls.append(gt_cls)

        if npr == 0:
            # nothing to score but we still need to count GT
            all_tp.append(np.zeros((0, n_iou), dtype=bool))
            all_conf.append(np.zeros(0, dtype=np.float32))
            all_pred_cls.append(np.zeros(0, dtype=np.int32))
            continue

        pred_boxes = pred_arr[:, :4]
        pred_conf  = pred_arr[:, 4]
        pred_cls   = pred_arr[:, 5].astype(np.int32)
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)

        if ngt == 0:
            all_tp.append(np.zeros((npr, n_iou), dtype=bool))
            continue

        # IoU matrix  (npr, ngt)
        iou_mat = _box_iou_numpy(pred_boxes, gt_boxes)
        cls_match = pred_cls[:, None] == gt_cls[None, :]  # (npr, ngt)

        tp = np.zeros((npr, n_iou), dtype=bool)
        for t, thr in enumerate(iou_thresholds):
            valid = (iou_mat >= thr) & cls_match
            pi, gi = np.where(valid)
            if pi.size == 0:
                continue
            ious = iou_mat[pi, gi]
            order = ious.argsort()[::-1]
            pi, gi = pi[order], gi[order]
            matched_pred, matched_gt = set(), set()
            for p, g in zip(pi, gi):
                if p in matched_pred or g in matched_gt:
                    continue
                tp[p, t] = True
                matched_pred.add(p)
                matched_gt.add(g)
        all_tp.append(tp)

    # --- aggregate across images --------------------------------------------
    tp     = np.concatenate(all_tp, axis=0)      if all_tp     else np.zeros((0, n_iou), dtype=bool)
    conf   = np.concatenate(all_conf, axis=0)     if all_conf   else np.zeros(0, dtype=np.float32)
    p_cls  = np.concatenate(all_pred_cls, axis=0) if all_pred_cls else np.zeros(0, dtype=np.int32)
    t_cls  = np.concatenate(all_target_cls, axis=0) if all_target_cls else np.zeros(0, dtype=np.int32)

    if conf.size == 0 and t_cls.size == 0:
        return _zero

    # sort by confidence (descending)
    order = conf.argsort()[::-1]
    tp    = tp[order]
    conf  = conf[order]
    p_cls = p_cls[order]

    unique_cls = np.unique(np.concatenate([p_cls, t_cls]))
    ap = np.zeros((len(unique_cls), n_iou))

    for ci, c in enumerate(unique_cls):
        c_mask  = p_cls == c
        n_gt_c  = (t_cls == c).sum()
        if n_gt_c == 0 or c_mask.sum() == 0:
            continue
        tp_c = tp[c_mask]
        for t in range(n_iou):
            tp_cum = np.cumsum(tp_c[:, t])
            fp_cum = np.cumsum(~tp_c[:, t])
            rec  = tp_cum / (n_gt_c + 1e-16)
            prec = tp_cum / (tp_cum + fp_cum + 1e-16)
            # all-point interpolation
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            for j in range(len(mpre) - 1, 0, -1):
                mpre[j - 1] = max(mpre[j - 1], mpre[j])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap[ci, t] = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    map50    = float(ap[:, 0].mean()) if ap.size > 0 else 0.0
    map50_95 = float(ap.mean())       if ap.size > 0 else 0.0

    total_tp = float(tp[:, 0].sum())
    precision = total_tp / (len(tp) + 1e-16)
    recall    = total_tp / (len(t_cls) + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _find_label_path(img_path: Path) -> Path | None:
    candidates: list[Path] = []

    candidates.append(img_path.with_suffix(".txt"))

    parent = img_path.parent
    if parent.name.lower() == "images":
        candidates.append(parent.parent / "labels" / f"{img_path.stem}.txt")
        candidates.append(parent / "labels" / f"{img_path.stem}.txt")

    img_str = str(img_path)
    if "\\images\\" in img_str.lower():
        parts = img_str.split("\\")
        for i, p in enumerate(parts):
            if p.lower() == "images":
                parts2 = parts.copy()
                parts2[i] = "labels"
                candidates.append(Path("\\".join(parts2)).with_suffix(".txt"))
                break

    for c in candidates:
        if c.exists():
            return c
    return None


def load_gt_boxes(
    img_path: Path,
    keep_class_id: int | None = None,
) -> tuple[list[tuple[int, float, float, float, float]], bool]:
    txt_path = _find_label_path(img_path)
    if txt_path is None:
        return ([], False)

    raw = txt_path.read_text().strip()
    if not raw:
        return ([], True)

    lines = raw.splitlines()
    out: list[tuple[int, float, float, float, float]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue
        if keep_class_id is None or cls_id == keep_class_id:
            out.append((cls_id, cx, cy, bw, bh))
    return (out, True)


def filter_gt_to_tile_region(
    gt_boxes: list[tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
    tile_size: int,
    img_filename: str,
) -> list[tuple[int, float, float, float, float]]:
    """Filter GT boxes to only those the tiled model can actually see.

    Mirrors what slice_and_preprocess.py does at training time:
      1. Spatial: keep only boxes that have sufficient overlap (>=50%) with at
         least one tile in the tiling grid for this image type.
      2. ROI mask: for youtube_snapshot images that get the exclusion polygon,
         drop boxes whose center falls inside the masked-out region.
    """
    tiles = get_tiles_for_image(img_filename, img_w, img_h, tile_size)
    apply_mask = should_apply_mask(img_filename)

    # Build exclusion mask for YouTube snapshots
    roi_mask = None
    if apply_mask:
        roi_mask = np.ones((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [get_exclusion_polygon(img_filename)], 0)

    filtered: list[tuple[int, float, float, float, float]] = []
    for cls, cx, cy, bw, bh in gt_boxes:
        # 1. ROI mask check (before expensive overlap computation)
        if roi_mask is not None:
            px = max(0, min(img_w - 1, int(cx * img_w)))
            py = max(0, min(img_h - 1, int(cy * img_h)))
            if roi_mask[py, px] == 0:
                continue

        # 2. Spatial: does this box overlap sufficiently with any tile?
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_w = bw * img_w
        abs_h = bh * img_h
        box_x1 = abs_cx - abs_w / 2.0
        box_y1 = abs_cy - abs_h / 2.0
        box_x2 = abs_cx + abs_w / 2.0
        box_y2 = abs_cy + abs_h / 2.0
        original_area = abs_w * abs_h

        if original_area <= 0:
            continue

        in_any_tile = False
        for (tx, ty) in tiles:
            ix1 = max(box_x1, tx)
            iy1 = max(box_y1, ty)
            ix2 = min(box_x2, tx + tile_size)
            iy2 = min(box_y2, ty + tile_size)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap_ratio = (ix2 - ix1) * (iy2 - iy1) / original_area
            if overlap_ratio >= MIN_OVERLAP_RATIO:
                in_any_tile = True
                break

        if in_any_tile:
            filtered.append((cls, cx, cy, bw, bh))

    return filtered


# ---------------------------------------------------------------------------
# Core inference (one frame)
# ---------------------------------------------------------------------------

def crop_and_mask_tiles(
    img_bgr: np.ndarray,
    tile_size: int,
    img_filename: str,
    seg_mask: bool = False,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Crop tiles and optionally apply ROI mask.

    YouTube snapshots -> 3 bottom tiles + ROI mask.
    Other images      -> full grid tiles, no mask.

    When seg_mask=True the frame was already masked by apply_seg_mask()
    (inclusion ROI + triangle), so the exclusion-polygon tile masking is
    skipped to keep the input identical to the segmentation pipeline.
    """
    h, w = img_bgr.shape[:2]
    tiles = get_tiles_for_image(img_filename, w, h, tile_size)
    apply_mask = should_apply_mask(img_filename) and not seg_mask

    # Only build ROI mask for images that need it
    mask_u8 = None
    if apply_mask:
        after = get_exclusion_polygon(img_filename) is EXCLUSION_POLYGON_KAANAPALI_AFTER
        mask_u8 = _get_roi_mask_u8(h, w, after)

    tile_imgs: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for (tx, ty) in tiles:
        tile = img_bgr[ty : ty + tile_size, tx : tx + tile_size]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            continue
        if mask_u8 is not None:
            tile_mask = mask_u8[ty : ty + tile_size, tx : tx + tile_size]
            tile = cv2.bitwise_and(tile, tile, mask=tile_mask)
        tile_imgs.append(tile)
        offsets.append((tx, ty))

    return tile_imgs, offsets


def infer_one_image(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    tile_size: int,
    iou_merge: float,
    device: str,
    img_filename: str = "",
    stage_times: dict | None = None,
    img_size: int | None = None,
    seg_mask: bool = False,
) -> list[Det]:
    h, w = img_bgr.shape[:2]

    # --- Stage 0: seg-style full-frame masking (same pixels the seg model sees)
    if seg_mask:
        img_bgr = apply_seg_mask(img_bgr, img_filename)

    # --- Stage 1: Tile + mask ---
    t_tile = time.perf_counter()
    tile_imgs, offsets = crop_and_mask_tiles(
        img_bgr, tile_size, img_filename, seg_mask=seg_mask)
    if stage_times is not None:
        stage_times["tile_mask_ms"] = (time.perf_counter() - t_tile) * 1000.0

    if not tile_imgs:
        return []

    # --- Stage 2: Model predict (batched, Ultralytics handles preprocessing) ---
    t_infer = time.perf_counter()
    results = model.predict(
        source=tile_imgs,
        imgsz=img_size or tile_size,
        conf=conf,
        device=device,
        verbose=False,
        save=False
    )
    if stage_times is not None:
        stage_times["infer_ms"] = (time.perf_counter() - t_infer) * 1000.0

    # --- Stage 3: Map boxes back + cross-tile NMS merge ---
    t_post = time.perf_counter()

    all_boxes = []
    all_scores = []
    all_cls = []

    for res, (tx, ty) in zip(results, offsets):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(np.int32)
        xyxy[:, 0] += tx
        xyxy[:, 1] += ty
        xyxy[:, 2] += tx
        xyxy[:, 3] += ty
        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_cls.append(cls_ids)

    if not all_boxes:
        if stage_times is not None:
            stage_times["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0
        return []

    boxes_np = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores_np = np.concatenate(all_scores, axis=0).astype(np.float32)
    cls_np = np.concatenate(all_cls, axis=0)

    # Cross-tile NMS merge (pure numpy)
    keep = nms_merge(boxes_np, scores_np, cls_np, iou_merge)

    dets_final: list[Det] = []
    for i in keep:
        x1, y1, x2, y2 = boxes_np[i]
        dets_final.append(
            Det(
                cls_id=int(cls_np[i]),
                conf=float(scores_np[i]),
                x1=float(max(0, x1)),
                y1=float(max(0, y1)),
                x2=float(min(w, x2)),
                y2=float(min(h, y2)),
            )
        )

    if stage_times is not None:
        stage_times["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0

    return dets_final


# ---------------------------------------------------------------------------
# Timing-only mode (--time-test)
# ---------------------------------------------------------------------------

def run_time_test(
    model: YOLO,
    images: list[Path],
    out_dir: Path,
    args,
    model_path: Path,
    device: str,
) -> int:
    """Time per-image inference and write a CSV + summary stats txt.

    CSV columns break the per-image time into stages so you can compare the
    'ideal' number (raw model predict) vs the 'realistic' number (full pipeline
    including tiling/ROI mask and the NMS merge postprocess).
    """
    fieldnames = ["image", "tile_mask_ms", "model_infer_ms", "postprocess_ms", "total_ms"]
    rows: list[dict] = []

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path.name}")
            continue

        stage_times: dict = {}
        t0 = time.perf_counter()
        infer_one_image(
            model=model,
            img_bgr=img,
            conf=args.conf,
            tile_size=args.tile_size,
            iou_merge=args.iou_merge,
            device=device,
            img_filename=img_path.name,
            stage_times=stage_times,
            img_size=args.img_size,
            seg_mask=args.seg_mask,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "image": img_path.name,
                "tile_mask_ms": stage_times.get("tile_mask_ms", 0.0),
                "model_infer_ms": stage_times.get("infer_ms", 0.0),
                "postprocess_ms": stage_times.get("postprocess_ms", 0.0),
                "total_ms": total_ms,
            }
        )
        if i % 50 == 0 or i == len(images):
            print(f"Timed {i}/{len(images)} images")

    if not rows:
        print("No images were successfully timed.")
        return 1

    csv_path = out_dir / "inference_times.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
            )

    def _stats(vals: list[float]) -> dict:
        return {
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    sections = [
        ("model_infer_ms  (YOLO predict only - 'ideal')", _stats([r["model_infer_ms"] for r in rows])),
        ("total_ms        (tile/mask + predict + NMS merge - 'realistic')", _stats([r["total_ms"] for r in rows])),
    ]

    txt_path = out_dir / "inference_time_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("INFERENCE TIME TEST\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model:        {model_path}\n")
        f.write(f"Device:       {device}\n")
        f.write(f"Tile size:    {args.tile_size}\n")
        f.write(f"Infer size:   {args.img_size}\n")
        f.write(f"Conf:         {args.conf}\n")
        f.write(f"IoU merge:    {args.iou_merge}\n")
        f.write(f"Images timed: {len(rows)}\n\n")
        for label, s in sections:
            f.write(f"{label}\n")
            f.write(f"  mean:   {s['mean']:.3f} ms\n")
            f.write(f"  median: {s['median']:.3f} ms\n")
            f.write(f"  min:    {s['min']:.3f} ms\n")
            f.write(f"  max:    {s['max']:.3f} ms\n\n")

    print(f"\nPer-image times:  {csv_path}")
    print(f"Summary stats:    {txt_path}")
    return 0


def _save_buffered_vis(
    vis_dets: dict,
    input_dir: Path,
    vis_dir: Path,
    names: dict | None,
    max_imgs: int,
    day_only: bool,
) -> int:
    """Draw + write an evenly spaced subset of buffered detection frames.

    Returns the number of images written."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(vis_dets)
    if day_only:
        candidates = [n for n in candidates if is_day_photo(n)]
    if max_imgs >= 0 and len(candidates) > max_imgs:
        idx = np.linspace(0, len(candidates) - 1, max_imgs).round().astype(int)
        candidates = [candidates[i] for i in idx]
    saved = 0
    for name in candidates:
        img = cv2.imread(str(input_dir / name))
        if img is None:
            continue
        vis = draw_dets(img, vis_dets[name], names=names)
        cv2.imwrite(str(vis_dir / name), vis)
        saved += 1
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiled (3x bottom) YOLO inference with merge back to full image.")
    # parser.add_argument("--model", type=str, default="runs/detect/training_results/runs/beach_detection_20260130_174145/weights/best.pt", help="Path to trained YOLO .pt model")
    parser.add_argument("--model", type=str, default="runs/detect/training_results/runs/beach_detection_20260401_115424/weights/best.pt", help="Path to trained YOLO .pt model")
    parser.add_argument("--input-dir", type=str, default="roboflow_data/Beach Counter.v16i.yolov8/test/images", help="Directory with input images")
    parser.add_argument("--output-dir", type=str, default="runs/detect/training_results/runs/beach_detection_20260401_115424/inference_on_v16_conf02_02", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for display/saving")
    parser.add_argument("--metrics-conf", type=float, default=0.20, help="Confidence threshold for mAP inference (low = full PR curve, like YOLO val)")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile crop size in pixels (should match the tile size used for training)")
    parser.add_argument("--img-size", type=int, default=None, help="Model input size (imgsz) for inference (default: same as --tile-size). Set larger than --tile-size for zoomed inference, e.g. --tile-size 640 --img-size 1280")
    parser.add_argument("--iou-merge", type=float, default=0.50, help="IoU threshold for merge NMS")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--gt-person-class-id", type=int, default=3, help="Class id to use for ground-truth visualization")
    parser.add_argument("--save-img", action="store_true", help="Save annotated full-size images")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format label txts in full-image coords")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA (reduces latency)")
    parser.add_argument("--show", action="store_true", help="Show a live window for stream-like viewing")
    parser.add_argument("--loop", action="store_true", help="Loop over input images continuously")
    parser.add_argument("--delay-ms", type=int, default=0, help="Optional delay between frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    parser.add_argument("--calc-metrics", action="store_true", help="Calculate mAP metrics (requires ground truth labels)")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes in dataset")
    parser.add_argument("--time-test", action="store_true", help="Timing-only mode: write per-image CSV + summary stats, no detections")
    parser.add_argument("--time-test-images", type=str, default=r"D:\Kaanapali", help="Image directory used in --time-test mode")
    parser.add_argument("--time-test-n", type=int, default=500, help="Max images to time in --time-test mode (0 = all)")
    parser.add_argument("--seg-mask", action="store_true", help="Apply the waterline segmentation pipeline's frame masking (inclusion ROI polygon + exclusion triangle) before tiling, and enable the saturation night gate. The detection model then sees pixel-identical inputs to test_viz_segformer.py.")
    parser.add_argument("--allow-night", action="store_true", help="Disable the saturation night gate (only relevant with --seg-mask)")
    parser.add_argument("--no-boxes-csv", action="store_true", help="Do not write the per-detection bounding-box CSV (detections.csv)")
    parser.add_argument("--max-save-img", type=int, default=-1, help="Cap the number of annotated images written to visualized/ (-1 = all). When capped, frames are buffered and an evenly spaced subset is saved at the end.")
    parser.add_argument("--save-day-only", action="store_true", help="Only save annotated images for frames whose local (HST) hour is in the day window [5, 20)")

    args = parser.parse_args()

    # img_size (model input) defaults to tile_size (crop size)
    args.img_size = args.img_size or args.tile_size

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.time_test:
        input_dir = Path(args.time_test_images).resolve()
        if args.output_dir == parser.get_default("output_dir"):
            args.output_dir = str(ROOT_DIR / "time_test_results")
    else:
        input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = out_dir / "labels"
    vis_dir = out_dir / "visualized"
    vis_gt_dir = out_dir / "visualized_gt"
    if args.save_txt:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_img:
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_gt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device.lower()
    if device == "auto":
        device = select_device()
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    model = YOLO(str(model_path))

    if args.half and device == "cuda":
        try:
            model.model.half()
            print("Using FP16 (half precision) for inference")
        except Exception as e:
            print(f"Could not enable FP16: {e}")

    names = getattr(model, "names", None)

    # Build GT class -> model class mapping (handles class ID mismatch between
    # original multi-class labels and single-class / re-indexed model)
    gt_to_model_cls: Dict[int, int] = {}
    if names and args.calc_metrics:
        model_name_to_id = {v.lower(): k for k, v in names.items()}
        # The GT filter keeps only gt_person_class_id; map it to the model's ID
        # for "person" (or whatever name matches).
        for gt_cls_name in ["person"]:
            if gt_cls_name in model_name_to_id:
                gt_to_model_cls[args.gt_person_class_id] = model_name_to_id[gt_cls_name]
        if gt_to_model_cls:
            print(f"GT class remap: {gt_to_model_cls}")

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return 0

    if args.time_test and args.time_test_n > 0:
        images = images[: args.time_test_n]

    print(f"Found {len(images)} images")

    # Warmup: run a few predict calls to reduce first-frame overhead
    # (CUDA kernel compilation, memory allocation, cache population)
    try:
        warm_img = cv2.imread(str(images[0]))
        if warm_img is not None:
            for _ in range(3):
                _ = infer_one_image(
                    model=model,
                    img_bgr=warm_img,
                    conf=args.conf,
                    tile_size=args.tile_size,
                    iou_merge=args.iou_merge,
                    device=device,
                    img_filename=images[0].name,
                    img_size=args.img_size,
                    seg_mask=args.seg_mask,
                )
    except Exception:
        pass

    if args.time_test:
        return run_time_test(model, images, out_dir, args, model_path, device)

    frames_total = 0
    time_total_s = 0.0
    run_idx = 0

    frame_metrics: list[dict] = []
    
    # Variables for metrics calculation
    all_predictions: List[List[Tuple[float, float, float, float, float, int]]] = []
    all_ground_truth: List[List[Tuple[int, float, float, float, float]]] = []
    img_sizes: List[Tuple[int, int]] = []

    # --seg-mask: same saturation night gate as test_viz_segformer.py —
    # IR night frames are skipped entirely and logged to night.txt
    night_gate = args.seg_mask and not args.allow_night
    night_log = (open(out_dir / "night.txt", "w", encoding="utf-8")
                 if night_gate else None)
    n_night = 0

    # Every detection is streamed to a single bounding-box CSV
    boxes_file = None
    boxes_writer = None
    if not args.no_boxes_csv:
        boxes_file = open(out_dir / "detections.csv", "w", newline="",
                          encoding="utf-8")
        boxes_writer = csv.writer(boxes_file)
        boxes_writer.writerow(
            ["image", "cls_id", "cls_name", "conf", "x1", "y1", "x2", "y2"])

    # Buffered visualization: when the number of saved images is capped (or
    # restricted to day photos), dets are kept in memory and an evenly spaced
    # subset is annotated at the end of the run
    buffered_vis = args.save_img and (args.max_save_img >= 0
                                      or args.save_day_only)
    vis_dets: dict = {}

    def _finish() -> None:
        if boxes_file is not None:
            boxes_file.close()
        if night_log is not None:
            night_log.close()
        if buffered_vis and vis_dets:
            saved = _save_buffered_vis(
                vis_dets, input_dir, vis_dir, names,
                max_imgs=args.max_save_img, day_only=args.save_day_only)
            print(f"Saved {saved} annotated example images to '{vis_dir}'")
        if night_gate:
            print(f"{n_night} night frames skipped (see night.txt)")

    while True:
        for img_path in images:
            t0 = time.perf_counter()

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if night_gate:
                sat = mean_saturation(img)
                if sat < NIGHT_SAT_MAX:
                    n_night += 1
                    night_log.write(img_path.name + "\n")
                    night_log.flush()
                    print(f"  NIGHT {img_path.name} (sat {sat:.1f}) - skipped")
                    continue

            stage_times: dict = {}
            dets = infer_one_image(
                model=model,
                img_bgr=img,
                conf=args.conf,
                tile_size=args.tile_size,
                iou_merge=args.iou_merge,
                device=device,
                img_filename=img_path.name,
                stage_times=stage_times,
                img_size=args.img_size,
                seg_mask=args.seg_mask,
            )

            if boxes_writer is not None:
                for d in dets:
                    cls_name = (str(names.get(d.cls_id, d.cls_id))
                                if names else str(d.cls_id))
                    boxes_writer.writerow(
                        [img_path.name, d.cls_id, cls_name, f"{d.conf:.4f}",
                         round(d.x1, 1), round(d.y1, 1),
                         round(d.x2, 1), round(d.y2, 1)])

            vis = None
            gt_boxes: list[tuple[int, float, float, float, float]] = []
            gt_found = False
            if args.show or args.calc_metrics or (args.save_img and not buffered_vis):
                gt_boxes, gt_found = load_gt_boxes(img_path, keep_class_id=int(args.gt_person_class_id))
                vis = draw_dets(img, dets, names=names, gt_boxes=gt_boxes)
            if buffered_vis:
                vis_dets[img_path.name] = dets
            
            # Collect data for metrics calculation if enabled
            if args.calc_metrics:
                h, w = img.shape[:2]
                img_sizes.append((w, h))

                # --- Low-conf inference pass for proper mAP ---
                # YOLO val uses conf=0.001; we do the same so the PR curve
                # is not truncated at the display conf threshold.
                if args.metrics_conf < args.conf:
                    metrics_dets = infer_one_image(
                        model=model,
                        img_bgr=img,
                        conf=args.metrics_conf,
                        tile_size=args.tile_size,
                        iou_merge=args.iou_merge,
                        device=device,
                        img_filename=img_path.name,
                        img_size=args.img_size,
                        seg_mask=args.seg_mask,
                    )
                else:
                    metrics_dets = dets
                
                # Collect predictions in format [x1, y1, x2, y2, conf, cls]
                pred_data = []
                for det in metrics_dets:
                    pred_data.append((det.x1, det.y1, det.x2, det.y2, det.conf, det.cls_id))
                all_predictions.append(pred_data)
                
                # Filter GT to tile region (spatial + ROI mask) then remap classes
                filtered_gt = filter_gt_to_tile_region(
                    gt_boxes, w, h, args.tile_size, img_path.name,
                )
                remapped_gt = [
                    (gt_to_model_cls.get(cls, cls), cx, cy, bw, bh)
                    for cls, cx, cy, bw, bh in filtered_gt
                ]
                all_ground_truth.append(remapped_gt)

            if args.save_txt:
                h, w = img.shape[:2]
                lines = [to_yolo_line(d, w=w, h=h) for d in dets]
                (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))

            if args.save_img and not buffered_vis and vis is not None:
                cv2.imwrite(str(vis_dir / img_path.name), vis)

                if gt_found:
                    vis_gt = draw_dets(img, [], names=None, gt_boxes=gt_boxes)
                    cv2.imwrite(str(vis_gt_dir / img_path.name), vis_gt)

            if args.show and vis is not None:
                cv2.imshow("tiled_inference", vis)

            t1 = time.perf_counter()
            dt = t1 - t0
            frames_total += 1
            time_total_s += dt
            if boxes_file is not None and frames_total % 200 == 0:
                boxes_file.flush()
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_avg = (frames_total / time_total_s) if time_total_s > 0 else 0.0

            frame_metrics.append(
                {
                    "frame": frames_total,
                    "image": img_path.name,
                    "dets": int(len(dets)),
                    "latency_ms": float(dt * 1000.0),
                    "fps_inst": float(fps_inst),
                    "fps_avg": float(fps_avg),
                    **{k: round(v, 2) for k, v in stage_times.items()},
                }
            )

            stage_str = " | ".join(f"{k}={v:.1f}" for k, v in stage_times.items())
            print(
                f"Frame {frames_total} | {img_path.name} | dets={len(dets)} | "
                f"{dt*1000.0:.1f} ms | fps(inst)={fps_inst:.2f} | fps(avg)={fps_avg:.2f} | {stage_str}"
            )

            if args.show:
                wait = args.delay_ms if args.delay_ms > 0 else 1
                key = cv2.waitKey(wait) & 0xFF
                if key == ord("q") or key == 27:
                    cv2.destroyAllWindows()
                    _finish()
                    print(f"\nDone. Outputs in: {out_dir}")
                    return 0
            elif args.delay_ms > 0:
                time.sleep(args.delay_ms / 1000.0)

            if args.max_frames and frames_total >= args.max_frames:
                if args.show:
                    cv2.destroyAllWindows()
                _finish()
                print(f"\nDone. Outputs in: {out_dir}")
                _write_metrics(out_dir, frame_metrics, args, model_path, device,
                              all_predictions, all_ground_truth, img_sizes)
                return 0

        run_idx += 1
        if not args.loop:
            break

    if args.show:
        cv2.destroyAllWindows()

    _finish()
    _write_metrics(out_dir, frame_metrics, args, model_path, device,
                  all_predictions, all_ground_truth, img_sizes)
    print(f"\nDone. Outputs in: {out_dir}")
    return 0


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(v[f])
    return float(v[f] + (k - f) * (v[c] - v[f]))


def _write_metrics(out_dir: Path, frame_metrics: list[dict], args, model_path: Path, device: str,
                    all_predictions: List = None, all_ground_truth: List = None, img_sizes: List = None) -> None:
    if not frame_metrics:
        return

    lat_ms = [float(m["latency_ms"]) for m in frame_metrics]
    total_s = float(sum(lat_ms) / 1000.0)
    frames = int(len(frame_metrics))
    avg_fps = float(frames / total_s) if total_s > 0 else 0.0

    summary = {
        "model": str(model_path),
        "device": device,
        "conf": float(args.conf),
        "metrics_conf": float(args.metrics_conf),
        "tile_size": int(args.tile_size),
        "img_size": int(args.img_size),
        "iou_merge": float(args.iou_merge),
        "seg_mask": bool(getattr(args, "seg_mask", False)),
        "half": bool(getattr(args, "half", False)),
        "frames": frames,
        "total_time_s": total_s,
        "avg_fps": avg_fps,
        "latency_ms": {
            "min": float(min(lat_ms)) if lat_ms else float("nan"),
            "mean": float(sum(lat_ms) / len(lat_ms)) if lat_ms else float("nan"),
            "p50": _percentile(lat_ms, 50),
            "p90": _percentile(lat_ms, 90),
            "p95": _percentile(lat_ms, 95),
            "max": float(max(lat_ms)) if lat_ms else float("nan"),
        },
    }

    # Calculate detection metrics if data is available and metrics calculation is enabled
    if (args.calc_metrics and all_predictions and all_ground_truth and img_sizes):
        print("Calculating detection metrics...")
        print(f"Found {len(all_predictions)} images with predictions")
        gt_count = sum(1 for gts in all_ground_truth if gts)
        print(f"Found {gt_count} images with ground truth labels")
        
        if gt_count == 0:
            print("Warning: No ground truth labels found. Cannot calculate detection metrics.")
            summary["detection_metrics"] = {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": "No ground truth labels found"
            }
        else:
            try:
                det_metrics = calculate_map_metrics(
                    all_predictions=all_predictions,
                    all_ground_truth=all_ground_truth, 
                    img_sizes=img_sizes,
                    conf_threshold=args.metrics_conf,
                    iou_threshold=0.45,  # Standard COCO evaluation threshold
                    num_classes=args.num_classes
                )
                summary["detection_metrics"] = det_metrics
                print(f"mAP50: {det_metrics['map50']:.4f}")
                print(f"mAP50-95: {det_metrics['map50_95']:.4f}")
                print(f"Precision: {det_metrics['precision']:.4f}")
                print(f"Recall: {det_metrics['recall']:.4f}")
                print(f"F1: {det_metrics['f1']:.4f}")
            except Exception as e:
                print(f"Error calculating detection metrics: {e}")
                summary["detection_metrics"] = {
                    "map50": 0.0,
                    "map50_95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "error": str(e)
                }

    payload = {"summary": summary, "frames": frame_metrics}

    json_path = out_dir / "inference_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    txt_path = out_dir / "inference_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("INFERENCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model:         {summary['model']}\n")
        f.write(f"Device:        {summary['device']}\n")
        f.write(f"Half:          {summary['half']}\n")
        f.write(f"Conf:          {summary['conf']}\n")
        f.write(f"Metrics conf:  {summary['metrics_conf']}\n")
        f.write(f"Tile size:     {summary['tile_size']}\n")
        f.write(f"Infer size:    {summary['img_size']}\n")
        f.write(f"IoU merge:     {summary['iou_merge']}\n")
        f.write(f"Frames:        {summary['frames']}\n")
        f.write(f"Total time (s):{summary['total_time_s']:.3f}\n")
        f.write(f"Avg FPS:       {summary['avg_fps']:.3f}\n")
        f.write("\nLatency (ms):\n")
        f.write(f"  min:  {summary['latency_ms']['min']:.3f}\n")
        f.write(f"  mean: {summary['latency_ms']['mean']:.3f}\n")
        f.write(f"  p50:  {summary['latency_ms']['p50']:.3f}\n")
        f.write(f"  p90:  {summary['latency_ms']['p90']:.3f}\n")
        f.write(f"  p95:  {summary['latency_ms']['p95']:.3f}\n")
        f.write(f"  max:  {summary['latency_ms']['max']:.3f}\n")
        
        # Add detection metrics if available
        if "detection_metrics" in summary:
            f.write("\nDETECTION METRICS\n")
            f.write("-" * 80 + "\n")
            det_metrics = summary["detection_metrics"]
            f.write(f"mAP50:         {det_metrics['map50']:.4f}\n")
            f.write(f"mAP50-95:      {det_metrics['map50_95']:.4f}\n")
            f.write(f"Precision:     {det_metrics['precision']:.4f}\n")
            f.write(f"Recall:        {det_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:      {det_metrics['f1']:.4f}\n")
            if "error" in det_metrics:
                f.write(f"Error:         {det_metrics['error']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
