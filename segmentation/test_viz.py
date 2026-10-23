"""
Visualize YOLO26-seg predictions on the test set.

Saves side-by-side panels (image | ground-truth mask | prediction overlay) to
  <run_dir>/test_viz/

Usage (standalone):
    python test_viz.py --model runs/segment/yolo-waterline-seg/train/weights/best.pt
    python test_viz.py --model best.pt --output-dir my_viz --conf 0.25 --max-images 20

Imported by train_fast_yolo.py and called after training completes.
"""
import argparse
import csv
import os
import re
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image

import config
from preprocess_masks import apply_roi_mask, get_tile_offsets


# ---------------------------------------------------------------------------
# Resume support — skip images whose panel already exists in the output dir
# ---------------------------------------------------------------------------
_OUTPUT_RE = re.compile(r"^(\d+)_(.+)$")


def _scan_existing_outputs(output_dir: str) -> tuple[set[str], int]:
    """
    Scan output_dir for existing panel files (named ``NNN_<base>.jpg``).

    Returns a (set_of_base_names, max_index) tuple.  ``base`` is the original
    image stem (without the leading index and without extension), so it can be
    matched against input filenames regardless of the numeric prefix.
    """
    bases: set[str] = set()
    max_idx = 0
    if not os.path.isdir(output_dir):
        return bases, max_idx
    for fname in os.listdir(output_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(fname)[0]
        m = _OUTPUT_RE.match(stem)
        if not m:
            continue
        idx = int(m.group(1))
        bases.add(m.group(2))
        if idx > max_idx:
            max_idx = idx
    return bases, max_idx


# ---------------------------------------------------------------------------
# Daytime filtering — filenames carry a UTC timestamp (..._YYYYMMDD_HHMMSS)
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def _local_hour(fname: str) -> float | None:
    """UTC timestamp in the filename → local wall-clock hour (HST = UTC-10)."""
    m = _TS_RE.search(os.path.basename(fname))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    local = dt + timedelta(hours=config.LOCAL_UTC_OFFSET_H)
    return local.hour + local.minute / 60 + local.second / 3600


_STATS_FIELDS = [
    "image", "roi_px",
    "px_background", "px_basebg", "px_beach", "px_water",
    "pct_background", "pct_basebg", "pct_beach", "pct_water",
    "pct_lowconf_fill", "pct_voronoi_fill",
]


# ---------------------------------------------------------------------------
# Colour palette  (one colour per class, BGRA)
# ---------------------------------------------------------------------------
CLASS_COLORS_BGR = [
    (0,   0,   255),  # 0  background      — red   (should never appear inside polygon)
    (0,   255, 255),  # 1  base-bg-layer   — yellow
    (0,   165, 255),  # 2  beach           — orange
    (255, 50,  50),   # 3  water           — bright blue
]


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an H×W uint8 class-index mask to an H×W×3 BGR image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS_BGR):
        out[mask == cls_id] = color
    return out


def _alpha_overlay(image_bgr: np.ndarray, mask_bgr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend prediction colours on top of the original image."""
    return cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)


def _fill_small_holes(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Close gaps inside each class mask (e.g. YOLO instance-seg slivers and
    flat instance edges) without expanding the class into other predicted classes.
    Water is filled first so gaps between water and the sky/horizon band become
    water, not background or base-background-layer.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    filled = mask.copy()
    # Fill in water-first order so gaps near the waterline take class 3
    for cls_id in [3, 2, 1]:
        binary = (mask == cls_id).astype(np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        fill_region = (closed == 1) & (filled == 0)
        filled[fill_region] = cls_id
    return filled


def _coverage_fill_region(
    mask: np.ndarray,
    image_bgr: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the region eligible for gap-filling, returned as uint8 (255 = fill).

    Priority: explicit roi_mask > non-black interior of a polygon-masked image
    > background components fully enclosed by predicted classes (unmasked-safe
    fallback: a whole failed tile connected to the outside is left alone).
    """
    h, w = mask.shape
    if roi_mask is not None:
        return roi_mask

    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        interior = gray > 10
        if interior.mean() < 0.95:  # polygon-masked input: interior = non-black
            region = np.zeros((h, w), np.uint8)
            region[interior] = 255
            # pull back a couple px so JPEG ringing along the polygon edge
            # doesn't count as interior
            return cv2.erode(region, np.ones((5, 5), np.uint8))

    region = np.zeros((h, w), np.uint8)
    num, comp = cv2.connectedComponents((mask == 0).astype(np.uint8), connectivity=4)
    border_ids = set(np.unique(
        np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]])).tolist())
    for cid in range(1, num):
        if cid not in border_ids:
            region[comp == cid] = 255
    return region


def _fill_uncovered(
    mask: np.ndarray,
    fill_region: np.ndarray,
    max_hole_px: int = 200_000,
    x_weight: int = 4,
) -> np.ndarray:
    """
    Fill uncovered pixels inside fill_region with the class of the nearest
    covered pixel (Voronoi fill).  Tiled inference leaves gaps where an
    instance is missed, a mask is clipped to its bounding box, or a whole
    tile returns nothing — those stay 0 inside the ROI without this.

    The distance transform runs on an x-stretched copy so vertical neighbours
    dominate: the scene is horizontally layered (sky/water/beach), so a missed
    region should inherit the class directly above/below it rather than a
    sideways wedge from a distant covered lobe.

    Components larger than max_hole_px are left as background — a gap that big
    means a whole tile failed, which should stay visible rather than be
    fabricated over.

    Most holes are already filled upstream by tiled_predict's low-confidence
    second pass (the model's own sub-threshold instances); this Voronoi fill is
    the residue fallback for slivers no instance covers at any confidence.
    """
    covered   = mask != 0
    uncovered = ~covered & (fill_region > 0)
    if not uncovered.any():
        return mask

    num, comp = cv2.connectedComponents(uncovered.astype(np.uint8), connectivity=4)
    sizes = np.bincount(comp.ravel(), minlength=num)
    keep  = (sizes <= max_hole_px)
    keep[0] = False
    uncovered &= keep[comp]
    if not uncovered.any():
        return mask

    # distanceTransformWithLabels: sources are the zero pixels of src.
    # Stretch x by x_weight so Euclidean distance penalises horizontal steps.
    h, w  = mask.shape
    sw    = w * x_weight
    m_big   = cv2.resize(mask, (sw, h), interpolation=cv2.INTER_NEAREST)
    unc_big = cv2.resize(uncovered.astype(np.uint8), (sw, h),
                         interpolation=cv2.INTER_NEAREST) > 0
    covered_big = m_big != 0

    src = (~covered_big).astype(np.uint8)           # 0 at covered pixels
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros(labels.max() + 1, dtype=np.uint8)
    lut[labels[covered_big]] = m_big[covered_big]   # each covered px → its class

    filled_big = m_big.copy()
    filled_big[unc_big] = lut[labels[unc_big]]
    filled = cv2.resize(filled_big, (w, h), interpolation=cv2.INTER_NEAREST)

    out = mask.copy()
    out[uncovered] = filled[uncovered]

    # Layer-consistency pass: the scene stacks top→bottom as basebg(1) →
    # water(3) → beach(2), so a pixel should never be *filled* with water below
    # a beach-covered pixel in the same column — flip those to beach.
    # Constraint uses the original predictions only, so it can't cascade.
    beach_col  = mask == 2
    from_below = np.maximum.accumulate(beach_col[::-1], axis=0)[::-1]
    beach_below = np.zeros_like(beach_col)
    beach_below[:-1] = from_below[1:]
    out[uncovered & (out == 3) & beach_below] = 2
    return out


def _pred_to_mask(result, h: int, w: int) -> np.ndarray:
    """
    Convert a single ultralytics Result object to an H×W uint8 class-index mask.
    Each pixel takes the class of whichever instance mask has the highest
    confidence at that location (background = 0).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return mask

    masks_data = result.masks.data.cpu().numpy()   # (N, H', W')
    classes    = result.boxes.cls.cpu().numpy().astype(int)   # (N,)
    confs      = result.boxes.conf.cpu().numpy()   # (N,)

    conf_map   = np.zeros((h, w), dtype=np.float32)

    for i, (m, cls_id, conf) in enumerate(zip(masks_data, classes, confs)):
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        region    = (m_resized > 0.5) & (conf > conf_map)
        mask[region]     = cls_id
        conf_map[region] = conf

    return mask


def tiled_predict(
    model,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float  = 0.45,
    tile_size: int = config.TILE_SIZE,
    overlap: float = 0.0,
    mode: str = config.TILE_MODE,
    tile_height: int = config.TILE_HEIGHT,
    fill_conf: float = 0.0,
    info: dict | None = None,
) -> np.ndarray:
    """
    Simple tiled inference: split image into tiles, run inference, place results back.
    For bottom mode: 3 tiles covering the bottom portion, resized directly to tile_size×tile_size.

    fill_conf: if > 0 and a tile leaves uncovered pixels inside the ROI, that tile
    is re-run at this lower confidence and the extra detections fill only the
    uncovered pixels — the model labels its own misses (e.g. a beach instance at
    conf 0.06 covering the foreground strip that the 0.15 pass dropped).
    info: optional dict that receives "lowconf_px" (bool map of pixels labelled
    by the low-confidence pass) so callers can measure how much of the mask came
    from below-threshold detections.
    """
    full_h, full_w = image_bgr.shape[:2]
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    lowconf_px = np.zeros((full_h, full_w), dtype=bool) if info is not None else None

    # Non-black interior of a polygon-masked input; low-conf fills stay inside it.
    interior = None
    if fill_conf > 0:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        inside = gray > 10
        if inside.mean() < 0.95:          # masked input
            interior = inside

    def _lowconf_mask(tile_resized):
        """Rerun one tile at fill_conf; returns the low-conf class mask."""
        r = model.predict(tile_resized, conf=fill_conf, iou=iou, verbose=False)[0]
        return _pred_to_mask(r, tile_size, tile_size)

    if mode == "bottom":
        col_edges = [round(full_w * c / 3) for c in range(4)]
        y0 = max(0, full_h - tile_height)
        y1 = full_h
        th = y1 - y0
        nat_tw = full_w // 3
        pad = int(round(overlap * nat_tw))

        for i in range(3):
            cx0 = col_edges[i]      # canonical left edge (no overlap)
            cx1 = col_edges[i + 1]  # canonical right edge (no overlap)

            # Expand tile by pad pixels on each side (clamped to image bounds)
            x0 = max(0, cx0 - pad)
            x1 = min(full_w, cx1 + pad)
            l_crop = cx0 - x0   # pixels to strip on left after inference
            r_crop = x1 - cx1   # pixels to strip on right after inference
            ext_tw = x1 - x0

            tile = image_bgr[y0:y1, x0:x1]
            tile_resized = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            results = model.predict(tile_resized, conf=conf, iou=iou, verbose=False)
            tile_mask = _pred_to_mask(results[0], tile_size, tile_size)

            # Resize to extended width, then crop back to canonical region
            tile_mask_ext = cv2.resize(tile_mask, (ext_tw, th), interpolation=cv2.INTER_NEAREST)
            strip = tile_mask_ext[:, l_crop: ext_tw - r_crop]

            if fill_conf > 0:
                need = strip == 0
                if interior is not None:
                    need &= interior[y0:y1, cx0:cx1]
                if need.any():
                    low_ext = cv2.resize(
                        _lowconf_mask(tile_resized), (ext_tw, th),
                        interpolation=cv2.INTER_NEAREST)
                    low_strip = low_ext[:, l_crop: ext_tw - r_crop]
                    fill_px = need & (low_strip != 0)
                    strip = strip.copy()
                    strip[fill_px] = low_strip[fill_px]
                    if lowconf_px is not None:
                        lowconf_px[y0:y1, cx0:cx1] |= fill_px

            full_mask[y0:y1, cx0:cx1] = strip

    else:  # mode == "all"
        grid_tw = full_w // 3
        grid_th = full_h // 2

        for row in range(2):
            for col in range(3):
                x0 = col * grid_tw
                x1 = (col + 1) * grid_tw
                y0 = row * grid_th
                y1 = (row + 1) * grid_th

                tile = image_bgr[y0:y1, x0:x1]
                tile_resized = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

                results = model.predict(tile_resized, conf=conf, iou=iou, verbose=False)
                tile_mask = _pred_to_mask(results[0], tile_size, tile_size)

                tile_mask_full = cv2.resize(tile_mask, (grid_tw, grid_th), interpolation=cv2.INTER_NEAREST)
                strip = tile_mask_full

                if fill_conf > 0:
                    need = strip == 0
                    if interior is not None:
                        need &= interior[y0:y1, x0:x1]
                    if need.any():
                        low_full = cv2.resize(
                            _lowconf_mask(tile_resized), (grid_tw, grid_th),
                            interpolation=cv2.INTER_NEAREST)
                        fill_px = need & (low_full != 0)
                        strip = strip.copy()
                        strip[fill_px] = low_full[fill_px]
                        if lowconf_px is not None:
                            lowconf_px[y0:y1, x0:x1] |= fill_px

                full_mask[y0:y1, x0:x1] = strip

    if info is not None:
        info["lowconf_px"] = lowconf_px
    return full_mask


def run_test_viz(
    model_path: str,
    output_dir: str | None = None,
    conf: float = 0.25,
    iou: float  = 0.45,
    max_images: int = 0,
    dataset_dir: str = config.OUTPUT_DATASET,
    split: str = "test",
    tiled: bool = True,
    overlap: float = 0.15,
    tile_mode: str = config.TILE_MODE,
    tile_height: int = config.TILE_HEIGHT,
    mask_first: bool = True,
    save_tiles: bool = False,
    skip_existing: bool = True,
    gap_fill: bool = True,
    fill_conf: float = 0.05,
    save_masks: bool = False,
    time_filter: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Run inference on every image in <dataset_dir>/<split>/ and save
    side-by-side panels to output_dir.

    When tiled=True (default), each image is split into 6 tiles, the model
    runs on each tile at 640×640, and results are stitched back to full resolution.
    Overlap fraction controls how much neighbouring tiles overlap (default 0.15).

    Returns the output_dir path.
    """
    from ultralytics import YOLO

    if output_dir is None:
        run_dir = os.path.dirname(os.path.dirname(model_path))  # …/weights/best.pt → run root
        output_dir = os.path.join(run_dir, "test_viz")

    os.makedirs(output_dir, exist_ok=True)

    model  = YOLO(model_path)

    # Use <dataset_dir>/<split>/ when it exists; otherwise treat dataset_dir
    # itself as a plain folder of images (no splits, no ground truth).
    if os.path.isdir(os.path.join(dataset_dir, split)):
        split_dir = os.path.join(dataset_dir, split)
    else:
        split_dir = dataset_dir

    image_files = sorted(
        f for f in os.listdir(split_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
        if not f.lower().endswith("_mask.png")
    )

    if time_filter == "day":
        stamped = [(f, _local_hour(f)) for f in image_files]
        if not any(h is not None for _, h in stamped):
            if verbose:
                print("--time day ignored: no filenames carry a UTC timestamp")
        else:
            image_files = [f for f, h in stamped
                           if h is not None and config.DAY_START_H <= h < config.DAY_END_H]
            skipped = len(stamped) - len(image_files)
            if verbose:
                print(f"--time day: kept {len(image_files)} images "
                      f"(local 05:00-20:00 = UTC-10 from filename), skipped {skipped}")

    if max_images > 0:
        image_files = image_files[:max_images]

    # --- Mask / stats outputs ---
    masks_dir    = None
    stats_file   = None
    stats_writer = None
    if save_masks:
        masks_dir  = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        stats_path   = os.path.join(output_dir, "mask_stats.csv")
        stats_file   = open(stats_path, "a", newline="", encoding="utf-8")
        stats_writer = csv.DictWriter(stats_file, fieldnames=_STATS_FIELDS)
        if stats_file.tell() == 0:
            stats_writer.writeheader()

    # --- Resume support: skip images whose panel already exists ---
    existing_bases: set[str] = set()
    start_idx = 1
    if skip_existing:
        existing_bases, max_idx = _scan_existing_outputs(output_dir)
        if existing_bases:
            start_idx = max_idx + 1

            def _done(f: str) -> bool:
                stem = os.path.splitext(f)[0]
                if stem not in existing_bases:
                    return False
                if masks_dir is not None and not os.path.exists(
                        os.path.join(masks_dir, stem + "_mask.png")):
                    return False   # panel exists but mask/stats still missing
                return True

            n_before = len(image_files)
            image_files = [f for f in image_files if not _done(f)]
            if verbose:
                print(f"Skipping {n_before - len(image_files)} already-done images; "
                      f"resuming at index {start_idx}")

    if verbose:
        mode = "tiled (3×2)" if tiled else "full-image"
        print(f"Visualizing {len(image_files)} images from '{split_dir}' [{mode} inference]")
        print(f"Saving panels to '{output_dir}'\n")

    for idx, fname in enumerate(image_files, start_idx):
        img_path  = os.path.join(split_dir, fname)
        base      = os.path.splitext(fname)[0]
        mask_path = os.path.join(split_dir, base + "_mask.png")

        image_bgr = cv2.imread(img_path)
        h, w      = image_bgr.shape[:2]

        # --- Ground truth (optional) ---
        has_gt = os.path.exists(mask_path)
        if has_gt:
            gt_mask    = np.array(Image.open(mask_path))
            gt_colored = _colorize_mask(gt_mask)
            gt_overlay = _alpha_overlay(image_bgr, gt_colored)

        # --- Prediction (tiled or full-image) ---
        infer_bgr = apply_roi_mask(image_bgr, fname) if mask_first else image_bgr
        pred_info: dict = {}
        if tiled:
            pred_mask = tiled_predict(model, infer_bgr, conf=conf, iou=iou, overlap=overlap,
                                     mode=tile_mode if tile_mode else config.TILE_MODE,
                                     tile_height=tile_height if tile_height else config.TILE_HEIGHT,
                                     fill_conf=fill_conf,
                                     info=pred_info if save_masks else None)
        else:
            results   = model.predict(infer_bgr, conf=conf, iou=iou, verbose=False)
            pred_mask = _pred_to_mask(results[0], h, w)

        pred_mask = _fill_small_holes(pred_mask)
        fill_region = _coverage_fill_region(pred_mask, infer_bgr)
        model_mask  = pred_mask          # model output only (before Voronoi fill)
        if gap_fill:
            pred_mask = _fill_uncovered(pred_mask, fill_region)
        pred_colored = _colorize_mask(pred_mask)
        pred_overlay = _alpha_overlay(image_bgr, pred_colored)

        # --- Per-tile debugging panels (optional) ---
        if tiled and save_tiles:
            tile_debug_dir = os.path.join(output_dir, "tiles", base)
            os.makedirs(tile_debug_dir, exist_ok=True)

            mode = tile_mode if tile_mode else config.TILE_MODE
            tile_h = tile_height if tile_height else config.TILE_HEIGHT
            
            if mode == "bottom":
                # Use exact same calculation as training script
                col_edges = [round(w * c / 3) for c in range(4)]
                y_start = h - tile_h
                offsets = [(col_edges[i], y_start, col_edges[i + 1], h) for i in range(3)]
            else:
                # Full 3×2 grid
                tile_w = w // 3
                tile_h_grid = h // 2
                offsets = [(col * tile_w, row * tile_h_grid, (col + 1) * tile_w, (row + 1) * tile_h_grid) 
                          for row in range(2) for col in range(3)]
            
            for tile_idx, (x0, y0, x1, y1) in enumerate(offsets):
                tile_img    = image_bgr[y0:y1, x0:x1]
                tile_resized = cv2.resize(tile_img, (config.TILE_SIZE, config.TILE_SIZE), interpolation=cv2.INTER_LINEAR)

                # Run inference on this tile only
                results   = model.predict(tile_resized, conf=conf, iou=iou, verbose=False)
                tile_mask = _pred_to_mask(results[0], config.TILE_SIZE, config.TILE_SIZE)
                tile_colored = _colorize_mask(tile_mask)
                tile_overlay = _alpha_overlay(tile_resized, tile_colored)

                # Simple 2-panel tile visualization (image | prediction)
                tile_panel = np.hstack([tile_resized, tile_overlay])
                tile_path = os.path.join(tile_debug_dir, f"tile{tile_idx}.jpg")
                cv2.imwrite(tile_path, tile_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # --- Assemble panel (drop GT column when no ground truth) ---
        if has_gt:
            columns = [image_bgr, gt_overlay, pred_overlay]
            labels  = ["Original", "Ground Truth", "Prediction"]
        else:
            columns = [image_bgr, pred_overlay]
            labels  = ["Original", "Prediction"]

        # --- Legend bar ---
        bar_h  = 28
        bar_w  = w * len(columns)
        bar    = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        seg_w  = bar_w // len(config.CLASS_NAMES)
        for cls_id, (name, color) in enumerate(zip(config.CLASS_NAMES, CLASS_COLORS_BGR)):
            x0 = cls_id * seg_w
            x1 = x0 + seg_w
            bar[:, x0:x1] = color
            cv2.putText(bar, f"{cls_id}:{name}", (x0 + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Header labels ---
        header_h = 22
        header   = np.zeros((header_h, bar_w, 3), dtype=np.uint8)
        for col_idx, label in enumerate(labels):
            cv2.putText(header, label, (col_idx * w + 4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        panel = np.vstack([
            header,
            np.hstack(columns),
            bar,
        ])

        out_path = os.path.join(output_dir, f"{idx:03d}_{base}.jpg")
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # --- Save class-index mask + per-class area stats ---
        stat_str = ""
        if stats_writer is not None:
            Image.fromarray(pred_mask.astype(np.uint8)).save(
                os.path.join(masks_dir, base + "_mask.png"))
            interior = fill_region > 0
            if interior.sum() < 0.05 * pred_mask.size:
                interior = pred_mask != 0   # unmasked input: polygon is unknown
            roi_n  = max(1, int(interior.sum()))
            counts = np.bincount(pred_mask[interior].ravel(),
                                 minlength=len(config.CLASS_NAMES))
            vor_px = int((interior & (model_mask == 0)).sum())
            low_px = int((interior & pred_info["lowconf_px"]).sum()) \
                if "lowconf_px" in pred_info else 0
            row = {
                "image": fname, "roi_px": roi_n,
                "px_background": int(counts[0]), "px_basebg": int(counts[1]),
                "px_beach": int(counts[2]),    "px_water": int(counts[3]),
                "pct_background": round(100 * counts[0] / roi_n, 2),
                "pct_basebg":     round(100 * counts[1] / roi_n, 2),
                "pct_beach":      round(100 * counts[2] / roi_n, 2),
                "pct_water":      round(100 * counts[3] / roi_n, 2),
                "pct_lowconf_fill":  round(100 * low_px / roi_n, 2),
                "pct_voronoi_fill":  round(100 * vor_px / roi_n, 2),
            }
            stats_writer.writerow(row)
            stats_file.flush()
            stat_str = (f"  [beach {row['pct_beach']}% | water {row['pct_water']}%"
                        f" | bg {row['pct_background']}%]")

        if verbose:
            print(f"  [{idx:3d}/{len(image_files)}] {out_path}{stat_str}")

    if stats_file is not None:
        stats_file.close()

    if verbose:
        print(f"\nDone. {len(image_files)} panels saved to '{output_dir}'")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize YOLO26-seg predictions vs ground truth on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          required=True,
                   help="Path to trained weights, e.g. runs/.../weights/best.pt")
    p.add_argument("--output-dir",     default=None,
                   help="Where to save panels (default: <run_dir>/test_viz/)")
    p.add_argument("--dataset-dir",    default=config.OUTPUT_DATASET,
                   help="Preprocessed dataset root")
    p.add_argument("--split",          default="test",
                   choices=["train", "valid", "test"])
    p.add_argument("--conf",           type=float, default=0.25,
                   help="Detection confidence for the main pass")
    p.add_argument("--fill-conf",      type=float, default=0.05, dest="fill_conf",
                   help="Second-pass confidence that fills still-uncovered ROI pixels "
                        "with the model's own low-confidence instances (0 = off)")
    p.add_argument("--iou",            type=float, default=0.45,
                   help="IoU threshold for NMS")
    p.add_argument("--max-images",     type=int,   default=0,
                   help="Max images to visualize (0 = all)")
    p.add_argument("--no-tile",        action="store_true",
                   help="Run inference on the full image instead of tiling (default: tiled)")
    p.add_argument("--overlap",        type=float, default=0.15,
                   help="Tile overlap fraction for tiled inference (0.0 = no overlap, 0.5 = 50% overlap)")
    p.add_argument("--tile-mode",      default=config.TILE_MODE,
                   choices=["all", "bottom"],
                   help="Tiling mode: 'all' for full grid, 'bottom' for 3 bottom tiles")
    p.add_argument("--tile-height",    type=int, default=config.TILE_HEIGHT,
                   help="Native tile height for 'bottom' mode (before resize)")
    p.add_argument("--no-mask",        action="store_true",
                   help="Skip the ROI polygon mask step (input photos are "
                        "already masked, or you want raw-frame inference)")
    p.add_argument("--save-tiles",     action="store_true",
                   help="Save per-tile debugging panels to <output_dir>/tiles/<image>/")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-render all panels even if they already exist in the output dir")
    p.add_argument("--no-gap-fill",    action="store_true",
                   help="Don't Voronoi-fill uncovered ROI pixels with the "
                        "nearest predicted class")
    p.add_argument("--save-masks",     action="store_true",
                   help="Also save class-index masks to <output_dir>/masks/ and "
                        "per-class area stats to <output_dir>/mask_stats.csv")
    p.add_argument("--time",           default=None, choices=["day"],
                   help="Only process images in this local-time window "
                        "('day' = 05:00-20:00 HST; filename timestamps are UTC)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test_viz(
        model_path=args.model,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        max_images=args.max_images,
        dataset_dir=args.dataset_dir,
        split=args.split,
        tiled=not args.no_tile,
        overlap=args.overlap,
        tile_mode=args.tile_mode,
        tile_height=args.tile_height,
        mask_first=not args.no_mask,
        save_tiles=args.save_tiles,
        skip_existing=not args.no_skip_existing,
        gap_fill=not args.no_gap_fill,
        fill_conf=args.fill_conf,
        save_masks=args.save_masks,
        time_filter=args.time,
    )
