#!/usr/bin/env python3
r"""
Run tiled YOLO inference on a directory of images and export the merged
full-resolution predictions as a YOLOv8-formatted dataset with a
70/20/10 train/val/test split.

This reuses the tiled-inference pipeline from `training/run_tiled_inference.py`
(images are sliced into 640x640 tiles, inferred on, then detections are mapped
back to full-image coordinates and NMS-merged across tiles) so the exported
labels are in *full image resolution* YOLO format - not tile-local coords.

Naming conventions from the rest of the repo are honored:
  - `youtube_snapshot_<digits>...` / `youtube_snapshot_Kaanapali...` images
    get the ROI exclusion mask applied and use the 3-bottom-tiles layout.
  - All other images use the full overlapping tile grid (no mask).

Usage example:

  python tiled_inference_to_dataset.py \
    --model 'C:\Users\JR\Documents\GitHub\beach-obj-detection-training\training_results_old\yolo26l_beach_detection_20260523_074516\weights\best.pt' \
    --input-dir "D:\Kaanapali" \
    --conf 0.20

Output goes to:  <input-dir parent>/<input-dir name>_yolo/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
  data.yaml

Deduplication:
  By default, any input image whose stem matches a file in the reference
  Roboflow dataset (--roboflow-dir) is skipped during inference so the
  exported dataset never duplicates images already in the training set.

  Pass --dedup to retroactively clean an already-generated _yolo dataset:
  it scans all splits in the output dir and removes images + labels whose
  stems match files in the reference dataset, then exits (no inference).

  python tiled_inference_to_dataset.py --dedup
  python tiled_inference_to_dataset.py --dedup --input-dir "D:\Kaanapali"
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Path setup: reuse tiled-inference + preprocessing helpers from the repo
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
TRAINING_DIR = ROOT_DIR / "training"
PREPROC_DIR = ROOT_DIR / "preprocessing"
for d in (TRAINING_DIR, PREPROC_DIR):
    if d.exists() and str(d) not in sys.path:
        sys.path.append(str(d))

from utils import patch_torch_load, select_device  # noqa: E402
from run_tiled_inference import (  # noqa: E402
    collect_images,
    crop_and_mask_tiles,
    infer_one_image,
    nms_merge,
    to_yolo_line,
    Det,
)

patch_torch_load()


# ---------------------------------------------------------------------------
# Dataset layout helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Roboflow renames files as:  <orig_stem>_<ext>.rf.<32 hex chars>
# e.g.  youtube_snapshot_Kaanapali_20260105_205654.jpg
#   ->  youtube_snapshot_Kaanapali_20260105_205654_jpg.rf.a1b2c3d4...
_ROBOFLOW_SUFFIX_RE = re.compile(
    r"_(jpg|jpeg|png|bmp|tif|tiff|webp)\.rf\.[a-f0-9]{32}$",
    re.IGNORECASE,
)


def normalize_roboflow_stem(stem: str) -> str:
    """Strip the Roboflow `_jpg.rf.<hash>` suffix to recover the original stem.

    e.g. "foo_jpg.rf.a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" -> "foo"
    If the stem doesn't match the Roboflow pattern, it's returned unchanged.
    """
    m = _ROBOFLOW_SUFFIX_RE.search(stem)
    if m:
        return stem[: m.start()]
    return stem


def build_split_dirs(out_root: Path) -> dict[str, dict[str, Path]]:
    """Create train/valid/test {images, labels} subdirs and return them."""
    layout: dict[str, dict[str, Path]] = {}
    for split in ("train", "valid", "test"):
        img_dir = out_root / split / "images"
        lbl_dir = out_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        layout[split] = {"images": img_dir, "labels": lbl_dir}
    return layout


def split_images(images: list[Path], train_r: float, val_r: float,
                 test_r: float, seed: int) -> dict[str, list[Path]]:
    """Shuffle (seeded) and partition images into train/valid/test lists."""
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        # Normalize so ratios sum to 1.0
        train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    shuffled = images[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    # Remainder goes to test so the counts always sum to n
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def write_data_yaml(out_root: Path, model: YOLO) -> None:
    """Write a YOLOv8 data.yaml pointing at the three splits."""
    names = getattr(model, "names", None)
    if not names:
        return

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for cls_id, cls_name in names.items():
            f.write(f"  {cls_id}: {cls_name}\n")


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def collect_roboflow_stems(roboflow_dir: Path) -> set[str]:
    """Collect normalized image stems from all splits in a Roboflow-style YOLO dataset.

    Roboflow appends `_jpg.rf.<hash>` to filenames; this strips that suffix so
    stems match the original filenames (e.g. in the Kaanapali input dir).

    Expected layout:  <roboflow_dir>/{train,valid,test}/images/*
    """
    stems: set[str] = set()
    if not roboflow_dir.exists():
        return stems

    for split in ("train", "valid", "test"):
        img_dir = roboflow_dir / split / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stems.add(normalize_roboflow_stem(p.stem))
    return stems


def dedup_existing_dataset(out_root: Path, roboflow_stems: set[str]) -> int:
    """Remove images + labels from an existing _yolo dataset whose stems
    match the reference set.  Returns the number of image files removed.

    Handles the `_new` suffix that the inference step appends to filenames:
    e.g. `foo_new.jpg` is checked against reference stem `foo`.
    """
    removed = 0
    for split in ("train", "valid", "test"):
        img_dir = out_root / split / "images"
        lbl_dir = out_root / split / "labels"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            # Strip the "_new" suffix to recover the original stem
            stem = p.stem
            if stem.endswith("_new"):
                stem = stem[: -len("_new")]
            if stem in roboflow_stems:
                p.unlink()
                lbl = lbl_dir / f"{p.stem}.txt"
                if lbl.exists():
                    lbl.unlink()
                removed += 1
                print(f"  removed [{split}] {p.name}")
    return removed


# ---------------------------------------------------------------------------
# Pipelined inference helpers (prefetch → GPU → postprocess)
# ---------------------------------------------------------------------------

@dataclass
class PrefetchedImage:
    """Result of the prefetch stage: image read + tiles cropped + masked."""
    img_path: Path
    img: np.ndarray
    tile_imgs: list
    offsets: list
    h: int
    w: int


@dataclass
class InferenceResult:
    """Result of the GPU inference stage: tile predictions ready for NMS."""
    prefetched: PrefetchedImage
    results: list  # ultralytics Results objects, one per tile


@dataclass
class PostprocessResult:
    """Result of the postprocess stage: final dets + CSV rows."""
    img_path: Path
    split_name: str
    out_name: str
    dets: list
    csv_rows: list  # list of [split, image, cls, cx, cy, bw, bh, conf]


def prefetch_image(img_path: Path, tile_size: int) -> PrefetchedImage | None:
    """Stage 1 (CPU/I/O): read image from disk, crop + mask tiles."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    tile_imgs, offsets = crop_and_mask_tiles(img, tile_size, img_path.name)
    return PrefetchedImage(img_path=img_path, img=img, tile_imgs=tile_imgs,
                           offsets=offsets, h=h, w=w)


def run_gpu_inference(model, prefetched: PrefetchedImage, conf: float,
                      tile_size: int, device: str) -> InferenceResult:
    """Stage 2 (GPU): run model.predict on the prefetched tiles."""
    if not prefetched.tile_imgs:
        return InferenceResult(prefetched=prefetched, results=[])
    results = model.predict(
        source=prefetched.tile_imgs,
        imgsz=tile_size, conf=conf, device=device,
        verbose=False, save=False,
    )
    return InferenceResult(prefetched=prefetched, results=results)


def postprocess_inference(
    result: InferenceResult, iou_merge: float,
    split_name: str, img_dst: Path, lbl_dst: Path, img_path: Path,
    save_visualized: bool, vis_dir: Path, names,
) -> PostprocessResult:
    """Stage 3 (CPU/I/O): NMS merge, write label, copy image, optionally viz."""
    prefetched = result.prefetched
    w, h = prefetched.w, prefetched.h
    out_name = img_dst.name

    # --- NMS merge across tiles ---
    all_boxes, all_scores, all_cls = [], [], []
    for res, (tx, ty) in zip(result.results, prefetched.offsets):
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

    dets: list[Det] = []
    if all_boxes:
        boxes_np = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores_np = np.concatenate(all_scores, axis=0).astype(np.float32)
        cls_np = np.concatenate(all_cls, axis=0)
        keep = nms_merge(boxes_np, scores_np, cls_np, iou_merge)
        for i in keep:
            x1, y1, x2, y2 = boxes_np[i]
            dets.append(Det(
                cls_id=int(cls_np[i]), conf=float(scores_np[i]),
                x1=float(max(0, x1)), y1=float(max(0, y1)),
                x2=float(min(w, x2)), y2=float(min(h, y2)),
            ))

    # --- Write label file (standard 5-col YOLO) ---
    lines = [to_yolo_line(d, w=w, h=h) for d in dets]
    lbl_dst.write_text("\n".join(lines))

    # --- Copy original image ---
    shutil.copy2(img_path, img_dst)

    # --- Optional visualization ---
    if save_visualized:
        from run_tiled_inference import draw_dets
        vis = draw_dets(prefetched.img, dets, names=names)
        cv2.imwrite(str(vis_dir / out_name), vis)

    # --- Build CSV rows ---
    csv_rows = []
    for d in dets:
        cx = ((d.x1 + d.x2) / 2.0) / w
        cy = ((d.y1 + d.y2) / 2.0) / h
        bw = (d.x2 - d.x1) / w
        bh = (d.y2 - d.y1) / h
        csv_rows.append([
            split_name, out_name, d.cls_id,
            f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}",
            f"{d.conf:.6f}",
        ])

    return PostprocessResult(
        img_path=prefetched.img_path, split_name=split_name,
        out_name=out_name, dets=dets, csv_rows=csv_rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run tiled YOLO inference on a directory of images and export "
            "full-resolution predictions as a YOLOv8 dataset (70/20/10 split)."
        ),
    )
    parser.add_argument(
        "--model", type=str,
        default=r"C:\Users\JR\Documents\GitHub\beach-obj-detection-training\training_results_old\yolo26l_beach_detection_20260523_074516\weights\best.pt",
        help="Path to trained YOLO .pt model",
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=r"D:\Kaanapali",
        help="Directory containing input images (any of the repo's naming conventions).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.20,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--tile-size", type=int, default=640,
        help="Tile size used for inference (must match training tile size)",
    )
    parser.add_argument(
        "--iou-merge", type=float, default=0.50,
        help="IoU threshold for cross-tile NMS merge",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Fraction of images for training (default 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.20,
        help="Fraction of images for validation (default 0.20)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10,
        help="Fraction of images for testing (default 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val/test split",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use FP16 inference on CUDA",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of prefetch + postprocess worker threads each (default 4). "
             "Prefetch threads read images and tile them ahead of GPU inference; "
             "postprocess threads do NMS + file writing. GPU inference itself "
             "stays single-threaded for safety.",
    )
    parser.add_argument(
        "--save-visualized", action="store_true",
        help="Also save annotated full-resolution images alongside the dataset",
    )
    parser.add_argument(
        "--roboflow-dir", type=str,
        default=r"roboflow_data\Beach Counter.v16i.yolov8",
        help="Reference Roboflow dataset dir used for dedup (train/valid/test/images layout).",
    )
    parser.add_argument(
        "--dedup", action="store_true",
        help="Dedup-only mode: remove images+labels from the existing _yolo output "
             "whose stems match files in --roboflow-dir, then exit (no inference).",
    )
    parser.add_argument(
        "--no-dedup-filter", action="store_true",
        help="Disable the default inference-time dedup filter (which skips input "
             "images whose stems match files in --roboflow-dir).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory not found or is not a directory: {input_dir}")
        return 1

    # Output dir: sibling of input dir, with "_yolo" appended to its name
    out_root = input_dir.parent / f"{input_dir.name}_yolo"

    roboflow_dir = Path(args.roboflow_dir)
    if not roboflow_dir.is_absolute():
        roboflow_dir = (ROOT_DIR / roboflow_dir).resolve()

    # ------------------------------------------------------------------
    # --dedup: standalone dedup mode (no inference, no model needed)
    # ------------------------------------------------------------------
    if args.dedup:
        if not out_root.exists():
            print(f"[ERROR] Output dataset not found: {out_root}")
            print("   Run inference first, then use --dedup to clean it up.")
            return 1

        roboflow_stems = collect_roboflow_stems(roboflow_dir)
        if not roboflow_stems:
            print(f"[WARN] No reference images found in: {roboflow_dir}")
            print("   Nothing to dedup against.")
            return 0

        print("Dedup-only mode")
        print("=" * 80)
        print(f"Output dataset:  {out_root}")
        print(f"Reference dir:   {roboflow_dir}")
        print(f"Reference stems: {len(roboflow_stems)}")
        print()
        print("Removing matching images + labels...")
        removed = dedup_existing_dataset(out_root, roboflow_stems)
        print()
        print("=" * 80)
        print(f"[OK] Dedup complete! Removed {removed} image(s) + their labels.")
        print(f"  Dataset: {out_root}")
        return 0

    # ------------------------------------------------------------------
    # Normal inference mode
    # ------------------------------------------------------------------
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device.lower()
    if device == "auto":
        device = select_device()

    # Collect reference stems for inference-time dedup filter (on by default)
    roboflow_stems: set[str] = set()
    if not args.no_dedup_filter:
        roboflow_stems = collect_roboflow_stems(roboflow_dir)
        if roboflow_stems:
            print(f"Dedup filter: ON  ({len(roboflow_stems)} reference stems from {roboflow_dir})")
        else:
            print(f"Dedup filter: no reference images found at {roboflow_dir} (skipping filter)")
    else:
        print("Dedup filter: OFF (--no-dedup-filter)")
    print()

    print("Tiled Inference -> YOLOv8 Dataset Export")
    print("=" * 80)
    print(f"Model:        {model_path}")
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {out_root}")
    print(f"Tile size:    {args.tile_size}")
    print(f"Conf thresh:  {args.conf}")
    print(f"IoU merge:    {args.iou_merge}")
    print(f"Device:       {device}")
    print(f"Split:        {args.train_ratio:.2f} / {args.val_ratio:.2f} / "
          f"{args.test_ratio:.2f}  (seed={args.seed})")
    print()

    # ------------------------------------------------------------------
    # Step 1: Collect + split images
    # ------------------------------------------------------------------
    images = collect_images(input_dir)
    if not images:
        print(f"[ERROR] No images found in: {input_dir}")
        return 1
    print(f"Found {len(images)} images")

    # Filter out images whose stems match the reference dataset (dedup)
    if roboflow_stems:
        before = len(images)
        images = [p for p in images if p.stem not in roboflow_stems]
        skipped = before - len(images)
        if skipped:
            print(f"  [SKIP] {skipped} image(s) already in reference dataset")
        if not images:
            print("[ERROR] All images were filtered out by dedup. Nothing to do.")
            return 0

    splits = split_images(
        images, args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
    )
    print(f"  train: {len(splits['train'])}")
    print(f"  valid: {len(splits['valid'])}")
    print(f"  test:  {len(splits['test'])}")
    print()

    # Reverse lookup: image path -> split name
    split_of: dict[Path, str] = {}
    for split_name, imgs in splits.items():
        for p in imgs:
            split_of[p] = split_name

    layout = build_split_dirs(out_root)
    vis_dir = out_root / "visualized"
    if args.save_visualized:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    if args.half and device == "cuda":
        try:
            model.model.half()
            print("Using FP16 (half precision) for inference")
        except Exception as e:
            print(f"Could not enable FP16: {e}")
    names = getattr(model, "names", None)
    print()

    # Warmup so the first image doesn't include CUDA init overhead
    try:
        warm_img = cv2.imread(str(images[0]))
        if warm_img is not None:
            for _ in range(3):
                _ = infer_one_image(
                    model=model, img_bgr=warm_img, conf=args.conf,
                    tile_size=args.tile_size, iou_merge=args.iou_merge,
                    device=device, img_filename=images[0].name,
                )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 3: Pipelined tiled inference (prefetch → GPU → postprocess)
    # ------------------------------------------------------------------
    n_workers = max(1, args.workers)
    print(f"Running tiled inference and writing dataset...  (workers={n_workers})")
    print("-" * 80)

    total_dets = 0
    images_with_dets = 0
    frames = 0
    t_start = time.perf_counter()

    # Single consolidated CSV with all detections (including confidence)
    csv_path = out_root / "detections.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["split", "image", "class_id", "cx", "cy", "bw", "bh", "conf"])

    # Pre-compute output paths for each image (needed by postprocess stage)
    out_paths: dict[Path, tuple[str, Path, Path]] = {}  # img_path -> (split, img_dst, lbl_dst)
    for img_path in images:
        sn = split_of[img_path]
        out_stem = f"{img_path.stem}_new"
        out_name = f"{out_stem}{img_path.suffix}"
        out_paths[img_path] = (
            sn,
            layout[sn]["images"] / out_name,
            layout[sn]["labels"] / f"{out_stem}.txt",
        )

    # --- Pipelined loop ---
    # Prefetch pool: reads images + tiles them (CPU/I/O, overlaps with GPU)
    # Postprocess pool: NMS merge + writes files (CPU/I/O, overlaps with GPU)
    with ThreadPoolExecutor(max_workers=n_workers) as prefetch_pool, \
         ThreadPoolExecutor(max_workers=n_workers) as post_pool:

        # Prime the prefetch pipeline with the first batch
        prefetch_queue: list[Future] = []
        for img_path in images[:n_workers]:
            prefetch_queue.append(
                prefetch_pool.submit(prefetch_image, img_path, args.tile_size)
            )
        next_idx = n_workers

        post_futures: list[Future] = []

        for img_path in images:
            # --- Get prefetched image (blocks until ready) ---
            prefetched = prefetch_queue.pop(0).result()

            # Submit next prefetch to keep the pipeline full
            if next_idx < len(images):
                prefetch_queue.append(
                    prefetch_pool.submit(prefetch_image, images[next_idx], args.tile_size)
                )
                next_idx += 1

            if prefetched is None:
                print(f"  [WARN] could not read: {img_path.name}")
                continue

            # --- GPU inference (main thread — single-threaded for safety) ---
            inf_result = run_gpu_inference(
                model, prefetched, args.conf, args.tile_size, device,
            )

            # --- Submit postprocessing to background threads ---
            sn, img_dst, lbl_dst = out_paths[img_path]
            post_futures.append(
                post_pool.submit(
                    postprocess_inference,
                    inf_result, args.iou_merge, sn, img_dst, lbl_dst,
                    img_path, args.save_visualized, vis_dir, names,
                )
            )

            frames += 1
            # Count raw tile detections (pre-NMS) for progress display;
            # actual post-NMS counts are tallied when collecting results below.
            raw_dets = sum(
                len(r.boxes) if r.boxes is not None else 0
                for r in inf_result.results
            )
            fps_inst = 1.0 / (time.perf_counter() - t_start + 1e-6) * frames
            print(
                f"  [{sn:5s}] {img_path.name:40s} "
                f"raw={raw_dets:3d}  tiles={len(prefetched.tile_imgs)}  "
                f"({prefetched.w}x{prefetched.h})"
            )

        # --- Collect postprocess results in order, write CSV ---
        for fut in post_futures:
            pp_result: PostprocessResult = fut.result()
            total_dets += len(pp_result.dets)
            if pp_result.dets:
                images_with_dets += 1
            for row in pp_result.csv_rows:
                csv_writer.writerow(row)

    csv_file.close()

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Step 4: data.yaml
    # ------------------------------------------------------------------
    write_data_yaml(out_root, model)

    print()
    print("=" * 80)
    print("[OK] Tiled inference -> YOLOv8 dataset export complete!")
    print("=" * 80)
    print(f"Output dir:       {out_root}")
    print(f"Images processed: {frames}")
    print(f"Images w/ dets:   {images_with_dets}")
    print(f"Total detections: {total_dets}")
    if elapsed > 0:
        print(f"Inference time:   {elapsed:.1f}s ({frames / elapsed:.2f} img/s)")
    print()
    print("Directory layout:")
    print(f"  train/images: {layout['train']['images']}")
    print(f"  train/labels: {layout['train']['labels']}")
    print(f"  valid/images: {layout['valid']['images']}")
    print(f"  valid/labels: {layout['valid']['labels']}")
    print(f"  test/images:  {layout['test']['images']}")
    print(f"  test/labels:  {layout['test']['labels']}")
    print(f"  data.yaml:    {out_root / 'data.yaml'}")
    print(f"  detections:   {csv_path}")
    if args.save_visualized:
        print(f"  visualized:   {vis_dir}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
