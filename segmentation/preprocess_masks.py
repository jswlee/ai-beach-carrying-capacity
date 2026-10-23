"""
Preprocessing script for waterline-segmentation.v9i.png-mask-semantic

Applies the region-of-interest polygon (from mask.py) to all semantic mask PNGs:
  - Pixels OUTSIDE the polygon are forced to 0 (background)
  - Pixels INSIDE the polygon keep their original class labels (1=base-bg, 2=beach, 3=water)

This corrects manual eye-balled annotations near the polygon boundary.

Output dataset is written to: waterline-segmentation.v9i.png-mask-semantic-preprocessed/
"""
import os
import shutil
import cv2
import numpy as np
from PIL import Image

from config import (
    POLYGON_PTS_LIST,
    POLYGON_PTS_AFTER_LIST,
    CAMERA_MOVE_CUTOFF,
    SOURCE_DATASET,
    OUTPUT_DATASET,
    SPLITS,
    NUM_CLASSES,
    CLASS_NAMES,
    TILE_SIZE,
    TILE_HEIGHT,
    TILE_MODE,
    TILE_COLS_FULL,
    TILE_ROWS_FULL,
    DEFAULT_PREVIEW_DIR,
    DEFAULT_PREVIEW_SPLIT,
    DEFAULT_PREVIEW_N,
)

POLYGON_PTS       = np.array(POLYGON_PTS_LIST, dtype=np.int32)
POLYGON_PTS_AFTER = np.array(POLYGON_PTS_AFTER_LIST, dtype=np.int32)


def build_roi_mask(height: int, width: int,
                   polygon: np.ndarray | None = None) -> np.ndarray:
    """Returns a binary uint8 mask (255 inside polygon, 0 outside)."""
    pts = POLYGON_PTS if polygon is None else polygon
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [pts], 255)
    return roi


def build_roi_mask_for_filename(height: int, width: int,
                                filename: str) -> np.ndarray:
    """ROI mask using the pre-/post-camera-move polygon for this filename."""
    pts = POLYGON_PTS_AFTER if filename >= CAMERA_MOVE_CUTOFF else POLYGON_PTS
    return build_roi_mask(height, width, polygon=pts)


def apply_roi_mask(image_bgr: np.ndarray, filename: str) -> np.ndarray:
    """
    Zero out pixels outside the ROI polygon, choosing the pre- or post-move
    polygon by filename.  If the image already looks polygon-masked
    (substantial pure-black region), it is returned unchanged.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if (gray <= 10).mean() > 0.3:
        return image_bgr
    roi = build_roi_mask_for_filename(image_bgr.shape[0], image_bgr.shape[1],
                                      filename)
    return cv2.bitwise_and(image_bgr, image_bgr, mask=roi)


def preprocess_mask(mask_path: str, out_path: str, roi_mask: np.ndarray | None = None) -> None:
    """
    Load a semantic mask PNG, zero-out pixels outside the ROI polygon, save result.
    If roi_mask is None it is computed from the mask's own dimensions.
    """
    mask = np.array(Image.open(mask_path))

    h, w = mask.shape[:2]
    if roi_mask is None or roi_mask.shape != (h, w):
        roi_mask = build_roi_mask(h, w)

    # Everything outside the polygon → class 0 (background)
    outside = roi_mask == 0
    mask[outside] = 0

    Image.fromarray(mask.astype(np.uint8)).save(out_path)


def run_preprocessing(
    source_dataset: str = SOURCE_DATASET,
    output_dataset: str = OUTPUT_DATASET,
    verbose: bool = True,
) -> str:
    """
    Process all splits of the source dataset and write to output_dataset.
    Returns the output_dataset path.
    """
    if verbose:
        print(f"Source : {source_dataset}")
        print(f"Output : {output_dataset}")
        print(f"Polygon: {len(POLYGON_PTS)} vertices\n")

    # Pre-build ROI mask at native resolution (1920×1080) if images are that size;
    # we'll rebuild per-image if dimensions differ.
    cached_roi: dict[tuple, np.ndarray] = {}

    for split in SPLITS:
        src_split = os.path.join(source_dataset, split)
        dst_split = os.path.join(output_dataset, split)

        if not os.path.isdir(src_split):
            if verbose:
                print(f"  Skipping missing split: {src_split}")
            continue

        os.makedirs(dst_split, exist_ok=True)

        files = os.listdir(src_split)
        mask_files = [f for f in files if f.lower().endswith("_mask.png")]
        image_files = [f for f in files if f.lower().endswith(".jpg")]
        other_files = [f for f in files if f not in mask_files and f not in image_files]

        # Copy non-mask files (images, CSVs, etc.) unchanged
        for f in image_files + other_files:
            src = os.path.join(src_split, f)
            dst = os.path.join(dst_split, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        if verbose:
            print(f"[{split}] Processing {len(mask_files)} masks...")

        for fname in mask_files:
            src_mask = os.path.join(src_split, fname)
            dst_mask = os.path.join(dst_split, fname)

            # Get mask dimensions to look up / build ROI
            tmp = np.array(Image.open(src_mask))
            h, w = tmp.shape[:2]
            key = (h, w)
            if key not in cached_roi:
                cached_roi[key] = build_roi_mask(h, w)
            roi = cached_roi[key]

            preprocess_mask(src_mask, dst_mask, roi)

        if verbose:
            print(f"  Done. Output: {dst_split}\n")

    # Copy top-level files (README, data.yaml if present)
    for fname in os.listdir(source_dataset):
        src = os.path.join(source_dataset, fname)
        dst = os.path.join(output_dataset, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Write data.yaml for the preprocessed dataset (3 active classes + background)
    yaml_path = os.path.join(output_dataset, "data.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write(f"names: {CLASS_NAMES}\n")
            f.write(f"nc: {NUM_CLASSES}\n")
        if verbose:
            print(f"Wrote {yaml_path}")

    if verbose:
        print("Preprocessing complete.")

    return output_dataset


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def get_tile_offsets(
    img_w: int,
    img_h: int,
    cols: int = TILE_COLS_FULL,
    rows: int = TILE_ROWS_FULL,
    overlap: float = 0.0,
    mode: str = TILE_MODE,
    tile_height: int = TILE_HEIGHT,
) -> list[tuple[int, int, int, int]]:
    """
    Divide the image into tiles, returning (x0, y0, x1, y1) rectangles.

    Args:
        img_w, img_h: Image dimensions
        cols, rows:   Grid columns/rows (used when mode="all")
        overlap:     Fractional overlap per tile (0.0 = none, 0.15 = 15%)
        mode:        "all" for full grid, "bottom" for 3 bottom tiles only
        tile_height: Native tile height for "bottom" mode (before resize)

    Returns:
        List of tile rectangles.
    """
    if mode == "bottom":
        # 3 tiles across the bottom, each tile_height tall
        col_edges = [round(img_w * c / 3) for c in range(4)]
        y0 = max(0, img_h - tile_height)
        y1 = img_h

        pad_x = int(round(overlap * img_w / 3))
        pad_y = int(round(overlap * tile_height))

        tiles = []
        for c in range(3):
            x0_raw = col_edges[c]
            x1_raw = col_edges[c + 1]
            x0 = max(0, x0_raw - pad_x)
            x1 = min(img_w, x1_raw + pad_x)
            tiles.append((x0, y0 - pad_y, x1, y1 + pad_y))
        return tiles

    else:  # mode == "all"
        col_edges = [round(img_w * c / cols) for c in range(cols + 1)]
        row_edges = [round(img_h * r / rows) for r in range(rows + 1)]

        pad_x = int(round(overlap * img_w / cols))
        pad_y = int(round(overlap * img_h / rows))

        tiles = []
        for r in range(rows):
            for c in range(cols):
                x0 = max(0, col_edges[c] - pad_x)
                y0 = max(0, row_edges[r] - pad_y)
                x1 = min(img_w, col_edges[c + 1] + pad_x)
                y1 = min(img_h, row_edges[r + 1] + pad_y)
                tiles.append((x0, y0, x1, y1))
        return tiles


def tile_image_and_mask(
    img_path: str,
    mask_path: str,
    out_dir: str,
    tile_size: int = TILE_SIZE,
    mode: str = TILE_MODE,
    tile_height: int = TILE_HEIGHT,
) -> list[str]:
    """
    Slice one image+mask pair into tiles based on mode, resize each to tile_size×tile_size,
    and write them to out_dir.  Returns list of written mask tile paths.
    """
    img  = cv2.imread(img_path)
    mask = np.array(Image.open(mask_path))
    h, w = img.shape[:2]

    base      = os.path.splitext(os.path.basename(img_path))[0]
    offsets   = get_tile_offsets(w, h, mode=mode, tile_height=tile_height)
    mask_paths = []

    for idx, (x0, y0, x1, y1) in enumerate(offsets):
        tile_img  = img[y0:y1, x0:x1]
        tile_mask = mask[y0:y1, x0:x1]

        tile_img_resized  = cv2.resize(tile_img,  (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        tile_mask_resized = cv2.resize(tile_mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

        tile_base      = f"{base}_tile{idx}"
        img_out_path   = os.path.join(out_dir, tile_base + ".jpg")
        mask_out_path  = os.path.join(out_dir, tile_base + "_mask.png")

        cv2.imwrite(img_out_path, tile_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        Image.fromarray(tile_mask_resized.astype(np.uint8)).save(mask_out_path)
        mask_paths.append(mask_out_path)

    return mask_paths


def tile_dataset(
    preprocessed_dataset: str = OUTPUT_DATASET,
    tile_size: int = TILE_SIZE,
    mode: str = TILE_MODE,
    tile_height: int = TILE_HEIGHT,
    verbose: bool = True,
) -> str:
    """
    Tile all images in the preprocessed dataset based on mode,
    writing to <preprocessed_dataset>-tiled/.
    Returns the tiled dataset path.
    """
    tiled_dataset = preprocessed_dataset.rstrip("/\\") + "-tiled"

    # Determine tile count for reporting
    tiles_per_image = 3 if mode == "bottom" else TILE_COLS_FULL * TILE_ROWS_FULL

    if verbose:
        print(f"Tiling dataset: {preprocessed_dataset} → {tiled_dataset}")
        if mode == "bottom":
            print(f"  Layout : bottom 3 tiles, {tile_height}px tall each ({tiles_per_image} tiles/image)")
        else:
            print(f"  Layout : {TILE_COLS_FULL} cols × {TILE_ROWS_FULL} rows ({tiles_per_image} tiles/image)")
        print(f"  Tile size: {tile_size}×{tile_size} (after resize)\n")

    for split in SPLITS:
        src_split = os.path.join(preprocessed_dataset, split)
        dst_split = os.path.join(tiled_dataset, split)
        if not os.path.isdir(src_split):
            continue
        os.makedirs(dst_split, exist_ok=True)

        image_files = [f for f in os.listdir(src_split) if f.lower().endswith(".jpg")]
        total_tiles = 0

        for fname in image_files:
            base      = os.path.splitext(fname)[0]
            img_path  = os.path.join(src_split, fname)
            mask_path = os.path.join(src_split, base + "_mask.png")
            if not os.path.exists(mask_path):
                continue
            tile_image_and_mask(img_path, mask_path, dst_split, tile_size, mode, tile_height)
            total_tiles += tiles_per_image

        # Copy non-image/mask files (CSVs etc.)
        for f in os.listdir(src_split):
            if not f.lower().endswith(".jpg") and not f.lower().endswith(".png"):
                src = os.path.join(src_split, f)
                dst = os.path.join(dst_split, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        if verbose:
            print(f"  [{split}] {len(image_files)} images → {total_tiles} tiles")

    if verbose:
        print(f"Tiling complete. Output: {tiled_dataset}\n")

    return tiled_dataset


def mask_to_yolo_labels(mask_path: str, label_path: str, class_ids: list[int] | None = None) -> int:
    """
    Convert a single semantic PNG mask to a YOLO segmentation label .txt file.

    Each class region is traced with cv2.findContours; each contour becomes one
    instance line in the YOLO polygon format:
        <class_id> <x1> <y1> <x2> <y2> ...   (all values normalised 0-1)

    Contours with fewer than 3 points, or covering < MIN_AREA pixels, are skipped.

    Returns the number of annotation lines written.
    """
    MIN_AREA = 50  # pixels — ignore tiny speckles

    if class_ids is None:
        class_ids = list(range(1, NUM_CLASSES))  # skip 0 (background)

    mask = np.array(Image.open(mask_path))
    h, w = mask.shape[:2]

    lines = []
    for cls_id in class_ids:
        binary = np.where(mask == cls_id, np.uint8(255), np.uint8(0))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            # Squeeze shape (N,1,2) → (N,2) and normalise
            pts = cnt.squeeze(1).astype(np.float32)
            if pts.ndim == 1:  # single point edge case
                continue
            pts[:, 0] /= w
            pts[:, 1] /= h
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cls_id} {coords}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return len(lines)


def convert_masks_to_yolo_labels(
    preprocessed_dataset: str = OUTPUT_DATASET,
    class_ids: list[int] | None = None,
    verbose: bool = True,
) -> None:
    """
    Walk every split of the preprocessed dataset and write a YOLO .txt label
    file next to each image, derived from the corresponding _mask.png.

    Output structure (Ultralytics expects images/ and labels/ siblings):
        <split>/
            <name>.jpg          ← already there
            <name>_mask.png     ← already there
            <name>.txt          ← written here (YOLO polygon labels)
    """
    if verbose:
        print(f"Converting masks → YOLO labels in '{preprocessed_dataset}'")

    for split in SPLITS:
        split_dir = os.path.join(preprocessed_dataset, split)
        if not os.path.isdir(split_dir):
            continue

        mask_files = [f for f in os.listdir(split_dir) if f.lower().endswith("_mask.png")]
        total_instances = 0

        for fname in mask_files:
            mask_path  = os.path.join(split_dir, fname)
            # Label lives next to the image, named <image_base>.txt
            img_base   = fname[: -len("_mask.png")]
            label_path = os.path.join(split_dir, img_base + ".txt")
            n = mask_to_yolo_labels(mask_path, label_path, class_ids)
            total_instances += n

        if verbose:
            print(f"  [{split}] {len(mask_files)} labels written, {total_instances} total instances")

    if verbose:
        print("Label conversion complete.\n")


def preview_first_n(
    preprocessed_dataset: str = OUTPUT_DATASET,
    split: str = DEFAULT_PREVIEW_SPLIT,
    n: int = DEFAULT_PREVIEW_N,
    save_dir: str = DEFAULT_PREVIEW_DIR,
) -> None:
    """
    Save side-by-side previews (image | colorized mask | overlay) for the first N pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Colormap: 0=black, 1=gray, 2=sandybrown, 3=deepskyblue
    CLASS_COLORS = {
        0: (0,   0,   0),    # background
        1: (128, 128, 128),  # base-background-layer-segmentation
        2: (194, 160, 100),  # beach
        3: (30,  144, 255),  # water
    }

    os.makedirs(save_dir, exist_ok=True)
    split_dir = os.path.join(preprocessed_dataset, split)

    all_files = sorted(os.listdir(split_dir))
    image_files = [f for f in all_files if f.lower().endswith(".jpg")][:n]

    if not image_files:
        print("No images found for preview.")
        return

    for i, fname in enumerate(image_files):
        base = os.path.splitext(fname)[0]
        img_path  = os.path.join(split_dir, fname)
        mask_path = os.path.join(split_dir, base + "_mask.png")

        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Colorize mask
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            color_mask[mask == cls_id] = color

        # Overlay: blend color mask onto image at 50% alpha
        overlay = (img.astype(np.float32) * 0.5 + color_mask.astype(np.float32) * 0.5).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(img);         axes[0].set_title("Image");         axes[0].axis("off")
        axes[1].imshow(color_mask);  axes[1].set_title("Mask (colorized)"); axes[1].axis("off")
        axes[2].imshow(overlay);     axes[2].set_title("Overlay");       axes[2].axis("off")

        patches = [
            mpatches.Patch(color=np.array(c)/255, label=CLASS_NAMES[cls_id])
            for cls_id, c in CLASS_COLORS.items()
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(fname, fontsize=10)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"preview_{i+1:02d}_{fname.replace('.jpg','.png')}")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved preview: {out_path}")

    print(f"\nPreviews saved to '{save_dir}/'")


if __name__ == "__main__":
    run_preprocessing()
    convert_masks_to_yolo_labels()
    preview_first_n()
