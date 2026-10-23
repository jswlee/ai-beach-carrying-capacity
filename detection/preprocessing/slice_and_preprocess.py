import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

TILE_SIZE = 640
OVERLAP_YOUTUBE = 0.2
OVERLAP_FULL = 0.1
MIN_OVERLAP_RATIO = 0.5
BOTTOM_BAND_HEIGHT = 640

EXCLUSION_POLYGON = np.array([
    [3, 1077], [1903, 1079], [1911, 1074], [1909, 571], [1852, 585],
    [1851, 610], [1792, 627], [1792, 660], [1759, 677], [1792, 692],
    [1654, 744], [1610, 728], [1446, 777], [1308, 808], [1284, 806],
    [1212, 826], [1156, 816], [906, 866], [806, 909], [806, 924],
    [775, 930], [776, 941], [709, 934], [635, 958], [595, 956],
    [363, 991], [254, 1017], [211, 1023], [188, 1018], [46, 1053]
], dtype=np.int32)

# youtube_snapshot_Kaanapali_* images after this timestamp were captured from a
# slightly different camera angle and need the shifted polygon below.
KAANAPALI_ANGLE_CHANGED_AFTER = "20260127_221431"  # YYYYMMDD_HHMMSS

EXCLUSION_POLYGON_KAANAPALI_AFTER = np.array([
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


def get_exclusion_polygon(img_filename):
    """Return the exclusion polygon for a given image filename.

    youtube_snapshot_Kaanapali_* images timestamped after
    KAANAPALI_ANGLE_CHANGED_AFTER use EXCLUSION_POLYGON_KAANAPALI_AFTER
    (camera angle shifted slightly); everything else uses EXCLUSION_POLYGON.
    """
    name = img_filename.lower()
    prefix = "youtube_snapshot_kaanapali_"
    if name.startswith(prefix):
        ts = name[len(prefix):len(prefix) + 15]  # 'YYYYMMDD_HHMMSS'
        if (
            len(ts) == 15
            and ts[:8].isdigit() and ts[8] == "_" and ts[9:].isdigit()
            and ts > KAANAPALI_ANGLE_CHANGED_AFTER
        ):
            return EXCLUSION_POLYGON_KAANAPALI_AFTER
    return EXCLUSION_POLYGON


def is_youtube_snapshot(img_filename):
    """Return True if filename starts with 'youtube_snapshot'."""
    return img_filename.lower().startswith("youtube_snapshot")


def should_apply_mask(img_filename):
    """Return True only for youtube_snapshot_TIMESTAMP or youtube_snapshot_Kaanapali images.

    Rules:
      - Filename must start with 'youtube_snapshot_'
      - The part after that prefix must either:
          * start with digits (timestamp-style), or
          * start with 'Kaanapali' (case-insensitive)
      - Other LOCATION-style names (letters only) should NOT be masked.
    """
    name = img_filename.lower()
    prefix = "youtube_snapshot_"
    if not name.startswith(prefix):
        return False

    suffix = name[len(prefix):]  # part after 'youtube_snapshot_'
    if not suffix:
        return False

    # Case 1: timestamp-like, starts with a digit
    if suffix[0].isdigit():
        return True

    # Case 2: Kaanapali prefix (case-insensitive)
    if suffix.startswith("kaanapali"):
        return True

    # All other LOCATION-style names -> do NOT apply mask
    return False


def is_box_in_exclusion_zone(x_norm, y_norm, width_px, height_px, mask):
    """Check if box center is in exclusion zone."""
    cx = int(x_norm * width_px)
    cy = int(y_norm * height_px)
    cx = max(0, min(cx, width_px - 1))
    cy = max(0, min(cy, height_px - 1))
    return mask[cy, cx] == 0


def get_tiles(img_w, img_h, tile_size, overlap):
    """Generate tile coordinates with overlap."""
    stride = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_eff = min(x, img_w - tile_size)
            y_eff = min(y, img_h - tile_size)
            tiles.append((x_eff, y_eff))
            if x + tile_size >= img_w:
                break
            x += stride
        if y + tile_size >= img_h:
            break
        y += stride
    return tiles


def get_bottom_band_tiles(img_w, img_h, tile_size):
    """Tile the bottom band with a flush (non-overlapping) grid of tile_size squares.

    The band is BOTTOM_BAND_HEIGHT px tall (or tile_size / img_h if those are
    larger/smaller), so the covered region is the same regardless of tile_size:
    tile_size=640 -> 3 tiles (same as the old get_bottom_three_tiles),
    tile_size=320 -> 12 tiles (2 rows x 6 cols).
    """
    band_h = min(img_h, max(BOTTOM_BAND_HEIGHT, tile_size))
    y_off = img_h - band_h
    return [
        (max(0, x), y + y_off)
        for x, y in get_tiles(img_w, band_h, tile_size, 0.0)
    ]


def bottom_band_top(img_h, tile_size):
    """Top y-coordinate of the tiled bottom band (for GT filtering)."""
    return img_h - min(img_h, max(BOTTOM_BAND_HEIGHT, tile_size))


def get_bottom_three_tiles(img_w, img_h, tile_size):
    """Generate exactly three horizontally spaced tiles aligned to the image bottom.

    Returns a list of tuples: (x0, y0)
    - x0/y0 define the top-left coordinate of a tile in the original image.
    """
    return get_bottom_band_tiles(img_w, img_h, tile_size)


def process_yolo_dataset(
    input_dir,
    output_dir,
    enable_slicing=False,
    person_only=False,
    person_class_id=3,
    img_size=640,
    min_pixel_size=0
):
    """
    Process YOLO dataset with optional slicing, selective masking, and class filtering.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        enable_slicing: If True, tile images into 640x640 patches with overlap
        person_only: If True, keep only person class annotations
        person_class_id: Class ID for person (default 3 for Beach Counter dataset)
        min_pixel_size: Filter out annotations where max(width, height) in pixels
                        is below this value. 0 disables filtering.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f"\nProcessing mode: {'SLICING' if enable_slicing else 'NO-SLICE'}")
    print(f"Person-only filtering: {'ENABLED' if person_only else 'DISABLED'}")
    if min_pixel_size > 0:
        print(f"Min pixel size filter: removing annotations < {min_pixel_size}px")
    print(f"Selective masking: youtube_snapshot_ images only\n")

    tiles_written = 0
    label_files_written = 0
    boxes_kept = 0
    boxes_dropped = 0
    boxes_filtered_small = 0

    for split in ["train", "valid", "test"]:
        img_src = input_path / split / "images"
        lbl_src = input_path / split / "labels"

        img_dst = output_path / split / "images"
        lbl_dst = output_path / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        print(f"Processing {split}...")
        image_files = list(img_src.glob("*.jpg")) + list(img_src.glob("*.png"))

        for img_file in tqdm(image_files):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Selective masking: only apply to youtube_snapshot_ images
            apply_mask = should_apply_mask(img_file.name)
            if apply_mask:
                mask = np.ones((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [get_exclusion_polygon(img_file.name)], 0)
                img_processed = img.copy()
                img_processed[mask == 0] = 0
            else:
                mask = None
                img_processed = img.copy()

            # Load and filter labels
            boxes = []
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                        except (ValueError, IndexError):
                            continue

                        # Person-only filtering
                        if person_only and cls != person_class_id:
                            continue

                        # Check exclusion zone only if mask is applied
                        if apply_mask and mask is not None:
                            if is_box_in_exclusion_zone(cx, cy, w, h, mask):
                                continue

                        # Filter by min pixel size
                        if min_pixel_size > 0:
                            abs_w = bw * w
                            abs_h = bh * h
                            size_pixels = (abs_w * abs_h) ** 0.5  # sqrt(width * height)
                            if size_pixels < min_pixel_size:
                                boxes_filtered_small += 1
                                continue

                        boxes.append((cls, cx, cy, bw, bh))

            # Process based on mode
            if enable_slicing:
                # SLICING MODE: Tile images and map boxes to tiles
                # youtube_snapshot images -> 3 bottom tiles; others -> full-resolution grid
                if is_youtube_snapshot(img_file.name):
                    tiles = get_bottom_band_tiles(w, h, img_size)
                else:
                    tiles = get_tiles(w, h, img_size, OVERLAP_FULL)
                for tile_idx, (tx, tile_y0) in enumerate(tiles):
                    tile_y1 = tile_y0
                    tile_y2 = tile_y0 + img_size
                    tile_img = img_processed[tile_y1:tile_y2, tx:tx + img_size]
                    tile_boxes = []
                    
                    for cls, bx, by, bw, bh in boxes:
                        abs_cx = bx * w
                        abs_cy = by * h
                        abs_w = bw * w
                        abs_h = bh * h

                        box_x1 = abs_cx - abs_w / 2.0
                        box_y1 = abs_cy - abs_h / 2.0
                        box_x2 = abs_cx + abs_w / 2.0
                        box_y2 = abs_cy + abs_h / 2.0

                        # Intersect with tile bounds
                        ix1 = max(box_x1, tx)
                        iy1 = max(box_y1, tile_y1)
                        ix2 = min(box_x2, tx + img_size)
                        iy2 = min(box_y2, tile_y2)

                        if ix2 <= ix1 or iy2 <= iy1:
                            continue

                        # Check overlap ratio
                        intersection_area = (ix2 - ix1) * (iy2 - iy1)
                        original_area = abs_w * abs_h
                        if original_area > 0:
                            overlap_ratio = intersection_area / original_area
                            if overlap_ratio < MIN_OVERLAP_RATIO:
                                continue
                        else:
                            continue

                        # Compute new box in tile coordinates
                        tx1 = ix1 - tx
                        tx2 = ix2 - tx
                        ty1 = (iy1 - tile_y1)
                        ty2 = (iy2 - tile_y1)

                        new_cx = ((tx1 + tx2) / 2.0) / img_size
                        new_cy = ((ty1 + ty2) / 2.0) / img_size
                        new_w = (tx2 - tx1) / img_size
                        new_h = (ty2 - ty1) / img_size

                        # Clamp to valid range
                        new_cx = max(0.0, min(1.0, new_cx))
                        new_cy = max(0.0, min(1.0, new_cy))
                        new_w = max(0.0, min(1.0, new_w))
                        new_h = max(0.0, min(1.0, new_h))

                        if new_w > 0.0 and new_h > 0.0:
                            # Remap class to 0 if person_only mode
                            output_cls = 0 if person_only else cls
                            tile_boxes.append(
                                f"{output_cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            )
                            boxes_kept += 1
                        else:
                            boxes_dropped += 1

                    tile_name = f"{img_file.stem}_tile_{tile_idx}.jpg"
                    cv2.imwrite(str(img_dst / tile_name), tile_img)
                    tiles_written += 1
                    if tile_boxes:
                        with open(lbl_dst / f"{img_file.stem}_tile_{tile_idx}.txt", "w") as f:
                            f.write("\n".join(tile_boxes))
                        label_files_written += 1
            else:
                # NO-SLICE MODE: Copy image and labels directly
                cv2.imwrite(str(img_dst / img_file.name), img_processed)
                
                if boxes:
                    output_boxes = []
                    for cls, cx, cy, bw, bh in boxes:
                        # Remap class to 0 if person_only mode
                        output_cls = 0 if person_only else cls
                        output_boxes.append(f"{output_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    
                    with open(lbl_dst / f"{img_file.stem}.txt", "w") as f:
                        f.write("\n".join(output_boxes))

    # Update data.yaml
    orig_yaml = input_path / "data.yaml"
    if orig_yaml.exists():
        with open(orig_yaml) as f:
            data = yaml.safe_load(f)
        data["path"] = str(output_path.absolute())
        data["train"] = str(output_path / "train" / "images")
        data["val"] = str(output_path / "valid" / "images")
        data["test"] = str(output_path / "test" / "images")
        
        # Update class info if person_only mode
        if person_only:
            data["nc"] = 1
            data["names"] = ["person"]
        
        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(data, f, sort_keys=False)

    mode_str = "sliced" if enable_slicing else "processed"
    print(f"\n✓ Success! {mode_str.capitalize()} dataset created at: {output_path}")
    if min_pixel_size > 0:
        print(f"  Annotations filtered (< {min_pixel_size}px): {boxes_filtered_small}")
    if enable_slicing:
        print("Slicing summary:")
        print(f"  Tiles written: {tiles_written}")
        print(f"  Label files written: {label_files_written}")
        print(f"  Boxes kept (across all tiles): {boxes_kept}")
        print(f"  Boxes dropped (across all tiles): {boxes_dropped}")
    return str(output_path / "data.yaml")
