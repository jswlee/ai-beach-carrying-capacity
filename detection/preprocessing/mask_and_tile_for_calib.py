#!/usr/bin/env python3
"""Mask + bottom-3 tiling for calibration images.

Given a directory of raw images, this script:

  input_dir/*.jpg|*.png|... ->
      1) applies the ROI mask (same as training preprocessing)
      2) extracts the 3 bottom 640x640 tiles
      3) writes them into input_dir/calib_tiled

No labels are involved; this is only for calibration (.hef) images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from apply_roi_mask import create_roi_mask
from slice_and_preprocess import get_bottom_three_tiles, TILE_SIZE


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def process_dir(input_dir: Path, tile_size: int = 640) -> None:
    images = collect_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return

    out_dir = input_dir / "calib_tiled"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Writing masked + tiled images to: {out_dir}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Apply ROI mask (same as training preprocessing)
        mask01 = create_roi_mask(img.shape)
        mask_u8 = (mask01.astype(np.uint8) * 255)
        img_masked = cv2.bitwise_and(img, img, mask=mask_u8)

        # Get 3 bottom tiles
        tiles = get_bottom_three_tiles(w, h, tile_size)
        for idx, (tx, ty) in enumerate(tiles):
            tile = img_masked[ty : ty + tile_size, tx : tx + tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            out_name = f"{img_path.stem}_tile_{idx}.jpg"
            cv2.imwrite(str(out_dir / out_name), tile)

        print(f"Processed: {img_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask + bottom-3 tiling for calibration images.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of raw images for calibration")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="Tile size (default from preprocessing.TILE_SIZE)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found or not a directory: {input_dir}")

    process_dir(input_dir, tile_size=args.tile_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
