"""Extract waterline y-position per frame from saved class-index masks.

For each mask in Kaanapali_full_segformer_v2/masks/, finds the largest
connected water component (class 3), takes its deepest (max-row) pixel per
column = the waterline in that column, then aggregates across columns:
median / mean / q10 / q90 of the per-column waterline y.

Larger y = waterline further down the image = water higher up the beach.

Output: Kaanapali_full_segformer_v2/waterline_stats.csv, keyed by image name
(mask filename minus '_mask.png').
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

MASK_DIR = r'Kaanapali_full_segformer_v2/masks'
OUT_CSV = r'Kaanapali_full_segformer_v2/waterline_stats.csv'
WATER = 3
MIN_COL_PX = 20        # a column needs >=20 water px to vote on the waterline


def waterline_y(mask):
    """Per-column deepest water row, restricted to the largest water blob."""
    water = (mask == WATER).astype(np.uint8)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(water, 8)
    if n <= 1:
        return np.full(mask.shape[1], np.nan)
    big = 1 + np.argmax(stats_[1:, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == big)
    # deepest water pixel per column
    wl = np.full(mask.shape[1], -1)
    np.maximum.at(wl, xs, ys)
    col_n = np.bincount(xs, minlength=mask.shape[1])
    wl = wl.astype(float)
    wl[(wl < 0) | (col_n < MIN_COL_PX)] = np.nan
    return wl


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(MASK_DIR, '*_mask.png')))
    for i, f in enumerate(files):
        m = np.array(cv2.imread(f, cv2.IMREAD_UNCHANGED))
        wl = waterline_y(m)
        ok = np.isfinite(wl)
        rows.append({
            'image': os.path.basename(f).replace('_mask.png', '.jpg'),
            'wl_y_median': float(np.nanmedian(wl)) if ok.any() else np.nan,
            'wl_y_mean': float(np.nanmean(wl)) if ok.any() else np.nan,
            'wl_y_q10': float(np.nanpercentile(wl, 10)) if ok.any() else np.nan,
            'wl_y_q90': float(np.nanpercentile(wl, 90)) if ok.any() else np.nan,
            'wl_n_cols': int(ok.sum()),
        })
        if (i + 1) % 2000 == 0:
            print(f'{i + 1}/{len(files)}')
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df[['wl_y_median', 'wl_y_mean', 'wl_n_cols']].describe().round(1))
    print('wrote', OUT_CSV)


if __name__ == '__main__':
    main()
