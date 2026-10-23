"""
Count people per image by zone (beach / water / other) by fusing:

  - SegFormer semantic masks   (Kaanapali_full_segformer_v2/masks/*_mask.png)
  - YOLO person detections     (run_tiled_inference.py detections.csv)

Zone rule: the seg class at the detection box's *foot point* (bottom-center)
decides where the person is standing:
    mask 2 -> beach,  mask 3 -> water,  mask 0/1 -> other (background)

One output row per image that the seg pipeline processed (i.e. every
non-night frame — these are exactly the frames the detection model saw too,
since it uses the same saturation night gate).

Usage:
    python count_people_zones.py \
        --detections "<cv-training>/.../kaanapali_segmask_inference/detections.csv" \
        --output Kaanapali_full_segformer_v2/people_zones.csv
"""
import argparse
import csv
import os
import re
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

import config

_TS_RE = re.compile(r"(\d{8})_(\d{6})")

ZONE_OF_CLASS = {0: "other", 1: "other", 2: "beach", 3: "water"}

_FIELDS = ["image", "utc_ts", "local_dt", "local_hour", "ts_suspect",
           "people_beach", "people_water", "people_other", "people_total",
           "n_detections"]


def _utc_dt(fname: str):
    m = _TS_RE.search(os.path.basename(fname))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_detections(path: str, person_class: str) -> dict:
    """detections.csv -> {image_name: [(x1, y1, x2, y2), ...]} (persons only)."""
    dets: dict = {}
    n_skip = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cls = (r.get("cls_name") or "").strip()
            if person_class and cls and cls.lower() != person_class.lower():
                n_skip += 1
                continue
            dets.setdefault(r["image"], []).append(
                (float(r["x1"]), float(r["y1"]),
                 float(r["x2"]), float(r["y2"])))
    if n_skip:
        print(f"Skipped {n_skip} non-'{person_class}' detections")
    return dets


def run(args) -> str:
    dets = load_detections(args.detections, args.person_class)
    mask_files = sorted(f for f in os.listdir(args.masks_dir)
                        if f.endswith("_mask.png"))
    print(f"{len(mask_files)} seg masks, "
          f"{sum(len(v) for v in dets.values())} detections on "
          f"{len(dets)} images")

    mask_set = set(mask_files)
    unmatched = [img for img in dets
                 if os.path.splitext(img)[0] + "_mask.png" not in mask_set]
    if unmatched:
        print(f"WARNING: {len(unmatched)} detected images have no seg mask "
              f"(e.g. {unmatched[0]}) — they are excluded from the counts")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows = n_persons = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for mf in mask_files:
            img_name = mf[:-len("_mask.png")] + ".jpg"
            mask = np.array(Image.open(os.path.join(args.masks_dir, mf)))
            h, w_ = mask.shape[:2]
            counts = {"beach": 0, "water": 0, "other": 0}
            img_dets = dets.get(img_name, [])
            for x1, y1, x2, y2 in img_dets:
                fx = int(round((x1 + x2) / 2.0))
                fy = int(round(y2))                      # foot point
                fx = max(0, min(w_ - 1, fx))
                fy = max(0, min(h - 1, fy))
                counts[ZONE_OF_CLASS.get(int(mask[fy, fx]), "other")] += 1
            dt = _utc_dt(img_name)
            ldt = dt + timedelta(hours=config.LOCAL_UTC_OFFSET_H) if dt else None
            # frames whose computed local hour falls outside the day window
            # passed the saturation gate anyway — contiguous runs of these are
            # capture-clock anomalies (day frames stamped ~10 h off).  They are
            # flagged, not dropped: exclude ts_suspect=1 from hourly analyses.
            lhr = (ldt.hour + ldt.minute / 60) if ldt else None
            ts_suspect = int(lhr is not None
                             and not (config.DAY_START_H <= lhr
                                      < config.DAY_END_H))
            w.writerow({
                "image": img_name,
                "utc_ts": dt.isoformat() if dt else "",
                "local_dt": ldt.isoformat() if ldt else "",
                "local_hour": (round(lhr, 3) if lhr is not None else ""),
                "ts_suspect": ts_suspect,
                "people_beach": counts["beach"],
                "people_water": counts["water"],
                "people_other": counts["other"],
                "people_total": sum(counts.values()),
                "n_detections": len(img_dets),
            })
            n_rows += 1
            n_persons += sum(counts.values())

    print(f"Wrote {n_rows} rows ({n_persons} people) to {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-image people counts by zone (beach/water/other).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--masks-dir",
                   default="Kaanapali_full_segformer_v2/masks")
    p.add_argument("--detections", required=True,
                   help="detections.csv written by run_tiled_inference.py")
    p.add_argument("--output",
                   default="Kaanapali_full_segformer_v2/people_zones.csv")
    p.add_argument("--person-class", default="person",
                   help="cls_name to count ('' counts all detections)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
