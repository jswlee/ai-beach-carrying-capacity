"""
Semantic-seg version of test_viz.py — same panels, masks, and mask_stats.csv,
but dense output (argmax): no tiling, no holes, no gap-fill needed.

Auto-detects backend from --model dir: HF SegFormer dir (config.json) or
DinoSeg dir (dinoseg_meta.json).

Behaviors specific to this script:
  - Night frames are skipped via mean-saturation (IR night frames are pure
    grayscale: sat==0).  --allow-night disables.  Confirmed night frames are
    logged in <output>/night.txt so later runs skip them without re-reading.
  - The ROI polygon is applied only to filenames sorting after MASK_AFTER
    (camera moved; earlier frames are analysed unmasked).
  - Post-inference, EXCLUDE_POLY is forced to background (recurrent bad region).
  - --workers N runs N model processes in parallel (VRAM is plentiful).

Usage:
    python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model
    python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
        --dataset-dir "D:/Kaanapali" --save-masks --workers 4
"""
import argparse
import csv
import os
import re
import shutil
import multiprocessing as mp
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dinoseg import is_dinoseg_dir, load_dinoseg
from preprocess_masks import apply_roi_mask
from test_viz import (
    CLASS_COLORS_BGR,
    _alpha_overlay,
    _colorize_mask,
    _local_hour,
    _scan_existing_outputs,
)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- this-run constants ---
TRI_AFTER = "youtube_snapshot_Kaanapali_20260127_221431"   # blackout triangle only after this
EXCLUDE_POLY = np.array([[1919, 543], [1800, 393], [1917, 396]], dtype=np.int32)
NIGHT_SAT_MAX = 5.0    # IR night frames have mean saturation == 0; storm days >= ~26

_STATS_FIELDS = [
    "image", "utc_ts", "local_dt", "local_hour", "mean_sat", "median_gray",
    "is_night", "roi_masked", "roi_px",
    "px_background", "px_basebg", "px_beach", "px_water",
    "pct_background", "pct_basebg", "pct_beach", "pct_water",
    "pct_lowconf_fill", "pct_voronoi_fill",
]

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def _utc_dt(fname: str):
    m = _TS_RE.search(os.path.basename(fname))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


class _OnnxWrap:
    """onnxruntime session exposing the model(pixel_values=).logits API."""
    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, pixel_values: torch.Tensor):
        from types import SimpleNamespace
        x = pixel_values.cpu().numpy()
        logits = self.sess.run(["logits"], {"pixel_values": x})[0]
        return SimpleNamespace(logits=torch.from_numpy(logits))


def _load_any_model(model_dir, device):
    onnx_path = os.path.join(model_dir, "model.onnx")
    if device.type == "cpu" and os.path.isfile(onnx_path):
        wrap = _OnnxWrap(onnx_path)
        imgsz = int(wrap.sess.get_inputs()[0].shape[-1])
        return wrap, imgsz, f"ONNX ({imgsz})"
    if is_dinoseg_dir(model_dir):
        model, train_imgsz = load_dinoseg(model_dir, device)
        return model, train_imgsz, f"DinoSeg (train {train_imgsz})"
    from transformers import AutoModelForSemanticSegmentation
    model = AutoModelForSemanticSegmentation.from_pretrained(model_dir).to(device).eval()
    return model, 512, "SegFormer"


def semantic_predict(model, image_bgr: np.ndarray, device, imgsz: int,
                     fast: bool = False) -> np.ndarray:
    """Dense class-index mask (H, W) — every pixel classified, no holes.

    fast=True argmaxes at model resolution then nearest-upsamples the label
    map — much cheaper on CPU; edges are blockier (~imgsz grid)."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
    t = (t - _MEAN) / _STD
    with torch.no_grad():
        logits = model(pixel_values=t.unsqueeze(0).to(device)).logits
    if fast:
        small = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# worker (module-level so spawn can pickle it; model lives in globals)
# ---------------------------------------------------------------------------
_G = {}


def _init_worker(model_dir, imgsz_arg, device_str, fast):
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        torch.set_num_threads(max(1, os.cpu_count()))
    device = torch.device(device_str)
    model, train_imgsz, tag = _load_any_model(model_dir, device)
    _G["model"], _G["dev"], _G["tag"] = model, device, tag
    _G["imgsz"], _G["fast"] = imgsz_arg or train_imgsz, fast


def _proc_one(task):
    """Process one image. Returns {'log','row','night'}."""
    (idx, fname, split_dir, output_dir, masks_dir, save_masks,
     allow_night, mask_first, night_dir) = task
    base = os.path.splitext(fname)[0]
    src_path = os.path.join(split_dir, fname)
    image_bgr = cv2.imread(src_path)
    h, w = image_bgr.shape[:2]
    dt = _utc_dt(fname)
    ldt = dt + timedelta(hours=config.LOCAL_UTC_OFFSET_H) if dt else None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sat = float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
    med = float(np.median(gray))
    row = {"image": fname,
           "utc_ts": dt.isoformat() if dt else "",
           "local_dt": ldt.isoformat() if ldt else "",
           "local_hour": round(ldt.hour + ldt.minute / 60, 3) if ldt else "",
           "mean_sat": round(sat, 2), "median_gray": round(med, 1)}

    if not allow_night and sat < NIGHT_SAT_MAX:
        row.update({"is_night": 1, "roi_masked": 0, "roi_px": ""})
        if night_dir:
            shutil.copy2(src_path, os.path.join(night_dir, fname))
        return {"night": True, "row": row,
                "log": f"  [{idx:3d}] {base}  NIGHT (sat {sat:.1f}) - skipped"}

    # ROI polygon on all frames (auto-picks pre/post-move polygon), plus the
    # exclusion triangle as EXTRA blackout for post-move frames
    masked = mask_first
    infer_bgr = apply_roi_mask(image_bgr.copy(), fname) if masked else image_bgr
    apply_tri = fname > TRI_AFTER
    tri = np.zeros((h, w), dtype=np.uint8)
    if apply_tri:
        cv2.fillPoly(tri, [EXCLUDE_POLY], 1)
        infer_bgr[tri.astype(bool)] = 0

    pred_mask = semantic_predict(_G["model"], infer_bgr, _G["dev"], _G["imgsz"],
                                 fast=_G["fast"])
    if apply_tri:
        pred_mask[tri.astype(bool)] = 0   # also force post-inference

    interior = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2GRAY) > 10
    interior &= ~tri.astype(bool)
    row["is_night"], row["roi_masked"] = 0, int(masked)

    mask_path = os.path.join(split_dir, base + "_mask.png")
    has_gt = os.path.exists(mask_path)
    columns = [image_bgr]
    labels = ["Original"]
    if has_gt:
        columns.append(_alpha_overlay(image_bgr,
                                      _colorize_mask(np.array(Image.open(mask_path)))))
        labels.append("Ground Truth")
    columns.append(_alpha_overlay(image_bgr, _colorize_mask(pred_mask)))
    labels.append("Prediction")

    bar_h, bar_w = 28, w * len(columns)
    bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    seg_w = bar_w // len(config.CLASS_NAMES)
    for cls_id, (name, color) in enumerate(zip(config.CLASS_NAMES, CLASS_COLORS_BGR)):
        x0 = cls_id * seg_w
        bar[:, x0:x0 + seg_w] = color
        cv2.putText(bar, f"{cls_id}:{name}", (x0 + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    header = np.zeros((22, bar_w, 3), dtype=np.uint8)
    for col_idx, label in enumerate(labels):
        cv2.putText(header, label, (col_idx * w + 4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    out_path = os.path.join(output_dir, f"{idx:03d}_{base}.jpg")
    cv2.imwrite(out_path, np.vstack([header, np.hstack(columns), bar]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    stat_str = ""
    if save_masks:
        Image.fromarray(pred_mask).save(os.path.join(masks_dir, base + "_mask.png"))
        roi_n = max(1, int(interior.sum()))
        counts = np.bincount(pred_mask[interior].ravel(), minlength=len(config.CLASS_NAMES))
        row.update({"roi_px": roi_n,
                    "px_background": int(counts[0]), "px_basebg": int(counts[1]),
                    "px_beach": int(counts[2]), "px_water": int(counts[3]),
                    "pct_background": round(100 * counts[0] / roi_n, 2),
                    "pct_basebg": round(100 * counts[1] / roi_n, 2),
                    "pct_beach": round(100 * counts[2] / roi_n, 2),
                    "pct_water": round(100 * counts[3] / roi_n, 2),
                    "pct_lowconf_fill": 0.0, "pct_voronoi_fill": 0.0})
        stat_str = (f"  [beach {row['pct_beach']}% | water {row['pct_water']}%"
                    f" | bg {row['pct_background']}%]")
    return {"night": False, "row": row, "log": f"  [{idx:3d}] {out_path}{stat_str}"}


# ---------------------------------------------------------------------------

def run_viz(args) -> str:
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.model)), "test_viz")
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks") if args.save_masks else None
    if masks_dir:
        os.makedirs(masks_dir, exist_ok=True)
    night_dir = os.path.join(output_dir, "night") if args.save_night else None
    if night_dir:
        os.makedirs(night_dir, exist_ok=True)

    split_dir = (os.path.join(args.dataset_dir, args.split)
                 if os.path.isdir(os.path.join(args.dataset_dir, args.split))
                 else args.dataset_dir)
    image_files = sorted(f for f in os.listdir(split_dir)
                         if f.lower().endswith((".jpg", ".png"))
                         and not f.lower().endswith("_mask.png"))

    if args.time == "day":
        stamped = [(f, _local_hour(f)) for f in image_files]
        if any(h is not None for _, h in stamped):
            image_files = [f for f, h in stamped
                           if h is not None and config.DAY_START_H <= h < config.DAY_END_H]
            print(f"--time day: kept {len(image_files)} images")
        else:
            print("--time day ignored: no filenames carry a UTC timestamp")

    if args.max_images > 0:
        image_files = image_files[:args.max_images]

    existing, start_idx = _scan_existing_outputs(output_dir)
    # frames already logged in a previous run's CSV count as done too —
    # night frames produce no panel, so this prevents duplicate rows on resume
    stats_path = os.path.join(output_dir, "mask_stats.csv")
    known_night = set()
    if os.path.exists(stats_path):
        with open(stats_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing.add(os.path.splitext(r["image"])[0])
                if r.get("is_night") == "1":
                    known_night.add(r["image"])
    if args.skip_existing and existing:
        def _done(f):
            stem = os.path.splitext(f)[0]
            if stem not in existing:
                return False
            return masks_dir is None or os.path.exists(
                os.path.join(masks_dir, stem + "_mask.png"))
        n0 = len(image_files)
        image_files = [f for f in image_files if not _done(f)]
        print(f"Skipping {n0 - len(image_files)} already-done; resuming at {start_idx}")

    # confirmed night frames are logged in night.txt (seeded from past CSV
    # rows) so later runs skip them without re-reading each image
    night_log = None
    if not args.allow_night:
        night_log_path = os.path.join(output_dir, "night.txt")
        logged_night = set()
        if os.path.exists(night_log_path):
            with open(night_log_path, encoding="utf-8") as f:
                logged_night = {ln.strip() for ln in f if ln.strip()}
        night_log = open(night_log_path, "a", encoding="utf-8")
        unlogged = sorted(known_night - logged_night)
        if unlogged:
            night_log.write("".join(n + "\n" for n in unlogged))
            night_log.flush()
        known_night |= logged_night
        n0 = len(image_files)
        image_files = [f for f in image_files if f not in known_night]
        if n0 > len(image_files):
            print(f"Skipping {n0 - len(image_files)} known night frames")

    stats_file = open(stats_path, "a", newline="", encoding="utf-8") if args.save_masks else None
    stats_writer = csv.DictWriter(stats_file, fieldnames=_STATS_FIELDS) if stats_file else None
    if stats_file and stats_file.tell() == 0:
        stats_writer.writeheader()

    print(f"Visualizing {len(image_files)} images from '{split_dir}'")
    print(f"Saving panels to '{output_dir}'\n")

    tasks = [(idx, f, split_dir, output_dir, masks_dir, args.save_masks,
              args.allow_night, args.mask_first, night_dir)
             for idx, f in enumerate(image_files, start_idx)]
    n_night = 0

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args.model, args.imgsz, args.device,
                                args.fast)) as pool:
            for res in pool.imap_unordered(_proc_one, tasks, chunksize=4):
                print(res["log"])
                if res["row"] and stats_writer:
                    stats_writer.writerow(res["row"])
                    stats_file.flush()
                if res["night"] and night_log:
                    night_log.write(res["row"]["image"] + "\n")
                    night_log.flush()
                n_night += res["night"]
    else:
        _init_worker(args.model, args.imgsz, args.device, args.fast)
        print(f"backend: {_G['tag']} (infer {_G['imgsz']})")
        for task in tasks:
            res = _proc_one(task)
            print(res["log"])
            if res["row"] and stats_writer:
                stats_writer.writerow(res["row"])
                stats_file.flush()
            if res["night"] and night_log:
                night_log.write(res["row"]["image"] + "\n")
                night_log.flush()
            n_night += res["night"]

    if stats_file:
        stats_file.close()
    if night_log:
        night_log.close()
    print(f"\nDone. {len(image_files) - n_night} panels to '{output_dir}'; "
          f"{n_night} night frames skipped")
    return output_dir


def parse_args():
    p = argparse.ArgumentParser(description="Semantic-seg version of test_viz.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",          required=True,
                   help="HF SegFormer dir or DinoSeg dir (dinoseg_meta.json)")
    p.add_argument("--output-dir",     default=None, dest="output_dir")
    p.add_argument("--dataset-dir",    default=config.OUTPUT_DATASET, dest="dataset_dir")
    p.add_argument("--split",          default="test", choices=["train", "valid", "test"])
    p.add_argument("--imgsz",          type=int, default=None,
                   help="Square inference size (default: train size for DinoSeg, 512 for SegFormer)")
    p.add_argument("--max-images",     type=int, default=0, dest="max_images")
    p.add_argument("--save-masks",     action="store_true", dest="save_masks")
    p.add_argument("--time",           default=None, choices=["day"])
    p.add_argument("--allow-night",    action="store_true", dest="allow_night",
                   help="Disable the saturation-based night filter")
    p.add_argument("--no-save-night",  action="store_false", dest="save_night",
                   help="Don't copy skipped night frames to output_dir/night/")
    p.add_argument("--workers",        type=int, default=1,
                   help="Parallel worker processes (each loads its own model)")
    p.add_argument("--device",         default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--fast",           action="store_true",
                   help="Argmax at model res + nearest upsample (cheaper, blockier edges)")
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--no-mask-first",  action="store_false", dest="mask_first")
    return p.parse_args()


if __name__ == "__main__":
    run_viz(parse_args())
