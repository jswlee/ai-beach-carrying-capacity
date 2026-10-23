# Kaanapali Beach Segmentation — Methodology

End-to-end documentation of the pipeline that produced `Kaanapali_full_segformer_v2/`
(masks, panels, and `mask_stats.csv` over ~20.7k webcam frames). All scripts needed to
reproduce it are snapshotted in `waterline_segformer_scripts/`.

## 1. Scientific goal

Quantify beach usage and water exposure at Kaanapali Beach from a fixed YouTube
webcam. For each timestamped frame we want the fraction of the scene that is
beach vs. water vs. background, a clean waterline for tide correlation, and
(eventually) people counts — to support brown-water exposure and
carrying-capacity analysis.

## 2. Source data

- **Frames**: `D:\Kaanapali\youtube_snapshot_Kaanapali_YYYYMMDD_HHMMSS.jpg`
  — 20,703 JPEGs, 1912×1080, ~15-minute capture cadence (with occasional gaps).
  Filenames carry **UTC** capture timestamps; local time = UTC − 10 h
  (`LOCAL_UTC_OFFSET_H = -10` in `config.py`).
- **Capture window (local/HST)**: **2026-01-05 10:56 → 2026-10-08 10:30** —
  ~9 months spanning winter→summer. After the saturation night gate this yields
  **11,516 daytime frames** for analysis (9,187 IR/night frames excluded).
- **Labeled set**: Roboflow export `train_data/waterline-segmentation.v11i.png-mask-semantic-preprocessed/`
  — 121 train / 35 valid / 19 test, all from Nov 15–23, 2025.
- **External**: hourly NOAA CO-OPS water level, Kahului Harbor station 1615680 —
  `external_data/NOAA_water_level/CO-OPS__1615680__hr.csv` (see §6 for the
  analysis notebook that joins it to the segmentation stats).

### Classes (pixel values in saved masks)

| id | name | notes |
|----|------|-------|
| 0 | background | sky, buildings, rocks, outside-ROI |
| 1 | base-background-layer-segmentation | mostly palm canopy; **ignored in training** (see §4) |
| 2 | beach | sand incl. detached wet-sand/sandbar patches |
| 3 | water | ocean |

### Camera move + ROI polygons

The camera view changed once during the capture window. `config.py` holds two
ROI polygons (`POLYGON_PTS_LIST` / `POLYGON_PTS_AFTER_LIST`) and
`CAMERA_MOVE_CUTOFF = "youtube_snapshot_Kaanapali_20260127_222935.jpg"`.
`apply_roi_mask(image, filename)` picks the correct polygon by filename and
zeroes everything outside it; already-masked images (>30% pure black) are
passed through untouched.

### Night frames

When the camera goes dark it switches to IR and the images become **literal
grayscale** — mean HSV saturation is exactly 0. Daytime frames, even the very
dark March 14–15 storm frames, keep saturation ≥ ~26. This is exploited as the
night filter in inference (§5), because a brightness threshold would wrongly
discard storm frames.

## 3. Data preparation — `preprocess_masks.py`

Converts the Roboflow semantic-mask export into the training layout:

- Applies the ROI polygon to each image (blackout outside polygon).
- Emits class-index `_mask.png` per image and YOLO-seg `.txt` labels.
- `data.yaml` carries the class names.
- Split: `train/` (121), `valid/` (35), `test/` (19).

## 4. Training — `train_fast.py`

**Model**: `nvidia/segformer-b2-finetuned-ade-512-512` (SegFormer-B2, ~24.7M
params), 4-class head initialized from the ADE20k checkpoint.

| hyperparameter | value |
|---|---|
| input size | 512×512 (frames letterbox-resized) |
| batch size | 8 |
| optimizer | AdamW, lr 6e-5, weight_decay 0.01 |
| schedule | cosine annealing, T_max = 50 epochs |
| max epochs | 200, early stop patience 15 on val mIoU |
| loss | cross-entropy with `ignore_index = 1` |
| workers | 4 |

**`ignore_index = 1`** (base-background / palm canopy): those pixels contribute
no gradient — deliberate, because the study only needs beach vs. water and a
clean waterline. Consequences to remember:

- The model emits **zero class-1 pixels**; canopy overhanging water/beach is
  absorbed into a supervised class (usually beach).
- Reported mIoU is over classes {0, 2, 3} only.

**Result**: best checkpoint at `trained_models/segformer-waterline-v11i/best_model`,
val mIoU ≈ **0.992** on the supervised classes.

### Why SegFormer over YOLO (the model-selection story)

The earlier pipeline ran YOLO instance segmentation (tiled, then full-frame).
Instance masks produced edge artifacts — perfect 90° corners at tile/instance
boundaries, coverage holes filled by a Voronoi fallback, water/beach competing
along the waterline. Measured on the 19-image test set (raw full-frame
predictions vs. GT, counting uncovered pixels as wrong):

| model | ROI uncovered | mean pixel IoU |
|---|---|---|
| YOLO26m @960 | 0.6% | 0.908 |
| YOLO26l @960 | 14.3% | 0.820 |
| SegFormer-B2 @512 | 0% (by construction — dense output) | ~0.992 mIoU (val) |

mAP50 (~0.94 for m) is an instance-level metric and does not reflect pixel
coverage — a single missed instance can leave 10% of an image unlabeled while
barely denting recall. Pixel IoU / coverage is the honest metric here, and
SegFormer wins it while eliminating the entire artifact class. A DINOv2-base +
light conv head (`dinoseg.py`, `train_dinov3.py`) was prototyped as an
alternative — works (~1.7× slower, 770² input) but is not the production model.

## 5. Inference — `test_viz_segformer.py`

The command that produced `Kaanapali_full_segformer_v2/`:

```
python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
    --dataset-dir "D:/Kaanapali" \
    --save-masks --workers 4 --output-dir Kaanapali_full_segformer_v2
```

### Per-frame pipeline, in order

1. **Night gate** — mean HSV saturation of the raw frame. If `sat < 5.0`
   (`NIGHT_SAT_MAX`), the frame is IR/night: skipped entirely (no inference),
   logged in the CSV with `is_night=1`, and (unless `--no-save-night`) copied
   to `output_dir/night/` for visual audit. 9,187 unique frames were filtered
   this way.
2. **ROI mask** — `apply_roi_mask` blackouts outside the pre/post-move polygon
   (chosen by filename vs. `CAMERA_MOVE_CUTOFF`).
3. **Exclusion triangle, pre-inference** — for filenames
   `> youtube_snapshot_Kaanapali_20260127_221431` (`TRI_AFTER`), the triangle
   `[[1919,543],[1800,393],[1917,396]]` (a palm-tree patch the model
   persistently mislabels) is blacked out **before** inference.
4. **Inference** — SegFormer at 512², logits bilinearly upsampled to native
   1080×1912, argmax → dense class-index mask (no holes possible).
5. **Exclusion triangle, post-inference** — same triangle forced to class 0 in
   the saved mask (belt-and-suspenders; it's already background because the
   input was blacked out).
6. **Statistics** — counted on `interior` = non-black pixels of the masked
   input **minus** the triangle. So the denominator is
   "inside ROI, outside triangle"; the triangle's ~8.8k px count toward
   `px_background` but are removed from `roi_px`. Percentages are of `roi_px`.
7. **Outputs** — class-index `_mask.png` into `masks/`, a panel
   (original | GT-if-present | prediction | class-color bar) into the output
   dir, and one CSV row.

### Parallelism

`--workers N` spawns N `spawn`-context processes, each loading its own model
copy (~2 GB VRAM for SegFormer). The parent collects results via
`imap_unordered` and is the sole CSV writer — no corruption. Output is
unordered in the log but filenames are deterministic (`{idx:03d}_{name}.jpg`).

### Resume

Default `--skip-existing`: a frame counts as done if its panel exists (and its
mask, when `--save-masks`). Additionally, any `image` already present in
`mask_stats.csv` counts as done — important because night frames produce no
panel and would otherwise be re-evaluated and duplicated in the CSV on every
resume (this happened once; the shipped CSV was deduplicated, see §7).

### Flags

| flag | effect |
|---|---|
| `--save-masks` | write masks + `mask_stats.csv` |
| `--workers N` | parallel processes |
| `--time day` | keep only frames with local hour ∈ [05:00, 20:00) |
| `--allow-night` | disable the saturation night filter |
| `--no-save-night` | don't copy skipped night frames to `night/` |
| `--no-mask-first` | skip ROI masking |
| `--no-skip-existing` | disable resume |
| `--imgsz N` | override inference size (default 512) |
| `--device cpu|cuda|auto` | force device |
| `--fast` | argmax at 512² + nearest upsample (~15% faster, blockier edges, ~0.6% pixels differ) |
| `--max-images N` | cap count |

### CPU / ONNX

`export_onnx.py <model_dir>` writes `model.onnx` (static square input).
When `--device cpu` and `model.onnx` exists in the model dir, it's used
automatically. Measured on this machine: torch CPU 548 ms/img → ONNX 442 ms/img
with `--fast`; bit-identical masks (0.0% diff). The larger CPU lever is
`--workers`; ONNX is ≈ parity single-threaded.

## 6. Outputs — `Kaanapali_full_segformer_v2/`

```
Kaanapali_full_segformer_v2/
  00000_youtube_snapshot_....jpg   # viz panels (11,516)
  masks/<name>_mask.png            # class-index masks (11,516), uint8 {0,2,3}
  night/                           # skipped IR frames (empty: run used --no-save-night)
  mask_stats.csv                   # 20,703 rows — every source frame, one row each
```

### `mask_stats.csv` schema

| column | meaning |
|---|---|
| `image` | source filename |
| `utc_ts`, `local_dt`, `local_hour` | parsed timestamp (UTC → HST −10 h) |
| `mean_sat`, `median_gray` | QC heuristics — the values behind the night decision |
| `is_night` | 1 = skipped (sat < 5); class columns blank |
| `roi_masked` | 1 = ROI polygon applied |
| `roi_px` | stats denominator: interior non-black px minus triangle |
| `px_background/basebg/beach/water` | raw pixel counts inside `roi_px` |
| `pct_*` | percent of `roi_px` |
| `pct_lowconf_fill`, `pct_voronoi_fill` | always 0 — kept for CSV compat with the YOLO script |

### Downstream analysis — `kaanapali_analysis.ipynb`

`kaanapali_analysis.ipynb` explores the stats: diurnal cycle of beach/water
area, long-term trends over the 9-month window, and the waterline↔tide
relationship against NOAA Kahului Harbor hourly water level
(`external_data/NOAA_water_level/CO-OPS__1615680__hr.csv`, pulled via the
CO-OPS API — URL documented in the notebook). Watch for the camera-move epoch
when interpreting trends — ROI geometry changes at `CAMERA_MOVE_CUTOFF`.

### Observed stats (day frames, n = 11,516)

- pct_beach: mean 30.2, range 6.4–51.1 — low end is high tide / storm surge
  (Mar 14–15 and Oct 4–5 show beach ~4–12%).
- pct_water: mean 69.5, range 48.5–93.2.
- `px_basebg` is 0 in every row — expected consequence of `ignore_index=1`.

## 7. Known caveats
- **Triangle is post-move geometry** — it only makes sense after the camera
  move; gating on `TRI_AFTER` is correct even though it differs from
  `CAMERA_MOVE_CUTOFF` by ~75 min (frames in between keep the triangle off).
- **`ignore_index=1`** — canopy is folded into neighboring classes; `basebg`
  columns exist but are always 0.
- **Single-week labels** — training labels cover Nov 15–23 2025 only; frames
  later in the season may drift (e.g., different sandbar states). The
  `export_hard_for_roboflow.py` script exports a diverse ~1/day hard-frame set
  as editable pre-annotations for the next labeling round — if you retrain,
  evaluate on held-out **days**, not random frames (adjacent frames leak).
- **Percentages share one denominator** (`roi_px`) and sum to ~100 within it;
  outside-ROI and triangle pixels are excluded by design.

## 8. Reproduction

```
# 1. prep labels (once, from a Roboflow export)
python preprocess_masks.py

# 2. train
python train_fast.py                     # -> trained_models/segformer-waterline-v11i/best_model

# 3. inference + stats (GPU, parallel)
python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
    --dataset-dir "D:/Kaanapali" --save-masks \
    --workers 4 --output-dir Kaanapali_full_segformer_v2

# optional: CPU run (auto-uses model.onnx if present)
python export_onnx.py trained_models/segformer-waterline-v11i/best_model
python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
    --dataset-dir "D:/Kaanapali" --save-masks \
    --device cpu --workers 6 --fast --output-dir Kaanapali_cpu
```

## 9. `waterline_segformer_scripts/` inventory

| file | role |
|---|---|
| `config.py` | class names, ROI polygons, cutoffs, timezone/day constants |
| `preprocess_masks.py` | Roboflow → masked images + `_mask.png` + YOLO labels |
| `train_fast.py` | SegFormer-B2 training |
| `test_viz_segformer.py` | dense inference, night gate, triangle, stats, panels (production) |
| `test_viz.py` | legacy YOLO viz + shared panel/stat helpers (imported by above) |
| `dinoseg.py` | DINO backbone + seg head (auto-detected model dir format) |
| `train_dinov3.py` | DinoSeg training (alternative branch, not used for v2) |
| `export_onnx.py` | optional ONNX export for CPU inference |
| `export_hard_for_roboflow.py` | hard-frame export for the next annotation round |

Scripts are snapshots of the working tree at the time `Kaanapali_full_segformer_v2`
was produced; they assume the dataset folders sit alongside them.
