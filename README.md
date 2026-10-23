# AI Beach Carrying Capacity

End-to-end pipeline that turns a fixed beach webcam into longitudinal
carrying-capacity analytics: **how many people are on the beach vs. in the
water at any given time, how usage changes over the season, and how occupancy
responds to weather (air temperature, tide).**

The system fuses two models over the same frames:

| Model | Task | Canonical weights |
|---|---|---|
| SegFormer-B2 (semantic segmentation) | per-pixel beach / water / background | `waterline-detection/trained_models/segformer-waterline-v11i/best_model` |
| YOLO26l-P2 (object detection) | person bounding boxes | `beach-obj-detection-training/training_results_old/yolo26l_p2_beach_detection_20260524_162410/weights/best.pt` (identical copy in `cv-training/training_results/…`, md5 `c2246c8e…fc90`) |

**`PAPER.md` is the source of truth** for methodology, data provenance, and
results. This README is the operational map.

## Repository layout

```
ai-beach-carrying-capacity/
├── README.md                    ← you are here (operations map)
├── PAPER.md                     ← methodology + results draft (source of truth)
├── segmentation/                ← SegFormer pipeline (from waterline-detection)
│   ├── config.py                ← ROI polygons, class names, cutoffs, TZ constants
│   ├── preprocess_masks.py      ← Roboflow semantic export → masked imgs + _mask.png + YOLO labels
│   ├── train_fast.py            ← SegFormer-B2 training (→ best_model)
│   ├── test_viz_segformer.py    ← production inference: night gate, ROI, triangle, masks + stats
│   ├── test_viz.py              ← shared viz/stat helpers (imported by the above)
│   ├── extract_waterline.py     ← per-frame waterline y-position from saved masks
│   ├── count_people_zones.py    ← fuses seg masks + detection boxes → per-image zone counts
│   ├── export_onnx.py           ← optional ONNX export for CPU inference
│   ├── dinoseg.py, train_dinov3.py ← DINOv2 prototype (not production)
│   ├── kaanapali_analysis.ipynb ← seg-stats ↔ tide exploratory notebook
│   ├── waterline_methodology.md ← full seg pipeline write-up
│   └── requirements.txt
├── detection/                   ← YOLO person-detection pipeline (from cv-training)
│   ├── training/
│   │   ├── preprocess_and_train.py  ← unified preprocess → train → merged tiled eval
│   │   ├── train_yolo.py     ← Ultralytics training wrapper + summaries
│   │   ├── run_tiled_inference.py   ← tiled inference + NMS merge + --seg-mask mode + detections.csv
│   │   ├── finetune_from_checkpoint.py, export_onnx.py, utils.py
│   ├── preprocessing/
│   │   ├── slice_and_preprocess.py  ← ROI exclusion poly, tiling, person-only filter
│   │   ├── apply_roi_mask.py        ← ROI mask builder/application
│   │   ├── mask_and_tile_for_calib.py, convert_tinyset_to_yolov8.py
│   ├── analyze_annotation_sizes.py, visualize_*.py,
│   │   export_test_predictions_to_csv.py, tiled_inference_to_dataset.py,
│   │   run_inference_to_dataset.py, test_model.py
│   ├── results/                 ← provenance records of the production run
│   │   ├── args.yaml, training_summary.txt, results.csv, merged_metrics.json
│   └── requirements.txt
├── data/                        ← inference results used by the analysis
│   ├── people_zones.csv         ← per-frame beach/water/other counts + ts_suspect flag
│   ├── mask_stats_clean.csv     ← deduped seg stats, all 21,639 frames
│   ├── waterline_stats.csv      ← per-frame waterline y-position
│   ├── detections.csv           ← all 284,325 person boxes
│   ├── frame_weather.csv        ← per-image HCDP weather join
│   ├── annotation_sizes.csv     ← per-box person annotation sizes (tiled train set)
│   ├── night.txt, night_detections.txt  ← night-gated frames (identical both models)
│   └── external/                ← hcdp_5min_lahaina.csv, NOAA tide CSVs
├── figures/                     ← images referenced by PAPER.md
└── analysis/
    ├── beach_capacity_analysis.ipynb  ← metrics notebook (self-contained on data/)
    └── calibrate_homography.py        ← GCP → homography → per-pixel m² area maps
```

## Source data products

All tabular inference results are copied into `data/` (see layout above).
Only the heavy/binary products remain in the source repos:

| Product | Path | Contents |
|---|---|---|
| Webcam corpus | `D:\Kaanapali\` | 21,639 JPEGs 1912×1080, ~15 min cadence, 2026-01-05 → 2026-10-20 (UTC names) |
| Seg outputs | `waterline-detection/Kaanapali_full_segformer_v2/` | panels, `masks/` (12,035 class-index PNGs), `mask_stats.csv` (all frames) + deduped `mask_stats_clean.csv`, `night.txt`, `waterline_stats.csv`, `people_zones.csv` |
| Det outputs | `cv-training/training_results/yolo26l_p2_beach_detection_20260524_162410/kaanapali_segmask_inference/` | `detections.csv` (284,325 boxes), `visualized/` (10 day examples), `night.txt`, `inference_metrics.*` |
| Weather | `waterline-detection/Dashboard/external_data/hcdp_summary.csv` | HCDP Lahaina WTP, 5-min Tair/RH/WS, 2025-12-31 → 2026-08-08 |
| Tide | `waterline-detection/external_data/NOAA_water_level/` | CO-OPS 1615680 (Kahului) hourly; TPT2799 predictions |

## Reproduction commands

```bash
# --- Segmentation (run from waterline-detection/ or segmentation/ alongside data)
python preprocess_masks.py                      # Roboflow export → preprocessed dataset (once)
python train_fast.py                            # → trained_models/segformer-waterline-v11i/best_model
python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
    --dataset-dir "D:/Kaanapali" --save-masks --workers 4 \
    --output-dir Kaanapali_full_segformer_v2
python extract_waterline.py                     # masks → waterline_stats.csv

# --- Detection (run from cv-training/ or detection/; needs ultralytics env)
python training/preprocess_and_train.py --enable-slicing --person-only --p2 \
    --model yolo26l.pt --run-name <run_name>    # preprocess + train + merged test eval
python training/run_tiled_inference.py \
    --model <best.pt> --input-dir "D:/Kaanapali" \
    --output-dir <out_dir> --seg-mask --save-img --max-save-img 10 \
    --save-day-only --device cuda --conf 0.20   # → detections.csv + night.txt + 10 day panels

# --- Fusion
python segmentation/count_people_zones.py \
    --detections <out_dir>/detections.csv \
    --masks-dir Kaanapali_full_segformer_v2/masks \
    --output Kaanapali_full_segformer_v2/people_zones.csv
```

`--seg-mask` makes the detector see **pixel-identical inputs** to the
segmenter (same inclusion ROI, post-move exclusion triangle, and saturation
night gate — verified by a byte-identical `night.txt`).

## Environment

One venv runs everything (scripts + notebook):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`requirements.txt` pulls CUDA 12.8 PyTorch wheels (torch 2.9.0+cu128,
torchvision 0.24.0+cu128) via the pytorch index URL; drop the `+cu128`
suffixes for CPU-only. Jupyter kernels: `python -m ipykernel install --user
--name ai-beach-carrying-capacity` makes the venv selectable in VS Code /
Jupyter.

The original per-repo envs still work (`waterline-detection/venv` with torch
2.9+cu130 & transformers 4.57.1; `cv-training/venv` with torch nightly+cu128 &
ultralytics 8.4.60). Nothing was deleted from the source repos — files here
are copies.

## Downstream

`waterline-detection/Dashboard/` hosts a Streamlit deployment (latest images,
detection analytics, segmentation maps, weather correlations) built on the
same outputs — it is not part of this pipeline's reproduction path.
