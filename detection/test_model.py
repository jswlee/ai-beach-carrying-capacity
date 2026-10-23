#!/usr/bin/env python3
"""
Standalone test script to evaluate a trained YOLO model on the test split
and save metrics for easy model comparison.

Usage example:

  python test_model.py \
    --model training_results/models/beach_detection_20251125_075147.pt \
    --data processed_images_ready_for_training_person/data.yaml \
    --img-size 1920

Outputs:
  test_results/
    <model_name>_<timestamp>/
      test_summary.txt      # human-readable metrics
      metrics.json / csv    # raw metrics from Ultralytics
      plots/ images/        # PR/F1/confusion/confidence plots & annotated images
"""

from pathlib import Path
import argparse
import sys
from datetime import datetime

from ultralytics import YOLO

# Add training directory to path for utils import
sys.path.append(str(Path(__file__).parent / "training"))
from utils import select_device, save_test_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on the test split and save metrics.")
    parser.add_argument(
        "--model",
        type=str,
        default="training_results/models/beach_detection_20260104_101723_best.pt",
        help="Path to trained YOLO .pt model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="processed_data/processed_images_v13/data.yaml",
        help="Path to dataset YAML file (must define test split)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1920,
        help="Image size for test inference",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size for test inference",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional custom name for this test run (used under test_results/)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for inference",
    )

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"❌ Dataset YAML not found: {data_yaml}")
        return 1

    device = select_device()
    print("YOLO Test Evaluation")
    print("=" * 80)
    print(f"Model:   {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {device}")
    print()

    # Test results root
    results_root = Path("test_results")
    results_root.mkdir(parents=True, exist_ok=True)

    # Build run directory name from model name and timestamp if not provided
    if args.name is not None:
        run_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"test_{model_path.stem}_{timestamp}"

    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))
    print()

    # Run validation on test split
    print("Running test-set evaluation (model.val on split='test')...")
    print("-" * 80)
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.img_size,
        device=device,
        batch=args.batch,
        project=str(run_dir),
        name="eval",
        save=True,     # save annotated images
        plots=True,    # save PR/F1/confusion plots
        verbose=True,
        conf=args.conf,
        save_txt=True
    )

    # Save summary
    save_test_summary(run_dir=run_dir, model_path=model_path, data_yaml=str(data_yaml), img_size=args.img_size, device=device, results=results)

    print()
    print("=" * 80)
    print("✓ Test evaluation complete!")
    print("=" * 80)
    print(f"Results directory: {run_dir}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
