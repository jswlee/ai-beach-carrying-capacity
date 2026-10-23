#!/usr/bin/env python3
"""
Unified preprocessing and training workflow for YOLO beach detection model.
This script:
1. Preprocesses raw annotated images by applying ROI masks (if not already done)
2. Trains a YOLO model on the processed data
3. Saves all training metrics and parameters to a text file
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add preprocessing directory to path
sys.path.append(str(Path(__file__).parent.parent / "preprocessing"))

from slice_and_preprocess import process_yolo_dataset

# Import training function
from train_yolo import train_yolo_model, run_test_inference, run_merged_test_inference
from utils import save_training_summary


def run_inference_only(args, raw_data_path, dataset_yaml_path):
    """Produce metrics/summary for an existing (possibly interrupted) training run.

    Skips training entirely and runs the same post-training steps against the
    weights already on disk, so a run stopped early still gets its test metrics.
    """
    run_dir = Path(args.run_dir).resolve() if args.run_dir else Path("training_results").resolve() / args.run_name
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        return 1

    weights = Path(args.weights).resolve() if args.weights else run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"Error: weights not found: {weights}")
        return 1

    print("Inference-only mode (skipping training)")
    print(f"Run directory: {run_dir}")
    print(f"Weights:       {weights}")
    print()

    # Timestamps are derived from files on disk since training isn't run here
    last_ckpt = run_dir / "weights" / "last.pt"
    start_time = datetime.fromtimestamp(run_dir.stat().st_ctime)
    end_time = datetime.fromtimestamp((last_ckpt if last_ckpt.exists() else weights).stat().st_mtime)
    save_training_summary(run_dir, weights, start_time, end_time)

    if args.enable_slicing:
        run_merged_test_inference(
            model_path=str(weights),
            raw_data_dir=str(raw_data_path),
            img_size=args.img_size,
            tile_size=args.tile_size,
            run_dir=run_dir,
            conf=args.conf,
            person_only=args.person_only,
            person_class_id=args.person_class_id,
        )
    else:
        run_test_inference(
            model_path=str(weights),
            dataset_yaml=str(dataset_yaml_path),
            img_size=args.img_size,
            run_dir=run_dir,
            conf=args.conf,
        )

    print("=" * 70)
    print("✓ Inference-only run completed!")
    print("=" * 70)
    print(f"Summary:   {run_dir / 'training_summary.txt'}")
    print(f"Test metrics: {run_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and train YOLO model for beach detection"
    )
    
    # Preprocessing arguments
    parser.add_argument("--raw-data-dir", type=str, default="roboflow_data/Beach Counter.v16i.yolov8", help="Directory containing raw annotated data from Roboflow")
    parser.add_argument("--processed-data-dir", type=str, default="processed_data/processed_images_v16_tiled", help="Directory for processed (ROI-masked) images")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing if processed data already exists")
    parser.add_argument("--enable-slicing", action="store_true", help="Enable image slicing into 640x640 tiles with overlap")
    parser.add_argument("--person-only", action="store_true", help="Keep only person class annotations (filter out all other classes)")
    parser.add_argument("--person-class-id", type=int, default=3, help="Class ID for person in the dataset (default: 3)")
    parser.add_argument("--min-pixel-size", type=int, default=0, help="Filter out annotations where max(width,height) < this many pixels. Appends '_sub{N}px_removed' to output dir name.")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=300, help="Total number of training epochs (resuming from a checkpoint will continue training until this total is reached)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training (increased for larger dataset)")
    parser.add_argument("--img-size", type=int, default=640, help="Network input size for training and inference (imgsz)")
    parser.add_argument("--tile-size", type=int, default=None, help="Crop/tile size in pixels for slicing and tiled inference (default: same as --img-size). Decouple to zoom: e.g. --tile-size 640 --img-size 1280 trains the model on 640px crops upscaled to 1280px input")
    # parser.add_argument("--model", type=str, default="training_results/models/beach_detection_20251125_133123.pt", help="Model to use for training")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use for training")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (increased for larger dataset)")
    parser.add_argument("--patience", type=int, default=90, help="Patience for early stopping")
    parser.add_argument("--warmup_epochs", type=int, default=30, help="Number of warmup epochs (reduced for larger dataset)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer to use for training")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for inference")
    parser.add_argument("--p2", action="store_true", help="Use P2 (stride-4) architecture variant for small-object detection")
    # Inference-only arguments (metrics for an existing / interrupted run)
    parser.add_argument("--inference-only", action="store_true", help="Skip training and only run test-set inference + summary for an existing run")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory for --inference-only (default: training_results/{run_name})")
    parser.add_argument("--weights", type=str, default=None, help="Weights to evaluate for --inference-only (default: {run_dir}/weights/best.pt)")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Optional checkpoint (.pt) to resume from, e.g. "
            "training_results/{run}/weights/last.pt. NOTE: Ultralytics restores the "
            "original args on resume, so --epochs is ignored and training continues to "
            "the checkpoint's own epoch total."
        ),
    )
    
    args = parser.parse_args()

    # Tile size defaults to img_size (same behavior as before the decoupling)
    args.tile_size = args.tile_size or args.img_size

    # Convert paths to absolute
    raw_data_path = Path(args.raw_data_dir).resolve()
    processed_data_dir = args.processed_data_dir
    if args.min_pixel_size > 0:
        processed_data_dir = f"{processed_data_dir}_sub{args.min_pixel_size}px_removed"
    processed_data_path = Path(processed_data_dir).resolve()
    
    print("=" * 70)
    print("Beach Detection - Unified Preprocessing and Training Workflow")
    print("=" * 70)
    print()
    
    # Step 1: Preprocessing
    dataset_yaml_path = processed_data_path / "data.yaml"
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"beach_detection_{timestamp}"
    
    # Inference-only short-circuits before preprocessing/training
    if args.inference_only:
        return run_inference_only(args, raw_data_path, dataset_yaml_path)
    
    if args.skip_preprocessing and dataset_yaml_path.exists():
        print(f"Skipping preprocessing - using existing data at: {processed_data_path}")
        print()
    else:
        print("Step 1: Preprocessing raw data with ROI masks")
        print("-" * 70)
        print(f"Input:  {raw_data_path}")
        print(f"Output: {processed_data_path}")
        print()
        
        if not raw_data_path.exists():
            print(f"Error: Raw data directory not found: {raw_data_path}")
            return 1
        
        try:
            # Process the dataset with ROI masks
            dataset_yaml_path = process_yolo_dataset(
                str(raw_data_path),
                str(processed_data_path),
                enable_slicing=args.enable_slicing,
                person_only=args.person_only,
                person_class_id=args.person_class_id,
                img_size=args.tile_size,
                min_pixel_size=args.min_pixel_size,
            )
            print()
            print(f"Preprocessing complete!")
            print(f"  Dataset YAML: {dataset_yaml_path}")
            print()
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return 1
    
    # Step 2: Training
    print("Step 2: Training YOLO model")
    print("-" * 70)
    
    print(f"Run name: {args.run_name}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size (imgsz): {args.img_size}")
    print(f"Tile size (crop): {args.tile_size}")
    print(f"Freeze layers: {args.freeze}")
    print(f"Learning rate: {args.lr0}")
    print(f"Patience: {args.patience}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Optimizer: {args.optimizer}")
    print()
    
    try:
        model_path, run_dir = train_yolo_model(
            dataset_yaml=str(dataset_yaml_path),
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            run_name=args.run_name,
            freeze=args.freeze,
            lr0=args.lr0,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            optimizer=args.optimizer,
            resume_from=args.resume_from,
            p2=args.p2,
        )
        
        # Run test-set inference and save predictions/metrics alongside training
        # outputs (run_dir comes from the Ultralytics trainer)
        if args.enable_slicing:
            # Merged tiled evaluation: run inference on raw images, NMS-merge
            # overlapping tile detections, then compute image-level metrics.
            run_merged_test_inference(
                model_path=model_path,
                raw_data_dir=str(raw_data_path),
                img_size=args.img_size,
                tile_size=args.tile_size,
                run_dir=run_dir,
                conf=args.conf,
                person_only=args.person_only,
                person_class_id=args.person_class_id,
            )
        else:
            run_test_inference(
                model_path=model_path,
                dataset_yaml=str(dataset_yaml_path),
                img_size=args.img_size,
                run_dir=run_dir,
                conf=args.conf,
            )

        print()
        print("=" * 70)
        print("✓ Workflow completed successfully!")
        print("=" * 70)
        print(f"Trained model: {model_path}")
        print(f"Training results: {run_dir}")
        print(f"Test inference results: {run_dir / 'test_inference'}")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Training failed: {e}")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())