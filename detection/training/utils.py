#!/usr/bin/env python3
"""
Utility functions for YOLO training and testing.
Handles file I/O, summary generation, and PyTorch compatibility patches.
"""

from pathlib import Path
import pandas as pd
import torch
import os
import yaml
from datetime import datetime


def patch_torch_load():
    """Fix for PyTorch 2.6+ weights_only=True default."""
    print("Setting torch.load to use weights_only=False for YOLO model loading")
    _original_load = torch.load
    
    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    
    torch.load = patched_load


def select_device() -> str:
    """Return best available device string for YOLO (cuda, mps, cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_training_summary(run_dir, model_path, start_time, end_time):
    """
    Save comprehensive training summary to a text file.
    Reads actual training config from Ultralytics' args.yaml.
    
    Args:
        run_dir: Directory where training results are saved (Path object)
        model_path: Path to the saved model
        start_time: Training start timestamp
        end_time: Training end timestamp
    
    Returns:
        Path to the summary file
    """
    summary_file = run_dir / "training_summary.txt"
    
    # Load actual training config from Ultralytics
    args_file = run_dir / "args.yaml"
    training_config = {}
    if args_file.exists():
        with open(args_file, 'r') as f:
            training_config = yaml.safe_load(f)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Timestamp information
        f.write("TRAINING TIMESTAMPS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start Time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:      {str(end_time - start_time).split('.')[0]}\n\n")
        
        # Training configuration (from actual args.yaml)
        if training_config:
            f.write("TRAINING CONFIGURATION (actual values used)\n")
            f.write("-" * 80 + "\n")
            
            # Group important params for readability
            important_params = [
                'model', 'data', 'epochs', 'batch', 'imgsz', 'device',
                'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
                'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
                'box', 'cls', 'dfl', 'freeze',
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                'shear', 'perspective', 'flipud', 'fliplr',
                'mosaic', 'mixup', 'copy_paste',
                'patience', 'close_mosaic'
            ]
            
            for key in important_params:
                if key in training_config:
                    f.write(f"{key:20s}: {training_config[key]}\n")
            f.write("\n")
        
        # Model information
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Path:          {model_path}\n")
        f.write(f"Model Size:          {os.path.getsize(model_path) / (1024**2):.2f} MB\n\n")
        
        # Results CSV summary
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                
                f.write("TRAINING METRICS SUMMARY\n")
                f.write("-" * 80 + "\n")
                
                # Find best epoch
                if 'metrics/mAP50-95(B)' in df.columns:
                    best_idx = df['metrics/mAP50-95(B)'].idxmax()
                    best_epoch = df.loc[best_idx]
                    
                    f.write(f"Best Epoch:          {int(best_epoch['epoch'])}\n")
                    f.write(f"Best mAP50-95:       {best_epoch['metrics/mAP50-95(B)']:.4f}\n")
                    f.write(f"Best mAP50:          {best_epoch['metrics/mAP50(B)']:.4f}\n")
                    f.write(f"Best Precision:      {best_epoch['metrics/precision(B)']:.4f}\n")
                    f.write(f"Best Recall:         {best_epoch['metrics/recall(B)']:.4f}\n\n")
                
                # Final epoch metrics
                final_epoch = df.iloc[-1]
                f.write(f"Final Epoch:         {int(final_epoch['epoch'])}\n")
                f.write(f"Final mAP50-95:      {final_epoch['metrics/mAP50-95(B)']:.4f}\n")
                f.write(f"Final mAP50:         {final_epoch['metrics/mAP50(B)']:.4f}\n")
                f.write(f"Final Precision:     {final_epoch['metrics/precision(B)']:.4f}\n")
                f.write(f"Final Recall:        {final_epoch['metrics/recall(B)']:.4f}\n\n")
                
                # Loss metrics
                f.write("FINAL LOSS VALUES\n")
                f.write("-" * 80 + "\n")
                if 'train/box_loss' in df.columns:
                    f.write(f"Train Box Loss:      {final_epoch['train/box_loss']:.4f}\n")
                if 'train/cls_loss' in df.columns:
                    f.write(f"Train Class Loss:    {final_epoch['train/cls_loss']:.4f}\n")
                if 'train/dfl_loss' in df.columns:
                    f.write(f"Train DFL Loss:      {final_epoch['train/dfl_loss']:.4f}\n")
                if 'val/box_loss' in df.columns:
                    f.write(f"Val Box Loss:        {final_epoch['val/box_loss']:.4f}\n")
                if 'val/cls_loss' in df.columns:
                    f.write(f"Val Class Loss:      {final_epoch['val/cls_loss']:.4f}\n")
                if 'val/dfl_loss' in df.columns:
                    f.write(f"Val DFL Loss:        {final_epoch['val/dfl_loss']:.4f}\n")
                f.write("\n")
                
                # Full epoch-by-epoch results
                f.write("EPOCH-BY-EPOCH RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                # Write column headers
                cols_to_show = ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                               'metrics/precision(B)', 'metrics/recall(B)', 
                               'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
                cols_available = [c for c in cols_to_show if c in df.columns]
                
                # Write header
                header = "  ".join([f"{col:>18s}" for col in cols_available])
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                
                # Write data rows
                for _, row in df.iterrows():
                    values = "  ".join([f"{row[col]:>18.6f}" if pd.notna(row[col]) else f"{'N/A':>18s}" 
                                       for col in cols_available])
                    f.write(values + "\n")
                
            except Exception as e:
                f.write(f"Error reading results CSV: {e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    # ASCII-only: this can be printed to a cp1252 console (e.g. piped output)
    print(f"Training summary saved to: {summary_file}")
    return summary_file


def save_test_summary(run_dir: Path, model_path: Path, data_yaml: str, img_size: int, device: str, results) -> Path:
    """Save a concise test metrics summary to a text file.

    Args:
        run_dir: Directory where test results are saved
        model_path: Path to the evaluated model
        data_yaml: Dataset YAML path
        img_size: Inference image size
        device: Device used (cuda/mps/cpu)
        results: Ultralytics Results object from model.val
    
    Returns:
        Path to the summary file
    """
    summary_file = run_dir / "test_summary.txt"

    metrics = getattr(results, "metrics", None)
    names = getattr(results, "names", None)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO TEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("TEST CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model:             {model_path}\n")
        f.write(f"Dataset:           {data_yaml}\n")
        f.write(f"Image size:        {img_size}\n")
        f.write(f"Device:            {device}\n")
        f.write(f"Results dir:       {run_dir}\n")
        f.write("\n")

        if metrics is not None:
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            try:
                # Ultralytics metrics typically expose these attributes
                f.write(f"mAP50-95 (all):   {metrics.map:0.4f}\n")
                f.write(f"mAP50 (all):      {metrics.map50:0.4f}\n")
                f.write(f"Precision (all):  {metrics.precision.mean():0.4f}\n")
                f.write(f"Recall (all):     {metrics.recall.mean():0.4f}\n")
            except Exception:
                f.write("Could not read overall metrics from results.metrics.\n")
            f.write("\n")

            # Per-class table if available
            if names is not None:
                try:
                    f.write("PER-CLASS METRICS (similar to console)\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Class':<15s} {'Prec':>8s} {'Rec':>8s} {'mAP50':>10s} {'mAP50-95':>10s}\n")
                    f.write("-" * 80 + "\n")
                    for cls_id, cls_name in names.items():
                        map50_95 = float(metrics.ap[cls_id].mean()) if metrics.ap is not None else float("nan")
                        map50 = float(metrics.ap50[cls_id]) if getattr(metrics, "ap50", None) is not None else float("nan")
                        prec = float(metrics.precision[cls_id]) if metrics.precision is not None else float("nan")
                        rec = float(metrics.recall[cls_id]) if metrics.recall is not None else float("nan")
                        f.write(
                            f"{cls_name:<15s} {prec:8.3f} {rec:8.3f} {map50:10.3f} {map50_95:10.3f}\n"
                        )
                    f.write("\n")
                except Exception:
                    f.write("Could not read per-class metrics from results.metrics.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF TEST SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Test summary saved to: {summary_file}")
    return summary_file
