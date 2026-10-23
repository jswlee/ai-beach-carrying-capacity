#!/usr/bin/env python3
"""
Analyze annotation pixel sizes in YOLO dataset format.
Calculates bounding box sizes in pixels for person annotations (class_id=3)
across train/valid/test splits.

Outputs:
  - Per-split and aggregate histograms with bins: 0-4, 4-8, 8-16, 16+ pixels
  - summary_statistics.txt with detailed metrics
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 3
BINS = [0, 4, 8, 16, 1000]          # upper bound kept finite for matplotlib
BIN_LABELS = ['0-4px', '4-8px', '8-16px', '16px+']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_image_dimensions(image_path):
    """Return (width, height) for an image file, or None on error."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def parse_yolo_line(line, img_w, img_h):
    """Parse a single YOLO annotation line.

    Returns (class_id, size_px) where size_px = sqrt(w_px * h_px),
    or (None, None) on parse failure.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None, None
    try:
        class_id = int(parts[0])
        w_px = float(parts[3]) * img_w
        h_px = float(parts[4]) * img_h
        return class_id, np.sqrt(w_px * h_px)
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------
def analyze_split(dataset_path, split_name):
    """Analyze one dataset split and return a list of person annotation sizes."""
    print(f"Analyzing {split_name} split...")

    labels_dir = os.path.join(dataset_path, split_name, 'labels')
    images_dir = os.path.join(dataset_path, split_name, 'images')

    if not os.path.exists(labels_dir):
        print(f"  Labels directory not found: {labels_dir}")
        return []

    person_sizes = []
    total_ann = 0
    person_ann = 0
    processed = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)

        # Find corresponding image (.jpg then .png)
        stem = label_file.replace('.txt', '')
        image_path = None
        for ext in ('.jpg', '.png'):
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"  Image not found for {label_file}")
            continue

        dims = get_image_dimensions(image_path)
        if dims is None:
            continue
        img_w, img_h = dims

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, size = parse_yolo_line(line, img_w, img_h)
                    if class_id is None:
                        continue
                    total_ann += 1
                    if class_id == PERSON_CLASS_ID:
                        person_sizes.append(size)
                        person_ann += 1
        except Exception as e:
            print(f"  Error processing {label_path}: {e}")
            continue

        processed += 1

    print(f"  Processed {processed} files — "
          f"{person_ann} person annotations out of {total_ann} total")
    return person_sizes


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def create_histograms(split_data, output_dir):
    """Save per-split and aggregate histograms to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    def _plot(sizes, title, path):
        plt.figure(figsize=(10, 6))
        if sizes:
            plt.hist(sizes, bins=BINS, alpha=0.7, edgecolor='black',
                     color='steelblue')
            plt.xticks(BINS[:-1], BIN_LABELS)
            mean_s, med_s = np.mean(sizes), np.median(sizes)
            plt.axvline(mean_s, color='red', ls='--',
                        label=f'Mean: {mean_s:.1f}px')
            plt.axvline(med_s, color='green', ls='--',
                        label=f'Median: {med_s:.1f}px')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No person annotations',
                     ha='center', va='center',
                     transform=plt.gca().transAxes)
        plt.xlabel('Person Bounding Box Size (pixels)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # Per-split
    for name, sizes in split_data.items():
        _plot(sizes,
              f'Person Annotation Size Distribution — {name.upper()}',
              os.path.join(output_dir, f'histogram_{name}.png'))

    # Aggregate
    all_sizes = [s for sizes in split_data.values() for s in sizes]
    _plot(all_sizes,
          'Person Annotation Size Distribution — ALL SPLITS',
          os.path.join(output_dir, 'histogram_aggregate.png'))


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _format_summary(split_data):
    """Return the full summary as a string."""
    lines = []
    w = lines.append

    w("=" * 80)
    w("PERSON ANNOTATION SIZE ANALYSIS SUMMARY")
    w("=" * 80)

    all_sizes = [s for sizes in split_data.values() for s in sizes]

    if all_sizes:
        arr = np.array(all_sizes)
        w("")
        w("PERSON METRICS (All Splits Combined):")
        w(f"  Total Annotations : {len(arr):,}")
        w(f"  Mean Size         : {np.mean(arr):.1f} px")
        w(f"  Median Size       : {np.median(arr):.1f} px")
        w(f"  Std Deviation     : {np.std(arr):.1f} px")
        w(f"  Min Size          : {np.min(arr):.1f} px")
        w(f"  Max Size          : {np.max(arr):.1f} px")

        hist, _ = np.histogram(arr, bins=BINS)
        w("")
        w("  Size Distribution:")
        for label, count in zip(BIN_LABELS, hist):
            pct = count / len(arr) * 100
            w(f"    {label}: {count:,} ({pct:.1f}%)")
    else:
        w("")
        w("No person annotations found!")

    w("")
    w("SPLIT BREAKDOWN:")
    for name in ('train', 'valid', 'test'):
        sizes = split_data.get(name)
        if sizes is None:
            continue
        w(f"")
        w(f"  {name.upper()} Split:")
        if sizes:
            a = np.array(sizes)
            w(f"    Count  : {len(a):,}")
            w(f"    Mean   : {np.mean(a):.1f} px")
            w(f"    Median : {np.median(a):.1f} px")
            w(f"    Range  : {np.min(a):.1f} – {np.max(a):.1f} px")

            hist, _ = np.histogram(a, bins=BINS)
            w(f"    Size Distribution:")
            for label, count in zip(BIN_LABELS, hist):
                pct = count / len(a) * 100
                w(f"      {label}: {count:,} ({pct:.1f}%)")
        else:
            w("    No person annotations")

    return "\n".join(lines)


def save_summary(split_data, output_dir):
    """Write summary_statistics.txt and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    text = _format_summary(split_data)
    path = os.path.join(output_dir, 'summary_statistics.txt')
    with open(path, 'w') as f:
        f.write(text + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Analyze person annotation sizes in a YOLO dataset')
    parser.add_argument('--dataset_path', type=str,
                        default='roboflow_data/Beach Counter.v16i.yolov8',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='annotation_analysis',
                        help='Output directory for results')
    args = parser.parse_args()

    # Collect sizes per split
    split_data = {}
    for split in ('train', 'valid', 'test'):
        sizes = analyze_split(args.dataset_path, split)
        if sizes:
            split_data[split] = sizes

    if not split_data:
        print("No data found! Check dataset path.")
        return

    create_histograms(split_data, args.output_dir)
    summary_path = save_summary(split_data, args.output_dir)

    print(f"\nAnalysis complete!")
    print(f"  Summary  : {summary_path}")
    print(f"  Histograms saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
