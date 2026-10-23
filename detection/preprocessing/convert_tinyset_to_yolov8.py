#!/usr/bin/env python3
"""
Convert TinyPerson COCO format dataset to YOLO v8 format.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [cx, cy, w, h] normalized.
    
    Args:
        coco_bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [cx, cy, w, h] normalized to [0, 1]
    """
    x, y, w, h = coco_bbox
    
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    cx_norm = cx / img_width
    cy_norm = cy / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return cx_norm, cy_norm, w_norm, h_norm


def convert_coco_to_yolo(
    coco_json_path,
    images_base_dir,
    output_dir,
    split_name="train",
    filter_ignore=True,
    merge_classes=False
):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_base_dir: Base directory containing images
        output_dir: Output directory for YOLO format dataset
        split_name: Name of the split (train/valid/test)
        filter_ignore: If True, skip annotations with ignore=True
        merge_classes: If True, merge all person classes into single class 0
    """
    print(f"\nConverting {split_name} split...")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {categories}")
    
    output_path = Path(output_dir)
    img_out_dir = output_path / split_name / "images"
    lbl_out_dir = output_path / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    images_base = Path(images_base_dir)
    processed_count = 0
    skipped_count = 0
    
    for img_id, img_info in tqdm(images.items(), desc=f"Processing {split_name}"):
        img_filename = img_info['file_name']
        img_src = images_base / img_filename
        if not img_src.exists():
            img_src = images_base / Path(img_filename).name
        if not img_src.exists():
            print(f"  Warning: Image not found: {img_filename}")
            skipped_count += 1
            continue
        
        img_width = img_info['width']
        img_height = img_info['height']
        
        img_dst = img_out_dir / img_src.name
        shutil.copy2(img_src, img_dst)
        
        yolo_labels = []
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                if filter_ignore and ann.get('ignore', False):
                    continue
                
                coco_class_id = ann['category_id']
                if merge_classes:
                    yolo_class_id = 0
                else:
                    yolo_class_id = coco_class_id - 1
                
                coco_bbox = ann['bbox']
                cx, cy, w, h = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print(f"  Warning: Invalid bbox for image {img_filename}: {cx}, {cy}, {w}, {h}")
                    continue
                
                yolo_labels.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        lbl_dst = lbl_out_dir / (img_src.stem + ".txt")
        if yolo_labels:
            with open(lbl_dst, 'w') as f:
                f.write('\n'.join(yolo_labels))
        else:
            lbl_dst.touch()
        
        processed_count += 1
    
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    
    return processed_count


def main():
    tiny_set_dir = Path("tiny_set")
    output_dir = Path("tiny_set_yolov8")
    
    train_json = tiny_set_dir / "annotations" / "tiny_set_train.json"
    train_images = tiny_set_dir / "train" / "train"
    
    if train_json.exists():
        convert_coco_to_yolo(
            coco_json_path=train_json,
            images_base_dir=train_images,
            output_dir=output_dir,
            split_name="train",
            filter_ignore=True,
            merge_classes=True
        )
    
    test_json = tiny_set_dir / "annotations" / "tiny_set_test.json"
    test_images = tiny_set_dir / "test" / "test"
    
    if test_json.exists():
        convert_coco_to_yolo(
            coco_json_path=test_json,
            images_base_dir=test_images,
            output_dir=output_dir,
            split_name="test",
            filter_ignore=True,
            merge_classes=True
        )
    
    print("\nCreating validation split from test...")
    test_img_dir = output_dir / "test" / "images"
    test_lbl_dir = output_dir / "test" / "labels"
    valid_img_dir = output_dir / "valid" / "images"
    valid_lbl_dir = output_dir / "valid" / "labels"
    
    if test_img_dir.exists():
        shutil.copytree(test_img_dir, valid_img_dir, dirs_exist_ok=True)
        shutil.copytree(test_lbl_dir, valid_lbl_dir, dirs_exist_ok=True)
        print(f"  Copied test set to valid set")
    
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['person']
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Dataset YAML: {yaml_path.absolute()}")
    print(f"\nYou can now train with:")
    print(f"  python training/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
