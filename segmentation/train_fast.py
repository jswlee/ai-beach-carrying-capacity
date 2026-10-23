"""
Fast CUDA-optimized waterline segmentation training script
Simplified for maximum GPU utilization
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation
from sklearn.metrics import jaccard_score
import torchvision.transforms.functional as TF

# Custom Dataset
class WaterlineDataset(Dataset):
    def __init__(self, split_path, image_size=512):
        self.split_path = split_path
        self.image_size = image_size
        
        # Find all image-mask pairs
        all_files = os.listdir(split_path)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg')) 
                      and not f.lower().endswith('_mask.png')]
        mask_files = {f: os.path.join(split_path, f) for f in all_files 
                     if f.lower().endswith('_mask.png')}
        
        self.pairs = []
        for img_file in image_files:
            img_base = os.path.splitext(img_file)[0]
            mask_file = img_base + "_mask.png"
            if mask_file in mask_files:
                self.pairs.append({
                    'image': os.path.join(split_path, img_file),
                    'mask': mask_files[mask_file]
                })
        
        print(f"Loaded {len(self.pairs)} pairs from {split_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load image and mask
        image = Image.open(pair['image']).convert('RGB')
        mask = Image.open(pair['mask']).convert('L')
        
        # Resize
        image = TF.resize(image, [self.image_size, self.image_size], 
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], 
                        interpolation=TF.InterpolationMode.NEAREST)
        
        # Convert to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Normalize image (ImageNet stats)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask

# Metrics function
def compute_iou(preds, labels, num_classes, ignore_idx):
    """Compute mean IoU across classes"""
    ious = []
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Filter out ignore index
    valid_mask = labels != ignore_idx
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    for class_id in range(num_classes):
        if class_id == ignore_idx:
            continue
        
        pred_binary = (preds == class_id)
        label_binary = (labels == class_id)
        
        if label_binary.sum() == 0 and pred_binary.sum() == 0:
            continue
        
        iou = jaccard_score(label_binary, pred_binary, zero_division=0.0)
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

# Validation function
def validate(model, dataloader, device, criterion, num_labels, ignore_index):
    model.eval()
    total_loss = 0
    all_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            5
            # Upsample logits to match mask size
            logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            # Compute IoU
            preds = torch.argmax(logits, dim=1)
            iou = compute_iou(preds, masks, num_labels, ignore_index)
            all_ious.append(iou)
    
    model.train()
    return total_loss / len(dataloader), np.mean(all_ious)

def main():
    # Enforce GPU-only execution (CUDA or MPS). Exit if neither is available.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("\n" + "="*60)
        print("ERROR: No GPU accelerator detected.")
        print("This script only runs with CUDA (NVIDIA) or MPS (Apple Silicon).")
        print("\nFix:")
        print("- NVIDIA: install CUDA-enabled PyTorch matching your CUDA build (e.g., cu130)")
        print("  Example: pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130")
        print("="*60 + "\n")
        raise SystemExit(1)
    
    # Dataset path
    dataset_path = "./train_data/waterline-segmentation.v11i.png-mask-semantic-preprocessed"
    
    # Load class labels
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    num_labels = len(class_names)
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for i, name in enumerate(class_names)}
    ignore_index = label2id.get('base-background-layer-segmentation', -100)
    
    print(f"Classes: {class_names}")
    print(f"Ignore index: {ignore_index}")
    
    # Create datasets
    train_dataset = WaterlineDataset(os.path.join(dataset_path, 'train'))
    valid_dataset = WaterlineDataset(os.path.join(dataset_path, 'valid'))
    test_dataset = WaterlineDataset(os.path.join(dataset_path, 'test'))
    
    # Determine data loader settings based on device
    if device == 'cuda':
        num_workers = 4
        use_pin_memory = True
    else:  # mps
        num_workers = 4
        use_pin_memory = False
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Increase if you have VRAM
        shuffle=True, 
        num_workers=num_workers,  # Parallel data loading
        pin_memory=use_pin_memory,  # Faster GPU transfer
        persistent_workers=num_workers > 0
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=max(2, num_workers//2),
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=max(2, num_workers//2),
        pin_memory=use_pin_memory
    )
    
    # Load model
    print("Loading model...")
    checkpoint = "nvidia/segformer-b2-finetuned-ade-512-512"
    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Move model to accelerator and enable optimizations
    model = model.to(device)
    model.train()
    
    # Enable benchmarking/AMP conditionally
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
    else:  # mps
        scaler = None
        use_amp = False  # AMP on MPS is not broadly recommended yet
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * 50  # 50 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 200
    patience = 15
    patience_counter = 0
    best_iou = 0.0
    save_dir = "./trained_models/segformer-waterline-v11i"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training (CUDA only)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=images)
                    logits = outputs.logits
                    
                    # Upsample logits to match mask size
                    logits = nn.functional.interpolate(
                        logits, 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    loss = criterion(logits, masks)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Non-AMP path (MPS)
                outputs = model(pixel_values=images)
                logits = outputs.logits
                
                logits = nn.functional.interpolate(
                    logits, 
                    size=masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation every epoch
        val_loss, val_iou = validate(model, valid_loader, device, criterion, num_labels, ignore_index)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            print(f"  ✓ New best mIoU! Saving model...")
            model.save_pretrained(os.path.join(save_dir, "best_model"))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(save_dir, "checkpoint_best.pt"))
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint_best.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_iou = validate(model, test_loader, device, criterion, num_labels, ignore_index)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test mIoU: {test_iou:.4f}")
    
    # Save final model
    model.save_pretrained(os.path.join(save_dir, "final_model"))
    print(f"\nTraining complete! Models saved to {save_dir}")
    print(f"Best validation mIoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
