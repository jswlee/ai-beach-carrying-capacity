"""Train DINOv3 ViT-B/16 frozen backbone + conv head on the waterline dataset.

Same dataset/loop conventions as train_fast.py (reuses its WaterlineDataset,
compute_iou, and validate). Only the head trains, so bigger resolution is cheap.

    python train_dinov3.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinoseg import DinoSeg
from train_fast import WaterlineDataset, compute_iou, validate

DATASET = "./train_data/waterline-segmentation.v11i.png-mask-semantic-preprocessed"
SAVE_DIR = "./trained_models/dinov3-waterline-v11i"
IMG_SIZE = 768          # backbone sees square 768; frames are 1912x1080 squished
BATCH = 4               # frozen backbone -> activations don't store grads
EPOCHS, PATIENCE = 200, 15
LR = 1e-3               # head-only training
IGNORE_INDEX = 1        # base-background-layer-segmentation (same as train_fast)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    model = DinoSeg(num_labels=4).cuda()
    # snap IMG_SIZE up to a backbone patch multiple (768->768 for /16, ->770 for /14)
    p = model.backbone.config.patch_size
    img_size = IMG_SIZE + (-IMG_SIZE % p)
    print(f"backbone {model.backbone_name} patch={p} -> img_size {img_size}")

    train_ds = WaterlineDataset(os.path.join(DATASET, "train"), img_size)
    valid_ds = WaterlineDataset(os.path.join(DATASET, "valid"), img_size)
    test_ds = WaterlineDataset(os.path.join(DATASET, "test"), img_size)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, BATCH, shuffle=False, num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=2,
                             pin_memory=True)

    opt = torch.optim.AdamW(model.head_parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader) * EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda")

    best_iou, bad = 0.0, 0
    for ep in range(EPOCHS):
        model.head.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {ep + 1}/{EPOCHS}"):
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(pixel_values=images).logits
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            train_loss += loss.item()

        val_loss, val_iou = validate(model, valid_loader, "cuda", criterion, 4, IGNORE_INDEX)
        print(f"ep {ep + 1}: train_loss {train_loss / len(train_loader):.4f} "
              f"val_loss {val_loss:.4f} val mIoU {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou, bad = val_iou, 0
            model.save(SAVE_DIR, img_size)
            print(f"  new best -> saved to {SAVE_DIR}")
        elif (bad := bad + 1) >= PATIENCE:
            print(f"early stop at epoch {ep + 1}")
            break

    # test eval on the best head
    from dinoseg import load_dinoseg
    model, _ = load_dinoseg(SAVE_DIR, "cuda")
    test_loss, test_iou = validate(model, test_loader, "cuda", criterion, 4, IGNORE_INDEX)
    print(f"\nTEST: loss {test_loss:.4f} mIoU {test_iou:.4f} (best val {best_iou:.4f})")


if __name__ == "__main__":
    main()
