"""DINOv3 frozen-backbone semantic segmentation model, shared by train/viz scripts.

Frozen ViT-B/16 backbone -> small conv head -> logits at input resolution.
Only the head (~2.4M params) is trained.
"""
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Ungated, works without HF auth (patch=14).
# Upgrade path: "facebook/dinov3-vitb16-pretrain-lvd1689m" (patch=16) — gated,
# requires accepting Meta's license on the HF page + `hf auth login`.
BACKBONE = "facebook/dinov2-base"


class DinoSeg(nn.Module):
    def __init__(self, num_labels: int = 4, backbone_name: str = BACKBONE):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone.requires_grad_(False)
        dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_labels, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.backbone.eval()  # guarantee frozen-eval even if outer .train() is called
        b, _, h, w = pixel_values.shape
        p = self.backbone.config.patch_size
        assert h % p == 0 and w % p == 0, \
            f"input {h}x{w} not divisible by patch size {p}"
        hs = self.backbone(pixel_values=pixel_values).last_hidden_state
        n = (h // p) * (w // p)
        tokens = hs[:, -n:, :]                       # patch tokens are the last N
        feats = tokens.transpose(1, 2).reshape(b, -1, h // p, w // p)
        logits = self.head(feats)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return SimpleNamespace(logits=logits)

    def head_parameters(self):
        return self.head.parameters()

    def save(self, out_dir: str, imgsz: int) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.head.state_dict(), os.path.join(out_dir, "dinoseg_head.pt"))
        with open(os.path.join(out_dir, "dinoseg_meta.json"), "w") as f:
            json.dump({"backbone": self.backbone_name,
                       "num_labels": self.num_labels, "imgsz": imgsz}, f)


def load_dinoseg(model_dir: str, device="cpu") -> tuple[DinoSeg, int]:
    """Rebuild DinoSeg from a saved dir. Returns (model, train_imgsz)."""
    with open(os.path.join(model_dir, "dinoseg_meta.json")) as f:
        meta = json.load(f)
    model = DinoSeg(num_labels=meta["num_labels"], backbone_name=meta["backbone"])
    model.head.load_state_dict(
        torch.load(os.path.join(model_dir, "dinoseg_head.pt"), map_location="cpu"))
    return model.to(device).eval(), meta["imgsz"]


def is_dinoseg_dir(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "dinoseg_meta.json"))
