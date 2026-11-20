"""Grad-CAM for timm classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.utils import CLASS_NAMES_PATH, load_class_mapping


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)."""

    def __init__(
        self,
        model: nn.Module,
        mapping_path: Path = CLASS_NAMES_PATH,
    ) -> None:
        if not (hasattr(model, "forward_features") and hasattr(model, "forward_head")):
            raise TypeError(
                "GradCAM expects a timm model with forward_features/forward_head"
            )
        self.model = model
        mapping = load_class_mapping(mapping_path)
        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = mapping["idx_to_class"]

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int | str] = None,
    ) -> tuple[np.ndarray, torch.Tensor, str]:
        """Return (cam, logits, target_label_name) for single-image batch x (1, 3, H, W)."""
        self.model.eval()
        with torch.enable_grad():
            feats = self.model.forward_features(x)
            feats.retain_grad()
            logits = self.model.forward_head(feats)
            if isinstance(target_class, str):
                cls = self.class_to_idx[target_class.upper()]
            elif isinstance(target_class, int):
                cls = target_class
            elif logits.shape[-1] == 1:
                cls = int((torch.sigmoid(logits[0, 0]) >= 0.5).item())
            else:
                cls = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            score = logits[0, 0] if logits.shape[-1] == 1 else logits[0, cls]
            if logits.shape[-1] == 1 and cls == 0:
                score = -score
            score.backward()
            grads = feats.grad[0]
            weights = grads.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * feats[0]).sum(0)).detach().float().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.astype(np.float32), logits.detach(), self.idx_to_class[cls]


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Upsample ``cam`` to the image size and alpha-blend a colour map over it."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)
    heat = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = (alpha * heat + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)
