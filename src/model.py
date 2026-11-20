"""Model construction and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from src.utils import (
    CLASS_NAMES_PATH,
    get_logger,
    load_class_mapping,
    save_class_mapping,
)

logger = get_logger(__name__)

# Candidate backbones (timm names with explicit pretrained tags).
BACKBONES: dict[str, str] = {
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
}


def build_model(
    backbone: str,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    num_classes: int = 2,
) -> nn.Module:
    """Create a timm classifier with ``num_classes`` output logits."""
    name = BACKBONES.get(backbone, backbone)
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Built %s (%.1fM params, num_classes=%d, pretrained=%s)",
        name,
        n_params,
        num_classes,
        pretrained,
    )
    return model


def save_checkpoint(model: nn.Module, path: Path, meta: dict[str, Any]) -> None:
    """Save weights plus metadata (including ``class_to_idx``) and write ``class_names.json``."""
    if "class_to_idx" not in meta:
        raise ValueError("Checkpoint metadata must include 'class_to_idx'")
    mapping_payload = save_class_mapping(meta["class_to_idx"], CLASS_NAMES_PATH)
    meta = dict(meta)
    meta["class_to_idx"] = mapping_payload["class_to_idx"]
    meta["idx_to_class"] = mapping_payload["idx_to_class"]
    meta["class_names"] = mapping_payload["class_names"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from :func:`save_checkpoint` output and verify ``class_names.json``."""
    mapping = load_class_mapping(CLASS_NAMES_PATH)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(ckpt["meta"])
    if (
        "class_to_idx" in meta
        and {k.upper(): int(v) for k, v in meta["class_to_idx"].items()}
        != mapping["class_to_idx"]
    ):
        raise RuntimeError(
            f"Label mapping mismatch between checkpoint ({meta['class_to_idx']}) "
            f"and {CLASS_NAMES_PATH} ({mapping['class_to_idx']})"
        )
    meta["class_to_idx"] = mapping["class_to_idx"]
    meta["idx_to_class"] = mapping["idx_to_class"]
    meta["class_names"] = mapping["class_names"]
    num_classes = len(mapping["class_to_idx"])
    model = build_model(meta["backbone"], pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
