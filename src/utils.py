"""Shared utilities: project paths, logging, seeding, device selection and class mapping."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
IMAGES_DIR: Path = DATA_DIR / "images"
MANIFEST_PATH: Path = DATA_DIR / "manifest.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CHECKPOINT_PATH: Path = MODELS_DIR / "best.pt"
CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

SEED: int = 42

# Keep pretrained-weight downloads inside the project so the cache is self-contained.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent, timestamped format."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger(name)


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> "torch.device":  # noqa: F821 - torch imported lazily
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty-printed JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    return json.loads(path.read_text())


def save_class_mapping(
    class_to_idx: dict[str, int], path: Path = CLASS_NAMES_PATH
) -> dict[str, Any]:
    """Persist ``class_to_idx`` and ``idx_to_class`` to ``models/class_names.json``."""
    normalized = {str(k).upper(): int(v) for k, v in class_to_idx.items()}
    idx_to_class = {
        str(v): k for k, v in sorted(normalized.items(), key=lambda kv: kv[1])
    }
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    payload: dict[str, Any] = {
        "class_to_idx": normalized,
        "idx_to_class": idx_to_class,
        "class_names": class_names,
    }
    save_json(payload, path)
    return payload


def load_class_mapping(path: Path = CLASS_NAMES_PATH) -> dict[str, Any]:
    """Load persisted class mapping (raises FileNotFoundError if missing)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping file missing at {path}. Run training (`python -m src.train`) "
            "to generate class_names.json."
        )
    data = load_json(path)
    if "class_to_idx" not in data or "idx_to_class" not in data:
        raise ValueError(f"Invalid class mapping schema in {path}: {data}")
    class_to_idx = {str(k).upper(): int(v) for k, v in data["class_to_idx"].items()}
    idx_to_class = {int(k): str(v).upper() for k, v in data["idx_to_class"].items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": class_names,
    }
