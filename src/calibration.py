"""Temperature scaling calibration and ECE diagnostics (fitted strictly on val)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.utils import MODELS_DIR, get_logger, load_json, save_json

logger = get_logger(__name__)

CALIBRATION_PATH: Path = MODELS_DIR / "calibration.json"
UNCERTAIN_LOW: float = 0.40
UNCERTAIN_HIGH: float = 0.60


def compute_ece(p_fake: np.ndarray, y_bin: np.ndarray, n_bins: int = 15) -> float:
    """Compute top-label Expected Calibration Error (Guo et al., 2017)."""
    p_fake = np.asarray(p_fake, dtype=np.float64)
    y_bin = np.asarray(y_bin, dtype=np.int64)
    if len(p_fake) == 0:
        return 0.0
    conf = np.where(p_fake >= 0.5, p_fake, 1.0 - p_fake)
    pred = (p_fake >= 0.5).astype(np.int64)
    correct = (pred == y_bin).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(p_fake))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(
                float(correct[mask].mean()) - float(conf[mask].mean())
            )
    return float(ece)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale raw logits by ``1 / temperature``."""
    return logits.float() / max(float(temperature), 1e-4)


def logits_to_p_fake(
    logits: torch.Tensor, fake_idx: int, temperature: float = 1.0
) -> np.ndarray:
    """Return ``P(FAKE)`` as a 1D float32 numpy array from ``(N, C)`` or ``(N, 1)`` logits."""
    scaled = apply_temperature(logits, temperature)
    if scaled.ndim == 1 or scaled.shape[-1] == 1:
        p1 = torch.sigmoid(scaled.view(-1))
        p_fake = p1 if fake_idx == 1 else (1.0 - p1)
    else:
        p_fake = torch.softmax(scaled, dim=-1)[:, fake_idx]
    return p_fake.detach().cpu().numpy().astype(np.float64)


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    fake_idx: int,
) -> dict[str, Any]:
    """Fit scalar temperature ``T > 0`` minimizing validation NLL via LBFGS."""
    val_logits_f = val_logits.detach().float().cpu()
    val_labels_l = val_labels.detach().long().cpu()
    y_bin = (val_labels_l.numpy() == fake_idx).astype(np.int64)

    p_before = logits_to_p_fake(val_logits_f, fake_idx=fake_idx, temperature=1.0)
    ece_before = compute_ece(p_before, y_bin)

    is_single = val_logits_f.ndim == 1 or val_logits_f.shape[-1] == 1
    if is_single:
        loss_fn = nn.BCEWithLogitsLoss()
        target_f = val_labels_l.float()
        nll_before = float(loss_fn(val_logits_f.view(-1), target_f).item())
    else:
        loss_fn = nn.CrossEntropyLoss()
        nll_before = float(loss_fn(val_logits_f, val_labels_l).item())

    log_t = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.05, max_iter=60, line_search_fn="strong_wolfe"
    )

    def _closure():
        optimizer.zero_grad()
        temp = torch.exp(log_t).clamp(0.1, 10.0)
        scaled = val_logits_f / temp
        loss = (
            loss_fn(scaled.view(-1), target_f)
            if is_single
            else loss_fn(scaled, val_labels_l)
        )
        loss.backward()
        return loss

    optimizer.step(_closure)
    temperature = float(torch.exp(log_t).clamp(0.1, 10.0).item())
    with torch.no_grad():
        scaled_after = val_logits_f / temperature
        nll_after = float(
            (
                loss_fn(scaled_after.view(-1), target_f)
                if is_single
                else loss_fn(scaled_after, val_labels_l)
            ).item()
        )
    # Guard against numerical failure: never return a temperature that worsens NLL.
    if nll_after > nll_before + 1e-6:
        logger.warning(
            "Temperature scaling did not improve val NLL (%.4f -> %.4f); keeping T=1.0",
            nll_before,
            nll_after,
        )
        temperature = 1.0
        nll_after = nll_before

    p_after = logits_to_p_fake(val_logits_f, fake_idx=fake_idx, temperature=temperature)
    ece_after = compute_ece(p_after, y_bin)
    logger.info(
        "Validation calibration: T=%.4f | val NLL %.4f -> %.4f | val ECE %.4f -> %.4f",
        temperature,
        nll_before,
        nll_after,
        ece_before,
        ece_after,
    )
    return {
        "temperature": temperature,
        "val_n": int(len(val_labels_l)),
        "val_nll_before": nll_before,
        "val_nll_after": nll_after,
        "val_ece_before": ece_before,
        "val_ece_after": ece_after,
        "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
    }


def save_calibration(payload: dict[str, Any], path: Path = CALIBRATION_PATH) -> None:
    """Persist temperature scaling metadata to ``models/calibration.json``."""
    save_json(payload, path)


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    """Load calibration settings (falls back to ``T=1.0`` with a warning if absent)."""
    if not path.exists():
        logger.warning("Calibration file missing at %s; using uncalibrated T=1.0", path)
        return {"temperature": 1.0, "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH]}
    return load_json(path)


def plot_reliability(
    p_before: np.ndarray,
    p_after: np.ndarray,
    y_bin: np.ndarray,
    ece_before: float,
    ece_after: float,
    path: Path,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram comparing uncalibrated vs. temperature-scaled predictions."""
    bg, fg, cyan, violet, grid = "#07090f", "#d7e3f4", "#22e4ff", "#a970ff", "#1c2333"
    edges = np.linspace(0.5, 1.0, n_bins + 1)

    def _curve(p: np.ndarray):
        conf = np.where(p >= 0.5, p, 1.0 - p)
        correct = ((p >= 0.5).astype(int) == y_bin).astype(float)
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
            if mask.sum() > 0:
                xs.append(float(conf[mask].mean()))
                ys.append(float(correct[mask].mean()))
        return xs, ys

    xb, yb = _curve(p_before)
    xa, ya = _curve(p_after)

    fig, ax = plt.subplots(figsize=(5.4, 4.4), facecolor=bg)
    ax.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=fg)
    ax.plot(
        [0.5, 1.0],
        [0.5, 1.0],
        ls="--",
        color="#5b667a",
        lw=1.2,
        label="Perfect calibration",
    )
    ax.plot(
        xb,
        yb,
        color=violet,
        marker="o",
        lw=2.0,
        label=f"Before T-scaling (ECE={ece_before:.4f})",
    )
    ax.plot(
        xa,
        ya,
        color=cyan,
        marker="s",
        lw=2.2,
        label=f"After T-scaling (ECE={ece_after:.4f})",
    )
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("Mean predicted confidence", color=fg)
    ax.set_ylabel("Empirical accuracy", color=fg)
    ax.set_title("Reliability diagram (test set, face-gated)", color=fg)
    ax.grid(color=grid, lw=0.6)
    ax.legend(facecolor=bg, edgecolor=grid, labelcolor=fg, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)
