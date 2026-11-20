"""Evaluation on face-gated test split, unseen generator, calibration, TTA, and robustness."""

from __future__ import annotations

import argparse
import io
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.calibration import (  # noqa: E402
    CALIBRATION_PATH,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    compute_ece,
    load_calibration,
    logits_to_p_fake,
    plot_reliability,
)
from src.data import (  # noqa: E402
    HELD_OUT_GENERATOR,
    balanced_face_subset,
    load_manifest,
    log_class_balance,
    make_loader,
    make_tta_loader,
    stratified_subset,
)
from src.model import load_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    CHECKPOINT_PATH,
    CLASS_NAMES_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    get_device,
    get_logger,
    load_class_mapping,
    load_json,
    save_json,
    seed_everything,
)

logger = get_logger(__name__)

# Plot palette for saved evaluation figures.
BG, FG, CYAN, VIOLET, AMBER, GRID = (
    "#07090f",
    "#d7e3f4",
    "#22e4ff",
    "#a970ff",
    "#ffb547",
    "#1c2333",
)


def autocast_ctx(device: torch.device, enabled: bool):
    """Return an autocast context for the device (no-op when AMP is disabled)."""
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    positive_index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_positive, labels)`` for single-view ``loader`` (used by train.py)."""
    if positive_index is None:
        positive_index = load_class_mapping(CLASS_NAMES_PATH)["class_to_idx"]["FAKE"]
    logits, labels = run_single_logits(model, loader, device, amp=amp)
    return logits_to_p_fake(logits, fake_idx=positive_index, temperature=1.0), labels


@torch.no_grad()
def run_single_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return raw ``(logits_cpu, labels_np)`` for a single-view DataLoader."""
    model.eval()
    logits_list, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            out = model(x)
        logits_list.append(out.float().cpu())
        labels.append(y.numpy())
    return torch.cat(logits_list, dim=0), np.concatenate(labels)


@torch.no_grad()
def run_tta_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return averaged 3-view TTA ``(logits_cpu, labels_np)`` from a TTA DataLoader."""
    model.eval()
    logits_list, labels = [], []
    for views, y in loader:
        b, v, c, h, w = views.shape
        flat = views.view(b * v, c, h, w).to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            out = model(flat)
        avg = out.float().view(b, v, -1).mean(dim=1)
        logits_list.append(avg.cpu())
        labels.append(y.numpy())
    return torch.cat(logits_list, dim=0), np.concatenate(labels)


def compute_metrics(
    p_positive: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    class_to_idx: Optional[dict[str, int]] = None,
) -> dict:
    """Binary metrics using ``class_to_idx`` from ``models/class_names.json``."""
    if class_to_idx is None:
        class_to_idx = load_class_mapping(CLASS_NAMES_PATH)["class_to_idx"]
    pos_idx = int(class_to_idx["FAKE"])
    neg_idx = int(class_to_idx["REAL"])
    y_bin = (y == pos_idx).astype(int)
    pred_bin = (p_positive >= threshold).astype(int)
    pred_idx = np.where(pred_bin == 1, pos_idx, neg_idx)
    ordered_indices = [i for _, i in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(y, pred_idx, labels=ordered_indices)
    has_both = len(np.unique(y_bin)) > 1
    auc = float(roc_auc_score(y_bin, p_positive)) if has_both else None
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y_bin, pred_bin)),
        "precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
        "roc_auc": auc if auc is not None else 0.0,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [
            k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])
        ],
        "threshold": threshold,
    }


def compute_uncertainty_stats(p_fake: np.ndarray, y_bin: np.ndarray) -> dict[str, Any]:
    """Report fraction and accuracy of predictions falling outside vs. inside [0.40, 0.60]."""
    uncertain = (p_fake >= UNCERTAIN_LOW) & (p_fake <= UNCERTAIN_HIGH)
    n_unc = int(uncertain.sum())
    n_dec = int((~uncertain).sum())
    dec_acc = (
        float(((p_fake[~uncertain] >= 0.5).astype(int) == y_bin[~uncertain]).mean())
        if n_dec > 0
        else 0.0
    )
    return {
        "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
        "uncertain_count": n_unc,
        "uncertain_rate": float(uncertain.mean()) if len(p_fake) else 0.0,
        "decisive_count": n_dec,
        "decisive_accuracy": dec_acc,
    }


def compute_per_generator_metrics(
    df: pd.DataFrame,
    p_positive: np.ndarray,
    y: np.ndarray,
    class_to_idx: dict[str, int],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Compute accuracy, mean confidence, and sample count per generator / real source."""
    if "generator" not in df.columns or len(df) == 0:
        return {}
    pos_idx = int(class_to_idx["FAKE"])
    neg_idx = int(class_to_idx["REAL"])
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_idx = np.where(p_positive >= threshold, pos_idx, neg_idx)
    gens = df["generator"].astype(str).to_numpy()
    out: dict[str, dict] = {}
    for gen_name in sorted(np.unique(gens)):
        mask = gens == gen_name
        y_g = y[mask]
        pred_g = pred_idx[mask]
        p_pos_g = p_positive[mask]
        true_cls_idx = int(y_g[0])
        true_cls_name = idx_to_class[true_cls_idx]
        p_true_cls = p_pos_g if true_cls_idx == pos_idx else (1.0 - p_pos_g)
        unc_g = (p_pos_g >= UNCERTAIN_LOW) & (p_pos_g <= UNCERTAIN_HIGH)
        out[str(gen_name)] = {
            "n": int(mask.sum()),
            "class": true_cls_name,
            "accuracy": float((pred_g == y_g).mean()),
            "mean_confidence": float(p_true_cls.mean()),
            "uncertain_count": int(unc_g.sum()),
        }
    return out


def jpeg_perturb(quality: int) -> Callable[[Image.Image], Image.Image]:
    """Re-encode as JPEG at a fixed quality."""

    def _f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _f


def resize_perturb(scale: float) -> Callable[[Image.Image], Image.Image]:
    """Downscale by ``scale`` (bilinear) then upscale back."""

    def _f(img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
        )
        return small.resize((w, h), Image.BILINEAR)

    return _f


def blur_perturb(radius: float = 1.5) -> Callable[[Image.Image], Image.Image]:
    """Apply fixed Gaussian blur."""

    def _f(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius))

    return _f


def social_media_perturb(
    scale: float = 0.5, quality: int = 55
) -> Callable[[Image.Image], Image.Image]:
    """Simulate a chained social-media re-encode: downscale+upscale followed by JPEG compression."""

    def _f(img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
        )
        up = small.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        up.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _f


ROBUSTNESS_SUITE: dict[str, Optional[Callable[[Image.Image], Image.Image]]] = {
    "original": None,
    "resize_0.5x": resize_perturb(0.5),
    "resize_0.25x": resize_perturb(0.25),
    "jpeg_q50": jpeg_perturb(50),
    "blur": blur_perturb(1.5),
    "social_media_chain": social_media_perturb(0.5, 55),
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def plot_confusion(cm: list[list[int]], class_names: list[str], path: Path) -> None:
    """Save a confusion-matrix heatmap with axis tick labels from ``class_names``."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.4), facecolor=BG)
    _style(ax)
    ax.imshow(cm_arr, cmap="cool", alpha=0.85)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = 100 * cm_arr[i, j] / max(1, cm_arr[i].sum())
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]:,}\n{pct:.2f}%",
                ha="center",
                va="center",
                color="#05070c",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xticks(list(range(len(class_names))), class_names)
    ax.set_yticks(list(range(len(class_names))), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (face-gated test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_roc(p_positive: np.ndarray, y_bin: np.ndarray, auc: float, path: Path) -> None:
    """Save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_bin, p_positive)
    fig, ax = plt.subplots(figsize=(5, 4.4), facecolor=BG)
    _style(ax)
    ax.plot(fpr, tpr, color=CYAN, lw=2.2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color=VIOLET, lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (face-gated test set)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_robustness(
    results: dict[str, dict],
    path: Path,
    before_after: Optional[dict[str, dict]] = None,
) -> None:
    """Save a bar chart of robustness accuracy (grouped Before vs. After when available)."""
    names = list(results)
    after_accs = [100 * results[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=BG)
    _style(ax)
    x = np.arange(len(names))
    if before_after:
        before_accs = [100 * before_after[n]["before_acc"] for n in names]
        w = 0.38
        b1 = ax.bar(
            x - w / 2,
            before_accs,
            width=w,
            color=VIOLET,
            label="Before degradation aug",
        )
        b2 = ax.bar(
            x + w / 2,
            after_accs,
            width=w,
            color=CYAN,
            label="After degradation aug + TTA",
        )
        for b, a in zip(b1, before_accs):
            ax.text(
                b.get_x() + b.get_width() / 2,
                a + 0.6,
                f"{a:.1f}",
                ha="center",
                color=FG,
                fontsize=8,
            )
        for b, a in zip(b2, after_accs):
            ax.text(
                b.get_x() + b.get_width() / 2,
                a + 0.6,
                f"{a:.1f}",
                ha="center",
                color=FG,
                fontsize=8,
            )
        min_v = min(before_accs + after_accs)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower left")
        ax.set_title(
            "Robustness before vs. after degradation augmentation (n=600 face-gated)"
        )
    else:
        min_v = min(after_accs)
        colors = [CYAN if n == "original" else VIOLET for n in names]
        bars = ax.bar(x, after_accs, color=colors)
        for b, a in zip(bars, after_accs):
            ax.text(
                b.get_x() + b.get_width() / 2,
                a + 0.6,
                f"{a:.1f}",
                ha="center",
                color=FG,
                fontsize=9,
            )
        ax.set_title("Robustness: test accuracy under perturbations")
    ax.set_xticks(x, names)
    ax.set_ylim(max(0, min(50, min_v - 8)), 103)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_per_generator(
    per_gen: dict[str, dict], unseen: Optional[dict], path: Path
) -> None:
    """Save a bar chart of accuracy per face-gated source and held-out unseen generator."""
    items = list(per_gen.items())
    if unseen and unseen.get("n", 0) > 0:
        items.append(
            (
                f"{unseen['generator']} (UNSEEN)",
                {
                    "accuracy": unseen["accuracy"],
                    "class": "UNSEEN",
                    "n": unseen["n"],
                },
            )
        )
    if not items:
        return
    names = [k for k, _ in items]
    accs = [100 * v["accuracy"] for _, v in items]
    colors = [
        AMBER
        if v.get("class") == "UNSEEN"
        else (CYAN if v.get("class") == "REAL" else VIOLET)
        for _, v in items
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=BG)
    _style(ax)
    bars = ax.bar(names, accs, color=colors)
    for b, a in zip(bars, accs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            a + 0.8,
            f"{a:.1f}%",
            ha="center",
            color=FG,
            fontsize=8.5,
        )
    ax.set_ylim(0, 106)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Per-source & per-generator accuracy (face-gated | cyan=REAL, violet=FAKE, amber=Unseen)"
    )
    ax.grid(axis="y", color=GRID, lw=0.6)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def evaluate(
    checkpoint: Path = CHECKPOINT_PATH,
    output: Path = METRICS_PATH,
    before_metrics: Optional[Path] = None,
    batch_size: int = 64,
    workers: int = 0,
    limit: Optional[int] = None,
    robustness_limit: int = 600,
    skip_plots: bool = False,
) -> dict:
    """Evaluate ``checkpoint`` on untouched face-gated test split, unseen generator & robustness."""
    seed_everything()
    device = get_device()
    mapping = load_class_mapping(CLASS_NAMES_PATH)
    class_to_idx: dict[str, int] = mapping["class_to_idx"]
    idx_to_class: dict[int, str] = mapping["idx_to_class"]
    class_names: list[str] = mapping["class_names"]
    pos_idx = class_to_idx["FAKE"]

    model, meta = load_checkpoint(checkpoint, device)
    img_size = int(meta["img_size"])
    amp = bool(meta.get("amp", False))

    if "temperature" in meta:
        temperature = float(meta["temperature"])
    elif (
        checkpoint.resolve() == CHECKPOINT_PATH.resolve() and CALIBRATION_PATH.exists()
    ):
        temperature = float(load_calibration(CALIBRATION_PATH).get("temperature", 1.0))
    else:
        temperature = 1.0
    logger.info(
        "Evaluating checkpoint %s (temperature T=%.4f)", checkpoint, temperature
    )

    df = load_manifest()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    unseen_df = df[df["split"] == "unseen_generator"].reset_index(drop=True)

    has_face_col = "has_face" in df.columns
    if has_face_col:
        in_scope_df = test_df[test_df["has_face"] == True].reset_index(drop=True)  # noqa: E712
        no_face_test_df = test_df[test_df["has_face"] == False].reset_index(drop=True)  # noqa: E712
        unseen_in_scope_df = unseen_df[unseen_df["has_face"] == True].reset_index(  # noqa: E712
            drop=True
        )
        unseen_no_face_df = unseen_df[unseen_df["has_face"] == False].reset_index(  # noqa: E712
            drop=True
        )
    else:
        in_scope_df = test_df
        no_face_test_df = test_df.iloc[0:0]
        unseen_in_scope_df = unseen_df
        unseen_no_face_df = unseen_df.iloc[0:0]

    out_of_scope_report = {
        "test_no_face_total": int(len(no_face_test_df)),
        "test_no_face_by_source": (
            no_face_test_df["generator"].value_counts().to_dict()
            if "generator" in no_face_test_df.columns
            else {}
        ),
        "coco_total_test": int((test_df["generator"] == "COCO").sum())
        if "generator" in test_df.columns
        else 0,
        "coco_in_scope_faces": int((in_scope_df["generator"] == "COCO").sum())
        if "generator" in in_scope_df.columns
        else 0,
        "coco_out_of_scope_no_face": int((no_face_test_df["generator"] == "COCO").sum())
        if "generator" in no_face_test_df.columns
        else 0,
        "unseen_no_face_total": int(len(unseen_no_face_df)),
    }
    logger.info("Out-of-scope (no_face gated): %s", out_of_scope_report)

    if limit and limit < len(in_scope_df):
        in_scope_df = stratified_subset(in_scope_df, limit).reset_index(drop=True)
        logger.warning(
            "Evaluating on a %d-image stratified face-gated test subset (--limit)",
            len(in_scope_df),
        )
    log_class_balance(in_scope_df, "test_faces", idx_to_class)

    t0 = time.time()
    single_loader = make_loader(
        in_scope_df,
        img_size,
        False,
        batch_size,
        workers,
        class_to_idx=class_to_idx,
        use_face_crop=True,
    )
    single_logits, y_single = run_single_logits(model, single_loader, device, amp=amp)
    y_bin = (y_single == pos_idx).astype(np.int64)
    p_single_uncal = logits_to_p_fake(single_logits, fake_idx=pos_idx, temperature=1.0)
    p_single_cal = logits_to_p_fake(
        single_logits, fake_idx=pos_idx, temperature=temperature
    )
    single_metrics = compute_metrics(p_single_cal, y_single, class_to_idx=class_to_idx)
    single_metrics["ece_before"] = compute_ece(p_single_uncal, y_bin)
    single_metrics["ece_after"] = compute_ece(p_single_cal, y_bin)
    logger.info(
        "Single-view (face_crop, n=%d): acc=%.4f f1=%.4f auc=%.4f ece=%.4f->%.4f (%.0fs)",
        single_metrics["n"],
        single_metrics["accuracy"],
        single_metrics["f1"],
        single_metrics["roc_auc"],
        single_metrics["ece_before"],
        single_metrics["ece_after"],
        time.time() - t0,
    )

    t0 = time.time()
    tta_loader = make_tta_loader(
        in_scope_df,
        img_size,
        batch_size=max(8, batch_size // 2),
        workers=workers,
        class_to_idx=class_to_idx,
    )
    tta_logits, y_tta = run_tta_logits(model, tta_loader, device, amp=amp)
    p_tta_uncal = logits_to_p_fake(tta_logits, fake_idx=pos_idx, temperature=1.0)
    p_tta_cal = logits_to_p_fake(tta_logits, fake_idx=pos_idx, temperature=temperature)
    clean = compute_metrics(p_tta_cal, y_tta, class_to_idx=class_to_idx)
    ece_before = compute_ece(p_tta_uncal, y_bin)
    ece_after = compute_ece(p_tta_cal, y_bin)
    clean["ece_before"] = ece_before
    clean["ece_after"] = ece_after
    clean["temperature"] = temperature
    clean["uncertainty"] = compute_uncertainty_stats(p_tta_cal, y_bin)
    logger.info(
        "3-view TTA (n=%d): acc=%.4f f1=%.4f auc=%.4f | ECE %.4f -> %.4f (T=%.4f) | uncertain=%.1f%% (%.0fs)",
        clean["n"],
        clean["accuracy"],
        clean["f1"],
        clean["roc_auc"],
        ece_before,
        ece_after,
        temperature,
        100 * clean["uncertainty"]["uncertain_rate"],
        time.time() - t0,
    )

    tta_comparison = {
        "views": ["face_crop", "flipped_face_crop", "full_frame"],
        "single_view_accuracy": single_metrics["accuracy"],
        "single_view_f1": single_metrics["f1"],
        "single_view_auc": single_metrics["roc_auc"],
        "tta_3view_accuracy": clean["accuracy"],
        "tta_3view_f1": clean["f1"],
        "tta_3view_auc": clean["roc_auc"],
        "delta_accuracy": clean["accuracy"] - single_metrics["accuracy"],
        "delta_f1": clean["f1"] - single_metrics["f1"],
        "delta_auc": clean["roc_auc"] - single_metrics["roc_auc"],
        "improved_accuracy": bool(clean["accuracy"] > single_metrics["accuracy"]),
    }

    per_generator = compute_per_generator_metrics(
        in_scope_df, p_tta_cal, y_tta, class_to_idx
    )
    for gen_name, gm in per_generator.items():
        logger.info(
            "In-scope test %s (%s, n=%d): acc=%.4f mean_conf=%.4f uncertain=%d",
            gen_name,
            gm["class"],
            gm["n"],
            gm["accuracy"],
            gm["mean_confidence"],
            gm["uncertain_count"],
        )

    unseen_report: Optional[dict] = None
    if len(unseen_in_scope_df) > 0:
        t0 = time.time()
        unseen_loader = make_tta_loader(
            unseen_in_scope_df,
            img_size,
            batch_size=max(8, batch_size // 2),
            workers=workers,
            class_to_idx=class_to_idx,
        )
        u_logits, _ = run_tta_logits(model, unseen_loader, device, amp=amp)
        u_probs = logits_to_p_fake(u_logits, fake_idx=pos_idx, temperature=temperature)
        u_pred_fake = (u_probs >= 0.5).astype(int)
        u_acc = float(u_pred_fake.mean())
        real_mask = y_tta == class_to_idx["REAL"]
        combined_probs = np.concatenate([p_tta_cal[real_mask], u_probs])
        combined_y = np.concatenate(
            [np.zeros(real_mask.sum(), dtype=int), np.ones(len(u_probs), dtype=int)]
        )
        u_auc = (
            float(roc_auc_score(combined_y, combined_probs))
            if real_mask.sum() > 0
            else 0.0
        )
        u_unc = ((u_probs >= UNCERTAIN_LOW) & (u_probs <= UNCERTAIN_HIGH)).sum()
        unseen_report = {
            "generator": HELD_OUT_GENERATOR,
            "n": int(len(unseen_in_scope_df)),
            "n_total_held_out": int(len(unseen_df)),
            "n_no_face_gated": int(len(unseen_no_face_df)),
            "true_class": "FAKE",
            "accuracy": u_acc,
            "detection_rate_recall": u_acc,
            "mean_fake_probability": float(u_probs.mean()),
            "roc_auc_vs_test_real": u_auc,
            "detected_fake": int(u_pred_fake.sum()),
            "missed_as_real": int((1 - u_pred_fake).sum()),
            "uncertain_count": int(u_unc),
        }
        logger.info(
            "Unseen %s (faces=%d, no_face=%d): acc=%.4f mean_p_fake=%.4f auc=%.4f (%.0fs)",
            HELD_OUT_GENERATOR,
            unseen_report["n"],
            unseen_report["n_no_face_gated"],
            unseen_report["accuracy"],
            unseen_report["mean_fake_probability"],
            unseen_report["roc_auc_vs_test_real"],
            time.time() - t0,
        )

    robust_slice = balanced_face_subset(
        in_scope_df, min(robustness_limit, len(in_scope_df)), seed=42
    )
    robustness: dict[str, dict] = {}
    for name, perturb in ROBUSTNESS_SUITE.items():
        t0 = time.time()
        r_loader = make_tta_loader(
            robust_slice,
            img_size,
            batch_size=max(8, batch_size // 2),
            workers=workers,
            pre_transform=perturb,
            class_to_idx=class_to_idx,
        )
        r_logits, r_y = run_tta_logits(model, r_loader, device, amp=amp)
        r_probs = logits_to_p_fake(r_logits, fake_idx=pos_idx, temperature=temperature)
        rm = compute_metrics(r_probs, r_y, class_to_idx=class_to_idx)
        robustness[name] = {k: rm[k] for k in ("accuracy", "f1", "roc_auc", "n")}
        logger.info(
            "Robustness %-19s (n=%d): acc=%.4f f1=%.4f auc=%.4f (%.0fs)",
            name,
            rm["n"],
            rm["accuracy"],
            rm["f1"],
            rm["roc_auc"],
            time.time() - t0,
        )

    robustness_before_after: Optional[dict[str, dict]] = None
    if before_metrics and before_metrics.exists():
        b_data = load_json(before_metrics)
        b_rob = b_data.get("robustness", {})
        robustness_before_after = {}
        for name, cur in robustness.items():
            prev = b_rob.get(name, {})
            b_acc = float(prev.get("accuracy", cur["accuracy"]))
            b_f1 = float(prev.get("f1", cur["f1"]))
            b_auc = float(prev.get("roc_auc", cur["roc_auc"]))
            robustness_before_after[name] = {
                "n": cur["n"],
                "before_acc": b_acc,
                "after_acc": cur["accuracy"],
                "delta_acc": cur["accuracy"] - b_acc,
                "before_f1": b_f1,
                "after_f1": cur["f1"],
                "before_auc": b_auc,
                "after_auc": cur["roc_auc"],
            }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not skip_plots:
        plot_confusion(
            clean["confusion_matrix"], class_names, REPORTS_DIR / "confusion_matrix.png"
        )
        plot_roc(p_tta_cal, y_bin, clean["roc_auc"], REPORTS_DIR / "roc_curve.png")
        plot_robustness(
            robustness,
            REPORTS_DIR / "robustness.png",
            before_after=robustness_before_after,
        )
        plot_per_generator(
            per_generator, unseen_report, REPORTS_DIR / "per_generator_accuracy.png"
        )
        plot_reliability(
            p_tta_uncal,
            p_tta_cal,
            y_bin,
            ece_before,
            ece_after,
            REPORTS_DIR / "calibration.png",
        )

    train_summary = {}
    summary_path = REPORTS_DIR / "train_summary.json"
    if summary_path.exists():
        train_summary = load_json(summary_path)

    logo_data = {}
    logo_path = REPORTS_DIR / "logo_results.json"
    if logo_path.exists():
        logo_data = load_json(logo_path)

    cal_file_data = (
        load_calibration(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else {}
    )

    metrics = {
        "split": "test (face-gated 30% held-out) + unseen_generator (Midjourney held-out 100%)",
        "backbone": meta["backbone"],
        "img_size": img_size,
        "test": clean,
        "single_view_test": single_metrics,
        "tta_comparison": tta_comparison,
        "calibration": {
            "temperature": temperature,
            "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            "test_ece_before": ece_before,
            "test_ece_after": ece_after,
            "val_calibration": cal_file_data,
        },
        "out_of_scope_no_face": out_of_scope_report,
        "per_generator": per_generator,
        "unseen_generator": unseen_report,
        "robustness": robustness,
        "robustness_before_after": robustness_before_after,
        "logo_generalization": logo_data,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "positive_class": "FAKE",
        "positive_class_index": pos_idx,
        "split_sizes": df["split"].value_counts().to_dict(),
        "in_scope_face_split_sizes": (
            df[df["has_face"] == True]["split"].value_counts().to_dict()  # noqa: E712
            if has_face_col
            else df["split"].value_counts().to_dict()
        ),
        "training": train_summary,
    }
    save_json(metrics, output)
    logger.info("Saved metrics to %s", output)
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate face-gated detector, TTA, calibration & robustness"
    )
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--output", type=str, default=str(METRICS_PATH))
    parser.add_argument("--before-metrics", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--robustness-limit", type=int, default=600)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    evaluate(
        checkpoint=Path(args.checkpoint),
        output=Path(args.output),
        before_metrics=Path(args.before_metrics) if args.before_metrics else None,
        batch_size=args.batch_size,
        workers=args.workers,
        limit=args.limit,
        robustness_limit=args.robustness_limit,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
