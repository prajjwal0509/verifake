"""Leave-One-Generator-Out (LOGO) cross-generator generalization benchmark."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.calibration import logits_to_p_fake  # noqa: E402
from src.data import (  # noqa: E402
    HELD_OUT_GENERATOR,
    balanced_face_subset,
    derive_class_to_idx_from_manifest,
    load_manifest,
    make_tta_loader,
)
from src.evaluate import run_tta_logits  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.train import fit  # noqa: E402
from src.utils import (  # noqa: E402
    CLASS_NAMES_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    get_device,
    get_logger,
    save_class_mapping,
    save_json,
    seed_everything,
)

logger = get_logger(__name__)

BG, FG, CYAN, VIOLET, AMBER, GRID = (
    "#07090f",
    "#d7e3f4",
    "#22e4ff",
    "#a970ff",
    "#ffb547",
    "#1c2333",
)


def plot_logo(folds: list[dict[str, Any]], mean_rate: float, path: Path) -> None:
    """Save bar chart of Leave-One-Generator-Out detection rates."""
    names = [f["generator"] for f in folds]
    rates = [100.0 * f["held_out_detection_rate"] for f in folds]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG)
    bars = ax.bar(names, rates, color=AMBER, width=0.52)
    for b, r in zip(bars, rates):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r + 1.2,
            f"{r:.1f}%",
            ha="center",
            color=FG,
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(
        100.0 * mean_rate,
        color=CYAN,
        ls="--",
        lw=1.6,
        label=f"Mean LOGO detection = {100.0 * mean_rate:.2f}%",
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Held-out detection rate (%)", color=FG)
    ax.set_title("Leave-One-Generator-Out (LOGO) generalization", color=FG)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def run_logo(
    generators: list[str],
    train_per_fold: int = 1200,
    val_per_fold: int = 400,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1.5e-4,
    img_size: int = 224,
) -> dict[str, Any]:
    """Run Leave-One-Generator-Out evaluation across ``generators``."""
    seed_everything(SEED)
    device = get_device()
    df = load_manifest()
    if "has_face" in df.columns:
        df = df[df["has_face"] == True].copy()  # noqa: E712

    class_to_idx = derive_class_to_idx_from_manifest(df)
    save_class_mapping(class_to_idx, CLASS_NAMES_PATH)
    pos_idx = class_to_idx["FAKE"]
    neg_idx = class_to_idx["REAL"]

    test_real_df = df[(df["split"] == "test") & (df["label"] == neg_idx)].reset_index(
        drop=True
    )
    if len(test_real_df) > 600:
        test_real_df = test_real_df.sample(n=600, random_state=SEED).reset_index(
            drop=True
        )

    folds: list[dict[str, Any]] = []
    tmp_ckpt = MODELS_DIR / "logo_fold_tmp.pt"

    for idx, held_out in enumerate(generators, start=1):
        logger.info("LOGO fold %d/%d: holding out %s", idx, len(generators), held_out)
        fold_train_pool = df[
            (df["split"] == "train")
            & (df["generator"] != held_out)
            & (df["generator"] != HELD_OUT_GENERATOR)
        ]
        fold_val_pool = df[
            (df["split"] == "val")
            & (df["generator"] != held_out)
            & (df["generator"] != HELD_OUT_GENERATOR)
        ]
        fold_train = balanced_face_subset(
            fold_train_pool, train_per_fold, seed=SEED + idx
        )
        fold_val = balanced_face_subset(fold_val_pool, val_per_fold, seed=SEED + idx)

        if tmp_ckpt.exists():
            tmp_ckpt.unlink()

        t0 = time.time()
        fit_res = fit(
            "convnext_tiny",
            fold_train,
            fold_val,
            class_to_idx=class_to_idx,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=0.05,
            img_size=img_size,
            label_smoothing=0.1,
            patience=99,
            workers=0,
            device=device,
            save_path=tmp_ckpt,
            resume_checkpoint=None,  # fresh ImageNet weights — never leak held-out generator
        )

        # Evaluate on all face-detected test samples of held_out + real baseline for ROC-AUC
        held_out_test_df = df[
            (df["split"] == "test") & (df["generator"] == held_out)
        ].reset_index(drop=True)
        fold_model, _ = load_checkpoint(tmp_ckpt, device)
        fake_loader = make_tta_loader(
            held_out_test_df,
            img_size,
            batch_size=max(8, batch_size // 2),
            workers=0,
            class_to_idx=class_to_idx,
        )
        real_loader = make_tta_loader(
            test_real_df,
            img_size,
            batch_size=max(8, batch_size // 2),
            workers=0,
            class_to_idx=class_to_idx,
        )
        fake_logits, _ = run_tta_logits(
            fold_model, fake_loader, device, amp=fit_res["amp"]
        )
        real_logits, _ = run_tta_logits(
            fold_model, real_loader, device, amp=fit_res["amp"]
        )
        p_fake_held = logits_to_p_fake(fake_logits, fake_idx=pos_idx, temperature=1.0)
        p_fake_real = logits_to_p_fake(real_logits, fake_idx=pos_idx, temperature=1.0)

        preds = (p_fake_held >= 0.5).astype(int)
        det_rate = float(preds.mean())
        auc_vs_real = float(
            roc_auc_score(
                np.concatenate([np.zeros(len(p_fake_real)), np.ones(len(p_fake_held))]),
                np.concatenate([p_fake_real, p_fake_held]),
            )
        )
        fold_rec = {
            "generator": held_out,
            "train_n": int(len(fold_train)),
            "val_n": int(len(fold_val)),
            "in_dist_val_acc": float(fit_res["best"]["val_accuracy"]),
            "held_out_n": int(len(held_out_test_df)),
            "detected_fake": int(preds.sum()),
            "missed_as_real": int((1 - preds).sum()),
            "held_out_detection_rate": det_rate,
            "held_out_mean_p_fake": float(p_fake_held.mean()),
            "held_out_roc_auc_vs_real": auc_vs_real,
            "seconds": round(time.time() - t0, 1),
        }
        folds.append(fold_rec)
        logger.info(
            "LOGO [%s]: in_dist_val=%.4f | held_out (n=%d) detection=%.4f mean_p_fake=%.4f auc_vs_real=%.4f (%.0fs)",
            held_out,
            fold_rec["in_dist_val_acc"],
            fold_rec["held_out_n"],
            det_rate,
            fold_rec["held_out_mean_p_fake"],
            auc_vs_real,
            fold_rec["seconds"],
        )

        del fold_model, fake_loader, real_loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    mean_det = float(np.mean([f["held_out_detection_rate"] for f in folds]))
    mean_auc = float(np.mean([f["held_out_roc_auc_vs_real"] for f in folds]))
    out = {
        "generators": generators,
        "train_per_fold": train_per_fold,
        "val_per_fold": val_per_fold,
        "epochs_per_fold": epochs,
        "folds": folds,
        "mean_detection_rate": mean_det,
        "mean_roc_auc_vs_real": mean_auc,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(out, REPORTS_DIR / "logo_results.json")
    plot_logo(folds, mean_det, REPORTS_DIR / "logo_generalization.png")
    logger.info(
        "LOGO complete: mean detection rate across %d folds = %.4f (mean AUC = %.4f)",
        len(folds),
        mean_det,
        mean_auc,
    )
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Leave-One-Generator-Out evaluation")
    parser.add_argument("--generators", nargs="+", default=["DALLE3", "FLUX", "SDXL"])
    parser.add_argument("--train-per-fold", type=int, default=1200)
    parser.add_argument("--val-per-fold", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()
    run_logo(
        generators=args.generators,
        train_per_fold=args.train_per_fold,
        val_per_fold=args.val_per_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
