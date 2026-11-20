"""Training: backbone benchmark, fine-tuning with early stopping, and val temperature scaling."""

from __future__ import annotations

import argparse
import gc
import math
import time
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.calibration import CALIBRATION_PATH, fit_temperature, save_calibration  # noqa: E402
from src.data import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    balanced_face_subset,
    derive_class_to_idx_from_manifest,
    load_manifest,
    log_class_balance,
    make_loader,
    make_tta_loader,
    stratified_subset,
)
from src.evaluate import autocast_ctx, compute_metrics, run_inference  # noqa: E402
from src.model import BACKBONES, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    CHECKPOINT_PATH,
    CLASS_NAMES_PATH,
    REPORTS_DIR,
    SEED,
    get_device,
    get_logger,
    save_class_mapping,
    save_json,
    seed_everything,
)

logger = get_logger(__name__)


def make_grad_scaler(device: torch.device, amp: bool) -> Optional[torch.amp.GradScaler]:
    """GradScaler for fp16 autocast on CUDA/MPS (None when AMP is off)."""
    if not amp or device.type == "cpu":
        return None
    return torch.amp.GradScaler(device.type)


def probe_amp(model: nn.Module, device: torch.device, img_size: int) -> bool:
    """Check that fp16 autocast + GradScaler works and does not degrade speed on this device."""
    if device.type == "cpu":
        return False
    if device.type == "mps":
        logger.warning(
            "MPS detected: PyTorch fp16 autocast + GradScaler on Apple MPS causes "
            "fp16<->fp32 cast overhead (measured ~35%% slower than native fp32) and "
            "GradScaler skips optimizer steps unnecessarily; using native fp32 on MPS."
        )
        return False
    try:
        scaler = make_grad_scaler(device, True)
        x = torch.randn(2, 3, img_size, img_size, device=device)
        with autocast_ctx(device, True):
            loss = model(x).float().mean()
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)
        logger.info("AMP (fp16 autocast + GradScaler) enabled on %s", device.type)
        return True
    except Exception as exc:  # pragma: no cover - device dependent
        logger.warning("AMP unavailable on %s (%s); training in fp32", device.type, exc)
        model.zero_grad(set_to_none=True)
        return False


def warmup_cosine(total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    """LR multiplier: linear warmup then cosine decay to ``min_ratio``."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    return fn


def fit(
    backbone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    class_to_idx: dict[str, int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    img_size: int,
    label_smoothing: float,
    patience: int,
    workers: int,
    device: torch.device,
    save_path: Optional[Any] = None,
    max_steps_per_epoch: Optional[int] = None,
    use_amp: bool = True,
    resume_checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Fine-tune ``backbone``; returns history and best validation metrics."""
    seed_everything(SEED)
    train_loader = make_loader(
        train_df, img_size, True, batch_size, workers, class_to_idx=class_to_idx
    )
    val_loader = make_loader(
        val_df, img_size, False, batch_size * 2, workers, class_to_idx=class_to_idx
    )
    train_dataset = train_loader.dataset
    logger.info(
        "train_dataset.class_to_idx = %s | train_dataset.classes = %s",
        train_dataset.class_to_idx,
        train_dataset.classes,
    )
    save_class_mapping(train_dataset.class_to_idx, CLASS_NAMES_PATH)

    model = build_model(
        backbone,
        pretrained=(resume_checkpoint is None),
        num_classes=len(train_dataset.class_to_idx),
    )
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Initialized weights from checkpoint: %s", resume_checkpoint)
    model = model.to(device)
    amp = probe_amp(model, device, img_size) if use_amp else False
    scaler = make_grad_scaler(device, amp)
    pos_idx = train_dataset.class_to_idx["FAKE"]

    steps_per_epoch = len(train_loader)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, min(steps_per_epoch, total_steps // 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, warmup_steps)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history: list[dict[str, float]] = []
    best: dict[str, Any] = {"val_accuracy": -1.0, "val_roc_auc": -1.0, "epoch": 0}
    bad_epochs = 0
    logger.info(
        "Training %s: %d steps/epoch, %d epochs, warmup %d steps",
        backbone,
        steps_per_epoch,
        epochs,
        warmup_steps,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if (step + 1) % 25 == 0 or (step + 1) == steps_per_epoch:
                logger.info(
                    "  ep %d step %d/%d loss %.4f lr %.2e %.1f img/s",
                    epoch,
                    step + 1,
                    steps_per_epoch,
                    running / seen,
                    scheduler.get_last_lr()[0],
                    seen / (time.time() - t0),
                )

        train_time = time.time() - t0
        probs, labels = run_inference(
            model, val_loader, device, amp, positive_index=pos_idx
        )
        val = compute_metrics(probs, labels, class_to_idx=train_dataset.class_to_idx)
        y_bin = (labels == pos_idx).astype(np.float32)
        val_loss = float(
            nn.functional.binary_cross_entropy(
                torch.tensor(np.clip(probs, 1e-7, 1 - 1e-7), dtype=torch.float32),
                torch.tensor(y_bin, dtype=torch.float32),
            )
        )
        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "val_accuracy": val["accuracy"],
            "val_f1": val["f1"],
            "val_roc_auc": val["roc_auc"],
            "epoch_seconds": train_time,
            "train_img_per_s": seen / train_time,
        }
        history.append(rec)
        logger.info(
            "Epoch %d | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_f1 %.4f | "
            "val_auc %.5f | %.0fs",
            epoch,
            rec["train_loss"],
            val_loss,
            val["accuracy"],
            val["f1"],
            val["roc_auc"],
            train_time,
        )

        improved = (val["accuracy"], val["roc_auc"]) > (
            best["val_accuracy"],
            best["val_roc_auc"],
        )
        if improved:
            best = {
                "val_accuracy": val["accuracy"],
                "val_roc_auc": val["roc_auc"],
                "val_f1": val["f1"],
                "epoch": epoch,
            }
            bad_epochs = 0
            if save_path is not None:
                save_checkpoint(
                    model,
                    save_path,
                    {
                        "backbone": backbone,
                        "timm_name": BACKBONES.get(backbone, backbone),
                        "img_size": img_size,
                        "normalize_mean": list(IMAGENET_MEAN),
                        "normalize_std": list(IMAGENET_STD),
                        "class_to_idx": train_dataset.class_to_idx,
                        "amp": amp,
                        "epoch": epoch,
                        "val": best,
                    },
                )
                logger.info("Saved new best checkpoint (epoch %d)", epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(
                    "Early stopping: no val improvement for %d epochs", patience
                )
                break

    out = {"backbone": backbone, "history": history, "best": best, "amp": amp}
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def plot_history(history: list[dict[str, float]], path) -> None:
    """Save loss / accuracy curves."""
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#07090f")
    for ax in axes:
        ax.set_facecolor("#07090f")
        ax.tick_params(colors="#d7e3f4")
        for s in ax.spines.values():
            s.set_color("#1c2333")
        ax.grid(color="#1c2333")
    axes[0].plot(
        ep,
        [h["train_loss"] for h in history],
        color="#22e4ff",
        marker="o",
        label="train (smoothed CE)",
    )
    axes[0].plot(
        ep,
        [h["val_loss"] for h in history],
        color="#a970ff",
        marker="o",
        label="val (BCE)",
    )
    axes[0].set_title("Loss", color="#d7e3f4")
    axes[1].plot(
        ep,
        [h["val_accuracy"] for h in history],
        color="#22e4ff",
        marker="o",
        label="val accuracy",
    )
    axes[1].plot(
        ep,
        [h["val_roc_auc"] for h in history],
        color="#a970ff",
        marker="o",
        label="val AUC",
    )
    axes[1].set_title("Validation", color="#d7e3f4")
    for ax in axes:
        ax.set_xlabel("epoch", color="#d7e3f4")
        ax.legend(facecolor="#07090f", edgecolor="#1c2333", labelcolor="#d7e3f4")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#07090f")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Train the deepfake detector")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument(
        "--skip-benchmark", action="store_true", help="use the first backbone directly"
    )
    p.add_argument(
        "--bench-train", type=int, default=2000, help="train images per benchmark run"
    )
    p.add_argument(
        "--bench-val", type=int, default=800, help="val images per benchmark run"
    )
    p.add_argument("--bench-epochs", type=int, default=1)
    p.add_argument(
        "--train-subset", type=int, default=None, help="cap on final training images"
    )
    p.add_argument(
        "--val-subset",
        type=int,
        default=None,
        help="cap on validation images per epoch",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (0 avoids macOS MPS spawn deadlock)",
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="optional checkpoint path to initialize weights from",
    )
    p.add_argument(
        "--faces-only",
        action="store_true",
        help="filter train/val to YuNet has_face==True",
    )
    p.add_argument(
        "--balanced-faces",
        action="store_true",
        help="balance train subset 50/50 across face-detected sources",
    )
    args = p.parse_args()

    seed_everything(SEED)
    device = get_device()
    logger.info("Device: %s", device)
    if device.type == "cpu":
        logger.warning("No GPU found; training on CPU subset (accuracy will be lower).")
        args.train_subset = min(args.train_subset or 4000, 4000)
        args.bench_train, args.bench_val, args.epochs = 1500, 500, min(args.epochs, 3)

    df = load_manifest()
    class_to_idx = derive_class_to_idx_from_manifest(df)
    mapping_payload = save_class_mapping(class_to_idx, CLASS_NAMES_PATH)
    idx_to_class = {int(k): v for k, v in mapping_payload["idx_to_class"].items()}
    logger.info(
        "Confirmed dataset class_to_idx = %s | idx_to_class = %s",
        class_to_idx,
        idx_to_class,
    )

    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    if args.faces_only and "has_face" in df.columns:
        train_df = train_df[train_df["has_face"] == True].copy()  # noqa: E712
        val_df = val_df[val_df["has_face"] == True].copy()  # noqa: E712
        logger.info("Filtered train/val to YuNet face-detected images only")
    log_class_balance(train_df, "train", idx_to_class)
    log_class_balance(val_df, "val", idx_to_class)

    hp = dict(
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        label_smoothing=args.label_smoothing,
        workers=args.workers,
        device=device,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark: list[dict[str, Any]] = []
    chosen = args.backbones[0]
    if not args.skip_benchmark and len(args.backbones) > 1:
        b_train = stratified_subset(train_df, args.bench_train)
        b_val = stratified_subset(val_df, args.bench_val)
        logger.info(
            "Backbone benchmark: %d train / %d val, %d epochs",
            len(b_train),
            len(b_val),
            args.bench_epochs,
        )
        for bb in args.backbones:
            t0 = time.time()
            res = fit(bb, b_train, b_val, epochs=args.bench_epochs, patience=99, **hp)
            res["seconds"] = time.time() - t0
            benchmark.append(res)
            logger.info(
                "Benchmark %s: val_acc %.4f val_auc %.5f (%.0fs)",
                bb,
                res["best"]["val_accuracy"],
                res["best"]["val_roc_auc"],
                res["seconds"],
            )
        chosen = max(
            benchmark,
            key=lambda r: (r["best"]["val_accuracy"], r["best"]["val_roc_auc"]),
        )["backbone"]
        save_json(benchmark, REPORTS_DIR / "backbone_benchmark.json")
        logger.info("Selected backbone: %s", chosen)
    elif (REPORTS_DIR / "backbone_benchmark.json").exists():
        from src.utils import load_json

        benchmark = load_json(REPORTS_DIR / "backbone_benchmark.json")

    final_train = (
        balanced_face_subset(train_df, args.train_subset)
        if args.balanced_faces
        else stratified_subset(train_df, args.train_subset)
    )
    final_val = stratified_subset(val_df, args.val_subset)
    if "generator" in final_train.columns:
        logger.info(
            "Final train generator mix: %s",
            final_train["generator"].value_counts().to_dict(),
        )
    if len(final_train) < len(train_df):
        logger.warning(
            "Final training uses a %d-image subset of %d train images",
            len(final_train),
            len(train_df),
        )
    logger.info(
        "Final training: %s on %d train / %d val images",
        chosen,
        len(final_train),
        len(final_val),
    )
    t0 = time.time()
    result = fit(
        chosen,
        final_train,
        final_val,
        epochs=args.epochs,
        patience=args.patience,
        save_path=CHECKPOINT_PATH,
        resume_checkpoint=args.resume_checkpoint,
        **hp,
    )
    plot_history(result["history"], REPORTS_DIR / "training_curves.png")

    # Fit temperature scaling strictly on validation 3-view TTA logits (never touch test/unseen).
    best_model, _ = load_checkpoint(CHECKPOINT_PATH, device)
    tta_val_loader = make_tta_loader(
        final_val,
        args.img_size,
        batch_size=max(8, args.batch_size // 2),
        workers=args.workers,
        class_to_idx=class_to_idx,
    )
    val_logits_list, val_labels_list = [], []
    best_model.eval()
    with torch.no_grad():
        for views, y in tta_val_loader:
            b, v, c, h, w = views.shape
            flat = views.view(b * v, c, h, w).to(device, non_blocking=True)
            with autocast_ctx(device, result["amp"]):
                logits = best_model(flat).float().view(b, v, -1).mean(dim=1)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(y)
    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)
    cal_report = fit_temperature(val_logits, val_labels, fake_idx=class_to_idx["FAKE"])
    save_calibration(cal_report, CALIBRATION_PATH)
    ckpt_blob = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ckpt_blob["meta"]["temperature"] = cal_report["temperature"]
    torch.save(ckpt_blob, CHECKPOINT_PATH)
    del best_model, tta_val_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    summary = {
        "backbone": chosen,
        "device": str(device),
        "amp": result["amp"],
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "calibration": cal_report,
        "train_images": len(final_train),
        "train_images_available": len(train_df),
        "val_images": len(final_val),
        "val_images_available": len(val_df),
        "best_epoch": result["best"]["epoch"],
        "best_val": result["best"],
        "epochs_run": len(result["history"]),
        "total_train_minutes": round((time.time() - t0) / 60, 1),
        "hyperparameters": {k: v for k, v in vars(args).items()},
        "history": result["history"],
        "benchmark": [
            {"backbone": r["backbone"], "best": r["best"], "seconds": r["seconds"]}
            for r in benchmark
        ],
    }
    save_json(summary, REPORTS_DIR / "train_summary.json")
    logger.info(
        "Training done: best val acc %.4f at epoch %d (%s)",
        result["best"]["val_accuracy"],
        result["best"]["epoch"],
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
