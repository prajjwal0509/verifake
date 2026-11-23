"""Frozen CLIP ViT-L/14 feature extractor, trainable probe heads, and ensemble calibration."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.augmentations import CLIP_MEAN, CLIP_STD
from src.utils import (  # noqa: E402 - imported early to configure HF_HOME
    CLASS_NAMES_PATH,
    SEED,
    get_logger,
    load_class_mapping,
    save_class_mapping,
    seed_everything,
)

logger = get_logger(__name__)

CLIP_BACKBONE: str = "vit_large_patch14_clip_224.openai"


class CLIPFeatureExtractor(nn.Module):
    """Frozen CLIP ViT-L/14 visual backbone returning L2-normalized embeddings."""

    def __init__(
        self,
        backbone: str = CLIP_BACKBONE,
        *,
        pretrained: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.img_size = int(img_size)
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size)
            feat = self.encoder(dummy)
            self.embed_dim = int(feat.shape[-1])
        logger.info(
            "Initialized frozen %s (embed_dim=%d, pretrained=%s)",
            backbone,
            self.embed_dim,
            pretrained,
        )

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, *, use_fp16: bool = True) -> torch.Tensor:
        """Compute L2-normalized float32 embeddings (B, embed_dim)."""
        self.encoder.eval()
        dev_type = x.device.type
        ctx = (
            torch.autocast(device_type=dev_type, dtype=torch.float16)
            if (use_fp16 and dev_type in ("mps", "cuda"))
            else nullcontext()
        )
        with ctx:
            feats = self.encoder(x)
        feats_f = feats.float()
        return nn.functional.normalize(feats_f, dim=-1)


class CLIPProbeHead(nn.Module):
    """Lightweight classification head on frozen CLIP embeddings."""

    def __init__(
        self,
        embed_dim: int = 768,
        *,
        head_type: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.head_type = str(head_type).lower()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        if self.head_type == "linear":
            self.net = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            raise ValueError(f"Unsupported head_type={head_type!r} (expected linear/mlp)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def save_clip_head(head: CLIPProbeHead, path: Path, meta: dict[str, Any]) -> None:
    """Persist trained CLIP probe head weights and metadata."""
    if "class_to_idx" not in meta:
        raise ValueError("meta must include class_to_idx")
    mapping = save_class_mapping(meta["class_to_idx"], CLASS_NAMES_PATH)
    payload_meta = {
        "backbone": CLIP_BACKBONE,
        "img_size": 224,
        "normalize_mean": list(CLIP_MEAN),
        "normalize_std": list(CLIP_STD),
        "head_type": head.head_type,
        "embed_dim": head.embed_dim,
        "hidden_dim": head.hidden_dim,
        "dropout": head.dropout,
        "num_classes": head.num_classes,
        "class_to_idx": mapping["class_to_idx"],
        "idx_to_class": mapping["idx_to_class"],
        "class_names": mapping["class_names"],
        **{k: v for k, v in meta.items() if k != "class_to_idx"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "meta": payload_meta}, path)


def load_clip_head(path: Path, device: torch.device) -> tuple[CLIPProbeHead, dict[str, Any]]:
    """Load CLIP probe head checkpoint and verify class mapping consistency."""
    mapping = load_class_mapping(CLASS_NAMES_PATH)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(ckpt["meta"])
    if {k.upper(): int(v) for k, v in meta["class_to_idx"].items()} != mapping["class_to_idx"]:
        raise RuntimeError(
            f"Label mapping mismatch between {path} ({meta['class_to_idx']}) and {CLASS_NAMES_PATH}"
        )
    head = CLIPProbeHead(
        embed_dim=int(meta["embed_dim"]),
        head_type=str(meta["head_type"]),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        dropout=float(meta.get("dropout", 0.3)),
        num_classes=int(meta.get("num_classes", 2)),
    )
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    return head, meta


def compute_fpr_threshold(
    real_scores: np.ndarray, target_fpr: float = 0.05
) -> tuple[float, float]:
    """Choose smallest threshold in [0, 1] achieving empirical FPR <= target_fpr on real_scores."""
    scores = np.sort(np.asarray(real_scores, dtype=np.float64))
    if scores.size == 0:
        return 0.5, 0.0
    uniq = np.unique(scores)
    candidates = [0.0]
    for i in range(len(uniq) - 1):
        candidates.append(0.5 * (float(uniq[i]) + float(uniq[i + 1])))
    candidates.append(float(uniq[-1]) + 1e-6)
    chosen = float(uniq[-1]) + 1e-6
    for t in sorted(set(candidates)):
        if float(np.mean(scores >= t)) <= target_fpr + 1e-12:
            chosen = float(t)
            break
    chosen = min(1.0, max(0.0, chosen))
    return chosen, float(np.mean(scores >= chosen))


def compute_binary_metrics(
    p_fake: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float = 0.5,
    fake_idx: int = 1,
) -> dict[str, Any]:
    """Compute binary metrics, FPR, and confusion matrix for P(FAKE) scores."""
    y_bin = (np.asarray(y_true) == fake_idx).astype(int)
    scores = np.asarray(p_fake, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_bin, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    auroc = float(roc_auc_score(y_bin, scores)) if len(np.unique(y_bin)) > 1 else None
    return {
        "n": int(len(y_bin)),
        "threshold": float(threshold),
        "auroc": auroc,
        "accuracy": float(accuracy_score(y_bin, pred)),
        "precision": float(precision_score(y_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_bin, pred, zero_division=0)),
        "real_fpr": float(fp / max(1, tn + fp)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _score_val_embeddings(
    head: CLIPProbeHead,
    val_feats_clean: torch.Tensor,
    val_labels: torch.Tensor,
    val_feats_pert: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    *,
    fake_idx: int,
    target_fpr: float,
    device: torch.device,
) -> dict[str, Any]:
    head.eval()
    with torch.no_grad():
        logits_c = head(val_feats_clean.to(device)).cpu()
    probs_c = torch.softmax(logits_c.float(), dim=-1)[:, fake_idx].numpy()
    y_np = val_labels.cpu().numpy()
    real_scores = probs_c[y_np != fake_idx]
    t_fpr5, val_fpr = compute_fpr_threshold(real_scores, target_fpr)
    clean_m = compute_binary_metrics(probs_c, y_np, threshold=t_fpr5, fake_idx=fake_idx)
    clean_05 = compute_binary_metrics(probs_c, y_np, threshold=0.5, fake_idx=fake_idx)

    pert_aucs: list[float] = []
    pert_recs: list[float] = []
    if val_feats_pert:
        for feats_p, labels_p in val_feats_pert.values():
            if len(feats_p) == 0:
                continue
            with torch.no_grad():
                logits_p = head(feats_p.to(device)).cpu()
            probs_p = torch.softmax(logits_p.float(), dim=-1)[:, fake_idx].numpy()
            y_p = labels_p.cpu().numpy()
            pm = compute_binary_metrics(probs_p, y_p, threshold=t_fpr5, fake_idx=fake_idx)
            if pm["auroc"] is not None:
                pert_aucs.append(float(pm["auroc"]))
            pert_recs.append(float(pm["recall"]))

    mean_pert_auc = float(np.mean(pert_aucs)) if pert_aucs else float(clean_m["auroc"] or 0.0)
    mean_pert_rec = float(np.mean(pert_recs)) if pert_recs else float(clean_m["recall"])
    clean_auc = float(clean_m["auroc"] or 0.0)
    clean_rec = float(clean_m["recall"])
    gen_score = 0.5 * (0.5 * (clean_auc + clean_rec) + 0.5 * (mean_pert_auc + mean_pert_rec))
    return {
        "gen_score": float(gen_score),
        "threshold_val_fpr5": float(t_fpr5),
        "val_real_fpr": float(val_fpr),
        "clean_auroc": clean_auc,
        "clean_recall_fpr5": clean_rec,
        "clean_recall_050": float(clean_05["recall"]),
        "clean_fpr_050": float(clean_05["real_fpr"]),
        "pert_mean_auroc": mean_pert_auc,
        "pert_mean_recall_fpr5": mean_pert_rec,
    }


def train_clip_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats_clean: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    val_feats_pert: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]] = None,
    head_type: str = "linear",
    hidden_dim: int = 256,
    dropout: float = 0.3,
    epochs: int = 40,
    min_epochs: int = 8,
    patience: int = 8,
    lr: Optional[float] = None,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    fake_idx: int = 1,
    target_fpr: float = 0.05,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> tuple[CLIPProbeHead, dict[str, Any]]:
    """Train linear or MLP head on cached CLIP embeddings with OOD+perturbed early stopping."""
    seed_everything(seed)
    dev = device or torch.device("cpu")
    embed_dim = int(train_feats.shape[-1])
    head = CLIPProbeHead(
        embed_dim=embed_dim,
        head_type=head_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=2,
    ).to(dev)

    opt_lr = lr if lr is not None else (1e-2 if head_type == "linear" else 1e-3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=opt_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    eff_batch = min(batch_size, max(32, len(train_feats) // 8))
    ds = TensorDataset(train_feats.float(), train_labels.long())
    loader = DataLoader(ds, batch_size=eff_batch, shuffle=True)

    best_score = -1.0
    best_state = copy.deepcopy(head.state_dict())
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss, seen = 0.0, 0
        for bx, by in loader:
            bx = bx.to(dev)
            by = by.to(dev)
            logits = head(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * bx.size(0)
            seen += bx.size(0)
        scheduler.step()

        metrics = _score_val_embeddings(
            head,
            val_feats_clean,
            val_labels,
            val_feats_pert,
            fake_idx=fake_idx,
            target_fpr=target_fpr,
            device=dev,
        )
        score = metrics["gen_score"]
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
            best_metrics = {**metrics, "epoch": epoch, "train_loss": running_loss / max(1, seen)}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch >= min_epochs and bad_epochs >= patience:
            break

    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_epoch, **best_metrics}


def fit_ensemble_weights(
    val_clip_logits: torch.Tensor,
    val_conv_logits: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    val_pert_logits: Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    fake_idx: int = 1,
    target_fpr: float = 0.05,
) -> dict[str, Any]:
    """Grid-search w in [0, 1] for w * clip + (1 - w) * convnext strictly on validation sets."""
    y_clean = val_labels.cpu().numpy()
    best: dict[str, Any] = {"gen_score": -1.0}

    for step in range(21):
        w = round(step * 0.05, 2)
        blend_c = w * val_clip_logits.float().cpu() + (1.0 - w) * val_conv_logits.float().cpu()
        p_c = torch.softmax(blend_c, dim=-1)[:, fake_idx].numpy()
        t_fpr5, val_fpr = compute_fpr_threshold(p_c[y_clean != fake_idx], target_fpr)
        mc = compute_binary_metrics(p_c, y_clean, threshold=t_fpr5, fake_idx=fake_idx)

        pert_aucs: list[float] = []
        pert_recs: list[float] = []
        if val_pert_logits:
            for c_l, v_l, y_p_t in val_pert_logits:
                blend_p = w * c_l.float().cpu() + (1.0 - w) * v_l.float().cpu()
                p_p = torch.softmax(blend_p, dim=-1)[:, fake_idx].numpy()
                mp = compute_binary_metrics(
                    p_p, y_p_t.cpu().numpy(), threshold=t_fpr5, fake_idx=fake_idx
                )
                if mp["auroc"] is not None:
                    pert_aucs.append(float(mp["auroc"]))
                pert_recs.append(float(mp["recall"]))

        clean_auc = float(mc["auroc"] or 0.0)
        clean_rec = float(mc["recall"])
        mean_pert_auc = float(np.mean(pert_aucs)) if pert_aucs else clean_auc
        mean_pert_rec = float(np.mean(pert_recs)) if pert_recs else clean_rec
        gen_score = 0.5 * (0.5 * (clean_auc + clean_rec) + 0.5 * (mean_pert_auc + mean_pert_rec))

        if gen_score > best["gen_score"]:
            best = {
                "clip_weight": float(w),
                "convnext_weight": float(round(1.0 - w, 2)),
                "threshold_val_fpr5": float(t_fpr5),
                "val_real_fpr": float(val_fpr),
                "clean_auroc": clean_auc,
                "clean_recall_fpr5": clean_rec,
                "pert_mean_auroc": mean_pert_auc,
                "pert_mean_recall_fpr5": mean_pert_rec,
                "gen_score": float(gen_score),
            }
    return best
