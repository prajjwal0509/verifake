"""Step 3 training, 6-fold LOGO comparison (CLIP head vs. ConvNeXt vs. ensemble), and read-only held-out evaluation."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.augmentations import (  # noqa: E402
    apply_eval_perturbation,
    build_clip_transforms,
    build_convnext_transforms,
)
from src.clip_detector import (  # noqa: E402
    CLIPFeatureExtractor,
    compute_binary_metrics,
    compute_fpr_threshold,
    fit_ensemble_weights,
    load_clip_head,
    save_clip_head,
    train_clip_head,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.face_crop import crop_face_or_full, gate_and_crop_face  # noqa: E402
from src.image_io import register_image_plugins, to_rgb_safe  # noqa: E402
from src.inference import DeepfakeDetector  # noqa: E402
from src.model import BACKBONES, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    CHECKPOINT_PATH,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    get_device,
    get_logger,
    load_class_mapping,
    load_json,
    save_json,
    seed_everything,
)

logger = get_logger("train_step3")

DEFAULT_MANIFEST: Path = DATA_DIR / "prepared_manifest.csv"
DEFAULT_CACHE_DIR: Path = DATA_DIR / "cache" / "clip_vit_l14"
DEFAULT_OUT_DIR: Path = REPORTS_DIR / "step3"
LOGO_FAMILIES: tuple[str, ...] = ("dalle3", "flux", "sd21", "sd3", "sdxl", "stylegan")
STEP3_EVAL_PERTS: tuple[str, ...] = ("original", "jpeg_q50", "webp_q80", "downscale_512", "downscale_256", "downscale_512_jpeg_q75")


def _load_rgb(path_str: str) -> Image.Image:
    p = Path(path_str)
    abs_p = p if p.is_absolute() else (PROJECT_ROOT / p)
    with Image.open(abs_p) as im:
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
        return to_rgb_safe(ImageOps.exif_transpose(im))


def sample_balanced_sources(
    df: pd.DataFrame,
    target_total: int,
    *,
    seed: int = SEED,
    allow_oversample: bool = True,
) -> pd.DataFrame:
    """Sample 50/50 REAL/FAKE balanced across generator families (FAKE) and real sources (REAL)."""
    if len(df) == 0 or target_total <= 0:
        return df.iloc[0:0].copy()

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]
    half = target_total // 2

    def _draw_by_group(sub: pd.DataFrame, group_col: str, total_quota: int, base_seed: int) -> pd.DataFrame:
        groups = sorted([(str(k), grp) for k, grp in sub.groupby(group_col)], key=lambda kv: kv[0])
        if not groups:
            return sub.iloc[0:0]
        parts = []
        per_group = max(1, total_quota // len(groups))
        rem = total_quota - per_group * len(groups)
        for idx, (_, grp) in enumerate(groups):
            q = per_group + (1 if idx < rem else 0)
            replace = allow_oversample and (q > len(grp))
            take = q if replace else min(q, len(grp))
            parts.append(grp.sample(n=take, replace=replace, random_state=base_seed + idx))
        return pd.concat(parts, ignore_index=True)

    r_pick = _draw_by_group(real_df, "generator", half, seed)
    f_pick = _draw_by_group(fake_df, "generator_family", target_total - half, seed + 100)
    combined = pd.concat([r_pick, f_pick], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed + 200).reset_index(drop=True)


def build_logo_fold_dataframes(
    non_test_df: pd.DataFrame,
    held_out_family: str,
    *,
    train_cap: int = 3600,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build strictly leak-free (fold_train, fold_val, held_out_fake, val_real_eval) for one LOGO fold."""
    if (non_test_df["split"] == "held_out_test").any():
        raise RuntimeError("held_out_test rows passed to LOGO fold builder!")

    val_real_all = non_test_df[non_test_df["split_role"] == "val_real"].sort_values("sha256").reset_index(drop=True)
    half = len(val_real_all) // 2
    val_real_cal = val_real_all.iloc[:half].reset_index(drop=True)
    val_real_eval = val_real_all.iloc[half:].reset_index(drop=True)

    if held_out_family == "flux":
        # flux is already completely outside train; train on all train rows, validate on seen_val + val_real_cal
        fold_train_pool = non_test_df[non_test_df["split"] == "train"]
        fold_val_fake = non_test_df[non_test_df["split_role"] == "seen_val_generator"]
        held_fake = non_test_df[non_test_df["generator_family"] == "flux"].reset_index(drop=True)
    else:
        fold_train_pool = non_test_df[
            (non_test_df["split"] == "train") & (non_test_df["generator_family"] != held_out_family)
        ]
        # Prefer unseen flux for early-stopping when holding out another generator
        fold_val_fake = non_test_df[
            (non_test_df["split"] == "val")
            & (non_test_df["label"] == 1)
            & (non_test_df["generator_family"] != held_out_family)
        ]
        held_fake = non_test_df[
            (non_test_df["split"] == "val") & (non_test_df["generator_family"] == held_out_family)
        ]
        if len(held_fake) < 120:
            # Seen test splits are never used here; combine non-train or train-side held_out only for evaluation
            held_fake = non_test_df[
                (non_test_df["generator_family"] == held_out_family) & (non_test_df["split"] != "held_out_test")
            ].reset_index(drop=True)

    assert not (fold_train_pool["generator_family"] == held_out_family).any()
    assert not (fold_train_pool["split"] == "held_out_test").any()

    fold_train = sample_balanced_sources(fold_train_pool, train_cap, seed=seed, allow_oversample=True)
    fold_val = pd.concat([fold_val_fake, val_real_cal], ignore_index=True).sample(
        frac=1.0, random_state=seed + 1
    ).reset_index(drop=True)
    return fold_train, fold_val, held_fake, val_real_eval


_FACE_CROP_CACHE: dict[str, tuple[tuple[int, int, int, int], bool]] = {}


def _cached_crop_face_or_full(path_str: str, rgb: Image.Image) -> Image.Image:
    cached = _FACE_CROP_CACHE.get(path_str)
    if cached is not None:
        bbox, upsampled = cached
        c = rgb.crop(bbox)
        if upsampled and min(c.size) < 256:
            scale_up = 256 / float(min(c.size))
            nw = max(256, int(round(c.width * scale_up)))
            nh = max(256, int(round(c.height * scale_up)))
            c = c.resize((nw, nh), Image.Resampling.LANCZOS)
        return c
    cropped, meta = crop_face_or_full(rgb)
    bbox_tuple = tuple(int(v) for v in meta["bbox"])
    _FACE_CROP_CACHE[path_str] = (bbox_tuple, bool(meta.get("upsampled", False)))
    return cropped


class Step3TrainDataset(Dataset):
    """Face/scene mixed dataset with RobustForensicTransform."""

    def __init__(self, df: pd.DataFrame, transform: Callable) -> None:
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path_str = self.paths[idx]
        rgb = _load_rgb(path_str)
        cropped = _cached_crop_face_or_full(path_str, rgb)
        view = cropped if np.random.random() < 0.70 else rgb
        return self.transform(view), self.labels[idx]


class TwoViewEvalDataset(Dataset):
    """Returns [face_crop, full_frame_padded] tensors + face_detected mask for clean or perturbed DF."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        *,
        perturbation: str = "original",
        require_face: bool = True,
    ) -> None:
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.perturbation = perturbation
        self.require_face = require_face

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rgb = _load_rgb(self.paths[idx])
        v_rgb = apply_eval_perturbation(rgb, self.perturbation)
        if self.require_face:
            cropped, meta = gate_and_crop_face(v_rgb)
            ok = bool(meta["face_detected"])
            c_img = cropped if cropped is not None else v_rgb
        else:
            c_img, meta = crop_face_or_full(v_rgb)
            ok = bool(meta["face_detected"])
        t = torch.stack([self.transform(c_img), self.transform(v_rgb)], dim=0)
        return t, int(self.labels[idx]), bool(ok)


@torch.no_grad()
def infer_convnext_two_view_logits(
    model: nn.Module,
    df: pd.DataFrame,
    transform: Callable,
    *,
    perturbation: str = "original",
    require_face: bool = True,
    batch_size: int = 32,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean_logits (N,2), crop_logits (N,2), full_logits (N,2), face_ok (N,))."""
    model.eval()
    ds = TwoViewEvalDataset(df, transform, perturbation=perturbation, require_face=require_face)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    crops, fulls, oks = [], [], []
    for views, _, ok in loader:
        b, v, c, h, w = views.shape
        flat = views.view(b * v, c, h, w).to(device, non_blocking=True)
        out = model(flat).float().cpu().view(b, v, -1)
        crops.append(out[:, 0])
        fulls.append(out[:, 1])
        oks.append(ok.bool())
    crop_l = torch.cat(crops, dim=0)
    full_l = torch.cat(fulls, dim=0)
    ok_t = torch.cat(oks, dim=0)
    return 0.5 * (crop_l + full_l), crop_l, full_l, ok_t


def _precompute_two_view_tensors(
    df: pd.DataFrame,
    transform: Callable,
    *,
    perturbation: str = "original",
) -> torch.Tensor:
    views = []
    for row in df.itertuples(index=False):
        path_str = str(row.path)
        rgb = _load_rgb(path_str)
        if perturbation == "original":
            c_img = _cached_crop_face_or_full(path_str, rgb)
            v_rgb = rgb
        else:
            v_rgb = apply_eval_perturbation(rgb, perturbation)
            c_img, _ = crop_face_or_full(v_rgb)
        views.append(torch.stack([transform(c_img), transform(v_rgb)], dim=0).half())
    return torch.stack(views, dim=0)


@torch.no_grad()
def _score_precomputed_views(
    model: nn.Module,
    views_fp16: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    crops, fulls = [], []
    n = views_fp16.shape[0]
    for i in range(0, n, batch_size):
        sub = views_fp16[i : i + batch_size]
        b, v, c, h, w = sub.shape
        flat = sub.view(b * v, c, h, w).to(device=device, dtype=torch.float32, non_blocking=True)
        out = model(flat).float().cpu().view(b, v, -1)
        crops.append(out[:, 0])
        fulls.append(out[:, 1])
    return 0.5 * (torch.cat(crops, dim=0) + torch.cat(fulls, dim=0))


def _warmup_cosine(total_steps: int, warmup_steps: int, min_ratio: float = 0.05):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def train_convnext_branch(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    save_path: Path,
    class_to_idx: dict[str, int],
    device: torch.device,
    val_pert_slice: Optional[pd.DataFrame] = None,
    val_es_cap: int = 600,
    seed: int = SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train ConvNeXt-Tiny from fresh ImageNet weights with RobustForensicTransform and early stopping."""
    seed_everything(seed)
    train_tf = build_convnext_transforms(train=True, img_size=224)
    eval_tf = build_convnext_transforms(train=False, img_size=224)
    train_loader = DataLoader(
        Step3TrainDataset(train_df, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = build_model("convnext_tiny", pretrained=True, num_classes=2, drop_rate=0.2).to(device)
    total_steps = max(1, len(train_loader) * epochs)
    warmup = max(1, min(len(train_loader), total_steps // 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine(total_steps, warmup))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    fake_idx = class_to_idx["FAKE"]
    if val_es_cap > 0 and len(val_df) > val_es_cap:
        half_cap = val_es_cap // 2
        v_fake = val_df[val_df["label"] == fake_idx]
        v_real = val_df[val_df["label"] != fake_idx]
        val_es_df = pd.concat(
            [
                v_fake.sample(n=min(half_cap, len(v_fake)), random_state=seed),
                v_real.sample(n=min(val_es_cap - half_cap, len(v_real)), random_state=seed + 1),
            ],
            ignore_index=True,
        )
    else:
        val_es_df = val_df

    pert_df = val_pert_slice if val_pert_slice is not None else val_es_df
    y_val_np = val_es_df["label"].to_numpy(dtype=int)
    y_pert_np = pert_df["label"].to_numpy(dtype=int)

    val_clean_views = _precompute_two_view_tensors(val_es_df, eval_tf, perturbation="original")
    val_pert_views = {
        p_name: _precompute_two_view_tensors(pert_df, eval_tf, perturbation=p_name)
        for p_name in ("jpeg_q50", "downscale_256", "downscale_512_jpeg_q75")
    }

    best_score = -1.0
    best_epoch = 0
    best_meta: dict[str, Any] = {}
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).float()
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        eval_bs = max(64, batch_size * 2)
        mean_l_c = _score_precomputed_views(model, val_clean_views, batch_size=eval_bs, device=device)
        p_c = torch.softmax(mean_l_c, dim=-1)[:, fake_idx].numpy()
        t_fpr5, _ = compute_fpr_threshold(p_c[y_val_np != fake_idx], 0.05)
        mc = compute_binary_metrics(p_c, y_val_np, threshold=t_fpr5, fake_idx=fake_idx)

        pert_aucs, pert_recs = [], []
        for p_name in ("jpeg_q50", "downscale_256", "downscale_512_jpeg_q75"):
            mean_l_p = _score_precomputed_views(model, val_pert_views[p_name], batch_size=eval_bs, device=device)
            p_p = torch.softmax(mean_l_p, dim=-1)[:, fake_idx].numpy()
            mp = compute_binary_metrics(p_p, y_pert_np, threshold=t_fpr5, fake_idx=fake_idx)
            if mp["auroc"] is not None:
                pert_aucs.append(float(mp["auroc"]))
            pert_recs.append(float(mp["recall"]))

        clean_auc = float(mc["auroc"] or 0.0)
        clean_rec = float(mc["recall"])
        mean_p_auc = float(np.mean(pert_aucs)) if pert_aucs else clean_auc
        mean_p_rec = float(np.mean(pert_recs)) if pert_recs else clean_rec
        gen_score = 0.5 * (0.5 * (clean_auc + clean_rec) + 0.5 * (mean_p_auc + mean_p_rec))
        logger.info(
            "  ConvNeXt ep %d/%d | loss %.4f | val clean AUC=%.4f Rec@5%%=%.3f | pert AUC=%.4f Rec@5%%=%.3f | %.0fs",
            ep,
            epochs,
            running / max(1, seen),
            clean_auc,
            clean_rec,
            mean_p_auc,
            mean_p_rec,
            time.time() - t0,
        )

        if gen_score > best_score + 1e-5:
            best_score = gen_score
            best_epoch = ep
            bad_epochs = 0
            best_meta = {
                "epoch": ep,
                "gen_score": float(gen_score),
                "threshold_val_fpr5": float(t_fpr5),
                "clean_auroc": clean_auc,
                "clean_recall_fpr5": clean_rec,
                "pert_mean_auroc": mean_p_auc,
                "pert_mean_recall_fpr5": mean_p_rec,
            }
            save_checkpoint(
                model,
                save_path,
                {
                    "backbone": "convnext_tiny",
                    "timm_name": BACKBONES["convnext_tiny"],
                    "img_size": 224,
                    "normalize_mean": list(IMAGENET_MEAN),
                    "normalize_std": list(IMAGENET_STD),
                    "class_to_idx": class_to_idx,
                    "amp": False,
                    "temperature": 1.0,
                    "epoch": ep,
                    "val": best_meta,
                },
            )
        else:
            bad_epochs += 1

        if ep >= min_epochs and bad_epochs >= patience:
            logger.info("Early stopping ConvNeXt at epoch %d (best epoch %d)", ep, best_epoch)
            break

    del val_clean_views, val_pert_views
    best_model, _ = load_checkpoint(save_path, device)
    best_model.eval()
    return best_model, best_meta


def _load_clip_caches(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tr = torch.load(cache_dir / "train_aug.pt", map_location="cpu", weights_only=False)
    va = torch.load(cache_dir / "val_eval.pt", map_location="cpu", weights_only=False)
    te_path = cache_dir / "held_out_test_eval.pt"
    te = torch.load(te_path, map_location="cpu", weights_only=False) if te_path.exists() else {}
    return tr, va, te


def _lookup_clip_rows(
    df: pd.DataFrame,
    sha_to_tr: dict[str, int],
    sha_to_va: dict[str, int],
    tr_cache: dict[str, Any],
    va_cache: dict[str, Any],
    *,
    augmented: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble cached CLIP embeddings for df rows strictly from train/val caches."""
    feats: list[torch.Tensor] = []
    labels: list[int] = []
    for sha, lbl in zip(df["sha256"].astype(str), df["label"].astype(int)):
        if sha in sha_to_tr:
            i = sha_to_tr[sha]
            if augmented:
                # Expand all K augmented passes for each row
                f_passes = tr_cache["aug_feats"][i]  # (K, D)
                for p in range(f_passes.shape[0]):
                    feats.append(f_passes[p].float())
                    labels.append(lbl)
            else:
                v = 0.5 * (tr_cache["clean_crop_feats"][i].float() + tr_cache["clean_full_feats"][i].float())
                feats.append(v)
                labels.append(lbl)
        elif sha in sha_to_va:
            i = sha_to_va[sha]
            v = 0.5 * (va_cache["crop_feats"][i, 0].float() + va_cache["full_feats"][i, 0].float())
            feats.append(v)
            labels.append(lbl)
        else:
            raise KeyError(f"sha256 missing from train/val CLIP cache: {sha}")
    return torch.stack(feats, dim=0), torch.tensor(labels, dtype=torch.long)


def _lookup_val_pert_dict(
    df: pd.DataFrame,
    sha_to_va: dict[str, int],
    va_cache: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    perts = va_cache["perturbations"]
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    va_idxs = [sha_to_va[s] for s in df["sha256"].astype(str) if s in sha_to_va]
    if not va_idxs:
        return out
    y_t = torch.tensor(
        [int(l) for s, l in zip(df["sha256"].astype(str), df["label"].astype(int)) if s in sha_to_va],
        dtype=torch.long,
    )
    for pj, pname in enumerate(perts):
        if pname == "original":
            continue
        ok_mask = va_cache["face_detected"][va_idxs, pj]
        if ok_mask.sum() < 10:
            continue
        sub_idxs = torch.tensor(va_idxs)[ok_mask]
        v = 0.5 * (
            va_cache["crop_feats"][sub_idxs, pj].float()
            + va_cache["full_feats"][sub_idxs, pj].float()
        )
        out[pname] = (v, y_t[ok_mask])
    return out


def _eval_cached_clip_eval_set(
    head: nn.Module,
    shas: list[str],
    labels: np.ndarray,
    sha_to_cache: dict[str, tuple[dict[str, Any], int]],
    *,
    pert_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits (N,2), face_ok (N,)) for clean/perturbed split using CLIP cache."""
    feats = []
    oks = []
    for sha in shas:
        cache, idx = sha_to_cache[sha]
        if "perturbations" in cache:
            pj = cache["perturbations"].index(pert_name)
            ok = bool(cache["face_detected"][idx, pj])
            if ok:
                v = 0.5 * (cache["crop_feats"][idx, pj].float() + cache["full_feats"][idx, pj].float())
            else:
                v = 0.5 * (cache["crop_feats"][idx, 0].float() + cache["full_feats"][idx, 0].float())
            feats.append(v)
            oks.append(ok)
        else:
            v = 0.5 * (cache["clean_crop_feats"][idx].float() + cache["clean_full_feats"][idx].float())
            feats.append(v)
            oks.append(True)
    x = torch.stack(feats, dim=0).to(device)
    with torch.no_grad():
        logits = head(x).cpu()
    return logits, torch.tensor(oks, dtype=torch.bool)


@torch.no_grad()
def _extract_clip_pert_feats(
    extractor: CLIPFeatureExtractor,
    df_sub: pd.DataFrame,
    sha_to_va: dict[str, int],
    va_cache: dict[str, Any],
    *,
    pert_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get two-view averaged CLIP embeddings for df_sub under pert_name, hitting va_cache when available."""
    n = len(df_sub)
    feats = torch.zeros((n, extractor.embed_dim), dtype=torch.float32)
    oks = torch.zeros(n, dtype=torch.bool)
    miss_rows: list[tuple[int, str]] = []
    pj = va_cache["perturbations"].index(pert_name)

    for i, sha in enumerate(df_sub["sha256"].astype(str)):
        if sha in sha_to_va:
            vi = sha_to_va[sha]
            ok = bool(va_cache["face_detected"][vi, pj])
            oks[i] = ok
            slot_j = pj if ok else 0
            feats[i] = 0.5 * (
                va_cache["crop_feats"][vi, slot_j].float()
                + va_cache["full_feats"][vi, slot_j].float()
            )
        else:
            miss_rows.append((i, str(df_sub["path"].iloc[i])))

    if miss_rows:
        clip_tf = build_clip_transforms(train=False, img_size=224)
        miss_df = pd.DataFrame(
            {"path": [p for _, p in miss_rows], "label": [0] * len(miss_rows)}
        )
        ds = TwoViewEvalDataset(miss_df, clip_tf, perturbation=pert_name, require_face=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        encoded = []
        ok_enc = []
        use_fp16 = device.type in ("mps", "cuda")
        for views, _, ok in loader:
            b, v, c, h, w = views.shape
            flat = views.view(b * v, c, h, w).to(device, non_blocking=True)
            f = extractor.extract_features(flat, use_fp16=use_fp16).cpu().view(b, v, -1)
            encoded.append(0.5 * (f[:, 0] + f[:, 1]))
            ok_enc.append(ok.bool())
        enc_t = torch.cat(encoded, dim=0)
        ok_t = torch.cat(ok_enc, dim=0)
        for k, (row_i, _) in enumerate(miss_rows):
            feats[row_i] = enc_t[k]
            oks[row_i] = bool(ok_t[k])

    return feats, oks


def run_logo_benchmark(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    non_test_df = face_df[face_df["split"] != "held_out_test"].reset_index(drop=True)

    tr_cache, va_cache, _ = _load_clip_caches(args.cache_dir)
    sha_to_tr = {s: i for i, s in enumerate(tr_cache["sha256"])}
    sha_to_va = {s: i for i, s in enumerate(va_cache["sha256"])}
    sha_lookup: dict[str, tuple[dict[str, Any], int]] = {
        **{s: (tr_cache, i) for i, s in enumerate(tr_cache["sha256"])},
        **{s: (va_cache, i) for i, s in enumerate(va_cache["sha256"])},
    }

    device = get_device() if args.device == "auto" else torch.device(args.device)
    clip_extractor = CLIPFeatureExtractor(pretrained=True, img_size=224).to(device)
    class_to_idx = load_class_mapping()["class_to_idx"]
    fake_idx = class_to_idx["FAKE"]
    eval_tf = build_convnext_transforms(train=False, img_size=224)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logo_folds_dir = args.out_dir / "logo_folds"
    logo_folds_dir.mkdir(parents=True, exist_ok=True)
    tmp_conv_ckpt = MODELS_DIR / "step3_logo_convnext_tmp.pt"
    fold_records: list[dict[str, Any]] = []

    for f_idx, held_out_fam in enumerate(args.logo_families, start=1):
        fold_file = logo_folds_dir / f"{held_out_fam}.json"
        if fold_file.exists() and not getattr(args, "no_resume", False):
            cached_fold = load_json(fold_file)
            if isinstance(cached_fold, list) and len(cached_fold) == 4:
                logger.info(
                    "=== LOGO Fold %d/%d '%s': resumed cached fold -> %s ===",
                    f_idx,
                    len(args.logo_families),
                    held_out_fam,
                    fold_file,
                )
                fold_records.extend(cached_fold)
                continue
        logger.info("=== LOGO Fold %d/%d: holding out '%s' ===", f_idx, len(args.logo_families), held_out_fam)
        fold_train, fold_val, held_fake, val_real_eval = build_logo_fold_dataframes(
            non_test_df, held_out_fam, train_cap=args.logo_train_cap, seed=args.seed + f_idx
        )
        fold_eval_df = pd.concat([held_fake, val_real_eval], ignore_index=True)
        val_pert_df = pd.concat(
            [
                fold_val[fold_val["label"] == 1].sample(n=min(120, int((fold_val["label"] == 1).sum())), random_state=args.seed),
                fold_val[fold_val["label"] == 0].sample(n=min(120, int((fold_val["label"] == 0).sum())), random_state=args.seed),
            ],
            ignore_index=True,
        )

        # 1. CLIP features for fold
        x_tr, y_tr = _lookup_clip_rows(fold_train, sha_to_tr, sha_to_va, tr_cache, va_cache, augmented=True)
        x_va, y_va = _lookup_clip_rows(fold_val, sha_to_tr, sha_to_va, tr_cache, va_cache, augmented=False)
        va_pert_dict = _lookup_val_pert_dict(fold_val, sha_to_va, va_cache)

        clip_lin, _ = train_clip_head(
            x_tr, y_tr, x_va, y_va, val_feats_pert=va_pert_dict, head_type="linear",
            epochs=40, min_epochs=8, patience=8, device=device, seed=args.seed + f_idx,
        )
        clip_mlp, _ = train_clip_head(
            x_tr, y_tr, x_va, y_va, val_feats_pert=va_pert_dict, head_type="mlp",
            epochs=40, min_epochs=8, patience=8, device=device, seed=args.seed + f_idx,
        )

        # 2. ConvNeXt-Tiny for fold (fresh ImageNet weights, > 3 epochs)
        if tmp_conv_ckpt.exists():
            tmp_conv_ckpt.unlink()
        conv_model, _ = train_convnext_branch(
            fold_train,
            fold_val,
            epochs=args.logo_epochs,
            min_epochs=args.logo_min_epochs,
            patience=args.logo_patience,
            batch_size=args.convnext_batch_size,
            lr=args.convnext_lr,
            weight_decay=0.05,
            save_path=tmp_conv_ckpt,
            class_to_idx=class_to_idx,
            device=device,
            val_pert_slice=val_pert_df,
            seed=args.seed + f_idx,
        )

        # Validation logits on fold_val for leak-free threshold & ensemble calibration
        fold_val_shas = fold_val["sha256"].astype(str).tolist()
        y_fv = fold_val["label"].to_numpy(dtype=int)
        val_cl_lin, _ = _eval_cached_clip_eval_set(clip_lin, fold_val_shas, y_fv, sha_lookup, pert_name="original", device=device)
        val_cl_mlp, _ = _eval_cached_clip_eval_set(clip_mlp, fold_val_shas, y_fv, sha_lookup, pert_name="original", device=device)
        val_cv_l, _, _, _ = infer_convnext_two_view_logits(
            conv_model, fold_val, eval_tf, perturbation="original", require_face=False, batch_size=args.convnext_batch_size * 2, device=device
        )

        # Held-out clean evaluation
        eval_shas = fold_eval_df["sha256"].astype(str).tolist()
        y_ev = fold_eval_df["label"].to_numpy(dtype=int)
        ev_cl_lin, _ = _eval_cached_clip_eval_set(clip_lin, eval_shas, y_ev, sha_lookup, pert_name="original", device=device)
        ev_cl_mlp, _ = _eval_cached_clip_eval_set(clip_mlp, eval_shas, y_ev, sha_lookup, pert_name="original", device=device)
        ev_cv_l, _, _, _ = infer_convnext_two_view_logits(
            conv_model, fold_eval_df, eval_tf, perturbation="original", require_face=False, batch_size=args.convnext_batch_size * 2, device=device
        )

        ens_fit_lin = fit_ensemble_weights(val_cl_lin, val_cv_l, torch.tensor(y_fv), fake_idx=fake_idx)
        ens_fit_mlp = fit_ensemble_weights(val_cl_mlp, val_cv_l, torch.tensor(y_fv), fake_idx=fake_idx)
        use_mlp_in_ens = ens_fit_mlp["gen_score"] > ens_fit_lin["gen_score"]
        ens_fit = ens_fit_mlp if use_mlp_in_ens else ens_fit_lin
        ev_clip_best = ev_cl_mlp if use_mlp_in_ens else ev_cl_lin
        val_clip_best = val_cl_mlp if use_mlp_in_ens else val_cl_lin
        w_clip = ens_fit["clip_weight"]

        # Perturbed LOGO slice (computed ONCE per fold across all 4 branches)
        pert_sub = pd.concat(
            [
                held_fake.sample(n=min(150, len(held_fake)), random_state=args.seed + 5),
                val_real_eval.sample(n=min(150, len(val_real_eval)), random_state=args.seed + 6),
            ],
            ignore_index=True,
        )
        y_pv = pert_sub["label"].to_numpy(dtype=int)
        pert_branch_logits: dict[str, list[torch.Tensor]] = {
            "clip_linear": [],
            "clip_mlp": [],
            "convnext_tiny": [],
            "ensemble": [],
        }
        for p_mode in ("jpeg_q50", "downscale_256"):
            cf_p, _ = _extract_clip_pert_feats(
                clip_extractor,
                pert_sub,
                sha_to_va,
                va_cache,
                pert_name=p_mode,
                batch_size=args.convnext_batch_size * 2,
                device=device,
            )
            with torch.no_grad():
                pl_lin = clip_lin(cf_p.to(device)).cpu()
                pl_mlp = clip_mlp(cf_p.to(device)).cpu()
            pl_cv, _, _, _ = infer_convnext_two_view_logits(
                conv_model,
                pert_sub,
                eval_tf,
                perturbation=p_mode,
                require_face=False,
                batch_size=args.convnext_batch_size * 2,
                device=device,
            )
            pl_clip_b = pl_mlp if use_mlp_in_ens else pl_lin
            pl_ens = w_clip * pl_clip_b + (1.0 - w_clip) * pl_cv
            pert_branch_logits["clip_linear"].append(pl_lin)
            pert_branch_logits["clip_mlp"].append(pl_mlp)
            pert_branch_logits["convnext_tiny"].append(pl_cv)
            pert_branch_logits["ensemble"].append(pl_ens)

        branches = {
            "clip_linear": (val_cl_lin, ev_cl_lin),
            "clip_mlp": (val_cl_mlp, ev_cl_mlp),
            "convnext_tiny": (val_cv_l, ev_cv_l),
            "ensemble": (
                w_clip * val_clip_best + (1.0 - w_clip) * val_cv_l,
                w_clip * ev_clip_best + (1.0 - w_clip) * ev_cv_l,
            ),
        }

        this_fold_records: list[dict[str, Any]] = []
        for bname, (v_l, e_l) in branches.items():
            vp = torch.softmax(v_l.float(), dim=-1)[:, fake_idx].numpy()
            t_fpr5, _ = compute_fpr_threshold(vp[y_fv != fake_idx], 0.05)
            ep = torch.softmax(e_l.float(), dim=-1)[:, fake_idx].numpy()
            m_fpr5 = compute_binary_metrics(ep, y_ev, threshold=t_fpr5, fake_idx=fake_idx)
            m_050 = compute_binary_metrics(ep, y_ev, threshold=0.5, fake_idx=fake_idx)

            p_aucs, p_recs = [], []
            for pl_b in pert_branch_logits[bname]:
                pp = torch.softmax(pl_b.float(), dim=-1)[:, fake_idx].numpy()
                mp = compute_binary_metrics(pp, y_pv, threshold=t_fpr5, fake_idx=fake_idx)
                if mp["auroc"] is not None:
                    p_aucs.append(float(mp["auroc"]))
                p_recs.append(float(mp["recall"]))

            this_fold_records.append(
                {
                    "fold": held_out_fam,
                    "branch": bname,
                    "held_out_n": int(len(held_fake)),
                    "val_real_eval_n": int(len(val_real_eval)),
                    "threshold_val_fpr5": float(t_fpr5),
                    "auroc": float(m_fpr5["auroc"] or 0.0),
                    "recall_fpr5": float(m_fpr5["recall"]),
                    "real_fpr_fpr5": float(m_fpr5["real_fpr"]),
                    "recall_050": float(m_050["recall"]),
                    "real_fpr_050": float(m_050["real_fpr"]),
                    "pert_mean_auroc": float(np.mean(p_aucs)) if p_aucs else 0.0,
                    "pert_mean_recall_fpr5": float(np.mean(p_recs)) if p_recs else 0.0,
                    "clip_weight": float(w_clip if bname == "ensemble" else (1.0 if "clip" in bname else 0.0)),
                }
            )

        save_json(this_fold_records, fold_file)
        fold_records.extend(this_fold_records)
        fold_summary = " | ".join(
            f"{r['branch']}: AUC={r['auroc']:.4f} Rec@5%={100 * r['recall_fpr5']:.1f}% PertRec={100 * r['pert_mean_recall_fpr5']:.1f}%"
            for r in this_fold_records
        )
        logger.info("Fold '%s' summary: %s", held_out_fam, fold_summary)

        del conv_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if tmp_conv_ckpt.exists():
        tmp_conv_ckpt.unlink()

    fdf = pd.DataFrame(fold_records)
    csv_path = args.out_dir / "logo_comparison.csv"
    fdf.to_csv(csv_path, index=False)

    summary_rows = []
    agg_summary: dict[str, Any] = {}
    for bname, grp in fdf.groupby("branch"):
        rec = {
            "branch": bname,
            "mean_auroc": float(grp["auroc"].mean()),
            "mean_recall_fpr5": float(grp["recall_fpr5"].mean()),
            "mean_real_fpr_fpr5": float(grp["real_fpr_fpr5"].mean()),
            "mean_recall_050": float(grp["recall_050"].mean()),
            "mean_real_fpr_050": float(grp["real_fpr_050"].mean()),
            "mean_pert_auroc": float(grp["pert_mean_auroc"].mean()),
            "mean_pert_recall_fpr5": float(grp["pert_mean_recall_fpr5"].mean()),
        }
        rec["logo_selection_score"] = 0.5 * (
            0.5 * (rec["mean_auroc"] + rec["mean_recall_fpr5"])
            + 0.5 * (rec["mean_pert_auroc"] + rec["mean_pert_recall_fpr5"])
        )
        agg_summary[bname] = rec
        summary_rows.append(
            [
                bname,
                f"{rec['mean_auroc']:.4f}",
                f"{100*rec['mean_recall_fpr5']:.1f}%",
                f"{100*rec['mean_real_fpr_fpr5']:.1f}%",
                f"{rec['mean_pert_auroc']:.4f}",
                f"{100*rec['mean_pert_recall_fpr5']:.1f}%",
                f"{100*rec['mean_recall_050']:.1f}%",
                f"{100*rec['mean_real_fpr_050']:.1f}%",
                f"{rec['logo_selection_score']:.4f}",
            ]
        )

    winner = max(agg_summary.values(), key=lambda x: x["logo_selection_score"])["branch"]

    fold_table_rows = [
        [
            r["fold"],
            r["branch"],
            str(r["held_out_n"]),
            f"{r['auroc']:.4f}",
            f"{100*r['recall_fpr5']:.1f}%",
            f"{100*r['real_fpr_fpr5']:.1f}%",
            f"{r['pert_mean_auroc']:.4f}",
            f"{100*r['pert_mean_recall_fpr5']:.1f}%",
            f"{100*r['recall_050']:.1f}%",
            f"{100*r['real_fpr_050']:.1f}%",
        ]
        for r in fold_records
    ]

    md_lines = [
        "# Step 3 Leave-One-Generator-Out (LOGO) Comparison\n",
        f"- Recommended branch by mean LOGO generalization (clean + perturbed `jpeg_q50` / `downscale_256`): **`{winner}`**",
        "- Strict isolation: `held_out_test` (`Midjourney`, `GPT_Image_1`, `test_real`, `seen_test`) was NOT touched.\n",
        "## Mean Across All 6 Generator Families\n",
        "| Branch | Clean AUROC | Clean Rec @<=5% | Real FPR @<=5% | Pert AUROC | Pert Rec @<=5% | Rec @ 0.50 | FPR @ 0.50 | LOGO Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sr in summary_rows:
        md_lines.append("| " + " | ".join(sr) + " |")
    md_lines.append("\n## Per-Fold Breakdown\n")
    md_lines.append("| Held-Out Generator | Branch | n | Clean AUROC | Clean Rec @<=5% | Real FPR @<=5% | Pert AUROC | Pert Rec @<=5% | Rec @ 0.50 | FPR @ 0.50 |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fr in fold_table_rows:
        md_lines.append("| " + " | ".join(fr) + " |")

    (args.out_dir / "logo_comparison.md").write_text("\n".join(md_lines) + "\n")
    save_json({"winner": winner, "summary": agg_summary, "folds": fold_records}, args.out_dir / "logo_comparison.json")
    logger.info("Saved LOGO comparison (winner=%s) -> %s", winner, args.out_dir / "logo_comparison.md")


def train_final_models(args: argparse.Namespace) -> None:
    """Train final CLIP-Linear, CLIP-MLP, ConvNeXt-v2, and ensemble strictly on train + val (unseen flux)."""
    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    if (face_df["split"] == "held_out_test").sum() == 0:
        raise RuntimeError("No held_out_test rows found in prepared manifest")

    train_df = face_df[face_df["split"] == "train"].reset_index(drop=True)
    val_df = face_df[face_df["split"] == "val"].reset_index(drop=True)

    # Focus OOD early-stopping on val_real + unseen_val_generator (flux family) clean & perturbed
    ood_val_df = val_df[
        val_df["split_role"].isin(["val_real", "unseen_val_generator"])
    ].reset_index(drop=True)

    tr_cache, va_cache, _ = _load_clip_caches(args.cache_dir)
    sha_to_tr = {s: i for i, s in enumerate(tr_cache["sha256"])}
    sha_to_va = {s: i for i, s in enumerate(va_cache["sha256"])}
    sha_lookup = {
        **{s: (tr_cache, i) for i, s in enumerate(tr_cache["sha256"])},
        **{s: (va_cache, i) for i, s in enumerate(va_cache["sha256"])},
    }

    device = get_device() if args.device == "auto" else torch.device(args.device)
    class_to_idx = load_class_mapping()["class_to_idx"]
    fake_idx = class_to_idx["FAKE"]

    # Balanced sampling for gradient steps (equal quota across seen fake generators & real sources)
    clip_train_df = sample_balanced_sources(train_df, target_total=min(len(train_df), 8000), seed=args.seed, allow_oversample=True)
    x_tr, y_tr = _lookup_clip_rows(clip_train_df, sha_to_tr, sha_to_va, tr_cache, va_cache, augmented=True)
    x_va, y_va = _lookup_clip_rows(ood_val_df, sha_to_tr, sha_to_va, tr_cache, va_cache, augmented=False)
    va_pert_dict = _lookup_val_pert_dict(ood_val_df, sha_to_va, va_cache)

    clip_lin, lin_meta = train_clip_head(
        x_tr, y_tr, x_va, y_va, val_feats_pert=va_pert_dict, head_type="linear",
        epochs=50, min_epochs=10, patience=10, device=device, seed=args.seed,
    )
    save_clip_head(
        clip_lin,
        MODELS_DIR / "clip_linear_v2.pt",
        {"class_to_idx": class_to_idx, "val_fpr5_threshold": lin_meta["threshold_val_fpr5"], "best_val": lin_meta},
    )

    clip_mlp, mlp_meta = train_clip_head(
        x_tr, y_tr, x_va, y_va, val_feats_pert=va_pert_dict, head_type="mlp",
        epochs=50, min_epochs=10, patience=10, device=device, seed=args.seed,
    )
    save_clip_head(
        clip_mlp,
        MODELS_DIR / "clip_mlp_v2.pt",
        {"class_to_idx": class_to_idx, "val_fpr5_threshold": mlp_meta["threshold_val_fpr5"], "best_val": mlp_meta},
    )
    logger.info("CLIP-Linear best val=%s | CLIP-MLP best val=%s", lin_meta, mlp_meta)

    conv_train_df = sample_balanced_sources(train_df, target_total=args.final_convnext_samples, seed=args.seed + 7, allow_oversample=True)
    val_pert_slice = pd.concat(
        [
            ood_val_df[ood_val_df["label"] == 1].sample(n=min(150, int((ood_val_df["label"] == 1).sum())), random_state=args.seed),
            ood_val_df[ood_val_df["label"] == 0].sample(n=min(150, int((ood_val_df["label"] == 0).sum())), random_state=args.seed),
        ],
        ignore_index=True,
    )
    conv_model, conv_meta = train_convnext_branch(
        conv_train_df,
        ood_val_df,
        epochs=args.final_epochs,
        min_epochs=args.final_min_epochs,
        patience=args.final_patience,
        batch_size=args.convnext_batch_size,
        lr=args.convnext_lr,
        weight_decay=0.05,
        save_path=MODELS_DIR / "convnext_v2.pt",
        class_to_idx=class_to_idx,
        device=device,
        val_pert_slice=val_pert_slice,
        seed=args.seed + 7,
    )

    # Fit ensemble blend weight strictly on full val set (ood + seen val real/fake, clean + perturbed slice)
    eval_tf = build_convnext_transforms(train=False, img_size=224)
    val_shas = val_df["sha256"].astype(str).tolist()
    y_val_t = torch.tensor(val_df["label"].to_numpy(dtype=int), dtype=torch.long)

    v_lin_l, _ = _eval_cached_clip_eval_set(clip_lin, val_shas, y_val_t.numpy(), sha_lookup, pert_name="original", device=device)
    v_mlp_l, _ = _eval_cached_clip_eval_set(clip_mlp, val_shas, y_val_t.numpy(), sha_lookup, pert_name="original", device=device)
    v_conv_l, _, _, _ = infer_convnext_two_view_logits(
        conv_model, val_df, eval_tf, perturbation="original", require_face=False, batch_size=args.convnext_batch_size * 2, device=device
    )

    best_clip_key = "clip_mlp" if mlp_meta["gen_score"] >= lin_meta["gen_score"] else "clip_linear"
    best_clip_l = v_mlp_l if best_clip_key == "clip_mlp" else v_lin_l
    best_clip_head = clip_mlp if best_clip_key == "clip_mlp" else clip_lin

    pert_logit_triples = []
    vp_shas = val_pert_slice["sha256"].astype(str).tolist()
    y_vp_t = torch.tensor(val_pert_slice["label"].to_numpy(dtype=int), dtype=torch.long)
    for p_name in ("jpeg_q50", "downscale_256", "downscale_512_jpeg_q75"):
        pc_l, ok_c = _eval_cached_clip_eval_set(best_clip_head, vp_shas, y_vp_t.numpy(), sha_lookup, pert_name=p_name, device=device)
        pv_l, _, _, ok_v = infer_convnext_two_view_logits(
            conv_model, val_pert_slice, eval_tf, perturbation=p_name, require_face=False, batch_size=args.convnext_batch_size * 2, device=device
        )
        both_ok = ok_c & ok_v
        if both_ok.sum() >= 20:
            pert_logit_triples.append((pc_l[both_ok], pv_l[both_ok], y_vp_t[both_ok]))

    ens_cfg = fit_ensemble_weights(
        best_clip_l, v_conv_l, y_val_t, val_pert_logits=pert_logit_triples, fake_idx=fake_idx
    )
    # Also compute val_real FPR=5% thresholds for each individual final branch on full val_df
    p_v_lin = torch.softmax(v_lin_l.float(), dim=-1)[:, fake_idx].numpy()
    p_v_mlp = torch.softmax(v_mlp_l.float(), dim=-1)[:, fake_idx].numpy()
    p_v_conv = torch.softmax(v_conv_l.float(), dim=-1)[:, fake_idx].numpy()
    val_real_mask = val_df["label"].to_numpy(dtype=int) != fake_idx

    t_lin, _ = compute_fpr_threshold(p_v_lin[val_real_mask], 0.05)
    t_mlp, _ = compute_fpr_threshold(p_v_mlp[val_real_mask], 0.05)
    t_conv, _ = compute_fpr_threshold(p_v_conv[val_real_mask], 0.05)

    step3_meta = {
        "best_clip_branch": best_clip_key,
        "clip_linear_checkpoint": "models/clip_linear_v2.pt",
        "clip_mlp_checkpoint": "models/clip_mlp_v2.pt",
        "convnext_checkpoint": "models/convnext_v2.pt",
        "thresholds_val_fpr5": {
            "clip_linear": float(t_lin),
            "clip_mlp": float(t_mlp),
            "convnext_v2": float(t_conv),
            "ensemble": float(ens_cfg["threshold_val_fpr5"]),
        },
        "ensemble": ens_cfg,
        "val_metrics": {
            "clip_linear": lin_meta,
            "clip_mlp": mlp_meta,
            "convnext_v2": conv_meta,
        },
    }
    save_json(step3_meta, MODELS_DIR / "step3_ensemble.json")
    logger.info("Saved final Step 3 artifacts and ensemble config -> %s", MODELS_DIR / "step3_ensemble.json")


def eval_heldout_once(args: argparse.Namespace) -> None:
    """Single read-only held_out_test evaluation comparing baseline best.pt, CLIP heads, ConvNeXt v2, and ensemble."""
    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    test_df = face_df[face_df["split"] == "held_out_test"].reset_index(drop=True)
    val_df = face_df[face_df["split"] == "val"].reset_index(drop=True)

    ens_meta = load_json(MODELS_DIR / "step3_ensemble.json")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    class_to_idx = load_class_mapping()["class_to_idx"]
    fake_idx = class_to_idx["FAKE"]
    eval_tf = build_convnext_transforms(train=False, img_size=224)

    clip_lin, _ = load_clip_head(MODELS_DIR / "clip_linear_v2.pt", device)
    clip_mlp, _ = load_clip_head(MODELS_DIR / "clip_mlp_v2.pt", device)
    conv_v2, _ = load_checkpoint(MODELS_DIR / "convnext_v2.pt", device)

    # Also compute baseline models/best.pt val_fpr5 threshold strictly on val_real (if not saved)
    base_det = DeepfakeDetector(checkpoint=CHECKPOINT_PATH, device=device)

    _, _, te_cache = _load_clip_caches(args.cache_dir)
    te_lookup = {s: (te_cache, i) for i, s in enumerate(te_cache["sha256"])}

    val_real_df = val_df[val_df["label"] == 0].reset_index(drop=True)
    # Deterministic subsample up to max_eval per source if needed, or full 5,436 clean + pert subset
    if args.heldout_clean_cap > 0:
        parts = [grp.sample(n=min(args.heldout_clean_cap, len(grp)), random_state=args.seed) for _, grp in test_df.groupby("generator")]
        clean_df = pd.concat(parts, ignore_index=True)
    else:
        clean_df = test_df

    pert_parts = [grp.sample(n=min(args.heldout_pert_per_source, len(grp)), random_state=args.seed + 11) for _, grp in clean_df.groupby("generator")]
    pert_df = pd.concat(pert_parts, ignore_index=True)

    cached_pert_subset = te_cache.get("pert_sha256_subset")
    if cached_pert_subset is not None:
        needed_shas = set(pert_df["sha256"].astype(str))
        if not needed_shas.issubset(set(cached_pert_subset)):
            missing_n = len(needed_shas - set(cached_pert_subset))
            raise RuntimeError(
                f"held_out_test_eval.pt pert_sha256_subset is missing {missing_n} required perturbation sha256s; "
                "re-extract held_out_test_eval.pt with matching --test-pert-per-source / --seed."
            )

    # Baseline predictions via batched helper on clean_df & val_real_df (or subsample val_real)
    def _score_baseline(sub: pd.DataFrame, pert_name: str) -> tuple[np.ndarray, np.ndarray]:
        from PIL import ImageOps as _IO

        base_det.model.eval()
        all_p, all_ok = [], []
        batch_t, refs = [], []
        for idx, row in enumerate(sub.itertuples(index=False)):
            v_rgb = apply_eval_perturbation(_load_rgb(str(row.path)), pert_name)
            cropped, meta = gate_and_crop_face(v_rgb)
            if cropped is None:
                all_p.append(np.nan)
                all_ok.append(False)
                continue
            all_p.append(0.0)
            all_ok.append(True)
            batch_t.extend([base_det.transform(cropped), base_det.transform(_IO.mirror(cropped)), base_det.transform(v_rgb)])
            refs.append(idx)
            if len(refs) >= 16:
                bt = torch.stack(batch_t, dim=0).to(device)
                with torch.no_grad():
                    lg = base_det.model(bt).float().cpu().view(len(refs), 3, -1).mean(dim=1)
                ps = torch.softmax(lg / base_det.temperature, dim=-1)[:, fake_idx].numpy()
                for ri, pv in zip(refs, ps):
                    all_p[ri] = float(pv)
                batch_t, refs = [], []
        if refs:
            bt = torch.stack(batch_t, dim=0).to(device)
            with torch.no_grad():
                lg = base_det.model(bt).float().cpu().view(len(refs), 3, -1).mean(dim=1)
            ps = torch.softmax(lg / base_det.temperature, dim=-1)[:, fake_idx].numpy()
            for ri, pv in zip(refs, ps):
                all_p[ri] = float(pv)
        return np.array(all_p, dtype=float), np.array(all_ok, dtype=bool)

    base_val_real_p, base_vr_ok = _score_baseline(val_real_df.sample(n=min(300, len(val_real_df)), random_state=args.seed), "original")
    base_t_fpr5, _ = compute_fpr_threshold(base_val_real_p[base_vr_ok], 0.05)

    y_clean = clean_df["label"].to_numpy(dtype=int)
    clean_shas = clean_df["sha256"].astype(str).tolist()
    c_lin_l, c_lin_ok = _eval_cached_clip_eval_set(clip_lin, clean_shas, y_clean, te_lookup, pert_name="original", device=device)
    c_mlp_l, _ = _eval_cached_clip_eval_set(clip_mlp, clean_shas, y_clean, te_lookup, pert_name="original", device=device)
    c_conv_l, _, _, conv_ok = infer_convnext_two_view_logits(
        conv_v2, clean_df, eval_tf, perturbation="original", require_face=True, batch_size=args.convnext_batch_size * 2, device=device
    )
    base_c_p, base_c_ok = _score_baseline(clean_df, "original")

    w_clip = float(ens_meta["ensemble"]["clip_weight"])
    best_clip_key = ens_meta["best_clip_branch"]
    best_clip_c_l = c_mlp_l if best_clip_key == "clip_mlp" else c_lin_l
    ens_c_l = w_clip * best_clip_c_l + (1.0 - w_clip) * c_conv_l

    thresholds = ens_meta["thresholds_val_fpr5"]
    model_preds = {
        "baseline_best_pt": (base_c_p, base_c_ok, base_t_fpr5),
        "clip_linear_v2": (torch.softmax(c_lin_l.float(), dim=-1)[:, fake_idx].numpy(), c_lin_ok.numpy(), thresholds["clip_linear"]),
        "clip_mlp_v2": (torch.softmax(c_mlp_l.float(), dim=-1)[:, fake_idx].numpy(), c_lin_ok.numpy(), thresholds["clip_mlp"]),
        "convnext_v2": (torch.softmax(c_conv_l.float(), dim=-1)[:, fake_idx].numpy(), conv_ok.numpy(), thresholds["convnext_v2"]),
        "ensemble_v2": (torch.softmax(ens_c_l.float(), dim=-1)[:, fake_idx].numpy(), (c_lin_ok & conv_ok).numpy(), thresholds["ensemble"]),
    }

    overall_rows = []
    source_rows = []
    gens = clean_df["generator"].astype(str).to_numpy()
    roles = clean_df["split_role"].astype(str).to_numpy()
    for mname, (probs, ok_mask, thr) in model_preds.items():
        m = compute_binary_metrics(probs[ok_mask], y_clean[ok_mask], threshold=thr, fake_idx=fake_idx)
        m05 = compute_binary_metrics(probs[ok_mask], y_clean[ok_mask], threshold=0.5, fake_idx=fake_idx)
        unseen_mask = ok_mask & (clean_df["split_role"] == "unseen_test_generator").to_numpy()
        real_mask = ok_mask & (y_clean == 0)
        unseen_auc = compute_binary_metrics(
            np.concatenate([probs[real_mask], probs[unseen_mask]]),
            np.concatenate([y_clean[real_mask], y_clean[unseen_mask]]),
            threshold=thr,
            fake_idx=fake_idx,
        )
        overall_rows.append(
            {
                "model": mname,
                "threshold_val_fpr5": float(thr),
                "n_face": int(ok_mask.sum()),
                "auroc": float(m["auroc"] or 0.0),
                "recall_fpr5": float(m["recall"]),
                "real_fpr_fpr5": float(m["real_fpr"]),
                "unseen_auroc": float(unseen_auc["auroc"] or 0.0),
                "unseen_recall_fpr5": float(unseen_auc["recall"]),
                "recall_050": float(m05["recall"]),
                "real_fpr_050": float(m05["real_fpr"]),
            }
        )
        real_scores = probs[real_mask]
        for gen_name in sorted(np.unique(gens)):
            gm = ok_mask & (gens == gen_name)
            if gm.sum() == 0:
                continue
            cls = int(y_clean[gm][0])
            role = str(roles[gm][0])
            if cls == 0:
                source_rows.append(
                    {
                        "model": mname,
                        "source": gen_name,
                        "class": "REAL",
                        "role": role,
                        "n": int(gm.sum()),
                        "metric": "FPR",
                        "rate_fpr5": float((probs[gm] >= thr).mean()),
                        "rate_050": float((probs[gm] >= 0.5).mean()),
                        "auroc_vs_real": None,
                    }
                )
            else:
                auc_g = compute_binary_metrics(
                    np.concatenate([real_scores, probs[gm]]),
                    np.concatenate([np.zeros(len(real_scores)), np.ones(gm.sum())]),
                    threshold=thr,
                    fake_idx=1,
                )["auroc"]
                source_rows.append(
                    {
                        "model": mname,
                        "source": gen_name,
                        "class": "FAKE",
                        "role": role,
                        "n": int(gm.sum()),
                        "metric": "Recall",
                        "rate_fpr5": float((probs[gm] >= thr).mean()),
                        "rate_050": float((probs[gm] >= 0.5).mean()),
                        "auroc_vs_real": float(auc_g or 0.0),
                    }
                )

    # Perturbation sweep on pert_df
    pert_rows = []
    y_pv = pert_df["label"].to_numpy(dtype=int)
    pv_shas = pert_df["sha256"].astype(str).tolist()
    for p_name in STEP3_EVAL_PERTS:
        p_lin, ok_lin = _eval_cached_clip_eval_set(clip_lin, pv_shas, y_pv, te_lookup, pert_name=p_name, device=device)
        p_mlp, _ = _eval_cached_clip_eval_set(clip_mlp, pv_shas, y_pv, te_lookup, pert_name=p_name, device=device)
        p_conv, _, _, ok_conv = infer_convnext_two_view_logits(
            conv_v2, pert_df, eval_tf, perturbation=p_name, require_face=True, batch_size=args.convnext_batch_size * 2, device=device
        )
        p_base, ok_base = _score_baseline(pert_df, p_name)
        p_clip_best = p_mlp if best_clip_key == "clip_mlp" else p_lin
        p_ens = w_clip * p_clip_best + (1.0 - w_clip) * p_conv
        for mname, (ps, ok_m, thr) in {
            "baseline_best_pt": (p_base, ok_base, base_t_fpr5),
            "clip_linear_v2": (torch.softmax(p_lin.float(), dim=-1)[:, fake_idx].numpy(), ok_lin.numpy(), thresholds["clip_linear"]),
            "clip_mlp_v2": (torch.softmax(p_mlp.float(), dim=-1)[:, fake_idx].numpy(), ok_lin.numpy(), thresholds["clip_mlp"]),
            "convnext_v2": (torch.softmax(p_conv.float(), dim=-1)[:, fake_idx].numpy(), ok_conv.numpy(), thresholds["convnext_v2"]),
            "ensemble_v2": (torch.softmax(p_ens.float(), dim=-1)[:, fake_idx].numpy(), (ok_lin & ok_conv).numpy(), thresholds["ensemble"]),
        }.items():
            pm = compute_binary_metrics(ps[ok_m], y_pv[ok_m], threshold=thr, fake_idx=fake_idx)
            pert_rows.append(
                {
                    "perturbation": p_name,
                    "model": mname,
                    "n_face": int(ok_m.sum()),
                    "auroc": float(pm["auroc"] or 0.0),
                    "recall_fpr5": float(pm["recall"]),
                    "real_fpr_fpr5": float(pm["real_fpr"]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(args.out_dir / "heldout_overall.csv", index=False)
    pd.DataFrame(source_rows).to_csv(args.out_dir / "heldout_per_source.csv", index=False)
    pd.DataFrame(pert_rows).to_csv(args.out_dir / "heldout_perturbations.csv", index=False)

    md = [
        "# Step 3 Held-Out Test Report (Read-Only)\n",
        f"Best CLIP branch from `val`: **`{best_clip_key}`** | Ensemble clip weight: **`{w_clip:.2f}`** | Unseen test generators: `Midjourney`, `GPT_Image_1`\n",
        "## 1. Overall Held-Out Test Comparison\n",
        "| Model | Val<=5% Thresh | Overall AUROC | Overall Recall @<=5% | Unseen Test AUROC | Unseen Recall @<=5% | Test Real FPR @<=5% | Recall @ 0.50 | Real FPR @ 0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overall_rows:
        md.append(
            f"| `{r['model']}` | {r['threshold_val_fpr5']:.4f} | {r['auroc']:.4f} | "
            f"{100*r['recall_fpr5']:.1f}% | {r['unseen_auroc']:.4f} | {100*r['unseen_recall_fpr5']:.1f}% | "
            f"{100*r['real_fpr_fpr5']:.1f}% | {100*r['recall_050']:.1f}% | {100*r['real_fpr_050']:.1f}% |"
        )
    md.append("\n## 2. Per-Source Breakdown (`@ Val<=5% FPR`)\n")
    md.append("| Model | Source | Class | Role | n | Metric | Rate @ Val<=5% | Rate @ 0.50 | AUROC vs Test Real |")
    md.append("|---|---|---|---|---:|---|---:|---:|---:|")
    for s in source_rows:
        auc_s = "-" if s["auroc_vs_real"] is None else f"{s['auroc_vs_real']:.4f}"
        md.append(
            f"| `{s['model']}` | `{s['source']}` | {s['class']} | `{s['role']}` | {s['n']} | {s['metric']} | "
            f"{100*s['rate_fpr5']:.1f}% | {100*s['rate_050']:.1f}% | {auc_s} |"
        )
    md.append("\n## 3. Perturbation Robustness (`@ Val<=5% FPR`)\n")
    md.append("| Perturbation | Model | n (Face) | AUROC | Recall @<=5% | Real FPR @<=5% |")
    md.append("|---|---|---:|---:|---:|---:|")
    for pr in pert_rows:
        md.append(
            f"| `{pr['perturbation']}` | `{pr['model']}` | {pr['n_face']} | {pr['auroc']:.4f} | "
            f"{100*pr['recall_fpr5']:.1f}% | {100*pr['real_fpr_fpr5']:.1f}% |"
        )
    (args.out_dir / "heldout_report.md").write_text("\n".join(md) + "\n")
    save_json({"overall": overall_rows, "per_source": source_rows, "perturbations": pert_rows}, args.out_dir / "heldout_summary.json")
    logger.info("Saved Step 3 held-out test report -> %s", args.out_dir / "heldout_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3 LOGO comparison, final training, and held-out eval.")
    parser.add_argument("--task", required=True, choices=["logo", "train-final", "eval-heldout"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)

    # LOGO options
    parser.add_argument("--logo-families", nargs="+", default=list(LOGO_FAMILIES))
    parser.add_argument("--logo-train-cap", type=int, default=2400)
    parser.add_argument("--logo-epochs", type=int, default=7)
    parser.add_argument("--logo-min-epochs", type=int, default=5)
    parser.add_argument("--logo-patience", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")

    # Final training & shared ConvNeXt hyperparameters
    parser.add_argument("--final-convnext-samples", type=int, default=4800)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--final-min-epochs", type=int, default=5)
    parser.add_argument("--final-patience", type=int, default=3)
    parser.add_argument("--convnext-batch-size", type=int, default=32)
    parser.add_argument("--convnext-lr", type=float, default=1e-4)

    # Held-out evaluation options
    parser.add_argument("--heldout-clean-cap", type=int, default=0, help="0 = all test images")
    parser.add_argument("--heldout-pert-per-source", type=int, default=30)
    args = parser.parse_args()

    register_image_plugins()
    if args.task == "logo":
        run_logo_benchmark(args)
    elif args.task == "train-final":
        train_final_models(args)
    else:
        eval_heldout_once(args)


if __name__ == "__main__":
    main()
