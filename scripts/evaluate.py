"""Folder-based evaluation, 11-perturbation robustness suite, and LOGO benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.calibration import (  # noqa: E402
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    apply_temperature,
)
from src.face_crop import crop_face_or_full, gate_and_crop_face  # noqa: E402
from src.image_io import decode_image_bytes  # noqa: E402
from src.inference import DeepfakeDetector, logits_to_probabilities  # noqa: E402
from src.provenance import inspect_provenance  # noqa: E402
from src.utils import (  # noqa: E402
    CHECKPOINT_PATH,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    get_device,
    get_logger,
    save_class_mapping,
    save_json,
    seed_everything,
)

logger = get_logger("eval_harness")

IMAGE_EXTS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".gif",
        ".avif",
        ".heic",
        ".heif",
    }
)

SEEN_BY_CURRENT_MODEL: frozenset[str] = frozenset(
    {"coco", "celeba", "celeba_hq", "ffhq", "dalle3", "flux", "sd21", "sd3", "sdxl", "stylegan"}
)

PERTURBATIONS: tuple[str, ...] = (
    "original",
    "metadata_stripped",
    "jpeg_q95",
    "jpeg_q85",
    "jpeg_q70",
    "jpeg_q50",
    "webp_q80",
    "downscale_1024",
    "downscale_512",
    "downscale_256",
    "downscale_512_jpeg_q75",
)

VIEW_DISAGREE_MAG: float = 1.0


@dataclass(frozen=True)
class EvalSample:
    path: Path
    class_name: str  # REAL or FAKE
    label: int  # 0=REAL, 1=FAKE
    generator: str  # "real" for real/ folder, else <generator_name>
    subsource: str  # e.g. "coco" or "ffhq" inside real/, else generator name
    rel_id: str


@dataclass
class CachedPixelResult:
    face_detected: bool
    p_uncal_fake: Optional[float]
    p_cal_fake: Optional[float]
    crop_delta: Optional[float]
    full_delta: Optional[float]
    views_disagree: bool


def _is_image_file(p: Path) -> bool:
    return (p.is_file() or p.is_symlink()) and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS


def _collect_sorted_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if _is_image_file(p)),
        key=lambda p: str(p.relative_to(folder)),
    )


def _deterministic_sample(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    items = list(paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    return sorted(items[:limit], key=lambda p: str(p))


def discover_real_folder(
    real_dir: Path,
    *,
    max_per_subsource: int,
    perturb_max_per_subsource: int,
    seed: int,
) -> tuple[list[EvalSample], list[EvalSample]]:
    """Collect clean and perturbation samples from real/ (grouped by immediate subfolder if present)."""
    if not real_dir.exists():
        return [], []
    by_subsource: dict[str, list[Path]] = {}
    for p in _collect_sorted_images(real_dir):
        rel_parts = p.relative_to(real_dir).parts
        sub = rel_parts[0] if len(rel_parts) >= 2 else "real"
        by_subsource.setdefault(sub, []).append(p)

    clean_samples: list[EvalSample] = []
    perturb_samples: list[EvalSample] = []
    for idx, (sub, paths) in enumerate(sorted(by_subsource.items())):
        picked = _deterministic_sample(paths, max_per_subsource, seed + idx)
        p_picked = (
            picked[:perturb_max_per_subsource]
            if perturb_max_per_subsource > 0
            else picked
        )
        for p in picked:
            clean_samples.append(
                EvalSample(
                    path=p,
                    class_name="REAL",
                    label=0,
                    generator="real",
                    subsource=sub,
                    rel_id=f"real/{sub}/{p.name}",
                )
            )
        for p in p_picked:
            perturb_samples.append(
                EvalSample(
                    path=p,
                    class_name="REAL",
                    label=0,
                    generator="real",
                    subsource=sub,
                    rel_id=f"real/{sub}/{p.name}",
                )
            )
    return clean_samples, perturb_samples


def discover_eval_dir(
    eval_dir: Path,
    *,
    max_per_folder: int,
    max_real_per_subsource: int,
    perturb_max_per_folder: int,
    perturb_max_real_per_subsource: int,
    seed: int,
) -> tuple[list[str], list[EvalSample], list[EvalSample]]:
    """Discover real/ and every <generator>/ folder under eval_dir."""
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")

    subdirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name.lower(),
    )
    real_dir = next((d for d in subdirs if d.name.lower() == "real"), None)
    gen_dirs = [d for d in subdirs if d.name.lower() != "real"]

    clean_all: list[EvalSample] = []
    perturb_all: list[EvalSample] = []
    if real_dir is not None:
        r_clean, r_pert = discover_real_folder(
            real_dir,
            max_per_subsource=max_real_per_subsource,
            perturb_max_per_subsource=perturb_max_real_per_subsource,
            seed=seed,
        )
        clean_all.extend(r_clean)
        perturb_all.extend(r_pert)

    discovered_generators: list[str] = []
    for idx, gdir in enumerate(gen_dirs):
        gname = gdir.name
        discovered_generators.append(gname)
        paths = _collect_sorted_images(gdir)
        picked = _deterministic_sample(paths, max_per_folder, seed + 100 + idx)
        p_picked = (
            picked[:perturb_max_per_folder] if perturb_max_per_folder > 0 else picked
        )
        for p in picked:
            clean_all.append(
                EvalSample(
                    path=p,
                    class_name="FAKE",
                    label=1,
                    generator=gname,
                    subsource=gname,
                    rel_id=f"{gname}/{p.name}",
                )
            )
        for p in p_picked:
            perturb_all.append(
                EvalSample(
                    path=p,
                    class_name="FAKE",
                    label=1,
                    generator=gname,
                    subsource=gname,
                    rel_id=f"{gname}/{p.name}",
                )
            )
    return discovered_generators, clean_all, perturb_all


def _encode_bytes(img: Image.Image, fmt: str, quality: Optional[int] = None) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG")
    elif fmt == "JPEG":
        img.save(buf, format="JPEG", quality=int(quality or 95))
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=int(quality or 80))
    else:
        raise ValueError(f"Unsupported perturbation format: {fmt}")
    return buf.getvalue()


def _resize_max_side_if_larger(
    img: Image.Image, max_side: int
) -> tuple[Image.Image, bool]:
    w, h = img.size
    if max(w, h) <= max_side:
        return img, False
    scale = max_side / float(max(w, h))
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS), True


def apply_perturbation(
    orig_rgb: Image.Image,
    orig_raw: bytes,
    orig_fmt: str,
    pert_name: str,
) -> tuple[Image.Image, bytes, str, bool]:
    """Return (rgb_image, raw_bytes_for_provenance, format_name, actually_resized)."""
    if pert_name == "original":
        return orig_rgb, orig_raw, orig_fmt, False
    if pert_name == "metadata_stripped":
        png_bytes = _encode_bytes(orig_rgb, "PNG")
        return orig_rgb, png_bytes, "PNG", False
    if pert_name in ("jpeg_q95", "jpeg_q85", "jpeg_q70", "jpeg_q50"):
        q = int(pert_name.split("_q")[1])
        jb = _encode_bytes(orig_rgb, "JPEG", quality=q)
        with Image.open(io.BytesIO(jb)) as im:
            return im.convert("RGB"), jb, "JPEG", False
    if pert_name == "webp_q80":
        wb = _encode_bytes(orig_rgb, "WEBP", quality=80)
        with Image.open(io.BytesIO(wb)) as im:
            return im.convert("RGB"), wb, "WEBP", False
    if pert_name in ("downscale_1024", "downscale_512", "downscale_256"):
        target = int(pert_name.split("_")[1])
        resized, did_resize = _resize_max_side_if_larger(orig_rgb, target)
        pb = _encode_bytes(resized, "PNG")
        return resized, pb, "PNG", did_resize
    if pert_name == "downscale_512_jpeg_q75":
        resized, did_resize = _resize_max_side_if_larger(orig_rgb, 512)
        jb = _encode_bytes(resized, "JPEG", quality=75)
        with Image.open(io.BytesIO(jb)) as im:
            return im.convert("RGB"), jb, "JPEG", did_resize
    raise ValueError(f"Unknown perturbation: {pert_name}")


def _flush_batch(
    detector: DeepfakeDetector,
    pending: list[tuple[str, bool, Image.Image, Image.Image]],
    cache: dict[str, CachedPixelResult],
) -> None:
    if not pending:
        return
    real_idx = detector.class_to_idx["REAL"]
    fake_idx = detector.class_to_idx["FAKE"]
    tensors: list[torch.Tensor] = []
    for _, _, rgb, cropped in pending:
        tensors.append(detector.transform(cropped))
        tensors.append(detector.transform(ImageOps.mirror(cropped)))
        tensors.append(detector.transform(rgb))

    batch = torch.stack(tensors, dim=0).to(detector.device, non_blocking=True)
    with torch.no_grad():
        logits = detector.model(batch).float().cpu()
    logits = logits.view(len(pending), 3, -1)

    for i, (pix_md5, fdet, _, _) in enumerate(pending):
        views = logits[i]
        crop_delta = float(views[0, fake_idx].item() - views[0, real_idx].item())
        full_delta = float(views[2, fake_idx].item() - views[2, real_idx].item())
        views_disagree = (
            (crop_delta * full_delta < 0.0)
            and abs(crop_delta) >= VIEW_DISAGREE_MAG
            and abs(full_delta) >= VIEW_DISAGREE_MAG
        )
        mean_logit = views.mean(dim=0, keepdim=True)
        _, _, p_unc, _ = logits_to_probabilities(mean_logit, detector.idx_to_class)
        scaled = apply_temperature(mean_logit, detector.temperature)
        _, _, p_cal, _ = logits_to_probabilities(scaled, detector.idx_to_class)
        cache[pix_md5] = CachedPixelResult(
            face_detected=fdet,
            p_uncal_fake=float(p_unc["FAKE"]),
            p_cal_fake=float(p_cal["FAKE"]),
            crop_delta=crop_delta,
            full_delta=full_delta,
            views_disagree=views_disagree,
        )


def score_samples_with_perturbations(
    detector: DeepfakeDetector,
    samples: list[EvalSample],
    *,
    perturbations: tuple[str, ...],
    require_face: bool,
    batch_size: int,
    pixel_cache: Optional[dict[str, CachedPixelResult]] = None,
    existing_orig_records: Optional[dict[str, dict[str, Any]]] = None,
    verify_first_parity: bool = False,
) -> list[dict[str, Any]]:
    """Score samples across perturbations with pixel-MD5 dedup cache and batched TTA."""
    if pixel_cache is None:
        pixel_cache = {}
    records: list[dict[str, Any]] = []
    pending_items: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    pending_batch: list[tuple[str, bool, Image.Image, Image.Image]] = []

    def _finalize_pending() -> None:
        nonlocal pending_items, pending_batch
        if not pending_batch:
            return
        _flush_batch(detector, pending_batch, pixel_cache)
        for rec, orig_rgb, _ in pending_items:
            res = pixel_cache[rec["_pix_md5"]]
            rec["face_detected"] = res.face_detected
            rec["p_uncal_fake"] = res.p_uncal_fake
            rec["p_cal_fake"] = res.p_cal_fake
            rec["crop_delta"] = res.crop_delta
            rec["full_delta"] = res.full_delta
            rec["views_disagree"] = res.views_disagree
        pending_items = []
        pending_batch = []

    parity_checked = not verify_first_parity

    for s in samples:
        orig_raw = s.path.read_bytes()
        orig_rgb, orig_fmt = decode_image_bytes(orig_raw)
        for pert in perturbations:
            if (
                pert == "original"
                and existing_orig_records is not None
                and s.rel_id in existing_orig_records
            ):
                records.append(dict(existing_orig_records[s.rel_id]))
                continue

            v_rgb, v_bytes, v_fmt, did_resize = apply_perturbation(
                orig_rgb, orig_raw, orig_fmt, pert
            )
            pix_md5 = hashlib.md5(v_rgb.tobytes()).hexdigest()
            prov = inspect_provenance(v_bytes, detected_format=v_fmt)
            c2pa_override = bool(prov.get("ai_verified_by_credentials", False))

            rec: dict[str, Any] = {
                "rel_id": s.rel_id,
                "path": str(s.path),
                "class_name": s.class_name,
                "label": s.label,
                "generator": s.generator,
                "subsource": s.subsource,
                "perturbation": pert,
                "width": v_rgb.width,
                "height": v_rgb.height,
                "actually_resized": did_resize,
                "c2pa_override": c2pa_override,
                "_pix_md5": pix_md5,
            }

            if pix_md5 in pixel_cache:
                res = pixel_cache[pix_md5]
                rec["face_detected"] = res.face_detected
                rec["p_uncal_fake"] = res.p_uncal_fake
                rec["p_cal_fake"] = res.p_cal_fake
                rec["crop_delta"] = res.crop_delta
                rec["full_delta"] = res.full_delta
                rec["views_disagree"] = res.views_disagree
                records.append(rec)
                continue

            if require_face:
                cropped, face_meta = gate_and_crop_face(v_rgb)
                fdet = bool(face_meta.get("face_detected", False))
            else:
                cropped, face_meta = crop_face_or_full(v_rgb)
                fdet = bool(face_meta.get("face_detected", False))

            if cropped is None:
                pixel_cache[pix_md5] = CachedPixelResult(
                    face_detected=False,
                    p_uncal_fake=None,
                    p_cal_fake=None,
                    crop_delta=None,
                    full_delta=None,
                    views_disagree=False,
                )
                rec["face_detected"] = False
                rec["p_uncal_fake"] = None
                rec["p_cal_fake"] = None
                rec["crop_delta"] = None
                rec["full_delta"] = None
                rec["views_disagree"] = False
                records.append(rec)
                continue

            if not parity_checked and pert == "original":
                prod = detector.predict(
                    orig_raw,
                    require_face=require_face,
                    raw_bytes=orig_raw,
                    use_tta=True,
                    detected_format=orig_fmt,
                )
                _flush_batch(detector, [(pix_md5, fdet, v_rgb, cropped)], pixel_cache)
                res = pixel_cache[pix_md5]
                prod_p = float((prod.get("pixel_model") or {})["probabilities"]["FAKE"])
                diff = abs(prod_p - float(res.p_cal_fake or 0.0))
                if diff > 1e-4:
                    raise RuntimeError(
                        f"Batched parity check failed on {s.path}: {res.p_cal_fake} vs {prod_p}"
                    )
                logger.info("Verified batched 3-view parity vs DeepfakeDetector.predict (diff=%.2e)", diff)
                parity_checked = True
                rec["face_detected"] = res.face_detected
                rec["p_uncal_fake"] = res.p_uncal_fake
                rec["p_cal_fake"] = res.p_cal_fake
                rec["crop_delta"] = res.crop_delta
                rec["full_delta"] = res.full_delta
                rec["views_disagree"] = res.views_disagree
                records.append(rec)
                continue

            records.append(rec)
            pending_items.append((rec, v_rgb, cropped))
            pending_batch.append((pix_md5, fdet, v_rgb, cropped))
            if len(pending_batch) >= batch_size:
                _finalize_pending()

    _finalize_pending()
    for rec in records:
        rec.pop("_pix_md5", None)
    return records


def compute_fpr_threshold(
    real_scores: np.ndarray, target_fpr: float = 0.05
) -> tuple[float, float]:
    """Return smallest threshold in [0, 1] achieving empirical FPR <= target_fpr on real_scores."""
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
    actual_fpr = float(np.mean(scores >= chosen))
    return chosen, actual_fpr


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _binary_summary(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    face_df = df[df["p_cal_fake"].notna()].copy()
    if len(face_df) == 0:
        return {"n_total": int(len(df)), "n_face": 0}
    y = face_df["label"].to_numpy(dtype=int)
    p = face_df["p_cal_fake"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    real_fpr = fp / max(1, tn + fp)
    fake_rec = tp / max(1, tp + fn)
    return {
        "threshold": float(threshold),
        "n_total": int(len(df)),
        "n_face": int(len(face_df)),
        "no_face_count": int(len(df) - len(face_df)),
        "auroc": _safe_auroc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(fake_rec),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "real_fpr": float(real_fpr),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "uncertain_rate": float(((p >= UNCERTAIN_LOW) & (p <= UNCERTAIN_HIGH)).mean()),
        "view_disagree_rate": float(face_df["views_disagree"].mean()),
    }


def _build_eval_tables(
    clean_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    discovered_generators: list[str],
    *,
    t_val_fpr5: float,
    t_oracle_eval_fpr5: float,
    manual_threshold: Optional[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    real_clean = clean_df[clean_df["label"] == 0]
    real_face = real_clean[real_clean["p_cal_fake"].notna()]
    real_scores_all = real_face["p_cal_fake"].to_numpy(dtype=float)

    # 1. Per-real-subsource table
    real_sub_rows: list[dict[str, Any]] = []
    for sub, grp in sorted(real_clean.groupby("subsource")):
        fg = grp[grp["p_cal_fake"].notna()]
        p = fg["p_cal_fake"].to_numpy(dtype=float)
        real_sub_rows.append(
            {
                "subsource": sub,
                "seen_in_train": "unseen" if sub.lower() not in SEEN_BY_CURRENT_MODEL else "seen",
                "n": int(len(grp)),
                "n_face": int(len(fg)),
                "fpr_0_50": float((p >= 0.5).mean()) if len(p) else None,
                "fpr_val_fpr5": float((p >= t_val_fpr5).mean()) if len(p) else None,
                "mean_p_fake": float(p.mean()) if len(p) else None,
                "uncertain_rate": float(((p >= UNCERTAIN_LOW) & (p <= UNCERTAIN_HIGH)).mean()) if len(p) else None,
                "view_disagree_rate": float(fg["views_disagree"].mean()) if len(fg) else None,
            }
        )

    # 2. Per-generator table on clean original images
    gen_rows: list[dict[str, Any]] = []
    for gname in discovered_generators:
        seen_status = "seen" if gname.lower() in SEEN_BY_CURRENT_MODEL else "unseen"
        grp = clean_df[(clean_df["label"] == 1) & (clean_df["generator"] == gname)]
        if len(grp) == 0:
            gen_rows.append(
                {
                    "generator": gname,
                    "seen_in_train": seen_status,
                    "n": 0,
                    "n_face": 0,
                    "no_face_rate": None,
                    "auroc_vs_real": None,
                    "recall_0_50_face": None,
                    "recall_0_50_e2e": None,
                    "recall_val_fpr5_face": None,
                    "recall_oracle_eval_fpr5_face": None,
                    "mean_p_fake": None,
                    "uncertain_rate": None,
                    "view_disagree_rate": None,
                    "c2pa_overrides": 0,
                }
            )
            continue
        fg = grp[grp["p_cal_fake"].notna()]
        p = fg["p_cal_fake"].to_numpy(dtype=float)
        if len(p) and len(real_scores_all):
            auc = _safe_auroc(
                np.concatenate([np.zeros(len(real_scores_all), dtype=int), np.ones(len(p), dtype=int)]),
                np.concatenate([real_scores_all, p]),
            )
        else:
            auc = None
        tp_050 = int((p >= 0.5).sum())
        gen_rows.append(
            {
                "generator": gname,
                "seen_in_train": seen_status,
                "n": int(len(grp)),
                "n_face": int(len(fg)),
                "no_face_rate": float(1.0 - len(fg) / len(grp)),
                "auroc_vs_real": auc,
                "recall_0_50_face": float((p >= 0.5).mean()) if len(p) else None,
                "recall_0_50_e2e": float(tp_050 / len(grp)),
                "recall_val_fpr5_face": float((p >= t_val_fpr5).mean()) if len(p) else None,
                "recall_oracle_eval_fpr5_face": float((p >= t_oracle_eval_fpr5).mean()) if len(p) else None,
                "mean_p_fake": float(p.mean()) if len(p) else None,
                "uncertain_rate": float(((p >= UNCERTAIN_LOW) & (p <= UNCERTAIN_HIGH)).mean()) if len(p) else None,
                "view_disagree_rate": float(fg["views_disagree"].mean()) if len(fg) else None,
                "c2pa_overrides": int(grp["c2pa_override"].sum()),
            }
        )

    # 3. Per-perturbation + generator x perturbation CSV rows
    orig_ref = (
        pert_df[pert_df["perturbation"] == "original"]
        .set_index("rel_id")[["p_cal_fake", "face_detected"]]
        .rename(columns={"p_cal_fake": "orig_p", "face_detected": "orig_face"})
    )
    pert_joined = pert_df.join(orig_ref, on="rel_id")

    pert_summary_rows: list[dict[str, Any]] = []
    tidy_csv_rows: list[dict[str, Any]] = []

    # Clean per-generator CSV rows (perturbation = clean_all)
    for gr in gen_rows:
        tidy_csv_rows.append(
            {
                "scope": "clean_folder",
                "perturbation": "original_full",
                "folder": gr["generator"],
                "class_name": "FAKE",
                "seen_in_train": gr["seen_in_train"],
                "n_total": gr["n"],
                "n_face": gr["n_face"],
                "no_face_rate": gr["no_face_rate"],
                "metric_type": "recall",
                "rate_0_50": gr["recall_0_50_face"],
                "rate_val_fpr5": gr["recall_val_fpr5_face"],
                "rate_oracle_eval_fpr5": gr["recall_oracle_eval_fpr5_face"],
                "auroc_vs_real": gr["auroc_vs_real"],
                "mean_p_fake": gr["mean_p_fake"],
                "uncertain_rate": gr["uncertain_rate"],
                "view_disagree_rate": gr["view_disagree_rate"],
                "c2pa_overrides": gr["c2pa_overrides"],
                "flip_rate_0_50": None,
                "mean_abs_dp": None,
                "max_abs_dp": None,
            }
        )

    for pert in PERTURBATIONS:
        sub = pert_joined[pert_joined["perturbation"] == pert]
        if len(sub) == 0:
            continue
        f_sub = sub[sub["p_cal_fake"].notna()]
        r_f = f_sub[f_sub["label"] == 0]
        k_f = f_sub[f_sub["label"] == 1]

        both_valid = f_sub[f_sub["orig_p"].notna()].copy()
        if pert == "original":
            flip_rate = 0.0
            mean_dp = 0.0
            max_dp = 0.0
        elif len(both_valid):
            dp = np.abs(both_valid["p_cal_fake"] - both_valid["orig_p"])
            flips = (both_valid["p_cal_fake"] >= 0.5) != (both_valid["orig_p"] >= 0.5)
            flip_rate = float(flips.mean())
            mean_dp = float(dp.mean())
            max_dp = float(dp.max())
        else:
            flip_rate, mean_dp, max_dp = None, None, None

        auroc = _safe_auroc(
            f_sub["label"].to_numpy(dtype=int),
            f_sub["p_cal_fake"].to_numpy(dtype=float),
        )
        p_real = r_f["p_cal_fake"].to_numpy(dtype=float)
        p_fake = k_f["p_cal_fake"].to_numpy(dtype=float)
        pert_summary_rows.append(
            {
                "perturbation": pert,
                "n_total": int(len(sub)),
                "n_resized": int(sub["actually_resized"].sum()),
                "no_face_rate": float(1.0 - len(f_sub) / len(sub)),
                "auroc": auroc,
                "real_fpr_0_50": float((p_real >= 0.5).mean()) if len(p_real) else None,
                "fake_recall_0_50": float((p_fake >= 0.5).mean()) if len(p_fake) else None,
                "real_fpr_val_fpr5": float((p_real >= t_val_fpr5).mean()) if len(p_real) else None,
                "fake_recall_val_fpr5": float((p_fake >= t_val_fpr5).mean()) if len(p_fake) else None,
                "mean_p_real": float(p_real.mean()) if len(p_real) else None,
                "mean_p_fake": float(p_fake.mean()) if len(p_fake) else None,
                "flip_rate_0_50": flip_rate,
                "mean_abs_dp": mean_dp,
                "max_abs_dp": max_dp,
            }
        )

        # Folder x Perturbation rows (real + each generator present in pert_joined)
        folders_in_pert = ["real"] + [
            g for g in discovered_generators if (sub["generator"] == g).any()
        ]
        for folder in folders_in_pert:
            fg_all = sub[sub["generator"] == folder]
            fg_ok = fg_all[fg_all["p_cal_fake"].notna()]
            is_real = folder == "real"
            pv = fg_ok["p_cal_fake"].to_numpy(dtype=float)
            bv = fg_ok[fg_ok["orig_p"].notna()]
            if pert == "original":
                fl, mdp, mxdp = 0.0, 0.0, 0.0
            elif len(bv):
                dpv = np.abs(bv["p_cal_fake"] - bv["orig_p"])
                flv = (bv["p_cal_fake"] >= 0.5) != (bv["orig_p"] >= 0.5)
                fl, mdp, mxdp = float(flv.mean()), float(dpv.mean()), float(dpv.max())
            else:
                fl, mdp, mxdp = None, None, None

            r_scores_p = r_f["p_cal_fake"].to_numpy(dtype=float)
            auc_folder = None
            if not is_real and len(pv) and len(r_scores_p):
                auc_folder = _safe_auroc(
                    np.concatenate([np.zeros(len(r_scores_p), dtype=int), np.ones(len(pv), dtype=int)]),
                    np.concatenate([r_scores_p, pv]),
                )

            tidy_csv_rows.append(
                {
                    "scope": "perturbation_folder",
                    "perturbation": pert,
                    "folder": folder,
                    "class_name": "REAL" if is_real else "FAKE",
                    "seen_in_train": "mixed" if is_real else ("seen" if folder.lower() in SEEN_BY_CURRENT_MODEL else "unseen"),
                    "n_total": int(len(fg_all)),
                    "n_face": int(len(fg_ok)),
                    "no_face_rate": float(1.0 - len(fg_ok) / max(1, len(fg_all))),
                    "metric_type": "fpr" if is_real else "recall",
                    "rate_0_50": float((pv >= 0.5).mean()) if len(pv) else None,
                    "rate_val_fpr5": float((pv >= t_val_fpr5).mean()) if len(pv) else None,
                    "rate_oracle_eval_fpr5": float((pv >= t_oracle_eval_fpr5).mean()) if len(pv) else None,
                    "auroc_vs_real": auc_folder,
                    "mean_p_fake": float(pv.mean()) if len(pv) else None,
                    "uncertain_rate": float(((pv >= UNCERTAIN_LOW) & (pv <= UNCERTAIN_HIGH)).mean()) if len(pv) else None,
                    "view_disagree_rate": float(fg_ok["views_disagree"].mean()) if len(fg_ok) else None,
                    "c2pa_overrides": int(fg_all["c2pa_override"].sum()),
                    "flip_rate_0_50": fl,
                    "mean_abs_dp": mdp,
                    "max_abs_dp": mxdp,
                }
            )

    overall: dict[str, Any] = {
        "at_0_50": _binary_summary(clean_df, 0.5),
        "at_val_fpr5": _binary_summary(clean_df, t_val_fpr5),
        "at_oracle_eval_fpr5": _binary_summary(clean_df, t_oracle_eval_fpr5),
    }
    if manual_threshold is not None:
        overall["at_manual"] = _binary_summary(clean_df, manual_threshold)

    return overall, real_sub_rows, gen_rows, pert_summary_rows, tidy_csv_rows


def _fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{100.0 * v:.{decimals}f}%"


def _fmt_num(v: Optional[float], decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{decimals}f}"


def _write_markdown_report(
    out_path: Path,
    *,
    checkpoint: Path,
    temperature: float,
    require_face: bool,
    val_real_n: int,
    t_val_fpr5: float,
    actual_val_fpr: float,
    t_oracle_eval_fpr5: float,
    actual_oracle_fpr: float,
    overall: dict[str, Any],
    real_sub_rows: list[dict[str, Any]],
    gen_rows: list[dict[str, Any]],
    pert_summary_rows: list[dict[str, Any]],
    tidy_csv_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# VeriFake Evaluation Report\n")
    lines.append(f"- Checkpoint: `{checkpoint}` (temperature `T = {temperature:.4f}`, face gate: `{require_face}`)")
    lines.append(f"- Operating thresholds:")
    lines.append(f"  - Fixed `0.5000` (existing production threshold)")
    lines.append(
        f"  - Validation `<=5% FPR` threshold: **`{t_val_fpr5:.4f}`** (calibrated strictly on `data/val_real/`, n={val_real_n}, val FPR = `{100*actual_val_fpr:.2f}%`)"
    )
    lines.append(
        f"  - Diagnostic oracle `<=5% FPR` on `data/eval/real/`: `{t_oracle_eval_fpr5:.4f}` (eval real FPR = `{100*actual_oracle_fpr:.2f}%`; **never use for model selection or tuning**)\n"
    )

    lines.append("## 1. Overall Clean Evaluation (`original`)\n")
    lines.append("| Operating Point | Threshold | n (Face) | AUROC | Accuracy | Precision | Recall (FAKE) | F1 | Real FPR | Uncertain % | View Disagree % | Confusion Matrix `[[TN, FP], [FN, TP]]` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label_key, title in [
        ("at_0_50", "Fixed 0.50"),
        ("at_val_fpr5", "Val <=5% FPR"),
        ("at_oracle_eval_fpr5", "Oracle Eval <=5% FPR (diag)"),
    ]:
        s = overall.get(label_key, {})
        if s.get("n_face", 0) == 0:
            continue
        lines.append(
            f"| {title} | {s['threshold']:.4f} | {s['n_face']}/{s['n_total']} | "
            f"{_fmt_num(s.get('auroc'))} | {_fmt_pct(s.get('accuracy'), 2)} | {_fmt_pct(s.get('precision'), 2)} | "
            f"{_fmt_pct(s.get('recall'), 2)} | {_fmt_pct(s.get('f1'), 2)} | {_fmt_pct(s.get('real_fpr'), 2)} | "
            f"{_fmt_pct(s.get('uncertain_rate'), 1)} | {_fmt_pct(s.get('view_disagree_rate'), 1)} | `{s['confusion_matrix']}` |"
        )

    lines.append("\n## 2. Real Photos Breakdown (`data/eval/real/`)\n")
    lines.append("| Source | Status | n (Face) | FPR @ 0.50 | FPR @ Val<=5% | Mean P(FAKE) | Uncertain % | View Disagree % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in real_sub_rows:
        lines.append(
            f"| `{r['subsource']}` | {r['seen_in_train']} | {r['n_face']}/{r['n']} | "
            f"{_fmt_pct(r['fpr_0_50'])} | {_fmt_pct(r['fpr_val_fpr5'])} | {_fmt_num(r['mean_p_fake'], 3)} | "
            f"{_fmt_pct(r['uncertain_rate'])} | {_fmt_pct(r['view_disagree_rate'])} |"
        )

    lines.append("\n## 3. Per-Generator Detection (`data/eval/<generator>/`)\n")
    lines.append("| Generator | Status | n (Face) | No-Face % | AUROC vs Real | Recall @ 0.50 (Face) | Recall @ 0.50 (E2E) | Recall @ Val<=5% | Recall @ Oracle 5% | Mean P(FAKE) | Uncertain % | View Disagree % | C2PA Overrides |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in gen_rows:
        lines.append(
            f"| `{g['generator']}` | {g['seen_in_train']} | {g['n_face']}/{g['n']} | "
            f"{_fmt_pct(g['no_face_rate'])} | {_fmt_num(g['auroc_vs_real'])} | "
            f"{_fmt_pct(g['recall_0_50_face'])} | {_fmt_pct(g['recall_0_50_e2e'])} | "
            f"{_fmt_pct(g['recall_val_fpr5_face'])} | {_fmt_pct(g['recall_oracle_eval_fpr5_face'])} | "
            f"{_fmt_num(g['mean_p_fake'], 3)} | {_fmt_pct(g['uncertain_rate'])} | "
            f"{_fmt_pct(g['view_disagree_rate'])} | {g['c2pa_overrides']} |"
        )

    if pert_summary_rows:
        lines.append("\n## 4. Perturbation Suite Summary\n")
        lines.append("| Perturbation | n | Resized | No-Face % | AUROC | Real FPR @ 0.50 | Fake Recall @ 0.50 | Real FPR @ Val<=5% | Fake Recall @ Val<=5% | Mean P Real | Mean P Fake | Flip % vs Orig | Mean \\|ΔP\\| | Max \\|ΔP\\| |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for pr in pert_summary_rows:
            lines.append(
                f"| `{pr['perturbation']}` | {pr['n_total']} | {pr['n_resized']} | {_fmt_pct(pr['no_face_rate'])} | "
                f"{_fmt_num(pr['auroc'])} | {_fmt_pct(pr['real_fpr_0_50'])} | {_fmt_pct(pr['fake_recall_0_50'])} | "
                f"{_fmt_pct(pr['real_fpr_val_fpr5'])} | {_fmt_pct(pr['fake_recall_val_fpr5'])} | "
                f"{_fmt_num(pr['mean_p_real'], 3)} | {_fmt_num(pr['mean_p_fake'], 3)} | "
                f"{_fmt_pct(pr['flip_rate_0_50'])} | {_fmt_num(pr['mean_abs_dp'], 3)} | {_fmt_num(pr['max_abs_dp'], 3)} |"
            )

        lines.append("\n## 5. Folder × Perturbation Matrix (`@ 0.50` / `@ Val<=5%`)\n")
        lines.append("Values show **FPR** for `real` and **Recall** for AI generator folders.\n")
        pert_rows = [r for r in tidy_csv_rows if r["scope"] == "perturbation_folder"]
        folders = []
        for r in pert_rows:
            if r["folder"] not in folders:
                folders.append(r["folder"])
        lookup = {(r["folder"], r["perturbation"]): r for r in pert_rows}
        perts_present = [p for p in PERTURBATIONS if any((f, p) in lookup for f in folders)]
        lines.append("| Folder (Metric) | " + " | ".join(f"`{p}`" for p in perts_present) + " |")
        lines.append("|---|" + "|".join("---:" for _ in perts_present) + "|")
        for f in folders:
            mtype = "FPR" if f == "real" else "Rec"
            cells = []
            for p in perts_present:
                row = lookup.get((f, p))
                if not row or row["rate_0_50"] is None:
                    cells.append("-")
                else:
                    cells.append(f"{100*row['rate_0_50']:.0f}% / {100*row['rate_val_fpr5']:.0f}%")
            lines.append(f"| `{f}` ({mtype}) | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _run_eval_mode(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = None if args.device == "auto" else torch.device(args.device)
    require_face = not args.no_face_gate

    detector = DeepfakeDetector(checkpoint=args.checkpoint, device=device)
    discovered_gens, clean_samples, perturb_samples = discover_eval_dir(
        args.eval_dir,
        max_per_folder=args.max_per_folder,
        max_real_per_subsource=args.max_real_per_subsource,
        perturb_max_per_folder=args.perturb_max_per_folder,
        perturb_max_real_per_subsource=args.perturb_max_real_per_subsource,
        seed=args.seed,
    )
    val_real_samples, _ = discover_real_folder(
        args.val_real_dir,
        max_per_subsource=args.val_real_max_per_subsource,
        perturb_max_per_subsource=0,
        seed=args.seed + 500,
    )
    logger.info(
        "Discovered %d generator folder(s) (%s) | clean eval=%d | perturb subset=%d | val_real=%d",
        len(discovered_gens),
        ", ".join(discovered_gens),
        len(clean_samples),
        len(perturb_samples),
        len(val_real_samples),
    )

    pixel_cache: dict[str, CachedPixelResult] = {}

    t0 = time.time()
    val_real_records = score_samples_with_perturbations(
        detector,
        val_real_samples,
        perturbations=("original",),
        require_face=require_face,
        batch_size=args.batch_size,
        pixel_cache=pixel_cache,
    )
    val_real_scores = np.array(
        [r["p_cal_fake"] for r in val_real_records if r["p_cal_fake"] is not None],
        dtype=float,
    )
    if len(val_real_scores) > 0:
        t_val_fpr5, actual_val_fpr = compute_fpr_threshold(val_real_scores, args.target_fpr)
    else:
        t_val_fpr5, actual_val_fpr = 0.5, 0.0
        logger.warning("No val_real images found at %s; falling back t_val_fpr5=0.5", args.val_real_dir)

    clean_records = score_samples_with_perturbations(
        detector,
        clean_samples,
        perturbations=("original",),
        require_face=require_face,
        batch_size=args.batch_size,
        pixel_cache=pixel_cache,
        verify_first_parity=True,
    )
    clean_df = pd.DataFrame(clean_records)
    orig_map = {r["rel_id"]: r for r in clean_records}

    eval_real_scores = clean_df.loc[
        (clean_df["label"] == 0) & (clean_df["p_cal_fake"].notna()), "p_cal_fake"
    ].to_numpy(dtype=float)
    t_oracle_eval_fpr5, actual_oracle_fpr = compute_fpr_threshold(
        eval_real_scores, args.target_fpr
    )

    if args.skip_perturbations or not perturb_samples:
        pert_records = clean_records
    else:
        pert_records = score_samples_with_perturbations(
            detector,
            perturb_samples,
            perturbations=PERTURBATIONS,
            require_face=require_face,
            batch_size=args.batch_size,
            pixel_cache=pixel_cache,
            existing_orig_records=orig_map,
        )
    pert_df = pd.DataFrame(pert_records)

    overall, real_sub_rows, gen_rows, pert_summary_rows, tidy_csv_rows = _build_eval_tables(
        clean_df,
        pert_df,
        discovered_gens,
        t_val_fpr5=t_val_fpr5,
        t_oracle_eval_fpr5=t_oracle_eval_fpr5,
        manual_threshold=args.threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "baseline_report.md"
    csv_path = args.out_dir / "per_generator_perturbation.csv"
    img_csv_path = args.out_dir / "per_image_scores.csv"
    json_path = args.out_dir / "summary.json"

    _write_markdown_report(
        report_path,
        checkpoint=args.checkpoint,
        temperature=detector.temperature,
        require_face=require_face,
        val_real_n=len(val_real_scores),
        t_val_fpr5=t_val_fpr5,
        actual_val_fpr=actual_val_fpr,
        t_oracle_eval_fpr5=t_oracle_eval_fpr5,
        actual_oracle_fpr=actual_oracle_fpr,
        overall=overall,
        real_sub_rows=real_sub_rows,
        gen_rows=gen_rows,
        pert_summary_rows=[] if args.skip_perturbations else pert_summary_rows,
        tidy_csv_rows=tidy_csv_rows,
    )

    if tidy_csv_rows:
        _write_csv(csv_path, tidy_csv_rows, list(tidy_csv_rows[0].keys()))

    all_img_rows = clean_records if args.skip_perturbations else pert_records
    if all_img_rows:
        cols = [
            "rel_id",
            "class_name",
            "label",
            "generator",
            "subsource",
            "perturbation",
            "width",
            "height",
            "actually_resized",
            "face_detected",
            "c2pa_override",
            "p_uncal_fake",
            "p_cal_fake",
            "crop_delta",
            "full_delta",
            "views_disagree",
        ]
        _write_csv(img_csv_path, [{c: r.get(c) for c in cols} for r in all_img_rows], cols)

    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "temperature": detector.temperature,
        "require_face": require_face,
        "target_fpr": args.target_fpr,
        "val_real_n": int(len(val_real_scores)),
        "threshold_val_fpr5": t_val_fpr5,
        "val_real_actual_fpr": actual_val_fpr,
        "threshold_oracle_eval_fpr5": t_oracle_eval_fpr5,
        "oracle_eval_real_actual_fpr": actual_oracle_fpr,
        "overall": overall,
        "real_subsources": real_sub_rows,
        "per_generator": gen_rows,
        "perturbations": [] if args.skip_perturbations else pert_summary_rows,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save_json(summary_payload, json_path)

    logger.info("Evaluation finished in %.1fs", summary_payload["elapsed_seconds"])
    logger.info("Report saved: %s", report_path)
    logger.info("CSV saved   : %s", csv_path)


def _run_logo_mode(args: argparse.Namespace) -> None:
    from src.data import balanced_face_subset, derive_class_to_idx_from_manifest, load_manifest, make_tta_loader
    from src.evaluate import run_tta_logits
    from src.model import load_checkpoint
    from src.train import fit
    from src.calibration import logits_to_p_fake

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    df = load_manifest(multi=True) if args.manifest == DATA_DIR / "multi_manifest.csv" else pd.read_csv(args.manifest)
    if "has_face" in df.columns:
        df = df[df["has_face"] == True].copy()  # noqa: E712

    class_to_idx = derive_class_to_idx_from_manifest(df)
    pos_idx = class_to_idx["FAKE"]
    neg_idx = class_to_idx["REAL"]
    excluded = set(args.exclude_generators or [])

    test_real_df = df[(df["split"] == "test") & (df["label"] == neg_idx)].reset_index(drop=True)
    fold_reports: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = MODELS_DIR / "logo_eval_tmp.pt"

    for idx, held_out in enumerate(args.logo_generators, start=1):
        fold_excl = excluded | {held_out}
        fold_train_pool = df[(df["split"] == "train") & (~df["generator"].isin(fold_excl))]
        fold_val_pool = df[(df["split"] == "val") & (~df["generator"].isin(fold_excl))]
        held_test_df = df[
            df["split"].isin(["test", "unseen_generator"]) & (df["generator"] == held_out)
        ].reset_index(drop=True)

        # Verify zero hash/pixel leakage between train/val and held-out test
        tv_sha = set(fold_train_pool["sha256"]) | set(fold_val_pool["sha256"])
        tv_pix = set(fold_train_pool["pixel_md5"]) | set(fold_val_pool["pixel_md5"])
        test_sha = set(held_test_df["sha256"]) | set(test_real_df["sha256"])
        test_pix = set(held_test_df["pixel_md5"]) | set(test_real_df["pixel_md5"])
        if tv_sha & test_sha or tv_pix & test_pix:
            raise RuntimeError(f"LOGO fold {held_out}: hash/pixel leakage detected!")
        if held_out in set(fold_train_pool["generator"]) | set(fold_val_pool["generator"]):
            raise RuntimeError(f"LOGO fold {held_out}: held-out generator leaked into train/val!")

        fold_train = balanced_face_subset(fold_train_pool, args.logo_train_per_fold, seed=args.seed + idx)
        fold_val = balanced_face_subset(fold_val_pool, args.logo_val_per_fold, seed=args.seed + idx)
        logger.info(
            "LOGO fold %d/%d [%s]: train=%d (gens=%s) | val=%d | held_test=%d | test_real=%d",
            idx,
            len(args.logo_generators),
            held_out,
            len(fold_train),
            sorted(fold_train["generator"].unique().tolist()),
            len(fold_val),
            len(held_test_df),
            len(test_real_df),
        )
        if args.logo_dry_run:
            fold_reports.append(
                {
                    "generator": held_out,
                    "dry_run": True,
                    "train_n": int(len(fold_train)),
                    "val_n": int(len(fold_val)),
                    "held_out_n": int(len(held_test_df)),
                    "test_real_n": int(len(test_real_df)),
                }
            )
            continue

        save_class_mapping(class_to_idx)
        if tmp_ckpt.exists():
            tmp_ckpt.unlink()
        t0 = time.time()
        fit_res = fit(
            "convnext_tiny",
            fold_train,
            fold_val,
            class_to_idx=class_to_idx,
            epochs=args.logo_epochs,
            batch_size=args.batch_size,
            lr=args.logo_lr,
            weight_decay=0.05,
            img_size=224,
            label_smoothing=0.1,
            patience=99,
            workers=0,
            device=device,
            save_path=tmp_ckpt,
            resume_checkpoint=None,
        )
        fold_model, _ = load_checkpoint(tmp_ckpt, device)

        val_real_only = fold_val[fold_val["label"] == neg_idx].reset_index(drop=True)
        vr_loader = make_tta_loader(val_real_only, 224, batch_size=max(8, args.batch_size // 2), workers=0, class_to_idx=class_to_idx)
        hk_loader = make_tta_loader(held_test_df, 224, batch_size=max(8, args.batch_size // 2), workers=0, class_to_idx=class_to_idx)
        tr_loader = make_tta_loader(test_real_df, 224, batch_size=max(8, args.batch_size // 2), workers=0, class_to_idx=class_to_idx)

        vr_logits, _ = run_tta_logits(fold_model, vr_loader, device, amp=fit_res["amp"])
        hk_logits, _ = run_tta_logits(fold_model, hk_loader, device, amp=fit_res["amp"])
        tr_logits, _ = run_tta_logits(fold_model, tr_loader, device, amp=fit_res["amp"])

        p_val_real = logits_to_p_fake(vr_logits, fake_idx=pos_idx, temperature=1.0)
        p_held = logits_to_p_fake(hk_logits, fake_idx=pos_idx, temperature=1.0)
        p_test_real = logits_to_p_fake(tr_logits, fake_idx=pos_idx, temperature=1.0)

        t_fpr5, _ = compute_fpr_threshold(p_val_real, args.target_fpr)
        rec_050 = float((p_held >= 0.5).mean())
        rec_fpr5 = float((p_held >= t_fpr5).mean())
        fpr_test_050 = float((p_test_real >= 0.5).mean())
        fpr_test_fpr5 = float((p_test_real >= t_fpr5).mean())
        auc = _safe_auroc(
            np.concatenate([np.zeros(len(p_test_real), dtype=int), np.ones(len(p_held), dtype=int)]),
            np.concatenate([p_test_real, p_held]),
        )
        fold_reports.append(
            {
                "generator": held_out,
                "train_n": int(len(fold_train)),
                "val_n": int(len(fold_val)),
                "held_out_n": int(len(held_test_df)),
                "threshold_val_fpr5": t_fpr5,
                "recall_0_50": rec_050,
                "recall_val_fpr5": rec_fpr5,
                "test_real_fpr_0_50": fpr_test_050,
                "test_real_fpr_val_fpr5": fpr_test_fpr5,
                "auroc_vs_test_real": auc,
                "seconds": round(time.time() - t0, 1),
            }
        )
        del fold_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    if args.logo_dry_run:
        logger.info("LOGO dry-run passed across %d folds with zero leakage.", len(fold_reports))
        return

    _write_csv(args.out_dir / "logo_results.csv", fold_reports, list(fold_reports[0].keys()))
    lines = ["# Leave-One-Generator-Out (LOGO) Report\n", "| Held-Out Generator | n | Val<=5% Thresh | Recall @ 0.50 | Recall @ Val<=5% | Test Real FPR @ Val<=5% | AUROC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fr in fold_reports:
        lines.append(
            f"| `{fr['generator']}` | {fr['held_out_n']} | {fr['threshold_val_fpr5']:.4f} | "
            f"{_fmt_pct(fr['recall_0_50'])} | {_fmt_pct(fr['recall_val_fpr5'])} | "
            f"{_fmt_pct(fr['test_real_fpr_val_fpr5'])} | {_fmt_num(fr['auroc_vs_test_real'])} |"
        )
    (args.out_dir / "logo_report.md").write_text("\n".join(lines) + "\n")
    save_json({"folds": fold_reports}, args.out_dir / "logo_results.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate VeriFake across generator folders and perturbations.")
    p.add_argument("--mode", choices=["eval", "logo"], default="eval")
    p.add_argument("--eval-dir", type=Path, default=DATA_DIR / "eval")
    p.add_argument("--val-real-dir", type=Path, default=DATA_DIR / "val_real")
    p.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "eval")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--max-per-folder", type=int, default=0, help="0 = all images per generator folder")
    p.add_argument("--max-real-per-subsource", type=int, default=0, help="0 = all images per real subsource")
    p.add_argument("--perturb-max-per-folder", type=int, default=0, help="0 = all clean-eval images per generator")
    p.add_argument("--perturb-max-real-per-subsource", type=int, default=0, help="0 = all clean-eval images per real subsource")
    p.add_argument("--val-real-max-per-subsource", type=int, default=0, help="0 = all val_real images per subsource")
    p.add_argument("--skip-perturbations", action="store_true")
    p.add_argument("--no-face-gate", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)

    # LOGO mode options
    p.add_argument("--manifest", type=Path, default=DATA_DIR / "multi_manifest.csv")
    p.add_argument("--logo-generators", nargs="+", default=["DALLE3", "FLUX", "SDXL"])
    p.add_argument("--exclude-generators", nargs="*", default=["Midjourney"])
    p.add_argument("--logo-train-per-fold", type=int, default=1200)
    p.add_argument("--logo-val-per-fold", type=int, default=400)
    p.add_argument("--logo-epochs", type=int, default=2)
    p.add_argument("--logo-lr", type=float, default=1.5e-4)
    p.add_argument("--logo-dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "logo":
        _run_logo_mode(args)
    else:
        _run_eval_mode(args)


if __name__ == "__main__":
    main()
