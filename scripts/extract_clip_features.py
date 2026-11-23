"""Extract and cache frozen CLIP ViT-L/14 embeddings across train (augmented), val, and held-out test."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.augmentations import (  # noqa: E402
    STEP3_CACHE_PERTURBATIONS,
    apply_eval_perturbation,
    build_clip_transforms,
)
from src.clip_detector import CLIP_BACKBONE, CLIPFeatureExtractor  # noqa: E402
from src.face_crop import crop_face_or_full, gate_and_crop_face  # noqa: E402
from src.image_io import register_image_plugins, to_rgb_safe  # noqa: E402
from src.utils import DATA_DIR, SEED, get_device, get_logger, seed_everything  # noqa: E402

logger = get_logger("extract_clip")

DEFAULT_MANIFEST: Path = DATA_DIR / "prepared_manifest.csv"
DEFAULT_CACHE_DIR: Path = DATA_DIR / "cache" / "clip_vit_l14"


def _load_rgb(rel_or_abs: str) -> Image.Image:
    p = Path(rel_or_abs)
    abs_p = p if p.is_absolute() else (PROJECT_ROOT / p)
    with Image.open(abs_p) as im:
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
        return to_rgb_safe(ImageOps.exif_transpose(im))


def _encode_tensors(
    extractor: CLIPFeatureExtractor,
    tensors: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size], dim=0).to(device, non_blocking=True)
        feats = extractor.extract_features(batch, use_fp16=use_fp16).cpu().half()
        out.append(feats)
    return torch.cat(out, dim=0) if out else torch.empty((0, extractor.embed_dim), dtype=torch.float16)


def _extract_train_cache(
    df_train: pd.DataFrame,
    extractor: CLIPFeatureExtractor,
    out_path: Path,
    *,
    aug_passes: int,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> None:
    if out_path.exists():
        logger.info("Using existing train CLIP cache: %s", out_path)
        return

    clean_tf = build_clip_transforms(train=False, img_size=224)
    aug_tf = build_clip_transforms(train=True, img_size=224)
    n = len(df_train)
    embed_dim = extractor.embed_dim
    aug_feats = torch.empty((n, aug_passes, embed_dim), dtype=torch.float16)
    full_feats = torch.empty((n, embed_dim), dtype=torch.float16)

    t0 = time.time()
    buf_tensors: list[torch.Tensor] = []
    # For each image store (aug_passes + 1) tensors: [pass_0(face), full_clean, pass_1..K-1]
    slots_per_img = aug_passes + 1
    chunk_start = 0
    next_log_at = 500

    def _flush(start_idx: int, count: int) -> None:
        if not buf_tensors:
            return
        feats = _encode_tensors(
            extractor, buf_tensors, batch_size=batch_size, device=device, use_fp16=use_fp16
        )
        feats = feats.view(count, slots_per_img, embed_dim)
        aug_feats[start_idx : start_idx + count, 0] = feats[:, 0]
        full_feats[start_idx : start_idx + count] = feats[:, 1]
        if aug_passes > 1:
            aug_feats[start_idx : start_idx + count, 1:] = feats[:, 2:]
        buf_tensors.clear()

    chunk_count = 0
    for idx, row in enumerate(df_train.itertuples(index=False)):
        rgb = _load_rgb(str(row.path))
        cropped, _ = crop_face_or_full(rgb)
        buf_tensors.append(clean_tf(cropped))
        buf_tensors.append(clean_tf(rgb))
        for p_idx in range(1, aug_passes):
            # Mix cropped face (70%) and full scene (30%) for augmented views
            src_img = cropped if (p_idx == 1 or (idx + p_idx) % 10 < 7) else rgb
            buf_tensors.append(aug_tf(src_img))
        chunk_count += 1
        if len(buf_tensors) >= batch_size * 4:
            _flush(chunk_start, chunk_count)
            chunk_start += chunk_count
            chunk_count = 0
            if (idx + 1) >= next_log_at or (idx + 1) == n:
                dt = time.time() - t0
                logger.info(
                    "  [train] %d/%d images (%.1f img/s)",
                    idx + 1,
                    n,
                    (idx + 1) / max(0.1, dt),
                )
                while next_log_at <= idx + 1:
                    next_log_at += 500

    _flush(chunk_start, chunk_count)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": extractor.backbone_name,
            "sha256": df_train["sha256"].astype(str).tolist(),
            "labels": torch.tensor(df_train["label"].to_numpy(dtype="int64")),
            "generator_family": df_train["generator_family"].astype(str).tolist(),
            "generator": df_train["generator"].astype(str).tolist(),
            "aug_feats": aug_feats,
            "clean_crop_feats": aug_feats[:, 0],
            "clean_full_feats": full_feats,
        },
        out_path,
    )
    logger.info("Saved train CLIP cache (%d rows, %d passes) in %.1fs -> %s", n, aug_passes, time.time() - t0, out_path)


def _extract_eval_cache(
    df_split: pd.DataFrame,
    split_name: str,
    extractor: CLIPFeatureExtractor,
    out_path: Path,
    *,
    perturbations: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
    pert_sha_subset: Optional[set[str]] = None,
) -> None:
    if out_path.exists():
        logger.info("Using existing %s CLIP cache: %s", split_name, out_path)
        return

    clean_tf = build_clip_transforms(train=False, img_size=224)
    n = len(df_split)
    n_perts = len(perturbations)
    embed_dim = extractor.embed_dim

    crop_feats = torch.zeros((n, n_perts, embed_dim), dtype=torch.float16)
    full_feats = torch.zeros((n, n_perts, embed_dim), dtype=torch.float16)
    face_ok = torch.zeros((n, n_perts), dtype=torch.bool)

    t0 = time.time()
    next_log_at = 250
    pending_tensors: list[torch.Tensor] = []
    pending_refs: list[tuple[int, int, int]] = []  # (row_i, pert_j, slot_0crop_1full)
    pix_to_result: dict[str, tuple[bool, int]] = {}  # per-image dedup key -> tensor idx if encoded

    def _flush_pending() -> None:
        if not pending_tensors:
            return
        feats = _encode_tensors(
            extractor, pending_tensors, batch_size=batch_size, device=device, use_fp16=use_fp16
        )
        for k, (ri, pj, slot) in enumerate(pending_refs):
            if slot == 0:
                crop_feats[ri, pj] = feats[k]
            else:
                full_feats[ri, pj] = feats[k]
        pending_tensors.clear()
        pending_refs.clear()

    # Keep exact pixel dedup per image so no-op downscales reuse identical tensor outputs
    for ri, row in enumerate(df_split.itertuples(index=False)):
        orig_rgb = _load_rgb(str(row.path))
        img_cache: dict[str, tuple[int, bool]] = {}
        for pj, pert_name in enumerate(perturbations):
            if pert_name != "original" and pert_sha_subset is not None and str(row.sha256) not in pert_sha_subset:
                continue
            v_rgb = apply_eval_perturbation(orig_rgb, pert_name)
            pmd5 = hashlib.md5(v_rgb.tobytes()).hexdigest()
            if pmd5 in img_cache:
                prev_j, prev_ok = img_cache[pmd5]
                face_ok[ri, pj] = prev_ok
                # Note: wait until pending flushed or copy after loop -> track link
                if prev_ok and not pending_refs:
                    crop_feats[ri, pj] = crop_feats[ri, prev_j]
                    full_feats[ri, pj] = full_feats[ri, prev_j]
                else:
                    pix_to_result[f"{ri}:{pj}"] = (prev_ok, prev_j)
                continue

            cropped, _ = gate_and_crop_face(v_rgb)
            if cropped is None:
                face_ok[ri, pj] = False
                img_cache[pmd5] = (pj, False)
                continue

            face_ok[ri, pj] = True
            img_cache[pmd5] = (pj, True)
            pending_tensors.append(clean_tf(cropped))
            pending_refs.append((ri, pj, 0))
            pending_tensors.append(clean_tf(v_rgb))
            pending_refs.append((ri, pj, 1))

        if len(pending_tensors) >= batch_size * 4:
            _flush_pending()
            if (ri + 1) >= next_log_at or (ri + 1) == n:
                dt = time.time() - t0
                logger.info(
                    "  [%s] %d/%d images (%.1f img/s)",
                    split_name,
                    ri + 1,
                    n,
                    (ri + 1) / max(0.1, dt),
                )
                while next_log_at <= ri + 1:
                    next_log_at += 250

    _flush_pending()
    # Apply deferred per-image pixel-cache copies
    for key, (ok_flag, src_pj) in pix_to_result.items():
        ri_s, pj_s = key.split(":")
        ri, pj = int(ri_s), int(pj_s)
        face_ok[ri, pj] = ok_flag
        if ok_flag:
            crop_feats[ri, pj] = crop_feats[ri, src_pj]
            full_feats[ri, pj] = full_feats[ri, src_pj]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": extractor.backbone_name,
            "perturbations": list(perturbations),
            "sha256": df_split["sha256"].astype(str).tolist(),
            "labels": torch.tensor(df_split["label"].to_numpy(dtype="int64")),
            "split_role": df_split["split_role"].astype(str).tolist(),
            "generator_family": df_split["generator_family"].astype(str).tolist(),
            "generator": df_split["generator"].astype(str).tolist(),
            "crop_feats": crop_feats,
            "full_feats": full_feats,
            "face_detected": face_ok,
            "pert_sha256_subset": sorted(pert_sha_subset) if pert_sha_subset is not None else None,
        },
        out_path,
    )
    logger.info(
        "Saved %s CLIP cache (%d rows x %d perts) in %.1fs -> %s",
        split_name,
        n,
        n_perts,
        time.time() - t0,
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache frozen CLIP ViT-L/14 features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--backbone", type=str, default=CLIP_BACKBONE)
    parser.add_argument("--train-aug-passes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--test-pert-per-source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    register_image_plugins()
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    extractor = CLIPFeatureExtractor(backbone=args.backbone, pretrained=True, img_size=224).to(device)

    use_fp16 = (not args.no_fp16) and (device.type in ("mps", "cuda"))
    if use_fp16:
        test_x = torch.randn(2, 3, 224, 224, device=device)
        f32 = extractor.extract_features(test_x, use_fp16=False)
        f16 = extractor.extract_features(test_x, use_fp16=True)
        diff = float((f32 - f16).abs().max().item())
        logger.info("CLIP %s fp16 parity check max_diff=%.2e", device.type, diff)
        if not torch.isfinite(f16).all() or diff > 0.05:
            logger.warning("Disabling fp16 due to numerical drift; using fp32")
            use_fp16 = False

    if args.download_only:
        logger.info("CLIP download/init complete (device=%s, embed_dim=%d).", device, extractor.embed_dim)
        return

    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    train_df = face_df[face_df["split"] == "train"].reset_index(drop=True)
    val_df = face_df[face_df["split"] == "val"].reset_index(drop=True)
    test_df = face_df[face_df["split"] == "held_out_test"].reset_index(drop=True)

    _extract_train_cache(
        train_df,
        extractor,
        args.cache_dir / "train_aug.pt",
        aug_passes=args.train_aug_passes,
        batch_size=args.batch_size,
        device=device,
        use_fp16=use_fp16,
    )
    _extract_eval_cache(
        val_df,
        "val",
        extractor,
        args.cache_dir / "val_eval.pt",
        perturbations=STEP3_CACHE_PERTURBATIONS,
        batch_size=args.batch_size,
        device=device,
        use_fp16=use_fp16,
    )
    test_pert_shas: Optional[set[str]] = None
    if args.test_pert_per_source > 0:
        pert_parts = [
            grp.sample(n=min(args.test_pert_per_source, len(grp)), random_state=args.seed + 11)
            for _, grp in test_df.groupby("generator")
        ]
        test_pert_shas = set(pd.concat(pert_parts, ignore_index=True)["sha256"].astype(str))

    _extract_eval_cache(
        test_df,
        "held_out_test",
        extractor,
        args.cache_dir / "held_out_test_eval.pt",
        perturbations=STEP3_CACHE_PERTURBATIONS,
        batch_size=args.batch_size,
        device=device,
        use_fp16=use_fp16,
        pert_sha_subset=test_pert_shas,
    )


if __name__ == "__main__":
    main()
