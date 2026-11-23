"""Incrementally extend cached CLIP ViT-L/14 embeddings to prepared_manifest_v2.csv without recomputing old rows."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.extract_clip_features import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    _encode_tensors,
    _load_rgb,
)
from src.augmentations import (  # noqa: E402
    STEP3_CACHE_PERTURBATIONS,
    apply_eval_perturbation,
    build_clip_transforms,
)
from src.clip_detector import CLIP_BACKBONE, CLIPFeatureExtractor  # noqa: E402
from src.face_crop import crop_face_or_full, gate_and_crop_face  # noqa: E402
from src.image_io import register_image_plugins  # noqa: E402
from src.utils import DATA_DIR, SEED, get_device, get_logger, seed_everything  # noqa: E402

logger = get_logger("update_clip_cache_v2")


def _build_train_v2(
    train_df: pd.DataFrame,
    old_cache: dict[str, Any],
    extractor: CLIPFeatureExtractor,
    out_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> None:
    if out_path.exists():
        logger.info("Using existing v2 train CLIP cache: %s", out_path)
        return

    old_sha_to_idx = {str(s): i for i, s in enumerate(old_cache["sha256"])}
    n = len(train_df)
    aug_passes = int(old_cache["aug_feats"].shape[1])
    embed_dim = int(old_cache["aug_feats"].shape[2])

    aug_feats = torch.empty((n, aug_passes, embed_dim), dtype=torch.float16)
    full_feats = torch.empty((n, embed_dim), dtype=torch.float16)

    new_indices: list[tuple[int, str]] = []
    reused = 0
    for row_i, row in enumerate(train_df.itertuples(index=False)):
        sha = str(row.sha256)
        if sha in old_sha_to_idx:
            oi = old_sha_to_idx[sha]
            aug_feats[row_i] = old_cache["aug_feats"][oi]
            full_feats[row_i] = old_cache["clean_full_feats"][oi]
            reused += 1
        else:
            new_indices.append((row_i, str(row.path)))

    logger.info("v2 train cache: reusing %d rows, extracting %d new rows", reused, len(new_indices))
    clean_tf = build_clip_transforms(train=False, img_size=224)
    aug_tf = build_clip_transforms(train=True, img_size=224)
    slots_per_img = aug_passes + 1

    t0 = time.time()
    buf_tensors: list[torch.Tensor] = []
    buf_row_idxs: list[int] = []

    def _flush() -> None:
        if not buf_tensors:
            return
        feats = _encode_tensors(
            extractor, buf_tensors, batch_size=batch_size, device=device, use_fp16=use_fp16
        ).view(len(buf_row_idxs), slots_per_img, embed_dim)
        for k, target_idx in enumerate(buf_row_idxs):
            aug_feats[target_idx, 0] = feats[k, 0]
            full_feats[target_idx] = feats[k, 1]
            if aug_passes > 1:
                aug_feats[target_idx, 1:] = feats[k, 2:]
        buf_tensors.clear()
        buf_row_idxs.clear()

    for j, (row_i, path_str) in enumerate(new_indices):
        rgb = _load_rgb(path_str)
        cropped, _ = crop_face_or_full(rgb)
        buf_tensors.append(clean_tf(cropped))
        buf_tensors.append(clean_tf(rgb))
        for p_idx in range(1, aug_passes):
            src_img = cropped if (p_idx == 1 or (j + p_idx) % 10 < 7) else rgb
            buf_tensors.append(aug_tf(src_img))
        buf_row_idxs.append(row_i)
        if len(buf_tensors) >= batch_size * 4:
            _flush()
            if (j + 1) % 250 == 0 or (j + 1) == len(new_indices):
                logger.info("  [train v2] %d/%d new images (%.1fs)", j + 1, len(new_indices), time.time() - t0)

    _flush()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": extractor.backbone_name,
            "sha256": train_df["sha256"].astype(str).tolist(),
            "labels": torch.tensor(train_df["label"].to_numpy(dtype="int64")),
            "generator_family": train_df["generator_family"].astype(str).tolist(),
            "generator": train_df["generator"].astype(str).tolist(),
            "aug_feats": aug_feats,
            "clean_crop_feats": aug_feats[:, 0],
            "clean_full_feats": full_feats,
        },
        out_path,
    )
    logger.info("Saved v2 train CLIP cache (%d total rows) in %.1fs -> %s", n, time.time() - t0, out_path)


def _build_val_v2(
    val_df: pd.DataFrame,
    old_cache: dict[str, Any],
    extractor: CLIPFeatureExtractor,
    out_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> None:
    if out_path.exists():
        logger.info("Using existing v2 val CLIP cache: %s", out_path)
        return

    old_sha_to_idx = {str(s): i for i, s in enumerate(old_cache["sha256"])}
    perts = tuple(old_cache["perturbations"])
    n = len(val_df)
    n_perts = len(perts)
    embed_dim = int(old_cache["crop_feats"].shape[2])

    crop_feats = torch.zeros((n, n_perts, embed_dim), dtype=torch.float16)
    full_feats = torch.zeros((n, n_perts, embed_dim), dtype=torch.float16)
    face_ok = torch.zeros((n, n_perts), dtype=torch.bool)

    new_indices: list[tuple[int, str]] = []
    reused = 0
    for row_i, row in enumerate(val_df.itertuples(index=False)):
        sha = str(row.sha256)
        if sha in old_sha_to_idx:
            oi = old_sha_to_idx[sha]
            crop_feats[row_i] = old_cache["crop_feats"][oi]
            full_feats[row_i] = old_cache["full_feats"][oi]
            face_ok[row_i] = old_cache["face_detected"][oi]
            reused += 1
        else:
            new_indices.append((row_i, str(row.path)))

    logger.info("v2 val cache: reusing %d rows, extracting %d new rows", reused, len(new_indices))
    clean_tf = build_clip_transforms(train=False, img_size=224)
    pending_tensors: list[torch.Tensor] = []
    pending_refs: list[tuple[int, int, int]] = []
    pix_to_result: dict[str, tuple[bool, int]] = {}

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

    t0 = time.time()
    for ri, path_str in new_indices:
        orig_rgb = _load_rgb(path_str)
        img_cache: dict[str, tuple[int, bool]] = {}
        for pj, pert_name in enumerate(perts):
            v_rgb = apply_eval_perturbation(orig_rgb, pert_name)
            pmd5 = hashlib.md5(v_rgb.tobytes()).hexdigest()
            if pmd5 in img_cache:
                prev_j, prev_ok = img_cache[pmd5]
                face_ok[ri, pj] = prev_ok
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

    _flush_pending()
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
            "perturbations": list(perts),
            "sha256": val_df["sha256"].astype(str).tolist(),
            "labels": torch.tensor(val_df["label"].to_numpy(dtype="int64")),
            "split_role": val_df["split_role"].astype(str).tolist(),
            "generator_family": val_df["generator_family"].astype(str).tolist(),
            "generator": val_df["generator"].astype(str).tolist(),
            "crop_feats": crop_feats,
            "full_feats": full_feats,
            "face_detected": face_ok,
        },
        out_path,
    )
    logger.info("Saved v2 val CLIP cache (%d total rows) in %.1fs -> %s", n, time.time() - t0, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally extend CLIP ViT-L/14 cache for prepared_manifest_v2.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "prepared_manifest_v2.csv")
    parser.add_argument("--base-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-cache-dir", type=Path, default=DATA_DIR / "cache" / "clip_vit_l14_v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    register_image_plugins()
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    extractor = CLIPFeatureExtractor(backbone=CLIP_BACKBONE, pretrained=True, img_size=224).to(device)
    use_fp16 = device.type in ("mps", "cuda")

    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    train_df = face_df[face_df["split"] == "train"].reset_index(drop=True)
    val_df = face_df[face_df["split"] == "val"].reset_index(drop=True)

    old_train = torch.load(args.base_cache_dir / "train_aug.pt", map_location="cpu", weights_only=False)
    old_val = torch.load(args.base_cache_dir / "val_eval.pt", map_location="cpu", weights_only=False)

    _build_train_v2(
        train_df, old_train, extractor, args.out_cache_dir / "train_aug.pt",
        batch_size=args.batch_size, device=device, use_fp16=use_fp16,
    )
    _build_val_v2(
        val_df, old_val, extractor, args.out_cache_dir / "val_eval.pt",
        batch_size=args.batch_size, device=device, use_fp16=use_fp16,
    )

    test_out = args.out_cache_dir / "held_out_test_eval.pt"
    if not test_out.exists():
        test_src = (args.base_cache_dir / "held_out_test_eval.pt").resolve()
        test_out.symlink_to(test_src)
        logger.info("Linked %s -> %s", test_out, test_src)


if __name__ == "__main__":
    main()
