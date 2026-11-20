"""Dataset preparation, hash-deduplicated splits, face cropping, augmentations, and DataLoaders."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split

from src.face_crop import crop_face_or_full, detect_primary_face
from src.utils import (
    CLASS_NAMES_PATH,
    DATA_DIR,
    IMAGES_DIR,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    SEED,
    get_logger,
    load_class_mapping,
    save_class_mapping,
    save_json,
)

logger = get_logger(__name__)

RAW_DEFACTIFY_DIR: Path = DATA_DIR / "raw_defactify"
RAW_RAPIDATA_DIR: Path = DATA_DIR / "raw_rapidata"
RAW_CELEBA_DIR: Path = DATA_DIR / "raw_celeba"
MULTI_MANIFEST_PATH: Path = DATA_DIR / "multi_manifest.csv"

HELD_OUT_GENERATOR: str = "Midjourney"

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
TEST_FRACTION: float = 0.30
VAL_FRACTION_OF_TRAIN: float = 0.10


def inspect_parquet_class_to_idx(parquet_path: Path) -> dict[str, int]:
    """Read the Hugging Face Arrow schema metadata from a Parquet shard to get ``class_to_idx``."""
    schema = pq.read_schema(parquet_path)
    meta = schema.metadata or {}
    raw_hf = meta.get(b"huggingface")
    if raw_hf:
        hf = json.loads(raw_hf.decode())
        names = hf.get("info", {}).get("features", {}).get("label", {}).get("names")
        if isinstance(names, list) and len(names) >= 2:
            return {str(name).upper(): idx for idx, name in enumerate(names)}
        if isinstance(names, dict):
            return {str(name).upper(): int(idx) for idx, name in names.items()}
    raise RuntimeError(
        f"Could not extract ClassLabel names from {parquet_path} schema metadata"
    )


def derive_class_to_idx_from_manifest(df: pd.DataFrame) -> dict[str, int]:
    """Derive ``class_to_idx`` directly from the dataset manifest paths/columns."""
    if "class_name" in df.columns:
        pairs = df[["class_name", "label"]].drop_duplicates()
        return {
            str(r["class_name"]).upper(): int(r["label"]) for _, r in pairs.iterrows()
        }
    mapping: dict[str, int] = {}
    for label_val, group in df.groupby("label"):
        first_path = Path(str(group["path"].iloc[0]))
        folder_name = first_path.parent.name.upper()
        mapping[folder_name] = int(label_val)
    return dict(sorted(mapping.items(), key=lambda kv: kv[1]))


def _persist_raw_bytes(
    raw: bytes,
    label: int,
    class_name: str,
    generator: str,
    dataset_source: str,
) -> dict:
    """Decode, hash, and write an image file under ``data/images/<real|fake>/``."""
    sha = hashlib.sha256(raw).hexdigest()
    out_dir = IMAGES_DIR / class_name.lower()
    out_path = out_dir / f"{sha}.jpg"
    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        pix_hash = hashlib.md5(np.asarray(rgb).tobytes()).hexdigest()
        if not out_path.exists():
            rgb.save(out_path, format="JPEG", quality=95)
    return {
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "label": int(label),
        "class_name": class_name.upper(),
        "generator": generator,
        "dataset_source": dataset_source,
        "sha256": sha,
        "pixel_md5": pix_hash,
        "width": w,
        "height": h,
    }


def log_class_balance(
    df: pd.DataFrame, name: str, idx_to_class: Optional[dict[int, str]] = None
) -> None:
    """Log per-class counts and percentages for a split."""
    if len(df) == 0:
        logger.info("%-16s n=0", name)
        return
    if idx_to_class is None:
        if CLASS_NAMES_PATH.exists():
            idx_to_class = load_class_mapping()["idx_to_class"]
        else:
            c2i = derive_class_to_idx_from_manifest(df)
            idx_to_class = {v: k for k, v in c2i.items()}
    counts = df["label"].value_counts().sort_index()
    parts = [
        f"{idx_to_class[i]}({i})={counts.get(i, 0):,} ({100 * counts.get(i, 0) / len(df):.2f}%)"
        for i in sorted(idx_to_class)
    ]
    logger.info("%-16s n=%-7s %s", name, f"{len(df):,}", " | ".join(parts))


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any byte- or pixel-level hash is shared between two splits."""
    for col in ("sha256", "pixel_md5"):
        groups = {s: set(g[col]) for s, g in df.groupby("split")}
        names = sorted(groups)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = groups[a] & groups[b]
                if overlap:
                    raise RuntimeError(f"Leak: {len(overlap)} {col} shared by {a}/{b}")
        logger.info("Leak check (%s): no overlap between %s", col, " / ".join(names))


def build_multi_generator_manifest(
    ffhq_sample: int = 6000,
    stylegan_sample: int = 3500,
    celeba_sample: int = 3500,
    celebahq_sample: int = 2000,
) -> pd.DataFrame:
    """Pool all verified datasets, deduplicate by hash, hold out Midjourney, and split 70/30 (+10% val)."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Base manifest missing at {MANIFEST_PATH}")

    c2i = inspect_parquet_class_to_idx(sorted(RAW_DIR.glob("*.parquet"))[0])
    saved_map = save_class_mapping(c2i, CLASS_NAMES_PATH)
    class_to_idx = saved_map["class_to_idx"]
    idx_to_class = {int(k): v for k, v in saved_map["idx_to_class"].items()}
    real_idx = class_to_idx["REAL"]
    fake_idx = class_to_idx["FAKE"]

    for cname in class_to_idx:
        (IMAGES_DIR / cname.lower()).mkdir(parents=True, exist_ok=True)

    dataset_inventory: list[dict] = []
    all_rows: list[dict] = []

    # 1. Base 140k dataset (FFHQ real + StyleGAN fake) - already decoded & hashed in manifest.csv
    base_df = pd.read_csv(MANIFEST_PATH)
    raw_140k_bytes = sum(f.stat().st_size for f in RAW_DIR.glob("*.parquet"))
    logger.info(
        "Dataset 1: TheKernel01/140k-Real-and-Fake-Faces | size=%.1f MB | total=%d (REAL=%d, FAKE=%d)",
        raw_140k_bytes / 1e6,
        len(base_df),
        int((base_df["label"] == real_idx).sum()),
        int((base_df["label"] == fake_idx).sum()),
    )
    dataset_inventory.append(
        {
            "name": "TheKernel01/140k-Real-and-Fake-Faces (mirror of xhlulu/140k-real-and-fake-faces)",
            "size_mb": round(raw_140k_bytes / 1e6, 1),
            "total_rows_verified": int(len(base_df)),
            "class_counts_verified": {
                "REAL (FFHQ)": int((base_df["label"] == real_idx).sum()),
                "FAKE (StyleGAN)": int((base_df["label"] == fake_idx).sum()),
            },
        }
    )
    ffhq_sub = (
        base_df[base_df["label"] == real_idx]
        .sample(n=ffhq_sample, random_state=SEED)
        .copy()
    )
    ffhq_sub["class_name"] = "REAL"
    ffhq_sub["generator"] = "FFHQ"
    ffhq_sub["dataset_source"] = "140k_FFHQ_StyleGAN"

    sg_sub = (
        base_df[base_df["label"] == fake_idx]
        .sample(n=stylegan_sample, random_state=SEED)
        .copy()
    )
    sg_sub["class_name"] = "FAKE"
    sg_sub["generator"] = "StyleGAN"
    sg_sub["dataset_source"] = "140k_FFHQ_StyleGAN"
    all_rows.extend(
        ffhq_sub[
            [
                "path",
                "label",
                "class_name",
                "generator",
                "dataset_source",
                "sha256",
                "pixel_md5",
                "width",
                "height",
            ]
        ].to_dict("records")
    )
    all_rows.extend(
        sg_sub[
            [
                "path",
                "label",
                "class_name",
                "generator",
                "dataset_source",
                "sha256",
                "pixel_md5",
                "width",
                "height",
            ]
        ].to_dict("records")
    )

    # 2. Defactify_Image_Dataset (COCO Real, SD21, SDXL, SD3, DALLE3, Midjourney v6)
    def_files = sorted(RAW_DEFACTIFY_DIR.glob("*.parquet"))
    if not def_files:
        raise FileNotFoundError(f"No parquet shards in {RAW_DEFACTIFY_DIR}")
    def_bytes = sum(f.stat().st_size for f in def_files)
    b_map = {0: "COCO", 1: "SD21", 2: "SDXL", 3: "SD3", 4: "DALLE3", 5: "Midjourney"}
    def_counts: dict[str, int] = {}
    for f in def_files:
        t = pq.read_table(f, columns=["Image", "Label_A", "Label_B"])
        imgs = t.column("Image").to_pylist()
        la = t.column("Label_A").to_pylist()
        lb = t.column("Label_B").to_pylist()
        for im, a, b in zip(imgs, la, lb):
            gen = b_map[int(b)]
            is_fake = int(a) == 1
            lab = fake_idx if is_fake else real_idx
            cname = "FAKE" if is_fake else "REAL"
            def_counts[gen] = def_counts.get(gen, 0) + 1
            all_rows.append(
                _persist_raw_bytes(im["bytes"], lab, cname, gen, "Defactify")
            )
    logger.info(
        "Dataset 2: Rajarshi-Roy-research/Defactify_Image_Dataset | size=%.1f MB | counts=%s",
        def_bytes / 1e6,
        def_counts,
    )
    dataset_inventory.append(
        {
            "name": "Rajarshi-Roy-research/Defactify_Image_Dataset",
            "size_mb": round(def_bytes / 1e6, 1),
            "total_rows_verified": sum(def_counts.values()),
            "class_counts_verified": def_counts,
        }
    )

    # 3. Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset (FLUX.1, DALL-E 3, Midjourney)
    rap_files = sorted(RAW_RAPIDATA_DIR.glob("*.parquet"))
    if not rap_files:
        raise FileNotFoundError(f"No parquet shards in {RAW_RAPIDATA_DIR}")
    rap_bytes = sum(f.stat().st_size for f in rap_files)
    rap_seen: set[str] = set()
    rap_counts: dict[str, int] = {}
    model_norm = {"flux": "FLUX", "dalle-3": "DALLE3", "mj": "Midjourney", "sd3": "SD3"}
    for f in rap_files:
        t = pq.read_table(f, columns=["image1", "image2", "model1", "model2"])
        for col_im, col_m in [("image1", "model1"), ("image2", "model2")]:
            for im, m in zip(t.column(col_im).to_pylist(), t.column(col_m).to_pylist()):
                raw = im["bytes"]
                sha = hashlib.sha256(raw).hexdigest()
                if sha in rap_seen:
                    continue
                rap_seen.add(sha)
                gen = model_norm.get(str(m).lower(), str(m))
                rap_counts[gen] = rap_counts.get(gen, 0) + 1
                all_rows.append(
                    _persist_raw_bytes(
                        raw, fake_idx, "FAKE", gen, "Rapidata_Flux_DALLE3_MJ"
                    )
                )
    logger.info(
        "Dataset 3: Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset | size=%.1f MB | unique_counts=%s",
        rap_bytes / 1e6,
        rap_counts,
    )
    dataset_inventory.append(
        {
            "name": "Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset",
            "size_mb": round(rap_bytes / 1e6, 1),
            "total_rows_verified": sum(rap_counts.values()),
            "class_counts_verified": rap_counts,
        }
    )

    # 4. flwrlabs/celeba (178x218 in-the-wild real faces) + mattymchen/celeba-hq (1024x1024 real portraits)
    celeba_file = RAW_CELEBA_DIR / "celeba_test_0.parquet"
    celebahq_file = RAW_CELEBA_DIR / "celebahq_val_0.parquet"
    if not celeba_file.exists() or not celebahq_file.exists():
        raise FileNotFoundError(f"Missing CelebA shards in {RAW_CELEBA_DIR}")

    t_ca = pq.read_table(celeba_file, columns=["image"])
    ca_imgs = t_ca.column("image").to_pylist()
    rng = random.Random(SEED)
    ca_indices = rng.sample(range(len(ca_imgs)), min(celeba_sample, len(ca_imgs)))
    for idx in ca_indices:
        all_rows.append(
            _persist_raw_bytes(
                ca_imgs[idx]["bytes"], real_idx, "REAL", "CelebA", "flwrlabs_celeba"
            )
        )

    t_hq = pq.read_table(celebahq_file, columns=["image"])
    hq_imgs = t_hq.column("image").to_pylist()
    hq_indices = rng.sample(range(len(hq_imgs)), min(celebahq_sample, len(hq_imgs)))
    for idx in hq_indices:
        all_rows.append(
            _persist_raw_bytes(
                hq_imgs[idx]["bytes"],
                real_idx,
                "REAL",
                "CelebA_HQ",
                "mattymchen_celeba_hq",
            )
        )

    logger.info(
        "Dataset 4a: flwrlabs/celeba | size=%.1f MB | rows_verified=%d (sampled=%d)",
        celeba_file.stat().st_size / 1e6,
        len(ca_imgs),
        len(ca_indices),
    )
    logger.info(
        "Dataset 4b: mattymchen/celeba-hq | size=%.1f MB | rows_verified=%d (sampled=%d)",
        celebahq_file.stat().st_size / 1e6,
        len(hq_imgs),
        len(hq_indices),
    )
    dataset_inventory.append(
        {
            "name": "flwrlabs/celeba (in-the-wild 178x218 real faces)",
            "size_mb": round(celeba_file.stat().st_size / 1e6, 1),
            "total_rows_verified": len(ca_imgs),
            "class_counts_verified": {"REAL (CelebA sampled)": len(ca_indices)},
        }
    )
    dataset_inventory.append(
        {
            "name": "mattymchen/celeba-hq (high-res 1024x1024 real faces)",
            "size_mb": round(celebahq_file.stat().st_size / 1e6, 1),
            "total_rows_verified": len(hq_imgs),
            "class_counts_verified": {"REAL (CelebA-HQ sampled)": len(hq_indices)},
        }
    )

    df = pd.DataFrame(all_rows)
    logger.info(
        "Pooled %d total images across all sources before deduplication", len(df)
    )

    # Deduplicate by decoded pixel MD5 and raw SHA-256
    label_sets = df.groupby("pixel_md5")["label"].nunique()
    conflicting = set(label_sets[label_sets > 1].index)
    n_before = len(df)
    df = df[~df["pixel_md5"].isin(conflicting)]
    df = df.drop_duplicates(subset="pixel_md5", keep="first")
    df = df.drop_duplicates(subset="sha256", keep="first").reset_index(drop=True)
    logger.info(
        "Deduplication: removed %d duplicate rows (%d conflicting pixel hashes) -> %d unique images",
        n_before - len(df),
        len(conflicting),
        len(df),
    )

    # Hold out Midjourney completely to measure unseen-generator generalization.
    is_held_out = df["generator"] == HELD_OUT_GENERATOR
    df.loc[is_held_out, "split"] = "unseen_generator"
    in_dist_df = df[~is_held_out].copy()

    # 70/30 split stratified jointly by (generator, label).
    strata = in_dist_df["generator"].astype(str) + "_" + in_dist_df["label"].astype(str)
    train_val, test = train_test_split(
        in_dist_df,
        test_size=TEST_FRACTION,
        stratify=strata,
        random_state=SEED,
    )
    tv_strata = (
        train_val["generator"].astype(str) + "_" + train_val["label"].astype(str)
    )
    train, val = train_test_split(
        train_val,
        test_size=VAL_FRACTION_OF_TRAIN,
        stratify=tv_strata,
        random_state=SEED,
    )
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    for split_name in ("train", "val", "test", "unseen_generator"):
        sub = df[df["split"] == split_name]
        log_class_balance(sub, split_name, idx_to_class)
        logger.info(
            "  %s generators: %s", split_name, sub["generator"].value_counts().to_dict()
        )

    verify_no_leakage(df)
    df.to_csv(MULTI_MANIFEST_PATH, index=False)
    save_json(dataset_inventory, REPORTS_DIR / "dataset_inventory.json")
    logger.info(
        "Saved multi-generator manifest (%d rows) to %s", len(df), MULTI_MANIFEST_PATH
    )
    return df


def load_manifest(multi: bool = True) -> pd.DataFrame:
    """Load the multi-generator manifest if available (else base manifest), verifying zero leakage."""
    target = (
        MULTI_MANIFEST_PATH
        if (multi and MULTI_MANIFEST_PATH.exists())
        else MANIFEST_PATH
    )
    if not target.exists():
        raise FileNotFoundError(
            f"Manifest missing at {target} - run `python -m src.data` first."
        )
    df = pd.read_csv(target)
    verify_no_leakage(df)
    return df


def stratified_subset(
    df: pd.DataFrame, n: Optional[int], seed: int = SEED
) -> pd.DataFrame:
    """Return a generator-and-class-stratified random subset of ``n`` rows."""
    if n is None or n >= len(df):
        return df
    strata = (
        df["generator"].astype(str) + "_" + df["label"].astype(str)
        if "generator" in df.columns
        else df["label"]
    )
    sub, _ = train_test_split(df, train_size=n, stratify=strata, random_state=seed)
    return sub


def _probe_face_row(rel_path: str) -> tuple[bool, float]:
    """Run YuNet on a single image path (CPU-only worker for ProcessPoolExecutor)."""
    with Image.open(PROJECT_ROOT / rel_path) as img:
        det = detect_primary_face(img.convert("RGB"))
    if det is None:
        return False, 0.0
    return True, round(float(det["score"]), 4)


def annotate_manifest_faces(workers: int = 6) -> pd.DataFrame:
    """Annotate multi_manifest.csv with YuNet has_face and face_score without changing splits."""
    if not MULTI_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Multi manifest missing at {MULTI_MANIFEST_PATH}")
    df = pd.read_csv(MULTI_MANIFEST_PATH)
    paths = df["path"].astype(str).tolist()
    logger.info(
        "Running OpenCV DNN YuNet face detection across %d images (%d workers)...",
        len(paths),
        workers,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_probe_face_row, paths, chunksize=64))
    df["has_face"] = [r[0] for r in results]
    df["face_score"] = [r[1] for r in results]

    for split_name in ("train", "val", "test", "unseen_generator"):
        sub = df[df["split"] == split_name]
        n_face = int(sub["has_face"].sum())
        logger.info(
            "Face gate [%-16s]: %5d / %5d (%.1f%%) detected",
            split_name,
            n_face,
            len(sub),
            100 * n_face / max(1, len(sub)),
        )
    for gen_name, grp in df.groupby("generator"):
        n_face = int(grp["has_face"].sum())
        logger.info(
            "  Generator %-12s: %4d / %4d (%.1f%%) have face",
            gen_name,
            n_face,
            len(grp),
            100 * n_face / max(1, len(grp)),
        )

    verify_no_leakage(df)
    df.to_csv(MULTI_MANIFEST_PATH, index=False)
    logger.info("Updated manifest with face annotations at %s", MULTI_MANIFEST_PATH)
    return df


def _sample_balanced_by_generator(
    class_df: pd.DataFrame, quota: int, seed: int
) -> pd.DataFrame:
    """Sample ``quota`` rows evenly across generator/source groups within one class."""
    if quota >= len(class_df) or "generator" not in class_df.columns:
        return class_df.sample(n=min(quota, len(class_df)), random_state=seed)
    groups = sorted(
        [(g, grp) for g, grp in class_df.groupby("generator")],
        key=lambda kv: len(kv[1]),
    )
    picks = []
    remaining = quota
    for idx, (_, grp) in enumerate(groups):
        rem_groups = len(groups) - idx
        target = remaining // rem_groups + (
            1 if idx >= len(groups) - (remaining % rem_groups) else 0
        )
        take = min(len(grp), target)
        if take > 0:
            picks.append(grp.sample(n=take, random_state=seed + idx))
            remaining -= take
    return pd.concat(picks, ignore_index=True)


def balanced_face_subset(
    df: pd.DataFrame, target_total: int, seed: int = SEED
) -> pd.DataFrame:
    """Filter to ``has_face == True`` and sample 50/50 REAL/FAKE balanced across generators."""
    work = df[df["has_face"] == True].copy() if "has_face" in df.columns else df.copy()  # noqa: E712
    classes = sorted(work["label"].unique())
    if len(classes) != 2:
        return stratified_subset(work, target_total, seed=seed)
    c0 = work[work["label"] == classes[0]]
    c1 = work[work["label"] == classes[1]]
    per_class = min(target_total // 2, len(c0), len(c1))
    s0 = _sample_balanced_by_generator(c0, per_class, seed=seed)
    s1 = _sample_balanced_by_generator(c1, per_class, seed=seed + 100)
    combined = pd.concat([s0, s1], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class RandomResizeDegradation:
    """Randomly downscale (0.25x-1.0x) and upscale back to original dimensions."""

    def __init__(
        self, scale: tuple[float, float] = (0.25, 1.0), p: float = 0.45
    ) -> None:
        self.scale = scale
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale)
        nw, nh = max(24, int(round(w * s))), max(24, int(round(h * s)))
        interp_down = random.choice([Image.BILINEAR, Image.BICUBIC, Image.BOX])
        interp_up = random.choice([Image.BILINEAR, Image.BICUBIC])
        return img.resize((nw, nh), interp_down).resize((w, h), interp_up)


class RandomJPEG:
    """Re-encode a PIL image as JPEG at a random quality (30-95)."""

    def __init__(self, quality: tuple[int, int] = (30, 95), p: float = 0.45) -> None:
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomSocialMediaReencode:
    """Chained social-media degradation: resize downscale+upscale followed by JPEG re-encode."""

    def __init__(
        self,
        scale: tuple[float, float] = (0.35, 0.85),
        quality: tuple[int, int] = (35, 75),
        p: float = 0.35,
    ) -> None:
        self.scale = scale
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale)
        nw, nh = max(32, int(round(w * s))), max(32, int(round(h * s)))
        resized = img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    """Apply PIL Gaussian blur with a random radius."""

    def __init__(
        self, radius: tuple[float, float] = (0.3, 2.2), p: float = 0.30
    ) -> None:
        self.radius = radius
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


class RandomGaussianNoise:
    """Add light zero-mean Gaussian noise to a [0, 1] tensor."""

    def __init__(
        self, std: tuple[float, float] = (0.005, 0.04), p: float = 0.30
    ) -> None:
        self.std = std
        self.p = p

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        import torch

        if random.random() >= self.p:
            return x
        return (x + torch.randn_like(x) * random.uniform(*self.std)).clamp_(0.0, 1.0)


class RandomSmallFaceInScene:
    """Simulate a distant small face (8%-22% width) in a scene, tightly cropped and upsampled."""

    def __init__(
        self,
        scale: tuple[float, float] = (0.08, 0.22),
        margin: tuple[float, float] = (0.18, 0.26),
        p: float = 0.30,
    ) -> None:
        self.scale = scale
        self.margin = margin
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        scene_size = 512
        # Smooth color mosaic background derived from same image (destroys giant-face silhouette, zero cross-class leakage)
        scene = (
            img.resize((12, 12), Image.Resampling.BILINEAR)
            .resize((scene_size, scene_size), Image.Resampling.BILINEAR)
            .filter(ImageFilter.GaussianBlur(radius=random.uniform(8.0, 14.0)))
        )
        s = random.uniform(*self.scale)
        fw = max(36, int(round(scene_size * s)))
        fh = max(36, int(round(fw * (img.height / max(1, img.width)))))
        small_face = img.resize((fw, fh), Image.Resampling.BILINEAR)

        max_x = max(1, scene_size - fw - 1)
        max_y = max(1, scene_size - fh - 1)
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)
        scene.paste(small_face, (px, py))

        m = random.uniform(*self.margin)
        side = max(fw, fh) * (1.0 + 2.0 * m)
        cx, cy = px + fw / 2.0, py + fh / 2.0
        x1 = max(0, int(round(cx - side / 2.0)))
        y1 = max(0, int(round(cy - side / 2.0)))
        x2 = min(scene_size, int(round(cx + side / 2.0)))
        y2 = min(scene_size, int(round(cy + side / 2.0)))
        tight = scene.crop((x1, y1, x2, y2))
        return tight.resize((256, 256), Image.Resampling.LANCZOS)


def build_transforms(
    img_size: int,
    train: bool,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> Callable:
    """Training transforms add small-face & degradation robustness; eval is clean resize + normalize."""
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    normalize = T.Normalize(mean, std)
    if not train:
        return T.Compose(
            [
                T.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            RandomSmallFaceInScene(scale=(0.08, 0.22), margin=(0.18, 0.26), p=0.30),
            RandomResizeDegradation(scale=(0.25, 1.0), p=0.45),
            RandomJPEG(quality=(30, 95), p=0.45),
            RandomSocialMediaReencode(scale=(0.35, 0.85), quality=(35, 75), p=0.35),
            T.RandomResizedCrop(
                img_size,
                scale=(0.6, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.5),
            RandomGaussianBlur(radius=(0.3, 2.2), p=0.30),
            T.ToTensor(),
            RandomGaussianNoise(std=(0.005, 0.04), p=0.30),
            normalize,
        ]
    )


class FaceDataset:
    """Map-style dataset over rows of the manifest, applying face-crop with full-frame fallback."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        pre_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        use_face_crop: bool = True,
    ) -> None:
        self.paths = [str(PROJECT_ROOT / p) for p in df["path"]]
        self.labels = df["label"].astype(int).tolist()
        self.generators = (
            df["generator"].astype(str).tolist()
            if "generator" in df.columns
            else ["unknown"] * len(self.paths)
        )
        self.transform = transform
        self.pre_transform = pre_transform
        self.use_face_crop = use_face_crop
        self.class_to_idx: dict[str, int] = (
            {str(k).upper(): int(v) for k, v in class_to_idx.items()}
            if class_to_idx is not None
            else derive_class_to_idx_from_manifest(df)
        )
        self.idx_to_class: dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.classes: list[str] = [
            self.idx_to_class[i] for i in sorted(self.idx_to_class)
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        if self.use_face_crop:
            img, _ = crop_face_or_full(img)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        return self.transform(img), self.labels[idx]


class TTAFaceDataset:
    """3-view Test-Time Augmentation dataset: [face_crop, flipped_face_crop, full_frame]."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        pre_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        self.paths = [str(PROJECT_ROOT / p) for p in df["path"]]
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.pre_transform = pre_transform
        self.class_to_idx: dict[str, int] = (
            {str(k).upper(): int(v) for k, v in class_to_idx.items()}
            if class_to_idx is not None
            else derive_class_to_idx_from_manifest(df)
        )
        self.idx_to_class: dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.classes: list[str] = [
            self.idx_to_class[i] for i in sorted(self.idx_to_class)
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        import torch

        with Image.open(self.paths[idx]) as img:
            pil = img.convert("RGB")
        if self.pre_transform is not None:
            pil = self.pre_transform(pil)
        cropped, _ = crop_face_or_full(pil)
        views = torch.stack(
            [
                self.transform(cropped),
                self.transform(ImageOps.mirror(cropped)),
                self.transform(pil),
            ],
            dim=0,
        )
        return views, self.labels[idx]


def make_loader(
    df: pd.DataFrame,
    img_size: int,
    train: bool,
    batch_size: int,
    workers: int,
    pre_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    class_to_idx: Optional[dict[str, int]] = None,
    use_face_crop: bool = True,
) -> "torch.utils.data.DataLoader":  # noqa: F821
    """Create a DataLoader for a manifest slice."""
    import torch

    ds = FaceDataset(
        df,
        build_transforms(img_size, train),
        pre_transform,
        class_to_idx=class_to_idx,
        use_face_crop=use_face_crop,
    )
    gen = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=gen,
        pin_memory=torch.cuda.is_available(),
    )


def make_tta_loader(
    df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    workers: int,
    pre_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    class_to_idx: Optional[dict[str, int]] = None,
) -> "torch.utils.data.DataLoader":  # noqa: F821
    """Create a DataLoader yielding 3-view TTA stacks of shape ``(B, 3, 3, H, W)``."""
    import torch

    ds = TTAFaceDataset(
        df,
        build_transforms(img_size, train=False),
        pre_transform=pre_transform,
        class_to_idx=class_to_idx,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build or annotate the multi-generator dataset manifest"
    )
    parser.add_argument(
        "--annotate-faces",
        action="store_true",
        help="Tag existing multi_manifest.csv with YuNet has_face",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ffhq-sample", type=int, default=6000)
    parser.add_argument("--stylegan-sample", type=int, default=3500)
    parser.add_argument("--celeba-sample", type=int, default=3500)
    parser.add_argument("--celebahq-sample", type=int, default=2000)
    args = parser.parse_args()
    if args.annotate_faces:
        annotate_manifest_faces(workers=args.workers)
    else:
        build_multi_generator_manifest(
            ffhq_sample=args.ffhq_sample,
            stylegan_sample=args.stylegan_sample,
            celeba_sample=args.celeba_sample,
            celebahq_sample=args.celebahq_sample,
        )


if __name__ == "__main__":
    main()
