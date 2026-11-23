"""Generalization-safe data ingestion, deduplication, splitting, and shortcut-leakage auditing."""

from __future__ import annotations

import hashlib
import io
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import MULTI_MANIFEST_PATH
from src.face_crop import detect_primary_face
from src.image_io import decode_image_bytes, register_image_plugins, to_rgb_safe
from src.utils import PROJECT_ROOT, SEED, get_logger

logger = get_logger(__name__)

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

PUBLIC_DATASETS: list[dict[str, str]] = [
    {
        "name": "GenImage",
        "source_url": "https://github.com/GenImage-Dataset/GenImage (HF mirror: nebula/GenImage-arrow)",
        "license": "CC BY-NC-SA 4.0 (Non-commercial research)",
        "availability": "Public benchmark (~1M image pairs across 8 generators)",
        "generators": "Midjourney, SD 1.4, SD 1.5, ADM, GLIDE, VQDM, BigGAN, Wukong",
        "real_sources": "ImageNet ILSVRC2012",
        "cli_flag": "--genimage-dir /path/to/GenImage",
    },
    {
        "name": "Synthbuster",
        "source_url": "Zenodo record 10066460 (+ RAISE-1k real raw images)",
        "license": "CC-BY-4.0 (Synthbuster); check RAISE-1k research license",
        "availability": "Public on Zenodo (~9,000 lossless PNGs)",
        "generators": "DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5, SD 1.3, SD 1.4, SD 2, SDXL, GLIDE",
        "real_sources": "RAISE-1k (raw DSLR photos, separate download)",
        "cli_flag": "--synthbuster-dir /path/to/synthbuster --raise1k-dir /path/to/raise1k",
    },
    {
        "name": "Community Forensics",
        "source_url": "huggingface.co/datasets/OwensLab/CommunityForensics (-Small / PublicEval)",
        "license": "CC BY-NC-SA 4.0 / Non-commercial + underlying generator model licenses",
        "availability": "Public on Hugging Face (~1.1 TB full / ~278 GB Small Parquet)",
        "generators": "4,803 open & commercial image generators",
        "real_sources": "Redistributable real sets in Small & PublicEval (e.g., COCO/FFHQ)",
        "cli_flag": "Extract/stream or pass extracted folders via --user-real-dir / --user-fake-dir",
    },
    {
        "name": "WildRF",
        "source_url": "github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld",
        "license": "Check author repository terms (academic research)",
        "availability": "Public links via author GitHub repo (social-media in-the-wild)",
        "generators": "Wild AI images from Reddit, X/Twitter, Facebook",
        "real_sources": "Wild real photos from Reddit, X/Twitter, Facebook",
        "cli_flag": "--wildrf-dir /path/to/WildRF",
    },
    {
        "name": "OpenFake",
        "source_url": "huggingface.co/datasets/ComplexDataLab/OpenFake",
        "license": "Verify Hugging Face dataset card terms before commercial use",
        "availability": "Partially ingested locally (1,620 rows in multi_manifest.csv)",
        "generators": "GPT_Image_1, FLUX variants, Midjourney 6",
        "real_sources": "OpenFake_Real (LAION, Pexels)",
        "cli_flag": "Already present in data/multi_manifest.csv",
    },
]

METADATA_FEATURES: list[str] = [
    "width",
    "height",
    "aspect_ratio",
    "max_side",
    "min_side",
    "file_bytes",
    "bpp",
    "format_code",
    "has_exif",
]

FORMAT_CODES: dict[str, int] = {
    "JPEG": 0,
    "MPO": 0,
    "PNG": 1,
    "WEBP": 2,
    "HEIF": 3,
    "AVIF": 4,
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def normalize_generator_family(gen_name: str, class_name: str) -> str:
    """Map generator variants into canonical families so related models never split across folds."""
    if str(class_name).strip().upper() == "REAL":
        return "real"
    slug = _slug(gen_name)
    if slug == "flux" or slug.startswith("flux_"):
        return "flux"
    if slug.startswith("midjourney") or slug == "mj":
        return "midjourney"
    if slug in ("dalle3", "dalle_3", "dall_e_3"):
        return "dalle3"
    if slug in ("dalle2", "dalle_2", "dall_e_2"):
        return "dalle2"
    if slug in ("gpt_image_1", "gpt_image", "gptimage1"):
        return "gpt_image_1"
    if slug in ("sd21", "sd_2_1", "sd_v2", "sd2"):
        return "sd21"
    if slug in ("sd3", "sd_3"):
        return "sd3"
    if slug in ("sdxl", "sd_xl"):
        return "sdxl"
    if slug in ("sd14", "sd_1_4", "sd_v1_4", "stable_diffusion_v_1_4"):
        return "sd14"
    if slug in ("sd15", "sd_1_5", "sd_v1_5", "stable_diffusion_v_1_5"):
        return "sd15"
    if slug in ("sd13", "sd_1_3"):
        return "sd13"
    if slug in ("firefly", "adobe_firefly"):
        return "firefly"
    if slug in ("gemini", "google_gemini", "imagen", "imagen3", "google_imagen"):
        return "gemini"
    return slug


def compute_dhash64(img: Image.Image) -> int:
    """Compute 64-bit horizontal gradient difference hash (dHash)."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).ravel()
    h = 0
    for bit in diff:
        h = (h << 1) | int(bool(bit))
    return h


def _format_to_code(fmt: str) -> int:
    return FORMAT_CODES.get(str(fmt).upper(), 5)


def _res_bucket(max_side: int) -> str:
    if max_side <= 256:
        return "<=256"
    if max_side <= 512:
        return "257-512"
    if max_side <= 1024:
        return "513-1024"
    return ">1024"


def _aspect_bucket(w: int, h: int) -> str:
    ar = float(w) / max(1.0, float(h))
    if 0.95 <= ar <= 1.05:
        return "square"
    return "portrait" if ar < 0.95 else "landscape"


def inspect_image_metadata_row(
    path: Path, *, compute_hashes: bool = True
) -> dict[str, Any]:
    """Read disk metadata, dimensions, EXIF flag, and dHash64 for one image."""
    register_image_plugins()
    raw = path.read_bytes()
    file_bytes = len(raw)
    sha = hashlib.sha256(raw).hexdigest() if compute_hashes else ""

    with Image.open(io.BytesIO(raw)) as im:
        fmt = (im.format or path.suffix.lstrip(".") or "UNKNOWN").upper()
        try:
            exif = im.getexif()
            has_exif = bool(exif and len(exif) > 0)
        except Exception:
            has_exif = False
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
        rgb = to_rgb_safe(ImageOps.exif_transpose(im))

    w, h = rgb.size
    pix_md5 = hashlib.md5(np.asarray(rgb).tobytes()).hexdigest() if compute_hashes else ""
    dhash64 = compute_dhash64(rgb)
    thumb16 = rgb.resize((16, 16), Image.Resampling.LANCZOS).tobytes()
    return {
        "file_format": fmt,
        "file_bytes": int(file_bytes),
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(float(w) / max(1.0, float(h)), 4),
        "max_side": int(max(w, h)),
        "min_side": int(min(w, h)),
        "bpp": round(float(file_bytes) * 8.0 / max(1, w * h), 4),
        "format_code": _format_to_code(fmt),
        "has_exif": bool(has_exif),
        "dhash64": f"{dhash64:016x}",
        "dhash_int": int(dhash64),
        "res_bucket": _res_bucket(max(w, h)),
        "aspect_bucket": _aspect_bucket(w, h),
        "sha256": sha,
        "pixel_md5": pix_md5,
        "_thumb16": thumb16,
    }


def _scan_existing_row(row_path: str) -> dict[str, Any]:
    abs_path = PROJECT_ROOT / row_path
    info = inspect_image_metadata_row(abs_path, compute_hashes=False)
    info["path"] = row_path
    return info


def ingest_existing_manifest(
    manifest_path: Path = MULTI_MANIFEST_PATH,
    *,
    workers: int = 8,
) -> pd.DataFrame:
    """Load existing multi_manifest.csv and enrich with disk metadata + dHash64."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    missing = [p for p in df["path"] if not (PROJECT_ROOT / str(p)).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image(s) missing on disk (first: {missing[0]})"
        )

    logger.info("Scanning %d existing manifest images (workers=%d)...", len(df), workers)
    paths = [str(p) for p in df["path"]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        scanned = list(ex.map(_scan_existing_row, paths, chunksize=64))
    scan_df = pd.DataFrame(scanned)

    out = df.copy()
    for col in (
        "file_format",
        "file_bytes",
        "aspect_ratio",
        "max_side",
        "min_side",
        "bpp",
        "format_code",
        "has_exif",
        "dhash64",
        "dhash_int",
        "res_bucket",
        "aspect_bucket",
        "_thumb16",
    ):
        out[col] = scan_df[col]
    out["width"] = scan_df["width"]
    out["height"] = scan_df["height"]
    out["generator_family"] = [
        normalize_generator_family(str(g), str(c))
        for g, c in zip(out["generator"], out["class_name"])
    ]
    return out


def ingest_user_folder(
    folder: Path,
    *,
    class_name: str,
    source_name: str,
    dataset_source: str = "user_dropin",
    run_face_gate: bool = True,
) -> pd.DataFrame:
    """Ingest a user or external image folder into prepared-manifest schema."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    cname = class_name.strip().upper()
    if cname not in ("REAL", "FAKE"):
        raise ValueError(f"class_name must be REAL or FAKE, got {class_name!r}")
    label = 0 if cname == "REAL" else 1
    files = sorted(
        (
            p
            for p in folder.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda p: str(p),
    )
    records: list[dict[str, Any]] = []
    for p in files:
        try:
            raw = p.read_bytes()
            rgb, fmt = decode_image_bytes(raw)
        except Exception as exc:
            logger.warning("Skipping unreadable image %s (%s)", p, exc)
            continue

        info = inspect_image_metadata_row(p, compute_hashes=True)
        has_face, face_score = False, 0.0
        if run_face_gate:
            det = detect_primary_face(rgb)
            if det is not None:
                has_face = True
                face_score = round(float(det["score"]), 4)

        try:
            rel_path = str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            rel_path = str(p.resolve())

        records.append(
            {
                "path": rel_path,
                "label": label,
                "class_name": cname,
                "generator": source_name,
                "generator_family": normalize_generator_family(source_name, cname),
                "dataset_source": dataset_source,
                "sha256": info["sha256"],
                "pixel_md5": info["pixel_md5"],
                "width": info["width"],
                "height": info["height"],
                "has_face": has_face,
                "face_score": face_score,
                "file_format": fmt,
                "file_bytes": info["file_bytes"],
                "aspect_ratio": info["aspect_ratio"],
                "max_side": info["max_side"],
                "min_side": info["min_side"],
                "bpp": info["bpp"],
                "format_code": _format_to_code(fmt),
                "has_exif": info["has_exif"],
                "dhash64": info["dhash64"],
                "dhash_int": info["dhash_int"],
                "res_bucket": info["res_bucket"],
                "aspect_bucket": info["aspect_bucket"],
                "_thumb16": info["_thumb16"],
            }
        )
    return pd.DataFrame(records)


def ingest_genimage_dir(root: Path, *, run_face_gate: bool = True) -> pd.DataFrame:
    """Ingest GenImage layout (<gen>/{train,val}/{nature,ai}/ or <root>/{nature,ai}/)."""
    if not root.exists():
        raise FileNotFoundError(f"GenImage root not found: {root}")
    dfs: list[pd.DataFrame] = []
    for ai_dir in sorted(root.rglob("ai")):
        if not ai_dir.is_dir():
            continue
        rel_parts = ai_dir.relative_to(root).parts
        gen_name = rel_parts[0] if len(rel_parts) >= 2 and rel_parts[0] not in ("train", "val") else "genimage_ai"
        dfs.append(
            ingest_user_folder(
                ai_dir,
                class_name="FAKE",
                source_name=gen_name,
                dataset_source="GenImage",
                run_face_gate=run_face_gate,
            )
        )
    for nat_dir in sorted(root.rglob("nature")):
        if not nat_dir.is_dir():
            continue
        dfs.append(
            ingest_user_folder(
                nat_dir,
                class_name="REAL",
                source_name="imagenet_nature",
                dataset_source="GenImage",
                run_face_gate=run_face_gate,
            )
        )
    return pd.concat([d for d in dfs if len(d)], ignore_index=True) if dfs else pd.DataFrame()


def ingest_synthbuster_dir(
    root: Path, *, raise1k_dir: Optional[Path] = None, run_face_gate: bool = True
) -> pd.DataFrame:
    """Ingest Synthbuster (<gen>/*.png) plus optional RAISE-1k real folder."""
    if not root.exists():
        raise FileNotFoundError(f"Synthbuster root not found: {root}")
    dfs: list[pd.DataFrame] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            dfs.append(
                ingest_user_folder(
                    sub,
                    class_name="FAKE",
                    source_name=sub.name,
                    dataset_source="Synthbuster",
                    run_face_gate=run_face_gate,
                )
            )
    if raise1k_dir is not None and raise1k_dir.exists():
        dfs.append(
            ingest_user_folder(
                raise1k_dir,
                class_name="REAL",
                source_name="raise1k",
                dataset_source="Synthbuster_RAISE1k",
                run_face_gate=run_face_gate,
            )
        )
    return pd.concat([d for d in dfs if len(d)], ignore_index=True) if dfs else pd.DataFrame()


def ingest_wildrf_dir(root: Path, *, run_face_gate: bool = True) -> pd.DataFrame:
    """Ingest WildRF layout (<platform>/{0_real,1_fake}/ or {real,fake}/)."""
    if not root.exists():
        raise FileNotFoundError(f"WildRF root not found: {root}")
    dfs: list[pd.DataFrame] = []
    for sub in sorted(root.rglob("*")):
        if not sub.is_dir():
            continue
        low = sub.name.lower()
        rel_parts = sub.relative_to(root).parts
        platform = rel_parts[0] if len(rel_parts) >= 2 else "wildrf"
        if low in ("0_real", "real"):
            dfs.append(
                ingest_user_folder(
                    sub,
                    class_name="REAL",
                    source_name=f"wildrf_{_slug(platform)}",
                    dataset_source="WildRF",
                    run_face_gate=run_face_gate,
                )
            )
        elif low in ("1_fake", "fake"):
            dfs.append(
                ingest_user_folder(
                    sub,
                    class_name="FAKE",
                    source_name=f"wildrf_{_slug(platform)}",
                    dataset_source="WildRF",
                    run_face_gate=run_face_gate,
                )
            )
    return pd.concat([d for d in dfs if len(d)], ignore_index=True) if dfs else pd.DataFrame()


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _pigeonhole_band_masks(k: int) -> list[tuple[int, int]]:
    base = 64 // k
    rem = 64 % k
    bands: list[tuple[int, int]] = []
    shift = 0
    for i in range(k):
        width = base + (1 if i < rem else 0)
        mask = (1 << width) - 1
        bands.append((shift, mask))
        shift += width
    return bands


def _thumbs_match(
    t1: bytes,
    t2: bytes,
    *,
    max_mad: float = 12.0,
    min_cos: float = 0.96,
) -> bool:
    """Verify two 16x16 RGB thumbnails match closely in color/structure, rejecting dHash collisions."""
    a = np.frombuffer(t1, dtype=np.uint8).astype(np.float32)
    b = np.frombuffer(t2, dtype=np.uint8).astype(np.float32)
    if float(np.mean(np.abs(a - b))) > max_mad:
        return False
    ac = a - float(a.mean())
    bc = b - float(b.mean())
    na = float(np.linalg.norm(ac))
    nb = float(np.linalg.norm(bc))
    if na < 1e-6 or nb < 1e-6:
        return na < 1e-6 and nb < 1e-6
    return float(np.dot(ac, bc) / (na * nb)) >= min_cos


def cluster_and_deduplicate(
    df: pd.DataFrame,
    *,
    max_hamming: int = 4,
    max_thumb_mad: float = 12.0,
    min_thumb_cos: float = 0.96,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop exact sha256/pixel_md5 duplicates, cluster near-dups by dHash+RGB thumb, and drop cross-class collisions."""
    if len(df) == 0:
        return df.copy(), {
            "input_rows": 0,
            "exact_sha_dropped": 0,
            "exact_pixel_dropped": 0,
            "output_rows": 0,
            "unique_clusters": 0,
            "near_dup_clusters": 0,
            "cross_class_conflicts_dropped": 0,
        }

    work = df.reset_index(drop=True).copy()
    n_in = len(work)
    work = work.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
    after_sha = len(work)
    work = work.drop_duplicates(subset=["pixel_md5"], keep="first").reset_index(drop=True)
    after_pix = len(work)

    n = len(work)
    dhash_ints = [int(x) for x in work["dhash_int"].tolist()]
    has_thumbs = "_thumb16" in work.columns and not work["_thumb16"].isna().any()
    thumbs: list[bytes] = work["_thumb16"].tolist() if has_thumbs else []
    uf = _UnionFind(n)

    # Group indices sharing exact 64-bit dHash first, verifying 16x16 RGB similarity
    by_hash: dict[int, list[int]] = {}
    for idx, hv in enumerate(dhash_ints):
        by_hash.setdefault(hv, []).append(idx)
    for idxs in by_hash.values():
        if len(idxs) < 2:
            continue
        if not has_thumbs:
            first = idxs[0]
            for other in idxs[1:]:
                uf.union(first, other)
        elif len(idxs) <= 256:
            for a_pos in range(len(idxs)):
                ia = idxs[a_pos]
                for b_pos in range(a_pos + 1, len(idxs)):
                    ib = idxs[b_pos]
                    if _thumbs_match(thumbs[ia], thumbs[ib], max_mad=max_thumb_mad, min_cos=min_thumb_cos):
                        uf.union(ia, ib)

    # Near-duplicate detection across unique dHashes via exact pigeonhole bucketing
    if max_hamming > 0 and len(by_hash) > 1:
        uniq_hashes = list(by_hash.keys())
        k = min(64, max_hamming + 1)
        bands = _pigeonhole_band_masks(k)
        seen_hash_pairs: set[tuple[int, int]] = set()
        for shift, mask in bands:
            buckets: dict[int, list[int]] = {}
            for u_idx, hv in enumerate(uniq_hashes):
                key = (hv >> shift) & mask
                buckets.setdefault(key, []).append(u_idx)
            for members in buckets.values():
                if len(members) < 2 or len(members) > 512:
                    continue
                for a_pos in range(len(members)):
                    ha = uniq_hashes[members[a_pos]]
                    idxs_a = by_hash[ha]
                    for b_pos in range(a_pos + 1, len(members)):
                        hb = uniq_hashes[members[b_pos]]
                        if (ha ^ hb).bit_count() > max_hamming:
                            continue
                        pair_key = (ha, hb) if ha < hb else (hb, ha)
                        if pair_key in seen_hash_pairs:
                            continue
                        seen_hash_pairs.add(pair_key)
                        idxs_b = by_hash[hb]
                        if not has_thumbs:
                            uf.union(idxs_a[0], idxs_b[0])
                        elif len(idxs_a) <= 64 and len(idxs_b) <= 64:
                            for ia in idxs_a:
                                for ib in idxs_b:
                                    if _thumbs_match(
                                        thumbs[ia],
                                        thumbs[ib],
                                        max_mad=max_thumb_mad,
                                        min_cos=min_thumb_cos,
                                    ):
                                        uf.union(ia, ib)

    raw_roots = [uf.find(i) for i in range(n)]
    root_to_cid: dict[int, int] = {}
    cids: list[int] = []
    for r in raw_roots:
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
        cids.append(root_to_cid[r])
    work["cluster_id"] = cids

    # Check for cross-class collisions (REAL and FAKE inside same perceptual cluster)
    cls_counts = work.groupby("cluster_id")["label"].nunique()
    bad_cids = set(cls_counts[cls_counts > 1].index.tolist())
    if bad_cids:
        logger.warning(
            "Dropping %d row(s) across %d cross-class near-duplicate cluster(s)",
            int(work["cluster_id"].isin(bad_cids).sum()),
            len(bad_cids),
        )
        work = work[~work["cluster_id"].isin(bad_cids)].reset_index(drop=True)

    work = work.drop(columns=["_thumb16"], errors="ignore")
    cluster_sizes = work["cluster_id"].value_counts()
    near_dup_clusters = int((cluster_sizes > 1).sum())
    stats = {
        "input_rows": int(n_in),
        "exact_sha_dropped": int(n_in - after_sha),
        "exact_pixel_dropped": int(after_sha - after_pix),
        "cross_class_conflicts_dropped": int(len(bad_cids)),
        "output_rows": int(len(work)),
        "unique_clusters": int(work["cluster_id"].nunique()),
        "near_dup_clusters": near_dup_clusters,
        "near_dup_extra_rows": int((cluster_sizes - 1).clip(lower=0).sum()),
    }
    return work, stats


def _split_cluster_ids(
    cids: list[int], train_frac: float, val_frac: float, rng: random.Random
) -> tuple[set[int], set[int], set[int]]:
    items = list(cids)
    rng.shuffle(items)
    n = len(items)
    if n <= 1:
        return set(items), set(), set()
    if n == 2:
        return {items[0]}, set(), {items[1]}
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * (1.0 - train_frac - val_frac))))
    if n_val + n_test >= n:
        n_test = 1
        n_val = 1 if n >= 3 else 0
    val_set = set(items[:n_val])
    test_set = set(items[n_val : n_val + n_test])
    train_set = set(items[n_val + n_test :])
    return train_set, val_set, test_set


def assign_generalization_splits(
    df: pd.DataFrame,
    *,
    val_generators: tuple[str, ...] = ("flux",),
    test_generators: tuple[str, ...] = ("midjourney", "gpt_image_1"),
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign cluster- and generator-family-disjoint train / val / held_out_test splits."""
    val_fams = {normalize_generator_family(g, "FAKE") for g in val_generators}
    test_fams = {normalize_generator_family(g, "FAKE") for g in test_generators}
    if val_fams & test_fams:
        raise ValueError(
            f"val_generators and test_generators overlap on family: {val_fams & test_fams}"
        )

    work = df.copy()
    if "generator_family" not in work.columns:
        work["generator_family"] = [
            normalize_generator_family(str(g), str(c))
            for g, c in zip(work["generator"], work["class_name"])
        ]

    cluster_split: dict[int, str] = {}

    # Assign FAKE clusters by generator_family with strict priority (held_out_test > val > train)
    fake_df = work[work["label"] == 1]
    seen_fake_fams: dict[str, set[int]] = {}
    for cid, grp in fake_df.groupby("cluster_id"):
        fams = set(grp["generator_family"].astype(str).tolist())
        if fams & test_fams:
            cluster_split[int(cid)] = "held_out_test"
        elif fams & val_fams:
            cluster_split[int(cid)] = "val"
        else:
            primary_fam = sorted(fams)[0]
            seen_fake_fams.setdefault(primary_fam, set()).add(int(cid))

    for idx, (fam, cid_set) in enumerate(sorted(seen_fake_fams.items())):
        rng = random.Random(seed + 100 + idx)
        tr, va, te = _split_cluster_ids(sorted(cid_set), 0.70, 0.10, rng)
        for cid in tr:
            cluster_split[cid] = "train"
        for cid in va:
            cluster_split[cid] = "val"
        for cid in te:
            cluster_split[cid] = "held_out_test"

    # Assign REAL clusters stratified by primary source
    real_df = work[work["label"] == 0]
    for idx, (_, grp) in enumerate(sorted(real_df.groupby("generator"))):
        cids = sorted(int(c) for c in grp["cluster_id"].unique().tolist() if int(c) not in cluster_split)
        rng = random.Random(seed + 500 + idx)
        tr, va, te = _split_cluster_ids(cids, 0.60, 0.15, rng)
        for cid in tr:
            cluster_split[cid] = "train"
        for cid in va:
            cluster_split[cid] = "val"
        for cid in te:
            cluster_split[cid] = "held_out_test"

    work["split"] = [cluster_split[int(c)] for c in work["cluster_id"]]
    roles: list[str] = []
    for sp, lbl, fam in zip(work["split"], work["label"], work["generator_family"]):
        if sp == "train":
            roles.append("train")
        elif int(lbl) == 0:
            roles.append("val_real" if sp == "val" else "test_real")
        elif fam in test_fams:
            roles.append("unseen_test_generator")
        elif fam in val_fams:
            roles.append("unseen_val_generator")
        elif sp == "val":
            roles.append("seen_val_generator")
        else:
            roles.append("seen_test_generator")
    work["split_role"] = roles
    verify_prepared_splits(work, val_families=val_fams, test_families=test_fams)
    return work


def verify_prepared_splits(
    df: pd.DataFrame,
    *,
    val_families: Optional[set[str]] = None,
    test_families: Optional[set[str]] = None,
) -> None:
    """Assert zero sha256, pixel_md5, cluster_id, or held-out generator-family leakage."""
    if len(df) > 0 and int(df.groupby("cluster_id")["label"].nunique().max()) > 1:
        raise RuntimeError("Prepared split leak: cross-class cluster_id remains in dataset")
    for col in ("sha256", "pixel_md5", "cluster_id"):
        groups = {s: set(g[col]) for s, g in df.groupby("split")}
        names = sorted(groups)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = groups[a] & groups[b]
                if overlap:
                    raise RuntimeError(
                        f"Prepared split leak: {len(overlap)} {col} shared between {a} and {b}"
                    )

    if test_families:
        bad = df[
            df["generator_family"].isin(test_families)
            & (df["split"] != "held_out_test")
        ]
        if len(bad) > 0:
            raise RuntimeError(
                f"Test generator family leaked outside held_out_test: {bad['generator_family'].unique().tolist()}"
            )
    if val_families:
        bad = df[df["generator_family"].isin(val_families) & (df["split"] != "val")]
        if len(bad) > 0:
            raise RuntimeError(
                f"Validation generator family leaked outside val: {bad['generator_family'].unique().tolist()}"
            )


def compute_matched_sampling_weights(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Compute inverse-frequency weights balancing REAL/FAKE across (res_bucket, aspect_bucket) cells."""
    weights = np.ones(len(df), dtype=np.float64)
    if len(df) == 0:
        return pd.Series(weights, index=df.index), {"matched_cells": 0, "matched_row_coverage": 0.0}

    work = df.copy()
    if "has_face" in work.columns:
        face_mask = work["has_face"].astype(bool).to_numpy()
    else:
        face_mask = np.ones(len(work), dtype=bool)
    train_mask = (work["split"] == "train").to_numpy() & face_mask

    matched_cells = 0
    unmatched_cells: list[str] = []
    cell_stats: list[dict[str, Any]] = []

    sub_idx = np.where(train_mask)[0]
    sub_df = work.iloc[sub_idx]
    for (rb, ab), grp in sub_df.groupby(["res_bucket", "aspect_bucket"]):
        r_idx = grp.index[grp["label"] == 0].to_numpy()
        f_idx = grp.index[grp["label"] == 1].to_numpy()
        cell_name = f"{rb}:{ab}"
        cell_stats.append(
            {
                "cell": cell_name,
                "n_real": int(len(r_idx)),
                "n_fake": int(len(f_idx)),
            }
        )
        if len(r_idx) > 0 and len(f_idx) > 0:
            matched_cells += 1
            target = float(min(len(r_idx), len(f_idx)))
            weights[r_idx] = target / float(len(r_idx))
            weights[f_idx] = target / float(len(f_idx))
        else:
            unmatched_cells.append(cell_name)
            weights[grp.index.to_numpy()] = 0.1

    if len(sub_idx) > 0 and weights[sub_idx].mean() > 0:
        weights[sub_idx] = weights[sub_idx] / weights[sub_idx].mean()

    matched_rows = sum(
        c["n_real"] + c["n_fake"] for c in cell_stats if c["n_real"] > 0 and c["n_fake"] > 0
    )
    info = {
        "train_face_rows": int(len(sub_idx)),
        "matched_cells": matched_cells,
        "unmatched_cells": unmatched_cells,
        "matched_row_coverage": float(matched_rows / max(1, len(sub_idx))),
        "cells": cell_stats,
    }
    return pd.Series(np.round(weights, 4), index=df.index), info


def _run_metadata_probe_df(df: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Fit 5-fold CV classifier strictly on non-semantic image metadata to quantify shortcuts."""
    if len(df) < 20 or df["label"].nunique() < 2:
        return {"n": int(len(df)), "auroc": None, "accuracy": None, "top_features": []}

    X = df[METADATA_FEATURES].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return {"n": int(len(df)), "auroc": None, "accuracy": None, "top_features": []}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    first_fold_imp: Optional[list[dict[str, Any]]] = None

    min_leaf = max(2, min(20, len(y) // 10))
    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        clf = HistGradientBoostingClassifier(
            max_depth=5,
            max_iter=100,
            min_samples_leaf=min_leaf,
            random_state=seed + fold_idx,
        )
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        if fold_idx == 0:
            imp = permutation_importance(
                clf, X[va], y[va], scoring="roc_auc", n_repeats=5, random_state=seed
            )
            pairs = sorted(
                zip(METADATA_FEATURES, imp.importances_mean),
                key=lambda kv: kv[1],
                reverse=True,
            )
            first_fold_imp = [
                {"feature": k, "auc_drop": round(float(v), 4)} for k, v in pairs
            ]

    return {
        "n": int(len(y)),
        "auroc": float(roc_auc_score(y, oof)),
        "accuracy": float(((oof >= 0.5).astype(int) == y).mean()),
        "top_features": first_fold_imp or [],
    }


def harmonize_image_pair_safe(
    img: Image.Image,
    rng: random.Random,
    *,
    target_max_sides: tuple[int, ...] = (384, 512, 768, 1024),
    allow_aspect_crop: bool = True,
) -> tuple[Image.Image, bytes, str]:
    """Class-agnostic resize, aspect-crop, metadata strip, and codec re-encoding."""
    work = img.convert("RGB")
    w, h = work.size

    if allow_aspect_crop and min(w, h) >= 128:
        # Shared aspect distribution across both classes: 65% square, 20% portrait, 15% landscape
        r = rng.random()
        if r < 0.65:
            target_ar = 1.0
        elif r < 0.85:
            target_ar = rng.uniform(0.75, 0.85)
        else:
            target_ar = rng.uniform(1.20, 1.40)

        cur_ar = float(w) / max(1.0, float(h))
        if cur_ar > target_ar * 1.03:
            nw = max(64, int(round(h * target_ar)))
            left = rng.randint(0, max(0, w - nw))
            work = work.crop((left, 0, left + nw, h))
        elif cur_ar < target_ar * 0.97:
            nh = max(64, int(round(w / target_ar)))
            top = rng.randint(0, max(0, h - nh))
            work = work.crop((0, top, w, top + nh))

    w, h = work.size
    target_max = rng.choice(target_max_sides)
    if max(w, h) > target_max:
        scale = target_max / float(max(w, h))
        nw = max(32, int(round(w * scale)))
        nh = max(32, int(round(h * scale)))
        interp = rng.choice(
            [
                Image.Resampling.LANCZOS,
                Image.Resampling.BILINEAR,
                Image.Resampling.BICUBIC,
            ]
        )
        work = work.resize((nw, nh), interp)

    # Shared codec mix strips EXIF/C2PA and equalizes format/compression statistics
    p = rng.random()
    buf = io.BytesIO()
    if p < 0.65:
        fmt = "JPEG"
        work.save(buf, format="JPEG", quality=rng.randint(45, 95))
    elif p < 0.85:
        fmt = "WEBP"
        work.save(buf, format="WEBP", quality=rng.randint(65, 95))
    else:
        fmt = "PNG"
        work.save(buf, format="PNG")
    raw_out = buf.getvalue()
    with Image.open(io.BytesIO(raw_out)) as reopened:
        out_img = reopened.convert("RGB")
    return out_img, raw_out, fmt


def _harmonized_probe_on_subset(
    sub_df: pd.DataFrame,
    *,
    target_max_sides: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    if len(sub_df) < 20 or sub_df["label"].nunique() < 2:
        return {"n": int(len(sub_df)), "raw": None, "harmonized": None}

    raw_probe = _run_metadata_probe_df(sub_df, seed=seed)
    rng = random.Random(seed)
    harm_rows: list[dict[str, Any]] = []
    for row in sub_df.itertuples(index=False):
        p = Path(str(row.path))
        abs_p = p if p.is_absolute() else (PROJECT_ROOT / p)
        with Image.open(abs_p) as im:
            im.load()
            rgb = to_rgb_safe(ImageOps.exif_transpose(im))
        h_img, h_bytes, h_fmt = harmonize_image_pair_safe(
            rgb, rng, target_max_sides=target_max_sides, allow_aspect_crop=True
        )
        w, h = h_img.size
        harm_rows.append(
            {
                "label": int(row.label),
                "width": int(w),
                "height": int(h),
                "aspect_ratio": float(w) / max(1.0, float(h)),
                "max_side": int(max(w, h)),
                "min_side": int(min(w, h)),
                "file_bytes": int(len(h_bytes)),
                "bpp": float(len(h_bytes)) * 8.0 / max(1, w * h),
                "format_code": _format_to_code(h_fmt),
                "has_exif": False,
            }
        )
    harm_df = pd.DataFrame(harm_rows)
    harm_probe = _run_metadata_probe_df(harm_df, seed=seed + 7)
    return {
        "n": int(len(sub_df)),
        "raw": raw_probe,
        "harmonized": harm_probe,
    }


def _sample_balanced_classes(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    c0 = df[df["label"] == 0]
    c1 = df[df["label"] == 1]
    k = min(per_class, len(c0), len(c1))
    if k == 0:
        return df.iloc[0:0]
    s0 = c0.sample(n=k, random_state=seed)
    s1 = c1.sample(n=k, random_state=seed + 1)
    return pd.concat([s0, s1], ignore_index=True)


def verify_harmonized_leakage(
    df: pd.DataFrame,
    sample_per_class: int = 400,
    seed: int = SEED,
) -> dict[str, Any]:
    """Verify raw vs. harmonized metadata shortcut AUROC on balanced face-detected samples."""
    face_df = df[df["has_face"].astype(bool)] if "has_face" in df.columns else df
    mixed_pool = _sample_balanced_classes(face_df, sample_per_class, seed=seed)
    hires_pool = _sample_balanced_classes(
        face_df[face_df["min_side"] >= 512], sample_per_class, seed=seed + 10
    )
    return {
        "mixed_face": _harmonized_probe_on_subset(
            mixed_pool, target_max_sides=(384, 512, 768, 1024), seed=seed + 20
        ),
        "hires_face_512plus": _harmonized_probe_on_subset(
            hires_pool, target_max_sides=(384, 512), seed=seed + 30
        ),
    }


def audit_shortcut_leakage(
    df: pd.DataFrame, *, harmonize_n: int = 350, seed: int = SEED
) -> dict[str, Any]:
    """Compute full metadata-shortcut AUROC, per-source profiles, and harmonized checks."""
    face_df = df[df["has_face"].astype(bool)] if "has_face" in df.columns else df
    raw_all = _run_metadata_probe_df(df, seed=seed)
    raw_face = _run_metadata_probe_df(face_df, seed=seed + 1)
    harm = verify_harmonized_leakage(df, sample_per_class=harmonize_n, seed=seed + 10)
    mixed_harm = harm["mixed_face"]
    hires_harm = harm["hires_face_512plus"]

    source_profiles: list[dict[str, Any]] = []
    for (cls, gen), grp in sorted(df.groupby(["class_name", "generator"])):
        ars = grp["aspect_ratio"].to_numpy(dtype=float)
        source_profiles.append(
            {
                "class_name": cls,
                "generator": gen,
                "generator_family": str(grp["generator_family"].iloc[0]),
                "n_total": int(len(grp)),
                "n_face": int(grp["has_face"].astype(bool).sum()) if "has_face" in grp.columns else int(len(grp)),
                "formats": grp["file_format"].value_counts().to_dict(),
                "med_max_side": int(grp["max_side"].median()),
                "p10_max_side": int(np.percentile(grp["max_side"], 10)),
                "p90_max_side": int(np.percentile(grp["max_side"], 90)),
                "square_pct": round(100.0 * float(np.mean(np.abs(ars - 1.0) <= 0.05)), 1),
                "med_bpp": round(float(grp["bpp"].median()), 2),
                "exif_pct": round(100.0 * float(grp["has_exif"].astype(bool).mean()), 1),
            }
        )

    return {
        "raw_all": raw_all,
        "raw_face": raw_face,
        "harmonized_mixed_face": mixed_harm,
        "harmonized_hires_face_512plus": hires_harm,
        "source_profiles": source_profiles,
    }
