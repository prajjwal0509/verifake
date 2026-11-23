"""Fetch non-gated modern OpenFake generators + InsightFace face-swap portraits into prepared_manifest_v2.csv."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data_pipeline import (  # noqa: E402
    _aspect_bucket,
    _format_to_code,
    _res_bucket,
    compute_dhash64,
)
from src.face_crop import detect_primary_face  # noqa: E402
from src.image_io import register_image_plugins, to_rgb_safe  # noqa: E402
from src.utils import DATA_DIR, SEED, get_logger, seed_everything  # noqa: E402

logger = get_logger("fetch_modern_data")

DEFAULT_BASE_MANIFEST: Path = DATA_DIR / "prepared_manifest.csv"
DEFAULT_OUT_MANIFEST: Path = DATA_DIR / "prepared_manifest_v2.csv"
MODERN_IMG_DIR: Path = DATA_DIR / "images" / "modern_v2"

# Generators strictly forbidden from train so val / held_out_test OOD stays zero-leak.
FORBIDDEN_TRAIN_SUBSTRINGS: tuple[str, ...] = (
    "midjourney",
    "gpt-image",
    "gpt_image",
    "gpt-4o",
    "flux",
)

# Priority modern OpenFake generator mappings -> (generator, generator_family)
MODERN_OPENFAKE_PREFIXES: list[tuple[str, str, str]] = [
    ("imagen-4", "OpenFake_Imagen4", "imagen"),
    ("imagen-3", "OpenFake_Imagen3", "imagen"),
    ("ideogram-", "OpenFake_Ideogram", "ideogram"),
    ("recraft-", "OpenFake_Recraft", "recraft"),
    ("grok-", "OpenFake_Grok", "grok"),
    ("qwen-image", "OpenFake_Qwen", "qwen"),
    ("hidream-", "OpenFake_HiDream", "hidream"),
    ("aurora", "OpenFake_Aurora", "aurora"),
    ("seedream", "OpenFake_Seedream", "seedream"),
    ("sd-3.5", "OpenFake_SD35", "sd3"),
    ("mystic", "OpenFake_Mystic", "mystic"),
    ("chroma", "OpenFake_Chroma", "chroma"),
]


def _classify_openfake_model(model_name: str, label_str: str) -> Optional[tuple[str, str, int]]:
    low = str(model_name or "").strip().lower()
    lbl_low = str(label_str or "").strip().lower()
    if lbl_low == "real":
        return "OpenFake_Real", "real", 0
    if lbl_low != "fake":
        return None
    if any(bad in low for bad in FORBIDDEN_TRAIN_SUBSTRINGS):
        return None
    for prefix, gen_name, fam_name in MODERN_OPENFAKE_PREFIXES:
        if prefix in low:
            return gen_name, fam_name, 1
    return None


def _compute_row_record(
    raw_bytes: bytes,
    *,
    rel_save_path: Path,
    label: int,
    generator: str,
    generator_family: str,
    dataset_source: str,
    split: str,
    split_role: str,
    cluster_id: int,
) -> Optional[dict[str, Any]]:
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            fmt = str(im.format or "JPEG").upper()
            exif = im.getexif()
            has_exif = bool(exif and len(exif) > 0)
            rgb = to_rgb_safe(ImageOps.exif_transpose(im))
    except Exception:
        return None

    w, h = rgb.size
    if min(w, h) < 48:
        return None
    det = detect_primary_face(rgb)
    if det is None:
        return None

    pix_md5 = hashlib.md5(rgb.tobytes()).hexdigest()
    abs_out = PROJECT_ROOT / rel_save_path
    abs_out.parent.mkdir(parents=True, exist_ok=True)
    abs_out.write_bytes(raw_bytes)

    aspect = round(float(w) / float(max(1, h)), 4)
    max_s = max(w, h)
    min_s = min(w, h)
    fbytes = len(raw_bytes)
    bpp = round((fbytes * 8.0) / float(max(1, w * h)), 4)
    dint = int(compute_dhash64(rgb))
    dhex = f"{dint:016x}"

    return {
        "path": str(rel_save_path),
        "label": int(label),
        "class_name": "FAKE" if int(label) == 1 else "REAL",
        "generator": str(generator),
        "dataset_source": str(dataset_source),
        "sha256": sha,
        "pixel_md5": pix_md5,
        "width": int(w),
        "height": int(h),
        "split": str(split),
        "has_face": True,
        "face_score": round(float(det["score"]), 4),
        "file_format": fmt,
        "file_bytes": int(fbytes),
        "aspect_ratio": aspect,
        "max_side": int(max_s),
        "min_side": int(min_s),
        "bpp": bpp,
        "format_code": int(_format_to_code(fmt)),
        "has_exif": bool(has_exif),
        "dhash64": dhex,
        "dhash_int": dint,
        "res_bucket": _res_bucket(max_s),
        "aspect_bucket": _aspect_bucket(w, h),
        "generator_family": str(generator_family),
        "cluster_id": int(cluster_id),
        "split_role": str(split_role),
        "match_weight": 1.0,
    }


def _fetch_sowaiba_faceswap(
    *,
    seen_shas: set[str],
    seen_pmd5: set[str],
    next_cluster_id: int,
    train_fake_target: int,
    train_real_target: int,
    val_fake_target: int,
    val_real_target: int,
) -> tuple[list[dict[str, Any]], int]:
    """Download non-gated Sowaiba01/Deepfake InsightFace face-swaps & paired reals with identity/pair split."""
    api = HfApi()
    files = api.list_repo_files("Sowaiba01/Deepfake", repo_type="dataset")
    fakes = sorted([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and f.lower().startswith("fake/")])
    reals = sorted([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and f.lower().startswith("real/")])
    logger.info("Sowaiba01/Deepfake inventory: %d fake, %d real files", len(fakes), len(reals))

    # Reserve non-overlapping ID windows so train/val never share paired source photos:
    # Train: first 70% of indices; Val: last 30% of indices (taken sequentially from distinct ends).
    split_f = int(len(fakes) * 0.70)
    split_r = int(len(reals) * 0.70)
    jobs = [
        ("train", "train", 1, "FaceSwap_InsightFace", "faceswap", fakes[:split_f], train_fake_target),
        ("train", "train", 0, "FaceSwap_Real", "real", reals[:split_r], train_real_target),
        ("val", "seen_val_generator", 1, "FaceSwap_InsightFace", "faceswap", fakes[split_f:], val_fake_target),
        ("val", "val_real", 0, "FaceSwap_Real", "real", reals[split_r:], val_real_target),
    ]

    rows: list[dict[str, Any]] = []

    def _download_file(repo_path: str) -> Optional[tuple[str, bytes]]:
        try:
            local_p = hf_hub_download("Sowaiba01/Deepfake", repo_path, repo_type="dataset")
            return repo_path, Path(local_p).read_bytes()
        except Exception:
            return None

    for split, role, lbl, gen, fam, candidates, quota in jobs:
        if quota <= 0 or not candidates:
            continue
        collected = 0
        # Pull in slightly oversized batches so face-gating misses still hit quota cleanly
        idx = 0
        while collected < quota and idx < len(candidates):
            need = max(8, int((quota - collected) * 1.35))
            batch = candidates[idx : idx + need]
            idx += len(batch)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(_download_file, rp) for rp in batch]
                for fut in as_completed(futs):
                    if collected >= quota:
                        continue
                    res = fut.result()
                    if res is None:
                        continue
                    rp, raw = res
                    sha = hashlib.sha256(raw).hexdigest()
                    if sha in seen_shas:
                        continue
                    ext = Path(rp).suffix.lower() or ".jpg"
                    rel_path = Path("data/images/modern_v2/faceswap") / split / gen / f"{sha[:24]}{ext}"
                    rec = _compute_row_record(
                        raw,
                        rel_save_path=rel_path,
                        label=lbl,
                        generator=gen,
                        generator_family=fam,
                        dataset_source="Sowaiba01_Deepfake_InsightFace",
                        split=split,
                        split_role=role,
                        cluster_id=next_cluster_id,
                    )
                    if rec is None or rec["pixel_md5"] in seen_pmd5:
                        if (PROJECT_ROOT / rel_path).exists():
                            (PROJECT_ROOT / rel_path).unlink(missing_ok=True)
                        continue
                    seen_shas.add(rec["sha256"])
                    seen_pmd5.add(rec["pixel_md5"])
                    next_cluster_id += 1
                    rows.append(rec)
                    collected += 1
        logger.info("  Collected %s/%s (%s): %d/%d", split, role, gen, collected, quota)
    return rows, next_cluster_id


def _fetch_openfake_modern(
    *,
    seen_shas: set[str],
    seen_pmd5: set[str],
    next_cluster_id: int,
    fake_target: int,
    real_target: int,
    per_gen_cap: int,
    max_pages: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch face-detected modern non-flux/non-heldout OpenFake train images + real photos via HF rows API."""
    rng = np.random.default_rng(seed)
    # Sample diverse offsets across OpenFake/core train (~2.3M rows)
    offsets = rng.choice(np.arange(0, 2_250_000, 100), size=min(max_pages, 500), replace=False)

    def _get_page(off: int) -> list[dict[str, Any]]:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=ComplexDataLab%2FOpenFake&config=core&split=train&offset={int(off)}&length=100"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
            return [r["row"] for r in data.get("rows", [])]
        except Exception:
            return []

    def _download_img(item: tuple[str, str, str, int, str]) -> Optional[tuple[bytes, str, str, int, str]]:
        src_url, gen, fam, lbl, ext = item
        try:
            req = urllib.request.Request(src_url, headers={"User-Agent": "VeriFake-Research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read(), gen, fam, lbl, ext
        except Exception:
            return None

    rows: list[dict[str, Any]] = []
    gen_counts: dict[str, int] = {}
    fake_collected = 0
    real_collected = 0

    for p_idx, off in enumerate(offsets):
        if fake_collected >= fake_target and real_collected >= real_target:
            break
        page_rows = _get_page(int(off))
        if not page_rows:
            continue

        to_fetch: list[tuple[str, str, str, int, str]] = []
        for r in page_rows:
            cls_info = _classify_openfake_model(r.get("model", ""), r.get("label", ""))
            if cls_info is None:
                continue
            gen, fam, lbl = cls_info
            if lbl == 1:
                if fake_collected >= fake_target or gen_counts.get(gen, 0) >= per_gen_cap:
                    continue
            else:
                if real_collected >= real_target:
                    continue
            img_info = r.get("image") or {}
            src_url = img_info.get("src")
            if not src_url:
                continue
            row_sha = str(r.get("image_sha256") or "")
            if row_sha and row_sha in seen_shas:
                continue
            path_part = urllib.parse.urlparse(src_url).path
            ext = Path(path_part).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            to_fetch.append((src_url, gen, fam, lbl, ext))

        if not to_fetch:
            continue

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_download_img, it) for it in to_fetch]
            for fut in as_completed(futs):
                dl = fut.result()
                if dl is None:
                    continue
                raw, gen, fam, lbl, ext = dl
                if lbl == 1 and (fake_collected >= fake_target or gen_counts.get(gen, 0) >= per_gen_cap):
                    continue
                if lbl == 0 and real_collected >= real_target:
                    continue
                sha = hashlib.sha256(raw).hexdigest()
                if sha in seen_shas:
                    continue
                rel_path = Path("data/images/modern_v2/openfake") / "train" / gen / f"{sha[:24]}{ext}"
                rec = _compute_row_record(
                    raw,
                    rel_save_path=rel_path,
                    label=lbl,
                    generator=gen,
                    generator_family=fam,
                    dataset_source="ComplexDataLab_OpenFake_core_train",
                    split="train",
                    split_role="train",
                    cluster_id=next_cluster_id,
                )
                if rec is None or rec["pixel_md5"] in seen_pmd5:
                    if (PROJECT_ROOT / rel_path).exists():
                        (PROJECT_ROOT / rel_path).unlink(missing_ok=True)
                    continue
                seen_shas.add(rec["sha256"])
                seen_pmd5.add(rec["pixel_md5"])
                next_cluster_id += 1
                rows.append(rec)
                if lbl == 1:
                    fake_collected += 1
                    gen_counts[gen] = gen_counts.get(gen, 0) + 1
                else:
                    real_collected += 1

        if (p_idx + 1) % 15 == 0:
            logger.info(
                "  OpenFake scan page %d/%d -> fake=%d/%d real=%d/%d | by_gen=%s",
                p_idx + 1,
                len(offsets),
                fake_collected,
                fake_target,
                real_collected,
                real_target,
                dict(sorted(gen_counts.items())),
            )
        time.sleep(0.05)

    logger.info("Finished OpenFake modern fetch: fake=%d, real=%d | %s", fake_collected, real_collected, dict(sorted(gen_counts.items())))
    return rows, next_cluster_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch modern OpenFake + face-swap datasets cleanly.")
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--faceswap-train-fake", type=int, default=480)
    parser.add_argument("--faceswap-train-real", type=int, default=320)
    parser.add_argument("--faceswap-val-fake", type=int, default=120)
    parser.add_argument("--faceswap-val-real", type=int, default=80)
    parser.add_argument("--openfake-fake-target", type=int, default=1000)
    parser.add_argument("--openfake-real-target", type=int, default=500)
    parser.add_argument("--openfake-per-gen-cap", type=int, default=200)
    parser.add_argument("--openfake-max-pages", type=int, default=140)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    register_image_plugins()
    seed_everything(args.seed)

    base_df = pd.read_csv(args.base_manifest)
    seen_shas = set(base_df["sha256"].astype(str))
    seen_pmd5 = set(base_df["pixel_md5"].astype(str))
    next_cluster_id = int(base_df["cluster_id"].max()) + 1

    fs_rows, next_cluster_id = _fetch_sowaiba_faceswap(
        seen_shas=seen_shas,
        seen_pmd5=seen_pmd5,
        next_cluster_id=next_cluster_id,
        train_fake_target=args.faceswap_train_fake,
        train_real_target=args.faceswap_train_real,
        val_fake_target=args.faceswap_val_fake,
        val_real_target=args.faceswap_val_real,
    )
    of_rows, next_cluster_id = _fetch_openfake_modern(
        seen_shas=seen_shas,
        seen_pmd5=seen_pmd5,
        next_cluster_id=next_cluster_id,
        fake_target=args.openfake_fake_target,
        real_target=args.openfake_real_target,
        per_gen_cap=args.openfake_per_gen_cap,
        max_pages=args.openfake_max_pages,
        seed=args.seed,
    )

    new_df = pd.DataFrame(fs_rows + of_rows)
    if len(new_df) == 0:
        raise RuntimeError("No new modern/faceswap rows fetched!")

    # Preserve exact column order of prepared_manifest.csv
    new_df = new_df[list(base_df.columns)]
    out_df = pd.concat([base_df, new_df], ignore_index=True)

    # Strict cross-split zero-leak assertions
    splits = {}
    for sp in ("train", "val", "held_out_test"):
        sub = out_df[out_df["split"] == sp]
        splits[sp] = (set(sub["sha256"].astype(str)), set(sub["pixel_md5"].astype(str)))

    for s1, s2 in [("train", "val"), ("train", "held_out_test"), ("val", "held_out_test")]:
        assert not (splits[s1][0] & splits[s2][0]), f"sha256 overlap between {s1} and {s2}"
        assert not (splits[s1][1] & splits[s2][1]), f"pixel_md5 overlap between {s1} and {s2}"

    train_gens = " ".join(out_df[out_df["split"] == "train"]["generator"].astype(str).unique()).lower()
    for bad in FORBIDDEN_TRAIN_SUBSTRINGS:
        assert bad not in train_gens, f"Forbidden generator '{bad}' leaked into train!"

    out_df.to_csv(args.out_manifest, index=False)
    logger.info("Wrote %s (%d total rows, +%d new)", args.out_manifest, len(out_df), len(new_df))
    print("\nAdded rows summary:")
    print(new_df.groupby(["split", "split_role", "class_name", "generator", "generator_family"]).size().to_string())


if __name__ == "__main__":
    main()
