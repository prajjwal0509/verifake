"""Populate gitignored symlinked eval folders (`data/eval/`) and disjoint `data/val_real/`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data import MULTI_MANIFEST_PATH, verify_no_leakage  # noqa: E402
from src.utils import DATA_DIR  # noqa: E402

UNSEEN_BY_CURRENT_MODEL: frozenset[str] = frozenset(
    {"midjourney", "gpt_image_1", "flux_openfake", "openfake_real"}
)


def _is_true_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.strip().str.lower() == "true"


def _ensure_symlink(src_abs: Path, dst_abs: Path) -> str:
    dst_abs.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(src_abs, dst_abs.parent)
    if dst_abs.is_symlink():
        if os.readlink(dst_abs) == rel_target and dst_abs.exists():
            return "unchanged"
        dst_abs.unlink()
        os.symlink(rel_target, dst_abs)
        return "updated"
    if dst_abs.exists():
        return "kept_regular_file"
    os.symlink(rel_target, dst_abs)
    return "created"


def _check_split_isolation(df: pd.DataFrame) -> None:
    verify_no_leakage(df)
    train_val = df[df["split"].isin(["train", "val"])]
    held_out = df[df["split"].isin(["test", "unseen_generator"])]
    for col in ("sha256", "pixel_md5"):
        overlap = set(train_val[col]) & set(held_out[col])
        if overlap:
            raise RuntimeError(
                f"Leakage detected: {len(overlap)} {col} shared between train/val and held-out splits"
            )
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    for col in ("sha256", "pixel_md5"):
        overlap = set(val_df[col]) & set(test_df[col])
        if overlap:
            raise RuntimeError(
                f"Leakage detected: {len(overlap)} {col} shared between val and test"
            )


def _populate_group(
    sub_df: pd.DataFrame, dst_dir: Path, stats: dict[str, int]
) -> tuple[int, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for row_idx, row in enumerate(sub_df.itertuples(index=False)):
        src_abs = PROJECT_ROOT / str(row.path)
        if not src_abs.exists():
            raise FileNotFoundError(f"Manifest image missing on disk: {src_abs}")
        dst_abs = dst_dir / src_abs.name
        if dst_abs.exists() and not dst_abs.is_symlink():
            pass
        elif dst_abs.is_symlink() and Path(os.path.realpath(dst_abs)) != src_abs.resolve():
            dst_abs = dst_dir / f"{row_idx}_{src_abs.name}"
        action = _ensure_symlink(src_abs, dst_abs)
        stats[action] = stats.get(action, 0) + 1
    med_w = int(round(sub_df["width"].median())) if len(sub_df) else 0
    med_h = int(round(sub_df["height"].median())) if len(sub_df) else 0
    return len(sub_df), f"{med_w}x{med_h}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create symlinked data/eval/ and data/val_real/ folders from multi_manifest.csv."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MULTI_MANIFEST_PATH,
        help="Path to multi-generator manifest CSV (default: data/multi_manifest.csv).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DATA_DIR / "eval",
        help="Target eval folder root (default: data/eval).",
    )
    parser.add_argument(
        "--val-real-dir",
        type=Path,
        default=DATA_DIR / "val_real",
        help="Disjoint validation real-photo root for FPR calibration (default: data/val_real).",
    )
    parser.add_argument(
        "--include-no-face",
        action="store_true",
        help="Include rows where has_face == False as well (default: face-detected only).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    _check_split_isolation(df)

    if not args.include_no_face and "has_face" in df.columns:
        work = df[_is_true_series(df["has_face"])].copy()
    else:
        work = df.copy()

    stats: dict[str, int] = {}
    inventory: list[tuple[str, str, str, int, str, str]] = []

    # 1. data/eval/real/<subsource>/ from split == test & class_name == REAL
    eval_real = work[(work["split"] == "test") & (work["class_name"] == "REAL")]
    for subsource, grp in sorted(eval_real.groupby("generator")):
        slug = str(subsource).lower()
        n, med = _populate_group(grp, args.eval_dir / "real" / slug, stats)
        seen = "unseen" if slug in UNSEEN_BY_CURRENT_MODEL else "seen"
        inventory.append(("eval/real", slug, "REAL", n, med, seen))

    # 2. data/eval/<generator>/ from split in (test, unseen_generator) & class_name == FAKE
    eval_fake = work[
        work["split"].isin(["test", "unseen_generator"]) & (work["class_name"] == "FAKE")
    ]
    for gen_name, grp in sorted(eval_fake.groupby("generator")):
        slug = str(gen_name).lower()
        n, med = _populate_group(grp, args.eval_dir / slug, stats)
        seen = "unseen" if slug in UNSEEN_BY_CURRENT_MODEL else "seen"
        inventory.append(("eval/fake", slug, "FAKE", n, med, seen))

    # 3. Empty drop-in folder for Gemini images
    gemini_dir = args.eval_dir / "gemini"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    gemini_existing = sum(1 for p in gemini_dir.rglob("*") if p.is_file())
    inventory.append(("eval/fake", "gemini", "FAKE", gemini_existing, "-", "unseen"))

    # 4. Disjoint data/val_real/<subsource>/ from split == val & class_name == REAL
    val_real = work[(work["split"] == "val") & (work["class_name"] == "REAL")]
    for subsource, grp in sorted(val_real.groupby("generator")):
        slug = str(subsource).lower()
        n, med = _populate_group(grp, args.val_real_dir / slug, stats)
        seen = "unseen" if slug in UNSEEN_BY_CURRENT_MODEL else "seen"
        inventory.append(("val_real", slug, "REAL", n, med, seen))

    print("Zero-leakage check passed (sha256 & pixel_md5 disjoint across train/val vs test/unseen_generator).")
    print(f"Eval dir     : {args.eval_dir}")
    print(f"Val-real dir : {args.val_real_dir}")
    print(f"Symlinks     : {stats}\n")
    print(f"{'Group':<11} {'Folder':<16} {'Class':<6} {'Count':>7} {'Median WxH':<12} {'best.pt Status'}")
    print("-" * 68)
    for grp, folder, cls, n, med, seen in inventory:
        print(f"{grp:<11} {folder:<16} {cls:<6} {n:>7,} {med:<12} {seen}")


if __name__ == "__main__":
    main()

