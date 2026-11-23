"""Organize, deduplicate, split by generator family, and audit shortcut leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data import MULTI_MANIFEST_PATH  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    PUBLIC_DATASETS,
    assign_generalization_splits,
    audit_shortcut_leakage,
    cluster_and_deduplicate,
    compute_matched_sampling_weights,
    ingest_existing_manifest,
    ingest_genimage_dir,
    ingest_synthbuster_dir,
    ingest_user_folder,
    ingest_wildrf_dir,
)
from src.utils import DATA_DIR, REPORTS_DIR, SEED, save_json  # noqa: E402


def _parse_named_dir(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty source name in {spec!r}")
    return name, Path(raw_path).expanduser()


def _print_public_datasets() -> None:
    print("Supported Public Datasets & Licenses:")
    print("-" * 80)
    for d in PUBLIC_DATASETS:
        print(f"- {d['name']}")
        print(f"  Source      : {d['source_url']}")
        print(f"  License     : {d['license']}")
        print(f"  Availability: {d['availability']}")
        print(f"  Generators  : {d['generators']}")
        print(f"  Real sources: {d['real_sources']}")
        print(f"  Ingest flag : {d['cli_flag']}")


def _fmt_num(v: Optional[float], decimals: int = 4) -> str:
    return "-" if v is None else f"{v:.{decimals}f}"


def _write_audit_markdown(
    out_path: Path,
    *,
    df: pd.DataFrame,
    dedup_stats: dict[str, Any],
    weight_info: dict[str, Any],
    leak_audit: dict[str, Any],
    val_generators: tuple[str, ...],
    test_generators: tuple[str, ...],
) -> None:
    lines: list[str] = []
    lines.append("# Data Pipeline & Shortcut Leakage Audit\n")
    lines.append("## 1. Deduplication & Near-Duplicate Clustering\n")
    lines.append(f"- Input rows: `{dedup_stats['input_rows']:,}`")
    lines.append(f"- Exact `sha256` duplicates dropped: `{dedup_stats['exact_sha_dropped']:,}`")
    lines.append(f"- Exact `pixel_md5` duplicates dropped: `{dedup_stats['exact_pixel_dropped']:,}`")
    lines.append(f"- Cross-class `dHash` conflict clusters dropped: `{dedup_stats['cross_class_conflicts_dropped']:,}`")
    lines.append(
        f"- Output unique images: **`{dedup_stats['output_rows']:,}`** grouped into **`{dedup_stats['unique_clusters']:,}`** perceptual clusters "
        f"(`{dedup_stats['near_dup_clusters']:,}` multi-image clusters keeping resized/recompressed variants in the same split)\n"
    )

    lines.append("## 2. Leak-Free Split Allocation\n")
    lines.append(f"- Held-out validation generator family (for early stopping / OOD model selection / calibration): `{list(val_generators)}`")
    lines.append(f"- Frozen held-out test generator families (never used for training or tuning): `{list(test_generators)}`")
    lines.append("- Note: `FLUX` and `FLUX_OpenFake` are grouped into canonical family `flux` so related models never leak across splits.\n")
    lines.append("| Split | Role | Class | Generators / Sources | Images | Face-Detected |")
    lines.append("|---|---|---|---:|---:|---:|")
    for (sp, role, cls), grp in df.groupby(["split", "split_role", "class_name"]):
        n_face = int(grp["has_face"].astype(bool).sum())
        gens = ", ".join(sorted(grp["generator"].astype(str).unique().tolist()))
        lines.append(f"| `{sp}` | `{role}` | {cls} | {gens} | {len(grp):,} | {n_face:,} |")

    lines.append("\n## 3. Metadata-Only Shortcut Leakage Probe (HistGradientBoosting 5-Fold CV)\n")
    lines.append("Features tested: `width, height, aspect_ratio, max_side, min_side, file_bytes, bpp, format_code, has_exif` (no pixel values).\n")
    lines.append("| Subset / Condition | n | Metadata-Only AUROC | Metadata-Only Acc | Top Leaking Features |")
    lines.append("|---|---:|---:|---:|---|")

    def _feat_str(rec: Optional[dict[str, Any]]) -> str:
        if not rec or not rec.get("top_features"):
            return "-"
        return ", ".join(f"`{f['feature']}` ({f['auc_drop']:+.3f})" for f in rec["top_features"][:4])

    for label, rec in [
        ("Raw manifest (all rows)", leak_audit["raw_all"]),
        ("Raw manifest (face-detected)", leak_audit["raw_face"]),
    ]:
        if rec:
            lines.append(
                f"| {label} | {rec['n']:,} | **`{_fmt_num(rec['auroc'])}`** | {_fmt_num(rec['accuracy'])} | {_feat_str(rec)} |"
            )

    hm = leak_audit["harmonized_mixed_face"]
    if hm and hm.get("raw") and hm.get("harmonized"):
        lines.append(
            f"| Sampled face-detected mixed-res (Raw) | {hm['n']:,} | `{_fmt_num(hm['raw']['auroc'])}` | {_fmt_num(hm['raw']['accuracy'])} | {_feat_str(hm['raw'])} |"
        )
        lines.append(
            f"| Sampled face-detected mixed-res (Harmonized `384-1024`) | {hm['n']:,} | `{_fmt_num(hm['harmonized']['auroc'])}` | {_fmt_num(hm['harmonized']['accuracy'])} | {_feat_str(hm['harmonized'])} |"
        )

    hh = leak_audit["harmonized_hires_face_512plus"]
    if hh and hh.get("raw") and hh.get("harmonized"):
        lines.append(
            f"| Sampled >=512px face-detected (Raw) | {hh['n']:,} | `{_fmt_num(hh['raw']['auroc'])}` | {_fmt_num(hh['raw']['accuracy'])} | {_feat_str(hh['raw'])} |"
        )
        lines.append(
            f"| **Sampled >=512px face-detected (Harmonized `384/512`)** | {hh['n']:,} | **`{_fmt_num(hh['harmonized']['auroc'])}`** | {_fmt_num(hh['harmonized']['accuracy'])} | {_feat_str(hh['harmonized'])} |"
        )

    lines.append("\n### Resolution × Aspect Cell Matching (`train`, face-detected)\n")
    lines.append(
        f"- Matched cells containing both REAL and FAKE: **`{weight_info['matched_cells']}`** covering **`{100 * weight_info['matched_row_coverage']:.1f}%`** of face-detected train rows"
    )
    if weight_info["unmatched_cells"]:
        lines.append(f"- Unmatched single-class cells (downweighted in training sampler): `{weight_info['unmatched_cells']}`")
    lines.append("\n| Resolution × Aspect Cell | REAL (train faces) | FAKE (train faces) |")
    lines.append("|---|---:|---:|")
    for c in weight_info["cells"]:
        lines.append(f"| `{c['cell']}` | {c['n_real']:,} | {c['n_fake']:,} |")

    lines.append("\n## 4. Per-Source Format, Resolution, Aspect & Compression Profile\n")
    lines.append("| Class | Source | Family | n (Face) | Formats | Median MaxSide (p10-p90) | Square % | Median bpp | EXIF % |")
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|")
    for sp in leak_audit["source_profiles"]:
        fmts = ", ".join(f"{k}:{v}" for k, v in sp["formats"].items())
        lines.append(
            f"| {sp['class_name']} | `{sp['generator']}` | `{sp['generator_family']}` | "
            f"{sp['n_total']:,} ({sp['n_face']:,}) | {fmts} | "
            f"{sp['med_max_side']} ({sp['p10_max_side']}-{sp['p90_max_side']}) | "
            f"{sp['square_pct']:.1f}% | {sp['med_bpp']:.2f} | {sp['exif_pct']:.1f}% |"
        )

    lines.append("\n## 5. Supported Public Datasets & Licenses\n")
    lines.append("| Dataset | License | Generators | Real Sources | How to Ingest |")
    lines.append("|---|---|---|---|---|")
    for d in PUBLIC_DATASETS:
        lines.append(
            f"| **{d['name']}** | {d['license']} | {d['generators']} | {d['real_sources']} | `{d['cli_flag']}` |"
        )

    lines.append("\n## 6. Key Findings & Gaps\n")
    lines.append(
        "1. Every image in `data/multi_manifest.csv` is currently saved as JPEG `quality=95` without EXIF, but **aspect ratio, `max_side`, and `bpp` still allow metadata alone to separate REAL from FAKE with near-perfect accuracy** because CelebA (`178×218`), FFHQ/StyleGAN (`256×256`), COCO (`640×480`), and 1024px generators have disjoint size/shape signatures.\n"
        "2. In `mixed` face-detected data, low-res sources (`CelebA 178×218`, `FFHQ 256×256`, `StyleGAN 256×256`) cannot reach high-res targets (`>=512px`) without upscaling. When evaluating sources with `min_side >= 512px` under class-agnostic harmonization (`384/512px` + aspect crop + shared JPEG/WebP/PNG), the metadata-only shortcut collapses toward random chance (`~0.50` AUROC).\n"
        "3. **Current real-photo diversity gap:** General in-the-wild real photos (`OpenFake_Real`: 700, `COCO`: 472 faces) are small compared to uniform face crops (`FFHQ`: 6,000, `CelebA`: 3,499). Adding your own phone camera, DSLR, and social-media compressed real photos via `--user-real-dir` (or RAISE-1k / WildRF) is strongly recommended before Step 3 training."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated, generator-split VeriFake manifest and audit shortcuts.")
    parser.add_argument("--list-public-datasets", action="store_true")
    parser.add_argument("--source-manifest", type=Path, default=MULTI_MANIFEST_PATH)
    parser.add_argument("--user-real-dir", action="append", default=[], help="NAME=PATH for real photos folder")
    parser.add_argument("--user-fake-dir", action="append", default=[], help="NAME=PATH for AI generator folder")
    parser.add_argument("--genimage-dir", type=Path, default=None)
    parser.add_argument("--synthbuster-dir", type=Path, default=None)
    parser.add_argument("--raise1k-dir", type=Path, default=None)
    parser.add_argument("--wildrf-dir", type=Path, default=None)
    parser.add_argument("--val-generators", nargs="+", default=["flux"])
    parser.add_argument("--test-generators", nargs="+", default=["midjourney", "gpt_image_1"])
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--harmonize-audit-n", type=int, default=350)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-manifest", type=Path, default=DATA_DIR / "prepared_manifest.csv")
    parser.add_argument("--report-out", type=Path, default=REPORTS_DIR / "data_audit.md")
    parser.add_argument("--json-out", type=Path, default=REPORTS_DIR / "data_audit.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.list_public_datasets:
        _print_public_datasets()
        return

    frames: list[pd.DataFrame] = []
    if args.source_manifest is not None and args.source_manifest.exists():
        frames.append(ingest_existing_manifest(args.source_manifest, workers=args.workers))
    for spec in args.user_real_dir:
        name, path = _parse_named_dir(spec)
        frames.append(ingest_user_folder(path, class_name="REAL", source_name=name))
    for spec in args.user_fake_dir:
        name, path = _parse_named_dir(spec)
        frames.append(ingest_user_folder(path, class_name="FAKE", source_name=name))
    if args.genimage_dir is not None:
        frames.append(ingest_genimage_dir(args.genimage_dir))
    if args.synthbuster_dir is not None:
        frames.append(ingest_synthbuster_dir(args.synthbuster_dir, raise1k_dir=args.raise1k_dir))
    if args.wildrf_dir is not None:
        frames.append(ingest_wildrf_dir(args.wildrf_dir))

    frames = [f for f in frames if len(f) > 0]
    if not frames:
        raise RuntimeError("No images ingested from manifest or user folders.")

    combined = pd.concat(frames, ignore_index=True)
    deduped, dedup_stats = cluster_and_deduplicate(combined, max_hamming=args.dhash_threshold)
    prepared = assign_generalization_splits(
        deduped,
        val_generators=tuple(args.val_generators),
        test_generators=tuple(args.test_generators),
        seed=args.seed,
    )
    weights, weight_info = compute_matched_sampling_weights(prepared)
    prepared["match_weight"] = weights

    leak_audit = audit_shortcut_leakage(
        prepared, harmonize_n=args.harmonize_audit_n, seed=args.seed
    )

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.out_manifest, index=False)
    _write_audit_markdown(
        args.report_out,
        df=prepared,
        dedup_stats=dedup_stats,
        weight_info=weight_info,
        leak_audit=leak_audit,
        val_generators=tuple(args.val_generators),
        test_generators=tuple(args.test_generators),
    )
    save_json(
        {
            "dedup": dedup_stats,
            "val_generators": list(args.val_generators),
            "test_generators": list(args.test_generators),
            "split_counts": prepared.groupby(["split", "split_role", "class_name"])
            .size()
            .reset_index(name="count")
            .to_dict("records"),
            "matched_sampling": weight_info,
            "leakage_audit": leak_audit,
            "public_datasets": PUBLIC_DATASETS,
        },
        args.json_out,
    )

    print(f"Prepared manifest : {args.out_manifest} ({len(prepared):,} rows, {dedup_stats['unique_clusters']:,} clusters)")
    print(f"Dedup summary     : {dedup_stats}")
    print(f"Metadata AUROC    : all={_fmt_num(leak_audit['raw_all']['auroc'])}, face={_fmt_num(leak_audit['raw_face']['auroc'])}")
    hh = leak_audit.get("harmonized_hires_face_512plus") or {}
    if hh.get("raw") and hh.get("harmonized"):
        print(
            f">=512px shortcut   : raw AUROC={_fmt_num(hh['raw']['auroc'])} -> harmonized AUROC={_fmt_num(hh['harmonized']['auroc'])}"
        )
    print(f"Audit report      : {args.report_out}")


if __name__ == "__main__":
    main()

