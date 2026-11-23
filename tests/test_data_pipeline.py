"""Unit tests for data deduplication, generator-family isolation, adapters, and shortcut mitigation."""

from __future__ import annotations

import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data_pipeline import (
    _run_metadata_probe_df,
    assign_generalization_splits,
    cluster_and_deduplicate,
    harmonize_image_pair_safe,
    ingest_genimage_dir,
    ingest_synthbuster_dir,
    ingest_user_folder,
    ingest_wildrf_dir,
    normalize_generator_family,
    verify_harmonized_leakage,
    verify_prepared_splits,
)


def _make_textured_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    base = np.asarray(
        Image.fromarray(coarse, mode="RGB").resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    noise = rng.normal(0, 14, (height, width, 3)).astype(np.float32)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestDataPipeline(unittest.TestCase):
    """Verify perceptual deduplication, family splits, dataset adapters, and shortcut reduction."""

    def test_01_perceptual_dedup_clusters_resized_and_recompressed_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / "real_cam"
            real_dir.mkdir()

            base = _make_textured_image(640, 640, seed=101)
            base.save(real_dir / "orig.png", format="PNG")
            # Resized & JPEG-recompressed variant of exact same image must share cluster_id
            base.resize((320, 320), Image.Resampling.LANCZOS).save(
                real_dir / "resized.jpg", format="JPEG", quality=60
            )
            # Exact byte copy must be dropped cleanly
            (real_dir / "zzz_exact_dup.png").write_bytes((real_dir / "orig.png").read_bytes())

            for i in range(6):
                _make_textured_image(512, 512, seed=200 + i).save(
                    real_dir / f"other_{i}.jpg", format="JPEG", quality=90
                )

            fake_dir = root / "sdxl"
            fake_dir.mkdir()
            for i in range(6):
                _make_textured_image(512, 512, seed=300 + i).save(
                    fake_dir / f"fake_{i}.png", format="PNG"
                )

            df_r = ingest_user_folder(real_dir, class_name="REAL", source_name="phone_cam", run_face_gate=False)
            df_f = ingest_user_folder(fake_dir, class_name="FAKE", source_name="sdxl", run_face_gate=False)
            deduped, stats = cluster_and_deduplicate(pd.concat([df_r, df_f], ignore_index=True), max_hamming=4)

            self.assertEqual(stats["exact_sha_dropped"], 1)
            orig_cid = int(deduped.loc[deduped["path"].str.endswith("orig.png"), "cluster_id"].iloc[0])
            res_cid = int(deduped.loc[deduped["path"].str.endswith("resized.jpg"), "cluster_id"].iloc[0])
            self.assertEqual(orig_cid, res_cid)

            splits = assign_generalization_splits(
                deduped, val_generators=(), test_generators=(), seed=42
            )
            verify_prepared_splits(splits)
            # Near-duplicate variants must live in the exact same split
            matched_splits = splits.loc[splits["cluster_id"] == orig_cid, "split"].unique().tolist()
            self.assertEqual(len(matched_splits), 1)

    def test_02_generator_family_grouping_keeps_flux_variants_together(self) -> None:
        self.assertEqual(normalize_generator_family("FLUX", "FAKE"), "flux")
        self.assertEqual(normalize_generator_family("FLUX_OpenFake", "FAKE"), "flux")
        self.assertEqual(normalize_generator_family("Midjourney", "FAKE"), "midjourney")
        self.assertEqual(normalize_generator_family("COCO", "REAL"), "real")

        rows = []
        idx = 0
        for cls, gen in [
            ("REAL", "phone"),
            ("REAL", "phone"),
            ("REAL", "phone"),
            ("REAL", "phone"),
            ("REAL", "phone"),
            ("FAKE", "SDXL"),
            ("FAKE", "SDXL"),
            ("FAKE", "SDXL"),
            ("FAKE", "FLUX"),
            ("FAKE", "FLUX_OpenFake"),
            ("FAKE", "Midjourney"),
            ("FAKE", "GPT_Image_1"),
        ]:
            idx += 1
            rows.append(
                {
                    "path": f"dummy_{idx}.jpg",
                    "label": 0 if cls == "REAL" else 1,
                    "class_name": cls,
                    "generator": gen,
                    "generator_family": normalize_generator_family(gen, cls),
                    "sha256": f"sha_{idx}",
                    "pixel_md5": f"pmd5_{idx}",
                    "dhash_int": idx * 999983,
                    "cluster_id": idx,
                    "width": 512,
                    "height": 512,
                    "has_face": True,
                }
            )
        df = pd.DataFrame(rows)
        out = assign_generalization_splits(
            df, val_generators=("flux",), test_generators=("midjourney", "gpt_image_1"), seed=42
        )
        self.assertTrue((out.loc[out["generator_family"] == "flux", "split"] == "val").all())
        self.assertTrue(
            (out.loc[out["generator_family"].isin(["midjourney", "gpt_image_1"]), "split"] == "held_out_test").all()
        )
        verify_prepared_splits(out, val_families={"flux"}, test_families={"midjourney", "gpt_image_1"})

    def test_03_public_dataset_folder_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # GenImage fixture
            g_root = root / "genimage"
            (g_root / "wukong" / "val" / "ai").mkdir(parents=True)
            (g_root / "wukong" / "val" / "nature").mkdir(parents=True)
            _make_textured_image(128, 128, 1).save(g_root / "wukong" / "val" / "ai" / "a.png")
            _make_textured_image(128, 128, 2).save(g_root / "wukong" / "val" / "nature" / "n.png")
            df_g = ingest_genimage_dir(g_root, run_face_gate=False)
            self.assertEqual(set(df_g["generator"]), {"wukong", "imagenet_nature"})

            # Synthbuster + RAISE-1k fixture
            s_root = root / "synthbuster"
            r_root = root / "raise1k"
            (s_root / "firefly").mkdir(parents=True)
            r_root.mkdir()
            _make_textured_image(128, 128, 3).save(s_root / "firefly" / "f.png")
            _make_textured_image(128, 128, 4).save(r_root / "r.png")
            df_s = ingest_synthbuster_dir(s_root, raise1k_dir=r_root, run_face_gate=False)
            self.assertEqual(set(df_s["generator"]), {"firefly", "raise1k"})

            # WildRF fixture
            w_root = root / "wildrf"
            (w_root / "reddit" / "0_real").mkdir(parents=True)
            (w_root / "reddit" / "1_fake").mkdir(parents=True)
            _make_textured_image(128, 128, 5).save(w_root / "reddit" / "0_real" / "r.jpg")
            _make_textured_image(128, 128, 6).save(w_root / "reddit" / "1_fake" / "f.jpg")
            df_w = ingest_wildrf_dir(w_root, run_face_gate=False)
            self.assertEqual(sorted(df_w["class_name"].unique()), ["FAKE", "REAL"])

    def test_04_shortcut_leakage_probe_and_harmonization_mitigation(self) -> None:
        # Deliberately create extreme metadata shortcut: REAL = 512x512 PNG, FAKE = 768x600 JPEG q70
        raw_rows = []
        harm_rows = []
        rng = random.Random(42)
        for i in range(40):
            img = _make_textured_image(512, 512, seed=10 + i)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw_b = buf.getvalue()
            raw_rows.append(
                {
                    "label": 0,
                    "width": 512,
                    "height": 512,
                    "aspect_ratio": 1.0,
                    "max_side": 512,
                    "min_side": 512,
                    "file_bytes": len(raw_b),
                    "bpp": len(raw_b) * 8.0 / (512 * 512),
                    "format_code": 1,
                    "has_exif": 1,
                }
            )
            h_img, h_b, h_fmt = harmonize_image_pair_safe(img, rng, target_max_sides=(384, 512))
            harm_rows.append(
                {
                    "label": 0,
                    "width": h_img.width,
                    "height": h_img.height,
                    "aspect_ratio": h_img.width / h_img.height,
                    "max_side": max(h_img.size),
                    "min_side": min(h_img.size),
                    "file_bytes": len(h_b),
                    "bpp": len(h_b) * 8.0 / (h_img.width * h_img.height),
                    "format_code": 0 if h_fmt == "JPEG" else (2 if h_fmt == "WEBP" else 1),
                    "has_exif": 0,
                }
            )

        for i in range(40):
            img = _make_textured_image(768, 600, seed=500 + i)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            raw_b = buf.getvalue()
            raw_rows.append(
                {
                    "label": 1,
                    "width": 768,
                    "height": 600,
                    "aspect_ratio": 768 / 600,
                    "max_side": 768,
                    "min_side": 600,
                    "file_bytes": len(raw_b),
                    "bpp": len(raw_b) * 8.0 / (768 * 600),
                    "format_code": 0,
                    "has_exif": 0,
                }
            )
            h_img, h_b, h_fmt = harmonize_image_pair_safe(img, rng, target_max_sides=(384, 512))
            harm_rows.append(
                {
                    "label": 1,
                    "width": h_img.width,
                    "height": h_img.height,
                    "aspect_ratio": h_img.width / h_img.height,
                    "max_side": max(h_img.size),
                    "min_side": min(h_img.size),
                    "file_bytes": len(h_b),
                    "bpp": len(h_b) * 8.0 / (h_img.width * h_img.height),
                    "format_code": 0 if h_fmt == "JPEG" else (2 if h_fmt == "WEBP" else 1),
                    "has_exif": 0,
                }
            )

        raw_res = _run_metadata_probe_df(pd.DataFrame(raw_rows), seed=42)
        harm_res = _run_metadata_probe_df(pd.DataFrame(harm_rows), seed=42)
        self.assertGreater(raw_res["auroc"], 0.95)
        self.assertLess(harm_res["auroc"], 0.65)

        with tempfile.TemporaryDirectory() as tmp:
            troot = Path(tmp)
            records = []
            for i in range(15):
                rp = troot / f"r_{i}.png"
                fp = troot / f"f_{i}.jpg"
                _make_textured_image(512, 512, 700 + i).save(rp, format="PNG")
                _make_textured_image(768, 600, 800 + i).save(fp, format="JPEG", quality=70)
                records.append(
                    {
                        "path": str(rp),
                        "label": 0,
                        "width": 512,
                        "height": 512,
                        "aspect_ratio": 1.0,
                        "max_side": 512,
                        "min_side": 512,
                        "file_bytes": rp.stat().st_size,
                        "bpp": rp.stat().st_size * 8.0 / (512 * 512),
                        "format_code": 1,
                        "has_exif": False,
                        "has_face": True,
                    }
                )
                records.append(
                    {
                        "path": str(fp),
                        "label": 1,
                        "width": 768,
                        "height": 600,
                        "aspect_ratio": 768 / 600,
                        "max_side": 768,
                        "min_side": 600,
                        "file_bytes": fp.stat().st_size,
                        "bpp": fp.stat().st_size * 8.0 / (768 * 600),
                        "format_code": 0,
                        "has_exif": False,
                        "has_face": True,
                    }
                )
            res = verify_harmonized_leakage(pd.DataFrame(records), sample_per_class=15, seed=42)
            self.assertGreater(res["hires_face_512plus"]["raw"]["auroc"], 0.95)
            self.assertLess(res["hires_face_512plus"]["harmonized"]["auroc"], 0.70)

    def test_05_dhash_collision_rejected_by_rgb_thumb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / "cam"
            fake_dir = root / "sd3"
            real_dir.mkdir()
            fake_dir.mkdir()

            # Construct two distinct images whose grayscale ramps monotonically left-to-right
            # (exact same dHash=0xffffffffffffffff) but have opposing RGB color distributions.
            ramp = np.linspace(10, 220, 256, dtype=np.uint8)[None, :].repeat(256, axis=0)
            arr_blue = np.stack([np.zeros_like(ramp), ramp, np.full_like(ramp, 230)], axis=-1)
            arr_orange = np.stack([np.full_like(ramp, 240), ramp, np.zeros_like(ramp)], axis=-1)

            img_blue = Image.fromarray(arr_blue, mode="RGB")
            img_orange = Image.fromarray(arr_orange, mode="RGB")

            img_blue.save(real_dir / "blue.png", format="PNG")
            img_blue.resize((128, 128), Image.Resampling.LANCZOS).save(
                real_dir / "blue_resized.jpg", format="JPEG", quality=75
            )
            img_orange.save(fake_dir / "orange.png", format="PNG")

            df_r = ingest_user_folder(real_dir, class_name="REAL", source_name="cam", run_face_gate=False)
            df_f = ingest_user_folder(fake_dir, class_name="FAKE", source_name="sd3", run_face_gate=False)
            deduped, stats = cluster_and_deduplicate(pd.concat([df_r, df_f], ignore_index=True), max_hamming=4)

            self.assertEqual(stats["cross_class_conflicts_dropped"], 0)
            self.assertEqual(len(deduped), 3)
            cid_blue = int(deduped.loc[deduped["path"].str.endswith("blue.png"), "cluster_id"].iloc[0])
            cid_blue_res = int(deduped.loc[deduped["path"].str.endswith("blue_resized.jpg"), "cluster_id"].iloc[0])
            cid_orange = int(deduped.loc[deduped["path"].str.endswith("orange.png"), "cluster_id"].iloc[0])
            self.assertEqual(cid_blue, cid_blue_res)
            self.assertNotEqual(cid_blue, cid_orange)
