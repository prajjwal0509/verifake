"""Unit tests for Step 3 augmentations, CLIP probe head, and LOGO fold isolation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.train_step3 import build_logo_fold_dataframes, sample_balanced_sources
from src.augmentations import (
    apply_eval_perturbation,
    build_clip_transforms,
    build_convnext_transforms,
    resize_pad_square,
)
from src.clip_detector import CLIPFeatureExtractor, load_clip_head, save_clip_head, train_clip_head


def _solid_or_grad(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 225, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestStep3Models(unittest.TestCase):
    """Fast CPU unit tests (zero network) for Step 3 preprocessing, CLIP head, and LOGO isolation."""

    def test_01_robust_forensic_transform_all_shapes(self) -> None:
        clip_tr = build_clip_transforms(train=True, img_size=224)
        clip_ev = build_clip_transforms(train=False, img_size=224)
        conv_tr = build_convnext_transforms(train=True, img_size=224)
        conv_ev = build_convnext_transforms(train=False, img_size=224)

        for w, h in ((512, 512), (1024, 765), (765, 1024), (178, 218)):
            img = _solid_or_grad(w, h, seed=w + h)
            for tf in (clip_tr, clip_ev, conv_tr, conv_ev):
                out = tf(img)
                self.assertEqual(tuple(out.shape), (3, 224, 224))
                self.assertTrue(torch.isfinite(out).all())

        for pname in ("original", "jpeg_q50", "webp_q80", "downscale_512", "downscale_256", "downscale_512_jpeg_q75"):
            pimg = apply_eval_perturbation(_solid_or_grad(640, 480, seed=7), pname)
            self.assertEqual(pimg.mode, "RGB")

    def test_02_resize_pad_square_preserves_aspect(self) -> None:
        # Bright red wide rectangle on gray background -> padded top/bottom strips must stay fill=(128,128,128)
        wide = Image.new("RGB", (1024, 512), (240, 20, 20))
        sq = resize_pad_square(wide, size=224, fill=(128, 128, 128))
        self.assertEqual(sq.size, (224, 224))
        arr = np.asarray(sq)
        # Top border row (y=5) is gray padding, center row (y=112) is red content
        self.assertEqual(tuple(int(v) for v in arr[5, 112]), (128, 128, 128))
        self.assertGreater(int(arr[112, 112, 0]), 220)

    def test_03_clip_probe_head_train_save_load_parity(self) -> None:
        rng = np.random.default_rng(42)
        d = 64
        proto = np.array([-1.0] * (d // 2) + [1.0] * (d // 2), dtype=np.float32)

        def _make_embs(n_per: int) -> tuple[torch.Tensor, torch.Tensor]:
            c0 = np.tile(-proto, (n_per, 1)) + rng.normal(0.0, 0.35, (n_per, d))
            c1 = np.tile(proto, (n_per, 1)) + rng.normal(0.0, 0.35, (n_per, d))
            x = torch.tensor(np.vstack([c0, c1]), dtype=torch.float32)
            x = torch.nn.functional.normalize(x, dim=-1)
            y = torch.tensor([0] * n_per + [1] * n_per, dtype=torch.long)
            return x, y

        x_tr, y_tr = _make_embs(80)
        x_va, y_va = _make_embs(30)

        for head_type in ("linear", "mlp"):
            head, meta = train_clip_head(
                x_tr,
                y_tr,
                x_va,
                y_va,
                head_type=head_type,
                hidden_dim=32,
                epochs=15,
                min_epochs=5,
                patience=5,
                seed=42,
            )
            self.assertGreater(meta["clean_auroc"], 0.95)
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_path = Path(tmp) / f"{head_type}.pt"
                save_clip_head(head, ckpt_path, {"class_to_idx": {"REAL": 0, "FAKE": 1}, "best_val": meta})
                loaded, loaded_meta = load_clip_head(ckpt_path, torch.device("cpu"))
                self.assertEqual(loaded_meta["head_type"], head_type)
                with torch.no_grad():
                    diff = (head(x_va) - loaded(x_va)).abs().max().item()
                self.assertLess(diff, 1e-6)

        # Also verify CLIPFeatureExtractor compiles cleanly without downloading weights
        ext = CLIPFeatureExtractor(pretrained=False, img_size=224)
        feats = ext.extract_features(torch.randn(2, 3, 224, 224), use_fp16=False)
        self.assertEqual(feats.shape, (2, ext.embed_dim))
        self.assertTrue(torch.allclose(feats.norm(dim=-1), torch.ones(2), atol=1e-4))

    def test_04_logo_split_excludes_held_out_test_and_fold_family(self) -> None:
        rows = []
        idx = 0
        for sp, role, cls, fam, gen, n in [
            ("train", "train", "REAL", "real", "COCO", 20),
            ("train", "train", "REAL", "real", "FFHQ", 40),
            ("train", "train", "FAKE", "dalle3", "DALLE3", 20),
            ("train", "train", "FAKE", "sdxl", "SDXL", 20),
            ("val", "val_real", "REAL", "real", "COCO", 12),
            ("val", "val_real", "REAL", "real", "FFHQ", 12),
            ("val", "unseen_val_generator", "FAKE", "flux", "FLUX", 15),
            ("val", "seen_val_generator", "FAKE", "sdxl", "SDXL", 10),
            ("held_out_test", "unseen_test_generator", "FAKE", "midjourney", "Midjourney", 15),
            ("held_out_test", "test_real", "REAL", "real", "FFHQ", 15),
        ]:
            for _ in range(n):
                idx += 1
                rows.append(
                    {
                        "path": f"dummy_{idx}.jpg",
                        "sha256": f"sha_{idx}",
                        "label": 0 if cls == "REAL" else 1,
                        "class_name": cls,
                        "generator": gen,
                        "generator_family": fam,
                        "split": sp,
                        "split_role": role,
                    }
                )
        df = pd.DataFrame(rows)
        non_test = df[df["split"] != "held_out_test"].reset_index(drop=True)
        for held in ("flux", "sdxl"):
            f_tr, f_val, f_held, f_vr = build_logo_fold_dataframes(non_test, held, train_cap=40, seed=42)
            self.assertFalse((f_tr["generator_family"] == held).any())
            self.assertFalse((f_tr["split"] == "held_out_test").any())
            self.assertFalse((f_val["generator_family"] == held).any())
            self.assertTrue((f_held["generator_family"] == held).all())
            self.assertGreater(len(f_vr), 0)

        bal = sample_balanced_sources(non_test[non_test["split"] == "train"], 40, seed=42)
        self.assertEqual(int((bal["label"] == 0).sum()), 20)
        self.assertEqual(int((bal["label"] == 1).sum()), 20)
