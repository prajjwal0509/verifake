"""Unit tests for Step 4 bounded calibration, view disagreement, and EnsembleDeepfakeDetector."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import load_manifest  # noqa: E402
from src.ensemble_inference import (  # noqa: E402
    EnsembleDeepfakeDetector,
    STEP3_ENSEMBLE_PATH,
    apply_calibrated_decision,
    fit_temperature_bounded_platt,
)
from src.utils import MODELS_DIR  # noqa: E402


class TestStep4Inference(unittest.TestCase):
    """CPU unit tests for Step 4 bounded calibration, abstention rules, and ensemble API."""

    def test_01_bounded_calibration_enforces_t_ge_1_and_lowers_nll_ece(self) -> None:
        rng = np.random.default_rng(7)
        y = np.array([0] * 200 + [1] * 200, dtype=np.int64)

        # Underconfident logits (unconstrained optimization would pick T < 1.0 to sharpen)
        under_z = np.where(y == 1, 0.35, -0.35) + rng.normal(0.0, 0.15, size=len(y))
        res_under = fit_temperature_bounded_platt(under_z, y)
        self.assertGreaterEqual(res_under["temperature"], 1.0)

        # Overconfident and shifted logits (REAL > 0, FAKE >> 0) -> T > 1.0 and bias < 0 must lower NLL/ECE
        over_z = np.where(y == 1, 8.0, 2.0) + rng.normal(0.0, 1.2, size=len(y))
        res_over = fit_temperature_bounded_platt(over_z, y)
        self.assertGreater(res_over["temperature"], 1.0)
        self.assertLess(res_over["val_nll_after"], res_over["val_nll_before"])
        self.assertLess(res_over["val_ece_after"], res_over["val_ece_before"])

    def test_02_view_disagreement_and_borderline_uncertain_rules(self) -> None:
        # Views strongly disagree around threshold 0.50 (z_thr = 0.0) despite average = +0.5
        vd = apply_calibrated_decision(
            crop_z=2.0,
            full_z=-1.0,
            clip_z=0.5,
            conv_z=0.5,
            temperature=1.0,
            bias=0.0,
            decision_threshold=0.50,
            uncertain_band=(0.35, 0.65),
            view_disagreement_margin=0.85,
        )
        self.assertTrue(vd["view_disagreement"])
        self.assertTrue(vd["uncertain"])
        self.assertEqual(vd["label"], "UNCERTAIN")
        self.assertEqual(vd["uncertain_reason"], "view_disagreement")

        # Strong CLIP vs ConvNeXt branch disagreement -> branch_disagreement UNCERTAIN
        bd_m = apply_calibrated_decision(
            crop_z=0.5,
            full_z=0.5,
            clip_z=2.0,
            conv_z=-1.5,
            temperature=1.0,
            bias=0.0,
            decision_threshold=0.50,
            uncertain_band=(0.35, 0.65),
            view_disagreement_margin=0.85,
            branch_disagreement_margin=1.15,
        )
        self.assertTrue(bd_m["branch_disagreement"])
        self.assertTrue(bd_m["uncertain"])
        self.assertEqual(bd_m["label"], "UNCERTAIN")
        self.assertEqual(bd_m["uncertain_reason"], "branch_disagreement")

        # Views agree mildly right at decision threshold -> borderline confidence
        bd = apply_calibrated_decision(
            crop_z=0.1,
            full_z=0.0,
            clip_z=0.05,
            conv_z=0.05,
            temperature=1.0,
            bias=0.0,
            decision_threshold=0.50,
            uncertain_band=(0.35, 0.65),
            view_disagreement_margin=0.85,
        )
        self.assertFalse(bd["view_disagreement"])
        self.assertTrue(bd["uncertain"])
        self.assertEqual(bd["label"], "UNCERTAIN")
        self.assertEqual(bd["uncertain_reason"], "borderline_confidence")

        # Views agree strongly on FAKE and REAL outside band
        fk = apply_calibrated_decision(
            crop_z=3.0,
            full_z=2.5,
            clip_z=2.8,
            conv_z=2.7,
            temperature=1.0,
            bias=0.0,
            decision_threshold=0.50,
            uncertain_band=(0.35, 0.65),
        )
        self.assertFalse(fk["uncertain"])
        self.assertEqual(fk["label"], "FAKE")
        self.assertIsNone(fk["uncertain_reason"])

        rl = apply_calibrated_decision(
            crop_z=-3.0,
            full_z=-2.5,
            clip_z=-2.8,
            conv_z=-2.7,
            temperature=1.0,
            bias=0.0,
            decision_threshold=0.50,
            uncertain_band=(0.35, 0.65),
        )
        self.assertFalse(rl["uncertain"])
        self.assertEqual(rl["label"], "REAL")
        self.assertIsNone(rl["uncertain_reason"])

    def test_03_ensemble_detector_end_to_end_cpu_smoke(self) -> None:
        if not (
            STEP3_ENSEMBLE_PATH.exists()
            and (MODELS_DIR / "clip_mlp_v2.pt").exists()
            and (MODELS_DIR / "convnext_v2.pt").exists()
        ):
            self.skipTest("Step 3 checkpoints not present")

        det = EnsembleDeepfakeDetector(
            pretrained_clip=True,
            device=torch.device("cpu"),
        )
        self.assertGreaterEqual(det.temperature, 1.0)

        # Face image smoke test (with Grad-CAM)
        df = load_manifest()
        sample_rel = df[(df["split"] == "test") & (df["has_face"] == True)]["path"].iloc[0]  # noqa: E712
        res_face = det.predict_with_gradcam(PROJECT_ROOT / str(sample_rel), require_face=True)
        self.assertEqual(res_face["status"], "ok")
        self.assertIn(res_face["label"], {"REAL", "FAKE", "UNCERTAIN"})
        self.assertIn(res_face["binary_label"], {"REAL", "FAKE"})
        self.assertIsNotNone(res_face["ensemble_details"])
        self.assertTrue(str(res_face["gradcam"]).startswith("data:image/png;base64,"))
        self.assertIn("crop", res_face["diagnostics"]["per_view_probabilities"])
        self.assertIn("full_frame", res_face["diagnostics"]["per_view_probabilities"])

        # Non-face rejection
        blank = Image.new("RGB", (256, 256), (120, 120, 120))
        res_noface = det.predict_with_gradcam(blank, require_face=True)
        self.assertEqual(res_noface["status"], "no_face")
        self.assertNotIn("label", res_noface)
        self.assertNotIn("gradcam", res_noface)
