"""Regression tests for label mapping, face gate, calibration, and C2PA."""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.calibration import (
    CALIBRATION_PATH,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    apply_temperature,
    load_calibration,
)
from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FaceDataset,
    build_transforms,
    load_manifest,
)
from src.face_crop import gate_and_crop_face
from src.inference import (
    DeepfakeDetector,
    _to_pil,
    get_detector,
    logits_to_probabilities,
    predict,
)
from src.metadata import inspect_content_credentials
from src.utils import (
    CHECKPOINT_PATH,
    CLASS_NAMES_PATH,
    METRICS_PATH,
    PROJECT_ROOT,
    load_class_mapping,
    load_json,
)


def _make_png_with_cabx(
    cabx_payload: bytes, width: int = 128, height: int = 128
) -> bytes:
    """Create a valid minimal gradient PNG containing a C2PA ``caBX`` metadata chunk."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(20, 220, width, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(40, 180, height, dtype=np.uint8)[:, None]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    raw = buf.getvalue()
    # Insert caBX right before IEND (last 12 bytes).
    cabx_len = struct.pack(">I", len(cabx_payload))
    crc = struct.pack(">I", zlib.crc32(b"caBX" + cabx_payload) & 0xFFFFFFFF)
    return raw[:-12] + cabx_len + b"caBX" + cabx_payload + crc + raw[-12:]


class TestLabelMappingAndInference(unittest.TestCase):
    """Verify class mapping, Face Gate, calibration, C2PA metadata, parity & test accuracy."""

    def test_01_class_names_and_calibration_files_exist(self) -> None:
        """``models/class_names.json`` and ``models/calibration.json`` must match checkpoint."""
        self.assertTrue(CLASS_NAMES_PATH.exists(), f"Missing {CLASS_NAMES_PATH}")
        self.assertTrue(CALIBRATION_PATH.exists(), f"Missing {CALIBRATION_PATH}")

        mapping = load_class_mapping(CLASS_NAMES_PATH)
        self.assertEqual(set(mapping["class_to_idx"].values()), {0, 1})

        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["meta"]["class_to_idx"], mapping["class_to_idx"])
        cal = load_calibration(CALIBRATION_PATH)
        self.assertGreater(cal["temperature"], 0.05)

    def test_02_missing_mapping_file_raises(self) -> None:
        """Loading the mapping or detector must fail loudly if ``class_names.json`` is missing."""
        missing_path = (
            Path(tempfile.gettempdir()) / "nonexistent_class_names_99999.json"
        )
        if missing_path.exists():
            missing_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_class_mapping(missing_path)
        with self.assertRaises(FileNotFoundError):
            DeepfakeDetector(checkpoint=CHECKPOINT_PATH, mapping_path=missing_path)

    def test_03_logits_calibration_and_uncertainty_band(self) -> None:
        """Verify softmax/sigmoid mapping, temperature scaling, and UNCERTAIN [0.40, 0.60] band."""
        idx_to_class = {0: "REAL", 1: "FAKE"}

        label, conf, probs, idx = logits_to_probabilities(
            torch.tensor([[0.2, 2.5]]), idx_to_class
        )
        self.assertEqual(idx, 1)
        self.assertEqual(label, "FAKE")
        self.assertAlmostEqual(conf, probs["FAKE"], places=6)

        # Higher temperature softens extreme confidence toward 0.5
        softened = apply_temperature(torch.tensor([[0.2, 2.5]]), temperature=3.0)
        _, soft_conf, _, _ = logits_to_probabilities(softened, idx_to_class)
        self.assertLess(soft_conf, conf)

        # Equal logits -> P(FAKE) = 0.50 -> inside [0.40, 0.60] uncertainty band
        _, _, eq_probs, _ = logits_to_probabilities(
            torch.tensor([[0.0, 0.0]]), idx_to_class
        )
        self.assertTrue(UNCERTAIN_LOW <= eq_probs["FAKE"] <= UNCERTAIN_HIGH)

    def test_04_preprocessing_parity_between_eval_dataset_and_inference(self) -> None:
        """Inference preprocessing must match eval dataset tensors for both face-crop and full-frame."""
        df = load_manifest()
        test_split = df[(df["split"] == "test") & (df["has_face"] == True)]  # noqa: E712
        hi_res = test_split[test_split["width"] > 256].head(2)
        test_df = pd.concat([test_split.head(2), hi_res], ignore_index=True)
        det = get_detector()

        eval_ds = FaceDataset(
            test_df, build_transforms(det.img_size, train=False), use_face_crop=True
        )
        eval_ds_raw = FaceDataset(
            test_df, build_transforms(det.img_size, train=False), use_face_crop=False
        )
        for i, rel_path in enumerate(test_df["path"]):
            pil = _to_pil(PROJECT_ROOT / rel_path)
            cropped, meta = gate_and_crop_face(pil)
            self.assertIsNotNone(cropped, f"Expected face on {rel_path}, got {meta}")

            ds_tensor, _ = eval_ds[i]
            inf_tensor = det.transform(cropped)
            self.assertTrue(torch.allclose(ds_tensor, inf_tensor, atol=1e-6))

            ds_raw_tensor, _ = eval_ds_raw[i]
            inf_raw_tensor = det.transform(pil)
            self.assertTrue(torch.allclose(ds_raw_tensor, inf_raw_tensor, atol=1e-6))

        self.assertEqual(tuple(det.meta["normalize_mean"]), IMAGENET_MEAN)
        self.assertEqual(tuple(det.meta["normalize_std"]), IMAGENET_STD)

    def test_05_face_gate_rejects_non_face_image(self) -> None:
        """Non-face images must return status='no_face' and NEVER emit REAL/FAKE."""
        non_face_bytes = _make_png_with_cabx(
            b"c2pa\x00trainedAlgorithmicMedia\x00ChatGPT"
        )
        res = predict(non_face_bytes, require_face=True)
        self.assertEqual(res["status"], "no_face")
        self.assertNotIn("label", res)
        self.assertNotIn("probabilities", res)
        self.assertIn("No face detected", res["message"])

        res_cam = get_detector().predict_with_gradcam(non_face_bytes, require_face=True)
        self.assertEqual(res_cam["status"], "no_face")
        self.assertNotIn("label", res_cam)
        self.assertNotIn("gradcam", res_cam)

    def test_06_content_credentials_c2pa_detection(self) -> None:
        """C2PA metadata inspection must detect caBX + trainedAlgorithmicMedia as supporting evidence."""
        png_bytes = _make_png_with_cabx(
            b"jumb\x00c2pa\x00trainedAlgorithmicMedia\x00OpenAI\x00ChatGPT"
        )
        cred = inspect_content_credentials(png_bytes)
        self.assertTrue(cred["has_c2pa"])
        self.assertEqual(cred["digital_source_type"], "trainedAlgorithmicMedia")
        self.assertIn("OpenAI", cred["generator_claims"])
        self.assertIn("ChatGPT", cred["generator_claims"])
        self.assertEqual(cred["status"], "ai_credentials_found")
        self.assertIn("Supporting evidence only", cred["disclaimer"])

    def test_07_predict_accuracy_on_20_real_and_20_fake_face_gated_test_images(
        self,
    ) -> None:
        """Run ``predict()`` on 20 random REAL and 20 random FAKE face-gated test images; fail if < 80%."""
        mapping = load_class_mapping(CLASS_NAMES_PATH)
        class_to_idx = mapping["class_to_idx"]
        df = load_manifest()
        test_df = df[(df["split"] == "test") & (df["has_face"] == True)]  # noqa: E712
        metrics = load_json(METRICS_PATH)

        total_correct = 0
        total_seen = 0

        for class_name, class_idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
            sample_df = test_df[test_df["label"] == class_idx].sample(
                n=20, random_state=42
            )
            correct = 0
            for rel_path in sample_df["path"]:
                res = predict(PROJECT_ROOT / rel_path, require_face=True, use_tta=True)
                self.assertEqual(res["status"], "ok")
                self.assertIn("binary_label", res)
                self.assertAlmostEqual(
                    res["confidence"],
                    res["probabilities"][res["binary_label"]],
                    places=6,
                )
                if res["binary_label"] == class_name:
                    correct += 1
            acc = correct / len(sample_df)
            total_correct += correct
            total_seen += len(sample_df)
            self.assertGreaterEqual(
                acc, 0.80, f"{class_name} accuracy < 80%: {acc * 100:.1f}%"
            )

        overall_acc = total_correct / total_seen
        reported_test_acc = float(metrics["test"]["accuracy"])
        self.assertGreaterEqual(overall_acc, 0.80)
        self.assertAlmostEqual(overall_acc, reported_test_acc, delta=0.10)

        for key in (
            "calibration",
            "tta_comparison",
            "out_of_scope_no_face",
            "robustness_before_after",
            "per_generator",
            "unseen_generator",
            "logo_generalization",
        ):
            self.assertIn(key, metrics, f"Missing '{key}' in {METRICS_PATH}")
        self.assertEqual(len(metrics["logo_generalization"]["folds"]), 3)


if __name__ == "__main__":
    unittest.main()
