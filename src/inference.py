"""Inference API: C2PA provenance check, YuNet face gate, 3-view TTA, calibration, and Grad-CAM."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

import torch
from PIL import Image, ImageOps

from src.calibration import (
    CALIBRATION_PATH,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    apply_temperature,
    load_calibration,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.face_crop import crop_face_or_full, gate_and_crop_face
from src.gradcam import GradCAM, overlay_heatmap
from src.image_io import encode_jpeg_preview_b64, register_image_plugins, to_rgb_safe
from src.model import load_checkpoint
from src.provenance import inspect_provenance
from src.utils import (
    CHECKPOINT_PATH,
    CLASS_NAMES_PATH,
    get_device,
    get_logger,
    load_class_mapping,
)

logger = get_logger(__name__)
register_image_plugins()

ImageInput = Union[Image.Image, bytes, str, Path]


def _to_pil(image: ImageInput) -> Image.Image:
    """Accept a PIL image, raw bytes or a file path and return an upright RGB PIL image."""
    register_image_plugins()
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
        if getattr(img, "is_animated", False):
            img.seek(0)
        img.load()
    else:
        img = Image.open(Path(image))
        if getattr(img, "is_animated", False):
            img.seek(0)
        img.load()
    img = ImageOps.exif_transpose(img)
    return to_rgb_safe(img)


def _png_b64(img: Image.Image, max_side: int = 640) -> str:
    """Encode a (downscaled) image as a base64 PNG data URL."""
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def logits_to_probabilities(
    logits: torch.Tensor,
    idx_to_class: dict[int, str],
) -> tuple[str, float, dict[str, float], int]:
    """Convert model logits into (label, confidence, probabilities, pred_idx)."""
    logits_1d = logits.float().view(-1)
    if logits_1d.numel() == 1:
        p1 = float(torch.sigmoid(logits_1d[0]).item())
        prob_by_idx = {0: 1.0 - p1, 1: p1}
    else:
        probs_tensor = torch.softmax(logits_1d, dim=0).cpu()
        prob_by_idx = {
            i: float(probs_tensor[i].item()) for i in range(probs_tensor.numel())
        }

    pred_idx = max(prob_by_idx, key=prob_by_idx.__getitem__)
    pred_label = idx_to_class[pred_idx]
    confidence = float(prob_by_idx[pred_idx])
    probabilities = {
        idx_to_class[i]: float(prob_by_idx[i]) for i in sorted(idx_to_class)
    }
    return pred_label, confidence, probabilities, pred_idx


class DeepfakeDetector:
    """Loads checkpoint, ``class_names.json``, and ``calibration.json`` to serve predictions."""

    def __init__(
        self,
        checkpoint: Path = CHECKPOINT_PATH,
        mapping_path: Path = CLASS_NAMES_PATH,
        calibration_path: Path = CALIBRATION_PATH,
        device: Optional[torch.device] = None,
    ) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint}; run `python -m src.train`."
            )
        mapping = load_class_mapping(mapping_path)
        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = mapping["idx_to_class"]
        self.class_names: list[str] = mapping["class_names"]

        self.device = device or get_device()
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        cal = load_calibration(calibration_path)
        self.temperature: float = float(
            cal.get("temperature", self.meta.get("temperature", 1.0))
        )
        self.img_size = int(self.meta["img_size"])
        mean = tuple(self.meta.get("normalize_mean", IMAGENET_MEAN))
        std = tuple(self.meta.get("normalize_std", IMAGENET_STD))
        self.transform = build_transforms(
            self.img_size, train=False, mean=mean, std=std
        )
        self.cam = GradCAM(self.model, mapping_path=mapping_path)
        self._lock = Lock()
        logger.info(
            "Loaded %s (epoch %s) on %s | class_to_idx=%s | T=%.4f",
            self.meta["backbone"],
            self.meta.get("epoch"),
            self.device,
            self.class_to_idx,
            self.temperature,
        )

    def _calibrate_view(self, view_logit: torch.Tensor) -> dict[str, Any]:
        scaled = apply_temperature(view_logit, self.temperature)
        binary_label, _, probs, _ = logits_to_probabilities(scaled, self.idx_to_class)
        p_fake = float(probs["FAKE"])
        uncertain = UNCERTAIN_LOW <= p_fake <= UNCERTAIN_HIGH
        return {
            "REAL": float(probs["REAL"]),
            "FAKE": p_fake,
            "label": "UNCERTAIN" if uncertain else binary_label,
        }

    def _tta_logits_with_views(
        self,
        pil: Image.Image,
        cropped: Image.Image,
    ) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
        """Run 3-view TTA (crop, flipped, full_frame) returning mean logits and calibrated per-view probs."""
        batch = torch.stack(
            [
                self.transform(cropped),
                self.transform(ImageOps.mirror(cropped)),
                self.transform(pil),
            ],
            dim=0,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
        per_view = {
            "crop": self._calibrate_view(logits[0:1]),
            "flipped": self._calibrate_view(logits[1:2]),
            "full_frame": self._calibrate_view(logits[2:3]),
        }
        return logits.mean(dim=0, keepdim=True), per_view

    def _build_pixel_signal(self, logits: torch.Tensor) -> dict[str, Any]:
        """Compute calibrated Pixel Model signal from aggregated logits."""
        _, _, uncal_probs, _ = logits_to_probabilities(logits, self.idx_to_class)
        scaled = apply_temperature(logits, self.temperature)
        binary_label, confidence, probs, pred_idx = logits_to_probabilities(
            scaled, self.idx_to_class
        )
        p_fake = float(probs["FAKE"])
        uncertain = UNCERTAIN_LOW <= p_fake <= UNCERTAIN_HIGH
        display_label = "UNCERTAIN" if uncertain else binary_label
        return {
            "status": "ok",
            "label": display_label,
            "binary_label": binary_label,
            "uncertain": uncertain,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probs,
            "uncalibrated_probabilities": uncal_probs,
            "temperature": self.temperature,
        }

    def _build_diagnostics(
        self,
        face_meta: dict[str, Any],
        per_view_probs: Optional[dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "face_detected": bool(face_meta.get("face_detected", False)),
            "face_score": float(face_meta.get("face_score", 0.0)),
            "image_resolution": face_meta.get("image_resolution"),
            "face_bbox": face_meta.get("raw_face_bbox") or face_meta.get("bbox"),
            "face_width_pct": float(face_meta.get("face_width_pct", 0.0)),
            "face_height_pct": float(face_meta.get("face_height_pct", 0.0)),
            "raw_crop_resolution": face_meta.get("raw_crop_resolution"),
            "crop_resolution": face_meta.get("crop_resolution"),
            "crop_margin": face_meta.get("crop_margin"),
            "upsampled": bool(face_meta.get("upsampled", False)),
            "small_face": bool(face_meta.get("small_face", False)),
            "warning": face_meta.get("warning"),
            "per_view_probabilities": per_view_probs,
        }

    def _assemble_response(
        self,
        *,
        pil: Image.Image,
        face_meta: dict[str, Any],
        provenance: dict[str, Any],
        pixel_signal: Optional[dict[str, Any]],
        per_view_probs: Optional[dict[str, dict[str, Any]]],
        use_tta: bool,
    ) -> dict[str, Any]:
        diagnostics = self._build_diagnostics(face_meta, per_view_probs)
        override = bool(provenance.get("ai_verified_by_credentials"))

        if pixel_signal is None and not override:
            return {
                "status": "no_face",
                "message": "No face detected. This tool analyzes faces only.",
                "verdict_display": "No face detected",
                "verdict_source": "face_gate",
                "override_applied": False,
                "pixel_model": {
                    "status": "no_face",
                    "label": None,
                    "binary_label": None,
                    "uncertain": False,
                    "confidence": None,
                    "probabilities": None,
                    "temperature": self.temperature,
                },
                "provenance": provenance,
                "content_credentials": provenance,
                "diagnostics": diagnostics,
                "preprocessing": face_meta,
                "image_size": list(pil.size),
                "model": self.meta["backbone"],
                "temperature": self.temperature,
                "class_to_idx": self.class_to_idx,
            }

        if override:
            label = "FAKE"
            binary_label = "FAKE"
            uncertain = False
            verdict_source = "metadata"
            verdict_display = "FAKE (source: metadata)"
            confidence = 1.0
            probs = {"REAL": 0.0, "FAKE": 1.0}
            class_index = self.class_to_idx["FAKE"]
            if pixel_signal is None:
                pixel_signal = {
                    "status": "no_face",
                    "label": None,
                    "binary_label": None,
                    "uncertain": False,
                    "confidence": None,
                    "probabilities": None,
                    "temperature": self.temperature,
                }
        else:
            assert pixel_signal is not None
            label = pixel_signal["label"]
            binary_label = pixel_signal["binary_label"]
            uncertain = pixel_signal["uncertain"]
            confidence = pixel_signal["confidence"]
            probs = pixel_signal["probabilities"]
            class_index = pixel_signal["class_index"]
            verdict_source = "pixel_model"
            if label == "REAL":
                if provenance.get("present") and provenance.get("signature_valid"):
                    verdict_display = "REAL (signed credentials, no AI assertion)"
                else:
                    verdict_display = "Likely real, not verified"
            elif label == "FAKE":
                verdict_display = "FAKE (source: pixel model)"
            else:
                verdict_display = "UNCERTAIN"

        return {
            "status": "ok",
            "label": label,
            "binary_label": binary_label,
            "uncertain": uncertain,
            "verdict_source": verdict_source,
            "verdict_display": verdict_display,
            "override_applied": override,
            "class_index": class_index,
            "confidence": confidence,
            "probabilities": probs,
            "uncalibrated_probabilities": pixel_signal.get(
                "uncalibrated_probabilities"
            ),
            "temperature": self.temperature,
            "tta_views": ["face_crop", "flipped_face_crop", "full_frame"]
            if use_tta
            else ["face_crop"],
            "pixel_model": pixel_signal,
            "provenance": provenance,
            "content_credentials": provenance,
            "diagnostics": diagnostics,
            "preprocessing": face_meta,
            "class_to_idx": self.class_to_idx,
            "image_size": list(pil.size),
            "model": self.meta["backbone"],
        }

    def _run_pixel_model(
        self,
        pil: Image.Image,
        cropped: Image.Image,
        use_tta: bool,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if use_tta:
            logits, per_view = self._tta_logits_with_views(pil, cropped)
        else:
            logits = self.model(self.transform(cropped).unsqueeze(0).to(self.device))
            per_view = {"crop": self._calibrate_view(logits)}
        return self._build_pixel_signal(logits), per_view

    @torch.no_grad()
    def predict(
        self,
        image: ImageInput,
        require_face: bool = True,
        raw_bytes: Optional[bytes] = None,
        use_tta: bool = True,
        detected_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run C2PA provenance check, YuNet face gate, 3-view TTA & calibration."""
        cred_source = raw_bytes if raw_bytes is not None else image
        provenance = inspect_provenance(cred_source, detected_format=detected_format)
        pil = _to_pil(image)

        if require_face:
            cropped, face_meta = gate_and_crop_face(pil)
        else:
            cropped, face_meta = crop_face_or_full(pil)

        pixel_signal: Optional[dict[str, Any]] = None
        per_view: Optional[dict[str, dict[str, Any]]] = None
        if cropped is not None:
            with self._lock:
                pixel_signal, per_view = self._run_pixel_model(
                    pil, cropped, use_tta=use_tta
                )

        return self._assemble_response(
            pil=pil,
            face_meta=face_meta,
            provenance=provenance,
            pixel_signal=pixel_signal,
            per_view_probs=per_view,
            use_tta=use_tta,
        )

    def predict_with_gradcam(
        self,
        image: ImageInput,
        require_face: bool = True,
        raw_bytes: Optional[bytes] = None,
        use_tta: bool = True,
        detected_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run C2PA provenance check, YuNet face gate, 3-view TTA, diagnostics & Grad-CAM."""
        cred_source = raw_bytes if raw_bytes is not None else image
        provenance = inspect_provenance(cred_source, detected_format=detected_format)
        pil = _to_pil(image)

        if require_face:
            cropped, face_meta = gate_and_crop_face(pil)
        else:
            cropped, face_meta = crop_face_or_full(pil)

        pixel_signal: Optional[dict[str, Any]] = None
        per_view: Optional[dict[str, dict[str, Any]]] = None
        cam_b64: Optional[str] = None
        cam_target: Optional[str] = None

        if cropped is not None:
            x_full = self.transform(pil).unsqueeze(0).to(self.device)
            with self._lock:
                pixel_signal, per_view = self._run_pixel_model(
                    pil, cropped, use_tta=use_tta
                )
                cam_target = pixel_signal["binary_label"]
                cam, _, _ = self.cam(x_full, target_class=cam_target)
            cam_b64 = _png_b64(overlay_heatmap(pil, cam))

        result = self._assemble_response(
            pil=pil,
            face_meta=face_meta,
            provenance=provenance,
            pixel_signal=pixel_signal,
            per_view_probs=per_view,
            use_tta=use_tta,
        )
        result["preview"] = encode_jpeg_preview_b64(pil)
        if cam_b64 is not None:
            result["gradcam"] = cam_b64
            result["gradcam_target"] = cam_target
        return result


_detector: Optional[DeepfakeDetector] = None


def get_detector(force_reload: bool = False) -> DeepfakeDetector:
    """Lazily construct (or reload) a process-wide detector."""
    global _detector
    if _detector is None or force_reload:
        _detector = DeepfakeDetector()
    return _detector


def predict(
    image: ImageInput,
    require_face: bool = True,
    raw_bytes: Optional[bytes] = None,
    use_tta: bool = True,
    detected_format: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience wrapper: ``predict(image)`` -> calibrated verdict + provenance + diagnostics."""
    return get_detector().predict(
        image,
        require_face=require_face,
        raw_bytes=raw_bytes,
        use_tta=use_tta,
        detected_format=detected_format,
    )
