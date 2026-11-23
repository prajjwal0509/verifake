"""Step 4 calibrated ensemble inference: CLIP ViT-L/14 + ConvNeXt-Tiny v2, bounded calibration, and view disagreement."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from src.augmentations import build_clip_transforms, build_convnext_transforms
from src.calibration import compute_ece
from src.clip_detector import CLIP_BACKBONE, CLIPFeatureExtractor, load_clip_head
from src.face_crop import crop_face_or_full, gate_and_crop_face
from src.gradcam import GradCAM, overlay_heatmap
from src.image_io import encode_jpeg_preview_b64, register_image_plugins, to_rgb_safe
from src.model import load_checkpoint
from src.provenance import inspect_provenance
from src.utils import (
    CLASS_NAMES_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    get_device,
    get_logger,
    load_class_mapping,
    load_json,
)

logger = get_logger(__name__)
register_image_plugins()

ImageInput = Union[Image.Image, bytes, str, Path]

STEP3_ENSEMBLE_PATH: Path = MODELS_DIR / "step3_ensemble.json"
STEP4_CALIBRATION_PATH: Path = MODELS_DIR / "step4_calibration.json"
DEFAULT_VIEW_DISAGREEMENT_MARGIN: float = 0.85
DEFAULT_BRANCH_DISAGREEMENT_MARGIN: float = 1.15


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    clipped = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _to_pil(image: ImageInput) -> Image.Image:
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
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_temperature_bounded_platt(
    val_z: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, Any]:
    """Fit Platt shift + temperature scaling ``(z + b) / T`` with ``T >= 1.0`` enforced strictly."""
    z_np = np.asarray(val_z, dtype=np.float32).reshape(-1)
    y_np = np.asarray(val_y, dtype=np.float32).reshape(-1)
    if z_np.size == 0:
        raise ValueError("Validation logits cannot be empty")

    z_t = torch.from_numpy(z_np)
    y_t = torch.from_numpy(y_np)
    y_bin = y_np.astype(np.int64)
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        nll_before = float(bce(z_t, y_t).item())
        p_before = torch.sigmoid(z_t).numpy().astype(np.float64)
        ece_before = compute_ece(p_before, y_bin)

    raw_t = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
    bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [raw_t, bias], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.clamp(1.0 + nn.functional.softplus(raw_t), min=1.0, max=10.0)
        scaled = (z_t + bias) / temp
        loss = bce(scaled, y_t)
        loss.backward()
        return loss

    optimizer.step(_closure)

    with torch.no_grad():
        temperature = float(
            torch.clamp(1.0 + nn.functional.softplus(raw_t), min=1.0, max=10.0).item()
        )
        bias_val = float(bias.item())
        scaled_after = (z_t + bias_val) / temperature
        nll_after = float(bce(scaled_after, y_t).item())
        p_after = torch.sigmoid(scaled_after).numpy().astype(np.float64)
        ece_after = compute_ece(p_after, y_bin)

    if (
        not math.isfinite(temperature)
        or not math.isfinite(bias_val)
        or not math.isfinite(nll_after)
        or nll_after > nll_before + 1e-6
    ):
        logger.warning(
            "Bounded calibration did not improve NLL (%.4f -> %.4f); reverting to T=1.0, b=0.0",
            nll_before,
            nll_after,
        )
        temperature = 1.0
        bias_val = 0.0
        nll_after = nll_before
        ece_after = ece_before

    temperature = max(1.0, temperature)
    logger.info(
        "Step 4 calibration (T>=1.0): T=%.4f b=%+.4f | NLL %.4f -> %.4f | ECE %.4f -> %.4f",
        temperature,
        bias_val,
        nll_before,
        nll_after,
        ece_before,
        ece_after,
    )
    return {
        "temperature": float(temperature),
        "bias": float(bias_val),
        "val_n": int(len(z_np)),
        "val_nll_before": float(nll_before),
        "val_nll_after": float(nll_after),
        "val_ece_before": float(ece_before),
        "val_ece_after": float(ece_after),
    }


def calibrate_logit(z: float, temperature: float, bias: float) -> float:
    """Apply bounded calibration ``(z + bias) / max(1.0, temperature)``."""
    return (float(z) + float(bias)) / max(1.0, float(temperature))


def apply_calibrated_decision(
    *,
    crop_z: float,
    full_z: float,
    clip_z: float,
    conv_z: float,
    temperature: float,
    bias: float,
    decision_threshold: float,
    uncertain_band: tuple[float, float],
    view_disagreement_margin: float = DEFAULT_VIEW_DISAGREEMENT_MARGIN,
    branch_disagreement_margin: float = DEFAULT_BRANCH_DISAGREEMENT_MARGIN,
) -> dict[str, Any]:
    """Combine calibrated views/branches with explicit view-disagreement & uncertain-band checks."""
    fused_z = 0.5 * (float(crop_z) + float(full_z))
    z_cal = calibrate_logit(fused_z, temperature, bias)
    crop_z_cal = calibrate_logit(crop_z, temperature, bias)
    full_z_cal = calibrate_logit(full_z, temperature, bias)
    clip_z_cal = calibrate_logit(clip_z, temperature, bias)
    conv_z_cal = calibrate_logit(conv_z, temperature, bias)

    p_fake = _sigmoid(z_cal)
    p_crop = _sigmoid(crop_z_cal)
    p_full = _sigmoid(full_z_cal)
    p_clip = _sigmoid(clip_z_cal)
    p_conv = _sigmoid(conv_z_cal)
    p_uncal = _sigmoid(fused_z)

    z_thr = _logit(decision_threshold)
    crop_delta = crop_z_cal - z_thr
    full_delta = full_z_cal - z_thr
    view_disagreement = bool(
        (crop_delta * full_delta < 0.0)
        and (abs(crop_delta) >= view_disagreement_margin)
        and (abs(full_delta) >= view_disagreement_margin)
    )

    clip_delta = clip_z_cal - z_thr
    conv_delta = conv_z_cal - z_thr
    branch_disagreement = bool(
        (clip_delta * conv_delta < 0.0)
        and (abs(clip_delta) >= branch_disagreement_margin)
        and (abs(conv_delta) >= branch_disagreement_margin)
    )

    binary_label = "FAKE" if p_fake >= decision_threshold else "REAL"
    band_lo, band_hi = float(uncertain_band[0]), float(uncertain_band[1])
    if view_disagreement:
        label = "UNCERTAIN"
        uncertain = True
        uncertain_reason: Optional[str] = "view_disagreement"
    elif branch_disagreement:
        label = "UNCERTAIN"
        uncertain = True
        uncertain_reason = "branch_disagreement"
    elif band_lo <= p_fake <= band_hi:
        label = "UNCERTAIN"
        uncertain = True
        uncertain_reason = "borderline_confidence"
    else:
        label = binary_label
        uncertain = False
        uncertain_reason = None

    def _entry(p: float) -> dict[str, Any]:
        b_lbl = "FAKE" if p >= decision_threshold else "REAL"
        lbl = "UNCERTAIN" if (band_lo <= p <= band_hi) else b_lbl
        return {
            "REAL": float(1.0 - p),
            "FAKE": float(p),
            "label": lbl,
            "binary_label": b_lbl,
        }

    confidence = float(p_fake if binary_label == "FAKE" else (1.0 - p_fake))
    return {
        "label": label,
        "binary_label": binary_label,
        "uncertain": uncertain,
        "uncertain_reason": uncertain_reason,
        "confidence": confidence,
        "probabilities": {"REAL": float(1.0 - p_fake), "FAKE": float(p_fake)},
        "uncalibrated_probabilities": {
            "REAL": float(1.0 - p_uncal),
            "FAKE": float(p_uncal),
        },
        "fused_logit": float(fused_z),
        "calibrated_logit": float(z_cal),
        "view_disagreement": view_disagreement,
        "branch_disagreement": branch_disagreement,
        "per_view": {"crop": _entry(p_crop), "full": _entry(p_full)},
        "per_branch": {"clip_mlp": _entry(p_clip), "convnext_v2": _entry(p_conv)},
    }


def _resolve_repo_path(path_or_str: Union[str, Path]) -> Path:
    p = Path(path_or_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class EnsembleDeepfakeDetector:
    """Step 4 dual-branch detector combining CLIP ViT-L/14 + ConvNeXt-Tiny v2, bounded calibration, and C2PA."""

    def __init__(
        self,
        *,
        ensemble_path: Path = STEP3_ENSEMBLE_PATH,
        calibration_path: Path = STEP4_CALIBRATION_PATH,
        mapping_path: Path = CLASS_NAMES_PATH,
        clip_checkpoint: Optional[Path] = None,
        convnext_checkpoint: Optional[Path] = None,
        clip_backbone: str = CLIP_BACKBONE,
        pretrained_clip: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble config missing at {ensemble_path}")

        mapping = load_class_mapping(mapping_path)
        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = mapping["idx_to_class"]
        self.class_names: list[str] = mapping["class_names"]
        self.fake_idx = int(self.class_to_idx["FAKE"])
        self.real_idx = int(self.class_to_idx["REAL"])

        ens_cfg = load_json(ensemble_path)
        self.best_clip_branch = str(ens_cfg.get("best_clip_branch", "clip_mlp"))
        self.clip_weight = float(ens_cfg["ensemble"]["clip_weight"])
        self.convnext_weight = 1.0 - self.clip_weight

        default_clip_key = (
            "clip_mlp_checkpoint"
            if self.best_clip_branch == "clip_mlp"
            else "clip_linear_checkpoint"
        )
        clip_ckpt = clip_checkpoint or _resolve_repo_path(ens_cfg[default_clip_key])
        conv_ckpt = convnext_checkpoint or _resolve_repo_path(
            ens_cfg["convnext_checkpoint"]
        )
        if not clip_ckpt.exists():
            raise FileNotFoundError(f"CLIP probe checkpoint missing at {clip_ckpt}")
        if not conv_ckpt.exists():
            raise FileNotFoundError(f"ConvNeXt checkpoint missing at {conv_ckpt}")

        self.device = device or get_device()
        self.clip_extractor = CLIPFeatureExtractor(
            backbone=clip_backbone, pretrained=pretrained_clip, img_size=224
        ).to(self.device)
        self.clip_head, self.clip_meta = load_clip_head(clip_ckpt, self.device)
        self.conv_model, self.conv_meta = load_checkpoint(conv_ckpt, self.device)
        self.img_size = int(self.conv_meta.get("img_size", 224))

        if calibration_path.exists():
            cal = load_json(calibration_path)
        else:
            logger.warning(
                "Step 4 calibration missing at %s; using fallback T=1.0, b=0.0, thr=0.5",
                calibration_path,
            )
            cal = {
                "temperature": 1.0,
                "bias": 0.0,
                "decision_threshold": 0.5,
                "uncertain_band": [0.40, 0.60],
                "view_disagreement_margin": DEFAULT_VIEW_DISAGREEMENT_MARGIN,
                "branch_disagreement_margin": DEFAULT_BRANCH_DISAGREEMENT_MARGIN,
            }

        self.temperature = max(1.0, float(cal.get("temperature", 1.0)))
        self.bias = float(cal.get("bias", 0.0))
        self.decision_threshold = float(cal.get("decision_threshold", 0.5))
        if "clip_weight" in cal:
            self.clip_weight = float(cal["clip_weight"])
            self.convnext_weight = 1.0 - self.clip_weight
        band = cal.get("uncertain_band", [0.40, 0.60])
        self.uncertain_band: tuple[float, float] = (float(band[0]), float(band[1]))
        self.view_disagreement_margin = float(
            cal.get("view_disagreement_margin", DEFAULT_VIEW_DISAGREEMENT_MARGIN)
        )
        self.branch_disagreement_margin = float(
            cal.get("branch_disagreement_margin", DEFAULT_BRANCH_DISAGREEMENT_MARGIN)
        )

        self.clip_transform = build_clip_transforms(train=False, img_size=224)
        self.conv_transform = build_convnext_transforms(
            train=False, img_size=self.img_size
        )
        self.transform = self.conv_transform
        self.cam = GradCAM(self.conv_model, mapping_path=mapping_path)
        self.meta: dict[str, Any] = {
            "backbone": f"CLIP-L/14({self.best_clip_branch})+convnext_tiny",
            "epoch": self.conv_meta.get("epoch"),
        }
        self._lock = Lock()
        logger.info(
            "Loaded EnsembleDeepfakeDetector on %s | w_clip=%.2f | T=%.4f b=%+.4f thr=%.4f band=[%.3f, %.3f]",
            self.device,
            self.clip_weight,
            self.temperature,
            self.bias,
            self.decision_threshold,
            self.uncertain_band[0],
            self.uncertain_band[1],
        )

    def _diff(self, logits_row: torch.Tensor) -> float:
        return float(logits_row[self.fake_idx].item() - logits_row[self.real_idx].item())

    @torch.no_grad()
    def _forward_two_views(
        self,
        pil: Image.Image,
        cropped: Image.Image,
        use_tta: bool = True,
    ) -> dict[str, float]:
        imgs = [cropped, pil] if use_tta else [cropped]
        clip_batch = torch.stack(
            [self.clip_transform(im) for im in imgs], dim=0
        ).to(self.device)
        conv_batch = torch.stack(
            [self.conv_transform(im) for im in imgs], dim=0
        ).to(self.device)

        use_fp16 = self.device.type in ("mps", "cuda")
        clip_feats = self.clip_extractor.extract_features(clip_batch, use_fp16=use_fp16)
        clip_logits = self.clip_head(clip_feats).float().cpu()
        conv_logits = self.conv_model(conv_batch).float().cpu()

        clip_crop_z = self._diff(clip_logits[0])
        conv_crop_z = self._diff(conv_logits[0])
        if len(imgs) > 1:
            clip_full_z = self._diff(clip_logits[1])
            conv_full_z = self._diff(conv_logits[1])
        else:
            clip_full_z = clip_crop_z
            conv_full_z = conv_crop_z

        crop_z = self.clip_weight * clip_crop_z + self.convnext_weight * conv_crop_z
        full_z = self.clip_weight * clip_full_z + self.convnext_weight * conv_full_z
        clip_z = 0.5 * (clip_crop_z + clip_full_z)
        conv_z = 0.5 * (conv_crop_z + conv_full_z)
        return {
            "crop_z": crop_z,
            "full_z": full_z,
            "clip_z": clip_z,
            "conv_z": conv_z,
        }

    def _evaluate_pixel_decision(
        self, view_logits: dict[str, float]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        dec = apply_calibrated_decision(
            crop_z=view_logits["crop_z"],
            full_z=view_logits["full_z"],
            clip_z=view_logits["clip_z"],
            conv_z=view_logits["conv_z"],
            temperature=self.temperature,
            bias=self.bias,
            decision_threshold=self.decision_threshold,
            uncertain_band=self.uncertain_band,
            view_disagreement_margin=self.view_disagreement_margin,
            branch_disagreement_margin=self.branch_disagreement_margin,
        )
        pred_idx = self.fake_idx if dec["binary_label"] == "FAKE" else self.real_idx
        pixel_signal = {
            "status": "ok",
            "label": dec["label"],
            "binary_label": dec["binary_label"],
            "uncertain": dec["uncertain"],
            "uncertain_reason": dec["uncertain_reason"],
            "class_index": pred_idx,
            "confidence": dec["confidence"],
            "probabilities": dec["probabilities"],
            "uncalibrated_probabilities": dec["uncalibrated_probabilities"],
            "temperature": self.temperature,
        }
        per_view_probs = {
            "crop": dec["per_view"]["crop"],
            "full_frame": dec["per_view"]["full"],
        }
        return pixel_signal, dec, per_view_probs

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
        decision: Optional[dict[str, Any]],
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
                "decision_threshold": self.decision_threshold,
                "uncertain_band": list(self.uncertain_band),
                "class_to_idx": self.class_to_idx,
            }

        if override:
            label = "FAKE"
            binary_label = "FAKE"
            uncertain = False
            uncertain_reason = None
            verdict_source = "metadata"
            verdict_display = "FAKE (source: metadata)"
            confidence = 1.0
            probs = {"REAL": 0.0, "FAKE": 1.0}
            class_index = self.fake_idx
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
            assert pixel_signal is not None and decision is not None
            label = pixel_signal["label"]
            binary_label = pixel_signal["binary_label"]
            uncertain = pixel_signal["uncertain"]
            uncertain_reason = decision["uncertain_reason"]
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
            elif uncertain_reason == "view_disagreement":
                verdict_display = "UNCERTAIN (face and full frame disagree)"
            elif uncertain_reason == "branch_disagreement":
                verdict_display = "UNCERTAIN (CLIP and ConvNeXt models disagree)"
            else:
                verdict_display = "UNCERTAIN"

        ensemble_details = (
            {
                "clip_branch": self.best_clip_branch,
                "clip_weight": self.clip_weight,
                "convnext_weight": self.convnext_weight,
                "fused_logit": decision["fused_logit"],
                "calibrated_logit": decision["calibrated_logit"],
                "per_branch": decision["per_branch"],
                "per_view": decision["per_view"],
            }
            if decision is not None
            else None
        )

        return {
            "status": "ok",
            "label": label,
            "binary_label": binary_label,
            "uncertain": uncertain,
            "uncertain_reason": uncertain_reason,
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
            "decision_threshold": self.decision_threshold,
            "uncertain_band": list(self.uncertain_band),
            "view_disagreement": bool(
                decision["view_disagreement"] if decision is not None else False
            ),
            "branch_disagreement": bool(
                decision["branch_disagreement"] if decision is not None else False
            ),
            "ensemble_details": ensemble_details,
            "tta_views": ["face_crop", "full_frame"] if use_tta else ["face_crop"],
            "pixel_model": pixel_signal,
            "provenance": provenance,
            "content_credentials": provenance,
            "diagnostics": diagnostics,
            "preprocessing": face_meta,
            "class_to_idx": self.class_to_idx,
            "image_size": list(pil.size),
            "model": self.meta["backbone"],
        }

    @torch.no_grad()
    def predict(
        self,
        image: ImageInput,
        require_face: bool = True,
        raw_bytes: Optional[bytes] = None,
        use_tta: bool = True,
        detected_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run C2PA provenance, face gate, CLIP+ConvNeXt ensemble, bounded calibration & disagreement checks."""
        cred_source = raw_bytes if raw_bytes is not None else image
        provenance = inspect_provenance(cred_source, detected_format=detected_format)
        pil = _to_pil(image)

        if require_face:
            cropped, face_meta = gate_and_crop_face(pil)
        else:
            cropped, face_meta = crop_face_or_full(pil)

        pixel_signal: Optional[dict[str, Any]] = None
        decision: Optional[dict[str, Any]] = None
        per_view: Optional[dict[str, dict[str, Any]]] = None
        if cropped is not None:
            with self._lock:
                view_logits = self._forward_two_views(pil, cropped, use_tta=use_tta)
                pixel_signal, decision, per_view = self._evaluate_pixel_decision(
                    view_logits
                )

        return self._assemble_response(
            pil=pil,
            face_meta=face_meta,
            provenance=provenance,
            pixel_signal=pixel_signal,
            decision=decision,
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
        """Run ensemble inference with ConvNeXt-v2 Grad-CAM overlay."""
        cred_source = raw_bytes if raw_bytes is not None else image
        provenance = inspect_provenance(cred_source, detected_format=detected_format)
        pil = _to_pil(image)

        if require_face:
            cropped, face_meta = gate_and_crop_face(pil)
        else:
            cropped, face_meta = crop_face_or_full(pil)

        pixel_signal: Optional[dict[str, Any]] = None
        decision: Optional[dict[str, Any]] = None
        per_view: Optional[dict[str, dict[str, Any]]] = None
        cam_b64: Optional[str] = None
        cam_target: Optional[str] = None

        if cropped is not None:
            x_full = self.conv_transform(pil).unsqueeze(0).to(self.device)
            with self._lock:
                view_logits = self._forward_two_views(pil, cropped, use_tta=use_tta)
                pixel_signal, decision, per_view = self._evaluate_pixel_decision(
                    view_logits
                )
                cam_target = pixel_signal["binary_label"]
                cam, _, _ = self.cam(x_full, target_class=cam_target)
            cam_b64 = _png_b64(overlay_heatmap(pil, cam))

        result = self._assemble_response(
            pil=pil,
            face_meta=face_meta,
            provenance=provenance,
            pixel_signal=pixel_signal,
            decision=decision,
            per_view_probs=per_view,
            use_tta=use_tta,
        )
        result["preview"] = encode_jpeg_preview_b64(pil)
        if cam_b64 is not None:
            result["gradcam"] = cam_b64
            result["gradcam_target"] = cam_target
        return result
