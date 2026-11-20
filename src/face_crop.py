"""OpenCV DNN (YuNet) face detection, face gating, and small-face cropping."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from src.utils import MODELS_DIR, get_logger

logger = get_logger(__name__)

YUNET_PATH: Path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

SMALL_FACE_WIDTH_PCT: float = 15.0
NORMAL_MARGIN: float = 0.35
SMALL_FACE_MARGIN: float = 0.20
SMALL_FACE_MIN_UPSAMPLE: int = 256

_detector_lock = Lock()
_yunet: Optional[Any] = None
_yunet_threshold: Optional[float] = None


def _get_yunet(score_threshold: float = 0.55) -> Optional[Any]:
    """Lazily initialize the OpenCV DNN YuNet face detector."""
    global _yunet, _yunet_threshold
    if not YUNET_PATH.exists():
        return None
    if _yunet is None or _yunet_threshold != score_threshold:
        _yunet = cv2.FaceDetectorYN.create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        _yunet_threshold = score_threshold
    return _yunet


def _run_yunet_at_side(
    rgb: np.ndarray,
    score_threshold: float,
    max_detect_side: int,
) -> Optional[dict[str, Any]]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_detect_side / max(w, h))
    dw, dh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)

    with _detector_lock:
        det = _get_yunet(score_threshold=score_threshold)
        if det is None:
            return None
        det.setInputSize((dw, dh))
        _, faces = det.detect(bgr)

    if faces is None or len(faces) == 0:
        return None

    best_row = None
    best_weight = -1.0
    for row in faces:
        fx, fy, fw, fh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        score = float(row[-1])
        if score < score_threshold or fw <= 12 or fh <= 12:
            continue
        weight = (fw * fh) * score
        if weight > best_weight:
            best_weight = weight
            best_row = (fx / scale, fy / scale, fw / scale, fh / scale, score)

    if best_row is None:
        return None

    x, y, bw, bh, score = best_row
    return {
        "bbox": [int(round(x)), int(round(y)), int(round(x + bw)), int(round(y + bh))],
        "score": float(score),
    }


def detect_primary_face(
    img: Image.Image,
    score_threshold: float = 0.55,
    max_detect_side: int = 1280,
) -> Optional[dict[str, Any]]:
    """Detect primary face in img via YuNet, retrying large scenes at 1920px."""
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    if min(w, h) < 48:
        return None

    det = _run_yunet_at_side(
        rgb, score_threshold=score_threshold, max_detect_side=max_detect_side
    )
    if det is None and max(w, h) > max_detect_side and max_detect_side < 1920:
        det = _run_yunet_at_side(
            rgb, score_threshold=score_threshold, max_detect_side=1920
        )
    return det


def gate_and_crop_face(
    img: Image.Image,
    margin: float = NORMAL_MARGIN,
    score_threshold: float = 0.55,
    tight_tile_max_side: int = 256,
) -> tuple[Optional[Image.Image], dict[str, Any]]:
    """Run YuNet face gate and return (face_crop_or_None, diagnostics)."""
    rgb_img = img.convert("RGB")
    w, h = rgb_img.size
    det = detect_primary_face(rgb_img, score_threshold=score_threshold)
    if det is None:
        return None, {
            "face_detected": False,
            "mode": "no_face",
            "bbox": None,
            "raw_face_bbox": None,
            "face_score": 0.0,
            "image_resolution": [w, h],
            "face_width_pct": 0.0,
            "face_height_pct": 0.0,
            "raw_crop_resolution": None,
            "crop_resolution": None,
            "crop_margin": None,
            "upsampled": False,
            "small_face": False,
            "warning": None,
        }

    score = round(float(det["score"]), 4)
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    face_width_pct = round(100.0 * bw / max(1, w), 2)
    face_height_pct = round(100.0 * bh / max(1, h), 2)
    small_face = bool(face_width_pct < SMALL_FACE_WIDTH_PCT)
    warning = "Small face: low confidence" if small_face else None

    if max(w, h) <= tight_tile_max_side:
        return rgb_img, {
            "face_detected": True,
            "mode": "yunet_verified_tile",
            "bbox": [0, 0, w, h],
            "raw_face_bbox": det["bbox"],
            "face_score": score,
            "image_resolution": [w, h],
            "face_width_pct": face_width_pct,
            "face_height_pct": face_height_pct,
            "raw_crop_resolution": [w, h],
            "crop_resolution": [w, h],
            "crop_margin": 0.0,
            "upsampled": False,
            "small_face": small_face,
            "warning": warning,
        }

    active_margin = SMALL_FACE_MARGIN if small_face else margin
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * active_margin)
    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(w, int(round(cx + side / 2.0)))
    ny2 = min(h, int(round(cy + side / 2.0)))
    if (nx2 - nx1) < 48 or (ny2 - ny1) < 48:
        return rgb_img, {
            "face_detected": True,
            "mode": "yunet_verified_tile",
            "bbox": [0, 0, w, h],
            "raw_face_bbox": det["bbox"],
            "face_score": score,
            "image_resolution": [w, h],
            "face_width_pct": face_width_pct,
            "face_height_pct": face_height_pct,
            "raw_crop_resolution": [w, h],
            "crop_resolution": [w, h],
            "crop_margin": 0.0,
            "upsampled": False,
            "small_face": small_face,
            "warning": warning,
        }

    cropped = rgb_img.crop((nx1, ny1, nx2, ny2))
    raw_crop_res = [cropped.width, cropped.height]
    upsampled = False
    if small_face and min(cropped.size) < SMALL_FACE_MIN_UPSAMPLE:
        scale_up = SMALL_FACE_MIN_UPSAMPLE / min(cropped.size)
        nw = max(SMALL_FACE_MIN_UPSAMPLE, int(round(cropped.width * scale_up)))
        nh = max(SMALL_FACE_MIN_UPSAMPLE, int(round(cropped.height * scale_up)))
        cropped = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
        upsampled = True

    return cropped, {
        "face_detected": True,
        "mode": "opencv_dnn_yunet_crop",
        "bbox": [nx1, ny1, nx2, ny2],
        "raw_face_bbox": det["bbox"],
        "face_score": score,
        "image_resolution": [w, h],
        "face_width_pct": face_width_pct,
        "face_height_pct": face_height_pct,
        "raw_crop_resolution": raw_crop_res,
        "crop_resolution": [cropped.width, cropped.height],
        "crop_margin": active_margin,
        "upsampled": upsampled,
        "small_face": small_face,
        "warning": warning,
    }


def crop_face_or_full(
    img: Image.Image,
    margin: float = NORMAL_MARGIN,
    score_threshold: float = 0.55,
    skip_if_tight_le: int = 256,
) -> tuple[Image.Image, dict[str, Any]]:
    """Face crop with full-frame fallback (for out-of-scope batch auditing only)."""
    rgb_img = img.convert("RGB")
    w, h = rgb_img.size
    cropped, meta = gate_and_crop_face(
        rgb_img,
        margin=margin,
        score_threshold=score_threshold,
        tight_tile_max_side=skip_if_tight_le,
    )
    if cropped is None:
        return rgb_img, {
            "face_detected": False,
            "mode": "full_frame_fallback",
            "bbox": [0, 0, w, h],
            "raw_face_bbox": None,
            "face_score": 0.0,
            "image_resolution": [w, h],
            "face_width_pct": 0.0,
            "face_height_pct": 0.0,
            "raw_crop_resolution": [w, h],
            "crop_resolution": [w, h],
            "crop_margin": 0.0,
            "upsampled": False,
            "small_face": False,
            "warning": None,
        }
    return cropped, meta
