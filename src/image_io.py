"""Content-based multi-format image decoding, validation, EXIF rotation, and preview encoding."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils import get_logger

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 12000
MAX_IMAGE_PIXELS = (
    64_000_000  # 64 MP cap prevents decompression bombs while allowing 48 MP photos
)
ALLOWED_FORMATS = frozenset(
    {"JPEG", "MPO", "PNG", "WEBP", "BMP", "TIFF", "GIF", "AVIF", "HEIF"}
)

_PLUGINS_REGISTERED = False


class ImageValidationError(ValueError):
    """Structured HTTP-friendly validation / decode error."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def register_image_plugins() -> None:
    """Configure Pillow pixel limits and register the HEIF/HEIC opener (idempotent)."""
    global _PLUGINS_REGISTERED
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    if _PLUGINS_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        logger.warning(
            "pillow-heif is not installed; HEIC/HEIF decoding will be unavailable."
        )
    _PLUGINS_REGISTERED = True


register_image_plugins()


def to_rgb_safe(img: Image.Image) -> Image.Image:
    """Convert any Pillow image mode (RGBA, palette, 16-bit, float, CMYK, L) cleanly to 8-bit RGB."""
    mode = img.mode
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        mode = img.mode

    if mode in ("RGBA", "LA", "PA", "RGBa"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")

    if mode.startswith("I;16") or mode == "I":
        arr = np.asarray(img, dtype=np.float32)
        if arr.size:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > 65535.0 or mn < 0.0:
                arr = (arr - mn) / max(mx - mn, 1.0) * 255.0
            elif mx > 255.0:
                arr = arr / 257.0
        u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(u8, mode="L").convert("RGB")

    if mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        if arr.size:
            mn, mx = float(arr.min()), float(arr.max())
            if mn >= 0.0 and mx <= 1.0:
                arr = arr * 255.0
            else:
                arr = (arr - mn) / max(mx - mn, 1e-6) * 255.0
        u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(u8, mode="L").convert("RGB")

    return img if mode == "RGB" else img.convert("RGB")


def decode_image_bytes(
    data: bytes,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> tuple[Image.Image, str]:
    """Validate bounds, detect format from bytes, orient via EXIF, and return (rgb_image, format)."""
    if not data:
        raise ImageValidationError(400, "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            413,
            f"File is too large ({len(data) / 1e6:.1f} MB). Maximum allowed size is {max_bytes // (1024 * 1024)} MB.",
        )

    register_image_plugins()
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            400,
            f"Image exceeds the maximum pixel limit ({MAX_IMAGE_SIDE:,} px per side / {MAX_IMAGE_PIXELS // 1_000_000} MP).",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            400,
            "The uploaded file could not be decoded as a supported image (JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF, HEIC/HEIF).",
        ) from exc

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(
            415,
            f"Unsupported image format '{fmt or 'unknown'}'. Supported formats: JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF, HEIC/HEIF.",
        )

    w, h = img.size
    if w > MAX_IMAGE_SIDE or h > MAX_IMAGE_SIDE or (w * h) > MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            400,
            f"Image dimensions ({w}×{h}) exceed the maximum allowed limit ({MAX_IMAGE_SIDE:,}px per side / {MAX_IMAGE_PIXELS // 1_000_000} MP).",
        )
    if min(w, h) < MIN_IMAGE_SIDE:
        raise ImageValidationError(
            400, f"Image is too small ({w}×{h}). Minimum side is {MIN_IMAGE_SIDE}px."
        )

    try:
        if fmt == "GIF" or getattr(img, "is_animated", False):
            img.seek(0)
        img.load()
        img = ImageOps.exif_transpose(img)
        rgb = to_rgb_safe(img)
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            400,
            f"Image exceeds the maximum pixel limit ({MAX_IMAGE_SIDE:,} px per side / {MAX_IMAGE_PIXELS // 1_000_000} MP).",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ImageValidationError(
            400, "The uploaded image file is corrupted or truncated."
        ) from exc

    rw, rh = rgb.size
    if (
        min(rw, rh) < MIN_IMAGE_SIDE
        or rw > MAX_IMAGE_SIDE
        or rh > MAX_IMAGE_SIDE
        or (rw * rh) > MAX_IMAGE_PIXELS
    ):
        raise ImageValidationError(
            400,
            f"Decoded image dimensions ({rw}×{rh}) are outside allowed bounds ({MIN_IMAGE_SIDE}px–{MAX_IMAGE_SIDE:,}px).",
        )

    return rgb, fmt


def encode_jpeg_preview_b64(
    img: Image.Image, max_side: int = 800, quality: int = 88
) -> str:
    """Encode an already-decoded RGB image as a JPEG data URL strictly for browser preview display."""
    thumb = to_rgb_safe(img).copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
