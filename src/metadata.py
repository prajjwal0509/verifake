"""Container metadata (C2PA / JUMBF / XMP / EXIF) inspection as supporting evidence."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from typing import Any, Union

from PIL import Image

from src.image_io import register_image_plugins

register_image_plugins()

DISCLAIMER = (
    "Supporting evidence only — metadata and Content Credentials can be stripped or "
    "forged and never override the visual model."
)

DIGITAL_SOURCE_TYPES: tuple[tuple[bytes, str], ...] = (
    (b"compositewithtrainedalgorithmicmedia", "compositeWithTrainedAlgorithmicMedia"),
    (b"trainedalgorithmicmedia", "trainedAlgorithmicMedia"),
    (b"algorithmicmedia", "algorithmicMedia"),
    (b"digitalcapture", "digitalCapture"),
)

GENERATOR_PATTERNS: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    ("OpenAI", (b"openai",)),
    ("ChatGPT", (b"chatgpt",)),
    ("DALL-E", (b"dall-e", b"dall\xc2\xb7e", b"dall\xe2\x80\x91e", b"dalle")),
    ("Midjourney", (b"midjourney",)),
    ("Stable Diffusion", (b"stable diffusion", b"stablediffusion")),
    ("FLUX", (b"black forest labs", b"flux.1", b"flux-1")),
    ("Adobe Firefly", (b"firefly",)),
    ("Google Imagen", (b"imagen",)),
    ("ComfyUI", (b"comfyui",)),
    ("Automatic1111", (b"automatic1111",)),
)

EXIF_TAGS_OF_INTEREST: dict[int, str] = {
    271: "Make",
    272: "Model",
    305: "Software",
    306: "DateTime",
}


def _extract_png_metadata(raw: bytes) -> tuple[list[bytes], bool]:
    """Extract metadata chunks (caBX, tEXt, iTXt, zTXt, eXIf) from a PNG byte stream."""
    chunks: list[bytes] = []
    has_cabx = False
    pos = 8
    n = len(raw)
    while pos + 8 <= n:
        length, ctype = struct.unpack(">I4s", raw[pos : pos + 8])
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > n:
            break
        payload = raw[data_start:data_end]
        if ctype == b"caBX":
            has_cabx = True
            chunks.append(payload)
        elif ctype in (b"tEXt", b"eXIf"):
            chunks.append(payload)
        elif ctype == b"zTXt":
            sep = payload.find(b"\x00")
            if sep != -1 and sep + 2 <= len(payload):
                chunks.append(payload[:sep])
                try:
                    chunks.append(zlib.decompress(payload[sep + 2 :]))
                except zlib.error:
                    pass
        elif ctype == b"iTXt":
            # keyword\0 + comp_flag(1) + comp_method(1) + lang\0 + translated\0 + text
            sep = payload.find(b"\x00")
            if sep != -1 and sep + 3 <= len(payload):
                comp_flag = payload[sep + 1]
                rest = payload[sep + 3 :]
                for _ in range(2):
                    idx = rest.find(b"\x00")
                    if idx == -1:
                        break
                    rest = rest[idx + 1 :]
                if comp_flag == 1:
                    try:
                        chunks.append(zlib.decompress(rest))
                    except zlib.error:
                        pass
                else:
                    chunks.append(rest)
        pos = data_end + 4
        if ctype == b"IEND":
            break
    return chunks, has_cabx


def _extract_jpeg_metadata(raw: bytes) -> tuple[list[bytes], bool]:
    """Extract APP1 (EXIF/XMP), APP11 (JUMBF/C2PA), APP13 (IPTC), and COM markers."""
    chunks: list[bytes] = []
    has_app11_c2pa = False
    pos = 2
    n = len(raw)
    while pos + 2 <= n:
        while pos < n and raw[pos] != 0xFF:
            pos += 1
        while pos < n and raw[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = raw[pos]
        pos += 1
        if marker in (0xD9, 0xDA):  # EOI or SOS (start of compressed scan)
            break
        if marker == 0x00 or marker == 0x01 or (0xD0 <= marker <= 0xD8):
            continue
        if pos + 2 > n:
            break
        seg_len = struct.unpack(">H", raw[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > n:
            break
        seg_data = raw[pos + 2 : pos + seg_len]
        if marker in (0xE1, 0xEB, 0xED, 0xFE):
            chunks.append(seg_data)
            if marker == 0xEB:
                low = seg_data.lower()
                if b"c2pa" in low or b"jumb" in low:
                    has_app11_c2pa = True
        pos += seg_len
    return chunks, has_app11_c2pa


def _extract_webp_metadata(raw: bytes) -> tuple[list[bytes], bool]:
    """Extract C2PA, JUMB, XMP, and EXIF chunks from a RIFF/WebP byte stream."""
    chunks: list[bytes] = []
    has_webp_c2pa = False
    pos = 12
    n = len(raw)
    while pos + 8 <= n:
        ctype, clen = struct.unpack("<4sI", raw[pos : pos + 8])
        data_start = pos + 8
        data_end = min(n, data_start + clen)
        if ctype in (b"C2PA", b"JUMB", b"XMP ", b"EXIF"):
            chunks.append(raw[data_start:data_end])
            if ctype in (b"C2PA", b"JUMB"):
                has_webp_c2pa = True
        pos = data_start + clen + (clen & 1)
    return chunks, has_webp_c2pa


def _extract_isobmff_metadata(raw: bytes) -> tuple[list[bytes], bool]:
    """Extract metadata/C2PA boxes from ISOBMFF containers (HEIC/HEIF/AVIF), skipping mdat pixel payloads."""
    chunks: list[bytes] = []
    has_isobmff_c2pa = False
    pos = 0
    n = len(raw)
    while pos + 8 <= n:
        box_size, box_type = struct.unpack(">I4s", raw[pos : pos + 8])
        hdr = 8
        if box_size == 1:
            if pos + 16 > n:
                break
            box_size = struct.unpack(">Q", raw[pos + 8 : pos + 16])[0]
            hdr = 16
        elif box_size == 0:
            box_size = n - pos
        if box_size < hdr or pos + box_size > n:
            break
        if box_type in (b"meta", b"uuid", b"jumb", b"c2pa", b"moov", b"udta"):
            payload = raw[pos + hdr : pos + box_size]
            chunks.append(payload)
            low = payload.lower()
            if box_type in (b"jumb", b"c2pa") or b"c2pa" in low or b"jumb" in low:
                has_isobmff_c2pa = True
        pos += box_size
    return chunks, has_isobmff_c2pa


def _extract_exif_dict(raw: bytes) -> dict[str, str]:
    """Read standard camera EXIF fields via Pillow without failing on corrupt tags."""
    register_image_plugins()
    out: dict[str, str] = {}
    try:
        with Image.open(io.BytesIO(raw)) as img:
            exif = img.getexif()
            if exif:
                for tag_id, tag_name in EXIF_TAGS_OF_INTEREST.items():
                    val = exif.get(tag_id)
                    if val is not None:
                        s = str(val).strip()
                        if s:
                            out[tag_name] = s[:80]
    except Exception:
        pass
    return out


def inspect_content_credentials(
    source: Union[bytes, bytearray, str, Path, Image.Image],
) -> dict[str, Any]:
    """Inspect C2PA / JUMBF / XMP / EXIF metadata and return a structured supporting summary."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        return {
            "status": "none",
            "has_c2pa": False,
            "digital_source_type": None,
            "generator_claims": [],
            "camera_exif": {},
            "summary": "No container metadata available (in-memory image).",
            "disclaimer": DISCLAIMER,
        }

    meta_chunks: list[bytes] = []
    has_container_c2pa = False
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        meta_chunks, has_container_c2pa = _extract_png_metadata(raw)
    elif raw[:2] == b"\xff\xd8":
        meta_chunks, has_container_c2pa = _extract_jpeg_metadata(raw)
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        meta_chunks, has_container_c2pa = _extract_webp_metadata(raw)
    elif len(raw) >= 12 and raw[4:8] == b"ftyp":
        meta_chunks, has_container_c2pa = _extract_isobmff_metadata(raw)

    blob_lower = b"\n".join(meta_chunks).lower()
    has_c2pa = bool(
        has_container_c2pa
        or b"c2pa" in blob_lower
        or b"jumb" in blob_lower
        or b"c2ma" in blob_lower
        or b"c2cl" in blob_lower
    )

    digital_source_type: str | None = None
    for needle, canonical in DIGITAL_SOURCE_TYPES:
        if needle in blob_lower:
            digital_source_type = canonical
            break

    generator_claims: list[str] = []
    for label, needles in GENERATOR_PATTERNS:
        if any(needle in blob_lower for needle in needles):
            generator_claims.append(label)

    camera_exif = _extract_exif_dict(raw)

    if (
        digital_source_type and "algorithmic" in digital_source_type.lower()
    ) or generator_claims:
        status = "ai_credentials_found"
        parts = []
        if has_c2pa:
            parts.append("C2PA JUMBF manifest present")
        if digital_source_type:
            parts.append(f"digitalSourceType={digital_source_type}")
        if generator_claims:
            parts.append(f"claims={', '.join(generator_claims)}")
        summary = " · ".join(parts)
    elif has_c2pa:
        status = "c2pa_present"
        summary = (
            "C2PA Content Credentials manifest present (no AI generator claim parsed)."
        )
    elif camera_exif.get("Make") or camera_exif.get("Model"):
        status = "camera_exif_only"
        cam = " ".join(
            v for v in (camera_exif.get("Make"), camera_exif.get("Model")) if v
        )
        summary = f"No C2PA manifest; camera EXIF present ({cam})."
    elif camera_exif.get("Software"):
        status = "none"
        summary = f"No C2PA Content Credentials found (Software tag: {camera_exif['Software']})."
    else:
        status = "none"
        summary = "No C2PA Content Credentials or camera EXIF metadata found."

    return {
        "status": status,
        "has_c2pa": has_c2pa,
        "digital_source_type": digital_source_type,
        "generator_claims": generator_claims,
        "camera_exif": camera_exif,
        "summary": summary,
        "disclaimer": DISCLAIMER,
    }
