"""Cryptographic C2PA Content Credentials check using c2pa-python."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import c2pa
from PIL import Image

from src.metadata import _extract_exif_dict

ABSENT_BADGE = "No credentials (this does not mean the image is real)"
AI_VERIFIED_BADGE = "Content Credentials: AI-generated"
PROVENANCE_DISCLAIMER = (
    "Missing or stripped Content Credentials never imply an image is real — social platforms "
    "and image exports routinely remove metadata."
)

FORMAT_TO_MIME: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "JPG": "image/jpeg",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
    "HEIF": "image/heic",
    "HEIC": "image/heic",
    "TIFF": "image/tiff",
    "TIF": "image/tiff",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}

AI_SOURCE_TOKENS: tuple[str, ...] = (
    "compositewithtrainedalgorithmicmedia",
    "trainedalgorithmicmedia",
    "algorithmicmedia",
    "compositesynthetic",
)

# Codes emitted purely because default local validation lacks an external TSA/root trust list
# while cryptographic signature + asset hash verification (`Valid`) succeeded cleanly.
NON_TAMPER_TRUST_CODES: frozenset[str] = frozenset(
    {
        "signingCredential.untrusted",
        "signingCredential.ocsp.skipped",
        "signingCredential.ocsp.notResponded",
        "timeStamp.untrusted",
    }
)


def _guess_mime(
    raw: bytes, detected_format: Optional[str] = None, suffix: str = ""
) -> str:
    if detected_format:
        mime = FORMAT_TO_MIME.get(detected_format.upper())
        if mime:
            return mime
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12].lower()
        if brand in (b"avif", b"avis"):
            return "image/avif"
        return "image/heic"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if suffix:
        ext_mime = FORMAT_TO_MIME.get(suffix.lstrip(".").upper())
        if ext_mime:
            return ext_mime
    return "image/jpeg"


def _is_absent_or_unsupported_c2pa_error(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    return (
        "ManifestNotFound" in name
        or "NotSupported" in name
        or "ManifestNotFound" in msg
        or "no JUMBF data found" in msg
        or "NotSupported:" in msg
    )


def _read_c2pa_store(raw: bytes, mime: str) -> Optional[dict[str, Any]]:
    """Read and parse C2PA manifest store using ``c2pa.Reader``."""
    try:
        with c2pa.Reader(mime, io.BytesIO(raw)) as reader:
            return json.loads(reader.json())
    except Exception as exc:
        if _is_absent_or_unsupported_c2pa_error(exc):
            return None
        # Some container variants work better via file-backed Reader
        ext = "." + mime.split("/")[-1].replace("jpeg", "jpg")
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        try:
            os.close(fd)
            Path(tmp_path).write_bytes(raw)
            with c2pa.Reader(tmp_path) as reader:
                return json.loads(reader.json())
        except Exception as inner_exc:
            if _is_absent_or_unsupported_c2pa_error(inner_exc):
                return None
            raise inner_exc
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _short_source_type(uri_or_name: str) -> str:
    return uri_or_name.rstrip("/").rsplit("/", 1)[-1]


def _format_software_agent(sa: Any) -> Optional[str]:
    if isinstance(sa, dict):
        name = str(sa.get("name") or "").strip()
        ver = str(sa.get("version") or "").strip()
        if name and ver:
            return f"{name} ({ver})"
        return name or ver or None
    if isinstance(sa, str) and sa.strip():
        return sa.strip()
    return None


def inspect_provenance(
    source: Union[bytes, bytearray, str, Path, Image.Image],
    detected_format: Optional[str] = None,
) -> dict[str, Any]:
    """Inspect cryptographic C2PA Content Credentials (and camera EXIF) before inference."""
    suffix = ""
    if isinstance(source, (str, Path)):
        p = Path(source)
        suffix = p.suffix
        raw = p.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        return {
            "status": "absent",
            "present": False,
            "signature_valid": False,
            "validation_state": None,
            "validation_status": [],
            "signer": None,
            "signed_time": None,
            "claim_generator": None,
            "software_agents": [],
            "digital_source_types": [],
            "digital_source_type": None,
            "ai_assertions": [],
            "declares_ai_generated": False,
            "ai_verified_by_credentials": False,
            "badge": ABSENT_BADGE,
            "camera_exif": {},
            "summary": f"{ABSENT_BADGE}.",
            "disclaimer": PROVENANCE_DISCLAIMER,
        }

    camera_exif = _extract_exif_dict(raw)
    mime = _guess_mime(raw, detected_format=detected_format, suffix=suffix)

    try:
        store = _read_c2pa_store(raw, mime)
    except Exception as exc:
        cam = " ".join(
            v for v in (camera_exif.get("Make"), camera_exif.get("Model")) if v
        )
        return {
            "status": "error",
            "present": False,
            "signature_valid": False,
            "validation_state": "Invalid",
            "validation_status": [
                {"code": "c2pa.read_error", "explanation": str(exc)[:180]}
            ],
            "signer": None,
            "signed_time": None,
            "claim_generator": None,
            "software_agents": [],
            "digital_source_types": [],
            "digital_source_type": None,
            "ai_assertions": [],
            "declares_ai_generated": False,
            "ai_verified_by_credentials": False,
            "badge": ABSENT_BADGE,
            "camera_exif": camera_exif,
            "summary": f"C2PA parse error ({exc}); {ABSENT_BADGE}."
            + (f" Camera EXIF only: {cam}." if cam else ""),
            "disclaimer": PROVENANCE_DISCLAIMER,
        }

    if not store or not store.get("active_manifest"):
        cam = " ".join(
            v for v in (camera_exif.get("Make"), camera_exif.get("Model")) if v
        )
        exif_note = (
            f" Camera EXIF tag present ({cam}) — EXIF can be spoofed." if cam else ""
        )
        return {
            "status": "absent",
            "present": False,
            "signature_valid": False,
            "validation_state": None,
            "validation_status": [],
            "signer": None,
            "signed_time": None,
            "claim_generator": None,
            "software_agents": [],
            "digital_source_types": [],
            "digital_source_type": None,
            "ai_assertions": [],
            "declares_ai_generated": False,
            "ai_verified_by_credentials": False,
            "badge": ABSENT_BADGE,
            "camera_exif": camera_exif,
            "summary": f"{ABSENT_BADGE}.{exif_note}",
            "disclaimer": PROVENANCE_DISCLAIMER,
        }

    manifests: dict[str, Any] = store.get("manifests", {})
    active_id: str = store["active_manifest"]
    active = manifests.get(active_id, {})

    validation_state = store.get("validation_state")
    raw_status = store.get("validation_status") or []
    validation_status = [
        {"code": str(s.get("code", "")), "explanation": str(s.get("explanation", ""))}
        for s in raw_status
        if isinstance(s, dict)
    ]

    val_results = store.get("validation_results") or {}
    active_res = val_results.get("activeManifest") or {}
    failure_codes = [
        str(x.get("code", ""))
        for x in (active_res.get("failure") or [])
        if isinstance(x, dict)
    ]
    success_codes = {
        str(x.get("code", ""))
        for x in (active_res.get("success") or [])
        if isinstance(x, dict)
    }
    all_status_codes = [s["code"] for s in validation_status]

    hard_failures = [
        c
        for c in (failure_codes + all_status_codes)
        if c and c not in NON_TAMPER_TRUST_CODES
    ]
    has_crypto_proof = bool(
        validation_state in ("Trusted", "Valid")
        and not hard_failures
        and (
            validation_state == "Trusted"
            or "claimSignature.validated" in success_codes
            or "assertion.dataHash.match" in success_codes
        )
    )
    signature_valid = has_crypto_proof

    sig_info = active.get("signature_info") or {}
    common_name = str(sig_info.get("common_name") or "").strip()
    issuer = str(sig_info.get("issuer") or "").strip()
    if common_name and issuer and common_name != issuer:
        signer: Optional[str] = f"{common_name} (Issuer: {issuer})"
    else:
        signer = common_name or issuer or None
    signed_time = sig_info.get("time")

    cg_parts: list[str] = []
    for cg in active.get("claim_generator_info") or []:
        if isinstance(cg, dict) and cg.get("name"):
            ver = f" {cg['version']}" if cg.get("version") else ""
            cg_parts.append(f"{cg['name']}{ver}".strip())
    if not cg_parts and active.get("claim_generator"):
        cg_parts.append(str(active["claim_generator"]))
    claim_generator: Optional[str] = ", ".join(dict.fromkeys(cg_parts)) or None

    digital_source_types: list[str] = []
    software_agents: list[str] = []
    ai_assertions: list[str] = []
    declares_ai = False

    # Inspect active manifest first, plus any ingredient manifests for composite lineage
    scan_manifests = [active] + [m for mid, m in manifests.items() if mid != active_id]
    for m_idx, manifest in enumerate(scan_manifests):
        prefix = "" if m_idx == 0 else "[ingredient] "
        for assertion in manifest.get("assertions") or []:
            if not isinstance(assertion, dict):
                continue
            alabel = str(assertion.get("label") or "")
            adata = assertion.get("data") or {}
            if alabel.startswith("c2pa.actions") and isinstance(adata, dict):
                for action in adata.get("actions") or []:
                    if not isinstance(action, dict):
                        continue
                    sa = _format_software_agent(action.get("softwareAgent"))
                    if sa and sa not in software_agents:
                        software_agents.append(sa)
                    dst = action.get("digitalSourceType")
                    if dst:
                        dst_str = str(dst)
                        short_dst = _short_source_type(dst_str)
                        if short_dst not in digital_source_types:
                            digital_source_types.append(short_dst)
                        if any(tok in short_dst.lower() for tok in AI_SOURCE_TOKENS):
                            declares_ai = True
                            note = f"{prefix}{short_dst}" + (f" · {sa}" if sa else "")
                            if note not in ai_assertions:
                                ai_assertions.append(note)

    if (
        software_agents
        and claim_generator
        and not any(a in claim_generator for a in software_agents)
    ):
        claim_generator = f"{claim_generator} / {', '.join(software_agents)}"
    elif software_agents and not claim_generator:
        claim_generator = ", ".join(software_agents)

    ai_verified = bool(signature_valid and declares_ai)
    if ai_verified:
        badge = AI_VERIFIED_BADGE
    elif declares_ai and not signature_valid:
        badge = "Content Credentials: AI claim (invalid signature / tampered)"
    elif signature_valid:
        badge = "Content Credentials: Present (signed)"
    else:
        badge = "Content Credentials: Present (invalid signature)"

    parts = [badge]
    if signer:
        parts.append(f"Signer: {signer}")
    if claim_generator:
        parts.append(f"Generator: {claim_generator}")
    if digital_source_types:
        parts.append(f"digitalSourceType: {', '.join(digital_source_types)}")
    parts.append(f"Validation: {validation_state or 'Unknown'}")
    if hard_failures:
        parts.append(f"Errors: {', '.join(hard_failures)}")

    return {
        "status": "present",
        "present": True,
        "signature_valid": signature_valid,
        "validation_state": validation_state,
        "validation_status": validation_status,
        "hard_failures": hard_failures,
        "signer": signer,
        "signed_time": signed_time,
        "claim_generator": claim_generator,
        "software_agents": software_agents,
        "digital_source_types": digital_source_types,
        "digital_source_type": digital_source_types[0]
        if digital_source_types
        else None,
        "ai_assertions": ai_assertions,
        "declares_ai_generated": declares_ai,
        "ai_verified_by_credentials": ai_verified,
        "badge": badge,
        "camera_exif": camera_exif,
        "summary": " · ".join(parts),
        "disclaimer": PROVENANCE_DISCLAIMER,
    }
