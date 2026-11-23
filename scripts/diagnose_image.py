"""Diagnose a single image step-by-step through VeriFake's detection pipeline."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from src.calibration import (  # noqa: E402
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    apply_temperature,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.face_crop import crop_face_or_full, gate_and_crop_face  # noqa: E402
from src.image_io import decode_image_bytes, register_image_plugins  # noqa: E402
from src.inference import DeepfakeDetector, logits_to_probabilities  # noqa: E402
from src.metadata import (  # noqa: E402
    _extract_isobmff_metadata,
    _extract_jpeg_metadata,
    _extract_png_metadata,
    _extract_webp_metadata,
    inspect_content_credentials,
)
from src.provenance import (  # noqa: E402
    _guess_mime,
    _read_c2pa_store,
    inspect_provenance,
)
from src.utils import CHECKPOINT_PATH  # noqa: E402

MARKER_PATTERNS: tuple[str, ...] = (
    "google",
    "gemini",
    "imagen",
    "synthid",
    "made with google",
    "digitalsourcetype",
    "trainedalgorithmicmedia",
    "iptc",
    "xmp",
    "credit",
)


def _header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")


def _extract_raw_chunks(raw: bytes) -> tuple[str, list[bytes]]:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        chunks, _ = _extract_png_metadata(raw)
        return "PNG", chunks
    if raw[:2] == b"\xff\xd8":
        chunks, _ = _extract_jpeg_metadata(raw)
        return "JPEG", chunks
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        chunks, _ = _extract_webp_metadata(raw)
        return "WEBP", chunks
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        chunks, _ = _extract_isobmff_metadata(raw)
        return "ISOBMFF", chunks
    return "OTHER", []


def _sanitize_snippet(raw_slice: bytes, max_len: int = 140) -> str:
    text = "".join(chr(b) if 32 <= b < 127 else " " for b in raw_slice)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _scan_marker_snippets(
    chunks: list[bytes], raw_info_items: dict[str, str]
) -> dict[str, list[str]]:
    sources: list[bytes] = list(chunks)
    for k, v in raw_info_items.items():
        sources.append(f"{k}: {v}".encode("utf-8", errors="ignore"))

    hits: dict[str, list[str]] = {}
    for marker in MARKER_PATTERNS:
        needle = marker.encode("ascii")
        snippets: list[str] = []
        seen: set[str] = set()
        for src in sources:
            low = src.lower()
            start = 0
            while len(snippets) < 3:
                idx = low.find(needle, start)
                if idx == -1:
                    break
                lo = max(0, idx - 45)
                hi = min(len(src), idx + len(needle) + 85)
                snip = _sanitize_snippet(src[lo:hi], max_len=140)
                if snip and snip not in seen:
                    seen.add(snip)
                    snippets.append(snip)
                start = idx + len(needle)
            if len(snippets) >= 3:
                break
        if snippets:
            hits[marker] = snippets
    return hits


def _inspect_raw_pillow_header(raw_bytes: bytes) -> dict[str, Any]:
    register_image_plugins()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        exif = img.getexif()
        orientation = exif.get(274) if exif else None
        exif_dict = (
            {str(k): str(v)[:120] for k, v in exif.items()} if exif else {}
        )
        info_summary: dict[str, str] = {}
        for k, v in (img.info or {}).items():
            if k in ("icc_profile", "exif"):
                info_summary[str(k)] = f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else str(v)[:120]
            elif isinstance(v, bytes):
                info_summary[str(k)] = _sanitize_snippet(v, max_len=160)
            else:
                info_summary[str(k)] = re.sub(r"\s+", " ", str(v)).strip()[:160]
        return {
            "raw_format": img.format,
            "raw_mode": img.mode,
            "raw_size": list(img.size),
            "exif_orientation": orientation,
            "has_icc_profile": "icc_profile" in (img.info or {}),
            "is_animated": bool(getattr(img, "is_animated", False)),
            "n_frames": int(getattr(img, "n_frames", 1)),
            "exif_top_tags": exif_dict,
            "info_keys": list((img.info or {}).keys()),
            "info_summary": info_summary,
        }


def _score_logit_row(
    row: torch.Tensor,
    idx_to_class: dict[int, str],
    real_idx: int,
    fake_idx: int,
    temperature: float,
) -> dict[str, Any]:
    row_2d = row.view(1, -1).float().cpu()
    logit_real = float(row_2d[0, real_idx].item())
    logit_fake = float(row_2d[0, fake_idx].item())
    delta = logit_fake - logit_real

    _, _, p_uncal, _ = logits_to_probabilities(row_2d, idx_to_class)
    scaled = apply_temperature(row_2d, temperature)
    binary_label, confidence, p_cal, _ = logits_to_probabilities(scaled, idx_to_class)
    p_cal_fake = float(p_cal["FAKE"])
    uncertain = UNCERTAIN_LOW <= p_cal_fake <= UNCERTAIN_HIGH
    return {
        "logit_real": logit_real,
        "logit_fake": logit_fake,
        "delta": delta,
        "delta_scaled": delta / max(float(temperature), 1e-4),
        "p_uncal_real": float(p_uncal["REAL"]),
        "p_uncal_fake": float(p_uncal["FAKE"]),
        "p_cal_real": float(p_cal["REAL"]),
        "p_cal_fake": p_cal_fake,
        "binary_label": binary_label,
        "display_label": "UNCERTAIN" if uncertain else binary_label,
        "uncertain": uncertain,
        "confidence": float(confidence),
    }


def _run_tta_on_pil(
    detector: DeepfakeDetector,
    rgb: Image.Image,
    cropped: Image.Image,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    real_idx = detector.class_to_idx["REAL"]
    fake_idx = detector.class_to_idx["FAKE"]
    batch = torch.stack(
        [
            detector.transform(cropped),
            detector.transform(ImageOps.mirror(cropped)),
            detector.transform(rgb),
        ],
        dim=0,
    ).to(detector.device)
    with torch.no_grad():
        view_logits = detector.model(batch)
    mean_logits = view_logits.mean(dim=0, keepdim=True)

    names = ("face_crop", "flipped_face_crop", "full_frame")
    per_view = {
        name: _score_logit_row(
            view_logits[i],
            detector.idx_to_class,
            real_idx,
            fake_idx,
            detector.temperature,
        )
        for i, name in enumerate(names)
    }
    tta_mean = _score_logit_row(
        mean_logits[0],
        detector.idx_to_class,
        real_idx,
        fake_idx,
        detector.temperature,
    )
    return per_view, tta_mean


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img.copy()
    scale = max_side / float(max(w, h))
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _reencode_image(img: Image.Image, fmt: str, quality: int) -> Image.Image:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=quality)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    buf.seek(0)
    with Image.open(buf) as reopened:
        return reopened.convert("RGB")


def _extract_native_patches(
    img: Image.Image, patch_size: int
) -> list[tuple[str, tuple[int, int, int, int], Image.Image]]:
    w, h = img.size
    if w < patch_size or h < patch_size:
        return []
    cx = (w - patch_size) // 2
    cy = (h - patch_size) // 2
    candidates = [
        ("center", (cx, cy, cx + patch_size, cy + patch_size)),
        ("top_left", (0, 0, patch_size, patch_size)),
        ("top_right", (w - patch_size, 0, w, patch_size)),
        ("bottom_left", (0, h - patch_size, patch_size, h)),
        ("bottom_right", (w - patch_size, h - patch_size, w, h)),
    ]
    patches: list[tuple[str, tuple[int, int, int, int], Image.Image]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for name, box in candidates:
        if box not in seen_boxes:
            seen_boxes.add(box)
            patches.append((name, box, img.crop(box)))
    return patches


def _classify_confidence_finding(
    p_uncal_fake: float,
    p_cal_fake: float,
    binary_label: str,
    expected: str,
    temperature: float,
) -> dict[str, Any]:
    uncal_in_band = UNCERTAIN_LOW <= p_uncal_fake <= UNCERTAIN_HIGH
    cal_in_band = UNCERTAIN_LOW <= p_cal_fake <= UNCERTAIN_HIGH
    dist_50 = abs(p_cal_fake - 0.50)

    if cal_in_band:
        bucket = "BORDERLINE (within uncertain band [0.40, 0.60])"
    elif (0.30 <= p_cal_fake < UNCERTAIN_LOW) or (UNCERTAIN_HIGH < p_cal_fake <= 0.70):
        status = "RIGHT" if binary_label == expected else "WRONG"
        bucket = (
            f"NEAR-BORDERLINE (just outside [0.40, 0.60], {status} vs expected={expected})"
        )
    else:
        if binary_label == expected:
            bucket = f"CONFIDENTLY CORRECT (matches expected={expected})"
        else:
            bucket = f"CONFIDENTLY WRONG (predicted {binary_label}, expected={expected})"

    crossed_note = "no boundary crossing"
    if uncal_in_band and not cal_in_band:
        crossed_note = (
            f"YES: temperature scaling (T={temperature:.4f} < 1.0) pushed P(FAKE) "
            f"out of [0.40, 0.60] from {p_uncal_fake:.4f} -> {p_cal_fake:.4f}"
        )
    elif not uncal_in_band and cal_in_band:
        crossed_note = (
            f"YES: temperature scaling moved P(FAKE) into [0.40, 0.60] "
            f"from {p_uncal_fake:.4f} -> {p_cal_fake:.4f}"
        )

    return {
        "expected": expected,
        "p_uncal_fake": p_uncal_fake,
        "p_cal_fake": p_cal_fake,
        "distance_from_0_50": dist_50,
        "temperature": temperature,
        "uncal_in_uncertain_band": uncal_in_band,
        "cal_in_uncertain_band": cal_in_band,
        "crossed_uncertain_boundary": crossed_note,
        "classification": bucket,
    }


def diagnose_one(
    path: Path,
    detector: DeepfakeDetector,
    *,
    expected: str,
    no_face_gate: bool,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    raw_bytes = path.read_bytes()
    header_info = _inspect_raw_pillow_header(raw_bytes)
    rgb, detected_fmt = decode_image_bytes(raw_bytes)

    _header(f"DIAGNOSIS REPORT: {path.name}")
    print(f"Path: {path.resolve()}")

    # Stage 1: Input & Decode
    print("\n[Stage 1] File & Decode")
    print(
        f"  File size       : {len(raw_bytes):,} bytes ({len(raw_bytes) / 1024:.1f} KB)"
    )
    print(
        f"  Raw header      : format={header_info['raw_format']} mode={header_info['raw_mode']} "
        f"size={header_info['raw_size'][0]}x{header_info['raw_size'][1]} "
        f"orientation={header_info['exif_orientation']} icc={header_info['has_icc_profile']} "
        f"animated={header_info['is_animated']} frames={header_info['n_frames']}"
    )
    print(
        f"  Decoded (app)   : format={detected_fmt} upright_size={rgb.width}x{rgb.height} mode={rgb.mode}"
    )

    # Stage 2: Provenance, C2PA & Container Metadata
    print("\n[Stage 2] Provenance, C2PA & Container Metadata")
    provenance = inspect_provenance(raw_bytes, detected_format=detected_fmt)
    mime = _guess_mime(raw_bytes, detected_format=detected_fmt, suffix=path.suffix)
    raw_c2pa_detail: dict[str, Any] = {}
    try:
        store = _read_c2pa_store(raw_bytes, mime)
    except Exception as exc:
        store = None
        raw_c2pa_detail["read_error"] = str(exc)

    if store:
        active_id = store.get("active_manifest")
        active_res = (store.get("validation_results") or {}).get("activeManifest") or {}
        raw_c2pa_detail = {
            "active_manifest": active_id,
            "success_codes": [
                s.get("code") for s in (active_res.get("success") or []) if isinstance(s, dict)
            ],
            "failure_codes": [
                f.get("code") for f in (active_res.get("failure") or []) if isinstance(f, dict)
            ],
        }

    meta_scan = inspect_content_credentials(raw_bytes)
    container_kind, raw_chunks = _extract_raw_chunks(raw_bytes)
    marker_hits = _scan_marker_snippets(raw_chunks, header_info["info_summary"])

    print(
        f"  inspect_provenance : status={provenance['status']} present={provenance['present']} "
        f"sig_valid={provenance['signature_valid']} val_state={provenance['validation_state']}"
    )
    print(
        f"  C2PA claims        : declares_ai={provenance['declares_ai_generated']} "
        f"override={provenance['ai_verified_by_credentials']} "
        f"claim_gen={provenance['claim_generator']} "
        f"dst={provenance['digital_source_types']}"
    )
    print(f"  C2PA hard_failures : {provenance.get('hard_failures', [])}")
    if raw_c2pa_detail:
        print(f"  Raw C2PA store     : {raw_c2pa_detail}")
    print(f"  Camera EXIF        : {provenance.get('camera_exif') or {}}")
    print(
        f"  inspect_content_creds (unwired): status={meta_scan['status']} "
        f"has_c2pa={meta_scan['has_c2pa']} dst={meta_scan['digital_source_type']} "
        f"claims={meta_scan['generator_claims']}"
    )
    print(
        f"  Container chunks   : kind={container_kind} num_chunks={len(raw_chunks)} "
        f"pillow_info_keys={header_info['info_keys']}"
    )
    if marker_hits:
        print("  Marker hits        :")
        for marker, snips in marker_hits.items():
            for snip in snips:
                print(f"    - [{marker}] {snip}")
    else:
        print("  Marker hits        : none (no google/gemini/imagen/synthid/xmp/iptc strings in chunks)")

    # Stage 3: Face Gate & Crop
    print("\n[Stage 3] Face Gate & Crop")
    gated_crop, gate_meta = gate_and_crop_face(rgb)
    face_detected = bool(gate_meta["face_detected"])
    used_fallback = False
    if gated_crop is not None:
        cropped_for_model = gated_crop
    else:
        used_fallback = True
        cropped_for_model, _ = crop_face_or_full(rgb)

    img_size = detector.img_size
    crop_w, crop_h = cropped_for_model.size
    print(
        f"  face_detected      : {face_detected} (mode={gate_meta['mode']}, score={gate_meta['face_score']})"
    )
    print(
        f"  raw_face_bbox      : {gate_meta['raw_face_bbox']}  expanded_bbox={gate_meta['bbox']}"
    )
    print(
        f"  face size %        : width={gate_meta['face_width_pct']}% height={gate_meta['face_height_pct']}% "
        f"small_face={gate_meta['small_face']} warning={gate_meta['warning']}"
    )
    print(
        f"  crop resolution    : raw={gate_meta['raw_crop_resolution']} final={gate_meta['crop_resolution']} "
        f"margin={gate_meta['crop_margin']} upsampled={gate_meta['upsampled']}"
    )
    print(
        f"  resize into {img_size}x{img_size}: crop {crop_w}x{crop_h} -> {img_size}x{img_size} "
        f"({crop_w / img_size:.2f}x W, {crop_h / img_size:.2f}x H) | "
        f"full_frame {rgb.width}x{rgb.height} -> {img_size}x{img_size} "
        f"({rgb.width / img_size:.2f}x W, {rgb.height / img_size:.2f}x H)"
    )
    if used_fallback:
        note = "require_face=False requested" if no_face_gate else "diagnostic full-frame fallback (prod returns no_face)"
        print(f"  NOTE               : {note}")

    # Stage 4: Raw Logits & Probabilities
    print("\n[Stage 4] Stage-by-Stage Raw Logits & Probabilities")
    print(
        f"  Mapping & T        : class_to_idx={detector.class_to_idx} | T={detector.temperature:.6f} "
        f"| uncertain_band=[{UNCERTAIN_LOW:.2f}, {UNCERTAIN_HIGH:.2f}] | device={detector.device}"
    )
    per_view, tta_mean = _run_tta_on_pil(detector, rgb, cropped_for_model)

    hdr = (
        f"  {'View':<18} {'L[REAL]':>9} {'L[FAKE]':>9} {'Delta':>9} "
        f"{'P_uncal(F)':>11} {'Delta/T':>9} {'P_cal(F)':>10} {'Label':<10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ("face_crop", "flipped_face_crop", "full_frame"):
        r = per_view[name]
        print(
            f"  {name:<18} {r['logit_real']:>9.4f} {r['logit_fake']:>9.4f} {r['delta']:>+9.4f} "
            f"{r['p_uncal_fake']:>11.4f} {r['delta_scaled']:>+9.4f} {r['p_cal_fake']:>10.4f} {r['display_label']:<10}"
        )
    print("  " + "-" * (len(hdr) - 2))
    print(
        f"  {'TTA_mean (3-view)':<18} {tta_mean['logit_real']:>9.4f} {tta_mean['logit_fake']:>9.4f} "
        f"{tta_mean['delta']:>+9.4f} {tta_mean['p_uncal_fake']:>11.4f} {tta_mean['delta_scaled']:>+9.4f} "
        f"{tta_mean['p_cal_fake']:>10.4f} {tta_mean['display_label']:<10}"
    )

    prod = detector.predict(
        raw_bytes,
        require_face=not no_face_gate,
        raw_bytes=raw_bytes,
        use_tta=True,
        detected_format=detected_fmt,
    )
    prod_pixel = prod.get("pixel_model") or {}
    parity_diff: Optional[float] = None
    if face_detected or no_face_gate:
        prod_p_fake = float(prod_pixel["probabilities"]["FAKE"])
        parity_diff = abs(prod_p_fake - tta_mean["p_cal_fake"])
        assert parity_diff < 1e-5, (
            f"Diagnostic vs production P(FAKE) mismatch: {tta_mean['p_cal_fake']} vs {prod_p_fake}"
        )
    print(
        f"  Production check   : status={prod.get('status')} verdict_display={prod.get('verdict_display')!r} "
        f"source={prod.get('verdict_source')} override={prod.get('override_applied')} "
        + (f"(pixel parity diff={parity_diff:.2e})" if parity_diff is not None else "(prod gated no_face)")
    )

    # Stage 5: Borderline vs. Confidently Wrong
    print("\n[Stage 5] Borderline vs. Confidently Wrong Assessment (Pixel Model)")
    conf_assess = _classify_confidence_finding(
        tta_mean["p_uncal_fake"],
        tta_mean["p_cal_fake"],
        tta_mean["binary_label"],
        expected,
        detector.temperature,
    )
    print(f"  Expected true label: {expected}")
    print(
        f"  P_uncal(FAKE)      : {conf_assess['p_uncal_fake']:.4f} (T=1.0)"
    )
    print(
        f"  P_cal(FAKE)        : {conf_assess['p_cal_fake']:.4f} (T={detector.temperature:.4f}, "
        f"|P_cal - 0.50| = {conf_assess['distance_from_0_50']:.4f})"
    )
    print(
        f"  Calibration shift  : {conf_assess['crossed_uncertain_boundary']}"
    )
    print(f"  Assessment         : {conf_assess['classification']}")

    # Stage 6: Resolution & Recompression Perturbation Test
    print("\n[Stage 6] Original vs. Resized & JPEG-Recompressed Perturbation Test")
    variants: list[tuple[str, Optional[Image.Image], str]] = [
        ("original", rgb, ""),
        ("jpeg_q95", _reencode_image(rgb, "JPEG", 95), ""),
        ("jpeg_q85", _reencode_image(rgb, "JPEG", 85), ""),
        ("jpeg_q70", _reencode_image(rgb, "JPEG", 70), ""),
        ("jpeg_q50", _reencode_image(rgb, "JPEG", 50), ""),
        ("webp_q80", _reencode_image(rgb, "WEBP", 80), ""),
        (
            "resize_1024_lossless",
            _resize_max_side(rgb, 1024) if max(rgb.size) > 1024 else None,
            "max_side <= 1024",
        ),
        (
            "resize_512_lossless",
            _resize_max_side(rgb, 512) if max(rgb.size) > 512 else None,
            "max_side <= 512",
        ),
        (
            "resize_256_lossless",
            _resize_max_side(rgb, 256) if max(rgb.size) > 256 else None,
            "max_side <= 256",
        ),
        (
            "resize_1024_jpeg_q75",
            _reencode_image(_resize_max_side(rgb, 1024), "JPEG", 75),
            "",
        ),
        (
            "resize_512_jpeg_q80",
            _reencode_image(_resize_max_side(rgb, 512), "JPEG", 80),
            "",
        ),
    ]

    phdr = (
        f"  {'Variant':<22} {'WxH':<11} {'Face':<5} {'Delta':>8} "
        f"{'P_unc(F)':>9} {'P_cal(F)':>9} {'Binary':<7} {'Display':<10} {'Changed?'}"
    )
    print(phdr)
    print("  " + "-" * (len(phdr) - 2))

    perturbation_rows: list[dict[str, Any]] = []
    orig_bin: Optional[str] = None
    orig_disp: Optional[str] = None

    for vname, vimg, skip_reason in variants:
        if vimg is None:
            print(f"  {vname:<22} {'-':<11} {'-':<5} {'-':>8} {'-':>9} {'-':>9} {'SKIP':<7} {'-':<10} ({skip_reason})")
            perturbation_rows.append({"variant": vname, "skipped": True, "reason": skip_reason})
            continue

        if no_face_gate:
            vcrop, vmeta = crop_face_or_full(vimg)
        else:
            vcrop, vmeta = gate_and_crop_face(vimg)

        vface = bool(vmeta.get("face_detected", False))
        dims = f"{vimg.width}x{vimg.height}"
        if vcrop is None:
            changed = "NO_FACE" if (orig_bin is not None) else "-"
            print(f"  {vname:<22} {dims:<11} {'No':<5} {'-':>8} {'-':>9} {'-':>9} {'NO_FACE':<7} {'no_face':<10} {changed}")
            perturbation_rows.append(
                {"variant": vname, "size": list(vimg.size), "face_detected": False, "changed": changed}
            )
            continue

        _, vtta = _run_tta_on_pil(detector, vimg, vcrop)
        if vname == "original":
            orig_bin = vtta["binary_label"]
            orig_disp = vtta["display_label"]
            changed = "baseline"
        else:
            flags: list[str] = []
            if orig_bin is not None and vtta["binary_label"] != orig_bin:
                flags.append(f"BINARY({orig_bin}->{vtta['binary_label']})")
            if orig_disp is not None and vtta["display_label"] != orig_disp:
                flags.append(f"VERDICT({orig_disp}->{vtta['display_label']})")
            changed = ", ".join(flags) if flags else "no"

        print(
            f"  {vname:<22} {dims:<11} {('Yes' if vface else 'No'):<5} {vtta['delta']:>+8.4f} "
            f"{vtta['p_uncal_fake']:>9.4f} {vtta['p_cal_fake']:>9.4f} "
            f"{vtta['binary_label']:<7} {vtta['display_label']:<10} {changed}"
        )
        perturbation_rows.append(
            {
                "variant": vname,
                "size": list(vimg.size),
                "face_detected": vface,
                "delta": vtta["delta"],
                "p_uncal_fake": vtta["p_uncal_fake"],
                "p_cal_fake": vtta["p_cal_fake"],
                "binary_label": vtta["binary_label"],
                "display_label": vtta["display_label"],
                "changed": changed,
            }
        )

    scored_ps = [r["p_cal_fake"] for r in perturbation_rows if "p_cal_fake" in r]
    bin_flips = [r["variant"] for r in perturbation_rows if r.get("changed", "").startswith("BINARY") or "BINARY(" in r.get("changed", "")]
    disp_flips = [r["variant"] for r in perturbation_rows if "VERDICT(" in r.get("changed", "") or r.get("changed") == "NO_FACE"]
    if scored_ps:
        swing = max(scored_ps) - min(scored_ps)
        print(
            f"  Perturbation summary: min P_cal(F)={min(scored_ps):.4f}, max P_cal(F)={max(scored_ps):.4f} "
            f"(swing={swing:.4f}) | binary_flips={bin_flips or 'none'} | verdict_changes={disp_flips or 'none'}"
        )

    # Stage 7: Native-Resolution 224x224 Patch Probe
    print(f"\n[Stage 7] Native-Resolution {img_size}x{img_size} Patch Probe (Diagnostic Only — Model Trained on Resized Crops)")
    patches = _extract_native_patches(cropped_for_model, img_size)
    patch_rows: list[dict[str, Any]] = []
    if not patches:
        print(
            f"  Skipped: crop size {cropped_for_model.width}x{cropped_for_model.height} < {img_size}x{img_size} "
            "(cannot extract unscaled native patches)."
        )
    else:
        mean = tuple(detector.meta.get("normalize_mean", IMAGENET_MEAN))
        std = tuple(detector.meta.get("normalize_std", IMAGENET_STD))
        patch_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        patch_batch = torch.stack([patch_tf(pimg) for _, _, pimg in patches], dim=0).to(
            detector.device
        )
        with torch.no_grad():
            patch_logits = detector.model(patch_batch)
        real_idx = detector.class_to_idx["REAL"]
        fake_idx = detector.class_to_idx["FAKE"]
        for i, (pname, box, _) in enumerate(patches):
            ps = _score_logit_row(
                patch_logits[i],
                detector.idx_to_class,
                real_idx,
                fake_idx,
                detector.temperature,
            )
            ps["patch"] = pname
            ps["box"] = list(box)
            patch_rows.append(ps)
            print(
                f"  {pname:<14} box={str(list(box)):<20} Delta={ps['delta']:>+8.4f} "
                f"P_uncal(F)={ps['p_uncal_fake']:.4f} P_cal(F)={ps['p_cal_fake']:.4f} [{ps['display_label']}]"
            )
        mean_patch_logit = patch_logits.mean(dim=0, keepdim=True)
        patch_mean = _score_logit_row(
            mean_patch_logit[0],
            detector.idx_to_class,
            real_idx,
            fake_idx,
            detector.temperature,
        )
        print(
            f"  {'patch_mean':<14} {'n=' + str(len(patches)):<20} Delta={patch_mean['delta']:>+8.4f} "
            f"P_uncal(F)={patch_mean['p_uncal_fake']:.4f} P_cal(F)={patch_mean['p_cal_fake']:.4f} [{patch_mean['display_label']}]"
        )

    return {
        "path": str(path),
        "stage1_decode": {**header_info, "decoded_format": detected_fmt, "decoded_size": [rgb.width, rgb.height]},
        "stage2_provenance": provenance,
        "stage2_raw_c2pa": raw_c2pa_detail,
        "stage2_content_credentials": meta_scan,
        "stage2_marker_hits": marker_hits,
        "stage3_face_gate": {**gate_meta, "used_fallback": used_fallback},
        "stage4_views": per_view,
        "stage4_tta_mean": tta_mean,
        "stage4_production": {
            "status": prod.get("status"),
            "verdict_display": prod.get("verdict_display"),
            "verdict_source": prod.get("verdict_source"),
            "override_applied": prod.get("override_applied"),
            "pixel_parity_diff": parity_diff,
        },
        "stage5_assessment": conf_assess,
        "stage6_perturbations": perturbation_rows,
        "stage7_native_patches": patch_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run step-by-step VeriFake pipeline diagnostics on one or more images."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Image file path(s) to diagnose.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Model checkpoint path (default: models/best.pt).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device to run inference on (default: auto).",
    )
    parser.add_argument(
        "--expected",
        choices=["FAKE", "REAL"],
        default="FAKE",
        help="Expected ground-truth class for misclassification assessment (default: FAKE).",
    )
    parser.add_argument(
        "--no-face-gate",
        action="store_true",
        help="Use full-frame fallback when YuNet detects no face (still reports YuNet detection).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full JSON diagnostics.",
    )
    args = parser.parse_args()

    device = None if args.device == "auto" else torch.device(args.device)
    detector = DeepfakeDetector(checkpoint=args.checkpoint, device=device)

    results = [
        diagnose_one(
            img_path,
            detector,
            expected=args.expected,
            no_face_gate=args.no_face_gate,
        )
        for img_path in args.images
    ]
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved JSON diagnostics to {args.json_out}")


if __name__ == "__main__":
    main()

