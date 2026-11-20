"""Tests for C2PA provenance, metadata override, and face crop diagnostics."""

from __future__ import annotations

import asyncio
import datetime
import io
import struct
import unittest

import c2pa
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID
from fastapi import UploadFile
from PIL import Image, ImageFilter
from starlette.datastructures import Headers

import app.main as app_main
from src.data import load_manifest
from src.face_crop import SMALL_FACE_MARGIN, SMALL_FACE_WIDTH_PCT
from src.inference import get_detector
from src.provenance import ABSENT_BADGE, AI_VERIFIED_BADGE, inspect_provenance
from src.utils import PROJECT_ROOT


def _build_test_c2pa_signer() -> tuple[ec.EllipticCurvePrivateKey, str]:
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "VeriFake Test Root CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VeriFake Test CA"),
        ]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Test AI Signer"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VeriFake Test Org"),
        ]
    )
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(2)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ObjectIdentifier("1.3.6.1.5.5.7.3.36")]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(
                ca_cert.extensions.get_extension_for_class(
                    x509.SubjectKeyIdentifier
                ).value
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    chain_pem = (
        leaf_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    ).decode()
    return leaf_key, chain_pem


def _sign_png_with_ai_c2pa(img: Image.Image) -> bytes:
    leaf_key, chain_pem = _build_test_c2pa_signer()

    def sign_cb(data: bytes) -> bytes:
        der = leaf_key.sign(data, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")

    src_buf = io.BytesIO()
    img.convert("RGB").save(src_buf, format="PNG")
    src_buf.seek(0)
    dst_buf = io.BytesIO()
    manifest_def = {
        "claim_generator_info": [{"name": "Test AI API", "version": "1.0"}],
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            "softwareAgent": {
                                "name": "TestGPT",
                                "version": "gpt-image",
                            },
                        }
                    ]
                },
            }
        ],
    }
    with c2pa.Signer.from_callback(
        sign_cb, c2pa.C2paSigningAlg.ES256, chain_pem
    ) as signer:
        with c2pa.Builder(manifest_def) as builder:
            builder.sign(signer, "image/png", src_buf, dst_buf)
    return dst_buf.getvalue()


def _make_upload(raw: bytes, filename: str, mime: str = "image/png") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(raw), filename=filename, headers=Headers({"content-type": mime})
    )


class TestProvenanceAndDiagnostics(unittest.TestCase):
    """Verify cryptographic C2PA provenance, combined verdict logic, small-face cropper & diagnostics."""

    real_face: Image.Image

    @classmethod
    def setUpClass(cls) -> None:
        cls.det = get_detector()
        app_main.state["detector"] = cls.det
        df = load_manifest()
        sample = df[
            (df["split"] == "test")
            & (df["generator"] == "CelebA_HQ")
            & (df["has_face"] == True)  # noqa: E712
        ].iloc[0]
        img = Image.open(PROJECT_ROOT / sample["path"]).convert("RGB")
        img.thumbnail((512, 512))
        cls.real_face = img

    def test_01_absent_credentials_never_marks_real_verified(self) -> None:
        """Unsigned real face must return absent credentials and 'Likely real, not verified'."""
        buf = io.BytesIO()
        self.real_face.save(buf, format="PNG")
        res = asyncio.run(
            app_main.predict(_make_upload(buf.getvalue(), "unsigned_real.png"))
        )

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["pixel_model"]["label"], "REAL")
        self.assertFalse(res["override_applied"])
        self.assertEqual(res["verdict_display"], "Likely real, not verified")
        self.assertEqual(res["provenance"]["status"], "absent")
        self.assertFalse(res["provenance"]["present"])
        self.assertEqual(res["provenance"]["badge"], ABSENT_BADGE)

    def test_02_valid_ai_c2pa_overrides_pixel_real_and_exposes_both_signals(
        self,
    ) -> None:
        """Valid C2PA trainedAlgorithmicMedia on a REAL face must override to FAKE (source: metadata)."""
        signed_png = _sign_png_with_ai_c2pa(self.real_face)
        prov = inspect_provenance(signed_png, detected_format="PNG")
        self.assertTrue(prov["present"])
        self.assertTrue(prov["signature_valid"])
        self.assertTrue(prov["declares_ai_generated"])
        self.assertTrue(prov["ai_verified_by_credentials"])
        self.assertEqual(prov["badge"], AI_VERIFIED_BADGE)
        self.assertIn("Test AI Signer", prov["signer"] or "")
        self.assertIn("TestGPT", prov["claim_generator"] or "")
        self.assertIn("trainedAlgorithmicMedia", prov["digital_source_types"])

        res = asyncio.run(
            app_main.predict(_make_upload(signed_png, "signed_override.png"))
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["label"], "FAKE")
        self.assertTrue(res["override_applied"])
        self.assertEqual(res["verdict_source"], "metadata")
        self.assertEqual(res["verdict_display"], "FAKE (source: metadata)")
        # Separate signals stay intact
        self.assertEqual(res["pixel_model"]["label"], "REAL")
        self.assertEqual(res["provenance"]["badge"], AI_VERIFIED_BADGE)

        # Also works even when no face is present in a signed AI graphic
        non_face_signed = _sign_png_with_ai_c2pa(
            Image.new("RGB", (160, 160), (40, 80, 120))
        )
        res_nf = asyncio.run(
            app_main.predict(_make_upload(non_face_signed, "signed_noface.png"))
        )
        self.assertEqual(res_nf["status"], "ok")
        self.assertEqual(res_nf["label"], "FAKE")
        self.assertEqual(res_nf["verdict_display"], "FAKE (source: metadata)")
        self.assertEqual(res_nf["pixel_model"]["status"], "no_face")

    def test_03_tampered_c2pa_fails_validation_and_does_not_override(self) -> None:
        """Modifying pixels after C2PA signing must fail assertion.dataHash validation and skip override."""
        signed_png = _sign_png_with_ai_c2pa(self.real_face)
        # Extract valid PNG caBX chunk from the signed file
        pos = 8
        cabx_chunk = b""
        while pos + 8 <= len(signed_png):
            length, ctype = struct.unpack(">I4s", signed_png[pos : pos + 8])
            end = pos + 8 + length + 4
            if ctype == b"caBX":
                cabx_chunk = signed_png[pos:end]
                break
            pos = end
        self.assertTrue(cabx_chunk, "Expected caBX chunk in signed PNG")

        # Tamper pixel values (darken by 25) and re-encode a clean PNG with the original caBX chunk transplanted
        tampered_arr = np.clip(
            np.asarray(self.real_face, dtype=np.int16) - 25, 0, 255
        ).astype(np.uint8)
        tbuf = io.BytesIO()
        Image.fromarray(tampered_arr).save(tbuf, format="PNG")
        t_raw = tbuf.getvalue()
        tampered_png = t_raw[:-12] + cabx_chunk + t_raw[-12:]

        res = asyncio.run(
            app_main.predict(_make_upload(tampered_png, "tampered_pixels.png"))
        )

        self.assertTrue(res["provenance"]["present"])
        self.assertFalse(res["provenance"]["signature_valid"])
        self.assertEqual(res["provenance"]["validation_state"], "Invalid")
        self.assertFalse(res["override_applied"])
        self.assertEqual(res["verdict_source"], "pixel_model")

    def test_04_small_face_diagnostics_tight_crop_upsample_and_per_view_probs(
        self,
    ) -> None:
        """Faces < 15% of image width must trigger tight 0.20 crop, >=256px upsample, and warning."""
        bg = (
            self.real_face.resize((10, 10), Image.Resampling.BILINEAR)
            .resize((900, 900), Image.Resampling.BILINEAR)
            .filter(ImageFilter.GaussianBlur(10))
        )
        # Paste a crisp 96x96 portrait (~10.7% of 900px width -> < 15%, raw tight crop < 256px so Lanczos upsamples)
        small_face = self.real_face.resize((96, 96), Image.Resampling.LANCZOS)
        bg.paste(small_face, (400, 320))

        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        res = asyncio.run(
            app_main.predict(_make_upload(buf.getvalue(), "small_face.png"))
        )

        diag = res["diagnostics"]
        self.assertTrue(diag["face_detected"])
        self.assertLess(diag["face_width_pct"], SMALL_FACE_WIDTH_PCT)
        self.assertTrue(diag["small_face"])
        self.assertEqual(diag["warning"], "Small face: low confidence")
        self.assertEqual(diag["crop_margin"], SMALL_FACE_MARGIN)
        self.assertTrue(diag["upsampled"])
        self.assertGreaterEqual(min(diag["crop_resolution"]), 256)

        pv = diag["per_view_probabilities"]
        self.assertEqual(set(pv.keys()), {"crop", "flipped", "full_frame"})
        for view_name, stats in pv.items():
            self.assertAlmostEqual(
                stats["REAL"] + stats["FAKE"], 1.0, places=4, msg=view_name
            )


if __name__ == "__main__":
    unittest.main()
