"""Tests for multi-format image decoding, EXIF orientation, and upload guards."""

from __future__ import annotations

import asyncio
import io
import struct
import unittest
import warnings
import zlib

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

import app.main as app_main
from src.data import load_manifest
from src.image_io import (
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    decode_image_bytes,
    register_image_plugins,
)
from src.inference import get_detector
from src.utils import PROJECT_ROOT


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _make_ihdr_only_png(width: int, height: int) -> bytes:
    """Create a tiny valid-header PNG claiming ``width x height`` to test decompression-bomb guards."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _make_upload(raw: bytes, filename: str, content_type: str | None) -> UploadFile:
    headers = Headers({} if content_type is None else {"content-type": content_type})
    return UploadFile(file=io.BytesIO(raw), filename=filename, headers=headers)


class TestMultiFormatSupport(unittest.TestCase):
    """Verify decoding and /predict across all supported image containers, modes, and safety bounds."""

    face_pil: Image.Image

    @classmethod
    def setUpClass(cls) -> None:
        register_image_plugins()
        app_main.state["detector"] = get_detector()

        df = load_manifest()
        sample = df[
            (df["split"] == "test")
            & (df["generator"] == "CelebA_HQ")
            & (df["has_face"] == True)  # noqa: E712
        ].iloc[0]
        img = Image.open(PROJECT_ROOT / sample["path"]).convert("RGB")
        # Resize moderately for fast encoding of HEIC/AVIF/TIFF fixtures in tests while keeping crisp face.
        img.thumbnail((384, 384))
        cls.face_pil = img

    def _assert_predict_ok(
        self, raw: bytes, filename: str, mime: str | None, expected_format: str
    ) -> dict:
        res = asyncio.run(app_main.predict(_make_upload(raw, filename, mime)))
        self.assertEqual(res.get("status"), "ok", f"Failed on {filename}: {res}")
        self.assertIn(res.get("label"), ("REAL", "FAKE", "UNCERTAIN"))
        self.assertIn(res.get("binary_label"), ("REAL", "FAKE"))
        self.assertEqual(res.get("detected_format"), expected_format)
        self.assertTrue(
            str(res.get("preview", "")).startswith("data:image/jpeg;base64,")
        )
        self.assertTrue(
            str(res.get("gradcam", "")).startswith("data:image/png;base64,")
        )
        return res

    def test_01_all_supported_formats_return_prediction(self) -> None:
        """JPG, PNG, WEBP, BMP, TIFF, GIF (frame 0), AVIF, and HEIC must all decode and predict cleanly."""
        w, h = self.face_pil.size
        format_cases: list[tuple[str, str, str, str, dict]] = [
            ("JPEG", "sample.jpg", "image/jpeg", "JPEG", {"quality": 92}),
            ("PNG", "sample.png", "image/png", "PNG", {}),
            ("WEBP", "sample.webp", "image/webp", "WEBP", {"quality": 90}),
            ("BMP", "sample.bmp", "image/bmp", "BMP", {}),
            ("TIFF", "sample.tiff", "image/tiff", "TIFF", {}),
            ("AVIF", "sample.avif", "image/avif", "AVIF", {"quality": 85}),
            ("HEIF", "sample.heic", "image/heic", "HEIF", {"quality": 88}),
        ]
        for save_fmt, fname, mime, expected_fmt, kwargs in format_cases:
            buf = io.BytesIO()
            self.face_pil.save(buf, format=save_fmt, **kwargs)
            res = self._assert_predict_ok(buf.getvalue(), fname, mime, expected_fmt)
            self.assertEqual(res["image_size"], [w, h], f"Size mismatch for {save_fmt}")

        # Animated 2-frame GIF: frame 0 has the face, frame 1 is solid black -> must use frame 0.
        gif_buf = io.BytesIO()
        black = Image.new("RGB", (w, h), (0, 0, 0))
        self.face_pil.save(
            gif_buf,
            format="GIF",
            save_all=True,
            append_images=[black],
            duration=100,
            loop=0,
        )
        gif_res = self._assert_predict_ok(
            gif_buf.getvalue(), "sample.gif", "image/gif", "GIF"
        )
        self.assertEqual(gif_res["image_size"], [w, h])

    def test_02_rotated_heic_and_jpeg_exif_transpose(self) -> None:
        """Rotated HEIC and JPEG (EXIF Orientation=6) must decode upright without double-rotation."""
        # Non-square crop so width != height proves orientation is restored upright.
        upright = self.face_pil.crop(
            (0, 20, self.face_pil.width, self.face_pil.height - 20)
        )
        uw, uh = upright.size
        self.assertNotEqual(uw, uh)

        stored_rotated = upright.transpose(
            Image.Transpose.ROTATE_90
        )  # needs 90 CW (Orientation=6) to appear upright
        exif = stored_rotated.getexif()
        exif[274] = 6  # Orientation = Rotate 90 CW

        heic_buf = io.BytesIO()
        stored_rotated.save(heic_buf, format="HEIF", quality=90, exif=exif.tobytes())
        heic_bytes = heic_buf.getvalue()
        dec_heic, fmt_heic = decode_image_bytes(heic_bytes)
        self.assertEqual(fmt_heic, "HEIF")
        self.assertEqual(dec_heic.size, (uw, uh))
        res_heic = self._assert_predict_ok(
            heic_bytes, "IMG_1001.HEIC", "image/heic", "HEIF"
        )
        self.assertEqual(res_heic["image_size"], [uw, uh])

        jpg_buf = io.BytesIO()
        stored_rotated.save(jpg_buf, format="JPEG", quality=92, exif=exif.tobytes())
        dec_jpg, fmt_jpg = decode_image_bytes(jpg_buf.getvalue())
        self.assertEqual(fmt_jpg, "JPEG")
        self.assertEqual(dec_jpg.size, (uw, uh))

    def test_03_content_detection_ignores_wrong_or_empty_mime_and_extension(
        self,
    ) -> None:
        """Browser sending HEIC as .bin with empty or application/octet-stream MIME must succeed."""
        buf = io.BytesIO()
        self.face_pil.save(buf, format="HEIF", quality=88)
        heic_bytes = buf.getvalue()

        for fake_mime in ("", "application/octet-stream", None):
            res = self._assert_predict_ok(
                heic_bytes, "iphone_upload.bin", fake_mime, "HEIF"
            )
            self.assertEqual(res["image_size"], list(self.face_pil.size))

    def test_04_edge_color_modes_rgba_palette_cmyk_and_16bit(self) -> None:
        """RGBA, Palette+transparency, CMYK, and 16-bit grayscale images must convert cleanly to RGB."""
        w, h = self.face_pil.size

        # RGBA PNG with alpha cutout
        rgba = self.face_pil.convert("RGBA")
        arr_rgba = np.array(rgba)
        arr_rgba[:12, :12, 3] = 0
        rgba_buf = io.BytesIO()
        Image.fromarray(arr_rgba, mode="RGBA").save(rgba_buf, format="PNG")
        out_rgba, _ = decode_image_bytes(rgba_buf.getvalue())
        self.assertEqual(out_rgba.mode, "RGB")
        # Transparent corner composited onto white, not black
        self.assertEqual(out_rgba.getpixel((0, 0)), (255, 255, 255))

        # Palette PNG with transparency
        p_img = Image.fromarray(arr_rgba, mode="RGBA").convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128
        )
        p_buf = io.BytesIO()
        p_img.save(p_buf, format="PNG")
        out_p, _ = decode_image_bytes(p_buf.getvalue())
        self.assertEqual(out_p.mode, "RGB")
        self.assertEqual(out_p.size, (w, h))

        # CMYK TIFF
        cmyk_buf = io.BytesIO()
        self.face_pil.convert("CMYK").save(cmyk_buf, format="TIFF")
        out_cmyk, fmt_cmyk = decode_image_bytes(cmyk_buf.getvalue())
        self.assertEqual(fmt_cmyk, "TIFF")
        self.assertEqual(out_cmyk.mode, "RGB")

        # 16-bit grayscale PNG: verify proportional scaling (NOT clipped to pure white 255)
        grad16 = np.tile(np.linspace(0, 65535, 64, dtype=np.uint16), (64, 1))
        buf16 = io.BytesIO()
        Image.fromarray(grad16).save(buf16, format="PNG")
        out16, _ = decode_image_bytes(buf16.getvalue())
        self.assertEqual(out16.mode, "RGB")
        px_arr = np.asarray(out16)
        self.assertLessEqual(int(px_arr[:, 0, 0].max()), 2)
        self.assertTrue(
            115 <= int(px_arr[0, 32, 0]) <= 140,
            f"Unexpected 16-bit midtone: {px_arr[0, 32, 0]}",
        )
        self.assertGreaterEqual(int(px_arr[:, -1, 0].min()), 253)

    def test_05_zero_recompression_preserves_lossless_pixels(self) -> None:
        """Decoding a PNG upload must pass exact pixels without intermediate JPEG re-compression."""
        raw_arr = np.asarray(self.face_pil)
        buf = io.BytesIO()
        self.face_pil.save(buf, format="PNG")
        decoded, fmt = decode_image_bytes(buf.getvalue())
        self.assertEqual(fmt, "PNG")
        self.assertTrue(np.array_equal(np.asarray(decoded), raw_arr))

    def test_06_rejects_corrupt_small_oversized_and_decompression_bombs(self) -> None:
        """Invalid images, tiny files, >15MB payloads, and huge pixel dimensions must fail cleanly."""
        # Corrupt / non-image disguised as image/jpeg
        with self.assertRaises(HTTPException) as ctx_bad:
            asyncio.run(
                app_main.predict(
                    _make_upload(b"definitely not an image", "fake.jpg", "image/jpeg")
                )
            )
        self.assertEqual(ctx_bad.exception.status_code, 400)
        self.assertIn("could not be decoded", ctx_bad.exception.detail)

        # Truncated JPEG
        jbuf = io.BytesIO()
        self.face_pil.save(jbuf, format="JPEG", quality=90)
        trunc = jbuf.getvalue()[: len(jbuf.getvalue()) // 2]
        with self.assertRaises(HTTPException) as ctx_trunc:
            asyncio.run(app_main.predict(_make_upload(trunc, "cut.jpg", "image/jpeg")))
        self.assertEqual(ctx_trunc.exception.status_code, 400)

        # Too small (< 32px)
        small_buf = io.BytesIO()
        Image.new("RGB", (16, 16), (120, 120, 120)).save(small_buf, format="PNG")
        with self.assertRaises(ImageValidationError) as ctx_small:
            decode_image_bytes(small_buf.getvalue())
        self.assertEqual(ctx_small.exception.status_code, 400)

        # Oversized (> 15 MB)
        with self.assertRaises(HTTPException) as ctx_big:
            asyncio.run(
                app_main.predict(
                    _make_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "huge.png", "image/png")
                )
            )
        self.assertEqual(ctx_big.exception.status_code, 413)

        # Decompression bomb checks: side > MAX_IMAGE_SIDE and total pixels > MAX_IMAGE_PIXELS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with self.assertRaises(ImageValidationError) as ctx_side:
                decode_image_bytes(_make_ihdr_only_png(MAX_IMAGE_SIDE + 100, 64))
            self.assertEqual(ctx_side.exception.status_code, 400)

            with self.assertRaises(ImageValidationError) as ctx_mp:
                decode_image_bytes(_make_ihdr_only_png(9000, 9000))  # 81 MP > 64 MP
            self.assertEqual(ctx_mp.exception.status_code, 400)

            with self.assertRaises(ImageValidationError) as ctx_bomb:
                decode_image_bytes(
                    _make_ihdr_only_png(20000, 20000)
                )  # > 2*MAX triggers Pillow DecompressionBombError
            self.assertEqual(ctx_bomb.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
