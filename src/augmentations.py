"""Aspect-preserving preprocessing and robustness augmentations for generalization training."""

from __future__ import annotations

import io
import random
from typing import Callable, Optional

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.data import IMAGENET_MEAN, IMAGENET_STD

CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

STEP3_CACHE_PERTURBATIONS: tuple[str, ...] = (
    "original",
    "jpeg_q50",
    "webp_q80",
    "downscale_512",
    "downscale_256",
    "downscale_512_jpeg_q75",
)


def resize_pad_square(
    img: Image.Image,
    size: int = 224,
    *,
    fill: tuple[int, int, int] = (128, 128, 128),
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    """Resize preserving aspect ratio so max(w, h) == size, then pad centered to square."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w == size and h == size:
        return rgb
    scale = size / float(max(1, w, h))
    nw = max(1, min(size, int(round(w * scale))))
    nh = max(1, min(size, int(round(h * scale))))
    resized = rgb.resize((nw, nh), resample)
    if nw == size and nh == size:
        return resized
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def random_native_or_resized_crop(
    img: Image.Image,
    size: int = 224,
    *,
    native_prob: float = 0.35,
    scale: tuple[float, float] = (0.65, 1.0),
    ratio: tuple[float, float] = (0.85, 1.18),
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Sample either an unscaled native size×size crop (when large enough) or an aspect-safe crop."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if min(w, h) >= max(size + 16, 240) and random.random() < native_prob:
        x0 = random.randint(0, w - size)
        y0 = random.randint(0, h - size)
        return rgb.crop((x0, y0, x0 + size, y0 + size))

    area = float(w * h)
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(*ratio)
        cw = int(round((target_area * aspect) ** 0.5))
        ch = int(round((target_area / aspect) ** 0.5))
        if 32 <= cw <= w and 32 <= ch <= h:
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            return rgb.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), resample)

    # Fallback: center square crop if <= aspect bounds, else aspect-safe padded square
    side = min(w, h)
    if max(w, h) <= int(1.35 * side) and side >= 48:
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return rgb.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), resample)
    return resize_pad_square(rgb, size=size, resample=resample)


class RobustForensicTransform:
    """Degradation & multi-codec augmentation pipeline resisting resolution/format shortcuts."""

    def __init__(
        self,
        size: int = 224,
        *,
        native_prob: float = 0.35,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        jitter: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        self.size = size
        self.native_prob = native_prob
        self.resample = resample
        self.jitter = jitter or T.ColorJitter(0.15, 0.15, 0.15, 0.02)

    def _reencode(self, img: Image.Image, *, forced_codec: Optional[str] = None, q_range: tuple[int, int] = (40, 95)) -> Image.Image:
        r = random.random()
        codec = forced_codec or ("JPEG" if r < 0.50 else ("WEBP" if r < 0.75 else "PNG"))
        if codec == "PNG":
            return img
        buf = io.BytesIO()
        if codec == "JPEG":
            img.save(buf, format="JPEG", quality=random.randint(*q_range))
        else:
            q_lo = max(50, q_range[0])
            img.save(buf, format="WEBP", quality=random.randint(q_lo, q_range[1]))
        buf.seek(0)
        with Image.open(buf) as reopened:
            return reopened.convert("RGB")

    def __call__(self, img: Image.Image) -> Image.Image:
        work = img.convert("RGB")
        if random.random() < 0.50:
            work = work.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.40:
            work = self.jitter(work)

        # Random downscale across 256..1024 without upscaling tiny sources
        w, h = work.size
        max_wh = max(w, h)
        if max_wh > 256 and random.random() < 0.50:
            target_max = random.randint(256, min(1024, max_wh))
            if target_max < max_wh:
                scale = target_max / float(max_wh)
                nw = max(48, int(round(w * scale)))
                nh = max(48, int(round(h * scale)))
                interp = random.choice(
                    [
                        Image.Resampling.LANCZOS,
                        Image.Resampling.BILINEAR,
                        Image.Resampling.BICUBIC,
                        Image.Resampling.BOX,
                    ]
                )
                work = work.resize((nw, nh), interp)

        # Mixed social-media chain (downscale + lossy codec) or codec + metadata strip
        if random.random() < 0.35:
            w, h = work.size
            max_wh = max(w, h)
            if max_wh > 256:
                target_max = random.randint(256, min(768, max_wh))
                if target_max < max_wh:
                    scale = target_max / float(max_wh)
                    nw = max(48, int(round(w * scale)))
                    nh = max(48, int(round(h * scale)))
                    work = work.resize((nw, nh), Image.Resampling.BILINEAR)
            codec = "JPEG" if random.random() < 0.75 else "WEBP"
            work = self._reencode(work, forced_codec=codec, q_range=(40, 80))
        else:
            work = self._reencode(work, q_range=(40, 95))

        if random.random() < 0.30:
            work = work.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 2.0)))

        return random_native_or_resized_crop(
            work,
            size=self.size,
            native_prob=self.native_prob,
            resample=self.resample,
        )


class TensorGaussianNoise:
    """Add mild Gaussian sensor noise directly in [0, 1] tensor space."""

    def __init__(self, std: tuple[float, float] = (0.004, 0.035), p: float = 0.30) -> None:
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        sigma = random.uniform(*self.std)
        return (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)


def build_clip_transforms(train: bool, img_size: int = 224) -> Callable:
    """Build aspect-preserving eval or forensic-augmented CLIP image transforms."""
    normalize = T.Normalize(CLIP_MEAN, CLIP_STD)
    if not train:
        return T.Compose(
            [
                T.Lambda(
                    lambda im: resize_pad_square(
                        im, size=img_size, resample=Image.Resampling.BICUBIC
                    )
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            RobustForensicTransform(
                size=img_size, native_prob=0.35, resample=Image.Resampling.BICUBIC
            ),
            T.ToTensor(),
            TensorGaussianNoise(std=(0.004, 0.035), p=0.30),
            normalize,
        ]
    )


def build_convnext_transforms(
    train: bool,
    img_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> Callable:
    """Build aspect-preserving eval or forensic-augmented ConvNeXt image transforms."""
    normalize = T.Normalize(mean, std)
    if not train:
        return T.Compose(
            [
                T.Lambda(
                    lambda im: resize_pad_square(
                        im, size=img_size, resample=Image.Resampling.BILINEAR
                    )
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            RobustForensicTransform(
                size=img_size, native_prob=0.35, resample=Image.Resampling.BILINEAR
            ),
            T.ToTensor(),
            TensorGaussianNoise(std=(0.004, 0.035), p=0.30),
            normalize,
        ]
    )


def apply_eval_perturbation(img: Image.Image, name: str) -> Image.Image:
    """Apply deterministic robustness perturbation for Step 3 validation/test benchmarks."""
    rgb = img.convert("RGB")
    if name == "original":
        return rgb
    if name in ("jpeg_q50", "jpeg_q70", "jpeg_q85", "jpeg_q95"):
        q = int(name.split("_q")[1])
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        with Image.open(buf) as reopened:
            return reopened.convert("RGB")
    if name == "webp_q80":
        buf = io.BytesIO()
        rgb.save(buf, format="WEBP", quality=80)
        buf.seek(0)
        with Image.open(buf) as reopened:
            return reopened.convert("RGB")
    if name in ("downscale_1024", "downscale_512", "downscale_256"):
        target = int(name.split("_")[1])
        w, h = rgb.size
        if max(w, h) <= target:
            return rgb
        scale = target / float(max(w, h))
        return rgb.resize(
            (max(32, int(round(w * scale))), max(32, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    if name == "downscale_512_jpeg_q75":
        small = apply_eval_perturbation(rgb, "downscale_512")
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        with Image.open(buf) as reopened:
            return reopened.convert("RGB")
    raise ValueError(f"Unknown eval perturbation: {name}")
