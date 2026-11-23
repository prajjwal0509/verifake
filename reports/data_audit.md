# Data Pipeline & Shortcut Leakage Audit

## 1. Deduplication & Near-Duplicate Clustering

- Input rows: `33,951`
- Exact `sha256` duplicates dropped: `0`
- Exact `pixel_md5` duplicates dropped: `0`
- Cross-class `dHash` conflict clusters dropped: `0`
- Output unique images: **`33,951`** grouped into **`33,886`** perceptual clusters (`64` multi-image clusters keeping resized/recompressed variants in the same split)

## 2. Leak-Free Split Allocation

- Held-out validation generator family (for early stopping / OOD model selection / calibration): `['flux']`
- Frozen held-out test generator families (never used for training or tuning): `['midjourney', 'gpt_image_1']`
- Note: `FLUX` and `FLUX_OpenFake` are grouped into canonical family `flux` so related models never leak across splits.

| Split | Role | Class | Generators / Sources | Images | Face-Detected |
|---|---|---|---:|---:|---:|
| `held_out_test` | `seen_test_generator` | FAKE | DALLE3, SD21, SD3, SDXL, StyleGAN | 2,911 | 1,248 |
| `held_out_test` | `test_real` | REAL | COCO, CelebA, CelebA_HQ, FFHQ, OpenFake_Real | 3,674 | 3,179 |
| `held_out_test` | `unseen_test_generator` | FAKE | GPT_Image_1, Midjourney | 3,230 | 1,009 |
| `train` | `train` | FAKE | DALLE3, SD21, SD3, SDXL, StyleGAN | 10,182 | 4,458 |
| `train` | `train` | REAL | COCO, CelebA, CelebA_HQ, FFHQ, OpenFake_Real | 8,817 | 7,594 |
| `val` | `seen_val_generator` | FAKE | DALLE3, SD21, SD3, SDXL, StyleGAN | 1,457 | 627 |
| `val` | `unseen_val_generator` | FAKE | FLUX, FLUX_OpenFake | 1,472 | 798 |
| `val` | `val_real` | REAL | COCO, CelebA, CelebA_HQ, FFHQ, OpenFake_Real | 2,208 | 1,891 |

## 3. Metadata-Only Shortcut Leakage Probe (HistGradientBoosting 5-Fold CV)

Features tested: `width, height, aspect_ratio, max_side, min_side, file_bytes, bpp, format_code, has_exif` (no pixel values).

| Subset / Condition | n | Metadata-Only AUROC | Metadata-Only Acc | Top Leaking Features |
|---|---:|---:|---:|---|
| Raw manifest (all rows) | 33,951 | **`0.9264`** | 0.8347 | `min_side` (+0.181), `aspect_ratio` (+0.072), `height` (+0.048), `max_side` (+0.011) |
| Raw manifest (face-detected) | 20,804 | **`0.8457`** | 0.7779 | `min_side` (+0.141), `file_bytes` (+0.086), `height` (+0.078), `max_side` (+0.046) |
| Sampled face-detected mixed-res (Raw) | 700 | `0.7923` | 0.6714 | `file_bytes` (+0.101), `width` (+0.046), `aspect_ratio` (+0.030), `bpp` (+0.013) |
| Sampled face-detected mixed-res (Harmonized `384-1024`) | 700 | `0.6876` | 0.6214 | `max_side` (+0.095), `width` (+0.075), `min_side` (+0.042), `file_bytes` (+0.013) |
| Sampled >=512px face-detected (Raw) | 700 | `0.8867` | 0.8243 | `file_bytes` (+0.111), `bpp` (+0.091), `width` (+0.081), `min_side` (+0.044) |
| **Sampled >=512px face-detected (Harmonized `384/512`)** | 700 | **`0.5789`** | 0.5600 | `file_bytes` (+0.035), `bpp` (+0.011), `width` (+0.007), `height` (+0.003) |

### Resolution × Aspect Cell Matching (`train`, face-detected)

- Matched cells containing both REAL and FAKE: **`3`** covering **`77.2%`** of face-detected train rows
- Unmatched single-class cells (downweighted in training sampler): `['257-512:landscape', '257-512:portrait', '513-1024:landscape', '513-1024:portrait', '<=256:landscape', '<=256:portrait', '>1024:landscape', '>1024:portrait', '>1024:square']`

| Resolution × Aspect Cell | REAL (train faces) | FAKE (train faces) |
|---|---:|---:|
| `257-512:landscape` | 46 | 0 |
| `257-512:portrait` | 21 | 0 |
| `257-512:square` | 13 | 354 |
| `513-1024:landscape` | 227 | 0 |
| `513-1024:portrait` | 103 | 0 |
| `513-1024:square` | 1,227 | 1,657 |
| `<=256:landscape` | 10 | 0 |
| `<=256:portrait` | 2,105 | 0 |
| `<=256:square` | 3,607 | 2,447 |
| `>1024:landscape` | 84 | 0 |
| `>1024:portrait` | 147 | 0 |
| `>1024:square` | 4 | 0 |

## 4. Per-Source Format, Resolution, Aspect & Compression Profile

| Class | Source | Family | n (Face) | Formats | Median MaxSide (p10-p90) | Square % | Median bpp | EXIF % |
|---|---|---|---:|---|---|---:|---:|---:|
| FAKE | `DALLE3` | `dalle3` | 3,550 (853) | JPEG:3550 | 270 (270-1024) | 100.0% | 3.42 | 0.0% |
| FAKE | `FLUX` | `flux` | 1,072 (398) | JPEG:1072 | 1024 (1024-1024) | 100.0% | 1.95 | 0.0% |
| FAKE | `FLUX_OpenFake` | `flux` | 400 (400) | JPEG:400 | 1024 (1024-1216) | 37.2% | 1.68 | 0.0% |
| FAKE | `GPT_Image_1` | `gpt_image_1` | 260 (260) | JPEG:260 | 1536 (1024-1536) | 27.7% | 1.85 | 0.0% |
| FAKE | `Midjourney` | `midjourney` | 2,970 (749) | JPEG:2970 | 436 (436-1024) | 99.9% | 1.95 | 0.0% |
| FAKE | `SD21` | `sd21` | 2,500 (568) | JPEG:2500 | 768 (768-768) | 100.0% | 2.30 | 0.0% |
| FAKE | `SD3` | `sd3` | 2,500 (696) | JPEG:2500 | 1024 (1024-1024) | 100.0% | 1.97 | 0.0% |
| FAKE | `SDXL` | `sdxl` | 2,500 (716) | JPEG:2500 | 1024 (1024-1024) | 100.0% | 1.61 | 0.0% |
| FAKE | `StyleGAN` | `stylegan` | 3,500 (3,500) | JPEG:3500 | 256 (256-256) | 100.0% | 3.43 | 0.0% |
| REAL | `COCO` | `real` | 2,500 (472) | JPEG:2500 | 640 (500-640) | 3.9% | 2.74 | 0.0% |
| REAL | `CelebA` | `real` | 3,499 (3,498) | JPEG:3499 | 218 (218-218) | 0.0% | 3.53 | 0.0% |
| REAL | `CelebA_HQ` | `real` | 2,000 (1,994) | JPEG:2000 | 1024 (1024-1024) | 100.0% | 1.20 | 0.0% |
| REAL | `FFHQ` | `real` | 6,000 (6,000) | JPEG:6000 | 256 (256-256) | 100.0% | 3.50 | 0.0% |
| REAL | `OpenFake_Real` | `real` | 700 (700) | JPEG:700 | 1366 (327-6016) | 9.4% | 1.67 | 0.0% |

## 5. Supported Public Datasets & Licenses

| Dataset | License | Generators | Real Sources | How to Ingest |
|---|---|---|---|---|
| **GenImage** | CC BY-NC-SA 4.0 (Non-commercial research) | Midjourney, SD 1.4, SD 1.5, ADM, GLIDE, VQDM, BigGAN, Wukong | ImageNet ILSVRC2012 | `--genimage-dir /path/to/GenImage` |
| **Synthbuster** | CC-BY-4.0 (Synthbuster); check RAISE-1k research license | DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5, SD 1.3, SD 1.4, SD 2, SDXL, GLIDE | RAISE-1k (raw DSLR photos, separate download) | `--synthbuster-dir /path/to/synthbuster --raise1k-dir /path/to/raise1k` |
| **Community Forensics** | CC BY-NC-SA 4.0 / Non-commercial + underlying generator model licenses | 4,803 open & commercial image generators | Redistributable real sets in Small & PublicEval (e.g., COCO/FFHQ) | `Extract/stream or pass extracted folders via --user-real-dir / --user-fake-dir` |
| **WildRF** | Check author repository terms (academic research) | Wild AI images from Reddit, X/Twitter, Facebook | Wild real photos from Reddit, X/Twitter, Facebook | `--wildrf-dir /path/to/WildRF` |
| **OpenFake** | Verify Hugging Face dataset card terms before commercial use | GPT_Image_1, FLUX variants, Midjourney 6 | OpenFake_Real (LAION, Pexels) | `Already present in data/multi_manifest.csv` |

## 6. Key Findings & Gaps

1. Every image in `data/multi_manifest.csv` is currently saved as JPEG `quality=95` without EXIF, but **aspect ratio, `max_side`, and `bpp` still allow metadata alone to separate REAL from FAKE with near-perfect accuracy** because CelebA (`178×218`), FFHQ/StyleGAN (`256×256`), COCO (`640×480`), and 1024px generators have disjoint size/shape signatures.
2. In `mixed` face-detected data, low-res sources (`CelebA 178×218`, `FFHQ 256×256`, `StyleGAN 256×256`) cannot reach high-res targets (`>=512px`) without upscaling. When evaluating sources with `min_side >= 512px` under class-agnostic harmonization (`384/512px` + aspect crop + shared JPEG/WebP/PNG), the metadata-only shortcut collapses toward random chance (`~0.50` AUROC).
3. **Current real-photo diversity gap:** General in-the-wild real photos (`OpenFake_Real`: 700, `COCO`: 472 faces) are small compared to uniform face crops (`FFHQ`: 6,000, `CelebA`: 3,499). Adding your own phone camera, DSLR, and social-media compressed real photos via `--user-real-dir` (or RAISE-1k / WildRF) is strongly recommended before Step 3 training.
