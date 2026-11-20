# VeriFake

VeriFake checks whether a face portrait is real or AI-generated using a ConvNeXt-Tiny classifier trained on real photos and diffusion and GAN generators.

## What it does

You upload a portrait in a browser or via `POST /predict` in JPEG, PNG, WEBP, BMP, TIFF, GIF, AVIF, or HEIC/HEIF up to 15 MB. First, a YuNet detector verifies that the image actually contains a human face so the classifier does not guess on landscapes or objects. For valid faces, the model scores three views of the image, applies validation-fitted temperature scaling to output calibrated probabilities and an uncertain band (`0.40` to `0.60`), and computes a Grad-CAM overlay highlighting which regions drove the score. Content Credentials (C2PA / JUMBF / XMP / EXIF) are inspected separately as supporting provenance evidence and never change the pixel prediction. Images are processed in memory and not stored.

![VeriFake analysis screen showing an AI-generated verdict and Grad-CAM heatmap](assets/ui_fake.png)

![VeriFake performance view with confusion matrix and ROC curve](assets/ui_performance.png)

## Results

All metrics below are measured on the untouched 5,586-image face-gated test split (`reports/metrics.json`). That split was never used for model selection, hyperparameter tuning, or calibration.

| Metric | Value | Notes |
| --- | ---: | --- |
| Accuracy (3-view TTA) | 96.26% | Single-view crop is 95.97% |
| Precision | 92.56% | 3,425 TN, 157 FP, 52 FN, 1,952 TP |
| Recall | 97.41% | 1,952 / 2,004 AI-generated faces caught |
| F1 score | 0.9492 | Up from 0.9453 single-view |
| ROC-AUC | 0.9947 | Held-out face-gated test split |
| Test ECE | 0.0148 | Down from 0.0436 via T = 0.6095 |
| Uncertain predictions | 29 / 5,586 | 96.58% accuracy on 5,557 decisive calls |
| Unseen Midjourney | 95.50% | n = 489 faces; AUC 0.9903 vs. test real |

Per-source accuracy across the test split and held-out unseen Midjourney set:

| Source | Class | Faces | Accuracy | Uncertain |
| --- | --- | ---: | ---: | ---: |
| SDXL | FAKE | 198 | 99.49% | 0 |
| DALLE3 | FAKE | 255 | 99.22% | 1 |
| FLUX | FAKE | 103 | 99.03% | 0 |
| CelebA_HQ | REAL | 598 | 98.83% | 1 |
| SD3 | FAKE | 206 | 98.06% | 1 |
| CelebA | REAL | 1,050 | 97.43% | 3 |
| SD21 | FAKE | 192 | 97.40% | 0 |
| StyleGAN | FAKE | 1,050 | 96.29% | 4 |
| FFHQ | REAL | 1,800 | 96.17% | 12 |
| Midjourney (unseen) | FAKE | 489 | 95.50% | 9 |
| COCO | REAL | 134 | 59.70% | 7 |

## How it works

I pooled images from five public datasets and removed 7 byte- and pixel-level duplicates (SHA-256 and decoded RGB MD5), leaving 32,331 unique images. Before any split, I held out every Midjourney image (`unseen_generator`, 2,710 total, 489 with detected faces) so I could measure performance on a generator family never seen in training or validation. The remaining images were split 70/30 stratified jointly by source generator and label (`18,660` train pool, `8,887` test), with 10% of the train pool (`2,074`) held out for validation (`src/data.py`). Keeping the 30% test set strictly untouched meant every design choice rested on validation scores rather than test leakage.

To pick the backbone, I trained `convnext_tiny.fb_in22k_ft_in1k` and `efficientnet_b3.ra2_in1k` for one epoch on the same subset (`reports/backbone_benchmark.json`). ConvNeXt-Tiny scored higher on validation accuracy (`73.0%` vs `71.5%`) and ROC-AUC (`0.7989` vs `0.7815`) while finishing faster, so I kept it and fine-tuned for 3 epochs on 3,600 generator-balanced face images (`1,000` validation images) to `96.10%` validation accuracy. Training included resize downsampling (`0.25x` to `1.0x`), JPEG compression (`q=30` to `95`), simulated social re-encoding, Gaussian blur, and additive noise so common web re-uploads would not easily fool the detector.

In earlier runs without a face gate, the classifier produced erratic probabilities on landscapes and indoor scenes. I added OpenCV's YuNet face detector (`models/face_detection_yunet_2023mar.onnx`, score threshold `0.55`) in `src/face_crop.py` to crop aligned faces with context padding and reject images with no face (`status: "no_face"`). During inference (`src/inference.py`), logits are averaged across three views: the face crop, a horizontal flip of the crop, and the full frame. That 3-view test-time augmentation added `+0.29%` test accuracy and `+0.0021` ROC-AUC over a single crop.

Raw logits from neural nets are often overconfident, so I fitted single-parameter temperature scaling (`T = 0.6095`) with LBFGS on the 1,000-image validation logits (`src/calibration.py`). Calibration reduced test expected calibration error from `0.0436` to `0.0148`, and calibrated `P(FAKE)` values between `0.40` and `0.60` return `UNCERTAIN`. Visual attribution uses Grad-CAM (`src/gradcam.py`) over ConvNeXt feature maps, and `src/provenance.py` inspects C2PA manifests via `c2pa-python` with a direct container-chunk parser fallback.

## Run it

Model weights (`106 MB`) are stored with Git LFS, so initialize Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/<your-username>/verifake.git
cd verifake
```

Create a virtual environment and install pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the FastAPI web app (opens http://localhost:8000 in your browser):

```bash
python -m app.main
```

To run headless on a specific port or send a single image from the terminal:

```bash
DEEPFAKE_NO_BROWSER=1 DEEPFAKE_PORT=8000 python -m app.main
curl -F "file=@path/to/portrait.jpg" http://127.0.0.1:8000/predict
```

The test suite and evaluation scripts expect the local dataset under `data/` (`data/multi_manifest.csv` and image files, excluded from git):

```bash
python -m unittest discover -s tests -v
python -m src.evaluate --checkpoint models/best.pt
```

## Known issues

- Unseen generator drop: When I ran 3-fold leave-one-generator-out benchmarks (`reports/logo_results.json`), holding out one diffusion model at a time along with Midjourney, mean detection dropped to `84.13%` (`87.84%` on DALLE3, `86.87%` on SDXL, and `77.67%` on FLUX; mean ROC-AUC `0.9285`). That is 12 to 21 percentage points lower than when those generators are included in training (`99.0%` to `99.5%`). On held-out Midjourney (`489` faces), `22` were misclassified as real and `9` fell in the uncertain band.
- Faces only: Any photo where YuNet finds no face is rejected out of scope (`3,301` of `8,887` test images and `2,221` of `2,710` Midjourney images).
- Weak on in-the-wild COCO scenes: `616` of `750` COCO test images had no detected face, and accuracy on the remaining `134` face crops is `59.70%`, well below `96.17%` on FFHQ and `98.83%` on CelebA-HQ.
- Compression and downscaling trade-offs: On the 600-image degradation test slice (`3`-view TTA), `0.25x` resizing drops accuracy to `89.17%`, Gaussian blur (`sigma = 1.5`) to `90.17%`, `0.5x` resizing to `90.83%`, and the social-media re-encoding chain to `92.17%` (versus `93.17%` on the unperturbed slice). Training with heavy degradations also nudged clean held-out test accuracy slightly down from `96.92%` (`reports/before_metrics.json`) to `96.26%`.
- Fragile metadata: Messaging platforms usually strip C2PA manifests on upload, and EXIF camera fields can be altered, so missing credentials do not imply an image is real.

## Project structure

```text
.
├── app/
│   ├── main.py              # FastAPI endpoints (/, /api/health, /api/metrics, /predict)
│   └── static/              # HTML/CSS/JS frontend and self-hosted fonts
├── assets/                  # UI screenshots used in README
├── models/
│   ├── best.pt              # Final ConvNeXt-Tiny checkpoint (Git LFS)
│   ├── calibration.json     # Validation-fitted temperature parameter
│   ├── class_names.json     # Explicit REAL/FAKE index mapping
│   └── face_detection_yunet_2023mar.onnx
├── reports/                 # Saved evaluation metrics and PNG plots
├── src/                     # Data pipeline, training, calibration, Grad-CAM, provenance
├── tests/                   # Unit and format regression tests
├── requirements.txt
└── LICENSE
```

## Data sources

- [TheKernel01/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) (`FFHQ` real and `StyleGAN` fake)
- [Rajarshi-Roy-research/Defactify_Image_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (`COCO` real and `SD21`, `SDXL`, `SD3`, `DALLE3`, `Midjourney` fake)
- [Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset](https://huggingface.co/datasets/Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset) (`FLUX`, `DALLE3`, `Midjourney` fake)
- [flwrlabs/celeba](https://huggingface.co/datasets/flwrlabs/celeba) (`CelebA` real portraits)
- [mattymchen/celeba-hq](https://huggingface.co/datasets/mattymchen/celeba-hq) (`CelebA-HQ` high-resolution real portraits)
