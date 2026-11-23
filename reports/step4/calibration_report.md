# Step 4 Calibrated Ensemble & Decision Logic Report

- Ensemble: `CLIP-L/14 (clip_mlp, w=0.65) + convnext_v2 (w=0.35)` (`fit_ood_weight=True`)
- Bounded calibration (fitted strictly on `val` clean + perturbed): **`T = 1.0000`** (`T >= 1.0` enforced), shift **`b = +0.8537`**
- Validation NLL: `0.1964 -> 0.1724` | Validation ECE: `0.0086 -> 0.0217`
- Decision threshold (clean `val_real` target FPR <= `4.0%`): **`0.4966`** (empirical `val_real` FPR `3.96%`)
- Uncertain band (+/- `0.55` log-odds): **`[0.3627, 0.6310]`** | View margin: `0.85` | Branch margin: `1.15`

## 1. Overall Held-Out Test Calibration & Selective Metrics (`5,435` face-detected images)

| Metric | Value |
|---|---:|
| AUROC | 0.9896 |
| Test NLL (before -> after) | 0.1640 -> 0.1440 |
| Test ECE (before -> after) | 0.0171 -> 0.0318 |
| Binary Recall (all samples @ 0.4966) | 92.6% |
| Binary Real FPR (all samples @ 0.4966) | 3.7% |
| Decided Recall (non-UNCERTAIN) | 96.1% |
| Decided Real FPR (non-UNCERTAIN) | 2.0% |
| UNCERTAIN Abstention Rate | 9.1% (493/5435) |
| Abstentions breakdown (view_disagreement / branch_disagreement / borderline) | 59 / 246 / 191 |
| Binary error rate on abstained samples | 30.0% |

## 2. Per-Source Breakdown on Held-Out Test

| Source | Class | Role | n | Metric | Binary Rate | Decided Rate | UNCERTAIN Rate | View/Branch Disagreements | AUROC vs Test Real |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `COCO` | REAL | `test_real` | 132 | FPR | 12.9% | 7.6% | 20.5% | 12 / 8 | - |
| `CelebA` | REAL | `test_real` | 874 | FPR | 0.5% | 0.1% | 2.2% | 0 / 16 | - |
| `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 1.0% | 0.4% | 2.4% | 0 / 6 | - |
| `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 100.0% | 100.0% | 1.3% | 2 / 0 | 0.9998 |
| `FFHQ` | REAL | `test_real` | 1500 | FPR | 2.2% | 0.9% | 5.3% | 0 / 34 | - |
| `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 95.0% | 98.7% | 10.8% | 15 / 8 | 0.9906 |
| `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 96.8% | 99.3% | 6.5% | 12 / 18 | 0.9953 |
| `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 33.7% | 31.0% | 33.7% | 14 / 17 | - |
| `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 97.5% | 99.1% | 6.7% | 2 / 5 | 0.9962 |
| `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 100.0% | 100.0% | 2.3% | 2 / 1 | 0.9977 |
| `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 98.6% | 100.0% | 2.0% | 0 / 1 | 0.9984 |
| `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 82.3% | 86.5% | 29.0% | 0 / 132 | 0.9765 |

## 3. Perturbation Robustness (`held_out_test`, `30`/source)

| Perturbation | n | AUROC | ECE (before -> after) | Binary Rec / FPR | Decided Rec / FPR | UNCERTAIN % | View/Branch Disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 359 | 0.9925 | 0.0433 -> 0.0404 | 97.1% / 6.7% | 98.5% / 4.4% | 7.8% | 7 / 11 |
| `jpeg_q50` | 344 | 0.9894 | 0.0460 -> 0.0547 | 96.5% / 4.8% | 97.3% / 4.1% | 10.8% | 8 / 11 |
| `webp_q80` | 352 | 0.9896 | 0.0410 -> 0.0495 | 96.6% / 10.2% | 97.9% / 3.8% | 8.2% | 7 / 8 |
| `downscale_512` | 331 | 0.9897 | 0.0416 -> 0.0345 | 97.3% / 8.2% | 98.8% / 6.9% | 9.4% | 7 / 10 |
| `downscale_256` | 274 | 0.9876 | 0.0368 -> 0.0314 | 94.4% / 6.9% | 99.2% / 5.0% | 10.9% | 0 / 14 |
| `downscale_512_jpeg_q75` | 330 | 0.9833 | 0.0359 -> 0.0369 | 97.3% / 10.3% | 97.1% / 7.3% | 9.7% | 9 / 11 |
