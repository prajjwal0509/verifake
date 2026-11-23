# Step 3 Held-Out Test Report (Read-Only)

Best CLIP branch from `val`: **`clip_mlp`** | Ensemble clip weight: **`0.45`** | Unseen test generators: `Midjourney`, `GPT_Image_1`

## 1. Overall Held-Out Test Comparison

| Model | Val<=5% Thresh | Overall AUROC | Overall Recall @<=5% | Unseen Test AUROC | Unseen Recall @<=5% | Test Real FPR @<=5% | Recall @ 0.50 | Real FPR @ 0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_best_pt` | 0.5600 | 0.9875 | 95.3% | 0.9818 | 93.0% | 6.1% | 95.7% | 6.3% |
| `clip_linear_v2` | 0.5222 | 0.9619 | 83.9% | 0.9326 | 74.2% | 5.3% | 84.7% | 5.7% |
| `clip_mlp_v2` | 0.1482 | 0.9786 | 90.5% | 0.9587 | 81.8% | 5.0% | 85.1% | 2.0% |
| `convnext_v2` | 0.1370 | 0.9549 | 79.7% | 0.9426 | 77.8% | 5.5% | 64.6% | 1.5% |
| `ensemble_v2` | 0.0490 | 0.9858 | 94.1% | 0.9728 | 88.3% | 6.1% | 81.6% | 0.7% |

## 2. Per-Source Breakdown (`@ Val<=5% FPR`)

| Model | Source | Class | Role | n | Metric | Rate @ Val<=5% | Rate @ 0.50 | AUROC vs Test Real |
|---|---|---|---|---:|---|---:|---:|---:|
| `baseline_best_pt` | `COCO` | REAL | `test_real` | 132 | FPR | 24.2% | 24.2% | - |
| `baseline_best_pt` | `CelebA` | REAL | `test_real` | 874 | FPR | 1.7% | 1.8% | - |
| `baseline_best_pt` | `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 1.4% | 1.4% | - |
| `baseline_best_pt` | `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 99.3% | 99.3% | 0.9957 |
| `baseline_best_pt` | `FFHQ` | REAL | `test_real` | 1500 | FPR | 3.2% | 3.4% | - |
| `baseline_best_pt` | `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 90.7% | 91.1% | 0.9812 |
| `baseline_best_pt` | `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 93.7% | 94.3% | 0.9821 |
| `baseline_best_pt` | `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 52.6% | 54.3% | - |
| `baseline_best_pt` | `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 97.5% | 98.3% | 0.9897 |
| `baseline_best_pt` | `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 96.9% | 97.7% | 0.9901 |
| `baseline_best_pt` | `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 99.3% | 100.0% | 0.9925 |
| `baseline_best_pt` | `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 96.3% | 96.4% | 0.9921 |
| `clip_linear_v2` | `COCO` | REAL | `test_real` | 132 | FPR | 12.1% | 12.1% | - |
| `clip_linear_v2` | `CelebA` | REAL | `test_real` | 874 | FPR | 0.8% | 1.0% | - |
| `clip_linear_v2` | `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 1.6% | 1.6% | - |
| `clip_linear_v2` | `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 98.7% | 98.7% | 0.9963 |
| `clip_linear_v2` | `FFHQ` | REAL | `test_real` | 1500 | FPR | 8.1% | 9.0% | - |
| `clip_linear_v2` | `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 34.0% | 37.5% | 0.8083 |
| `clip_linear_v2` | `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 88.1% | 88.5% | 0.9756 |
| `clip_linear_v2` | `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 8.0% | 8.0% | - |
| `clip_linear_v2` | `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 92.4% | 92.4% | 0.9876 |
| `clip_linear_v2` | `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 80.8% | 83.1% | 0.9714 |
| `clip_linear_v2` | `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 96.6% | 96.6% | 0.9939 |
| `clip_linear_v2` | `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 91.2% | 91.4% | 0.9839 |
| `clip_mlp_v2` | `COCO` | REAL | `test_real` | 132 | FPR | 16.7% | 9.1% | - |
| `clip_mlp_v2` | `CelebA` | REAL | `test_real` | 874 | FPR | 0.5% | 0.0% | - |
| `clip_mlp_v2` | `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 1.6% | 0.8% | - |
| `clip_mlp_v2` | `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 100.0% | 99.3% | 0.9996 |
| `clip_mlp_v2` | `FFHQ` | REAL | `test_real` | 1500 | FPR | 6.4% | 2.5% | - |
| `clip_mlp_v2` | `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 47.1% | 33.2% | 0.8738 |
| `clip_mlp_v2` | `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 93.9% | 88.1% | 0.9880 |
| `clip_mlp_v2` | `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 17.1% | 6.9% | - |
| `clip_mlp_v2` | `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 98.3% | 96.6% | 0.9958 |
| `clip_mlp_v2` | `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 94.6% | 91.5% | 0.9917 |
| `clip_mlp_v2` | `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 99.3% | 97.3% | 0.9975 |
| `clip_mlp_v2` | `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 96.9% | 92.2% | 0.9933 |
| `convnext_v2` | `COCO` | REAL | `test_real` | 132 | FPR | 20.5% | 5.3% | - |
| `convnext_v2` | `CelebA` | REAL | `test_real` | 874 | FPR | 0.3% | 0.0% | - |
| `convnext_v2` | `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 2.2% | 0.6% | - |
| `convnext_v2` | `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 100.0% | 99.3% | 0.9993 |
| `convnext_v2` | `FFHQ` | REAL | `test_real` | 1500 | FPR | 7.3% | 2.1% | - |
| `convnext_v2` | `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 50.6% | 29.0% | 0.8536 |
| `convnext_v2` | `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 87.2% | 73.4% | 0.9734 |
| `convnext_v2` | `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 14.9% | 4.0% | - |
| `convnext_v2` | `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 97.5% | 96.6% | 0.9961 |
| `convnext_v2` | `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 100.0% | 98.5% | 0.9990 |
| `convnext_v2` | `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 100.0% | 100.0% | 0.9995 |
| `convnext_v2` | `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 67.0% | 41.8% | 0.9383 |
| `ensemble_v2` | `COCO` | REAL | `test_real` | 132 | FPR | 25.0% | 5.3% | - |
| `ensemble_v2` | `CelebA` | REAL | `test_real` | 874 | FPR | 0.0% | 0.0% | - |
| `ensemble_v2` | `CelebA_HQ` | REAL | `test_real` | 498 | FPR | 1.8% | 0.0% | - |
| `ensemble_v2` | `DALLE3` | FAKE | `seen_test_generator` | 151 | Recall | 100.0% | 100.0% | 1.0000 |
| `ensemble_v2` | `FFHQ` | REAL | `test_real` | 1500 | FPR | 7.9% | 0.7% | - |
| `ensemble_v2` | `GPT_Image_1` | FAKE | `unseen_test_generator` | 259 | Recall | 60.6% | 27.8% | 0.9087 |
| `ensemble_v2` | `Midjourney` | FAKE | `unseen_test_generator` | 749 | Recall | 97.9% | 88.7% | 0.9949 |
| `ensemble_v2` | `OpenFake_Real` | REAL | `test_real` | 175 | FPR | 18.9% | 2.9% | - |
| `ensemble_v2` | `SD21` | FAKE | `seen_test_generator` | 119 | Recall | 100.0% | 96.6% | 0.9992 |
| `ensemble_v2` | `SD3` | FAKE | `seen_test_generator` | 130 | Recall | 100.0% | 99.2% | 0.9996 |
| `ensemble_v2` | `SDXL` | FAKE | `seen_test_generator` | 147 | Recall | 100.0% | 99.3% | 0.9999 |
| `ensemble_v2` | `StyleGAN` | FAKE | `seen_test_generator` | 701 | Recall | 98.0% | 80.3% | 0.9935 |

## 3. Perturbation Robustness (`@ Val<=5% FPR`)

| Perturbation | Model | n (Face) | AUROC | Recall @<=5% | Real FPR @<=5% |
|---|---|---:|---:|---:|---:|
| `original` | `baseline_best_pt` | 359 | 0.9734 | 95.2% | 14.0% |
| `original` | `clip_linear_v2` | 359 | 0.9598 | 85.2% | 4.0% |
| `original` | `clip_mlp_v2` | 359 | 0.9726 | 91.9% | 8.0% |
| `original` | `convnext_v2` | 359 | 0.9557 | 84.7% | 6.0% |
| `original` | `ensemble_v2` | 359 | 0.9824 | 94.7% | 9.3% |
| `jpeg_q50` | `baseline_best_pt` | 344 | 0.9714 | 93.0% | 11.7% |
| `jpeg_q50` | `clip_linear_v2` | 344 | 0.9541 | 84.4% | 5.5% |
| `jpeg_q50` | `clip_mlp_v2` | 344 | 0.9635 | 93.5% | 11.7% |
| `jpeg_q50` | `convnext_v2` | 344 | 0.9553 | 85.9% | 4.8% |
| `jpeg_q50` | `ensemble_v2` | 344 | 0.9806 | 94.0% | 12.4% |
| `webp_q80` | `baseline_best_pt` | 352 | 0.9701 | 94.1% | 15.0% |
| `webp_q80` | `clip_linear_v2` | 352 | 0.9517 | 82.9% | 4.8% |
| `webp_q80` | `clip_mlp_v2` | 352 | 0.9649 | 91.2% | 8.8% |
| `webp_q80` | `convnext_v2` | 352 | 0.9549 | 85.9% | 6.8% |
| `webp_q80` | `ensemble_v2` | 352 | 0.9788 | 94.6% | 10.9% |
| `downscale_512` | `baseline_best_pt` | 331 | 0.9692 | 93.5% | 16.4% |
| `downscale_512` | `clip_linear_v2` | 331 | 0.9476 | 82.7% | 6.2% |
| `downscale_512` | `clip_mlp_v2` | 331 | 0.9585 | 91.4% | 8.2% |
| `downscale_512` | `convnext_v2` | 331 | 0.9523 | 82.2% | 4.1% |
| `downscale_512` | `ensemble_v2` | 331 | 0.9756 | 92.4% | 7.5% |
| `downscale_256` | `baseline_best_pt` | 274 | 0.9642 | 94.4% | 23.8% |
| `downscale_256` | `clip_linear_v2` | 274 | 0.9325 | 79.2% | 3.8% |
| `downscale_256` | `clip_mlp_v2` | 274 | 0.9569 | 86.8% | 3.8% |
| `downscale_256` | `convnext_v2` | 274 | 0.9388 | 72.2% | 0.8% |
| `downscale_256` | `ensemble_v2` | 274 | 0.9691 | 86.8% | 2.3% |
| `downscale_512_jpeg_q75` | `baseline_best_pt` | 330 | 0.9670 | 92.4% | 16.6% |
| `downscale_512_jpeg_q75` | `clip_linear_v2` | 330 | 0.9389 | 83.2% | 8.3% |
| `downscale_512_jpeg_q75` | `clip_mlp_v2` | 330 | 0.9522 | 90.8% | 10.3% |
| `downscale_512_jpeg_q75` | `convnext_v2` | 330 | 0.9527 | 83.8% | 9.0% |
| `downscale_512_jpeg_q75` | `ensemble_v2` | 330 | 0.9726 | 94.1% | 12.4% |
