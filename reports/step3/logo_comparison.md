# Step 3 Leave-One-Generator-Out (LOGO) Comparison

- Recommended branch by mean LOGO generalization (clean + perturbed `jpeg_q50` / `downscale_256`): **`ensemble`**
- Strict isolation: `held_out_test` (`Midjourney`, `GPT_Image_1`, `test_real`, `seen_test`) was NOT touched.

## Mean Across All 6 Generator Families

| Branch | Clean AUROC | Clean Rec @<=5% | Real FPR @<=5% | Pert AUROC | Pert Rec @<=5% | Rec @ 0.50 | FPR @ 0.50 | LOGO Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_linear | 0.8989 | 59.3% | 4.9% | 0.9147 | 60.3% | 61.2% | 6.1% | 0.7522 |
| clip_mlp | 0.9279 | 68.0% | 3.8% | 0.9354 | 69.7% | 66.6% | 3.7% | 0.8099 |
| convnext_tiny | 0.8677 | 62.2% | 4.7% | 0.8768 | 64.3% | 63.1% | 4.1% | 0.7524 |
| ensemble | 0.9186 | 71.5% | 4.1% | 0.9252 | 73.1% | 66.6% | 2.1% | 0.8223 |

## Per-Fold Breakdown

| Held-Out Generator | Branch | n | Clean AUROC | Clean Rec @<=5% | Real FPR @<=5% | Pert AUROC | Pert Rec @<=5% | Rec @ 0.50 | FPR @ 0.50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dalle3 | clip_linear | 702 | 0.9339 | 71.5% | 5.4% | 0.9575 | 74.3% | 76.2% | 8.0% |
| dalle3 | clip_mlp | 702 | 0.9701 | 81.3% | 4.2% | 0.9755 | 83.3% | 78.8% | 3.7% |
| dalle3 | convnext_tiny | 702 | 0.9373 | 59.1% | 4.9% | 0.9629 | 62.3% | 57.1% | 4.2% |
| dalle3 | ensemble | 702 | 0.9782 | 86.3% | 4.8% | 0.9858 | 88.3% | 75.6% | 2.2% |
| flux | clip_linear | 798 | 0.8665 | 55.3% | 4.9% | 0.8881 | 59.3% | 58.4% | 6.4% |
| flux | clip_mlp | 798 | 0.8996 | 63.7% | 5.1% | 0.9090 | 69.3% | 63.0% | 4.8% |
| flux | convnext_tiny | 798 | 0.8569 | 52.1% | 5.2% | 0.8405 | 55.7% | 55.1% | 6.0% |
| flux | ensemble | 798 | 0.9078 | 64.5% | 4.5% | 0.9096 | 68.0% | 59.8% | 3.4% |
| sd21 | clip_linear | 449 | 0.9691 | 80.8% | 4.5% | 0.9704 | 82.7% | 88.6% | 9.1% |
| sd21 | clip_mlp | 449 | 0.9858 | 86.4% | 2.2% | 0.9844 | 88.0% | 89.5% | 3.8% |
| sd21 | convnext_tiny | 449 | 0.9827 | 89.1% | 5.1% | 0.9825 | 88.3% | 88.2% | 4.9% |
| sd21 | ensemble | 449 | 0.9934 | 95.1% | 2.4% | 0.9918 | 94.7% | 93.3% | 1.8% |
| sd3 | clip_linear | 566 | 0.8970 | 50.7% | 6.2% | 0.8976 | 49.3% | 54.2% | 7.0% |
| sd3 | clip_mlp | 566 | 0.9504 | 69.4% | 3.6% | 0.9505 | 69.0% | 73.9% | 4.8% |
| sd3 | convnext_tiny | 566 | 0.9657 | 76.3% | 4.2% | 0.9729 | 79.3% | 83.4% | 6.0% |
| sd3 | ensemble | 566 | 0.9727 | 82.5% | 3.8% | 0.9770 | 81.7% | 77.0% | 2.7% |
| sdxl | clip_linear | 569 | 0.9734 | 83.3% | 4.7% | 0.9790 | 83.3% | 84.9% | 4.9% |
| sdxl | clip_mlp | 569 | 0.9847 | 91.9% | 4.2% | 0.9865 | 91.7% | 91.6% | 4.0% |
| sdxl | convnext_tiny | 569 | 0.9941 | 96.1% | 4.5% | 0.9964 | 98.0% | 94.6% | 3.2% |
| sdxl | ensemble | 569 | 0.9941 | 97.0% | 4.2% | 0.9957 | 98.3% | 93.7% | 2.7% |
| stylegan | clip_linear | 352 | 0.7537 | 13.9% | 3.6% | 0.7957 | 12.7% | 4.8% | 1.5% |
| stylegan | clip_mlp | 352 | 0.7766 | 15.1% | 3.6% | 0.8067 | 16.7% | 3.1% | 1.0% |
| stylegan | convnext_tiny | 352 | 0.4697 | 0.6% | 4.5% | 0.5059 | 2.0% | 0.0% | 0.1% |
| stylegan | ensemble | 352 | 0.6652 | 3.4% | 5.0% | 0.6911 | 7.3% | 0.0% | 0.0% |
