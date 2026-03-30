# DeepCoral

This directory stores ATCNet + DeepCORAL LOSO results on BCI2a.

## Purpose

- Serve as a domain adaptation baseline using the DeepCORAL method.
- Provide comparison with other domain adaptation approaches (DANN, CDAN, C3DAN).

## Contents

- `ATCNet_DeepCORAL_bcic2a_loso_cdan_seed-0_*`
- `ATCNet_DeepCORAL_bcic2a_loso_cdan_seed-1_*`
- `ATCNet_DeepCORAL_bcic2a_loso_cdan_seed-2_*`
- `ATCNet_DeepCORAL_bcic2a_loso_cdan_seed-3_*`
- `ATCNet_DeepCORAL_bcic2a_loso_cdan_seed-4_*`

Each run folder includes per-subject outputs and a `results.txt` summary.

## How to interpret

- DeepCORAL aligns source and target domain distributions by minimizing the distance between their second-order statistics (covariances).
- Compare with DANN and CDAN to evaluate different domain adaptation strategies.
- Keep comparisons seed-aligned when analyzing performance differences.

## Notes

- The `cdan` token in folder names refers to the training pipeline used, not the actual method (which is DeepCORAL).
