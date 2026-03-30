# EEGTCNet

This directory stores EEGTCNet LOSO results on BCI2a.

## Purpose

- Provide a temporal-convolution EEG baseline for comparison with EEGNet and ATCNet.
- Support model-family ablation and robustness analysis across seeds.

## Contents

- `EEGTCNet_bcic2a_loso_seed-0_*`
- `EEGTCNet_bcic2a_loso_seed-1_*`
- `EEGTCNet_bcic2a_loso_seed-2_*`
- `EEGTCNet_bcic2a_loso_seed-3_*`
- `EEGTCNet_bcic2a_loso_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Compare with `results/EEGNet` to understand gains from temporal-convolution design.
- Compare with `results/ATCNet_Baseline` for broader architecture-level benchmarking.

## Notes

- Keep seed-aligned comparisons (for example seed-2 vs seed-2) for fair analysis.
