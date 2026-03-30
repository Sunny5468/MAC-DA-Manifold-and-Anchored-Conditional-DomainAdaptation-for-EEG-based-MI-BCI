# MSCFormer

This directory stores MSCFormer LOSO results on BCI2a.

## Purpose

- Provide transformer-style baseline results for architecture-family comparison.
- Benchmark MSCFormer against convolutional and adaptation-based pipelines.

## Contents

- `MSCFormer_bcic2a_loso_seed-0_*`
- `MSCFormer_bcic2a_loso_seed-1_*`
- `MSCFormer_bcic2a_loso_seed-2_*`
- `MSCFormer_bcic2a_loso_seed-3_*`
- `MSCFormer_bcic2a_loso_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Compare with `results/ATCNet_Baseline`, `results/EEGNet`, and `results/EEGTCNet`
  to evaluate architecture-specific strengths.
- Keep seed-aligned comparisons whenever possible.

## Notes

- This folder is for model-level baseline comparison, not domain-adaptation ablation.
