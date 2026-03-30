# EEGNet

This directory stores EEGNet LOSO results on BCI2a.

## Purpose

- Provide a lightweight non-ATCNet baseline for cross-model comparison.
- Serve as a reference when discussing architecture-level performance gaps.

## Contents

- `EEGNet_bcic2a_loso_seed-0_*`
- `EEGNet_bcic2a_loso_seed-1_*`
- `EEGNet_bcic2a_loso_seed-2_*`
- `EEGNet_bcic2a_loso_seed-3_*`
- `EEGNet_bcic2a_loso_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Use seed-matched comparisons against other model families when possible.
- Use multi-seed averages for claims instead of single-run numbers.

## Notes

- Folder names encode dataset mode, seed, augmentation flag, GPU id, and timestamp.
