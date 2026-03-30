# TSSEFFNet

This directory stores TSSEFFNet LOSO results on BCI2a.

## Purpose

- Provide TSSEFFNet benchmark results in the same protocol as other model baselines.
- Support cross-model robustness analysis across seeds.

## Contents

- `TSSEFFNet_bcic2a_loso_seed-0_*`
- `TSSEFFNet_bcic2a_loso_seed-1_*`
- `TSSEFFNet_bcic2a_loso_seed-2_*`
- `TSSEFFNet_bcic2a_loso_seed-3_*`
- `TSSEFFNet_bcic2a_loso_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Use this folder as one model-family reference in multi-model comparison tables.
- Prefer multi-seed summary statistics for reporting.

## Notes

- Naming convention follows the same seed/timestamp format used across `results/`.
