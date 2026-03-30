# CDAN_ATCNet

This directory stores ATCNet + CDAN LOSO results on BCI2a.

## Purpose

- Serve as the main domain-adversarial baseline for ablation and comparison.
- Provide seed-matched references for evaluating CCCORAL effects.

## Contents

- `ATCNet_CDAN_bcic2a_loso_cdan_seed-0_*`
- `ATCNet_CDAN_bcic2a_loso_cdan_seed-1_*`
- `ATCNet_CDAN_bcic2a_loso_cdan_seed-2_*`
- `ATCNet_CDAN_bcic2a_loso_cdan_seed-3_*`
- `ATCNet_CDAN_bcic2a_loso_cdan_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Use this directory as the direct counterpart when computing `CCCORAL - No-CCCORAL` deltas.
- Keep comparisons seed-aligned (for example seed-0 vs seed-0).

## Notes

- The `cdan` token in names indicates CDAN training mode in this project.
