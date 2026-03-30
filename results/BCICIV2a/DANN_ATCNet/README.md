# DANN_ATCNet

This directory stores ATCNet + DANN LOSO results on BCI2a.

## Purpose

- Provide a classic adversarial adaptation baseline (DANN) for method-level comparison.
- Contrast DANN against CDAN/C3DAN to analyze conditional alignment benefits.

## Contents

- `ATCNet_DANN_bcic2a_loso_cdan_seed-0_*`
- `ATCNet_DANN_bcic2a_loso_cdan_seed-1_*`
- `ATCNet_DANN_bcic2a_loso_cdan_seed-2_*`
- `ATCNet_DANN_bcic2a_loso_cdan_seed-3_*`
- `ATCNet_DANN_bcic2a_loso_cdan_seed-4_*`

This run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Use this directory as the feature-level adversarial baseline counterpart of CDAN/C3DAN.
- Prefer seed-aligned comparison when reporting deltas across methods.

## Notes

- Although the folder name contains `cdan` in part of the run token, this directory is intended for DANN experiments.
- Seed-3 currently uses merged artifacts (`...2101_merge_20260317_1533`), where full metrics are partially incomplete.
- See `summarize.md` for detailed statistics and completeness notes.
