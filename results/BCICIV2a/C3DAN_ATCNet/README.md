# C3DAN_ATCNet

This directory stores ATCNet + CDAN + CCCORAL (C3DAN) LOSO results on BCI2a.

## Purpose

- Track the combined method results for seed-wise comparison against CDAN and baseline.
- Preserve the primary C3DAN outputs used in performance analysis.

## Contents

- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_seed-0_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_seed-1_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_seed-2_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_seed-3_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Compare this directory with `results/CDAN_ATCNet` to measure CCCORAL contribution.
- Compare this directory with `results/ATCNet_Baseline` to measure total adaptation gain.

## Notes

- Folder name uses `CDAN_CCCORAL` to indicate the combined adaptation setting.
