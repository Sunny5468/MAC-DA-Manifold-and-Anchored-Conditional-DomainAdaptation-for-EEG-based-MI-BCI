# ATCNet_Baseline

This directory stores baseline ATCNet LOSO results on BCI2a without domain-adversarial adaptation.

## Purpose

- Provide the reference performance for comparison with DANN, CDAN, and CDAN+CCCORAL.
- Keep baseline runs grouped by seed for reproducibility.

## Contents

- `ATCNet_bcic2a_loso_seed-0_*`
- `ATCNet_bcic2a_loso_seed-1_*`
- `ATCNet_bcic2a_loso_seed-2_*`
- `ATCNet_bcic2a_loso_seed-3_*`
- `ATCNet_bcic2a_loso_seed-4_*`

Each run folder usually includes per-subject outputs and a `results.txt` summary.

## How to interpret

- Use this directory as the baseline anchor in method comparison tables.
- For statistical claims, compare seed-matched runs against CDAN/DANN/C3DAN directories.

## Notes

- Naming fields encode dataset mode, seed, augmentation flag, GPU id, and timestamp.
