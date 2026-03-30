# EA-ATCNet

This directory stores the EA-enabled ATCNet LOSO results on BCI Competition IV 2a.

## Method

EA-ATCNet uses:
- **ATCNet** as the backbone classifier
- **EA (Euclidean Alignment)** as preprocessing before model training
- **No domain adaptation module** (no CDAN / DANN / CORAL / CCCORAL)

This setting is used as an EA-only reference to measure how much gain comes from EA preprocessing itself.

## Contents

- `ATCNet_bcic2a_loso_ea_seed-0_*`
- `ATCNet_bcic2a_loso_ea_seed-1_*`
- `ATCNet_bcic2a_loso_ea_seed-2_*`
- `ATCNet_bcic2a_loso_ea_seed-3_*`
- `ATCNet_bcic2a_loso_ea_seed-4_*`
- `summarize.md`

Each run folder includes:
- `results.txt` - per-subject accuracy, loss, kappa, and summary statistics
- `confmats/` - confusion matrix plots per subject
- `curves/` - training curves per subject
- `config.yaml` - experiment configuration

## How to interpret

- Compare with `results/ATCNet_Baseline` to isolate EA preprocessing gain.
- Compare with `results/CDAN_ATCNet` and `results/C3DAN_ATCNet` to measure additional domain adaptation gain over EA-only baseline.
- Subject 2 is still a hard target in this setup, while subjects 1, 3, and 8 remain relatively strong.

## Summary (5 seeds, 9 subjects)

| Metric | Mean ± Std |
| --- | ---: |
| Accuracy (%) | 64.42 ± 12.52 |
| Kappa (%) | 52.56 ± 16.69 |
| Loss (x100) | 94.29 ± 35.26 |
