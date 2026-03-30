# DABAN_ATCNet

This directory stores ATCNet + MI-DABAN (Domain Adversarial Batch Normalization with Mutual Information) LOSO results on BCI Competition IV 2a.

## Method

MI-DABAN applies domain adaptation through:
- **Domain Adversarial Batch Normalization (DABN)** — separate BN statistics for source/target domains with adversarial alignment
- **Mutual Information (MI) regularization** — encourages discriminative feature learning across domains

Model: `ATCNet_MI_DABAN`, #Params: 286,774

## Contents

- `ATCNet_MI_DABAN_bcic2a_loso_cdan_seed-0_*`
- `ATCNet_MI_DABAN_bcic2a_loso_cdan_seed-1_*`
- `ATCNet_MI_DABAN_bcic2a_loso_cdan_seed-2_*`
- `ATCNet_MI_DABAN_bcic2a_loso_cdan_seed-3_*`
- `ATCNet_MI_DABAN_bcic2a_loso_cdan_seed-4_*`

Each run folder includes:
- `results.txt` — per-subject accuracy, loss, kappa, and summary statistics
- `confmats/` — confusion matrix plots per subject
- `curves/` — training accuracy and loss curves per subject
- `tsne/` — t-SNE visualizations (per-subject class distribution and source/target domain)
- `config.yaml` — experiment configuration

## How to interpret

- Compare with `results/C3DAN_ATCNet` (CDAN + CCCORAL) to evaluate DABAN vs. explicit adversarial + CORAL alignment.
- Compare with `results/DANN_ATCNet` to compare DABAN vs. standard DANN.
- Compare with `results/ATCNet_Baseline` to measure total adaptation gain.

## Known Issues

- Seed-2 has partial N/A values for Train Time, Test Time, and Loss in subjects 1-6 (recovered/merged results from interrupted run).

## Summary (5 seeds, 9 subjects)

| Metric | Mean ± Std |
| --- | ---: |
| Accuracy (%) | 65.07 ± 15.34 |
| Kappa (%) | 53.43 ± 20.46 |
| Loss (×100) | 94.53 ± 58.97 |
