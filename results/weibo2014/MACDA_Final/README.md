# MACDA_Final

This directory stores ATCNet + CDAN + CCCORAL + EA LOSO results on Weibo2014.

## Method

MACDA_Final uses multi-component adaptation with:
- CDAN-based adversarial alignment
- CCCORAL correlation alignment regularization
- EA preprocessing in the data pipeline

## Contents

- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_ea_seed-0_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_ea_seed-1_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_ea_seed-2_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_ea_seed-3_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_ea_seed-4_*
- summarize.md

Each run folder includes:
- results.txt: per-subject accuracy, loss, kappa, and summary statistics
- confmats/: confusion matrix plots per subject
- curves/: training accuracy and loss curves per subject
- tsne/: t-SNE visualizations (per-subject class distribution and source/target domain)
- config.yaml: experiment configuration

## Summary (5 seeds, 10 subjects)

| Metric | Mean ± Std |
| --- | ---: |
| Accuracy (%) | 53.70 ± 10.60 |
| Kappa (%) | 38.27 ± 14.13 |
| Loss (x100) | 118.20 ± 28.18 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
