# C3DAN

This directory stores ATCNet + CDAN + CCCORAL LOSO results on Weibo2014.

## Method

C3DAN setting:
- Backbone: ATCNet
- Adaptation: CDAN + Class-Conditional CORAL (CCCORAL)
- Dataset protocol: Weibo2014 LOSO
- Seeds: 0-4

## Contents

- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_seed-0_aug-True_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_seed-1_aug-True_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_seed-2_aug-True_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_seed-3_aug-True_*
- ATCNet_CDAN_CCCORAL_weibo2014_loso_cdan_seed-4_aug-True_*
- summarize.md

Each run folder includes:
- results.txt: per-subject accuracy, loss, kappa, and summary statistics
- confmats/: confusion matrix plots per subject
- curves/: training accuracy and loss curves per subject
- tsne/: t-SNE visualizations
- config.yaml: experiment configuration

## Summary (5 seeds, 10 subjects)

| Metric | Mean +- Std |
| --- | ---: |
| Accuracy (%) | 46.83 +- 11.76 |
| Kappa (%) | 29.10 +- 15.68 |
| Loss (x100) | 129.79 +- 28.32 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
- All seed-subject combinations are present.
