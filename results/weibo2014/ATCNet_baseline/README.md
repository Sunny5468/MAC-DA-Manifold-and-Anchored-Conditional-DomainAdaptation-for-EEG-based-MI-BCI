# ATCNet_baseline

This directory stores ATCNet baseline LOSO results on Weibo2014 (no domain adaptation module).

## Method

Baseline setting:
- Backbone: ATCNet
- Dataset protocol: Weibo2014 LOSO
- Adaptation: disabled
- Seeds: 0-4

## Contents

- ATCNet_weibo2014_loso_seed-0_aug-False_*
- ATCNet_weibo2014_loso_seed-1_aug-False_*
- ATCNet_weibo2014_loso_seed-2_aug-False_*
- ATCNet_weibo2014_loso_seed-3_aug-False_*
- ATCNet_weibo2014_loso_seed-4_aug-False_*
- summarize.md

Each run folder includes:
- results.txt: per-subject accuracy, loss, kappa, and summary statistics
- confmats/: confusion matrix plots per subject
- curves/: training accuracy and loss curves per subject
- tsne/: t-SNE visualizations
- config.yaml: experiment configuration

## Summary (5 seeds, available rows)

| Metric | Mean +- Std |
| --- | ---: |
| Accuracy (%) | 45.91 +- 12.12 |
| Kappa (%) | 27.89 +- 16.15 |
| Loss (x100) | 152.06 +- 58.36 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
- All seed-subject combinations are present.
