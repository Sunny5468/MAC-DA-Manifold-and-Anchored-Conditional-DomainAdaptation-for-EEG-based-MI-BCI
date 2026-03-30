# CDAN

This directory stores ATCNet + CDAN LOSO results on Weibo2014.

## Method

CDAN setting:
- Backbone: ATCNet
- Adaptation: Conditional Domain Adversarial Network (CDAN)
- Dataset protocol: Weibo2014 LOSO
- Seeds: 0-4

## Contents

- ATCNet_CDAN_weibo2014_loso_cdan_seed-0_aug-False_*
- ATCNet_CDAN_weibo2014_loso_cdan_seed-1_aug-False_*
- ATCNet_CDAN_weibo2014_loso_cdan_seed-2_aug-False_*
- ATCNet_CDAN_weibo2014_loso_cdan_seed-3_aug-False_*
- ATCNet_CDAN_weibo2014_loso_cdan_seed-4_aug-False_*
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
| Accuracy (%) | 50.76 +- 12.02 |
| Kappa (%) | 34.35 +- 16.03 |
| Loss (x100) | 124.19 +- 27.89 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
- All seed-subject combinations are present.
