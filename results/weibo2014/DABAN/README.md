# DABAN

This directory stores ATCNet + DABAN LOSO results on Weibo2014.

## Method

DABAN setting:
- Backbone: ATCNet
- Adaptation: DABAN (Domain Adaptation with Balanced Adversarial Network)
- Dataset protocol: Weibo2014 LOSO
- Seeds: 0-4

## Contents

- ATCNet_MI_DABAN_weibo2014_loso_cdan_seed-0_aug-True_*
- ATCNet_MI_DABAN_weibo2014_loso_cdan_seed-1_aug-True_*
- ATCNet_MI_DABAN_weibo2014_loso_cdan_seed-2_aug-True_*
- ATCNet_MI_DABAN_weibo2014_loso_cdan_seed-3_aug-True_*
- ATCNet_MI_DABAN_weibo2014_loso_cdan_seed-4_aug-True_*
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
| Accuracy (%) | 46.96 +- 10.29 |
| Kappa (%) | 29.27 +- 13.72 |
| Loss (x100) | 127.61 +- 26.37 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
- All seed-subject combinations are present.
