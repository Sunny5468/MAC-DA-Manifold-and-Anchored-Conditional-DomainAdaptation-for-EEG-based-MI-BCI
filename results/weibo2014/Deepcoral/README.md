# Deepcoral

This directory stores ATCNet + DeepCORAL LOSO results on Weibo2014.

## Method

DeepCORAL setting:
- Backbone: ATCNet
- Adaptation: DeepCORAL
- Dataset protocol: Weibo2014 LOSO
- Seeds: 0-4

## Contents

- ATCNet_DeepCORAL_weibo2014_loso_cdan_seed-0_aug-True_*
- ATCNet_DeepCORAL_weibo2014_loso_cdan_seed-1_aug-True_*
- ATCNet_DeepCORAL_weibo2014_loso_cdan_seed-2_aug-True_*
- ATCNet_DeepCORAL_weibo2014_loso_cdan_seed-3_aug-True_*
- ATCNet_DeepCORAL_weibo2014_loso_cdan_seed-4_aug-True_*
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
| Accuracy (%) | 47.21 +- 10.99 |
| Kappa (%) | 29.61 +- 14.65 |
| Loss (x100) | 126.47 +- 24.71 |

## Notes

- Statistics are aggregated from all results.txt files under this folder.
- Parsed rows: 50 (seeds: 0-4, subjects: 1-10).
- All seed-subject combinations are present.
