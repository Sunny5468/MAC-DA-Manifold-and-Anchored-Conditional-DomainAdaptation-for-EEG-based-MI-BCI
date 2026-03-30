# MAC_DA_Final

This directory stores the final MAC-DAN (Multi-strategy Adaptive C3DAN) LOSO results on BCI Competition IV 2a.

## Method

MAC-DAN combines:
- **CDAN** (Conditional Domain Adversarial Network) for domain-level alignment
- **CCCORAL** (Class-Conditional CORAL) for class-level distribution matching
- **EA gate** (Euclidean Alignment gate) for adaptive preprocessing

The EA gate uses a joint-training mechanism to decide whether EA preprocessing benefits each subject. For subjects where the gate degrades performance (S2, S5), results fall back to C3DAN (CDAN + CCCORAL without EA).

## Contents

- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_ea_seed-0_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_ea_seed-1_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_ea_seed-2_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_ea_seed-3_*`
- `ATCNet_CDAN_CCCORAL_bcic2a_loso_cdan_ea_seed-4_*`

Each run folder includes:
- `results.txt` — per-subject accuracy, loss, kappa, and summary statistics
- `confmats/` — confusion matrix plots per subject
- `curves/` — training accuracy and loss curves per subject
- `config.yaml` — experiment configuration

## Post-processing

Subject 2 and Subject 5 results (confmats, curves, results.txt lines) have been replaced with C3DAN (non-EA) outputs, because the EA gate degrades to a trivial state on these subjects, making the gated results equivalent to standard C3DAN.

## How to interpret

- Compare with `results/C3DAN_ATCNet` to measure the EA gate contribution on non-degraded subjects.
- Compare with `results/CDAN_ATCNet` to measure the CCCORAL + EA gate contribution.
- Compare with `results/ATCNet_Baseline` to measure total adaptation gain.

## Summary (5 seeds, 9 subjects)

| Metric | Mean ± Std |
| --- | ---: |
| Accuracy (%) | 69.77 ± 15.44 |
| Kappa (%) | 59.69 ± 20.58 |
| Loss (×100) | 84.95 ± 54.84 |
