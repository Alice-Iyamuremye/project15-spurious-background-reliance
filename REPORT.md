# Project 15 — Report

## Title
Spurious Background Reliance in Bird Classification: Detecting and
Mitigating Background Shortcuts with ERM and Group DRO on the
Waterbirds Benchmark

## Headline Result
Group DRO improves worst-group accuracy from 68.85% (ERM) to 85.51% — a
+16.66 percentage-point improvement, while also improving average
accuracy from 86.61% to 90.29%.

## Problem Statement
Standard image classifiers learn spurious correlations instead of the
actual class-defining features. In the Waterbirds dataset, models
trained with standard ERM (Empirical Risk Minimization) exploit the
background as a shortcut: they classify birds by the surrounding
landscape rather than the bird's actual features. This causes them to
fail on minority subgroups (e.g., waterbirds on land).

## Methodology
- **Dataset:** Waterbirds (WILDS benchmark, 11,788 images, 4 subgroups)
- **Backbone:** ImageNet-pretrained ResNet-50 (23.5M params)
- **Baseline:** ERM — minimize average cross-entropy loss
- **Mitigation:** Group DRO — re-weight groups by loss via dual ascent
- **Evaluation:** Per-subgroup test accuracy, worst-group accuracy, gap
- **Visualization:** Grad-CAM saliency maps

## Results Summary

| Method   | Average | Worst-group | Gap    |
|----------|---------|-------------|--------|
| ERM      | 86.61%  | 68.85%      | 17.77 pp |
| Group DRO| 90.29%  | 85.51%      | 4.78 pp  |

Per-subgroup accuracy:
| Subgroup              | ERM    | Group DRO | Delta     |
|-----------------------|--------|-----------|-----------|
| landbird-on-land      | 98.80% | 94.50%    | -4.30 pp  |
| landbird-on-water     | 85.81% | 87.23%    | +1.42 pp  |
| waterbird-on-land     | 68.85% | 85.51%    | +16.66 pp |
| waterbird-on-water    | 92.99% | 93.93%    | +0.94 pp  |

## Conclusion
ERM exploits background shortcuts in the Waterbirds dataset, producing
a 17.77-percentage-point gap between average and worst-group accuracy.
Group DRO closes this gap dramatically, achieving 85.51% worst-group
accuracy (+16.66 pp). Grad-CAM saliency maps confirm the shortcut
visually: ERM attends to background, while Group DRO attends to the
bird itself.

## Files in this delivery
- `project15_waterbirds.ipynb` — presentation notebook (20 cells)
- `erm/` — source code (9 Python files)
- `REPORT.md` — this report
