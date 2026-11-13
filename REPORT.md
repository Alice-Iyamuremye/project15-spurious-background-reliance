# Project 15 — Waterbirds Robustness Report

## Title
Spurious Background Reliance in Bird Classification: ERM and Group DRO on the WILDS Waterbirds Benchmark

## Source of truth and reporting rule
The reported test metrics in this document are taken from the saved test-evaluation outputs in the **Project15ML notebook** (`project15ml (1).ipynb`), specifically the ERM and Group DRO test evaluation and side-by-side comparison cells. Where older markdown text, draft reports, or repository summaries disagree with those saved outputs, the saved test outputs are used. These are results from the notebook's recorded run, not a claim of independent reruns.

## Headline test results

| Method | Average test accuracy | Worst-group test accuracy | Average minus worst-group |
|---|---:|---:|---:|
| ERM | 82.98% | 63.55% | 19.43 pp |
| Group DRO | 86.13% | 73.99% | 12.14 pp |

Group DRO improves worst-group test accuracy by **10.44 percentage points** and average test accuracy by **3.15 percentage points** in the saved run.

## Per-subgroup test accuracy

| Subgroup | ERM | Group DRO | Difference |
|---|---:|---:|---:|
| Landbird-on-land | 98.54% | 94.15% | -4.39 pp |
| Landbird-on-water | 74.81% | 84.97% | +10.16 pp |
| Waterbird-on-land | 63.55% | 73.99% | +10.44 pp |
| Waterbird-on-water | 95.02% | 91.43% | -3.58 pp |

The waterbird-on-land subgroup is the worst-performing subgroup for both models in the saved test evaluation. Group DRO improves performance on both minority subgroups in this run, while accuracy decreases on the two majority subgroups.

## Methodology supported by the notebook

- **Dataset:** WILDS Waterbirds, 11,788 images and four bird-type/background subgroups.
- **Data split:** official train, validation and test splits; the notebook records 4,795 training, 1,199 validation and 5,794 test examples.
- **Backbone:** ImageNet-pretrained **ResNet-18**, with a two-class classification head.
- **ERM:** standard empirical risk minimisation using cross-entropy loss.
- **Group DRO:** subgroup-aware loss weighting updated by an exponential/dual-ascent-style rule.
- **Checkpoint selection:** the training code selects checkpoints using worst-group validation accuracy.
- **Evaluation:** per-subgroup test accuracy, average accuracy and worst-group accuracy.
- **Visual inspection:** Grad-CAM heatmaps are generated to inspect model attention.

## Interpretation

The saved test outputs show a higher worst-group accuracy for Group DRO than ERM, alongside a modest increase in average accuracy. This is evidence of improved subgroup performance for this recorded experiment. It does **not** establish that ERM causally relied on background features or that Group DRO eliminated shortcut learning.

Grad-CAM is treated as qualitative inspection only. Heatmaps can suggest where model activations are concentrated, but they do not prove which features causally determine predictions.

## Limitations

- The headline figures describe one saved notebook run; the report does not claim multi-seed statistical significance.
- Only one dataset and one backbone (ResNet-18) are evaluated.
- Group DRO uses subgroup labels during training; those labels may not be available in deployment settings.
- The two minority subgroups have relatively few training examples, limiting what can be concluded about generalisation.
- Grad-CAM is qualitative and should not be treated as causal evidence.
- The results do not demonstrate robustness under arbitrary distribution shifts or real-world deployment conditions.

## Reproducibility note

The notebook records the data setup, training, validation-based checkpoint selection, test evaluation and Grad-CAM inspection workflow. Training and evaluation outputs preserved in the notebook are the basis for the metrics above. A fresh run may differ because of training stochasticity and environment differences.
