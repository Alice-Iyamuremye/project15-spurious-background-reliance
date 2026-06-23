
# Project 15 — Spurious Background Reliance in Bird Classification

Detecting and mitigating background shortcuts in bird classification using
**ERM** (Empirical Risk Minimization) and **Group DRO** (Distributionally
Robust Optimization) on the **Waterbirds** benchmark.

## Headline Result

| Method    | Average | Worst-group | Gap    |
|-----------|---------|-------------|--------|
| ERM       | 83.65%  | 64.64%      | 19.01pp|
| Group DRO | 86.11%  | 73.36%      | 12.75pp|

Group DRO improves worst-group accuracy by **+8.7 percentage points**.

## What's in this repo

- `project15_waterbirds.ipynb` — Presentation notebook (20 cells)
- `REPORT.md` — Detailed results and analysis
- `project15_spec.md` — Planning specification
- `erm/` — Source code (9 Python files)

## How to run

See the notebook cells 1-9 for setup. Cells 10+ train and evaluate the models
(requires ~80 minutes on a single GPU).

## Reference

Sagawa et al. (2019), "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020.
