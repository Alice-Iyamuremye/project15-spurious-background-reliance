# Project 15 — Spurious Background Reliance in Bird Classification
## Project Specification & Planning Document

---

## 1. Project Title

**Spurious Background Reliance in Bird Classification: Detecting, Visualizing, and Mitigating Background Shortcuts on the Waterbirds Dataset.**

A more concise variant:
**Background Shortcuts in Bird Classification — Worst-Group Robustness on Waterbirds.**

---

## 2. Problem Statement

Given the **Waterbirds** dataset (a subset of CUB-200, re-grouped into *landbirds* vs *waterbirds* with deliberate background bias), train an image classifier and answer:

> *Does the classifier actually recognize the bird species, or does it learn a spurious shortcut based on the background (water vs land)?*

More concretely, the project must:

1. **Train** one or more classifiers on the biased training distribution.
2. **Evaluate** performance not just on average, but **per subgroup** (4 combinations of {bird type} × {background type}).
3. **Diagnose** the failure mode using **saliency maps** (Grad-CAM or similar) to verify whether the model attends to the bird or the background.
4. **Mitigate** the shortcut with at least one robustness method and demonstrate that worst-group accuracy improves.

---

## 3. Main Question Type

This project sits at the intersection of two task categories:

| Dimension | Value |
|---|---|
| **Surface task** | **Classification** — binary image classification (landbird vs waterbird). |
| **Deeper question** | **Robustness** — specifically **subgroup robustness** under spurious correlation. |

So the framing is:

> *Binary classification **evaluated under a robustness protocol** — strict subgroup / worst-group analysis.*

This is **not** retrieval, segmentation, or recommendation. It is classification + robustness evaluation + interpretability.

---

## 4. What the Project Compares / Tests / Demonstrates

The project must **compare** at least two things and **demonstrate** at least one phenomenon:

### Required comparisons
- **A. Average accuracy vs Worst-group accuracy** for the same trained model. This shows that average accuracy alone is misleading.
- **B. Standard training (ERM) vs a robustness method** (e.g., Group DRO, reweighting, or strong augmentation). This shows whether mitigation works.

### Required demonstrations
- **C. Saliency evidence** that an ERM model attends to the background, while a mitigated model attends more to the bird.
- **D. The four-subgroup breakdown table** showing where failures concentrate.

### Optional / stretch
- **E. Per-class confusion** on minority subgroups (e.g., landbird-on-water misclassified as waterbird).
- **F. Effect of model capacity / pretraining** on shortcut reliance.

---

## 5. Special Requirements

The problem statement specifies a **strict evaluation protocol**. The required special handling is:

### ✅ Subgroup analysis (REQUIRED)
Every reported metric must be broken down by the 4 subgroups:
1. Landbird + Land background (majority, easy)
2. Waterbird + Water background (majority, easy)
3. Landbird + Water background (minority, hard)
4. Waterbird + Land background (minority, hard)

### ✅ Worst-group accuracy (REQUIRED)
The headline number must be **worst-group accuracy**, not average. This is the WILDS protocol.

### ✅ Saliency visualization (REQUIRED)
Generate Grad-CAM (or occlusion / integrated gradients) heatmaps for representative examples, especially from the **failing minority subgroups**. This is the qualitative evidence that the shortcut exists.

### ✅ Mitigation comparison (REQUIRED)
Compare at least ERM (baseline) against one mitigation strategy. Showing only ERM is incomplete.

### Other WILDS-style considerations
- **No data leakage**: backgrounds seen at test time may differ from training; this is by design.
- **Fixed train/val/test split** as provided by the WILDS package — do not re-split.
- **Group labels** (`background == "water"` etc.) are available from the dataset metadata; mitigation methods that need group labels (like Group DRO) are allowed to use them at training time.

---

## 6. What Counts as a Good Solution vs an Incomplete / Misleading One

### ✅ A good solution
1. Trains **at least two models** (ERM + one mitigation method).
2. Reports a **per-subgroup accuracy table** for both models.
3. Reports **average and worst-group accuracy** as the primary numbers, with worst-group highlighted.
4. Shows that the mitigation method **meaningfully improves worst-group accuracy** (e.g., from ~60% to ~80%+).
5. Provides **saliency maps** on at least 4–8 examples that visually demonstrate the background-attention problem in ERM and its reduction in the mitigated model.
6. Discusses **why** the mitigation works (or doesn't), referencing the loss weighting or training objective.
7. Uses the **official WILDS train/val/test splits** and reports val worst-group accuracy for model selection.

### ❌ An incomplete or misleading solution
1. **Reporting only average accuracy** — this hides the very problem the project is designed to expose.
2. **Reporting only overall accuracy** without subgroup breakdown — same issue.
3. **No saliency visualization** — without it, you have no evidence the shortcut exists; you only have an indirect signal.
4. **Re-splitting the data** or shuffling groups — invalidates the WILDS protocol.
5. **Testing the mitigation on the same biased metric** — mitigation must be evaluated by worst-group accuracy, not average.
6. **Choosing the model on average validation accuracy** — this rewards shortcut learning; must be chosen by **worst-group validation accuracy**.
7. **Reporting one model** and calling it "robust" — robustness claims require a comparison.
8. **Hand-waving saliency** without real Grad-CAM output — showing "the model might focus on background" without visualizing it is not evidence.

---

## 7. Realistic Scope

### Dataset size justification

The Waterbirds dataset is **small enough to use in full**:

| Split | Images | Notes |
|---|---|---|
| Train | ~4,795 | Heavily biased: ~93% majority subgroups |
| Validation | ~4,799 | Used for worst-group model selection |
| Test | ~4,799 | Final reported metrics |

- Total ≈ **9.6k train+val, ~4.8k test**. This is a tiny dataset by modern standards.
- **No subsetting is needed.** Using a subset would actually weaken the project because:
  - The minority subgroups (~7% each) would become too small to evaluate reliably.
  - The point of the dataset is the *bias distribution*, which is fixed by design.

### Compute / model choices

A realistic scope balances rigor with feasibility:

| Choice | Recommendation | Reason |
|---|---|---|
| **Backbone** | ResNet-18 (ImageNet pretrained) | Fast enough to train multiple variants; standard WILDS baseline. |
| **Epochs** | 50–100 | Converges well for this size; Group DRO needs more epochs. |
| **Mitigation methods** | (a) Group DRO; (b) upweighting minority groups (optional 3rd: stronger augmentation) | Group DRO is the canonical WILDS baseline; gives a strong comparison point. |
| **Saliency tool** | Grad-CAM on the final conv layer of ResNet | Standard, fast, interpretable. |
| **Saliency examples** | 4–8 images covering all 4 subgroups, before/after mitigation | Enough to tell a clear visual story. |

### Time budget (rough)
- Data loading + EDA + subgroup inspection: ~1 hour
- ERM training (1 model): ~30–60 min on a single GPU
- Group DRO training (1 model): ~60–90 min
- Saliency generation + plots: ~1 hour
- Report writing: ~2–3 hours

This is well within the scope of a single project.

---

## 8. One-paragraph Project Pitch (for proposals / readme)

> *We tackle the problem of **spurious background reliance** in image classifiers using the Waterbirds benchmark, where landbirds and waterbirds are photographed on land and water backgrounds respectively. Standard empirical risk minimization (ERM) achieves high average accuracy by exploiting the background→class shortcut, but performs near-randomly on the minority subgroups (e.g., landbirds on water). We train an ERM baseline and a Group DRO model, evaluate both under the strict **worst-group accuracy** protocol with full subgroup breakdowns, and use **Grad-CAM saliency maps** to demonstrate that the ERM model attends to the background while the robust model attends to the bird itself. The project demonstrates that average accuracy hides shortcut learning, that subgroup-aware training mitigates it, and that saliency maps provide direct visual evidence.*

