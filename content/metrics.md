# Model Evaluation Metrics in Credit

A credit model is evaluated against multiple criteria — no single metric is sufficient. All of the metrics below are built from the same four primitives:

- **TP (True Positive):** correctly predicted defaults
- **TN (True Negative):** correctly predicted non-defaults
- **FP (False Positive):** good customers wrongly predicted as defaulters
- **FN (False Negative):** defaulters wrongly predicted as good customers

## Accuracy

> Accuracy = (TP + TN) / (TP + TN + FP + FN)

Accuracy answers "how often is the model right?" In credit, it is almost always misleading. Defaults are rare — typically 5–15% of a portfolio. A model that predicts "no default" for everyone is 90%+ accurate but useless, because it catches zero defaulters. Accuracy alone hides the failure mode the bank cares about most: the rare-but-expensive false negative (Zhang et al., 2026).

## Precision and Recall

> **Precision = TP / (TP + FP)** — of the customers the model flagged as risky, what share actually defaulted?
>
> **Recall = TP / (TP + FN)** — of the customers who actually defaulted, what share did the model catch?

These have direct business interpretations:

- **Low precision** means the model wrongly flags too many good customers as risky. The bank rejects profitable applicants and loses revenue.
- **Low recall** means the model misses too many real defaulters. The bank issues loans that go bad and absorbs the losses.

Banks adjust the precision-recall balance to reflect their risk appetite. A conservative lender tolerates lower precision (more rejections) to push recall up (fewer missed defaulters). An aggressive lender tolerates lower recall to push precision up (fewer rejected applicants). The choice is a policy decision, not a statistical one.

## F1 Score

> F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 combines precision and recall into a single metric using a harmonic mean. The harmonic mean penalises imbalance: a model with precision 0.9 and recall 0.1 has F1 ≈ 0.18, not 0.5. F1 is useful for comparing classifiers when both error types matter, but in credit it is usually secondary to ranking metrics such as AUC and Gini.

## ROC and AUC

A Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate at every possible classification threshold. The Area Under the Curve (AUC) is the area below this curve. It has a clean interpretation: **AUC is the probability that the model assigns a higher risk score to a randomly chosen defaulter than to a randomly chosen non-defaulter** (Sudjianto & Burakov, 2025).

A practical scale for AUC:

- 0.5 = no skill (random guessing)
- 0.6 – 0.7 = weak
- 0.7 – 0.8 = acceptable
- 0.8 – 0.9 = strong
- \> 0.9 = excellent (or check for leakage)

For context, the project benchmarks are AUC = 0.68 (baseline logistic regression) and AUC = 0.82 (reference LightGBM). This is the gap the feature-engineering work has to close — through better features, not through switching model classes.

AUC is the dominant metric in credit because lending is fundamentally a **ranking problem**. Banks do not simply classify customers as "good" or "bad" — they assign a probability of default and set a cut-off based on policy. A model that ranks customers well is more valuable than a model that classifies well at a single arbitrary threshold (Sudjianto & Burakov, 2025).

## Gini Coefficient

> Gini = 2 × AUC − 1

The Gini coefficient is a linear transformation of AUC. For AUC = 0.82, Gini = 0.64. It measures the same thing as AUC — ranking quality — but rescaled to run from 0 (no skill) to 1 (perfect separation). Gini is deeply embedded in banking regulation, Basel reporting, and traditional scorecard documentation, which is why it persists alongside AUC.

## Metric Summary

| Metric | Question Answered | Credit Interpretation |
|---|---|---|
| Accuracy | How often is the model right? | Misleading on imbalanced data |
| Precision | Are flagged borrowers really risky? | Avoid rejecting profitable customers |
| Recall | Did we catch the actual defaulters? | Reduce loss from bad loans |
| F1 | Balance of precision and recall | Useful comparison metric |
| AUC | How well does the model rank risk? | Core scorecard performance metric |
| Gini | How strong is risk separation? | Banking-standard regulatory metric |

A model with mediocre accuracy but strong AUC can still be commercially valuable. The opposite — high accuracy but weak AUC — is almost always useless in production lending.
