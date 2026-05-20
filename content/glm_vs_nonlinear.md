# Generalised Linear Models vs Non-Linear Models for Classification

## What a Generalised Linear Model Is

A GLM is an extension of ordinary linear regression that allows the outcome variable to follow a wider family of distributions — including binary outcomes (default / no default), counts, and rates (Lindsey, 1999). A GLM combines three ingredients: a probability distribution for the outcome (the "random component"), a linear combination of input features called the **linear predictor**, and a **link function** that connects the two (Heit et al., 2024; Lindsey, 1999).

The linear predictor takes the form:

> η = β₀ + β₁x₁ + β₂x₂ + … + βₖxₖ

For binary classification, the GLM of choice is **logistic regression**, which uses the logit link:

> logit(p) = ln(p / (1 − p)) = η

Rearranging for the probability of default:

> P(default | X) = 1 / (1 + e^(−η))

Logistic regression is "linear" in the sense that the features enter the model linearly on the log-odds scale, but the relationship between the features and the predicted probability is non-linear — an S-shaped curve that asymptotes at 0 and 1. This is a subtle but important point: GLMs accommodate non-linearity in the outcome through the link function while keeping the relationship between predictors and the linear predictor linear (Heit et al., 2024).

## What a Non-Linear Classifier Is

Non-linear classifiers — decision trees, random forests, gradient-boosted machines such as LightGBM and XGBoost, and neural networks — do not assume any fixed functional form. They learn the shape of the decision boundary directly from the data, which lets them capture interactions and threshold effects that a GLM would only see if explicitly engineered (Scholbeck et al., 2024; Zhang et al., 2026).

This flexibility is both their strength and their weakness. On rich data with complex patterns they often outperform GLMs in raw predictive accuracy (Zhang et al., 2026). On simpler problems the gap narrows considerably, particularly once careful feature engineering is applied to the GLM side (Sudjianto & Burakov, 2025).

## Why GLMs Dominate in Credit

Despite a lower ceiling on predictive performance, logistic regression remains the dominant model class in retail credit for three reasons:

1. **Interpretability.** Each coefficient β captures the change in log-odds of default for a one-unit change in that feature, holding everything else constant. A non-linear model's "feature importance" is harder to translate into a defensible story (Scholbeck et al., 2024).
2. **Stability.** GLMs are less prone to overfitting noisy patterns and tend to be more stable across time periods, which matters for ongoing monitoring and revalidation (Sudjianto & Burakov, 2025).
3. **Regulatory acceptance.** Banking regulators expect models they can audit. A scorecard derived from a logistic regression is auditable in a way that an ensemble of 500 boosted trees is not (Dianta et al., 2026; Sharma et al., 2023).

## The Gap Is Smaller Than It Appears

A common assumption is that switching from logistic regression to a non-linear model unlocks a large performance jump. Recent work suggests this gap is largely an artefact of weak feature engineering. Sudjianto and Burakov (2025) show that logistic regression with properly binned WoE-transformed features achieves AUC values in the 0.82–0.84 range on realistic credit data, matching constrained XGBoost. The implication is sharp: **optimal feature engineering matters more than the choice of model class.** Most of the predictive lift comes from transforming the features sensibly, not from adopting a more flexible learner.
