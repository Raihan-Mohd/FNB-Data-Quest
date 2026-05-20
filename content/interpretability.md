# Interpretability vs Complexity

## The Core Trade-off

Models exist on a spectrum from highly interpretable (logistic regression, decision rules, simple scorecards) to opaque (deep neural networks, large ensembles). More flexible models generally capture more complex patterns at the cost of harder interpretation (Dianta et al., 2026; Zhang et al., 2026). This is the central tension in applied credit modelling.

## Why Credit Picks Interpretability

In lending, a model decision has real consequences. A rejected applicant in many jurisdictions has a legal right to a reason for that rejection — "adverse action notice" requirements under, for example, the US Equal Credit Opportunity Act. An institution cannot say "the neural network said no." It must point to specific reasons. Logistic regression makes this trivial: each coefficient is a reason, and the contribution of each feature to a particular decision can be read directly off the model.

Zhang et al. (2026) demonstrate this concretely. In a bond default study comparing logistic regression, random forests, and XGBoost, they find that logistic regression has the highest interpretability consistency between LIME and SHAP explanation methods, while XGBoost has the lowest. More complex models do not just have lower interpretability — their explanations are also less stable, which further undermines their use in regulated decisions.

## Explainability Tools (XAI) and Their Limits

Explainable AI (XAI) techniques such as LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations) attempt to extract reason-codes from black-box models (Dianta et al., 2026; Zhang et al., 2026). These methods can mitigate the interpretability problem but do not eliminate it. They are post-hoc approximations — explanations *of* the model, not the model itself — and their reliability depends on the underlying model class (Zhang et al., 2026). For regulated credit decisions, an intrinsically interpretable model is preferred over an explainable wrapper around an opaque one.

## When Complexity Is Worth It

This is not a blanket argument against non-linear models. Where decisions are low-stakes, where outcomes do not affect protected groups, or where retraining is cheap, the predictive gain from complexity may be worth the interpretability cost. Credit is not such a domain. The project brief enforces this constraint deliberately: the final model must be a logistic regression.
