# Weight of Evidence and Information Value

## Weight of Evidence (WoE)

WoE is a transformation that re-encodes a feature so its values reflect their **strength of evidence for default**. For each bin (or category) *i*, WoE is defined as:

> WoE_i = ln( %Goods_i / %Bads_i )

where %Goods_i and %Bads_i are the share of non-defaulters and defaulters falling into bin *i* (Sharma et al., 2023; Sudjianto & Burakov, 2025).

In plain language: a positive WoE means the bin contains a higher share of good customers than bad customers, relative to the overall population, so observing a customer in this bin is evidence *against* default. A negative WoE points the other way. WoE = 0 means the bin contributes nothing to the prediction.

**Why WoE is useful in credit:**

- **It handles non-linearity inside a linear model.** A feature such as income may have a non-linear relationship with default risk. By binning income and replacing each bin with its WoE, the GLM gets to use a non-linear transformation of the original feature while remaining linear in the parameters.
- **It handles missing values gracefully.** Missing values can be treated as their own bin with its own WoE, rather than being dropped or imputed arbitrarily.
- **It produces monotonic, interpretable variables.** A well-binned WoE-encoded feature has a clear direction — higher WoE means lower risk — which makes coefficients easy to read.
- **It unifies categorical and continuous features** under one transformation framework (Sharma et al., 2023).

## Information Value (IV)

IV summarises the predictive strength of a feature in a single number by aggregating WoE across bins:

> IV = Σᵢ (%Goods_i − %Bads_i) × WoE_i

In essence, IV asks: "how much does this feature pull the distribution of good and bad customers apart?" The wider the separation, the higher the IV.

Conventional rules of thumb for interpreting IV (Sharma et al., 2023):

| IV value | Interpretation |
|---|---|
| < 0.02 | Not predictive |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| 0.30 – 0.50 | Strong |
| > 0.50 | Suspiciously strong — investigate for leakage |

The last row matters in practice. An IV above 0.5 is often a signal that the feature contains information that would not be available at the time of application — i.e., target leakage. This is a useful diagnostic to keep on hand during feature engineering.

Sudjianto and Burakov (2025) place WoE and IV on a more rigorous theoretical footing by proving that **IV is mathematically equivalent to the Population Stability Index (PSI) computed between good and bad outcomes** — and that PSI itself is the Jeffreys divergence, a symmetric extension of Kullback–Leibler divergence from information theory. This unification matters because it shows that the industry rules of thumb are not arbitrary: they correspond to well-defined information-theoretic quantities, and they admit formal statistical inference such as confidence intervals on IV (Sudjianto & Burakov, 2025).

## Why WoE and IV Are the Backbone of Credit Scorecards

WoE and IV remain dominant in regulated credit modelling because they combine three properties that few alternatives match: they are theoretically grounded, they preserve interpretability of the underlying GLM, and they produce features that regulators recognise (Sudjianto & Burakov, 2025; Sharma et al., 2023). Traditional credit scorecards are essentially logistic regressions on WoE-transformed variables, often translated into a point-based score system that loan officers can read at a glance.
