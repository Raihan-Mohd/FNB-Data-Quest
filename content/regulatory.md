# Regulatory Considerations for Feature Selection

The dataset for this project is simulated, but that does not exempt the model from regulatory scrutiny. Regulators (the US Consumer Financial Protection Bureau, the UK Financial Conduct Authority, GDPR authorities in the EU, and equivalent bodies under Basel frameworks) evaluate not just predictive power, but also fairness, transparency, governance, and the economic justification for each feature (Sudjianto & Burakov, 2025; Dianta et al., 2026). A feature that improves AUC but fails these tests is unlikely to survive validation, regardless of the data source.

## Protected Attributes and Their Proxies

Direct demographic identifiers — race, ethnicity, religion, gender, nationality, disability, marital status — are unambiguously prohibited in credit decisions in most jurisdictions. Less obvious but equally problematic are **proxy features** that correlate strongly with protected attributes: postcode, school attended, surname, browsing behaviour, device type, language preference. These can produce **disparate impact** even when the protected attribute is never explicitly used (Sudjianto & Burakov, 2025).

Postcode is the canonical example. In many countries it correlates closely with race and socio-economic status, which means a model that uses postcode is implicitly using race. The fact that the data is simulated does not change the methodological point: building a model around proxies normalises a pattern that would be discriminatory in production.

## Target Leakage and Look-Ahead Bias

A feature has **target leakage** when it contains information that would not actually be available at the time of the decision. Examples include "late fee paid," "collections activity," or any post-issuance behaviour. Including these inflates AUC artificially and produces a model that fails the moment it is deployed.

**Look-ahead bias** is the timing version of leakage: using future information to predict past events. Even in simulated data, careless timestamp alignment can produce this problem. Regulators treat leakage as a governance failure regardless of intent.

## Invasive Behavioural and Non-Economic Features

Phone battery level, GPS movement patterns, typing speed, contact-list size, time of application, screen brightness — these may correlate with default outcomes statistically, but they fail on three grounds: weak causal connection, privacy concerns, and lack of business justification. A regulator asks: "why should phone battery level determine loan approval?" If the answer is "because it correlates," that is not enough.

The same applies to features that are merely statistically predictive but economically irrational — favourite colour, horoscope, mouse-movement patterns. Pure correlation without economic explanation is increasingly rejected.

## Personally Identifiable Information (PII)

Even in a simulated dataset, features such as names, social security numbers, device IDs, or anything that could be used to re-identify a real person are problematic under GDPR, CCPA, and equivalent frameworks. Regulators prefer aggregated or bucketed forms ("age 25–34" rather than exact date of birth; "income band" rather than exact income).

## Opaque Feature Constructions

Features built from latent embeddings, deep neural-net outputs, or aggressive automatic feature engineering may capture useful signal but are difficult to validate and monitor (Dianta et al., 2026). Regulators prefer parsimonious, intuitively-named features that map to underlying economic concepts. This is part of why WoE-encoded variables — themselves transformations, but transparent ones — remain acceptable while neural-net embeddings do not.

## Features Regulators Generally Accept

Where the feature-engineering effort should focus:

- Income, debt-to-income ratio
- Repayment history, delinquency counts
- Credit utilisation, credit age
- Number of recent hard inquiries
- Verified employment information
- Loan amount, term, purpose

These are economically explainable, historically validated, auditable, and easy to justify to a regulator. They are also the natural candidates for WoE transformation in a logistic-regression scorecard.

## Implications for the Loan Book Data

Several columns in this project's dataset require explicit treatment:

- **`age`** — Permitted in some jurisdictions but restricted in others. Should be discussed transparently and tested for disparate impact.
- **`region`** — A clear postcode-style proxy. High disparate-impact risk. Either drop or test for fairness with explicit evidence.
- **`email_domain_type`, `phone_verified`** — Borderline. Defensible only as fraud signals; should be reviewed.
- **`branch_code_id`** — Geographic proxy concerns similar to `region`.
- **`application_dow`** — No economic rationale. Should be excluded.
- **`months_at_current_address`** — Acceptable. Linked to residential stability.

The choice of which features to include is itself a defensible governance decision, and the model documentation should explain why each questionable feature was kept or removed.
