"""
Central modelling pipeline. Port of Person A's fnb_app.py analytics.

load_and_engineer() — feature engineering on raw loan book
train_models()      — fits baseline + improved logistic regression

Both functions are cached so the app only runs them once per session.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler


# ── Feature lists ──────────────────────────────────────────────────────────
NUMERIC_BASELINE = [
    "age", "annual_income", "employment_length_years", "num_open_accounts",
    "num_delinquencies_2yr", "total_revolving_balance", "credit_utilisation_pct",
    "months_since_oldest_account", "num_hard_inquiries_6mo", "loan_amount",
    "interest_rate", "dti_ratio", "months_since_last_delinquency",
    "pct_accounts_current", "months_at_current_address",
]

ENGINEERED = [
    "age", "log_annual_income", "employment_length_years", "num_open_accounts",
    "num_delinquencies_2yr", "log_revolving", "credit_utilisation_capped",
    "months_since_oldest_account", "num_hard_inquiries_6mo", "log_loan_amount",
    "interest_rate", "dti_ratio", "months_since_last_delinquency_filled",
    "pct_accounts_current", "months_at_current_address", "ever_delinquent",
    "loan_to_income", "log_income_per_acct", "delinquency_rate",
    "monthly_debt_burden", "high_inquiries", "high_utilisation", "high_dti",
    "many_delinquencies", "low_pct_current", "new_credit_history",
    "short_employment", "dti_x_utilisation", "delinq_x_inquiries",
    "interest_x_dti", "app_month", "app_quarter",
]

ENGINEERING_DECISIONS = [
    ("Missing delinquency → binary flag + fill 999",
     "49.9% missing = never delinquent. Default rate 8% vs 23%.",
     "Largest single AUC gain"),
    ("Log transform: income, loan amount, revolving balance",
     "Right-skewed numerics; log makes relationship more linear.",
     "Better logistic fit"),
    ("Loan-to-income ratio",
     "Raw loan amount ignores affordability context.",
     "Added ratio feature"),
    ("Delinquency rate (delinquencies / accounts)",
     "Normalises for portfolio size; 2 delinquencies on 2 accounts > 2 on 20.",
     "Better risk signal"),
    ("High utilisation flag (≥70%)",
     "Non-linear: risk spikes above 70% threshold.",
     "Captures threshold effect"),
    ("Cap credit utilisation at 100%",
     "19 rows above 100% are data errors; capping prevents distortion.",
     "Data quality fix"),
    ("Interaction: DTI × utilisation",
     "Both high together signals severe overextension.",
     "Captures combined risk"),
    ("Interaction: interest rate × DTI",
     "High rate on high debt burden = compounding risk.",
     "Domain knowledge"),
    ("Standardise categoricals (home_ownership, purpose)",
     "14 and 21 raw variants collapsed to 4 and 7.",
     "Prevents category leakage"),
    ("Parse mixed date formats → month / quarter",
     "24k rows had MM/DD/YYYY vs YYYY-MM-DD; seasonal features extracted.",
     "Data quality + new features"),
]


# ── Feature engineering ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Engineering features...")
def load_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply Person A's full feature engineering pipeline to the raw loan book."""
    d = df_raw.copy()

    # Parse dates (mixed format in the data)
    d["application_date"] = pd.to_datetime(
        d["application_date"], format="mixed", errors="coerce"
    )

    # Log transforms
    d["log_annual_income"]   = np.log1p(d["annual_income"])
    d["log_revolving"]       = np.log1p(d["total_revolving_balance"])
    d["log_loan_amount"]     = np.log1p(d["loan_amount"])
    d["log_income_per_acct"] = np.log1p(
        d["annual_income"] / d["num_open_accounts"].replace(0, np.nan)
    )

    # Cap utilisation at 100 (19 error rows)
    d["credit_utilisation_capped"] = d["credit_utilisation_pct"].clip(upper=100)

    # Delinquency signal — 49.9% missing = never delinquent
    d["months_since_last_delinquency_filled"] = d["months_since_last_delinquency"].fillna(999)
    d["ever_delinquent"] = d["months_since_last_delinquency"].notna().astype(int)

    # Ratio features
    d["loan_to_income"]      = d["loan_amount"] / d["annual_income"].replace(0, np.nan)
    d["delinquency_rate"]    = d["num_delinquencies_2yr"] / d["num_open_accounts"].replace(0, np.nan)
    d["monthly_debt_burden"] = d["dti_ratio"] * d["annual_income"] / 12

    # Binary threshold flags
    d["high_inquiries"]     = (d["num_hard_inquiries_6mo"]    >= 3).astype(int)
    d["high_utilisation"]   = (d["credit_utilisation_capped"] >= 70).astype(int)
    d["high_dti"]           = (d["dti_ratio"]                 >= 0.40).astype(int)
    d["many_delinquencies"] = (d["num_delinquencies_2yr"]     >= 2).astype(int)
    d["low_pct_current"]    = (d["pct_accounts_current"]      < 80).astype(int)
    d["new_credit_history"] = (d["months_since_oldest_account"] < 60).astype(int)
    d["short_employment"]   = (d["employment_length_years"]   < 2).astype(int)

    # Interactions
    d["dti_x_utilisation"]  = d["dti_ratio"] * d["credit_utilisation_capped"]
    d["delinq_x_inquiries"] = d["num_delinquencies_2yr"] * d["num_hard_inquiries_6mo"]
    d["interest_x_dti"]     = d["interest_rate"] * d["dti_ratio"]

    # Temporal features
    d["app_month"]   = d["application_date"].dt.month.fillna(0).astype(int)
    d["app_quarter"] = d["application_date"].dt.quarter.fillna(0).astype(int)

    # Standardise categoricals before one-hot
    d["loan_purpose"] = (
        d["loan_purpose"].astype(str).str.strip().str.lower()
         .str.replace(r"[\s\-]+", "_", regex=True)
    )
    d["home_ownership"] = d["home_ownership"].astype(str).str.strip().str.upper()

    # One-hot encode
    d = pd.get_dummies(
        d, columns=["home_ownership", "loan_purpose", "email_domain_type"],
        drop_first=False
    )
    return d


# ── Model training ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models — this takes ~15 seconds...")
def train_models(df_raw: pd.DataFrame) -> dict:
    """
    Fit baseline and improved logistic regression on Person A's engineered data.

    Returns a dict with AUC scores, predictions, coefficients, and fitted objects
    needed by the views.
    """
    d = load_and_engineer(df_raw)
    train = d[d["set"] == "train"].copy()
    test  = d[d["set"] == "test"].copy()
    y_train, y_test = train["default_flag"], test["default_flag"]

    dummy_cols = [
        c for c in train.columns
        if any(c.startswith(p) for p in ["home_ownership_", "loan_purpose_", "email_domain_type_"])
    ]
    all_eng = ENGINEERED + dummy_cols
    # Ensure test has same columns as train
    for col in all_eng:
        if col not in test.columns:
            test[col] = 0

    # ── Baseline: raw numerics only ────────────────────────────────────────
    imp1 = SimpleImputer(strategy="mean")
    sc1  = StandardScaler()
    Xtr1 = sc1.fit_transform(imp1.fit_transform(train[NUMERIC_BASELINE]))
    Xte1 = sc1.transform(imp1.transform(test[NUMERIC_BASELINE]))
    lr1  = LogisticRegression(max_iter=1000, random_state=42)
    lr1.fit(Xtr1, y_train)
    base_train_auc = roc_auc_score(y_train, lr1.predict_proba(Xtr1)[:, 1])
    base_test_auc  = roc_auc_score(y_test,  lr1.predict_proba(Xte1)[:, 1])

    # ── Improved: engineered features ──────────────────────────────────────
    sc2  = StandardScaler()
    Xtr2 = sc2.fit_transform(train[all_eng].fillna(0))
    Xte2 = sc2.transform(test[all_eng].fillna(0))
    lr2  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr2.fit(Xtr2, y_train)
    impr_train_auc = roc_auc_score(y_train, lr2.predict_proba(Xtr2)[:, 1])
    impr_test_auc  = roc_auc_score(y_test,  lr2.predict_proba(Xte2)[:, 1])

    probs_test = lr2.predict_proba(Xte2)[:, 1]

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, probs_test)

    # Precision-recall curve data
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs_test)

    # Coefficients
    coef_df = (
        pd.DataFrame({"feature": all_eng, "coef": lr2.coef_[0]})
          .assign(abs_coef=lambda x: x["coef"].abs())
          .sort_values("abs_coef", ascending=False)
          .head(20)
          .reset_index(drop=True)
    )
    coef_df["direction"] = coef_df["coef"].apply(
        lambda x: "↑ risk" if x > 0 else "↓ risk"
    )

    # Approval strategy curve
    thresholds_curve = np.linspace(0.05, 0.95, 91)
    vol, risk_arr, avoided = [], [], []
    y_arr = y_test.values
    for t in thresholds_curve:
        app = probs_test < t
        vol.append(app.sum() / len(y_arr))
        risk_arr.append((app & (y_arr == 1)).sum() / max(app.sum(), 1))
        avoided.append((~app & (y_arr == 1)).sum() / max(y_arr.sum(), 1))

    return {
        "base_train_auc":  base_train_auc,
        "base_test_auc":   base_test_auc,
        "impr_train_auc":  impr_train_auc,
        "impr_test_auc":   impr_test_auc,
        "probs_test":      probs_test,
        "y_test":          y_test.values,
        "fpr":             fpr,
        "tpr":             tpr,
        "prec_arr":        prec_arr,
        "rec_arr":         rec_arr,
        "coef_df":         coef_df,
        "thresholds_curve": thresholds_curve,
        "vol_curve":       np.array(vol),
        "risk_curve":      np.array(risk_arr),
        "avoided_curve":   np.array(avoided),
        "avg_loan":        float(df_raw[df_raw["set"] == "test"]["loan_amount"].mean()),
        "n_test":          len(y_test),
        "feature_names":   all_eng,
    }
