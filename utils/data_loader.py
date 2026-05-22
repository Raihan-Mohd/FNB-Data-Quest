"""Cached data loaders. Centralises file paths and caching policy."""
from pathlib import Path
import pandas as pd
import streamlit as st

_DISPLAY_NAMES: dict[str, str] = {
    "applicant_id_hash": "Applicant ID",
    "age": "Age",
    "annual_income": "Annual Income",
    "employment_length_years": "Employment Length (Years)",
    "home_ownership": "Home Ownership",
    "region": "Region",
    "num_open_accounts": "Number of Open Accounts",
    "num_delinquencies_2yr": "Delinquencies (2 Years)",
    "total_revolving_balance": "Total Revolving Balance",
    "credit_utilisation_pct": "Credit Utilisation (%)",
    "months_since_oldest_account": "Months Since Oldest Account",
    "num_hard_inquiries_6mo": "Hard Inquiries (6 Months)",
    "loan_amount": "Loan Amount",
    "interest_rate": "Interest Rate",
    "loan_purpose": "Loan Purpose",
    "dti_ratio": "DTI Ratio",
    "months_since_last_delinquency": "Months Since Last Delinquency",
    "pct_accounts_current": "% Accounts Current",
    "application_date": "Application Date",
    "application_dow": "Application Day of Week",
    "branch_code_id": "Branch Code",
    "months_at_current_address": "Months at Current Address",
    "email_domain_type": "Email Domain Type",
    "phone_verified": "Phone Verified",
    "default_flag": "Default Flag",
    "set": "Set",
    "credit_utilisation_capped": "Credit Utilisation (Capped)",
    "ever_delinquent": "Ever Delinquent",
    "months_since_last_delinquency_filled": "Months Since Last Delinquency",
    "loan_to_income": "Loan to Income Ratio",
    "income_per_account": "Income per Account",
    "delinquency_rate": "Delinquency Rate",
    "monthly_debt_burden": "Monthly Debt Burden",
    "log_annual_income": "Annual Income (Log)",
    "log_loan_amount": "Loan Amount (Log)",
    "log_revolving": "Total Revolving Balance (Log)",
    "log_income_per_acct": "Income per Account (Log)",
    "high_inquiries": "High Inquiries Flag",
    "high_utilisation": "High Utilisation Flag",
    "high_dti": "High DTI Flag",
    "many_delinquencies": "Many Delinquencies Flag",
    "low_pct_current": "Low % Accounts Current Flag",
    "new_credit_history": "New Credit History Flag",
    "short_employment": "Short Employment Flag",
    "dti_x_utilisation": "DTI x Utilisation",
    "delinq_x_inquiries": "Delinquencies x Inquiries",
    "interest_x_dti": "Interest Rate x DTI",
}


def col_label(name: str) -> str:
    """Return a human-readable display name for a DataFrame column."""
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())

LOAN_BOOK_PATH = Path(__file__).resolve().parent.parent / "loan_book.csv"


@st.cache_data(show_spinner="Loading loan book...")
def load_loan_book() -> pd.DataFrame:
    """Load the full loan book CSV. Cached for the session."""
    if not LOAN_BOOK_PATH.exists():
        st.error(f"Data file not found at {LOAN_BOOK_PATH}. Place loan_book.csv in /data.")
        st.stop()
    df = pd.read_csv(LOAN_BOOK_PATH, parse_dates=["application_date"])
    return df


@st.cache_data
def get_train_test_split(df: pd.DataFrame):
    """Split the dataframe based on the existing 'set' column."""
    train = df[df["set"] == "train"].copy()
    test = df[df["set"] == "test"].copy()
    return train, test


@st.cache_data
def get_feature_columns() -> dict:
    """Return feature columns grouped by type. Used by view modules to populate selectors."""
    return {
        "numeric": [
            "age", "annual_income", "employment_length_years", "num_open_accounts",
            "num_delinquencies_2yr", "total_revolving_balance", "credit_utilisation_pct",
            "months_since_oldest_account", "num_hard_inquiries_6mo", "loan_amount",
            "interest_rate", "dti_ratio", "months_since_last_delinquency",
            "pct_accounts_current", "months_at_current_address",
        ],
        "categorical": [
            "home_ownership", "region", "loan_purpose", "application_dow",
            "branch_code_id", "email_domain_type", "phone_verified",
        ],
        "target": "default_flag",
        "id": "applicant_id_hash",
        "date": "application_date",
    }
