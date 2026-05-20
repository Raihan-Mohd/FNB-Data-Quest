"""Cached data loaders. Centralises file paths and caching policy."""
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOAN_BOOK_PATH = DATA_DIR / "loan_book.csv"


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
