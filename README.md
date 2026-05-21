# DataQuest 2026 — Interpretable Credit Models

Interactive credit-modelling tool built with Streamlit. Covers EDA, scorecard-style logistic regression with WoE/IV, and a business decision dashboard.

## Model Results

| Model | AUC |
|---|---|
| Given baseline (competition) | 0.68 |
| **Our logistic regression** | **0.8064** |
| Reference LightGBM ceiling | 0.82 |

Our improved logistic regression closes **~92% of the gap** between the baseline and the LightGBM ceiling, using only feature engineering — no non-linear models.

Key engineering wins:
- Treating missing delinquency as a signal (`ever_delinquent` flag + fill with 999)
- Log transforms on skewed income / loan / balance features
- Risk threshold flags (high DTI, high utilisation ≥ 70%)
- Interaction terms: DTI × utilisation, interest rate × DTI
- Derived ratios: loan-to-income, delinquency rate per account

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Important:** `loan_book.csv` must be in the project root (same folder as `app.py`). It is already there — do not move it.

## App Structure

Six tabs, covering every deliverable in the competition brief:

| Tab | Contents |
|---|---|
| 📋 Data Quality | Missing values, categorical inconsistencies, date format issues, outliers |
| 🔍 Univariate Explorer | Distribution by default, WoE chart, default-rate chart, IV leaderboard |
| 🔗 Bivariate Explorer | Scatter / heatmap of any two features, full correlation matrix |
| 🤖 Model Performance | ROC curve, top-20 feature coefficients, engineering decisions table |
| 📚 Research | GLMs vs non-linear models, WoE/IV explanation, metrics guide, regulatory notes |
| 💼 Business Dashboard | Approval threshold slider, volume vs risk curve, precision–recall curve |

## Project Layout

```
FNB-Data-Quest/
├── app.py              # single entry point — all tabs live here
├── fnb_app.py          # original working draft (kept for reference)
├── requirements.txt
├── loan_book.csv       # dataset (project root)
├── .venv/              # local virtual environment
├── .streamlit/         # theme config
├── views/              # legacy multi-page shell (superseded by app.py)
├── core/               # legacy analytics modules (superseded by app.py)
├── content/            # research markdown (superseded by app.py)
├── components/         # reusable UI bits
└── utils/              # cached loaders
```

## Common Gotchas

**`streamlit: command not found`** — the venv isn't activated. Re-run `source .venv/bin/activate` then try again.

**`ModuleNotFoundError`** — run `pip install -r requirements.txt` inside the activated venv.

**Data file not found** — `loan_book.csv` must be in the project root, not inside `data/`. The path is resolved relative to `app.py` at startup.

**Port already in use** — another Streamlit process is running. Stop it with `Ctrl+C`, or run `streamlit run app.py --server.port 8502`.

## Deliverables Checklist

- [x] Interactive EDA tool (univariate + bivariate + data quality)
- [x] WoE and IV for all numeric features
- [x] Improved logistic regression (AUC 0.8064 vs 0.68 baseline)
- [x] Research section (GLMs, WoE/IV, metrics, regulatory concerns)
- [x] Business decision dashboard (threshold slider, volume/risk, precision/recall)
- [x] Feature engineering justification table
- [x] Reproducible single-command run
