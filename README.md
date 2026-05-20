# DataQuest 2026 — Interpretable Credit Models

Interactive credit-modelling tool built with Streamlit. Covers EDA, scorecard-style logistic regression with WoE/IV, and a business decision dashboard.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
dataquest_2026/
├── app.py                  # entry point + navigation
├── requirements.txt
├── .streamlit/config.toml  # theme
├── data/                   # loan_book.csv lives here
├── views/                  # one file per page
├── content/                # research markdown
├── core/                   # analytics (Person A)
├── components/             # reusable UI bits
└── utils/                  # cached loaders
```

## Adding Data

Place `loan_book.csv` in the `data/` directory. The data loader caches it on first load.

## Editing Research Content

Research lives in `content/*.md`. Edit the markdown directly; the Research page renders it on reload.

## For Contributors

- **App shell / views / content** — owner: Person B
- **EDA analytics, modelling, metrics in `core/`** — owner: Person A
- **Reusable components in `components/`** — shared
