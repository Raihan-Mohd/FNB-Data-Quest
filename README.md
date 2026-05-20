# DataQuest 2026 — Interpretable Credit Models

Interactive credit-modelling tool built with Streamlit. Covers EDA, scorecard-style logistic regression with WoE/IV, and a business decision dashboard.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

Run this command in vs code terminal before running the webapp

```bash
python3 -m venv .venv
source .venv/bin/activate
```
You'll know it worked when your terminal prompt is prefixed with (.venv).

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

Common gotchas

streamlit: command not found → the venv isn't activated, or pip install failed. Re-activate (source .venv/bin/activate or the PowerShell equivalent) and re-run pip install -r requirements.txt.

ModuleNotFoundError: No module named 'views' → you're running from the wrong directory. Make sure your terminal is inside dataquest_2026/, not its parent.

Data file not found error in the app → the CSV isn't in data/loan_book.csv. Check the path exactly.


Port already in use → another Streamlit process is running. Stop it with Ctrl+C in its terminal, or run streamlit run app.py --server.port 8502 to use a different port.
