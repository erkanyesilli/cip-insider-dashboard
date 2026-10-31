# Insider Purchases and Stock Returns: Dashboard

Streamlit dashboard for the CIP project of group 201 (HSLU, MSc Applied Data Science, HS2026).
It shows how stocks perform after corporate insiders publicly report open-market purchases (SEC Form 4, code P).

## What it shows

| Tab | Content |
|---|---|
| Overview | Key figures, median and trimmed mean path of the excess return from 20 trading days before to 63 after entry, distribution |
| Comparison (drill-down) | Median with 95% CI and median paths by role, cluster vs. isolated purchase, purchase value, price level, quarter, 10b5-1 plan |
| Single company | Stock vs. SPY with all insider purchases, table of who bought when |
| Data & method | Cleaning steps from 585,091 raw rows to 16,296 filings, Yahoo data issue, method |

## Data

- **SEC Insider Transactions Data Sets**, 2024 Q1–2025 Q4 (Form 4, code P)
- **Yahoo Finance** daily adjusted close prices (via `yfinance`), SPY as market benchmark

`data/` contains pre-aggregated files built from the output of the group's pipeline
(repository `CIP_HS2026_201`, scripts `02_load_data.py` → `05_ResearchQuestion.py`).
The app itself only filters and aggregates (vectorized with NumPy/pandas).

## Method in brief

- **Event** = one Form 4 filing (several trades in one filing count once)
- **Entry** = close on the first trading day after the filing (no look-ahead)
- **Excess return** = stock return − SPY return over 30 / 90 calendar days
- **Cluster** = at least 2 different insiders of the same company filed within the 30 days up to the filing
- Trimmed mean (5% on each side) as in the group's analysis, median confidence intervals from order statistics

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run app.py
```
