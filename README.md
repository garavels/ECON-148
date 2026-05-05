<p align="center">
  <img src="assets/logo.png" alt="ECON 148 — Local Housing Price Nowcasting" width="260" />
</p>

<h1 align="center">Local Housing Price Nowcasting</h1>
<p align="center"><em>Econ 148 — Data Science for Economists · UC Berkeley</em></p>

---

## Overview

This project builds a **monthly nowcast of local house-price changes** for three structurally different U.S. metros — **San Francisco, Austin, and Cleveland** — and tests whether non-linear models extract signal beyond classical baselines.

**Target.** Monthly log return of the Zillow Home Value Index (ZHVI):

$$y_t = \ln\!\left(\frac{\text{ZHVI}_t}{\text{ZHVI}_{t-1}}\right)$$

Log returns make the target stationary and comparable across metros with very different price levels.

**Models compared**

| Family | Model | Role |
|---|---|---|
| Time series | ARIMA (per metro) | Univariate baseline |
| Linear | OLS with metro dummies | Pooled linear baseline |
| Tree ensemble | XGBoost | Non-linear pooled model |
| Neural net | 2-layer MLP (16 → 8, ReLU) | Non-linear pooled model |

A single chronological split (**train ≤ 2022-12**, **test 2023-01 → latest**) means the test window covers the post-2023 rate-shock regime — a genuine out-of-distribution stress test.

## Data Sources

| Series | Source | Frequency | Purpose |
|---|---|---|---|
| ZHVI, new listings, days-to-pending, price cuts | [Zillow Research](https://www.zillow.com/research/data/) | Monthly (metro) | Local housing market dynamics |
| 30-yr mortgage rate, UNRATE, CPI, housing starts, STLFSI | [FRED](https://fred.stlouisfed.org/) | Weekly / monthly | National macro context |
| Metro-level unemployment (LAUS, SA) | FRED | Monthly (metro) | Local labor market health |
| Median income, median home value, vacancy rate | [Census ACS 5-yr 2024](https://www.census.gov/programs-surveys/acs) | Cross-section | Per-metro fixed effects |

Leading indicators are **lagged one month** so the model only uses information available at nowcast time. The 2024 ACS enters as a per-metro structural bias (constant within city), so it cannot leak across the train/test boundary.

## Repository Structure

```
ECON-148/
├── notebooks/
│   └── housing_nowcast_main.ipynb   # End-to-end pipeline (load → features → models → eval)
├── data/
│   ├── raw/                         # Zillow, FRED, ACS source CSVs
│   └── cleaned/                     # Processed panels
├── outputs/
│   ├── figures/                     # ZHVI trajectories, SHAP plots, prediction charts, etc.
│   └── tables/                      # Error metrics, top SHAP features
├── writeup/                         # Final report
└── assets/                          # README assets
```

## Pipeline (Notebook Sections)

0. **Setup** — libraries, paths, seed, metro/date config
1. **Load raw data** — Zillow metros, FRED macros, metro LAUS, ACS snapshot
2. **Feature engineering** — log returns, lags, metro fixed effects
3. **Train / test split** — chronological, no leakage
4. **ARIMA baseline** (univariate, per metro)
5. **OLS baseline** (pooled, with metro dummies)
6. **XGBoost** (non-linear, pooled)
7. **Two-layer MLP** (16 → 8 ReLU)
8. **Evaluation** — MAE in log-returns and dollars
9. **Economic interpretability** — SHAP per metro, feature importance
10. **Summary and interpretation**

## Reproducibility

- `SEED = 0` is passed to every stochastic operation.
- Macro series prefer a live `pandas_datareader` pull, with a fallback to `data/raw/` CSVs for full offline reproducibility.
- Paths resolve from the repository root, so the notebook runs identically regardless of working directory.

## Running

```bash
git clone https://github.com/garavels/ECON-148.git
cd ECON-148
jupyter lab notebooks/housing_nowcast_main.ipynb
```

The first cell installs dependencies:

```bash
pip install -q pandas_datareader shap xgboost scikit-learn statsmodels
```

Then **Run All**.

## Key Outputs

Selected figures saved to [`outputs/figures/`](outputs/figures):

- `zhvi_levels.png` — ZHVI trajectories by metro
- `predictions_vs_actual.png` — model predictions vs realized returns
- `mae_dollars_bar.png` — head-to-head error comparison
- `shap_per_metro.png`, `shap_heatmap.png` — feature attribution by city
- `xgb_feature_importance.png` — global feature ranking

Error tables in [`outputs/tables/`](outputs/tables): `errors_logret.csv`, `mae_dollars.csv`, `shap_top_features.csv`.
