"""Transform the raw panel into a modelling frame.

Applies the stationarity / leading-indicator transforms the user specified:

  - ZHVI            -> log return (target: y_t = ln(ZHVI_t / ZHVI_{t-1}))
  - MORTGAGE30US    -> first difference in pp, lagged 1
  - CPIAUCSL        -> YoY log change (inflation)
  - HOUST           -> log level, lagged 1
  - UNRATE (nat/met)-> first difference, lagged 1
  - STLFSI4         -> level (already standardised)
  - new_listings    -> log level, lagged 1
  - days_pending    -> log level, lagged 1
  - price_cuts      -> level (% already stationary-ish), lagged 1
  - ACS static      -> per-metro fixed-effect numeric + one-hot metro
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "y_logret"
LAG1_FEATURES = [
    "d_mortgage30", "d_unrate_nat", "d_unrate_metro",
    "log_houst", "log_new_listings", "log_days_pending",
    "price_cuts", "stlfsi",
]
CONTEMPORANEOUS_FEATURES = [
    "cpi_yoy",
]
STATIC_FEATURES = [
    "acs_median_income", "acs_median_home_value", "acs_vacancy_rate",
]


def _per_metro(df: pd.DataFrame, fn) -> pd.DataFrame:
    return df.groupby("metro", group_keys=False).apply(fn)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (metro, date) with target + model-ready features."""
    df = panel.sort_values(["metro", "date"]).copy()

    # --- Target: monthly log return of ZHVI ---
    df["y_logret"] = _per_metro(df, lambda g: np.log(g["zhvi"]).diff())

    # --- Stationarity transforms (applied per metro where needed) ---
    df["d_mortgage30"] = df["MORTGAGE30US"].diff()
    df["d_unrate_nat"] = df["UNRATE"].diff()
    df["d_unrate_metro"] = _per_metro(df, lambda g: g["unrate_metro"].diff())

    df["cpi_yoy"] = np.log(df["CPIAUCSL"]).diff(12)

    df["log_houst"] = np.log(df["HOUST"].where(df["HOUST"] > 0))
    df["log_new_listings"] = np.log(df["new_listings"].where(df["new_listings"] > 0))
    df["log_days_pending"] = np.log(df["days_pending"].where(df["days_pending"] > 0))

    df["price_cuts"] = df["price_cuts"]
    df["stlfsi"] = df["STLFSI4"]

    # --- Lag the leading indicators by one month per metro ---
    for col in LAG1_FEATURES:
        df[col] = _per_metro(df, lambda g, c=col: g[c].shift(1))

    # --- One-hot metro (cross-sectional fixed effect) ---
    metro_dummies = pd.get_dummies(df["metro"], prefix="metro", dtype=float)
    df = pd.concat([df, metro_dummies], axis=1)

    feature_cols = (
        LAG1_FEATURES
        + CONTEMPORANEOUS_FEATURES
        + STATIC_FEATURES
        + list(metro_dummies.columns)
    )

    keep = ["metro", "date", TARGET] + feature_cols
    out = df[keep].copy()
    return out


def chronological_split(
    feat: pd.DataFrame, train_end: str, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = feat[feat["date"] <= pd.Timestamp(train_end)].copy()
    test = feat[feat["date"] >= pd.Timestamp(test_start)].copy()
    return train, test


def feature_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if c not in {"metro", "date", TARGET}]


if __name__ == "__main__":
    from .data_loader import build_raw_panel
    panel = build_raw_panel(prefer_api=False)
    feat = build_features(panel)
    print(feat.shape)
    print(feat.head(3))
    print("\nnon-null by column:")
    print(feat.notna().sum())
