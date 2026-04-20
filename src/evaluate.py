"""Evaluation: per-metro MAE/RMSE and SHAP feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .models import Predictions


def error_metrics(preds: Predictions) -> pd.DataFrame:
    """MAE / RMSE on log returns and on dollar-value ZHVI changes."""
    df = preds.frame.copy()
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    df["sq_err"] = (df["y_true"] - df["y_pred"]) ** 2

    rows = []
    for metro, g in df.groupby("metro"):
        rows.append({
            "model": preds.name,
            "metro": metro,
            "n": len(g),
            "MAE_logret": g["abs_err"].mean(),
            "RMSE_logret": np.sqrt(g["sq_err"].mean()),
        })
    rows.append({
        "model": preds.name,
        "metro": "ALL",
        "n": len(df),
        "MAE_logret": df["abs_err"].mean(),
        "RMSE_logret": np.sqrt(df["sq_err"].mean()),
    })
    return pd.DataFrame(rows)


def attach_dollar_errors(preds: Predictions, panel: pd.DataFrame) -> pd.DataFrame:
    """Convert log-return errors into dollar-ZHVI errors using previous-month level.

    dollar_err_t ≈ ZHVI_{t-1} * (exp(y_pred) - exp(y_true))
    """
    df = preds.frame.merge(
        panel[["metro", "date", "zhvi"]], on=["metro", "date"], how="left"
    )
    df = df.sort_values(["metro", "date"])
    df["zhvi_prev"] = df.groupby("metro")["zhvi"].shift(1)
    df["zhvi_pred"] = df["zhvi_prev"] * np.exp(df["y_pred"])
    df["zhvi_true"] = df["zhvi_prev"] * np.exp(df["y_true"])
    df["dollar_abs_err"] = (df["zhvi_pred"] - df["zhvi_true"]).abs()
    return df


def dollar_mae(preds: Predictions, panel: pd.DataFrame) -> pd.DataFrame:
    df = attach_dollar_errors(preds, panel).dropna(subset=["dollar_abs_err"])
    out = df.groupby("metro")["dollar_abs_err"].mean().rename("MAE_dollars").reset_index()
    out["model"] = preds.name
    return out[["model", "metro", "MAE_dollars"]]


def shap_per_metro(
    model: xgb.XGBRegressor,
    test: pd.DataFrame,
    feats: list[str],
) -> dict[str, pd.Series]:
    """Compute mean |SHAP| per feature, separately for each metro on the test set."""
    explainer = shap.TreeExplainer(model)
    X = test[feats].astype(float)
    sv = explainer.shap_values(X)
    sv_df = pd.DataFrame(sv, columns=feats, index=test.index)
    sv_df["metro"] = test["metro"].values

    out = {}
    for metro, g in sv_df.groupby("metro"):
        imp = g[feats].abs().mean().sort_values(ascending=False)
        out[metro] = imp
    return out


def shap_summary_table(shap_by_metro: dict[str, pd.Series], top_k: int = 10) -> pd.DataFrame:
    """Wide table: one row per feature, one column per metro, ranked by total impact."""
    df = pd.DataFrame(shap_by_metro)
    df["total"] = df.sum(axis=1)
    df = df.sort_values("total", ascending=False).drop(columns="total").head(top_k)
    return df
