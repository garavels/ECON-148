"""Baseline (OLS, ARIMA) and ML (XGBoost) nowcasting models.

All models output predictions of the monthly ZHVI log return `y_logret`
aligned on (metro, date) so they share a single evaluation harness.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET, feature_columns


@dataclass
class Predictions:
    """Aligned test-set predictions from one model."""
    name: str
    frame: pd.DataFrame  # columns: metro, date, y_true, y_pred


def _drop_na(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    tr = train.dropna(subset=cols + [TARGET]).copy()
    te = test.dropna(subset=cols + [TARGET]).copy()
    return tr, te


# ---------- OLS (pooled, with metro dummies) ----------

def fit_predict_ols(train: pd.DataFrame, test: pd.DataFrame) -> Predictions:
    feats = feature_columns(train)
    tr, te = _drop_na(train, test, feats)
    X_tr = sm.add_constant(tr[feats].astype(float), has_constant="add")
    X_te = sm.add_constant(te[feats].astype(float), has_constant="add")
    model = sm.OLS(tr[TARGET].astype(float), X_tr).fit()
    y_pred = model.predict(X_te)
    out = te[["metro", "date"]].copy()
    out["y_true"] = te[TARGET].values
    out["y_pred"] = y_pred.values
    return Predictions("OLS", out)


# ---------- ARIMA (per metro, univariate) ----------

def fit_predict_arima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 1),
) -> Predictions:
    preds = []
    for metro in sorted(train["metro"].unique()):
        tr = train[train["metro"] == metro].dropna(subset=[TARGET]).sort_values("date")
        te = test[test["metro"] == metro].dropna(subset=[TARGET]).sort_values("date")
        if len(tr) < 24 or te.empty:
            continue
        y_train = tr[TARGET].astype(float).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(y_train, order=order).fit()
            fc = fit.forecast(steps=len(te))
        block = te[["metro", "date"]].copy()
        block["y_true"] = te[TARGET].values
        block["y_pred"] = np.asarray(fc)
        preds.append(block)
    return Predictions(f"ARIMA{order}", pd.concat(preds, ignore_index=True))


# ---------- XGBoost (pooled, with metro dummies + static ACS) ----------

def fit_predict_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict | None = None,
    seed: int = 0,
) -> tuple[Predictions, xgb.XGBRegressor, list[str]]:
    feats = feature_columns(train)
    tr, te = _drop_na(train, test, feats)
    params = params or dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(tr[feats].astype(float), tr[TARGET].astype(float))
    y_pred = model.predict(te[feats].astype(float))
    out = te[["metro", "date"]].copy()
    out["y_true"] = te[TARGET].values
    out["y_pred"] = y_pred
    return Predictions("XGBoost", out), model, feats


# ---------- 2-layer MLP (pooled, with standardisation) ----------

def fit_predict_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hidden: tuple[int, int] = (16, 8),
    alpha: float = 1e-1,
    seed: int = 0,
) -> tuple[Predictions, TransformedTargetRegressor, list[str]]:
    """Feed-forward net with 2 hidden layers.

    Both inputs and target are standardised — log returns have std ~0.005 and
    ACS features have scale ~1e6, so without scaling the optimiser can't
    converge to a useful solution. Heavy L2 (alpha=0.1) plus a small net
    prevents extrapolation blow-ups in the 2023+ rate regime.
    """
    feats = feature_columns(train)
    tr, te = _drop_na(train, test, feats)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=3000,
            tol=1e-6,
            random_state=seed,
        )),
    ])
    model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    model.fit(tr[feats].astype(float), tr[TARGET].astype(float))
    y_pred = model.predict(te[feats].astype(float))
    out = te[["metro", "date"]].copy()
    out["y_true"] = te[TARGET].values
    out["y_pred"] = y_pred
    return Predictions(f"MLP{hidden}", out), model, feats
