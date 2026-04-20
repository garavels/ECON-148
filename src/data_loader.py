"""Load Zillow + FRED data into tidy monthly panels.

Prefers pandas_datareader (keyless FRED) for fresh pulls; falls back to the
raw CSVs in data/raw when the network is unavailable.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_RAW, METROS, ZILLOW_FILES, FRED_NATIONAL,
    START, END,
)


# ---------- Zillow metro series ----------

def _load_zillow_wide(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    date_cols = [c for c in df.columns if re.match(r"\d{4}-\d{2}-\d{2}", c)]
    long = df.melt(id_vars=id_cols, value_vars=date_cols,
                   var_name="date", value_name=value_name)
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp("M")
    return long[["RegionName", "date", value_name]]


def load_zillow_panel(metros: list[str] | None = None) -> pd.DataFrame:
    """Return a long panel (metro, date) with all four Zillow series joined."""
    metros = metros or list(METROS.keys())
    regions = [METROS[m]["zillow_region"] for m in metros]

    frames = []
    for name, fname in ZILLOW_FILES.items():
        df = _load_zillow_wide(DATA_RAW / fname, value_name=name)
        df = df[df["RegionName"].isin(regions)]
        frames.append(df)

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["RegionName", "date"], how="outer")

    out = out.rename(columns={"RegionName": "metro"}).sort_values(["metro", "date"])
    return out.reset_index(drop=True)


# ---------- FRED macro series ----------

def _load_fred_csv(path: Path, series: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date"}).set_index("date")
    # FRED csvs sometimes use "." for missing
    s = pd.to_numeric(df[series], errors="coerce")
    return s.rename(series)


def _to_monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    """Resample any frequency to monthly (month-end)."""
    m = s.resample("ME")
    if how == "mean":
        return m.mean()
    if how == "last":
        return m.last()
    raise ValueError(how)


def load_fred_national(prefer_api: bool = True) -> pd.DataFrame:
    """Load the five national FRED series and resample to monthly.

    CSVs in data/raw are the snapshot source; if prefer_api and network is
    available, we refresh via pandas_datareader. Both paths end at the same
    tidy monthly DataFrame keyed on date.
    """
    series_map = dict(FRED_NATIONAL)  # {code: csv_filename}
    frames: dict[str, pd.Series] = {}

    if prefer_api:
        try:
            from pandas_datareader import data as pdr
            for code in series_map:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = pdr.DataReader(code, "fred", START, END)[code]
                frames[code] = s
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"FRED API fetch failed ({e}); falling back to CSVs.")
            frames = {}

    if not frames:
        for code, fname in series_map.items():
            frames[code] = _load_fred_csv(DATA_RAW / fname, code)

    # Weekly/daily → monthly means; monthly series stay monthly
    monthly = {code: _to_monthly(s, how="mean") for code, s in frames.items()}
    df = pd.concat(monthly.values(), axis=1)
    df.columns = list(monthly.keys())
    df.index.name = "date"
    return df


# ---------- Metro-level LAUS unemployment (FRED API only) ----------

def load_metro_unemployment(metros: list[str] | None = None) -> pd.DataFrame:
    """Pull metro-level unemployment rate (LAUS, SA) from FRED for each metro."""
    from pandas_datareader import data as pdr
    metros = metros or list(METROS.keys())
    rows = []
    for m in metros:
        code = METROS[m]["laus_code"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = pdr.DataReader(code, "fred", START, END)[code]
        s.index = s.index.to_period("M").to_timestamp("M")
        s.index.name = "date"
        rows.append(pd.DataFrame({"metro": m, "unrate_metro": s.values}, index=s.index))
    out = pd.concat(rows).reset_index().sort_values(["metro", "date"])
    return out


# ---------- ACS Census (static 2024 fixed effects) ----------

_MONEY_RE = re.compile(r"[^\d.\-]")


def _parse_money(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    if s in {"-", "(X)", "N", ""}:
        return np.nan
    return float(_MONEY_RE.sub("", s)) if _MONEY_RE.sub("", s) else np.nan


def _parse_percent(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").strip()
    if s in {"-", "(X)", "N", ""}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _find_metro_col(cols: list[str], needle: str, suffix: str) -> str | None:
    for c in cols:
        if needle in c and c.endswith(suffix):
            return c
    return None


def load_acs_static() -> pd.DataFrame:
    """Return per-metro static Census features:
    median_income, median_home_value, vacancy_rate (all ACS 5Y 2024).
    """
    # 1) Median household income (S1901)
    s1901 = pd.read_csv(DATA_RAW / "ACSST5Y2024.S1901-2026-04-20T012535.csv")
    # 2) Median home value (B25077)
    b25077 = pd.read_csv(DATA_RAW / "ACSDT5Y2024.B25077-2026-04-20T011959.csv")
    # 3) DP04 occupancy for vacancy rate
    dp04 = pd.read_csv(DATA_RAW / "ACSDP5Y2024.DP04-2026-04-20T011849.csv")

    recs = []
    for m, meta in METROS.items():
        needle = meta["acs_col_contains"]

        # Median household income is encoded in a row — S1901 uses the "Median income (dollars)"
        # row under the "Households" estimate column.
        inc = np.nan
        col = _find_metro_col(list(s1901.columns), needle, "Households!!Estimate")
        if col is not None:
            # Find the "Median income (dollars)" row
            label_col = s1901.columns[0]
            hit = s1901[s1901[label_col].astype(str).str.contains("Median income", na=False)]
            if not hit.empty:
                inc = _parse_money(hit.iloc[0][col])

        # Median home value from B25077
        hv = np.nan
        col = _find_metro_col(list(b25077.columns), needle, "Estimate")
        if col is not None:
            label_col = b25077.columns[0]
            hit = b25077[b25077[label_col].astype(str).str.contains("Median value", na=False)]
            if not hit.empty:
                hv = _parse_money(hit.iloc[0][col])

        # Vacancy rate from DP04 — "Vacant housing units" percent column
        vac = np.nan
        col = _find_metro_col(list(dp04.columns), needle, "Percent")
        if col is not None:
            label_col = dp04.columns[0]
            hit = dp04[dp04[label_col].astype(str).str.contains("Vacant housing units", na=False)]
            if not hit.empty:
                vac = _parse_percent(hit.iloc[0][col])

        recs.append({
            "metro": m,
            "acs_median_income": inc,
            "acs_median_home_value": hv,
            "acs_vacancy_rate": vac,
        })
    return pd.DataFrame(recs)


# ---------- Orchestration ----------

def build_raw_panel(prefer_api: bool = True) -> pd.DataFrame:
    """Join everything into a long monthly panel: one row per (metro, date)."""
    zillow = load_zillow_panel()
    fred = load_fred_national(prefer_api=prefer_api).reset_index()
    fred["date"] = pd.to_datetime(fred["date"]).dt.to_period("M").dt.to_timestamp("M")

    metro_unrate = load_metro_unemployment()
    metro_unrate["date"] = pd.to_datetime(metro_unrate["date"]).dt.to_period("M").dt.to_timestamp("M")

    acs = load_acs_static()

    panel = zillow.merge(fred, on="date", how="left")
    panel = panel.merge(metro_unrate, on=["metro", "date"], how="left")
    panel = panel.merge(acs, on="metro", how="left")

    panel = panel.sort_values(["metro", "date"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    p = build_raw_panel(prefer_api=False)
    print(p.shape)
    print(p.tail())
    print(p.isna().sum())
