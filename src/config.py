"""Project-wide configuration: paths, target metros, date range, and feature codes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "cleaned"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"

for p in (DATA_CLEAN, OUT_FIG, OUT_TAB):
    p.mkdir(parents=True, exist_ok=True)

METROS = {
    "San Francisco, CA": {
        "zillow_region": "San Francisco, CA",
        "laus_code": "SANF806URN",
        "acs_col_contains": "San Francisco-Oakland-Fremont",
    },
    "Austin, TX": {
        "zillow_region": "Austin, TX",
        "laus_code": "AUST448URN",
        "acs_col_contains": "Austin-Round Rock",
    },
    "Cleveland, OH": {
        "zillow_region": "Cleveland, OH",
        "laus_code": "CLEV439URN",
        "acs_col_contains": "Cleveland, OH Metro Area",
    },
}

ZILLOW_FILES = {
    "zhvi":         "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "new_listings": "Metro_new_listings_uc_sfrcondo_sm_month.csv",
    "days_pending": "Metro_med_doz_pending_uc_sfrcondo_sm_month.csv",
    "price_cuts":   "Metro_perc_listings_price_cut_uc_sfrcondo_sm_month.csv",
}

FRED_NATIONAL = {
    "MORTGAGE30US": "MORTGAGE30US.csv",
    "UNRATE":       "UNRATE.csv",
    "CPIAUCSL":     "CPIAUCSL.csv",
    "HOUST":        "HOUST.csv",
    "STLFSI4":      "STLFSI4.csv",
}

START = "2000-01-01"
END   = "2026-03-31"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
