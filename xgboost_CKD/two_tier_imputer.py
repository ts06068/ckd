
"""
two_tier_imputer.py
═══════════════════════════════════════════════════════════════════════════════
Standalone module for CKD FibroScan prediction pipeline.
Extracted from model_full_CKD_v4-2.ipynb (Cell 4 + Cell 14).

Note on Sex encoding
────────────────────
Sex is stored as numeric 1.0 / 2.0 in the source data.
map_sex_to_binary() is intentionally absent — pd.to_numeric() handles it.

Usage (loading saved imputer in a new instance)
───────────────────────────────────────────────
    import sys
    sys.path.insert(0, "/path/to/models/src")
    from two_tier_imputer import TwoTierImputer   # ← must come BEFORE joblib.load
    import joblib
    imp = joblib.load("/path/to/models/scenario_B/imputer.joblib")
    df_imputed = imp.transform(df_new)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

# ══════════════════════════════════════════════════════════════════════════════
# Constants  (mirrored from Cell 4 — single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

CAP_CUTOFF_BY_HOSPITAL: dict[int, pd.Timestamp] = {
    10: pd.Timestamp("2012-03-14"),
    20: pd.Timestamp("2013-01-10"),
}

TIER1_FEATURES: list[str] = ["Age", "Sex"]
TIER2_FEATURES: list[str] = [
    "BMI", "HTN", "DM", "Baseline_Cr",
    "LSM_kPa", "Baseline_AST", "Baseline_ALT",
]
CAP_FEATURE: list[str] = ["CAP"]

MICE_AUX_COLS: list[str] = ["Hospital", "Fibroscan_Date_numeric"]


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions  (Cell 14 — needed for pre-imputation data prep)
# ══════════════════════════════════════════════════════════════════════════════

def coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Cast all columns to numeric via pd.to_numeric (coerce errors to NaN).
    Sex is already stored as 1.0 / 2.0 — no special mapping needed.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_fibroscan_date_numeric(
    df: pd.DataFrame,
    ref: pd.Timestamp = pd.Timestamp("2000-01-01"),
) -> pd.DataFrame:
    """Add Fibroscan_Date_numeric (days since ref) as MICE auxiliary variable."""
    df = df.copy()
    df["Fibroscan_Date_numeric"] = (
        pd.to_datetime(df["Fibroscan_Date"], errors="coerce") - ref
    ).dt.days.astype(float)
    return df


def is_cap_era(df: pd.DataFrame) -> pd.Series:
    """Return True for rows whose FibroScan date is on/after the hospital's
    CAP introduction date (hospital-specific MNAR cutoff).
    """
    dates  = pd.to_datetime(df["Fibroscan_Date"], errors="coerce")
    result = pd.Series(False, index=df.index)
    for hosp, cutoff in CAP_CUTOFF_BY_HOSPITAL.items():
        result |= (df["Hospital"] == hosp) & (dates >= cutoff)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TwoTierImputer
# ══════════════════════════════════════════════════════════════════════════════

class TwoTierImputer:
    """
    Two-tier imputer for CKD FibroScan cohort.

    Tier 1 — MCAR-like (Age, Sex)
        SimpleImputer(median, keep_empty_features=True)

    Tier 2 — MAR (BMI, HTN, DM, Baseline_Cr, LSM_kPa, AST, ALT)
        IterativeImputer / MICE (BayesianRidge)

    CAP — MNAR
        Scenario A (include_cap=False): caller excludes pre-cutoff rows.
        Scenario B (include_cap=True) : MICE + Hospital + Fibroscan_Date_numeric
                                        as auxiliary variables.

    Parameters
    ----------
    include_cap : bool
    aux_cols    : list[str] | None
    random_state: int
    """

    def __init__(
        self,
        include_cap: bool = False,
        aux_cols: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.include_cap  = include_cap
        self.aux_cols     = aux_cols or []
        self.random_state = random_state

        self._imp_t1:  SimpleImputer    | None = None
        self._imp_t2:  IterativeImputer | None = None
        self._imp_cap: IterativeImputer | None = None
        self._t1_cols:  list[str] | None = None
        self._t2_cols:  list[str] | None = None
        self._cap_cols: list[str] | None = None

    def fit(self, df_train: pd.DataFrame) -> "TwoTierImputer":
        # Tier 1
        self._t1_cols = [c for c in TIER1_FEATURES if c in df_train.columns]
        self._imp_t1  = SimpleImputer(strategy="median", keep_empty_features=True)
        self._imp_t1.fit(df_train[self._t1_cols])
        print(f"[IMPUTE] Tier-1 (SimpleImputer/median) fit: {self._t1_cols}")

        # Tier 2
        self._t2_cols = [c for c in TIER2_FEATURES if c in df_train.columns]
        self._imp_t2  = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=self.random_state,
            max_iter=10,
            sample_posterior=False,
        )
        self._imp_t2.fit(df_train[self._t2_cols])
        print(f"[IMPUTE] Tier-2 (MICE/BayesianRidge) fit : {self._t2_cols}")

        # CAP — MNAR, Scenario B only
        if self.include_cap and "CAP" in df_train.columns:
            aux_present    = [c for c in self.aux_cols if c in df_train.columns]
            self._cap_cols = ["CAP"] + aux_present
            self._imp_cap  = IterativeImputer(
                estimator=BayesianRidge(),
                random_state=self.random_state,
                max_iter=10,
                sample_posterior=False,
            )
            self._imp_cap.fit(df_train[self._cap_cols])
            print(f"[IMPUTE] CAP MICE fit (aux included)   : {self._cap_cols}")
            print("         Note: aux cols are NOT written back after imputation")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._imp_t1 is None:
            raise RuntimeError("TwoTierImputer has not been fitted yet.")
        out = df.copy()
        out[self._t1_cols] = self._imp_t1.transform(out[self._t1_cols])
        out[self._t2_cols] = self._imp_t2.transform(out[self._t2_cols])
        if self.include_cap and self._imp_cap is not None:
            imputed    = self._imp_cap.transform(out[self._cap_cols])
            out["CAP"] = imputed[:, 0]
        return out

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)

    def __repr__(self) -> str:
        return (
            f"TwoTierImputer("
            f"include_cap={self.include_cap}, "
            f"aux_cols={self.aux_cols}, "
            f"random_state={self.random_state}, "
            f"fitted={self._imp_t1 is not None})"
        )
