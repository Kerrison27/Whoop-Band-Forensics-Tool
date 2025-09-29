# analysis/ml.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Optional sklearn import (graceful fallback if missing)
try:
    from sklearn.ensemble import IsolationForest
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# ----------------------------
# Feature configuration
# ----------------------------
FEATURE_COLS: List[str] = [

    "resting_hr",
    "hrv_ms",
    "sleep_efficiency_pct",
    "asleep_min",
    "respiratory_rate",
    "spo2_pct",
    "skin_temp_c",
    "day_strain",
    "steps",
    "cadence_spm",
    "stress_score",
    "recovery_score_pct",
]


# ----------------------------
# Robust scaling (median/MAD)
# ----------------------------
def _fit_robust_scaler(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    med: Dict[str, float] = {}
    mad: Dict[str, float] = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        m = np.nanmedian(x)
        # 1.4826 * median(|x - median(x)|) ~ std if normal
        mad_c = 1.4826 * np.nanmedian(np.abs(x - m))
        if not np.isfinite(mad_c) or mad_c == 0:
            mad_c = 1.0
        med[c] = float(m) if np.isfinite(m) else 0.0
        mad[c] = float(mad_c)
    return {"median": med, "mad": mad}


def _transform_robust(df: pd.DataFrame, cols: List[str], scaler: Dict[str, Any]) -> np.ndarray:
    out = []
    n = len(df)
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
        else:
            x = np.full(n, np.nan)
        m = scaler["median"].get(c, 0.0)
        s = scaler["mad"].get(c, 1.0)
        z = (x - m) / (s if s != 0 else 1.0)
        z = np.where(np.isfinite(z), z, 0.0)
        out.append(z)
    return np.vstack(out).T


def _top_contributors(row: pd.Series, scaler: Dict[str, Any], cols: List[str], k: int = 3) -> str:
    contribs = []
    for c in cols:
        if c not in row:
            continue
        x = row[c]
        if pd.isna(x):
            continue
        m = scaler["median"].get(c, 0.0)
        s = scaler["mad"].get(c, 1.0)
        z = (x - m) / (s if s != 0 else 1.0)
        contribs.append((c, abs(float(z))))
    contribs.sort(key=lambda t: t[1], reverse=True)
    top = [f"{name} (|z|≈{round(val, 1)})" for name, val in contribs[:k] if np.isfinite(val)]
    return ", ".join(top)


# ----------------------------
# Rolling baselines (bands)
# ----------------------------
def add_robust_bands(
    df: pd.DataFrame,
    cols: List[str] | None = None,
    window: int = 30
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    cols = cols or [
        "resting_hr", "hrv_ms", "respiratory_rate", "spo2_pct",
        "skin_temp_c", "sleep_efficiency_pct", "asleep_min"
    ]

    minp = max(3, window // 3)
    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        med = x.rolling(window=window, min_periods=minp).median()
        mad = (x - med).abs().rolling(window=window, min_periods=minp).median()
        band = 1.4826 * mad
        out[f"{c}_baseline"] = med
        out[f"{c}_low_band"] = med - band
        out[f"{c}_high_band"] = med + band
    return out


# ----------------------------
# Model training / scoring
# ----------------------------
def _available_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def _model_path(case_dir: Path) -> Path:
    return Path(case_dir) / "models" / "anomaly_iforest.pkl"


def train_or_load_model(
    case_dir: Path,
    df: pd.DataFrame,
    min_history: int = 30,
    flag_rate: float = 0.05,
) -> Dict[str, Any]:
    case_dir = Path(case_dir)
    (case_dir / "models").mkdir(parents=True, exist_ok=True)
    mpath = _model_path(case_dir)

    # Load if present and usable
    if mpath.exists():
        try:
            with open(mpath, "rb") as f:
                mdl = pickle.load(f)
            if set(mdl.get("feature_cols", [])) <= set(df.columns):
                return mdl
        except Exception:
            pass  # retrain

    # Sort & prep
    if "date" in df.columns:
        df = df.sort_values("date")

    cols = _available_cols(df)
    if len(cols) == 0:
        return {"type": "none", "feature_cols": [], "scaler": None, "threshold": None}

    # If limited history, still save a fallback model so scoring works
    if len(df) < max(min_history, 20):
        scaler = _fit_robust_scaler(df, cols)
        mdl = {
            "type": "fallback",
            "feature_cols": cols,
            "scaler": scaler,
            "threshold": 3.0,  # mean |z| > 3 flags by default
        }
        with open(mpath, "wb") as f:
            pickle.dump(mdl, f)
        return mdl

    # Fit scaler on available history
    scaler = _fit_robust_scaler(df, cols)
    X = _transform_robust(df, cols, scaler)

    if HAVE_SK:
        # IsolationForest (multivariate). We'll use our own quantile threshold.
        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            bootstrap=True,
            random_state=42,
        )
        iso.fit(X)

        # Higher = more anomalous (invert decision_function)
        raw = -iso.decision_function(X)
        thr = float(np.quantile(raw, 1.0 - flag_rate))

        mdl = {
            "type": "iforest",
            "feature_cols": cols,
            "scaler": scaler,
            "threshold": thr,
            "sk_model": iso,
        }
    else:
        # Fallback score: mean absolute robust-z across features
        mean_abs_z = np.mean(np.abs(X), axis=1)
        thr = float(np.quantile(mean_abs_z, 1.0 - flag_rate))
        mdl = {
            "type": "fallback",
            "feature_cols": cols,
            "scaler": scaler,
            "threshold": thr,
        }

    with open(mpath, "wb") as f:
        pickle.dump(mdl, f)
    return mdl


def score_with_model(df: pd.DataFrame, model: Dict[str, Any]) -> pd.DataFrame:
    if model.get("type") in (None, "none"):
        return df

    cols = [c for c in model.get("feature_cols", []) if c in df.columns]
    if not cols:
        return df

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    X = _transform_robust(out, cols, model["scaler"])
    thr = float(model.get("threshold", np.nan))

    if model["type"] == "iforest" and HAVE_SK and "sk_model" in model:
        raw = -model["sk_model"].decision_function(X)  # higher = more anomalous
        score = raw
    else:
        score = np.mean(np.abs(X), axis=1)

    out["ml_score"] = score
    if np.isfinite(thr):
        out["ml_flag"] = out["ml_score"] >= thr
    else:
        out["ml_flag"] = False

    # Light-weight explainability: top contributors by |z|
    contribs: List[str] = []
    for _, row in out.iterrows():
        contribs.append(_top_contributors(row, model["scaler"], cols, k=3))
    out["ml_top_feats"] = contribs

    # Normalized 0..100 version for UI (does not affect flagging)
    s = out["ml_score"].astype(float).to_numpy()
    s_pct = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9) * 100.0
    out["ml_score_pct"] = s_pct

    return out


__all__ = [
    "FEATURE_COLS",
    "add_robust_bands",
    "train_or_load_model",
    "score_with_model",
]
