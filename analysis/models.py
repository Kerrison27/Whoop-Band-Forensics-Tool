# analysis/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ------------------------
# Feature engineering
# ------------------------

BASE_COLS = [
    "resting_hr",
    "hrv_ms",
    "day_strain",
    "asleep_min",
    "sleep_efficiency_pct",
    "sleep_debt_min",
]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column in the DataFrame.")

    work = df.copy()
    work = work.sort_values("date").reset_index(drop=True)

    # Ensure all expected cols exist (fill missing with NaN)
    for c in BASE_COLS:
        if c not in work.columns:
            work[c] = np.nan

    # 7-day rolling means (min_periods=1 to work from the first day)
    for c in BASE_COLS:
        work[f"{c}_ma7"] = work[c].rolling(window=7, min_periods=1).mean()

    return work


def _feature_matrix_for_iso(df: pd.DataFrame) -> pd.DataFrame:
    feats = [
        # raw features (except resting_hr)
        "hrv_ms", "day_strain", "asleep_min", "sleep_efficiency_pct", "sleep_debt_min",
        # rolling means (including resting_hr)
        "resting_hr_ma7", "hrv_ms_ma7", "day_strain_ma7", "asleep_min_ma7",
        "sleep_efficiency_pct_ma7", "sleep_debt_min_ma7",
    ]
    return df[feats].copy()


def _feature_matrix_for_ridge(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Predict resting_hr using other metrics + rolling means.
    Target: resting_hr
    """
    feats = [
        "hrv_ms", "day_strain", "asleep_min", "sleep_efficiency_pct", "sleep_debt_min",
        "resting_hr_ma7", "hrv_ms_ma7", "day_strain_ma7", "asleep_min_ma7",
        "sleep_efficiency_pct_ma7", "sleep_debt_min_ma7",
    ]
    X = df[feats].copy()
    y = df["resting_hr"].copy()
    return X, y


# ------------------------
# Artifacts & paths
# ------------------------

@dataclass
class ModelPaths:
    iso_path: Path
    ridge_path: Path

    @staticmethod
    def for_case(case_dir: Path) -> "ModelPaths":
        models_dir = Path(case_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return ModelPaths(
            iso_path=models_dir / "isolation_forest.joblib",
            ridge_path=models_dir / "resting_hr_baseline.joblib",
        )


# ------------------------
# Training
# ------------------------

def train_iso_model(history_df: pd.DataFrame, case_dir: Path) -> Path:
    """
    Train IsolationForest on historical daily data and save joblib.
    Returns path to the saved model.
    """
    paths = ModelPaths.for_case(case_dir)

    feats_df = _prepare_features(history_df)
    X = _feature_matrix_for_iso(feats_df).to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scl = scaler.fit_transform(X_imp)

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    iso.fit(X_scl)

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": iso,
            "feature_order": list(_feature_matrix_for_iso(feats_df).columns),
        },
        paths.iso_path,
    )
    return paths.iso_path


def train_rhr_baseline(history_df: pd.DataFrame, case_dir: Path) -> Path:
    """
    Train a Ridge regressor to predict resting_hr from other metrics.
    Stores model + imputer + scaler + sigma (std of residuals).
    """
    paths = ModelPaths.for_case(case_dir)

    feats_df = _prepare_features(history_df)
    X, y = _feature_matrix_for_ridge(feats_df)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scl = scaler.fit_transform(X_imp)

    # Simple, stable baseline model
    model = Ridge(alpha=1.0, random_state=42)
    # For rows where y is NaN, drop them from training
    mask = ~y.isna()
    model.fit(X_scl[mask], y[mask])

    # Compute sigma on the training set (only where we had y)
    preds = model.predict(X_scl[mask])
    resid = y[mask].to_numpy() - preds
    sigma = float(np.nanstd(resid)) if resid.size else 1.0  # avoid zero

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "feature_order": list(X.columns),
            "sigma": sigma,
            "k": 2.0,  # 2-sigma default threshold for flags
        },
        paths.ridge_path,
    )
    return paths.ridge_path


# ------------------------
# Scoring
# ------------------------

def models_available(case_dir: Path) -> bool:
    paths = ModelPaths.for_case(case_dir)
    return paths.iso_path.exists() and paths.ridge_path.exists()


def score_iso_model(df: pd.DataFrame, case_dir: Path) -> pd.DataFrame:
    """
    Adds columns:
      - iso_score (decision_function; higher => more normal)
      - iso_flag  (True for anomaly)
    """
    paths = ModelPaths.for_case(case_dir)
    bundle = joblib.load(paths.iso_path)

    feats_df = _prepare_features(df)
    X = feats_df[bundle["feature_order"]].to_numpy(dtype=float)

    X_imp = bundle["imputer"].transform(X)
    X_scl = bundle["scaler"].transform(X_imp)

    # decision_function: higher is more normal
    scores = bundle["model"].decision_function(X_scl)
    # predict: 1 normal, -1 anomaly
    preds = bundle["model"].predict(X_scl)
    flags = preds == -1

    out = df.copy()
    out["iso_score"] = scores
    out["iso_flag"] = flags
    return out


def score_rhr_baseline(df: pd.DataFrame, case_dir: Path) -> pd.DataFrame:
    """
    Predict resting_hr and flag large positive residuals.
    Adds columns:
      - rhr_pred
      - rhr_residual (actual - pred)
      - rhr_z        (residual / sigma)
      - rhr_flag     (True if rhr_z > k)
    """
    paths = ModelPaths.for_case(case_dir)
    bundle = joblib.load(paths.ridge_path)

    feats_df = _prepare_features(df)
    X = feats_df[bundle["feature_order"]].to_numpy(dtype=float)

    X_imp = bundle["imputer"].transform(X)
    X_scl = bundle["scaler"].transform(X_imp)

    pred = bundle["model"].predict(X_scl)

    out = df.copy()
    if "resting_hr" in out.columns:
        residual = out["resting_hr"].to_numpy(dtype=float) - pred
    else:
        residual = np.full(shape=pred.shape, fill_value=np.nan, dtype=float)

    sigma = float(bundle.get("sigma", 1.0))
    k = float(bundle.get("k", 2.0))
    z = residual / (sigma if sigma > 0 else 1.0)
    flag = z > k

    out["rhr_pred"] = pred
    out["rhr_residual"] = residual
    out["rhr_z"] = z
    out["rhr_flag"] = flag
    return out
