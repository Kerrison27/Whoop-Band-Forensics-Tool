# analysis/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import hashlib
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# Use Matplotlib only for PNG exports (no extra deps)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================
# Threshold configuration
# ==============================
@dataclass
class Thresholds:
    elevated_hr_bpm: float = 90.0
    low_hrv_ms: float = 40.0
    high_strain: float = 14.0
    poor_sleep_eff_pct: float = 85.0
    short_sleep_min: float = 360.0
    sleep_debt_min: float = 60.0


# ==============================
# CSV helpers (robust parsing)
# ==============================
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def _pick_first(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_any_datetime(s: pd.Series) -> pd.Series:
    """Parse timestamps from strings or epoch seconds/ms. Returns datetime64[ns]."""
    # numeric → try epoch
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.dropna().gt(1e12).any():   # epoch ms
            return pd.to_datetime(s_num, unit="ms", errors="coerce")
        if s_num.dropna().gt(1e9).any():    # epoch s
            return pd.to_datetime(s_num, unit="s", errors="coerce")
        return pd.to_datetime(s, errors="coerce")

    # strings/datetime
    dt = pd.to_datetime(s, errors="coerce")
    # If looks like HH:MM or HH:MM:SS without a date, parse as timedelta anchored to 1970-01-01
    if dt.isna().mean() > 0.5 and s.astype(str).str.contains(r"^\d{1,2}:\d{2}(:\d{2})?$").any():
        td = pd.to_timedelta(s, errors="coerce")
        base = pd.Timestamp("1970-01-01")
        return base + td
    return dt

def _coerce_duration_to_min(series: pd.Series) -> pd.Series:
    """
    Convert duration-like columns to minutes.
    Handles strings like HH:MM:SS and numeric hours/seconds/minutes.
    """
    s = series.copy()
    as_str = s.astype(str)
    if as_str.str.contains(r"^\d{1,2}:\d{2}(:\d{2})?$").all():
        td = pd.to_timedelta(as_str, errors="coerce")
        return td.dt.total_seconds() / 60.0

    s_num = pd.to_numeric(s, errors="coerce")
    # If typical value is <24 (and non-integers present), likely hours → minutes
    if s_num.dropna().between(0, 24).mean() > 0.7 and (s_num.dropna() % 1 != 0).any():
        return s_num * 60.0
    # If very large numbers (often seconds) → minutes
    if s_num.dropna().gt(10000).mean() > 0.5:
        return s_num / 60.0
    # Otherwise assume already minutes
    return s_num

def _coerce_pct(series: pd.Series) -> pd.Series:
    """Ensure percentages are 0..100. If mostly 0..1, multiply by 100."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().between(0, 1.5).mean() > 0.7:  # mostly 0..1
        s = s * 100.0
    return s.clip(lower=0, upper=100)

def _read_csv_bytes_flex(path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Read a CSV robustly (UTF-8-sig, cp1252 fallback) and return (df, sha256hex).
    """
    raw_bytes = Path(path).read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    last_err = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            return df, sha
        except Exception as e:
            last_err = e
    # Last resort: default
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
        return df, sha
    except Exception as e:
        raise e if last_err is None else last_err


# ==============================
# Sleep CSV normalisation
# ==============================
def _normalise_sleep(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a wide variety of sleep CSV schemas and return daily rows with:
      date (required), and any of: asleep_min, sleep_efficiency_pct, sleep_debt_min,
      respiratory_rate, spo2_pct, hrv_ms (all optional)
    """
    if raw is None or raw.empty:
        raise ValueError("Sleep CSV appears to be empty.")

    df = _clean_cols(raw)

    # date-like column
    date_candidates = [
        "date", "day", "cycle_date",
        "start", "start_time", "start_datetime", "start_local", "start_utc",
        "bedtime_start", "sleep_start", "start_(local)",
    ]
    dcol = _pick_first(df, date_candidates)
    if dcol is None:
        maybe = [c for c in df.columns if ("date" in c) or ("start" in c)]
        dcol = maybe[0] if maybe else None
    if dcol is None:
        raise KeyError("Could not find a date/start column in sleeps CSV.")

    dt = _parse_any_datetime(df[dcol])
    if dt.isna().all():
        raise KeyError("Sleep CSV has a date/start column but its values could not be parsed.")

    # normalized date column
    out = pd.DataFrame({"date": pd.to_datetime(dt).dt.normalize()})

    # asleep minutes
    asleep_candidates = [
        "asleep_min", "minutes_asleep", "time_asleep", "sleep_time", "sleep_duration",
        "asleep", "duration", "total_sleep_time", "time_in_sleep",
    ]
    acol = _pick_first(df, asleep_candidates)
    if acol is not None:
        out["asleep_min"] = _coerce_duration_to_min(df[acol])

    # sleep efficiency %
    eff_candidates = [
        "sleep_efficiency_pct", "sleep_efficiency_percent", "efficiency", "sleep_efficiency",
        "sleep_performance_pct", "sleep_performance"
    ]
    ecol = _pick_first(df, eff_candidates)
    if ecol is not None:
        out["sleep_efficiency_pct"] = _coerce_pct(df[ecol])

    # sleep debt minutes
    debt_candidates = ["sleep_debt_min", "sleep_debt_mins", "sleep_debt", "debt"]
    d2 = _pick_first(df, debt_candidates)
    if d2 is not None:
        s = pd.to_numeric(df[d2], errors="coerce")
        if s.dropna().between(0, 24).mean() > 0.7:  # hours → minutes
            s = s * 60.0
        out["sleep_debt_min"] = s

    # optional vitals in sleep exports
    if "respiratory_rate" in df.columns:
        out["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")
    if "spo2" in df.columns:
        out["spo2_pct"] = _coerce_pct(df["spo2"])
    if "spo2_pct" in df.columns:
        out["spo2_pct"] = _coerce_pct(df["spo2_pct"])
    for hrv_col in ["hrv_ms", "rmssd", "nocturnal_hrv_rmssd_millis"]:
        if hrv_col in df.columns:
            out["hrv_ms"] = pd.to_numeric(df[hrv_col], errors="coerce")
            break

    # group per calendar date (be tolerant if nothing to aggregate)
    agg = {
        "asleep_min": "sum",
        "sleep_efficiency_pct": "mean",
        "sleep_debt_min": "mean",
        "respiratory_rate": "mean",
        "spo2_pct": "mean",
        "hrv_ms": "mean",
    }
    out["date"] = pd.to_datetime(out["date"]).dt.date
    agg_dict = {k: v for k, v in agg.items() if k in out.columns}

    if agg_dict:
        out = out.groupby("date", as_index=False).agg(agg_dict)
    else:
        # Keep unique dates only (no strict requirement to have sleep metrics)
        out = out[["date"]].drop_duplicates()

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    # IMPORTANT CHANGE: do NOT raise even if no asleep/efficiency/debt present.
    # Return date-only (plus any optional vitals) so merge can still proceed.
    return out

def load_sleep_csv(path: str | Path) -> Tuple[pd.DataFrame, str]:
    raw, sha = _read_csv_bytes_flex(Path(path))
    df = _normalise_sleep(raw)
    return df, sha


# ==============================
# Physio CSV normalisation
# ==============================
def _normalise_physio(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise "daily" physiological CSV to columns:
      date, resting_hr, hrv_ms (optional), day_strain (optional),
      respiratory_rate, spo2_pct, skin_temp_c, steps, cadence_spm,
      stress_score, recovery_score_pct (all optional)
    """
    if raw is None or raw.empty:
        raise ValueError("Physiological CSV appears to be empty.")

    df = _clean_cols(raw)

    # date-like
    date_candidates = ["date", "day", "cycle_date", "timestamp", "start", "start_time", "summary_date"]
    dcol = _pick_first(df, date_candidates)
    if dcol is None:
        maybe = [c for c in df.columns if ("date" in c) or ("start" in c) or ("time" in c)]
        dcol = maybe[0] if maybe else None
    if dcol is None:
        raise KeyError("Could not find a date/timestamp column in physio CSV.")

    dt = _parse_any_datetime(df[dcol])
    if dt.isna().all():
        raise KeyError("Physio CSV has a date/timestamp column but its values could not be parsed.")

    out = pd.DataFrame({"date": pd.to_datetime(dt).dt.normalize()})

    # resting HR
    rhr_candidates = ["resting_hr", "resting_heart_rate", "rhr", "avg_resting_heart_rate", "resting_heart_rate_bpm"]
    rcol = _pick_first(df, rhr_candidates)
    if rcol is not None:
        out["resting_hr"] = pd.to_numeric(df[rcol], errors="coerce")

    # HRV (daily)
    hrv_candidates = ["hrv_ms", "rmssd", "daily_rmssd_ms", "hrv"]
    hcol = _pick_first(df, hrv_candidates)
    if hcol is not None:
        out["hrv_ms"] = pd.to_numeric(df[hcol], errors="coerce")

    # strain/effort
    strain_candidates = ["day_strain", "strain", "strain_score", "whoop_strain"]
    scol = _pick_first(df, strain_candidates)
    if scol is not None:
        out["day_strain"] = pd.to_numeric(df[scol], errors="coerce")

    # optional vitals
    opt_map: Dict[str, List[str]] = {
        "respiratory_rate": ["respiratory_rate", "rr", "breaths_per_min"],
        "spo2_pct": ["spo2_pct", "spo2", "oxygen_saturation"],
        "skin_temp_c": ["skin_temp_c", "skin_temperature_c", "wrist_temp_c", "temperature_deviation"],
        "steps": ["steps", "total_steps", "step_count"],
        "cadence_spm": ["cadence_spm", "avg_cadence", "mean_cadence"],
        "stress_score": ["stress_score", "garmin_stress", "stress"],
        "recovery_score_pct": ["recovery_score_pct", "recovery_score", "whoop_recovery"],
    }
    for out_col, cands in opt_map.items():
        col = _pick_first(df, cands)
        if col is not None:
            if out_col.endswith("_pct"):
                out[out_col] = _coerce_pct(df[col])
            else:
                out[out_col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate per calendar date; tolerate missing metrics
    agg_phys = {
        "resting_hr": "mean",
        "hrv_ms": "mean",
        "day_strain": "mean",
        "respiratory_rate": "mean",
        "spo2_pct": "mean",
        "skin_temp_c": "mean",
        "steps": "sum",
        "cadence_spm": "mean",
        "stress_score": "mean",
        "recovery_score_pct": "mean",
    }
    out["date"] = pd.to_datetime(out["date"]).dt.date
    agg_dict = {k: v for k, v in agg_phys.items() if k in out.columns}

    if agg_dict:
        out = out.groupby("date", as_index=False).agg(agg_dict)
    else:
        # No metric columns found — keep unique dates so merge still works
        out = out[["date"]].drop_duplicates()

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    return out

def load_physio_csv(path: str | Path) -> Tuple[pd.DataFrame, str]:
    raw, sha = _read_csv_bytes_flex(Path(path))
    df = _normalise_physio(raw)
    return df, sha


# ==============================
# Merge + rule-based anomalies
# ==============================
def merge_daily(physio_df: pd.DataFrame, sleep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join by date, then prefer physio HRV if both present.
    """
    if (physio_df is None or physio_df.empty) and (sleep_df is None or sleep_df.empty):
        return pd.DataFrame(columns=["date"]).assign(date=pd.to_datetime([]))

    if physio_df is None or physio_df.empty:
        merged = sleep_df.copy()
    elif sleep_df is None or sleep_df.empty:
        merged = physio_df.copy()
    else:
        merged = pd.merge(
            physio_df, sleep_df, on="date", how="outer", suffixes=("", "_sleep")
        )
        # Prefer physio HRV; fill from sleep if missing
        if "hrv_ms" not in merged.columns and "hrv_ms_sleep" in merged.columns:
            merged.rename(columns={"hrv_ms_sleep": "hrv_ms"}, inplace=True)
        elif "hrv_ms" in merged.columns and "hrv_ms_sleep" in merged.columns:
            merged["hrv_ms"] = merged["hrv_ms"].fillna(merged["hrv_ms_sleep"])
            merged.drop(columns=["hrv_ms_sleep"], inplace=True, errors="ignore")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged

def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def detect_anomalies(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """
    Apply rule-based thresholds and produce boolean flags + any_anomaly.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    # Coerce numeric for safety
    num_cols = [
        "resting_hr", "hrv_ms", "day_strain",
        "asleep_min", "sleep_efficiency_pct", "sleep_debt_min",
    ]
    out = _to_num(out, num_cols)

    # Flags
    out["elevated_hr"]      = (out["resting_hr"] > th.elevated_hr_bpm) if "resting_hr" in out.columns else False
    out["low_hrv"]          = (out["hrv_ms"] < th.low_hrv_ms) if "hrv_ms" in out.columns else False
    out["high_strain_flag"] = (out["day_strain"] > th.high_strain) if "day_strain" in out.columns else False
    out["poor_sleep"]       = (out["sleep_efficiency_pct"] < th.poor_sleep_eff_pct) if "sleep_efficiency_pct" in out.columns else False
    out["short_sleep"]      = (out["asleep_min"] < th.short_sleep_min) if "asleep_min" in out.columns else False
    out["sleep_debt_flag"]  = (out["sleep_debt_min"] > th.sleep_debt_min) if "sleep_debt_min" in out.columns else False

    flag_cols = [c for c in ["elevated_hr","low_hrv","high_strain_flag","poor_sleep","short_sleep","sleep_debt_flag"] if c in out.columns]
    out["any_anomaly"] = out[flag_cols].any(axis=1) if flag_cols else False

    return out


# ==============================
# Output saving (CSV + PNG charts)
# ==============================
def _plot_line(dates: pd.Series, values: pd.Series, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(dates, values, marker="o", linewidth=1.5, markersize=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.autofmt_xdate()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_case_outputs(case_dir: Path, analysed: pd.DataFrame) -> None:
    """
    Write output/daily.csv and a handful of PNG charts if columns exist.
    """
    case_dir = Path(case_dir)
    out_dir = case_dir / "output"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = analysed.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Write CSV (include ML columns if present)
    csv_path = out_dir / "daily.csv"
    df.to_csv(csv_path, index=False)

    # Plots for common metrics
    metrics: List[Tuple[str, str, str]] = [
        ("resting_hr", "Resting HR (bpm)", "resting_hr.png"),
        ("hrv_ms", "HRV (ms)", "hrv_ms.png"),
        ("day_strain", "Strain", "day_strain.png"),
        ("sleep_efficiency_pct", "Sleep Efficiency (%)", "sleep_efficiency_pct.png"),
        ("asleep_min", "Asleep (min)", "asleep_min.png"),
        ("sleep_debt_min", "Sleep Debt (min)", "sleep_debt_min.png"),
        # Optional metrics if present:
        ("respiratory_rate", "Respiratory Rate (rpm)", "respiratory_rate.png"),
        ("spo2_pct", "SpO₂ (%)", "spo2_pct.png"),
        ("skin_temp_c", "Skin Temp (°C)", "skin_temp_c.png"),
        ("steps", "Steps", "steps.png"),
        ("stress_score", "Stress Score", "stress_score.png"),
        ("recovery_score_pct", "Recovery (%)", "recovery_score_pct.png"),
        ("ml_score", "ML Anomaly Score", "ml_score.png"),
        ("ml_score_pct", "ML Anomaly Score (%)", "ml_score_pct.png"),
    ]
    if "date" in df.columns:
        for col, title, fname in metrics:
            if col in df.columns and df[col].notna().any():
                try:
                    _plot_line(df["date"], pd.to_numeric(df[col], errors="coerce"), title, "", plots_dir / fname)
                except Exception:
                    pass


# ==============================
# (Optional) Robust bands for charts
# ==============================
def add_robust_bands(df: pd.DataFrame, cols: List[str] | None = None, window: int = 30) -> pd.DataFrame:
    """
    Adds columns like: metric_baseline, metric_low_band, metric_high_band
    using rolling median ± 1.4826*MAD.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    cols = cols or ["resting_hr", "hrv_ms", "respiratory_rate", "spo2_pct",
                    "skin_temp_c", "sleep_efficiency_pct", "asleep_min"]
    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        med = x.rolling(window=window, min_periods=max(3, window // 3)).median()
        mad = (x - med).abs().rolling(window=window, min_periods=max(3, window // 3)).median()
        band = 1.4826 * mad
        out[f"{c}_baseline"] = med
        out[f"{c}_low_band"] = med - band
        out[f"{c}_high_band"] = med + band
    return out
