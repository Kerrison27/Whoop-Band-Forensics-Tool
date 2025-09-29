# analysis/main.py
# -------------------------------------------------------------
# Wearable forensics: read Whoop CSVs, build daily metrics,
# flag anomalies, save CSVs/plots, and store in SQLite.
# -------------------------------------------------------------

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths (no edits needed) ----------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = HERE / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
DB_PATH = OUTPUT_DIR / "wearable_forensics.db"

PHYSIO_FILE = r"C:\Users\Alex\Documents\Whoop\physiological_cycles.csv"
SLEEPS_FILE  = r"C:\Users\Alex\Documents\Whoop\sleeps.csv"


# ---------- Helpers ----------
def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_time_col(df: pd.DataFrame) -> str:
    df.columns = [c.strip() for c in df.columns]
    if "Cycle start time" in df.columns:
        return "Cycle start time"
    for c in df.columns:
        if c.lower().replace(" ", "") == "cyclestarttime":
            return c
    raise KeyError(
        "Could not find the time column. Expected 'Cycle start time'. "
        f"Available columns: {list(df.columns)}"
    )


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- Readers ----------
def read_physio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = _find_time_col(df)

    ts = pd.to_datetime(df[tcol], format="mixed", dayfirst=False, errors="coerce")
    df["Date"] = ts.dt.date

    # expected columns -> output names
    mapping = {
        "Resting heart rate (bpm)": "resting_hr",
        "Heart rate variability (ms)": "hrv_ms",
        "Day Strain": "day_strain",
    }
    missing = [k for k in mapping if k not in df.columns]
    if missing:
        raise KeyError(f"[Physio] Missing columns: {missing}\nHave: {list(df.columns)}")

    df = df[["Date", *mapping.keys()]].rename(columns=mapping)
    df = _coerce_numeric(df, ["resting_hr", "hrv_ms", "day_strain"])

    # multiple rows per date -> take mean (daily)
    df = (
        df.groupby("Date", as_index=False)
        .agg({"resting_hr": "mean", "hrv_ms": "mean", "day_strain": "mean"})
    )
    return df


def read_sleeps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = _find_time_col(df)

    ts = pd.to_datetime(df[tcol], format="mixed", dayfirst=False, errors="coerce")
    df["Date"] = ts.dt.date

    mapping = {
        "Sleep efficiency %": "sleep_efficiency_pct",
        "Asleep duration (min)": "asleep_min",
        "Sleep debt (min)": "sleep_debt_min",
    }
    missing = [k for k in mapping if k not in df.columns]
    if missing:
        raise KeyError(f"[Sleep] Missing columns: {missing}\nHave: {list(df.columns)}")

    df = df[["Date", *mapping.keys()]].rename(columns=mapping)
    df = _coerce_numeric(df, ["sleep_efficiency_pct", "asleep_min", "sleep_debt_min"])

    df = (
        df.groupby("Date", as_index=False)
        .agg(
            {
                "sleep_efficiency_pct": "mean",
                "asleep_min": "sum",       # sum across naps/night for total sleep
                "sleep_debt_min": "mean",
            }
        )
    )
    return df


# ---------- Merge + anomalies ----------
def merge_daily(physio: pd.DataFrame, sleeps: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(physio, sleeps, on="Date", how="outer").sort_values("Date")
    # make Date a proper date again after merge
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.date
    return merged


def compute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # thresholds (data-driven with sensible baselines)
    hr_thresh = np.nanpercentile(out["resting_hr"], 90)          # high HR
    hrv_thresh = np.nanpercentile(out["hrv_ms"], 10)             # low HRV
    strain_thresh = np.nanpercentile(out["day_strain"], 75)      # high strain
    eff_thresh = 85.0                                            # poor sleep efficiency
    short_sleep_min = 360.0                                      # < 6h
    debt_thresh = 60.0                                           # > 60 min debt

    out["elevated_hr"] = (out["resting_hr"] > hr_thresh).astype("Int64")
    out["low_hrv"] = (out["hrv_ms"] < hrv_thresh).astype("Int64")
    out["high_strain"] = (out["day_strain"] > strain_thresh).astype("Int64")
    out["poor_sleep"] = (out["sleep_efficiency_pct"] < eff_thresh).astype("Int64")
    out["short_sleep"] = (out["asleep_min"] < short_sleep_min).astype("Int64")
    out["sleep_debt"] = (out["sleep_debt_min"] > debt_thresh).astype("Int64")

    out["any_anomaly"] = (
        out[["elevated_hr", "low_hrv", "high_strain", "poor_sleep", "short_sleep", "sleep_debt"]]
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .astype("Int64")
    )
    return out


# ---------- Outputs ----------
def save_csvs(df_daily: pd.DataFrame, df_anom: pd.DataFrame):
    daily_path = OUTPUT_DIR / "merged_daily.csv"
    anom_path = OUTPUT_DIR / "anomalies_daily.csv"
    df_daily.to_csv(daily_path, index=False)
    df_anom.to_csv(anom_path, index=False)
    return daily_path, anom_path


def _lineplot(df: pd.DataFrame, col: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(df["Date"])
    ax.plot(x, df[col], linewidth=1.5)
    ax.set_title(col)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    out_path = PLOTS_DIR / f"{col}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def save_plots(df: pd.DataFrame) -> list[Path]:
    plots = []
    for col, ylabel in [
        ("resting_hr", "bpm"),
        ("hrv_ms", "ms"),
        ("day_strain", "score"),
        ("sleep_efficiency_pct", "%"),
        ("asleep_min", "minutes"),
        ("sleep_debt_min", "minutes"),
    ]:
        if col in df.columns:
            plots.append(_lineplot(df, col, ylabel))
    return plots


def write_sqlite(df_daily: pd.DataFrame, df_anom: pd.DataFrame, db_path: Path):
    con = sqlite3.connect(db_path)
    try:
        df_daily.to_sql("daily_metrics", con, if_exists="replace", index=False)
        df_anom.to_sql("anomalies", con, if_exists="replace", index=False)
    finally:
        con.close()


# ---------- Main ----------
def main():
    _ensure_dirs()

    # Load
    physio = read_physio(PHYSIO_FILE)
    sleeps = read_sleeps(SLEEPS_FILE)
    merged = merge_daily(physio, sleeps)
    anomalies = compute_anomalies(merged)

    # Stats
    print("\nLoaded records:")
    print(f"  physio rows: {len(physio)}")
    print(f"  sleeps rows: {len(sleeps)}")
    print(f"  merged rows: {len(merged)}")

    # Anomaly counts
    counts = {
        "elevated_hr_events": int(anomalies["elevated_hr"].fillna(0).sum()),
        "low_hrv_events": int(anomalies["low_hrv"].fillna(0).sum()),
        "high_strain_events": int(anomalies["high_strain"].fillna(0).sum()),
        "poor_sleep_events": int(anomalies["poor_sleep"].fillna(0).sum()),
        "short_sleep_events": int(anomalies["short_sleep"].fillna(0).sum()),
        "sleep_debt_events": int(anomalies["sleep_debt"].fillna(0).sum()),
        "days_with_any_anomaly": int(anomalies["any_anomaly"].fillna(0).sum()),
    }
    print("\nAnomaly summary:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # Save artifacts
    daily_csv, anom_csv = save_csvs(merged, anomalies)
    write_sqlite(merged, anomalies, DB_PATH)
    plot_paths = save_plots(merged)

    print("\nSaved plots:")
    for p in plot_paths:
        print(f" - {p}")

    print(f"\nSaved SQLite DB to: {DB_PATH}")
    print(f"\nCSV outputs in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
