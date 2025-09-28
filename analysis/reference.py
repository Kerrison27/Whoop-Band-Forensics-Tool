# analysis/reference.py
from __future__ import annotations

from typing import List, Dict, Any
import math
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Reference ranges & thresholds (based on the table you provided) + lightweight rules.
# NOTE: We only flag metrics that actually exist in your analysed dataframe.
# --------------------------------------------------------------------------------------

# Human-friendly display names for columns we might see in your daily CSV
DISPLAY = {
    "resting_hr": "Resting Heart Rate (bpm)",
    "hrv_ms": "Heart Rate Variability (ms)",
    "sleep_efficiency_pct": "Sleep Efficiency (%)",
    "asleep_min": "Asleep (min)",
    "day_strain": "Strain (score)",
    "sleep_debt_min": "Sleep Debt (min)",
    # Optional if present in your dataset:
    "respiratory_rate": "Respiratory Rate (rpm)",
    "spo2_pct": "Blood Oxygen Saturation (%)",
    "skin_temp_c": "Skin Temperature (°C)",
    "steps": "Step Count (daily)",
    "cadence_spm": "Cadence (steps/min)",
    "pace_min_per_km": "Pace (min/km)",
    "stress_score": "Stress Level (0–100)",
    "recovery_score_pct": "Recovery (% 0–100)",
}

# Normal ranges & thresholds distilled from your table (numeric rules used by the flagger).
# For metrics where personal baseline matters (HRV), we use both absolute and relative rules.
RANGES: Dict[str, Dict[str, Any]] = {
    "resting_hr": {
        "normal_low": 60.0,
        "normal_high": 100.0,
        "flag_low": 50.0,     # bradycardia (non-athlete)
        "flag_high": 100.0,   # tachycardia at rest
        "notes": "Athletes may normally be 40–60 bpm.",
    },
    "hrv_ms": {
        # HRV is highly individual – absolute low is a rough floor; also watch relative drops.
        "absolute_low": 20.0,             # consistently < ~20 ms is poor
        "rel_drop_frac": 0.30,            # >30% drop vs 7-day rolling baseline
        "rolling_window": 7,
        "notes": "Use baseline trends; large sustained drops are concerning.",
    },
    "sleep_efficiency_pct": {
        "normal_low": 85.0,  # <85% is poor
    },
    "asleep_min": {
        # Adults generally need 7–9 h (420–540 min). Tool already has 360 min short-sleep threshold.
        "short_flag": 360.0,     # <6 h
        "very_short_flag": 240.0 # <4 h
    },
    "day_strain": {
        # Heuristic threshold already used in pipeline
        "high_flag": 14.0,
    },
    "sleep_debt_min": {
        "high_flag": 60.0,
    },
    # Optional metrics if present
    "respiratory_rate": {
        "rest_high": 20.0,  # >20 rpm at rest is fast
        "rest_very_high": 24.0,
        "sleep_delta_flag": 2.0,  # >+2 rpm vs personal sleep baseline is suspicious
        "rolling_window": 7,
    },
    "spo2_pct": {
        "low_warn": 95.0,
        "low_flag": 90.0,
        "low_critical": 88.0,
    },
    "skin_temp_c": {
        # Use deviation from person baseline when possible
        "fever_abs": 38.0,        # wrist/skin ≥38°C is very high
        "delta_warn": 1.0,        # +/- 1°C vs baseline notable
        "delta_flag": 2.0,        # +/- 2°C vs baseline strong flag
        "rolling_window": 14,
    },
    "steps": {
        "very_low_day": 1000,     # <1000 steps likely device off/incapacitated
        "very_high_day": 20000,   # >20k steps unusual for many unless special day
    },
    "cadence_spm": {
        "run_lower_bound": 130.0,  # >130 spm suggests running
        "implausibly_high": 200.0, # above this likely artifact
    },
    "pace_min_per_km": {
        # We don't hard-flag pace alone; context with cadence is better.
    },
    "stress_score": {
        "high_zone": 75.0,  # sustained high stress
    },
    "recovery_score_pct": {
        "low_red": 33.0,  # WHOOP-like red zone
    },
}

# Explanations (concise health-condition context you asked for)
EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "resting_hr": {
        "title": "Resting Heart Rate",
        "body": (
            "• High (>100 bpm) at rest: fever/illness, dehydration, anxiety/panic, pain, hyperthyroid, anemia, stimulant use.\n"
            "• Low (<50 bpm) if not athletic: medication (beta-blockers), conduction disease, hypothyroid; if athletic, often normal.\n"
            "• Forensic: nocturnal spikes during claimed sleep can indicate stress, disturbance, or struggle."
        ),
    },
    "hrv_ms": {
        "title": "Heart Rate Variability (RMSSD, ms)",
        "body": (
            "• Low HRV or >30% drop vs baseline: acute stress, overtraining, impending illness, alcohol, poor sleep.\n"
            "• Chronically low HRV may reflect poor recovery or chronic stress. Compare to personal baseline."
        ),
    },
    "sleep_efficiency_pct": {
        "title": "Sleep Efficiency (%)",
        "body": (
            "• <85%: fragmented or poor sleep—insomnia, environmental disturbance, pain, alcohol, sleep disorders.\n"
            "• Forensic: long awake periods can contradict 'I was asleep' claims."
        ),
    },
    "asleep_min": {
        "title": "Total Sleep (min)",
        "body": (
            "• <360 min (<6 h): short sleep; <240 min (<4 h): very short—impairs cognition and recovery.\n"
            "• Could reflect stress, illness, shift work, or device removal."
        ),
    },
    "day_strain": {
        "title": "Daily Strain",
        "body": (
            "• High strain suggests heavy exertion or stress load.\n"
            "• If unexpected given reported behavior, investigate activity timeline."
        ),
    },
    "sleep_debt_min": {
        "title": "Sleep Debt (min)",
        "body": (
            "• >60 min sleep debt indicates accumulating insufficiency; correlates with low HRV, high RHR."
        ),
    },
    "respiratory_rate": {
        "title": "Respiratory Rate (rpm)",
        "body": (
            "• >20 rpm at rest (esp. during sleep) suggests anxiety, pain, infection, or respiratory/cardiac strain.\n"
            "• A rise >2 rpm vs personal sleep baseline often precedes illness."
        ),
    },
    "spo2_pct": {
        "title": "Blood Oxygen Saturation (%)",
        "body": (
            "• <95% is below normal at sea level; <90% is hypoxemia; <88% is serious.\n"
            "• Dips during sleep suggest apnea; sudden drops align with respiratory compromise."
        ),
    },
    "skin_temp_c": {
        "title": "Skin Temperature (°C)",
        "body": (
            "• +2°C above baseline suggests fever/illness; sharp drop to ambient suggests device removal or cold exposure."
        ),
    },
    "steps": {
        "title": "Daily Steps",
        "body": (
            "• <1000 steps: likely device off or immobility; >20k: atypical surge.\n"
            "• Night-time steps contradict sleep; bursts can mark flight/struggle."
        ),
    },
    "cadence_spm": {
        "title": "Cadence (steps/min)",
        "body": (
            "• >130 spm suggests running; >200 spm likely artifact.\n"
            "• Night-time running cadence is suspicious without explanation."
        ),
    },
    "stress_score": {
        "title": "Stress Level (0–100)",
        "body": (
            "• Sustained >75 indicates high physiological stress.\n"
            "• Forensic spikes can mark confrontation, panic, or pain."
        ),
    },
    "recovery_score_pct": {
        "title": "Recovery (%)",
        "body": (
            "• <33% (red): poor recovery—low HRV, high RHR, illness, sleep loss."
        ),
    },
}

# Real case studies to surface in the UI (concise)
CASE_STUDIES: List[Dict[str, Any]] = [
    {
        "title": "State v. Richard Dabate (‘Fitbit Murder’)",
        "device": "Fitbit",
        "signals": ["steps/movement timeline"],
        "blurb": (
            "Victim’s Fitbit showed movement for nearly an hour after the alleged time of death, "
            "contradicting the husband’s account. Conviction hinged partly on wearable timeline."
        ),
    },
    {
        "title": "State v. George Burch / VanderHeyden case",
        "device": "Fitbit",
        "signals": ["steps (alibi)"],
        "blurb": (
            "Boyfriend’s Fitbit recorded ~no steps during the murder window, supporting his alibi; "
            "investigation shifted, and Burch was convicted."
        ),
    },
    {
        "title": "R v. Caroline Nilsson (South Australia)",
        "device": "Apple Watch",
        "signals": ["heart rate pattern, motion"],
        "blurb": (
            "Victim’s watch showed a burst then rapid fall in heart rate indicating time of attack, "
            "contradicting a prolonged struggle claim."
        ),
    },
    {
        "title": "People v. Anthony Aiello (San Jose)",
        "device": "Fitbit",
        "signals": ["heart rate timeline"],
        "blurb": (
            "Victim’s Fitbit spiked then flatlined while the suspect was still on premises per CCTV—"
            "wearable HR timeline placed the fatal event."
        ),
    },
    {
        "title": "Mark ‘Iceman’ Fellows (UK gangland murders)",
        "device": "Garmin Forerunner",
        "signals": ["GPS route, speed"],
        "blurb": (
            "GPS ‘run’ data matched reconnaissance and escape routes; key evidence supporting conviction."
        ),
    },
    {
        "title": "Hussein K. (Germany)",
        "device": "Apple Health",
        "signals": ["stair-climbing/steps"],
        "blurb": (
            "Health app logged climbing consistent with dragging the victim and returning; validated by test reenactment."
        ),
    },
    {
        "title": "Personal Injury (Calgary, 2014)",
        "device": "Fitbit",
        "signals": ["activity level vs. baseline"],
        "blurb": (
            "Plaintiff’s reduced activity quantified damages; early civil use of wearable data in court."
        ),
    },
    {
        "title": "Bartis v. Biomet, Inc.",
        "device": "Fitbit (discovery)",
        "signals": ["step counts"],
        "blurb": (
            "Court compelled production of step counts (but limited scope) to assess claimed mobility impairment."
        ),
    },
    {
        "title": "Ohio Pacemaker Arson Case",
        "device": "Implantable pacemaker",
        "signals": ["heart rhythm vs claimed activity"],
        "blurb": (
            "Heart rhythm data inconsistent with stated frantic escape; charges followed. Not a wearable, but same principle."
        ),
    },
]


def _rolling_baseline(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isinf(x))):
        return ""
    if isinstance(x, float):
        # smart rounding
        if abs(x) >= 100:
            return f"{x:.0f}"
        if abs(x) >= 10:
            return f"{x:.1f}"
        return f"{x:.2f}"
    return str(x)


def _add_flag(flags: List[Dict[str, Any]], date, metric_key: str, value, rule: str, why: str, severity: str = "warn"):
    flags.append({
        "date": pd.to_datetime(date),
        "metric": DISPLAY.get(metric_key, metric_key),
        "value": value,
        "rule": rule,
        "why": why,
        "severity": severity,
    })


def build_flag_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect the analysed daily dataframe and return a tidy table of out-of-range flags
    based on the provided clinical/forensic guidance. Only flags metrics present in df.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a working copy, ensure datetime
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    flags: List[Dict[str, Any]] = []

    # -------- Resting HR --------
    if "resting_hr" in d.columns:
        for _, r in d[["date", "resting_hr"]].dropna().iterrows():
            hr = float(r["resting_hr"])
            if hr > RANGES["resting_hr"]["flag_high"]:
                _add_flag(flags, r["date"], "resting_hr", hr, ">100 bpm at rest",
                          "Tachycardia at rest – fever/illness, dehydration, anxiety, pain, etc.", "warn")
            elif hr < RANGES["resting_hr"]["flag_low"]:
                _add_flag(flags, r["date"], "resting_hr", hr, "<50 bpm at rest",
                          "Bradycardia (normal in athletes; otherwise meds/conduction issues).", "info")

    # -------- HRV (ms) --------
    if "hrv_ms" in d.columns:
        # Absolute low
        for _, r in d[["date", "hrv_ms"]].dropna().iterrows():
            hrv = float(r["hrv_ms"])
            if hrv < RANGES["hrv_ms"]["absolute_low"]:
                _add_flag(flags, r["date"], "hrv_ms", hrv, "<~20 ms (absolute low)",
                          "Very low HRV – high stress, illness, or overreaching.", "warn")
        # Relative drop vs 7-day baseline
        hrv_ma = _rolling_baseline(d["hrv_ms"], RANGES["hrv_ms"]["rolling_window"])
        for idx, (date, hrv, base) in enumerate(zip(d["date"], d["hrv_ms"], hrv_ma)):
            if pd.isna(hrv) or pd.isna(base) or base <= 0:
                continue
            drop = (base - hrv) / base
            if drop >= RANGES["hrv_ms"]["rel_drop_frac"]:
                _add_flag(flags, date, "hrv_ms", float(hrv),
                          f">= {int(RANGES['hrv_ms']['rel_drop_frac']*100)}% drop vs 7-day baseline",
                          "Acute HRV suppression – stress/illness likely.", "warn")

    # -------- Sleep Efficiency --------
    if "sleep_efficiency_pct" in d.columns:
        poor = d["sleep_efficiency_pct"] < RANGES["sleep_efficiency_pct"]["normal_low"]
        for _, r in d.loc[poor, ["date", "sleep_efficiency_pct"]].dropna().iterrows():
            _add_flag(flags, r["date"], "sleep_efficiency_pct", float(r["sleep_efficiency_pct"]),
                      "<85% (poor efficiency)", "Fragmented/inefficient sleep.", "info")

    # -------- Total Sleep (min) --------
    if "asleep_min" in d.columns:
        very_short = d["asleep_min"] < RANGES["asleep_min"]["very_short_flag"]
        short = (d["asleep_min"] >= RANGES["asleep_min"]["very_short_flag"]) & (d["asleep_min"] < RANGES["asleep_min"]["short_flag"])
        for _, r in d.loc[very_short, ["date", "asleep_min"]].dropna().iterrows():
            _add_flag(flags, r["date"], "asleep_min", float(r["asleep_min"]),
                      "<240 min (very short)", "Severely short sleep; impairment likely.", "warn")
        for _, r in d.loc[short, ["date", "asleep_min"]].dropna().iterrows():
            _add_flag(flags, r["date"], "asleep_min", float(r["asleep_min"]),
                      "<360 min (short)", "Short sleep; suboptimal recovery.", "info")

    # -------- Day Strain --------
    if "day_strain" in d.columns:
        hi = d["day_strain"] > RANGES["day_strain"]["high_flag"]
        for _, r in d.loc[hi, ["date", "day_strain"]].dropna().iterrows():
            _add_flag(flags, r["date"], "day_strain", float(r["day_strain"]),
                      f"> {RANGES['day_strain']['high_flag']}", "High exertion/stress load.", "info")

    # -------- Sleep Debt --------
    if "sleep_debt_min" in d.columns:
        hi = d["sleep_debt_min"] > RANGES["sleep_debt_min"]["high_flag"]
        for _, r in d.loc[hi, ["date", "sleep_debt_min"]].dropna().iterrows():
            _add_flag(flags, r["date"], "sleep_debt_min", float(r["sleep_debt_min"]),
                      f"> {int(RANGES['sleep_debt_min']['high_flag'])} min", "Accumulating sleep debt.", "info")

    # -------- Respiratory Rate --------
    if "respiratory_rate" in d.columns:
        rr = d["respiratory_rate"].astype(float)
        # absolute high at rest (we assume daily values are resting/sleeping means if provided)
        hi = rr > RANGES["respiratory_rate"]["rest_high"]
        very_hi = rr > RANGES["respiratory_rate"]["rest_very_high"]
        for _, r in d.loc[very_hi, ["date", "respiratory_rate"]].dropna().iterrows():
            _add_flag(flags, r["date"], "respiratory_rate", float(r["respiratory_rate"]),
                      ">24 rpm (very high at rest)", "Respiratory distress / illness likely.", "warn")
        for _, r in d.loc[hi & ~very_hi, ["date", "respiratory_rate"]].dropna().iterrows():
            _add_flag(flags, r["date"], "respiratory_rate", float(r["respiratory_rate"]),
                      ">20 rpm (high at rest)", "Tachypnea—pain, anxiety, infection, or lung/cardiac strain.", "info")
        # relative rise vs baseline
        rr_base = _rolling_baseline(rr, RANGES["respiratory_rate"]["rolling_window"])
        for date, val, base in zip(d["date"], rr, rr_base):
            if pd.isna(val) or pd.isna(base):
                continue
            if (val - base) >= RANGES["respiratory_rate"]["sleep_delta_flag"]:
                _add_flag(flags, date, "respiratory_rate", float(val),
                          f">= +{RANGES['respiratory_rate']['sleep_delta_flag']} rpm vs 7-day baseline",
                          "RR rise—often precedes illness.", "info")

    # -------- SpO2 --------
    if "spo2_pct" in d.columns:
        spo2 = d["spo2_pct"].astype(float)
        crit = spo2 < RANGES["spo2_pct"]["low_critical"]
        low = (spo2 >= RANGES["spo2_pct"]["low_critical"]) & (spo2 < RANGES["spo2_pct"]["low_flag"])
        warn = (spo2 >= RANGES["spo2_pct"]["low_flag"]) & (spo2 < RANGES["spo2_pct"]["low_warn"])
        for _, r in d.loc[crit, ["date", "spo2_pct"]].dropna().iterrows():
            _add_flag(flags, r["date"], "spo2_pct", float(r["spo2_pct"]),
                      "<88% (critical)", "Severe hypoxemia—urgent if sustained.", "warn")
        for _, r in d.loc[low, ["date", "spo2_pct"]].dropna().iterrows():
            _add_flag(flags, r["date"], "spo2_pct", float(r["spo2_pct"]),
                      "<90% (low)", "Hypoxemia—respiratory compromise possible.", "warn")
        for _, r in d.loc[warn, ["date", "spo2_pct"]].dropna().iterrows():
            _add_flag(flags, r["date"], "spo2_pct", float(r["spo2_pct"]),
                      "<95% (below normal)", "Below normal oxygen for sea level.", "info")

    # -------- Skin Temperature --------
    if "skin_temp_c" in d.columns:
        t = d["skin_temp_c"].astype(float)
        # absolute high
        high_abs = t >= RANGES["skin_temp_c"]["fever_abs"]
        for _, r in d.loc[high_abs, ["date", "skin_temp_c"]].dropna().iterrows():
            _add_flag(flags, r["date"], "skin_temp_c", float(r["skin_temp_c"]),
                      "≥38°C (fever on skin)", "Likely fever/illness (context dependent).", "warn")
        # relative deviation vs baseline
        base = _rolling_baseline(t, RANGES["skin_temp_c"]["rolling_window"])
        for date, val, b in zip(d["date"], t, base):
            if pd.isna(val) or pd.isna(b):
                continue
            delta = float(val - b)
            if abs(delta) >= RANGES["skin_temp_c"]["delta_flag"]:
                _add_flag(flags, date, "skin_temp_c", float(val),
                          f"Δ≥{RANGES['skin_temp_c']['delta_flag']}°C vs 14-day baseline",
                          "Strong deviation—fever or device removal/cold exposure.", "warn")
            elif abs(delta) >= RANGES["skin_temp_c"]["delta_warn"]:
                _add_flag(flags, date, "skin_temp_c", float(val),
                          f"Δ≥{RANGES['skin_temp_c']['delta_warn']}°C vs 14-day baseline",
                          "Notable deviation—check illness or environment.", "info")

    # -------- Steps (daily) --------
    if "steps" in d.columns:
        steps = d["steps"].astype(float)
        very_low = steps < RANGES["steps"]["very_low_day"]
        very_high = steps > RANGES["steps"]["very_high_day"]
        for _, r in d.loc[very_low, ["date", "steps"]].dropna().iterrows():
            _add_flag(flags, r["date"], "steps", float(r["steps"]),
                      "<1000/day (very low)", "Possible device off or immobility.", "info")
        for _, r in d.loc[very_high, ["date", "steps"]].dropna().iterrows():
            _add_flag(flags, r["date"], "steps", float(r["steps"]),
                      ">20000/day (surge)", "Unusual surge—special event/exertion.", "info")

    # -------- Cadence / Pace (if present) --------
    if "cadence_spm" in d.columns:
        cad = d["cadence_spm"].astype(float)
        run = cad > RANGES["cadence_spm"]["run_lower_bound"]
        too_high = cad > RANGES["cadence_spm"]["implausibly_high"]
        for _, r in d.loc[too_high, ["date", "cadence_spm"]].dropna().iterrows():
            _add_flag(flags, r["date"], "cadence_spm", float(r["cadence_spm"]),
                      ">200 spm (implausible)", "Likely artifact or device shake.", "info")
        for _, r in d.loc[run & ~too_high, ["date", "cadence_spm"]].dropna().iterrows():
            _add_flag(flags, r["date"], "cadence_spm", float(r["cadence_spm"]),
                      ">130 spm (running)", "Running cadence—check if expected at this time.", "info")

    # -------- Stress / Recovery (if present) --------
    if "stress_score" in d.columns:
        s = d["stress_score"].astype(float)
        hi = s >= RANGES["stress_score"]["high_zone"]
        for _, r in d.loc[hi, ["date", "stress_score"]].dropna().iterrows():
            _add_flag(flags, r["date"], "stress_score", float(r["stress_score"]),
                      "≥75 (high stress zone)", "Sustained physiological stress.", "info")

    if "recovery_score_pct" in d.columns:
        rec = d["recovery_score_pct"].astype(float)
        low = rec <= RANGES["recovery_score_pct"]["low_red"]
        for _, r in d.loc[low, ["date", "recovery_score_pct"]].dropna().iterrows():
            _add_flag(flags, r["date"], "recovery_score_pct", float(r["recovery_score_pct"]),
                      "≤33% (red)", "Poor recovery—illness, stress, or sleep loss.", "info")

    if not flags:
        return pd.DataFrame()

    out = pd.DataFrame(flags).sort_values(["date", "severity"], ascending=[True, True])
    # Pretty-print numeric values
    out["value"] = out["value"].map(_fmt)
    return out


def normal_ranges_table() -> pd.DataFrame:
    """
    Returns a dataframe mirroring your provided reference table (metric, normal ranges,
    normal fluctuations, abnormal thresholds, notes).
    """
    rows = [
        {
            "Metric (Units)": "Resting Heart Rate (bpm)",
            "Normal Range": "60–100 (athletes 40–60)",
            "Normal Fluctuations": "Lowest in deep sleep; ±1–3 bpm day-to-day around baseline.",
            "Abnormal / Suspicious": ">100 (tachycardia) or <50 if not athletic; sudden spikes >120 at rest/sleep.",
            "Notes": "Females slightly higher on average; ensure true rest; nocturnal spikes during ‘sleep’ are suspicious.",
        },
        {
            "Metric (Units)": "Maximum Heart Rate (exercise, bpm)",
            "Normal Range": "≈220 − age (±10 bpm individual variation).",
            "Normal Fluctuations": "Reached only at intense exertion; fairly stable.",
            "Abnormal / Suspicious": "Near-max during non-exercise periods; inability to reach >85% with max effort.",
            "Notes": "Wrist HR less accurate at very high HR; forensic near-max at rest suggests fight/flight.",
        },
        {
            "Metric (Units)": "HRV (RMSSD, ms)",
            "Normal Range": "Highly individual; ~20–70 typical; 60–100 in young fit; athletes can be >100.",
            "Normal Fluctuations": "Higher at night; varies with stress, sleep, caffeine; watch trends.",
            "Abnormal / Suspicious": "Consistently <~20 ms or ≥30% drop vs baseline for >1 day.",
            "Notes": "Declines with age; compare to personal baseline; device method differs.",
        },
        {
            "Metric (Units)": "Sleep Stages (% of night)",
            "Normal Range": "Light ~50%; Deep 15–20%; REM 20–25%; Awake 5–10%.",
            "Normal Fluctuations": "Deep early night; REM near morning; varies with stress/recovery.",
            "Abnormal / Suspicious": "Deep <10% or REM <15% consistently; many awakenings; stage anomalies.",
            "Notes": "Wearable staging is approximate; use trends; forensic contradictions of claimed sleep.",
        },
        {
            "Metric (Units)": "Respiratory Rate (rpm)",
            "Normal Range": "12–20 at rest; ~13–18 asleep.",
            "Normal Fluctuations": "Very stable when healthy; REM slightly higher; rises with activity.",
            "Abnormal / Suspicious": ">20 at rest; >24 clearly abnormal; +>2 vs baseline (sleep) suggests illness.",
            "Notes": "Most accurate at rest; forensic spikes during sleep suggest anxiety/pain/respiratory strain.",
        },
        {
            "Metric (Units)": "SpO₂ (%)",
            "Normal Range": "95–100 at sea level (older/altitude 90–95).",
            "Normal Fluctuations": "Minor dips at night; typically stays ≥94–95.",
            "Abnormal / Suspicious": "<95 below normal; <90 hypoxemia; <88 serious; repeated >4% dips (apnea).",
            "Notes": "Wrist estimates can artifact with motion; sudden drops + HR change more credible.",
        },
        {
            "Metric (Units)": "Skin Temperature (°C)",
            "Normal Range": "≈33–37 on wrist; device tracks deviations vs baseline.",
            "Normal Fluctuations": "Rises at night; ±1°C typical day-to-day; +0.5°C luteal phase (female).",
            "Abnormal / Suspicious": "≥38°C skin; ±2°C vs baseline; sharp drop to ambient → device off.",
            "Notes": "Peripheral temp lags core; interpret with HR/RR; forensic removal detection.",
        },
        {
            "Metric (Units)": "Step Count (per day)",
            "Normal Range": "Lifestyle dependent; 7k–10k active; 4k–6k common.",
            "Normal Fluctuations": "Diurnal; near-zero during sleep; large day-to-day swings with routine.",
            "Abnormal / Suspicious": "<1000 very low (off/immobile); >20k surge; steps at odd hours vs claimed sleep.",
            "Notes": "Arm-motion artifacts possible; align with timeline/location.",
        },
        {
            "Metric (Units)": "Cadence / Pace",
            "Normal Range": "Walk ~60–115 spm; Run ~160–180 spm; pace varies with fitness.",
            "Normal Fluctuations": "Cadence rises with speed; session-specific.",
            "Abnormal / Suspicious": ">200 spm (artifact); running cadence during supposed rest; implausible pace.",
            "Notes": "Use with steps/GPS; forensic to infer running/chasing.",
        },
        {
            "Metric (Units)": "Stress / Recovery (0–100)",
            "Normal Range": "Device-specific; low stress / high recovery when well-rested.",
            "Normal Fluctuations": "Lowest overnight; daytime spikes with stress; recovery changes day-to-day.",
            "Abnormal / Suspicious": "Sustained stress ≥75 or recovery ≤33%; sudden spikes at key times.",
            "Notes": "Proprietary scales; use relatively and correlate with HR/HRV/sleep.",
        },
    ]
    return pd.DataFrame(rows)


# What this module exposes
__all__ = [
    "build_flag_table",
    "normal_ranges_table",
    "EXPLANATIONS",
    "CASE_STUDIES",
    "DISPLAY",
    "RANGES",
]
