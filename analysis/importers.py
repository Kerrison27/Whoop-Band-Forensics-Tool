# analysis/importers.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
from .pipeline import _sha256_file, _to_numeric

def load_whoop_json(pathlike):
    p = Path(pathlike)
    sha = _sha256_file(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        data = [data]

    df = pd.json_normalize(data)
    # Map common keys (extend as needed)
    colmap = {
        "date": ["date", "calendar_date", "start", "start_time"],
        "resting_hr": ["resting_hr", "resting_heart_rate_bpm","resting_heart_rate"],
        "hrv_ms": ["hrv_ms","hrv","heart_rate_variability_ms"],
        "respiratory_rate": ["respiratory_rate","resp_rate","breaths_per_min"],
        "spo2_pct": ["spo2_pct","spo2","blood_oxygen_pct","spo2_percent"],
        "sleep_efficiency_pct": ["sleep_efficiency_pct","sleep_efficiency"],
        "asleep_min": ["asleep_min","asleep_minutes","sleep_asleep_minutes"],
        "day_strain": ["day_strain","strain","strain_score"],
        "skin_temp_c": ["skin_temp_c","wrist_temp_c","skin_temperature_c"],
        "steps": ["steps","step_count"],
        "cadence_spm": ["cadence_spm"]
    }
    out = {}
    for std, candidates in colmap.items():
        for c in candidates:
            if c in df.columns:
                out[std] = df[c]
                break
        if std not in out:
            out[std] = np.nan

    out = pd.DataFrame(out)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    num_cols = [c for c in out.columns if c != "date"]
    for c in num_cols:
        out[c] = _to_numeric(out[c])
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out, sha
