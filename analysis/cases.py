# analysis/cases.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd

# -----------------------
# Case study data model
# -----------------------

@dataclass(frozen=True)
class CaseStudy:
    id: str
    title: str
    year: int | str
    jurisdiction: str
    device: str
    data_types: List[str]        # e.g., ['steps', 'hr', 'gps']
    pattern_tags: List[str]      # heuristics your tool can match
    summary: str                 # short paragraph
    key_points: List[str]        # bullet highlights (concise)
    refs: List[str]              # short ref strings (no external fetch)

# Pattern tag glossary (what your tool can infer)
PATTERN_DESCRIPTIONS: Dict[str, str] = {
    "hr_spike_event": "Sudden heart-rate elevation at rest/overnight.",
    "nocturnal_event_possible": "Overnight disturbance (poor sleep + elevated HR).",
    "steps_evidence_available": "Daily or intraday step logs present.",
    "inactivity_exoneration": "Very low movement during a critical window.",
    "gps_evidence_available": "GPS routes or distances present.",
    "rr_elevated": "Respiratory rate elevated relative to expected rest/sleep.",
    "spo2_drops": "Oxygen saturation dips into abnormal range.",
    "device_removed_possible": "Signals suggest device removal (e.g., temp drop/no HR/no steps).",
}

# -----------------------
# Curated case studies
# -----------------------

CASE_STUDIES: List[CaseStudy] = [

    CaseStudy(
        id="dabate_fitbit_murder",
        title="State v. Richard Dabate — 'Fitbit Murder'",
        year=2022,
        jurisdiction="USA (Connecticut)",
        device="Fitbit",
        data_types=["steps", "timeline"],
        pattern_tags=["steps_evidence_available"],
        summary=(
            "Victim’s Fitbit showed substantial movement after the alleged time of death, "
            "contradicting the suspect’s timeline. Combined digital evidence led to conviction."
        ),
        key_points=[
            "Fitbit logged movement for ~1 hour after claimed death time.",
            "Wearable timelines can corroborate or contradict alibis.",
        ],
        refs=["Fitbit movement vs. stated timeline; conviction upheld."]
    ),

    CaseStudy(
        id="burch_vanderheyden",
        title="State v. George Burch (Nicole VanderHeyden)",
        year=2018,
        jurisdiction="USA (Wisconsin)",
        device="Fitbit",
        data_types=["steps"],
        pattern_tags=["steps_evidence_available", "inactivity_exoneration"],
        summary=(
            "Boyfriend’s Fitbit showed only a handful of steps during the murder window, "
            "supporting his claim that he was asleep; charges refocused on the actual perpetrator."
        ),
        key_points=[
            "Wearable inactivity helped exonerate an initial suspect.",
            "Court accepted wearable records as admissible with proper certification."
        ],
        refs=["Fitbit inactivity consistent with sleeping; conviction of Burch upheld."]
    ),

    CaseStudy(
        id="aiello_fitbit_hr_flatline",
        title="People v. Anthony Aiello (Karen Navarra)",
        year=2018,
        jurisdiction="USA (California)",
        device="Fitbit",
        data_types=["hr", "timeline"],
        pattern_tags=["hr_spike_event", "nocturnal_event_possible"],
        summary=(
            "Victim’s heart rate spiked and then flatlined within minutes; the device’s HR timeline "
            "helped fix time of attack and contradicted the suspect’s alibi."
        ),
        key_points=[
            "HR spike followed by abrupt cessation can mark attack window.",
            "Aligning HR timeline with cameras/ALPR can place suspects."
        ],
        refs=["Fitbit HR timeline used to set time of death."]
    ),

    CaseStudy(
        id="nilsson_applewatch",
        title="R v. Caroline Nilsson (Myrna Nilsson)",
        year="2016–2020",
        jurisdiction="Australia (South Australia)",
        device="Apple Watch",
        data_types=["hr", "motion"],
        pattern_tags=["hr_spike_event", "nocturnal_event_possible"],
        summary=(
            "Apple Watch data captured a brief burst of activity followed by sharp HR decline, "
            "contradicting claims of a prolonged struggle."
        ),
        key_points=[
            "Wearable HR/motion narrowed time-of-assault window to minutes.",
            "Bail denied partly citing smartwatch evidence strength."
        ],
        refs=["Apple Watch HR/motion used to dispute narrative."]
    ),

    CaseStudy(
        id="fellows_garmin_gps",
        title="R v. Mark 'Iceman' Fellows — Garmin Recon",
        year=2019,
        jurisdiction="UK (Manchester)",
        device="Garmin Forerunner",
        data_types=["gps", "routes", "speed"],
        pattern_tags=["gps_evidence_available"],
        summary=(
            "GPS routes from a Garmin watch revealed reconnaissance trips and movements "
            "consistent with the ambush and getaway, forming key evidence."
        ),
        key_points=[
            "Pre-crime recon route matched scene approach and egress.",
            "Speed changes (bike→walk) suggested lying in wait."
        ],
        refs=["Garmin GPS tracklog helped secure conviction."]
    ),

    CaseStudy(
        id="risley_false_report",
        title="Commonwealth v. Jeannine Risley — False Report",
        year=2015,
        jurisdiction="USA (Pennsylvania)",
        device="Fitbit",
        data_types=["steps"],
        pattern_tags=["steps_evidence_available", "nocturnal_event_possible"],
        summary=(
            "Fitbit showed the complainant was up and moving when she reported she was asleep and attacked; "
            "data supported charges for filing a false report."
        ),
        key_points=[
            "Step timeline disproved claimed sleep at incident time.",
            "Wearables can refute fabricated accounts."
        ],
        refs=["Fitbit steps contradicted narrative; misdemeanor charges."]
    ),

    CaseStudy(
        id="husseinK_apple_health",
        title="State v. Hussein K. — Apple Health Stair-Climb",
        year=2018,
        jurisdiction="Germany (Freiburg)",
        device="iPhone Health",
        data_types=["stairs", "motion"],
        pattern_tags=["nocturnal_event_possible"],
        summary=(
            "Health app recorded 'climbing stairs' during the attack window; investigators replicated "
            "the movement at the scene to validate the pattern before conviction."
        ),
        key_points=[
            "Phone motion/altitude events can tie movements to terrain.",
            "On-scene replication validated digital trace."
        ],
        refs=["Apple Health stair-climb aligned with crime scene embankment."]
    ),

    CaseStudy(
        id="calgary_civil_fitbit",
        title="Calgary Personal Injury — Fitbit for Damages",
        year=2014,
        jurisdiction="Canada (Alberta)",
        device="Fitbit",
        data_types=["steps", "activity_level"],
        pattern_tags=["steps_evidence_available"],
        summary=(
            "Plaintiff’s post-injury step/activity data, compared to population norms, supported claims of reduced capacity."
        ),
        key_points=[
            "Wearable data quantified functional loss.",
            "Analytics vs. population benchmarks used in damages."
        ],
        refs=["First reported courtroom use of wearable data in civil matter."]
    ),

    CaseStudy(
        id="bartis_biomet_discovery",
        title="Bartis et al. v. Biomet — Discovery of Step Data",
        year=2020,
        jurisdiction="USA (Missouri, Federal)",
        device="Fitbit",
        data_types=["steps"],
        pattern_tags=["steps_evidence_available"],
        summary=(
            "Court compelled production of daily step counts (but limited scope), balancing relevance with privacy in product-liability litigation."
        ),
        key_points=[
            "Step counts deemed relevant to claimed mobility limits.",
            "Court narrowed discovery to minimize privacy impact."
        ],
        refs=["Order: steps relevant; HR/GPS excluded as overbroad."]
    ),

    CaseStudy(
        id="ohio_pacemaker_arson",
        title="Ohio Pacemaker Arson Case — Cardiac Telemetry",
        year=2016,
        jurisdiction="USA (Ohio)",
        device="Pacemaker (implantable)",
        data_types=["hr_rhythm"],
        pattern_tags=["hr_spike_event"],
        summary=(
            "Pacemaker logs (not a wearable) contradicted the suspect’s account of frantic escape during a fire, leading to charges."
        ),
        key_points=[
            "Medical telemetry can act like a 'witness'.",
            "Timeline contradictions by physiologic data."
        ],
        refs=["Cardiac device trend vs. claimed exertion."]
    ),
]

# -----------------------
# Pattern inference
# -----------------------

def infer_patterns_from_daily(df: pd.DataFrame) -> Dict[str, Any]:

    patterns: Dict[str, Any] = {k: False for k in PATTERN_DESCRIPTIONS.keys()}

    cols = set(c.lower() for c in df.columns)

    # Availability patterns
    if "daily_steps" in cols or "steps" in cols:
        patterns["steps_evidence_available"] = True
    if "gps_distance_km" in cols or "gps_km" in cols or "gps" in cols:
        patterns["gps_evidence_available"] = True

    # Elevated HR + overnight disturbance proxy
    has_elevated_hr = "elevated_hr" in cols and df["elevated_hr"].fillna(False).any()
    has_poor_sleep = "poor_sleep" in cols and df["poor_sleep"].fillna(False).any()
    has_short_sleep = "short_sleep" in cols and df["short_sleep"].fillna(False).any()

    if has_elevated_hr:
        patterns["hr_spike_event"] = True
    if has_elevated_hr and (has_poor_sleep or has_short_sleep):
        patterns["nocturnal_event_possible"] = True

    # Inactivity / exoneration proxy
    low_steps_days = 0
    if "daily_steps" in cols:
        low_steps_days = int((df["daily_steps"].fillna(0) < 1000).sum())
    very_low_strain_days = 0
    if "day_strain" in cols:
        very_low_strain_days = int((pd.to_numeric(df["day_strain"], errors="coerce").fillna(0) < 2.0).sum())

    if low_steps_days >= 1 or very_low_strain_days >= 1:
        patterns["inactivity_exoneration"] = True

    # Respiratory rate / SpO2
    if "respiratory_rate" in cols:
        rr = pd.to_numeric(df["respiratory_rate"], errors="coerce")
        if (rr > 20).any():
            patterns["rr_elevated"] = True

    if "spo2_min" in cols or "spo2" in cols:
        spo2 = pd.to_numeric(df.get("spo2_min", df.get("spo2")), errors="coerce")
        if (spo2 < 90).any():
            patterns["spo2_drops"] = True

    # Device removal (if you later add this column)
    if "device_removed" in cols:
        patterns["device_removed_possible"] = bool(df["device_removed"].fillna(False).any())

    # Counts for reasoning
    patterns["_counts"] = {
        "elevated_hr_days": int(df.get("elevated_hr", pd.Series([])).fillna(False).sum()) if "elevated_hr" in cols else 0,
        "poor_sleep_days": int(df.get("poor_sleep", pd.Series([])).fillna(False).sum()) if "poor_sleep" in cols else 0,
        "short_sleep_days": int(df.get("short_sleep", pd.Series([])).fillna(False).sum()) if "short_sleep" in cols else 0,
        "low_steps_days": low_steps_days,
        "very_low_strain_days": very_low_strain_days,
    }
    return patterns


# -----------------------
# Recommendation logic
# -----------------------

def _score_case(cs: CaseStudy, patterns: Dict[str, Any]) -> Tuple[int, List[str]]:

    reasons: List[str] = []
    score = 0

    def bump(tag: str, pts: int, msg: str):
        nonlocal score
        if patterns.get(tag, False):
            score += pts
            reasons.append(msg)

    # Mapping of tags → cases
    if cs.id in {"aiello_fitbit_hr_flatline", "nilsson_applewatch", "ohio_pacemaker_arson"}:
        bump("hr_spike_event", 3, "HR spike/abrupt change observed in this dataset.")
        bump("nocturnal_event_possible", 2, "Overnight disturbance suggested by HR + sleep flags.")

    if cs.id in {"dabate_fitbit_murder", "risley_false_report", "calgary_civil_fitbit", "bartis_biomet_discovery", "burch_vanderheyden"}:
        bump("steps_evidence_available", 3, "Step/activity data present.")
        if cs.id == "burch_vanderheyden":
            bump("inactivity_exoneration", 3, "Periods of very low movement/inactivity present.")
        if cs.id == "dabate_fitbit_murder":
            # We can't verify 'movement after TOD' without an alleged time; still suggest when steps exist.
            reasons.append("Timeline contradictions can be probed when step logs exist.")
            score += 1

    if cs.id == "fellows_garmin_gps":
        bump("gps_evidence_available", 4, "GPS/route data appears available; consider recon patterns.")

    if cs.id == "husseinK_apple_health":
        bump("nocturnal_event_possible", 2, "Overnight disturbance pattern could align with stair/motion events.")
        # No direct stair count—still useful precedent if motion/altitude fields are available.

    # General medical stress overlays
    if cs.id in {"aiello_fitbit_hr_flatline", "nilsson_applewatch"}:
        if patterns.get("rr_elevated", False) or patterns.get("spo2_drops", False):
            score += 1
            reasons.append("Breathing/O₂ anomalies coincide with HR events.")

    # Device removal precedent (not tied to a single case above, but relevant for analysis generally)
    if patterns.get("device_removed_possible", False):
        reasons.append("Possible device removal detected; use cases as context for interpreting gaps.")
        score += 1

    return score, reasons


def suggest_relevant_cases(df: pd.DataFrame, top_n: int = 4) -> List[Dict[str, Any]]:

    patterns = infer_patterns_from_daily(df)
    scored: List[Tuple[int, CaseStudy, List[str]]] = []
    for cs in CASE_STUDIES:
        score, reasons = _score_case(cs, patterns)
        if score > 0:
            scored.append((score, cs, reasons))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, cs, reasons in scored[:top_n]:
        out.append({
            "id": cs.id,
            "title": cs.title,
            "year": cs.year,
            "jurisdiction": cs.jurisdiction,
            "device": cs.device,
            "data_types": cs.data_types,
            "summary": cs.summary,
            "key_points": cs.key_points,
            "refs": cs.refs,
            "matched_score": score,
            "matched_reasons": reasons,
        })
    # Always include at least 2 civil-use precedents if nothing matched strongly
    if not out:
        out = [c for c in [
            {
                "id": cs.id, "title": cs.title, "year": cs.year, "jurisdiction": cs.jurisdiction,
                "device": cs.device, "data_types": cs.data_types, "summary": cs.summary,
                "key_points": cs.key_points, "refs": cs.refs, "matched_score": 1,
                "matched_reasons": ["Useful civil precedent when step/activity data exists."]
            }
            for cs in CASE_STUDIES if cs.id in {"calgary_civil_fitbit", "bartis_biomet_discovery"}
        ]]
    return out


__all__ = [
    "CaseStudy",
    "CASE_STUDIES",
    "PATTERN_DESCRIPTIONS",
    "infer_patterns_from_daily",
    "suggest_relevant_cases",
]
