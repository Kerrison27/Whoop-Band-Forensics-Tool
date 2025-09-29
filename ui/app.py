# ui/app.py
from __future__ import annotations

from pathlib import Path
import sys
from urllib.parse import quote_plus  # fallback search links

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# pastels for charts
COLORWAY = ["#6C93B0", "#A4B59C", "#C5A0B6", "#BFAE88", "#95B8D6", "#BFB7C7"]

pio.templates["jpn_pastel"] = dict(
    layout=dict(
        colorway=COLORWAY,
        font=dict(family="Inter, 'Noto Sans JP', system-ui", size=13, color="#1C1F24"),
        paper_bgcolor="rgba(0,0,0,0)",   # let our CSS show through
        plot_bgcolor="#F5F7F2",
        xaxis=dict(gridcolor="#E5E1DA", zerolinecolor="#E5E1DA"),
        yaxis=dict(gridcolor="#E5E1DA", zerolinecolor="#E5E1DA"),
        margin=dict(l=40,r=20,t=50,b=40),
        hoverlabel=dict(bgcolor="#FAFAF7", bordercolor="#ECE7E1", font_color="#1C1F24"),
    )
)
px.defaults.template = "jpn_pastel"
px.defaults.color_discrete_sequence = COLORWAY


# -------------------------------------------------
# Ensure project root is on sys.path (fixes imports)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_ROOT = ROOT / "data" / "cases"

# ----------------
# Local modules
# ----------------
try:
    from database.db import (
        init_db, list_cases, create_case, get_case_id_by_name, add_file,
        save_daily, daily_summary
    )
except ModuleNotFoundError:
    st.error(
        "Couldn't import `database.db`. Make sure your project has a "
        "`database/db.py` file and that you're running Streamlit from the project root.\n\n"
        "Tip: In a terminal at the project root, run:\n"
        "`streamlit run ui/app.py`"
    )
    st.stop()

from analysis.pipeline import (
    load_physio_csv, load_sleep_csv, merge_daily,
    detect_anomalies, Thresholds, save_case_outputs
)

# Case-study suggestions
try:
    from analysis.cases import suggest_relevant_cases, PATTERN_DESCRIPTIONS  # noqa: F401
    HAVE_CASE_SUGGEST = True
except Exception:
    HAVE_CASE_SUGGEST = False

# Reference ranges / explanations / curated case studies
try:
    from analysis.reference import build_flag_table, EXPLANATIONS, CASE_STUDIES, normal_ranges_table
    HAVE_REFERENCE = True
except Exception:
    HAVE_REFERENCE = False

# ------------- boot -------------
st.set_page_config(page_title="Wearable Forensics", layout="wide")

# inject CSS (robust Windows utf-8 read)
css_path = ROOT / "ui" / "styles.css"
if css_path.exists():
    try:
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8', errors='ignore')}</style>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# init DB
init_db()

# ------------- sidebar: case selection / creation -------------
st.sidebar.header("Case")
cases = list(list_cases())
case_options = ["(Create new case)"] + [f"{c['id']}: {c['name']}" for c in cases]
choice = st.sidebar.selectbox("Select case", options=case_options)

if choice == "(Create new case)":
    with st.sidebar.form("new_case_form", clear_on_submit=True):
        new_name = st.text_input("New case name", placeholder="e.g., Case_2025_08_JohnDoe")
        submitted = st.form_submit_button("Create case")
    if submitted:
        if not new_name.strip():
            st.sidebar.error("Case name is required.")
            st.stop()
        if get_case_id_by_name(new_name):
            st.sidebar.error("Case name already exists.")
            st.stop()
        new_id = create_case(new_name)
        st.sidebar.success(f"Created case #{new_id}: {new_name}. Re-run and select it.")
        st.stop()
else:
    case_id = int(choice.split(":")[0])
    case_dir = CASES_ROOT / str(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    st.sidebar.write(f"**Case folder:** `{case_dir}`")

if choice == "(Create new case)":
    st.markdown('<div class="app-title">Wearable Forensics</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Create a case in the sidebar to begin.</div>', unsafe_allow_html=True)
    st.stop()

# ------------- header -------------
st.markdown('<div class="app-title">Wearable Forensics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload WHOOP/fitness CSVs, run the analysis, and explore interactive charts.</div>',
    unsafe_allow_html=True
)

# ------------- tabs -------------
if HAVE_REFERENCE:
    tab_upload, tab_dash, tab_files, tab_explain = st.tabs(
        ["Upload & Analyse", "Dashboard", "Outputs", "Explain & Case Studies"]
    )
else:
    tab_upload, tab_dash, tab_files = st.tabs(["Upload & Analyse", "Dashboard", "Outputs"])

# ========== TAB 1: Upload & Analyse ==========
with tab_upload:
    with st.expander("Analysis thresholds", expanded=False):
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        th = Thresholds(
            elevated_hr_bpm=c1.number_input("Elevated HR (bpm) >", value=90.0, step=1.0),
            low_hrv_ms=c2.number_input("Low HRV (ms) <", value=40.0, step=1.0),
            high_strain=c3.number_input("High Strain >", value=14.0, step=0.1),
            poor_sleep_eff_pct=c4.number_input("Poor Sleep Efficiency (%) <", value=85.0, step=1.0),
            short_sleep_min=c5.number_input("Short Sleep (min) <", value=360.0, step=10.0),
            sleep_debt_min=c6.number_input("Sleep Debt (min) >", value=60.0, step=10.0),
        )
        st.caption("Adjust thresholds and re-run analysis after uploading data.")

    st.markdown('<div class="card"><h4>Upload data</h4>', unsafe_allow_html=True)
    with st.form("upload_form"):
        physio_file = st.file_uploader("Physiological CSV", type=["csv"])
        sleep_file = st.file_uploader("Sleeps CSV", type=["csv"])
        run_btn = st.form_submit_button("Save & Run Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        if not physio_file or not sleep_file:
            st.error("Please upload both CSV files.")
            st.stop()

        # Save raw files
        uploads_dir = case_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        physio_path = uploads_dir / "physiological_cycles.csv"
        sleep_path = uploads_dir / "sleeps.csv"
        physio_path.write_bytes(physio_file.getvalue())
        sleep_path.write_bytes(sleep_file.getvalue())

        # Load + normalise
        physio_df, physio_sha = load_physio_csv(physio_path)
        sleep_df, sleep_sha = load_sleep_csv(sleep_path)

        # Save file records
        add_file(case_id, "physio", physio_file.name, physio_path, physio_sha)
        add_file(case_id, "sleep", sleep_file.name, sleep_path, sleep_sha)

        # Merge + anomalies
        merged = merge_daily(physio_df, sleep_df)
        analysed = detect_anomalies(merged, th)

        # --- ML: learn personal baseline & score anomalies (multivariate) ---
        from analysis.ml import train_or_load_model, score_with_model, add_robust_bands
        try:
            model = train_or_load_model(case_dir, analysed)
            analysed = score_with_model(analysed, model)
            analysed = add_robust_bands(analysed)  # optional bands for charts
        except Exception as e:
            st.warning(f"ML scoring skipped: {e}")

        # Persist to DB (avoid adding transient ML cols)
        rows = analysed.copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        rows_for_db = rows.drop(columns=["ml_score", "ml_flag", "ml_top_feats", "ml_score_pct"], errors="ignore")
        save_daily(case_id, rows_for_db.to_dict(orient="records"))

        # Save case outputs (CSV + PNG charts) — includes ML cols
        save_case_outputs(case_dir, analysed)

        st.success("✅ Data saved and analysis completed.")
        st.dataframe(analysed.tail(10), use_container_width=True)

# Helper to read case daily CSV (for dashboard/outputs)
def _read_case_daily(case_dir: Path) -> pd.DataFrame | None:
    p = case_dir / "output" / "daily.csv"
    if not p.exists():
        return None
    df_local = pd.read_csv(p)
    if "date" in df_local.columns:
        df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
    return df_local

# ========== TAB 2: Dashboard ==========
with tab_dash:
    st.markdown('<div class="card"><h4>Anomaly summary <span class="badge">Live</span></h4>', unsafe_allow_html=True)
    summary = daily_summary(case_id)
    if not summary:
        st.info("No analysis saved yet for this case. Go to **Upload & Analyse** first.")
    else:
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Elevated HR", summary.get('elevated_hr_events', 0))
        m2.metric("Low HRV", summary.get('low_hrv_events', 0))
        m3.metric("High Strain", summary.get('high_strain_events', 0))
        m4.metric("Poor Sleep", summary.get('poor_sleep_events', 0))
        m5.metric("Short Sleep", summary.get('short_sleep_events', 0))
        m6.metric("Sleep Debt", summary.get('sleep_debt_events', 0))
        m7.metric("Days w/ Any", summary.get('days_with_any_anomaly', 0))
    st.markdown('</div>', unsafe_allow_html=True)

    daily_df = _read_case_daily(case_dir)
    if daily_df is None or daily_df.empty:
        st.info("Charts will appear after you run an analysis.")
    else:
        st.markdown('<div class="card"><h4>Interactive charts</h4>', unsafe_allow_html=True)
        left, right = st.columns(2)

        # Resting HR
        with left:
            if "resting_hr" in daily_df.columns:
                fig = px.line(daily_df, x="date", y="resting_hr", markers=True, title="Resting HR (bpm)")
                st.plotly_chart(fig, use_container_width=True)

        # HRV
        with right:
            if "hrv_ms" in daily_df.columns:
                fig = px.line(daily_df, x="date", y="hrv_ms", markers=True, title="HRV (ms)")
                st.plotly_chart(fig, use_container_width=True)

        left2, right2 = st.columns(2)
        with left2:
            if "day_strain" in daily_df.columns:
                fig = px.line(daily_df, x="date", y="day_strain", markers=True, title="Strain")
                st.plotly_chart(fig, use_container_width=True)
        with right2:
            if "sleep_efficiency_pct" in daily_df.columns:
                fig = px.line(daily_df, x="date", y="sleep_efficiency_pct", markers=True, title="Sleep Efficiency (%)")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ML panel ---
        if "ml_score" in daily_df.columns or "ml_score_pct" in daily_df.columns:
            st.markdown('<div class="card"><h4>ML anomaly score</h4>', unsafe_allow_html=True)
            ycol = "ml_score_pct" if "ml_score_pct" in daily_df.columns else "ml_score"
            title = "ML anomaly score (higher = more anomalous)"
            fig = px.line(daily_df, x="date", y=ycol, markers=True, title=title)
            st.plotly_chart(fig, use_container_width=True)

            if "ml_flag" in daily_df.columns and daily_df["ml_flag"].any():
                st.markdown("**Flagged by ML**")
                ml_recent = daily_df[daily_df["ml_flag"] == 1].sort_values("date", ascending=False).head(20)
                show_cols = [c for c in ["date", "resting_hr", "hrv_ms", "day_strain",
                                         "sleep_efficiency_pct", "asleep_min", "ml_score", "ml_score_pct", "ml_flag"]
                             if c in ml_recent.columns]
                st.dataframe(ml_recent[show_cols], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Recent anomalies table
        st.markdown('<div class="card"><h4>Recent anomalies</h4>', unsafe_allow_html=True)
        flags = ["elevated_hr", "low_hrv", "high_strain_flag", "poor_sleep", "short_sleep", "sleep_debt_flag"]
        existing_flags = [f for f in flags if f in daily_df.columns]
        recent = (
            daily_df[daily_df[existing_flags].any(axis=1)]
            .sort_values("date", ascending=False)
            .head(20)
            if existing_flags else pd.DataFrame()
        )
        if recent.empty:
            st.write("No recent anomalies.")
        else:
            show_cols = ["date", "resting_hr", "hrv_ms", "day_strain", "sleep_efficiency_pct", "asleep_min", "sleep_debt_min"] + existing_flags
            show_cols = [c for c in show_cols if c in recent.columns]
            st.dataframe(recent[show_cols], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ========== TAB 3: Outputs ==========
with tab_files:
    st.markdown('<div class="card"><h4>Downloads</h4>', unsafe_allow_html=True)
    out_dir = case_dir / "output"
    plots_dir = out_dir / "plots"
    daily_csv = out_dir / "daily.csv"

    if daily_csv.exists():
        csv_bytes = daily_csv.read_bytes()
        st.download_button(
            label="Download daily CSV",
            data=csv_bytes,
            file_name="daily.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No outputs yet. Run an analysis first.")

    if plots_dir.exists():
        st.markdown("#### Charts (PNG)")
        cols = st.columns(2)
        plot_files = [
            "resting_hr.png", "hrv_ms.png", "day_strain.png",
            "sleep_efficiency_pct.png", "asleep_min.png", "sleep_debt_min.png",
        ]
        i = 0
        for name in plot_files:
            p = plots_dir / name
            if p.exists():
                cols[i % 2].image(str(p), use_container_width=True, caption=name)
                i += 1
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Simple report export (Markdown) ---
    if daily_csv.exists():
        if st.button("Export case report (Markdown)"):
            import io, datetime as _dt
            df_rep = pd.read_csv(daily_csv)
            n_days = len(df_rep)
            last = df_rep["date"].max() if "date" in df_rep.columns else "N/A"

            # Pull summary counts from DB
            summ = daily_summary(case_id) or {}
            lines = []
            lines.append(f"# Case Report — Case #{case_id}")
            lines.append(f"_Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
            lines.append("")
            lines.append(f"**Days analysed:** {n_days}")
            lines.append(f"**Last date:** {last}")
            lines.append("")
            lines.append("## Flag counts")
            for k, label in [
                ("elevated_hr_events", "Elevated HR"),
                ("low_hrv_events", "Low HRV"),
                ("high_strain_events", "High Strain"),
                ("poor_sleep_events", "Poor Sleep"),
                ("short_sleep_events", "Short Sleep"),
                ("sleep_debt_events", "Sleep Debt"),
                ("days_with_any_anomaly", "Days with any anomaly"),
            ]:
                lines.append(f"- {label}: {summ.get(k, 0)}")
            lines.append("")
            lines.append("## Notes")
            lines.append("- Thresholds and ML parameters are configurable in the app.")
            lines.append("- See `output/plots/` for PNG charts and `output/daily.csv` for the full table.")
            report_md = "\n".join(lines).encode("utf-8")

            st.download_button(
                "Download report.md",
                data=report_md,
                file_name="case_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ========== TAB 4: Explain & Case Studies ==========
if HAVE_REFERENCE:
    with tab_explain:
        st.markdown('<div class="card"><h4>Out-of-range flags & explanations</h4>', unsafe_allow_html=True)
        explain_df = _read_case_daily(case_dir)
        if explain_df is None or explain_df.empty:
            st.info("No analysis yet. Go to **Upload & Analyse** first.")
        else:
            flags_tbl = build_flag_table(explain_df)
            if flags_tbl.empty:
                st.success("No out-of-range flags based on reference ranges and trends.")
            else:
                st.dataframe(flags_tbl, use_container_width=True)

            st.markdown("### What the flags mean")
            cols = st.columns(2)
            keys = [
                "resting_hr", "hrv_ms", "sleep_efficiency_pct", "asleep_min",
                "respiratory_rate", "spo2_pct", "skin_temp_c", "steps",
                "cadence_spm", "stress_score", "recovery_score_pct"
            ]
            for i, k in enumerate(keys):
                if k in EXPLANATIONS:
                    with cols[i % 2].expander(EXPLANATIONS[k]["title"], expanded=False):
                        st.write(EXPLANATIONS[k]["body"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h4>Reference ranges & thresholds</h4>', unsafe_allow_html=True)
        try:
            ref_tbl = normal_ranges_table()
            st.dataframe(ref_tbl, use_container_width=True)
        except Exception:
            st.info("Reference table unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Dynamic "Relevant case studies" (based on current case data)
        if HAVE_CASE_SUGGEST:
            st.markdown('<div class="card"><h4>Relevant case studies</h4>', unsafe_allow_html=True)
            try:
                suggest_df = _read_case_daily(case_dir)
                if suggest_df is None or suggest_df.empty:
                    st.info("No case-study suggestions yet (need analysed data).")
                else:
                    recs = suggest_relevant_cases(suggest_df)
                    if not recs:
                        st.info("No case-study suggestions yet.")
                    else:
                        for r in recs:
                            header = f"{r.get('title','(Untitled)')}  •  {r.get('jurisdiction','')}  •  {r.get('year','')}  —  {', '.join(r.get('data_types', []))}"
                            with st.expander(header):
                                st.write(r.get("summary", ""))
                                reasons = r.get("matched_reasons") or []
                                if reasons:
                                    st.markdown("**Why this surfaced:** " + "; ".join(reasons))
                                kps = r.get("key_points") or []
                                if kps:
                                    st.markdown("**Key points**")
                                    for kp in kps:
                                        st.markdown(f"- {kp}")
                                refs = r.get("refs") or []
                                if refs:
                                    st.caption("References: " + " | ".join(refs))
                                # Link to web coverage (use provided or fallback search)
                                search_url = r.get("search_url")
                                if not search_url:
                                    q = f"{r.get('title','')} {r.get('jurisdiction','')} {r.get('device','')}"
                                    search_url = "https://www.google.com/search?q=" + quote_plus(q)
                                st.markdown(f"[Open web coverage for this case]({search_url})")
            except Exception as e:
                st.warning(f"Could not generate case-study suggestions: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Static, curated case studies list
        st.markdown('<div class="card"><h4>Real case studies</h4>', unsafe_allow_html=True)
        for cs in CASE_STUDIES:
            with st.expander(f"{cs['title']} — {cs['device']} • {', '.join(cs['signals'])}", expanded=False):
                st.write(cs["blurb"])
        st.markdown('</div>', unsafe_allow_html=True)
