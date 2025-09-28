# ui/report.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def export_report_pdf(dest: Path, case_name: str, hashes: dict, summary: dict, flags_tbl: pd.DataFrame | None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(dest), pagesize=A4)
    W, H = A4
    y = H - 2*cm

    def line(txt, size=12, color=colors.black, dy=14):
        nonlocal y
        c.setFont("Helvetica", size)
        c.setFillColor(color)
        c.drawString(2*cm, y, txt)
        y -= dy

    line(f"Wearable Forensics — Case: {case_name}", 14)
    line("Data Integrity (SHA-256):", 12)
    for label, sha in hashes.items():
        line(f"  - {label}: {sha}", 10)

    line("Summary:", 12, dy=12)
    for k, v in (summary or {}).items():
        line(f"  - {k}: {v}", 10)

    if flags_tbl is not None and not flags_tbl.empty:
        line("Flagged Out-of-Range Events (first 15):", 12, dy=12)
        # show first few concise rows
        for _, row in flags_tbl.head(15).iterrows():
            line(f"  {row.get('date','')}: {row.get('metric','')} = {row.get('value','')}  [{row.get('reason','')}]", 9, dy=11)
    else:
        line("No out-of-range flags.", 12)

    c.showPage()
    c.save()
