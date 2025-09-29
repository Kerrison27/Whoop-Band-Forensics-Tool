# database/db.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import sqlite3

# Location of the SQLite file
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.sqlite3"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def init_db() -> None:
    with _connect() as con:
        # --- cases
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # --- investigators
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS investigators (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              email TEXT
            );
            """
        )

        # --- files uploaded to a case
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              case_id INTEGER NOT NULL,
              kind TEXT NOT NULL,            -- 'physio' | 'sleep'
              original_name TEXT NOT NULL,
              stored_path TEXT NOT NULL,
              sha256 TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
            );
            """
        )

        # --- dataset import log
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_import_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              case_id INTEGER NOT NULL,
              investigator_id INTEGER,
              filename TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
              FOREIGN KEY(investigator_id) REFERENCES investigators(id) ON DELETE SET NULL
            );
            """
        )

        # --- daily analysis
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              case_id INTEGER NOT NULL,
              date TEXT NOT NULL,

              resting_hr REAL,
              hrv_ms REAL,
              day_strain REAL,
              asleep_min REAL,
              sleep_efficiency_pct REAL,
              sleep_debt_min REAL,

              -- rule-based flags
              elevated_hr INTEGER,
              low_hrv INTEGER,
              high_strain_flag INTEGER,
              poor_sleep INTEGER,
              short_sleep INTEGER,
              sleep_debt_flag INTEGER,
              any_anomaly INTEGER,

              -- ML fields (optional; your app currently doesn't persist these)
              iso_score REAL,
              iso_flag INTEGER,
              rhr_pred REAL,
              rhr_residual REAL,
              rhr_z REAL,
              rhr_flag INTEGER,

              UNIQUE(case_id, date),
              FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
            );
            """
        )

        # speed up lookups
        con.execute("CREATE INDEX IF NOT EXISTS idx_daily_case_date ON daily(case_id, date);")


def list_cases() -> Iterable[sqlite3.Row]:
    with _connect() as con:
        cur = con.execute("SELECT id, name, created_at FROM cases ORDER BY id DESC;")
        return cur.fetchall()


def get_case_id_by_name(name: str) -> Optional[int]:
    with _connect() as con:
        r = con.execute("SELECT id FROM cases WHERE name = ?;", (name,)).fetchone()
        return int(r["id"]) if r else None


def create_case(name: str) -> int:
    with _connect() as con:
        cur = con.execute("INSERT INTO cases (name) VALUES (?);", (name,))
        return int(cur.lastrowid)


def add_file(case_id: int, kind: str, original_name: str, stored_path, sha256: str) -> int:
    stored_path = str(stored_path)
    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO files (case_id, kind, original_name, stored_path, sha256)
            VALUES (?, ?, ?, ?, ?);
            """,
            (case_id, kind, original_name, stored_path, sha256),
        )
        # optional: also log to dataset_import_log
        con.execute(
            """
            INSERT INTO dataset_import_log (case_id, filename, sha256)
            VALUES (?, ?, ?);
            """,
            (case_id, original_name, sha256),
        )
        return int(cur.lastrowid)


def _coerce_bool(v) -> int:
    # pandas bool -> int for SQLite
    if v is True:
        return 1
    if v is False:
        return 0
    try:
        return 1 if int(v) != 0 else 0
    except Exception:
        return 0


def save_daily(case_id: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # Allowed columns (must match table)
    cols = [
        "resting_hr", "hrv_ms", "day_strain", "asleep_min",
        "sleep_efficiency_pct", "sleep_debt_min",
        "elevated_hr", "low_hrv", "high_strain_flag", "poor_sleep",
        "short_sleep", "sleep_debt_flag", "any_anomaly",
        # ML optionals — will only be included if present in the dicts
        "iso_score", "iso_flag", "rhr_pred", "rhr_residual", "rhr_z", "rhr_flag",
    ]

    with _connect() as con:
        for r in rows:
            if "date" not in r:
                continue
            date = str(r["date"])

            # Build column/value lists dynamically
            field_names = []
            placeholders = []
            values = []

            for c in cols:
                if c in r:
                    field_names.append(c)
                    placeholders.append("?")
                    val = r[c]
                    if c.endswith("_flag") or c in ("elevated_hr", "low_hrv", "poor_sleep", "short_sleep", "any_anomaly", "high_strain_flag"):
                        val = _coerce_bool(val)
                    values.append(val)

            if not field_names:
                # at minimum we need something to write besides the date
                continue

            # UPSERT
            assignments = ", ".join(f"{c}=excluded.{c}" for c in field_names)
            sql = (
                f"INSERT INTO daily (case_id, date, {', '.join(field_names)}) "
                f"VALUES (?, ?, {', '.join(placeholders)}) "
                f"ON CONFLICT(case_id, date) DO UPDATE SET {assignments};"
            )
            con.execute(sql, (case_id, date, *values))


def daily_summary(case_id: int) -> Dict[str, int]:
    q = """
    SELECT
      SUM(COALESCE(elevated_hr,0))              AS elevated_hr_events,
      SUM(COALESCE(low_hrv,0))                  AS low_hrv_events,
      SUM(COALESCE(high_strain_flag,0))         AS high_strain_events,
      SUM(COALESCE(poor_sleep,0))               AS poor_sleep_events,
      SUM(COALESCE(short_sleep,0))              AS short_sleep_events,
      SUM(COALESCE(sleep_debt_flag,0))          AS sleep_debt_events,
      SUM(CASE WHEN COALESCE(any_anomaly,0)=1 THEN 1 ELSE 0 END) AS days_with_any_anomaly
    FROM daily
    WHERE case_id = ?;
    """
    with _connect() as con:
        r = con.execute(q, (case_id,)).fetchone()
        if not r:
            return {}
        return {k: int(r[k]) if r[k] is not None else 0 for k in r.keys()}
