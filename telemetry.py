import sqlite3
from datetime import datetime
from pathlib import Path

DB_FILE = Path("./telemetry.db").resolve()


def init_telemetry_db():
    """Creates the SQLite telemetry table if it does not exist."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prompt TEXT,
                dispatches INTEGER,
                latency_sec REAL,
                chart_generated INTEGER
            )
        """)
        conn.commit()


def log_agent_run(prompt: str, dispatches: int, latency_sec: float, chart_generated: bool):
    """Logs a completed Supervisor execution trace."""
    init_telemetry_db()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (timestamp, prompt, dispatches, latency_sec, chart_generated)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                prompt[:200],
                dispatches,
                round(latency_sec, 2),
                1 if chart_generated else 0,
            ),
        )
        conn.commit()


def get_telemetry_summary() -> dict:
    """Returns aggregate metrics and the 20 most recent runs."""
    init_telemetry_db()
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT 20"
        ).fetchall()

        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_runs,
                 COALESCE(AVG(latency_sec), 0.0) as avg_latency,
                COALESCE(AVG(dispatches), 0.0) as avg_dispatches,
                COALESCE(SUM(chart_generated), 0) as total_charts
            FROM agent_runs
        """).fetchone()

    return {
        "total_runs": stats["total_runs"],
        "avg_latency": round(stats["avg_latency"], 2),
        "avg_dispatches": round(stats["avg_dispatches"], 1),
        "total_charts": stats["total_charts"],
        "recent_runs": [dict(r) for r in rows],
    }