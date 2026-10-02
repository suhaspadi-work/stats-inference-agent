"""
SQLite persistence layer for the product app: who has which analysis
sessions, and what Q&A history exists within each. Deliberately separate
from the agent's own LocalDiskStore (which holds dataset CSVs and
LangGraph's own thread checkpoints) -- this is purely product-level
metadata: whose session is this, what did they ask, what did they get back.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "app_data.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates the schema if it doesn't already exist. Safe to call on every app startup."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            thread_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            question_text TEXT NOT NULL,
            report_text TEXT,
            test_result_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session(username: str, dataset_name: str, thread_id: str) -> int:
    """Called once, when a user uploads a new dataset. Returns the new session's id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO sessions (username, dataset_name, thread_id, created_at) VALUES (?, ?, ?, ?)",
        (username, dataset_name, thread_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def list_sessions(username: str) -> list[sqlite3.Row]:
    """Past sessions for the sidebar, most recent first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE username = ? ORDER BY created_at DESC", (username,)
    ).fetchall()
    conn.close()
    return rows


def get_session(session_id: int) -> sqlite3.Row | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return row


def record_message(session_id: int, question_text: str, report_text: str | None, test_result: dict | None) -> None:
    """Called once a question has run to completion (or failed), for the session's history."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (session_id, question_text, report_text, test_result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            session_id, question_text, report_text,
            json.dumps(test_result) if test_result else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_messages(session_id: int) -> list[sqlite3.Row]:
    """A session's Q&A history, in chronological order."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)
    ).fetchall()
    conn.close()
    return rows