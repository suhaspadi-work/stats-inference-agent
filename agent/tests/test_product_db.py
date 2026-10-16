"""
M1.5 step P1: the product's SQLite layer (product/db.py): the safe migration of an old database, stage reports on
messages, the explanations table. db.DB_PATH is patched to a temporary file.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

import db  # noqa: E402

MESSAGE_COLUMNS = ["id", "session_id", "question_text", "report_text", "test_result_json", "created_at",
                   "stage_reports_json"]
EXPLANATION_COLUMNS = ["id", "session_id", "message_id", "round", "method_id", "question", "answer", "source",
                       "created_at"]


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = tmp_path / "t.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    return p


def _columns(path, table):
    conn = sqlite3.connect(path)
    names = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    conn.close()
    return names


def _session(name="data"):
    return db.create_session("ana", name, f"thread-{name}")


def test_init_db_twice(path):
    db.init_db()
    db.init_db()
    assert _columns(path, "messages") == MESSAGE_COLUMNS
    assert _columns(path, "explanations") == EXPLANATION_COLUMNS


def test_an_old_database_migrates(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
                 "dataset_name TEXT NOT NULL, thread_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL "
                 "REFERENCES sessions(id), question_text TEXT NOT NULL, report_text TEXT, test_result_json TEXT, "
                 "created_at TEXT NOT NULL)")
    conn.execute("INSERT INTO sessions (username, dataset_name, thread_id, created_at) VALUES ('ana', 'old', 't', "
                 "'2026-01-01T00:00:00')")
    conn.execute("INSERT INTO messages (session_id, question_text, report_text, test_result_json, created_at) "
                 "VALUES (1, 'old question', 'old report', NULL, '2026-01-01T00:00:01')")
    conn.commit()
    conn.close()
    db.init_db()
    db.init_db()
    assert _columns(path, "messages") == MESSAGE_COLUMNS
    (row,) = db.list_messages(1)
    assert (row["question_text"], row["report_text"], row["stage_reports_json"]) == ("old question", "old report", None)
    assert db.list_sessions("ana")[0]["dataset_name"] == "old"


def test_record_message(path):
    db.init_db()
    session = _session()
    reports = [{"stage": "stage1", "title": "Stage 1", "sections": [{"heading": "Checked", "text": "x"}]}]
    first = db.record_message(session, "q1", "r1", {"p_value": 0.01}, stage_reports=reports)
    second = db.record_message(session, "q2", "r2", None)
    assert isinstance(first, int) and second > first
    one, two = db.list_messages(session)
    assert json.loads(one["stage_reports_json"]) == reports
    assert json.loads(one["test_result_json"]) == {"p_value": 0.01}
    assert two["stage_reports_json"] is None and two["test_result_json"] is None


def test_a_value_json_cannot_serialize(path):
    db.init_db()
    session = _session()
    message_id = db.record_message(session, "q", "r", {"levels": {"a"}, "p": np.float32(0.5)},
                                   stage_reports=[{"odd": {1, 2}}])
    (row,) = db.list_messages(session)
    assert row["id"] == message_id
    stored = json.loads(row["test_result_json"])
    assert stored["levels"] == str({"a"}) and stored["p"] == str(np.float32(0.5))
    assert json.loads(row["stage_reports_json"]) == [{"odd": str({1, 2})}]


def test_explanations(path):
    db.init_db()
    session, other = _session("a"), _session("b")
    message = db.record_message(session, "q", "r", None)
    first = db.record_explanation(session, message, 1, "student_t_test", None, "It compares.", "model")
    second = db.record_explanation(session, message, 2, "ols", "Why OLS?", "Because.", "glossary_fallback")
    db.record_explanation(other, None, 1, "ols", None, "Elsewhere.", "model")
    assert second > first
    rows = db.list_explanations(session)
    assert [dict(r) | {"created_at": None} for r in rows] == [
        {"id": first, "session_id": session, "message_id": message, "round": 1, "method_id": "student_t_test",
         "question": None, "answer": "It compares.", "source": "model", "created_at": None},
        {"id": second, "session_id": session, "message_id": message, "round": 2, "method_id": "ols",
         "question": "Why OLS?", "answer": "Because.", "source": "glossary_fallback", "created_at": None},
    ]
    assert all(r["created_at"] for r in rows)
    assert [r["answer"] for r in db.list_explanations(other)] == ["Elsewhere."]


def test_explanations_with_the_same_time_keep_their_order(path, monkeypatch):
    db.init_db()
    session = _session()

    class Frozen:
        @staticmethod
        def now(tz=None):
            from datetime import datetime
            return datetime(2026, 1, 1, tzinfo=tz)
    monkeypatch.setattr(db, "datetime", Frozen)
    ids = [db.record_explanation(session, None, 1, "ols", None, f"a{i}", "model") for i in range(3)]
    assert [r["id"] for r in db.list_explanations(session)] == ids
