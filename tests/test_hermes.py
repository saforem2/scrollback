"""Tests for the Hermes Agent source adapter (synthetic SQLite databases)."""

import json
import os
import sqlite3
from pathlib import Path

from scrollback.archivefmt import to_archive_json
from scrollback.sources.hermes import HermesSource


def _db(tmp_path: Path) -> Path:
    path = tmp_path / ".hermes" / "state.db"
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, title TEXT, model TEXT,
                parent_session_id TEXT, started_at REAL, ended_at REAL,
                last_activity_at REAL, cwd TEXT, input_tokens INTEGER,
                output_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, reasoning_tokens INTEGER,
                estimated_cost_usd REAL, actual_cost_usd REAL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                content TEXT, tool_call_id TEXT, tool_calls TEXT,
                tool_name TEXT, effect_disposition TEXT, timestamp REAL,
                reasoning TEXT, reasoning_content TEXT, active INTEGER,
                compacted INTEGER, display_kind TEXT, display_metadata TEXT,
                display_order INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "20260102_030405_abcdef", "tui", "Fix widgets", "gpt-5", None,
                1_735_786_800.0, None, 1_735_786_900.0, "/work/widgets",
                10, 20, 30, 40, 5, 0.25, None,
            ),
        )
        tool_calls = [{
            "id": "call-1", "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"pytest"}'},
        }]
        rows = [
            (1, "user", "fix it", None, None, None, None, None, 1, 0, None, None, 1),
            (2, "assistant", "working", None, json.dumps(tool_calls), None, None,
             "thinking", 1, 0, None, None, 2),
            (3, "tool", "passed", "call-1", None, "terminal", "success", None,
             1, 0, None, None, 3),
            (4, "user", "secret scaffold", None, None, None, None, None,
             1, 0, "hidden", None, 4),
            (5, "assistant", "rewound", None, None, None, None, None,
             0, 0, None, None, 5),
            (6, "assistant", "model context", None, None, None, None, None,
             1, 0, None, '{"model_only":true}', 6),
            (7, "system", "visible malformed metadata", None, None, None, None, None,
             1, 0, None, "not-json", 7),
        ]
        conn.executemany(
            """INSERT INTO messages
               (id, session_id, role, content, tool_call_id, tool_calls, tool_name,
                effect_disposition, reasoning_content, active, compacted, display_kind,
                display_metadata, display_order, timestamp)
               VALUES (?, '20260102_030405_abcdef', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1735786800)""",
            rows,
        )
    return path


def test_hermes_lists_and_loads_visible_messages(tmp_path):
    path = _db(tmp_path)
    source = HermesSource(db_path=path)
    before = os.stat(path).st_mtime_ns

    sessions = list(source.list_sessions())
    assert len(sessions) == 1
    meta = sessions[0]
    assert meta.source == "hermes"
    assert meta.title == "Fix widgets"
    assert meta.directory == "/work/widgets"
    assert meta.agent == "tui"
    assert meta.message_count == 4
    assert meta.tokens_cache_read == 30
    assert meta.cost == 0.25

    session = source.load_session(meta.id)
    assert session is not None
    assert session.message_count == len(session.messages) == 4
    assert [message.role for message in session.messages] == [
        "user", "assistant", "tool", "system",
    ]
    assert [part.type for part in session.messages[1].parts] == ["reasoning", "text", "tool"]
    assert session.messages[1].parts[2].tool_name == "terminal"
    assert "pytest" in session.messages[1].parts[2].text
    assert session.messages[2].parts[0].tool_status == "success"
    assert '"session"' in to_archive_json(session)
    assert os.stat(path).st_mtime_ns == before


def test_hermes_pages_messages_and_builds_resume_command(tmp_path):
    source = HermesSource(db_path=_db(tmp_path))
    assert [message.id for message in source.load_messages(
        "20260102_030405_abcdef", offset=1, limit=1
    )] == ["2"]
    session = next(iter(source.list_sessions()))
    assert source.resume_command(session) == (
        "cd /work/widgets && hermes --resume 20260102_030405_abcdef"
    )


def test_hermes_tolerates_minimal_legacy_schema(tmp_path):
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)"
        )
        conn.execute("INSERT INTO sessions VALUES ('legacy', 1735786800)")
        conn.execute("INSERT INTO messages VALUES (1, 'legacy', 'user', ?)", (
            "\x00json:" + json.dumps([{"type": "text", "text": "hello"}]),
        ))

    source = HermesSource(db_path=path)
    session = source.load_session("legacy")
    assert session is not None
    assert session.title == "(untitled)"
    assert session.message_count == 1
    assert session.messages[0].text == "hello"
