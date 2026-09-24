"""Tests for OpenCode V2's projected-session SQLite schema."""

import json
import os
import sqlite3
from pathlib import Path

from scrollback.sources.opencode import OpenCodeSource


def _make_v2_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE session_v2 (
                id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT,
                directory TEXT, title TEXT, time_created INTEGER,
                time_updated INTEGER, agent TEXT, model TEXT,
                cost REAL, tokens_input INTEGER, tokens_output INTEGER,
                tokens_cache_read INTEGER, tokens_cache_write INTEGER,
                tokens_reasoning INTEGER
            );
            CREATE TABLE session_message (
                id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER,
                time_created INTEGER, time_updated INTEGER, data TEXT
            );
            -- V2 leaves the legacy tables present after migration.
            CREATE TABLE session (id TEXT PRIMARY KEY);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT);
            """
        )
        conn.execute(
            """INSERT INTO session_v2 VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "ses_v2", "project", None, "/work/v2", "V2 session", 1000, 9000,
                "build", json.dumps({"id": "gpt-5", "providerID": "openai"}),
                0.5, 10, 20, 30, 40, 5,
            ),
        )
        messages = [
            ("msg_user", "user", 1, {"text": "fix it", "time": {"created": 1100}}),
            ("msg_assistant", "assistant", 2, {
                "agent": "build",
                "model": {"id": "gpt-5", "providerID": "openai"},
                "time": {"created": 1200},
                "content": [
                    {"type": "reasoning", "text": "inspect first"},
                    {"type": "text", "text": "working"},
                    {"type": "tool", "id": "call_1", "name": "shell", "state": {
                        "status": "completed", "input": {"command": "pytest"},
                        "content": [{"type": "text", "text": "passed"}],
                    }},
                ],
            }),
            ("msg_system", "system", 3, {
                "text": "location changed", "time": {"created": 1300},
            }),
            ("msg_compact", "compaction", 4, {
                "summary": "Earlier work summary", "time": {"created": 1400},
            }),
        ]
        conn.executemany(
            "INSERT INTO session_message VALUES (?, 'ses_v2', ?, ?, ?, ?, ?)",
            [(mid, kind, seq, created["time"]["created"], created["time"]["created"],
              json.dumps(created)) for mid, kind, seq, created in messages],
        )


def test_opencode_v2_lists_and_loads_projected_messages(tmp_path):
    path = tmp_path / "opencode.db"
    _make_v2_db(path)
    source = OpenCodeSource(db_path=path)
    before = os.stat(path).st_mtime_ns

    listed = list(source.list_sessions())
    assert len(listed) == 1
    assert listed[0].id == "ses_v2"
    assert listed[0].message_count == 4
    assert listed[0].model == "gpt-5"
    assert listed[0].tokens_cache_read == 30

    session = source.load_session("ses_v2")
    assert session is not None
    assert session.message_count == len(session.messages) == 4
    assert [message.role for message in session.messages] == [
        "user", "assistant", "system", "assistant",
    ]
    assert [part.type for part in session.messages[1].parts] == [
        "reasoning", "text", "tool",
    ]
    tool = session.messages[1].parts[2]
    assert tool.tool_name == "shell"
    assert tool.tool_status == "completed"
    assert "pytest" in tool.text
    assert "passed" in tool.text
    assert session.messages[3].parts[0].type == "compaction"
    assert os.stat(path).st_mtime_ns == before


def test_opencode_v2_pages_messages_by_sequence(tmp_path):
    path = tmp_path / "opencode.db"
    _make_v2_db(path)
    source = OpenCodeSource(db_path=path)

    page = source.load_messages("ses_v2", offset=1, limit=2)
    assert [message.id for message in page] == ["msg_assistant", "msg_system"]
