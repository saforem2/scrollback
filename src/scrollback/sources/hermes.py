"""Hermes Agent source adapter (read-only SQLite).

Hermes stores session metadata and messages in ``~/.hermes/state.db``.
The database is opened with SQLite's ``mode=ro`` URI so browsing cannot
create files, migrate schemas, or interfere with the agent's writer.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models import Message, Part, Session, _to_dt
from .base import Source

_DEFAULT_DB = Path.home() / ".hermes" / "state.db"
_CONTENT_JSON_PREFIX = "\x00json:"
_ROLES = {"user", "assistant", "system", "tool"}


def _env_db() -> Path:
    override = os.environ.get("SCROLLBACK_HERMES_DB")
    return Path(override).expanduser() if override else _DEFAULT_DB


class HermesSource(Source):
    name = "hermes"
    label = "Hermes"

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _env_db()

    def is_available(self) -> bool:
        return self._db_path.is_file()

    def location(self) -> Path | None:
        return self._db_path if self.is_available() else None

    def resume_command(self, session: Session) -> str | None:
        command = f"hermes --resume {shlex.quote(session.id)}"
        if session.directory:
            return f"cd {shlex.quote(session.directory)} && {command}"
        return command

    def _connect(self) -> sqlite3.Connection:
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    def list_sessions(self) -> Iterator[Session]:
        if not self.is_available():
            return iter(())
        return self._list_sessions()

    def _list_sessions(self) -> Iterator[Session]:
        with self._connect() as conn:
            columns = self._columns(conn, "sessions")
            message_columns = self._columns(conn, "messages")
            if not {"id", "started_at"} <= columns:
                return
            select = _session_select(columns)
            count = _message_count_subquery(message_columns)
            rows = conn.execute(
                f"SELECT {select}, {count} AS visible_message_count "
                "FROM sessions s ORDER BY updated_at DESC, s.started_at DESC, s.id DESC"
            ).fetchall()
        for row in rows:
            yield _session_from_row(row)

    def load_session(self, session_id: str) -> Session | None:
        meta = self.load_session_meta(session_id)
        if meta is None:
            return None
        messages = tuple(self.load_messages(session_id))
        return Session(
            id=meta.id,
            source=meta.source,
            title=meta.title,
            directory=meta.directory,
            created=meta.created,
            updated=meta.updated,
            model=meta.model,
            agent=meta.agent,
            parent_id=meta.parent_id,
            message_count=meta.message_count,
            cost=meta.cost,
            tokens_input=meta.tokens_input,
            tokens_output=meta.tokens_output,
            tokens_cache_read=meta.tokens_cache_read,
            tokens_cache_write=meta.tokens_cache_write,
            tokens_reasoning=meta.tokens_reasoning,
            messages=messages,
            raw=meta.raw,
        )

    def load_session_meta(self, session_id: str) -> Session | None:
        if not self.is_available():
            return None
        with self._connect() as conn:
            columns = self._columns(conn, "sessions")
            message_columns = self._columns(conn, "messages")
            if not {"id", "started_at"} <= columns:
                return None
            row = conn.execute(
                f"SELECT {_session_select(columns)}, "
                f"{_message_count_subquery(message_columns)} AS visible_message_count "
                "FROM sessions s WHERE s.id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row is not None else None

    def load_messages(
        self, session_id: str, *, offset: int = 0, limit: int | None = None
    ) -> list[Message]:
        if not self.is_available():
            return []
        with self._connect() as conn:
            columns = self._columns(conn, "messages")
            if not {"id", "session_id", "role"} <= columns:
                return []
            rows = conn.execute(
                f"SELECT * FROM messages m WHERE m.session_id = ? "
                f"AND {_visible_message_predicate(columns, 'm')} "
                f"ORDER BY {_message_order(columns, 'm')} LIMIT ? OFFSET ?",
                (session_id, -1 if limit is None else limit, offset),
            ).fetchall()
        return [_message_from_row(row) for row in rows]


def _column(columns: set[str], name: str, fallback: str = "NULL") -> str:
    return f"s.{name}" if name in columns else fallback


def _session_select(columns: set[str]) -> str:
    updated = _coalesce(columns, ("last_activity_at", "ended_at", "started_at"))
    title_columns = [f"NULLIF(s.{name}, '')" for name in ("title", "display_name") if name in columns]
    title = (
        f"COALESCE({', '.join(title_columns)}, '(untitled)')"
        if title_columns
        else "'(untitled)'"
    )
    cost = _coalesce(columns, ("actual_cost_usd", "estimated_cost_usd"))
    return ", ".join(
        (
            "s.id AS id",
            f"{title} AS title",
            f"{_column(columns, 'cwd')} AS directory",
            "s.started_at AS created_at",
            f"{updated} AS updated_at",
            f"{_column(columns, 'model')} AS model",
            f"{_column(columns, 'source')} AS agent",
            f"{_column(columns, 'parent_session_id')} AS parent_id",
            f"{cost} AS cost",
            f"{_column(columns, 'input_tokens')} AS tokens_input",
            f"{_column(columns, 'output_tokens')} AS tokens_output",
            f"{_column(columns, 'cache_read_tokens')} AS tokens_cache_read",
            f"{_column(columns, 'cache_write_tokens')} AS tokens_cache_write",
            f"{_column(columns, 'reasoning_tokens')} AS tokens_reasoning",
        )
    )


def _coalesce(
    columns: set[str], names: tuple[str, ...], *, fallback: str = "NULL"
) -> str:
    available = [_column(columns, name) for name in names if name in columns]
    if not available:
        return fallback
    if len(available) == 1:
        return f"COALESCE({available[0]}, {fallback})"
    return f"COALESCE({', '.join(available)}, {fallback})"


def _visible_message_predicate(columns: set[str], alias: str = "messages") -> str:
    conditions = []
    if "active" in columns:
        conditions.append(f"COALESCE({alias}.active, 1) = 1")
    if "display_kind" in columns:
        conditions.append(f"COALESCE({alias}.display_kind, '') <> 'hidden'")
    if "display_metadata" in columns:
        conditions.append(
            f"(NOT json_valid({alias}.display_metadata) OR "
            f"COALESCE(json_extract({alias}.display_metadata, '$.model_only'), 0) = 0)"
        )
    conditions.append(f"{alias}.role IN ('user', 'assistant', 'system', 'tool')")
    return " AND ".join(conditions)


def _message_count_subquery(columns: set[str]) -> str:
    if not {"session_id", "role"} <= columns:
        return "0"
    return (
        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND "
        f"{_visible_message_predicate(columns, 'm')})"
    )


def _message_order(columns: set[str], alias: str) -> str:
    if "display_order" in columns:
        return f"COALESCE({alias}.display_order, {alias}.id), {alias}.id"
    if "timestamp" in columns:
        return f"{alias}.timestamp, {alias}.id"
    return f"{alias}.id"


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=str(row["id"]),
        source="hermes",
        title=row["title"] or "(untitled)",
        directory=row["directory"],
        created=_to_dt(_epoch_ms(row["created_at"])),
        updated=_to_dt(_epoch_ms(row["updated_at"])),
        model=row["model"],
        agent=row["agent"],
        parent_id=row["parent_id"],
        message_count=row["visible_message_count"],
        cost=row["cost"],
        tokens_input=row["tokens_input"],
        tokens_output=row["tokens_output"],
        tokens_cache_read=row["tokens_cache_read"],
        tokens_cache_write=row["tokens_cache_write"],
        tokens_reasoning=row["tokens_reasoning"],
    )


def _epoch_ms(value: Any) -> Any:
    return value * 1000 if isinstance(value, (int, float)) else value


def _message_from_row(row: sqlite3.Row) -> Message:
    raw = dict(row)
    role = raw.get("role") if raw.get("role") in _ROLES else "assistant"
    parts: list[Part] = []
    reasoning = raw.get("reasoning_content") or raw.get("reasoning")
    if reasoning:
        parts.append(Part(id=f"{row['id']}:reasoning", type="reasoning", text=str(reasoning)))

    content = _decode_content(raw.get("content"))
    text = _render_content(content)
    if role == "tool":
        parts.append(
            Part(
                id=f"{row['id']}:tool-result",
                type="tool",
                text=text,
                tool_name=raw.get("tool_name"),
                tool_status=raw.get("effect_disposition") or "result",
                raw={"tool_call_id": raw.get("tool_call_id")},
            )
        )
    elif text:
        parts.append(Part(id=f"{row['id']}:text", type="text", text=text))

    for index, call in enumerate(_json_list(raw.get("tool_calls"))):
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name")
        arguments = function.get("arguments", call.get("arguments"))
        parts.append(
            Part(
                id=str(call.get("id") or f"{row['id']}:tool-call:{index}"),
                type="tool",
                text=_render_tool_call(name, arguments),
                tool_name=name,
                tool_status="call",
                raw=call,
            )
        )

    return Message(
        id=str(row["id"]),
        role=role,
        created=_to_dt(_epoch_ms(raw.get("timestamp"))),
        parts=tuple(parts),
        raw={
            key: value
            for key, value in raw.items()
            if key
            not in {
                "content",
                "display_identity",
                "reasoning",
                "reasoning_content",
                "tool_calls",
            }
            and not isinstance(value, (bytes, bytearray))
        },
    )


def _decode_content(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(value[len(_CONTENT_JSON_PREFIX) :])
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _render_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif item is not None:
                texts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(texts)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)


def _json_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _render_tool_call(name: str | None, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            pass
    rendered = json.dumps(arguments, ensure_ascii=False) if not isinstance(arguments, str) else arguments
    return f"$ {name or 'tool'} {rendered or ''}".rstrip()
