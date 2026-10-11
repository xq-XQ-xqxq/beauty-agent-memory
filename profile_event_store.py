"""SQLite-backed dual-channel storage: Profile (mutable) + Event (immutable).

设计要点：
- Profile 通道按 (user_id, profile_key) 唯一约束存储；SINGLE 类型直接覆盖；
  SET 类型在调用层负责合并/移除集合元素，存储层只持久化最终 JSON。
- Event 通道只追加，永不更新或删除；按 created_at 排序输出时间线。
- 所有写入都携带 source_turn_hash，用于幂等去重（与 processed_turn_store 思路一致）。
- 不引入向量数据库；新方案通过 Profile key 精确读取、Event 按时间线扫描，
  避免现有方案因向量检索召回不准导致的混入问题。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from memory_schema import MemoryKeySpec

from profile_event_schema import (
    EVENT_SPEC_MAP,
    PROFILE_SPEC_MAP,
    get_spec,
    is_event_key,
    is_profile_key,
)


SCHEMA_VERSION = "profile-event-v1"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileEventStore:
    """Two-table SQLite store: user_profile (mutable) + user_event (immutable)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------
    # Connection & schema bootstrap
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT NOT NULL,
                    profile_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    source_event_id TEXT,
                    source_turn_hash TEXT,
                    updated_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    PRIMARY KEY (user_id, profile_key)
                );

                CREATE TABLE IF NOT EXISTS user_event (
                    event_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT,
                    description TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    conversation_id TEXT,
                    scene_major TEXT,
                    scene_minor TEXT,
                    source_turn_hash TEXT,
                    created_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    PRIMARY KEY (event_id)
                );

                CREATE TABLE IF NOT EXISTS user_profile_history (
                    user_id TEXT NOT NULL,
                    profile_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    source_turn_hash TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_conflict (
                    user_id TEXT NOT NULL,
                    profile_key TEXT NOT NULL,
                    old_value_json TEXT NOT NULL,
                    new_value_json TEXT NOT NULL,
                    old_evidence_json TEXT NOT NULL,
                    new_evidence_json TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_user_event_user_type_time
                    ON user_event(user_id, event_type, created_at);
                CREATE INDEX IF NOT EXISTS idx_user_profile_user
                    ON user_profile(user_id);
                CREATE INDEX IF NOT EXISTS idx_profile_history_user_key
                    ON user_profile_history(user_id, profile_key);
                CREATE INDEX IF NOT EXISTS idx_conflict_user
                    ON memory_conflict(user_id, resolved);
                """
            )
            # 迁移：给已有 user_event 表加 resolution_status 列
            try:
                connection.execute("SELECT resolution_status FROM user_event LIMIT 1")
            except sqlite3.OperationalError:
                connection.execute(
                    "ALTER TABLE user_event ADD COLUMN resolution_status TEXT DEFAULT NULL"
                )

    # ------------------------------------------------------------------
    # Profile channel
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str, profile_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json, evidence_json, source_event_id, updated_at FROM user_profile "
                "WHERE user_id = ? AND profile_key = ?",
                (user_id, profile_key),
            ).fetchone()
        if row is None:
            return None
        return {
            "value": _loads(row["value_json"]),
            "evidence": _loads(row["evidence_json"]),
            "source_event_id": row["source_event_id"],
            "updated_at": row["updated_at"],
        }

    def get_all_profile(self, user_id: str) -> dict[str, dict[str, Any]]:
        """Return {profile_key: {value, evidence, source_event_id, updated_at}}."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT profile_key, value_json, evidence_json, source_event_id, updated_at "
                "FROM user_profile WHERE user_id = ? ORDER BY profile_key",
                (user_id,),
            ).fetchall()
        return {
            row["profile_key"]: {
                "value": _loads(row["value_json"]),
                "evidence": _loads(row["evidence_json"]),
                "source_event_id": row["source_event_id"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def upsert_profile(
        self,
        user_id: str,
        profile_key: str,
        value: dict[str, Any],
        evidence: list[str],
        source_event_id: str | None = None,
        source_turn_hash: str | None = None,
    ) -> str:
        """SINGLE/SET 都按 key 覆盖；调用方负责 SET 的合并语义。"""

        if not is_profile_key(profile_key):
            raise ValueError(f"{profile_key} 不是 Profile 通道的 key")
        updated_at = _utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profile (
                    user_id, profile_key, value_json, evidence_json,
                    source_event_id, source_turn_hash, updated_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, profile_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    evidence_json = excluded.evidence_json,
                    source_event_id = excluded.source_event_id,
                    source_turn_hash = excluded.source_turn_hash,
                    updated_at = excluded.updated_at,
                    schema_version = excluded.schema_version
                """,
                (
                    user_id,
                    profile_key,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    json.dumps(list(evidence), ensure_ascii=False),
                    source_event_id,
                    source_turn_hash,
                    updated_at,
                    SCHEMA_VERSION,
                ),
            )
        return updated_at

    # ------------------------------------------------------------------
    # Event channel
    # ------------------------------------------------------------------

    def add_event(
        self,
        user_id: str,
        event_type: str,
        fields: dict[str, Any],
        evidence: list[str],
        description: str = "",
        occurred_at: str | None = None,
        conversation_id: str | None = None,
        scene_major: str | None = None,
        scene_minor: str | None = None,
        source_turn_hash: str | None = None,
        event_id: str | None = None,
    ) -> str:
        """Append an immutable event record. Returns the event_id."""

        if not is_event_key(event_type):
            raise ValueError(f"{event_type} 不是 Event 通道的 key")
        spec: MemoryKeySpec = EVENT_SPEC_MAP[event_type]
        eid = event_id or _new_event_id(user_id, event_type)
        body = description or _describe_event(spec, fields)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_event (
                    event_id, user_id, event_type, occurred_at, description,
                    fields_json, evidence_json, conversation_id, scene_major, scene_minor,
                    source_turn_hash, created_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    eid,
                    user_id,
                    event_type,
                    occurred_at,
                    body,
                    json.dumps(fields, ensure_ascii=False, sort_keys=True),
                    json.dumps(list(evidence), ensure_ascii=False),
                    conversation_id,
                    scene_major,
                    scene_minor,
                    source_turn_hash,
                    _utcnow(),
                    SCHEMA_VERSION,
                ),
            )
        return eid

    def get_events(
        self,
        user_id: str,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read events for a user, optionally filtered by type, newest first."""

        sql = (
            "SELECT event_id, user_id, event_type, occurred_at, description, "
            "fields_json, evidence_json, conversation_id, scene_major, scene_minor, "
            "source_turn_hash, created_at FROM user_event WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_to_event(row) for row in rows]

    def get_all_events_grouped(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return {event_type: [event, ...]} for prompt context building."""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self.get_events(user_id):
            grouped.setdefault(event["event_type"], []).append(event)
        return grouped

    def has_event_with_signature(
        self,
        user_id: str,
        event_type: str,
        signature: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Pre-filter existing events whose match_fields signature equals the input.

        Used by the dedup module before invoking the LLM same/different judgment.
        Returns full event rows so the LLM can compare fields, not just signatures.
        """

        from profile_event_schema import match_signature

        candidates: list[dict[str, Any]] = []
        for event in self.get_events(user_id, event_type=event_type):
            existing_sig = match_signature(event_type, event["fields"])
            if existing_sig == signature:
                candidates.append(event)
        return candidates

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def list_users(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM user_profile "
                "UNION SELECT DISTINCT user_id FROM user_event ORDER BY user_id"
            ).fetchall()
        return [row["user_id"] for row in rows]

    def reset_user(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM user_event WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM user_profile_history WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM memory_conflict WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------
    # Profile history & rollback
    # ------------------------------------------------------------------

    def archive_profile(
        self, user_id: str, profile_key: str,
        value: dict[str, Any], evidence: list[str],
        source_turn_hash: str | None = None,
    ) -> None:
        """在 UPDATE 之前归档旧值到 user_profile_history。"""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO user_profile_history "
                "(user_id, profile_key, value_json, evidence_json, archived_at, source_turn_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, profile_key,
                 json.dumps(value, ensure_ascii=False, sort_keys=True),
                 json.dumps(list(evidence), ensure_ascii=False),
                 _utcnow(), source_turn_hash),
            )

    def get_profile_history(
        self, user_id: str, profile_key: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT value_json, evidence_json, archived_at, source_turn_hash "
                "FROM user_profile_history WHERE user_id = ? AND profile_key = ? "
                "ORDER BY archived_at DESC LIMIT ?",
                (user_id, profile_key, limit),
            ).fetchall()
        return [
            {"value": _loads(r["value_json"]), "evidence": _loads(r["evidence_json"]),
             "archived_at": r["archived_at"], "source_turn_hash": r["source_turn_hash"]}
            for r in rows
        ]

    def rollback_profile(
        self, user_id: str, profile_key: str, to_archived_at: str | None = None,
    ) -> bool:
        """回滚 Profile 到历史值。不指定时间则回滚到最近一条。"""
        history = self.get_profile_history(user_id, profile_key, limit=1 if not to_archived_at else 50)
        if not history:
            return False
        target = history[0] if not to_archived_at else next(
            (h for h in history if h["archived_at"] == to_archived_at), None
        )
        if target is None:
            return False
        self.upsert_profile(
            user_id=user_id, profile_key=profile_key,
            value=target["value"], evidence=target["evidence"],
            source_turn_hash=target.get("source_turn_hash"),
        )
        return True

    # ------------------------------------------------------------------
    # Memory conflict
    # ------------------------------------------------------------------

    def record_conflict(
        self, user_id: str, profile_key: str,
        old_value: dict[str, Any], new_value: dict[str, Any],
        old_evidence: list[str], new_evidence: list[str],
    ) -> None:
        """记录 SINGLE Profile 枚举值冲突。"""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memory_conflict "
                "(user_id, profile_key, old_value_json, new_value_json, "
                "old_evidence_json, new_evidence_json, detected_at, resolved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (user_id, profile_key,
                 json.dumps(old_value, ensure_ascii=False, sort_keys=True),
                 json.dumps(new_value, ensure_ascii=False, sort_keys=True),
                 json.dumps(old_evidence, ensure_ascii=False),
                 json.dumps(new_evidence, ensure_ascii=False),
                 _utcnow()),
            )

    def get_conflicts(self, user_id: str, unresolved_only: bool = True) -> list[dict[str, Any]]:
        sql = ("SELECT profile_key, old_value_json, new_value_json, "
               "old_evidence_json, new_evidence_json, detected_at, resolved "
               "FROM memory_conflict WHERE user_id = ?")
        if unresolved_only:
            sql += " AND resolved = 0"
        sql += " ORDER BY detected_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, (user_id,)).fetchall()
        return [
            {"profile_key": r["profile_key"],
             "old_value": _loads(r["old_value_json"]),
             "new_value": _loads(r["new_value_json"]),
             "old_evidence": _loads(r["old_evidence_json"]),
             "new_evidence": _loads(r["new_evidence_json"]),
             "detected_at": r["detected_at"],
             "resolved": bool(r["resolved"])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Event resolution tagging
    # ------------------------------------------------------------------

    def tag_event_resolution(
        self, event_id: str, status: str,
    ) -> None:
        """标记事件的解决状态：resolved / unresolved。"""
        with self._connect() as connection:
            connection.execute(
                "UPDATE user_event SET resolution_status = ? WHERE event_id = ?",
                (status, event_id),
            )

    def tag_conversation_events(
        self, user_id: str, conversation_id: str, status: str,
    ) -> int:
        """批量标记某会话下所有事件的解决状态。返回受影响行数。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE user_event SET resolution_status = ? "
                "WHERE user_id = ? AND conversation_id = ?",
                (status, user_id, conversation_id),
            )
            return cursor.rowcount

    def close(self) -> None:
        # SQLite connections are short-lived (per-call) here; nothing to close.
        return None

    def __enter__(self) -> "ProfileEventStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "user_id": row["user_id"],
        "event_type": row["event_type"],
        "occurred_at": row["occurred_at"],
        "description": row["description"],
        "fields": _loads(row["fields_json"]),
        "evidence": _loads(row["evidence_json"]),
        "conversation_id": row["conversation_id"],
        "scene_major": row["scene_major"],
        "scene_minor": row["scene_minor"],
        "source_turn_hash": row["source_turn_hash"],
        "created_at": row["created_at"],
    }


def _new_event_id(user_id: str, event_type: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    safe_user = "".join(ch if ch.isalnum() else "_" for ch in user_id)[:32]
    safe_type = event_type.replace(".", "_")
    return f"EVT_{safe_user}_{safe_type}_{stamp}"


def _describe_event(spec: MemoryKeySpec, fields: dict[str, Any]) -> str:
    """Compact one-line description used when LLM does not provide one."""

    body = "；".join(f"{key}={val}" for key, val in fields.items() if val not in (None, "", []))
    return f"[{spec.dimension}][{spec.key}][{spec.scope}] {body}".strip()


def render_profile_text(profile_key: str, value: dict[str, Any] | list[Any]) -> str:
    spec = PROFILE_SPEC_MAP.get(profile_key)
    if spec is None:
        return json.dumps(value, ensure_ascii=False)
    # 兼容 dict 和 list 两种格式（SET 类型初始值可能是 list）
    if isinstance(value, dict):
        body = "；".join(f"{k}={v}" for k, v in value.items() if v not in (None, "", []))
    else:
        body = "；".join(str(v) for v in value if v not in (None, "", []))
    return f"[Profile][{profile_key}][{spec.scope}] {body}"


def render_event_text(event: dict[str, Any]) -> str:
    event_type = event.get("event_type", "")
    spec = EVENT_SPEC_MAP.get(event_type)
    scope = spec.scope if spec else ""
    occurred = event.get("occurred_at") or event.get("created_at") or ""
    body = event.get("description") or json.dumps(event.get("fields", {}), ensure_ascii=False)
    return f"[Event][{event_type}][{scope}] @ {occurred} | {body}"
