"""Independent idempotency ledger for processed conversation turns."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


class ProcessedTurnStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_turns (
                    user_id TEXT NOT NULL,
                    source_turn_hash TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, source_turn_hash)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def get(self, user_id: str, source_turn_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM processed_turns WHERE user_id = ? AND source_turn_hash = ?",
                (user_id, source_turn_hash),
            ).fetchone()
        if row is None:
            return None
        try:
            result = json.loads(row[0])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None

    def record(self, user_id: str, source_turn_hash: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO processed_turns
                    (user_id, source_turn_hash, processed_at, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    source_turn_hash,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
