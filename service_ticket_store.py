"""售后工单草稿存储（人在回路 / Human-in-the-loop 的权限边界演示）。

核心设计：
- Agent（LLM）只能调用 create_aftersales_ticket_draft 创建 **pending（待审核）** 草稿；
- approve / reject 两个方法 **不注册为 LLM 工具**，只能由人工客服通过代码/UI 调用；
- 已审核（approved/rejected）的工单不可再次审核，保证状态机不可逆跳转。

状态机：
    pending ──approve──▶ approved
       │
       └──reject───▶ rejected

面试考点：
- 为什么写操作要人工确认：LLM 可能误解用户意图或被注入诱导，直接写业务系统（退款/工单）
  会造成真实损失；草稿 + 人工批准把"不可逆动作"的决定权留给人。
- 权限分层如何在代码上保证：而不是只在 prompt 里要求"别批准"——LLM 根本拿不到批准接口。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 合法的工单类型与状态（应用层校验与数据库 CHECK 共用同一份常量，避免两处漂移）
TICKET_TYPES = {"return", "exchange", "refund", "allergy_case"}
TICKET_STATUSES = {"pending", "approved", "rejected"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_create_table_sql() -> str:
    """建表 DDL。枚举 CHECK 由 TICKET_TYPES/TICKET_STATUSES 常量生成。

    纵深防御：应用层校验给模型/调用方友好错误，数据库层 CHECK 兜底非法写入
    （绕过 store 直接写库、未来新增调用方漏校验等）。状态一致性也由库保证：
    - pending 工单不得带审核人/审核时间；
    - approved/rejected 必须有审核人和审核时间；
    - rejected 必须有驳回原因。
    """
    type_enum = ", ".join(f"'{name}'" for name in sorted(TICKET_TYPES))
    status_enum = ", ".join(f"'{name}'" for name in sorted(TICKET_STATUSES))
    return f"""
CREATE TABLE IF NOT EXISTS service_ticket (
    ticket_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    ticket_type      TEXT NOT NULL CHECK (ticket_type IN ({type_enum})),
    reason           TEXT NOT NULL,
    product_name     TEXT DEFAULT '',
    evidence_ids     TEXT DEFAULT '[]',
    status           TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ({status_enum})),
    created_by       TEXT NOT NULL DEFAULT 'agent',
    reviewed_by      TEXT DEFAULT '',
    review_note      TEXT DEFAULT '',
    created_at       TEXT NOT NULL,
    reviewed_at      TEXT DEFAULT '',
    CHECK (
        (status = 'pending' AND reviewed_by = '' AND reviewed_at = '')
        OR
        (status <> 'pending' AND reviewed_by <> '' AND reviewed_at <> '')
    ),
    CHECK (status <> 'rejected' OR review_note <> '')
)
"""


_CREATE_INDEX_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ticket_user ON service_ticket(user_id)"
)
_CREATE_INDEX_STATUS_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ticket_status ON service_ticket(status)"
)


class ServiceTicketStore:
    """SQLite 持久化的售后工单草稿存储。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        """建表；检测旧版无 CHECK 约束的表并自动迁移（重建表，SQLite 无法事后加 CHECK）。"""
        create_table_sql = _build_create_table_sql()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='service_ticket'"
            ).fetchone()
            if row is not None and "CHECK" not in (row[0] or "").upper():
                # 迁移：旧表改名 → 建新表（含 CHECK）→ 拷数据 → 删旧表（旧索引随之删除）
                connection.execute(
                    "ALTER TABLE service_ticket RENAME TO service_ticket_legacy"
                )
                connection.execute(create_table_sql)
                connection.execute(_CREATE_INDEX_USER_SQL)
                connection.execute(_CREATE_INDEX_STATUS_SQL)
                connection.execute(
                    "INSERT INTO service_ticket SELECT * FROM service_ticket_legacy"
                )
                connection.execute("DROP TABLE service_ticket_legacy")
            else:
                connection.execute(create_table_sql)
                connection.execute(_CREATE_INDEX_USER_SQL)
                connection.execute(_CREATE_INDEX_STATUS_SQL)

    # ------------------------------------------------------------------
    # Agent 可调用：创建草稿（唯一暴露给 LLM 的写操作，落库即 pending）
    # ------------------------------------------------------------------

    def create_draft(
        self,
        *,
        user_id: str,
        ticket_type: str,
        reason: str,
        product_name: str = "",
        evidence_ids: list[str] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        """创建售后工单草稿。状态恒为 pending，等待人工审核。"""
        if ticket_type not in TICKET_TYPES:
            raise ValueError(
                f"ticket_type 取值非法: {ticket_type}，合法值: {sorted(TICKET_TYPES)}"
            )
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("reason 不能为空")

        ticket_id = "TKT_" + uuid.uuid4().hex[:12]
        now = _utcnow_iso()
        evidence_json = json.dumps(evidence_ids or [], ensure_ascii=False)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO service_ticket
                    (ticket_id, user_id, ticket_type, reason, product_name,
                     evidence_ids, status, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    ticket_id, user_id, ticket_type, reason, product_name,
                    evidence_json, created_by, now,
                ),
            )

        return self.get_ticket(ticket_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 仅人工可调用：审核（不注册为 LLM 工具）
    # ------------------------------------------------------------------

    def approve(self, ticket_id: str, reviewer: str = "human_agent") -> dict[str, Any]:
        """人工批准工单草稿。pending → approved。"""
        return self._review(ticket_id, status="approved", reviewer=reviewer, note="")

    def reject(
        self,
        ticket_id: str,
        note: str = "",
        reviewer: str = "human_agent",
    ) -> dict[str, Any]:
        """人工驳回工单草稿。pending → rejected，需填写驳回原因。"""
        return self._review(
            ticket_id, status="rejected", reviewer=reviewer, note=note
        )

    def _review(
        self,
        ticket_id: str,
        *,
        status: str,
        reviewer: str,
        note: str,
    ) -> dict[str, Any]:
        ticket = self.get_ticket(ticket_id)
        if ticket is None:
            raise ValueError(f"工单不存在: {ticket_id}")
        if ticket["status"] != "pending":
            raise ValueError(
                f"工单 {ticket_id} 当前状态为 {ticket['status']}，已审核工单不可重复操作"
            )
        if status == "rejected" and not note.strip():
            raise ValueError("驳回工单必须填写原因（review note）")

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE service_ticket
                   SET status = ?, reviewed_by = ?, review_note = ?, reviewed_at = ?
                 WHERE ticket_id = ? AND status = 'pending'
                """,
                (status, reviewer, note, _utcnow_iso(), ticket_id),
            )
        updated = self.get_ticket(ticket_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM service_ticket WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_tickets(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM service_ticket WHERE 1=1"
        params: list[Any] = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if status:
            if status not in TICKET_STATUSES:
                raise ValueError(f"status 取值非法: {status}")
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 100)))

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]  # type: ignore[arg-type]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "ticket_id": row["ticket_id"],
            "user_id": row["user_id"],
            "ticket_type": row["ticket_type"],
            "reason": row["reason"],
            "product_name": row["product_name"],
            "evidence_ids": json.loads(row["evidence_ids"] or "[]"),
            "status": row["status"],
            "created_by": row["created_by"],
            "reviewed_by": row["reviewed_by"],
            "review_note": row["review_note"],
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
        }
