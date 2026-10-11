"""售后工单存储测试：应用层状态机 + 数据库层 CHECK 纵深防御 + 旧表迁移。"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from service_ticket_store import ServiceTicketStore


# ---------------------------------------------------------------------------
# 应用层状态机（HITL 边界）
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return ServiceTicketStore(tmp_path / "tickets.db")


def test_draft_is_always_pending(store):
    draft = store.create_draft(
        user_id="u1", ticket_type="return", reason="不适合肤质"
    )
    assert draft["status"] == "pending"
    assert draft["ticket_id"].startswith("TKT_")


def test_invalid_ticket_type_rejected_at_app_layer(store):
    with pytest.raises(ValueError, match="ticket_type 取值非法"):
        store.create_draft(user_id="u1", ticket_type="chargeback", reason="x")


def test_approve_flow(store):
    draft = store.create_draft(user_id="u1", ticket_type="refund", reason="重复扣款")
    approved = store.approve(draft["ticket_id"], reviewer="cs_001")
    assert approved["status"] == "approved"
    assert approved["reviewed_by"] == "cs_001"
    assert approved["reviewed_at"]


def test_reject_requires_note(store):
    draft = store.create_draft(user_id="u1", ticket_type="return", reason="x")
    with pytest.raises(ValueError, match="必须填写原因"):
        store.reject(draft["ticket_id"], note="   ")


def test_reviewed_ticket_cannot_be_reviewed_again(store):
    draft = store.create_draft(user_id="u1", ticket_type="return", reason="x")
    store.approve(draft["ticket_id"])
    with pytest.raises(ValueError, match="已审核工单不可重复操作"):
        store.reject(draft["ticket_id"], note="改判")


# ---------------------------------------------------------------------------
# 数据库层 CHECK：绕过应用层直接写库也必须被拦（纵深防御）
# ---------------------------------------------------------------------------

_RAW_COLUMNS = (
    "ticket_id, user_id, ticket_type, reason, product_name, evidence_ids, "
    "status, created_by, reviewed_by, review_note, created_at, reviewed_at"
)


def _raw_insert(store, **overrides):
    values = {
        "ticket_id": "TKT_raw001",
        "user_id": "u1",
        "ticket_type": "return",
        "reason": "测试",
        "product_name": "",
        "evidence_ids": "[]",
        "status": "pending",
        "created_by": "agent",
        "reviewed_by": "",
        "review_note": "",
        "created_at": "2026-10-11T00:00:00+00:00",
        "reviewed_at": "",
    }
    values.update(overrides)
    placeholders = ", ".join("?" for _ in values)
    with store._connect() as connection:
        connection.execute(
            f"INSERT INTO service_ticket ({_RAW_COLUMNS}) VALUES ({placeholders})",
            tuple(values.values()),
        )


def test_db_rejects_unknown_status(store):
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(store, status="cancelled")


def test_db_rejects_unknown_ticket_type(store):
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(store, ticket_type="chargeback")


def test_db_rejects_rejected_without_note(store):
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(
            store,
            status="rejected",
            reviewed_by="cs_001",
            reviewed_at="2026-10-11T00:01:00+00:00",
        )


def test_db_rejects_approved_without_reviewer(store):
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(
            store,
            status="approved",
            reviewed_at="2026-10-11T00:01:00+00:00",
        )


def test_db_rejects_pending_with_review_info(store):
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(
            store,
            reviewed_by="cs_001",
            reviewed_at="2026-10-11T00:01:00+00:00",
        )


def test_db_accepts_well_formed_rejected_row(store):
    _raw_insert(
        store,
        status="rejected",
        reviewed_by="cs_001",
        review_note="超出 7 天无理由期限",
        reviewed_at="2026-10-11T00:01:00+00:00",
    )
    ticket = store.get_ticket("TKT_raw001")
    assert ticket is not None
    assert ticket["status"] == "rejected"


# ---------------------------------------------------------------------------
# 旧表迁移：无 CHECK 约束的旧版表 → 自动重建为新表，数据保留
# ---------------------------------------------------------------------------

def test_legacy_table_without_check_is_migrated(tmp_path):
    db_path = tmp_path / "tickets.db"

    # 手工构造旧版表（无任何 CHECK），并写入一条历史工单
    legacy_connection = sqlite3.connect(db_path)
    legacy_connection.executescript(
        """
        CREATE TABLE service_ticket (
            ticket_id        TEXT PRIMARY KEY,
            user_id          TEXT NOT NULL,
            ticket_type      TEXT NOT NULL,
            reason           TEXT NOT NULL,
            product_name     TEXT DEFAULT '',
            evidence_ids     TEXT DEFAULT '[]',
            status           TEXT NOT NULL DEFAULT 'pending',
            created_by       TEXT NOT NULL DEFAULT 'agent',
            reviewed_by      TEXT DEFAULT '',
            review_note      TEXT DEFAULT '',
            created_at       TEXT NOT NULL,
            reviewed_at      TEXT DEFAULT ''
        );
        INSERT INTO service_ticket
            (ticket_id, user_id, ticket_type, reason, created_at)
        VALUES ('TKT_legacy01', 'u9', 'return', '历史工单', '2026-09-01T00:00:00+00:00');
        """
    )
    legacy_connection.commit()
    legacy_connection.close()

    # 初始化即触发迁移
    migrated = ServiceTicketStore(db_path)

    # 历史数据仍可读
    ticket = migrated.get_ticket("TKT_legacy01")
    assert ticket is not None
    assert ticket["ticket_type"] == "return"
    assert ticket["status"] == "pending"

    # 迁移后新约束生效：非法写入被库层拦截
    with pytest.raises(sqlite3.IntegrityError):
        with migrated._connect() as connection:
            connection.execute(
                "INSERT INTO service_ticket "
                "(ticket_id, user_id, ticket_type, reason, status, created_at) "
                "VALUES ('TKT_bad', 'u9', 'return', 'x', 'cancelled', '2026-10-11T00:00:00+00:00')"
            )

    # 应用层正常功能不受迁移影响
    draft = migrated.create_draft(
        user_id="u9", ticket_type="exchange", reason="颜色不符"
    )
    assert draft["status"] == "pending"
