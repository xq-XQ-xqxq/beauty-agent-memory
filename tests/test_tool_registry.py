"""工具注册表测试：参数校验、超时、失败回填、写工具权限边界。

用 FakeRetriever 避免构造 HybridRetriever（会触发真实 embedding）。
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_retriever import RetrievedDoc
from profile_event_store import ProfileEventStore
from service_ticket_store import ServiceTicketStore
from tool_registry import TOOL_TIMEOUT_SEC, ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# Fake 组件
# ---------------------------------------------------------------------------

class FakeRetriever:
    """鸭子类型：只需实现 search。"""

    def search(self, query, doc_type=None, top_k=5):
        if not query or not str(query).strip():
            return []
        return [
            RetrievedDoc(
                doc_id="prod_003",
                doc_type=doc_type or "product",
                title="水杨酸清洁面膜",
                content="测试内容",
                metadata={},
                score=1.0,
            )
        ]


class SlowRetriever:
    """模拟慢工具（超过超时阈值）。"""

    def search(self, query, doc_type=None, top_k=5):
        time.sleep(1.0)
        return []


@pytest.fixture()
def registry(tmp_path):
    store = ProfileEventStore(tmp_path / "profile.db")
    tickets = ServiceTicketStore(tmp_path / "tickets.db")
    return ToolRegistry(
        store=store,
        retriever=FakeRetriever(),
        ticket_store=tickets,
    )


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def test_unknown_tool_returns_error(registry):
    result = registry.execute("not_exist_tool", {})
    assert not result.ok
    assert "未注册的工具" in result.error


def test_missing_required_param_returns_error(registry):
    result = registry.execute("search_knowledge_base", {})
    assert not result.ok
    assert "缺少必填参数" in result.error
    assert "query" in result.error


def test_invalid_doc_type_raises_structured_error(registry):
    result = registry.execute(
        "search_knowledge_base",
        {"query": "测试", "doc_type": "not_a_type"},
    )
    assert not result.ok
    assert "doc_type 取值非法" in result.error


def test_top_k_is_clamped(registry):
    # top_k=999 应被夹紧到 5，不报错
    result = registry.execute(
        "search_knowledge_base",
        {"query": "痘痘", "top_k": 999},
    )
    assert result.ok
    assert isinstance(result.result, list)
    assert len(result.result) <= 5


def test_search_returns_structured_docs(registry):
    result = registry.execute(
        "search_knowledge_base",
        {"query": "水杨酸", "doc_type": "product"},
    )
    assert result.ok
    first = result.result[0]
    assert first["doc_id"] == "prod_003"
    assert "score" in first


def test_get_user_profile_empty_user_id_rejected(registry):
    result = registry.execute("get_user_profile", {"user_id": "   "})
    assert not result.ok
    assert "user_id" in result.error


def test_get_user_events_invalid_event_type_rejected(registry):
    """event_type 必须是 schema 枚举内的合法记忆键（防 LLM 自造类型）。"""
    result = registry.execute(
        "get_user_events",
        {"user_id": "u1", "event_type": "complaint"},
    )
    assert not result.ok
    assert "event_type 取值非法" in result.error


def test_get_user_events_valid_event_type_accepted(registry):
    result = registry.execute(
        "get_user_events",
        {"user_id": "u1", "event_type": "dissatisfaction.escalation_intent"},
    )
    assert result.ok
    assert result.result == []


# ---------------------------------------------------------------------------
# 超时控制
# ---------------------------------------------------------------------------

def test_tool_timeout_returns_structured_error(tmp_path):
    """工具执行超过超时阈值时，返回结构化超时错误（不卡死 loop）。"""
    store = ProfileEventStore(tmp_path / "profile.db")
    reg = ToolRegistry(
        store=store,
        retriever=SlowRetriever(),
        ticket_store=None,
        tool_timeout=0.3,
    )
    result = reg.execute("search_knowledge_base", {"query": "慢查询"})
    assert not result.ok
    assert "超时" in result.error
    assert result.latency_ms >= 200


def test_default_timeout_is_10_seconds():
    assert TOOL_TIMEOUT_SEC == 10.0


# ---------------------------------------------------------------------------
# 写工具：售后工单草稿 + 人工审核权限边界
# ---------------------------------------------------------------------------

def test_ticket_tool_not_registered_without_store(tmp_path):
    """未提供 ticket_store 时，写工具不注册（工具白名单即权限边界）。"""
    store = ProfileEventStore(tmp_path / "profile.db")
    reg = ToolRegistry(store=store, retriever=FakeRetriever(), ticket_store=None)
    assert "create_aftersales_ticket_draft" not in reg.tool_names
    # 4 个只读工具仍在
    assert len(reg.tool_names) == 4


def test_create_ticket_draft_is_pending(registry):
    result = registry.execute(
        "create_aftersales_ticket_draft",
        {
            "user_id": "u100",
            "ticket_type": "return",
            "reason": "用户使用水杨酸面膜后泛红，明确要求退货",
            "product_name": "水杨酸清洁面膜",
            "evidence_message_ids": [1, 2],
        },
    )
    assert result.ok
    assert result.result["status"] == "pending"
    ticket_id = result.result["ticket_id"]
    assert ticket_id.startswith("TKT_")

    # 落库状态确实是 pending
    ticket = registry.ticket_store.get_ticket(ticket_id)
    assert ticket["status"] == "pending"
    assert ticket["evidence_ids"] == ["msg_1", "msg_2"]
    assert ticket["created_by"] == "agent"


def test_ticket_invalid_type_rejected(registry):
    result = registry.execute(
        "create_aftersales_ticket_draft",
        {"user_id": "u100", "ticket_type": "delete_account", "reason": "x"},
    )
    assert not result.ok
    assert "ticket_type 取值非法" in result.error


def test_ticket_missing_reason_rejected(registry):
    result = registry.execute(
        "create_aftersales_ticket_draft",
        {"user_id": "u100", "ticket_type": "refund", "reason": "   "},
    )
    assert not result.ok
    assert "reason" in result.error


def test_ticket_human_approval_flow(registry):
    """草稿只能由人工 approve；批准后状态不可逆，重复审核报错。"""
    created = registry.execute(
        "create_aftersales_ticket_draft",
        {
            "user_id": "u101",
            "ticket_type": "allergy_case",
            "reason": "用户过敏要求登记",
        },
    )
    ticket_id = created.result["ticket_id"]

    # 人工批准（走 store API，LLM 拿不到这个接口）
    approved = registry.ticket_store.approve(ticket_id, reviewer="human_客服01")
    assert approved["status"] == "approved"
    assert approved["reviewed_by"] == "human_客服01"

    # 已批准工单不可再次审核
    with pytest.raises(ValueError, match="不可重复操作"):
        registry.ticket_store.reject(ticket_id, note="晚了")


def test_ticket_reject_requires_note(registry):
    created = registry.execute(
        "create_aftersales_ticket_draft",
        {"user_id": "u102", "ticket_type": "exchange", "reason": "想换货"},
    )
    ticket_id = created.result["ticket_id"]
    with pytest.raises(ValueError, match="驳回"):
        registry.ticket_store.reject(ticket_id, note="  ")


def test_list_tickets_filter_by_status(registry):
    for i in range(3):
        registry.execute(
            "create_aftersales_ticket_draft",
            {"user_id": f"u{i}", "ticket_type": "refund", "reason": f"原因{i}"},
        )
    pending = registry.ticket_store.list_tickets(status="pending")
    assert len(pending) == 3
    approved = registry.ticket_store.list_tickets(status="approved")
    assert approved == []
