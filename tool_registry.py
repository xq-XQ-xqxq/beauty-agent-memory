"""工具注册表与执行器（Function Calling）。

设计原则：
- 工具接口用 OpenAI Function Calling 标准的 JSON Schema 描述，便于 LLM 直接调用。
- 执行器只负责：参数校验 → 调用底层接口 → 结构化返回；不做任何 LLM 推理。
- 底层复用 KnowledgeRetriever / ProfileEventStore，零新增依赖。

5 个工具（4 只读 + 1 写草稿）：
1. search_knowledge_base(query, doc_type?, top_k?) — 主动检索 FAQ/SOP/产品/成分文档
2. get_user_profile(user_id, profile_key?) — 读取用户画像（不指定 key 时返回全部）
3. get_user_events(user_id, event_type?, limit?) — 读取用户事件流
4. check_ingredient_safety(ingredient_name, skin_type?) — 查成分安全（结合用户肤质）
5. create_aftersales_ticket_draft(...) — 【写】创建售后工单草稿（pending），必须人工审核

权限边界（人在回路）：
- 工具 1-4 为只读；工具 5 是唯一写操作，且只能落 pending 草稿。
- approve/reject 不注册为 LLM 工具，仅人工客服可调用（见 service_ticket_store.py）。
- 所有工具统一超时控制（默认 10s），超时/异常以结构化错误回填给模型。

LLM 与工具分工：
- LLM 负责：决定何时调用、调用哪个工具、用什么参数；解读工具返回结果。
- 工具负责：执行确定性查询，返回结构化数据；不做语义推断。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from hybrid_retriever import HybridRetriever
from knowledge_retriever import RetrievedDoc
from profile_event_schema import EVENT_KEYS
from profile_event_store import ProfileEventStore, render_event_text, render_profile_text
from service_ticket_store import ServiceTicketStore


# 单工具执行超时（秒）。超时后结构化错误回填给 LLM。
# 注意：Python 线程无法强杀，超时只保证"调用方不再等待并拿到错误"，
# 底层线程可能稍后自然结束；本项目工具均为本地毫秒级查询，风险可控。
TOOL_TIMEOUT_SEC = 10.0

# 共享线程池：所有工具执行共用，避免每次调用新建线程
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-exec")


# ---------------------------------------------------------------------------
# Tool call result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallResult:
    """单次工具调用的执行结果（含元信息，供 UI 展示轨迹）。"""

    name: str
    arguments: dict[str, Any]
    result: Any
    latency_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_display(self) -> dict[str, Any]:
        """转成可序列化的展示字典（截断大字段，避免 UI 爆炸）。"""
        result_str = _stringify(self.result)
        if len(result_str) > 600:
            result_str = result_str[:600] + "..."
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": result_str,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "ok": self.ok,
        }


def _stringify(value: Any) -> str:
    import json

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """单个工具的 schema 描述（OpenAI Function Calling 兼容格式）。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Any] = field(repr=False)

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 chat.completions.create(tools=...) 接收的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool handlers（底层执行函数）
# ---------------------------------------------------------------------------

def _handle_search_knowledge_base(
    retriever: HybridRetriever,
    *,
    query: str,
    doc_type: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """检索知识库。doc_type 可选 product/ingredient/faq/policy。"""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    top_k = max(1, min(int(top_k) if isinstance(top_k, (int, float)) else 3, 5))
    if doc_type and doc_type not in {"product", "ingredient", "faq", "policy"}:
        raise ValueError(f"doc_type 取值非法: {doc_type}")

    results: list[RetrievedDoc] = retriever.search(query, doc_type=doc_type, top_k=top_k)
    return [
        {
            "doc_id": d.doc_id,
            "doc_type": d.doc_type,
            "title": d.title,
            "content": d.content,
            "score": round(d.score, 4),
        }
        for d in results
    ]


def _handle_get_user_profile(
    store: ProfileEventStore,
    *,
    user_id: str,
    profile_key: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """读取用户画像。profile_key 为空时返回全部画像。"""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 不能为空")

    if profile_key:
        record = store.get_profile(user_id, profile_key)
        if record is None:
            return {"found": False, "profile_key": profile_key, "value": None}
        return {
            "found": True,
            "profile_key": profile_key,
            "value": record["value"],
            "evidence": record["evidence"],
            "updated_at": record["updated_at"],
            "render": render_profile_text(profile_key, record["value"]),
        }

    all_profile = store.get_all_profile(user_id)
    return [
        {
            "profile_key": key,
            "value": rec["value"],
            "updated_at": rec["updated_at"],
            "render": render_profile_text(key, rec["value"]),
        }
        for key, rec in all_profile.items()
    ]


def _handle_get_user_events(
    store: ProfileEventStore,
    *,
    user_id: str,
    event_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """读取用户事件流。event_type 为空时返回所有类型。"""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 不能为空")
    if event_type and event_type not in EVENT_KEYS:
        raise ValueError(
            f"event_type 取值非法: {event_type}；合法取值: {', '.join(EVENT_KEYS)}"
        )
    limit = max(1, min(int(limit) if isinstance(limit, (int, float)) else 10, 30))

    events = store.get_events(user_id, event_type=event_type, limit=limit)
    return [
        {
            "event_id": e["event_id"],
            "event_type": e["event_type"],
            "occurred_at": e.get("occurred_at") or e.get("created_at"),
            "description": e.get("description", ""),
            "fields": e["fields"],
            "render": render_event_text(e),
        }
        for e in events
    ]


# 肤质 → 建议禁用/慎用成分（用于成分安全检查的快速判定）
SKIN_TYPE_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    "sensitive": {
        "avoid": ["水杨酸", "果酸", "A醇", "视黄醇", "高浓度烟酰胺", "酒精"],
        "caution": ["烟酰胺", "维C"],
    },
    "dry": {
        "avoid": ["水杨酸", "高岭土", "酒精"],
        "caution": ["果酸"],
    },
    "oily": {
        "avoid": ["矿物油", "凡士林"],
        "caution": [],
    },
    "combination": {
        "avoid": [],
        "caution": ["高浓度水杨酸", "高浓度果酸"],
    },
    "normal": {
        "avoid": [],
        "caution": [],
    },
}


def _handle_check_ingredient_safety(
    retriever: HybridRetriever,
    *,
    ingredient_name: str,
    skin_type: str | None = None,
) -> dict[str, Any]:
    """查成分安全性。先查知识库 ingredient 类型，再叠加肤质约束。"""

    if not isinstance(ingredient_name, str) or not ingredient_name.strip():
        raise ValueError("ingredient_name 不能为空")

    # 1. 知识库检索成分详情
    docs = retriever.search(ingredient_name, doc_type="ingredient", top_k=2)
    kb_info = [
        {"title": d.title, "content": d.content, "doc_id": d.doc_id}
        for d in docs
    ]

    # 2. 肤质约束叠加（本地规则，避免 LLM 推断）
    constraints: dict[str, Any] = {}
    if skin_type and skin_type in SKIN_TYPE_CONSTRAINTS:
        rules = SKIN_TYPE_CONSTRAINTS[skin_type]
        lowered = ingredient_name.lower()
        # 匹配：成分名包含禁忌/慎用关键词
        matched_avoid = [kw for kw in rules["avoid"] if kw in ingredient_name or kw in lowered]
        matched_caution = [kw for kw in rules["caution"] if kw in ingredient_name or kw in lowered]
        constraints = {
            "skin_type": skin_type,
            "should_avoid": len(matched_avoid) > 0,
            "should_caution": len(matched_caution) > 0,
            "matched_avoid_keywords": matched_avoid,
            "matched_caution_keywords": matched_caution,
        }
    else:
        constraints = {"skin_type": skin_type or "unknown", "should_avoid": False, "should_caution": False}

    return {
        "ingredient": ingredient_name,
        "kb_docs": kb_info,
        "constraints": constraints,
    }


def _handle_create_aftersales_ticket_draft(
    ticket_store: ServiceTicketStore,
    *,
    user_id: str,
    ticket_type: str,
    reason: str,
    product_name: str = "",
    evidence_message_ids: list[int] | None = None,
) -> dict[str, Any]:
    """创建售后工单草稿（写操作）。

    权限边界：此工具只能创建 pending 草稿，批准/驳回只能由人工客服完成，
    Agent 没有批准接口。客服回复中不得承诺工单一定通过或承诺退款金额。
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 不能为空")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 不能为空（需说明用户诉求与事实依据）")

    evidence = [f"msg_{i}" for i in (evidence_message_ids or []) if isinstance(i, int)]

    ticket = ticket_store.create_draft(
        user_id=user_id,
        ticket_type=ticket_type,
        reason=reason.strip(),
        product_name=(product_name or "").strip(),
        evidence_ids=evidence,
        created_by="agent",
    )
    return {
        "ticket_id": ticket["ticket_id"],
        "status": "pending",
        "ticket_type": ticket["ticket_type"],
        "notice": (
            "工单草稿已创建，状态为待人工审核（pending）。"
            "你无权批准或驳回；请告知客服已生成草稿、需人工确认，"
            "且不要向用户承诺退款金额或处理结果。"
        ),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """工具注册表 + 执行器。

    用法：
        registry = ToolRegistry(store=store, retriever=retriever)
        schemas = registry.openai_schemas()         # 给 LLM 的工具描述
        result = registry.execute("search_knowledge_base", {"query": "长痘"})
    """

    def __init__(
        self,
        store: ProfileEventStore,
        retriever: HybridRetriever | None = None,
        ticket_store: ServiceTicketStore | None = None,
        tool_timeout: float = TOOL_TIMEOUT_SEC,
    ) -> None:
        if retriever is None:
            retriever = HybridRetriever()
        self._store = store
        self._retriever = retriever
        self._ticket_store = ticket_store
        self._tool_timeout = tool_timeout
        self._tools: dict[str, ToolSpec] = {}
        self._register_all()

    def _register_all(self) -> None:
        self._register(ToolSpec(
            name="search_knowledge_base",
            description=(
                "检索美妆知识库，获取产品详情、成分说明、FAQ 或售后政策。"
                "当用户提到具体产品名、成分名、症状或售后问题时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本，如\"长痘用什么\"、\"烟酰胺成分\"、\"退货政策\"",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["product", "ingredient", "faq", "policy"],
                        "description": "限定文档类型；不指定则全类型检索",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回前 K 条，默认 3，最大 5",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
            handler=lambda **kwargs: _handle_search_knowledge_base(self._retriever, **kwargs),
        ))

        self._register(ToolSpec(
            name="get_user_profile",
            description=(
                "读取用户画像（Profile 通道：肤质、敏感度、偏好、目标等长期属性）。"
                "需要确认用户历史肤质、过敏、目标时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户 ID"},
                    "profile_key": {
                        "type": "string",
                        "description": "画像 key，如 body.skin_type、preference.product_attribute；不指定则返回全部",
                    },
                },
                "required": ["user_id"],
            },
            handler=lambda **kwargs: _handle_get_user_profile(self._store, **kwargs),
        ))

        self._register(ToolSpec(
            name="get_user_events",
            description=(
                "读取用户事件流（Event 通道：购买经历、过敏史、投诉、情绪等）。"
                "需要避免重复询问已确认事项或检索历史投诉时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户 ID"},
                    "event_type": {
                        "type": "string",
                        "enum": list(EVENT_KEYS),
                        "description": (
                            "事件类型过滤。查投诉/升级诉求用 dissatisfaction.escalation_intent；"
                            "查不满原因用 dissatisfaction.reason；查产品使用反应用 "
                            "experience.product_reaction；不指定则返回全部类型"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回条数上限，默认 10，最大 30",
                        "default": 10,
                    },
                },
                "required": ["user_id"],
            },
            handler=lambda **kwargs: _handle_get_user_events(self._store, **kwargs),
        ))

        self._register(ToolSpec(
            name="check_ingredient_safety",
            description=(
                "查询成分安全性。结合知识库成分文档与用户肤质约束，"
                "返回是否禁忌/慎用以及知识库详情。用户询问成分安全性时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ingredient_name": {
                        "type": "string",
                        "description": "成分名称，如\"烟酰胺\"、\"水杨酸\"、\"A醇\"",
                    },
                    "skin_type": {
                        "type": "string",
                        "enum": ["dry", "oily", "combination", "sensitive", "normal"],
                        "description": "肤质；不指定则只返回知识库信息，不判定禁忌",
                    },
                },
                "required": ["ingredient_name"],
            },
            handler=lambda **kwargs: _handle_check_ingredient_safety(self._retriever, **kwargs),
        ))

        # 写工具（仅当工单存储可用时注册）：创建售后工单草稿，需人工审核
        if self._ticket_store is not None:
            self._register(ToolSpec(
                name="create_aftersales_ticket_draft",
                description=(
                    "创建售后工单【草稿】（退货/换货/退款/过敏案例），用于明确表达售后诉求的用户。"
                    "草稿创建后状态为 pending，必须由人工客服审核后才生效。"
                    "仅在用户明确提出退货/换货/退款/过敏投诉时调用；闲聊和普通咨询不要调用。"
                    "调用后不得向用户承诺退款金额或处理结果。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "用户 ID"},
                        "ticket_type": {
                            "type": "string",
                            "enum": ["return", "exchange", "refund", "allergy_case"],
                            "description": "工单类型：退货/换货/退款/过敏案例",
                        },
                        "reason": {
                            "type": "string",
                            "description": "工单原因，客观描述用户诉求与事实（引用对话中的具体信息），不写推断",
                        },
                        "product_name": {
                            "type": "string",
                            "description": "涉及产品名（如对话中提及）",
                        },
                        "evidence_message_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "支撑该工单的对话消息编号（从1开始），1-3 条",
                        },
                    },
                    "required": ["user_id", "ticket_type", "reason"],
                },
                handler=lambda **kwargs: _handle_create_aftersales_ticket_draft(
                    self._ticket_store, **kwargs
                ),
            ))

    @property
    def ticket_store(self) -> ServiceTicketStore | None:
        return self._ticket_store

    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """返回 tools=[...] 格式的工具 schema 列表，供 OpenAI 调用使用。"""
        return [self._tools[name].to_openai_schema() for name in self._tools]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """执行一次工具调用，返回结构化结果（含错误信息与耗时）。"""
        spec = self._tools.get(name)
        if spec is None:
            return ToolCallResult(
                name=name,
                arguments=arguments,
                result=None,
                latency_ms=0,
                error=f"未注册的工具: {name}",
            )

        # 参数校验：required 字段必须存在
        missing = [
            field for field in spec.parameters.get("required", [])
            if not arguments.get(field)
        ]
        if missing:
            return ToolCallResult(
                name=name,
                arguments=arguments,
                result=None,
                latency_ms=0,
                error=f"缺少必填参数: {missing}",
            )

        started = perf_counter()
        try:
            # 在线程池中执行，施加超时上限（防止慢工具卡死整个 agent loop）
            future = _TOOL_EXECUTOR.submit(spec.handler, **arguments)
            result = future.result(timeout=self._tool_timeout)
            latency_ms = round((perf_counter() - started) * 1000)
            return ToolCallResult(
                name=name,
                arguments=arguments,
                result=result,
                latency_ms=latency_ms,
            )
        except FuturesTimeout:
            latency_ms = round((perf_counter() - started) * 1000)
            return ToolCallResult(
                name=name,
                arguments=arguments,
                result=None,
                latency_ms=latency_ms,
                error=(
                    f"工具执行超时（>{self._tool_timeout}s）。"
                    "请缩小查询范围或改用其他工具，不要重复同样的调用。"
                ),
            )
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            return ToolCallResult(
                name=name,
                arguments=arguments,
                result=None,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    def execute_batch(self, calls: list[dict[str, Any]]) -> list[ToolCallResult]:
        """批量执行工具调用。calls 格式: [{"name": "...", "arguments": {...}}, ...]"""
        return [
            self.execute(call.get("name", ""), call.get("arguments", {}) or {})
            for call in calls
        ]


# ---------------------------------------------------------------------------
# Default factory（便于在 label_engine / streamlit_app 中复用）
# ---------------------------------------------------------------------------

def build_default_registry(
    db_path: Path | str | None = None,
) -> ToolRegistry:
    """构建默认工具注册表（连接默认 SQLite + 知识库）。

    注意：默认 DB 路径与 streamlit_app.pe_store() 保持一致（data/profile_event.db），
    否则 Agent 模式会读到空数据库。
    """
    if db_path is None:
        db_path = Path(__file__).parent / "data" / "profile_event.db"
    store = ProfileEventStore(db_path)
    ticket_store = ServiceTicketStore(
        Path(db_path).parent / "service_tickets.sqlite3"
    )
    return ToolRegistry(
        store=store,
        retriever=HybridRetriever(),
        ticket_store=ticket_store,
    )
