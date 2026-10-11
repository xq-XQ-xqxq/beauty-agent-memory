"""Second mapping: full customer memory to dynamic agent-facing insights."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).parent / ".env")


ALLOWED_PRIORITIES = {"urgent", "high", "normal"}
PRIORITY_RANK = {"urgent": 0, "high": 1, "normal": 2}
ALLOWED_ACTIONS = {
    "empathize",
    "clarify",
    "confirm_need",
    "avoid_repeat",
    "safety_attention",
    "follow_up",
}
MAX_LABELS = 5
MAX_SUGGESTIONS = 2
MAX_RECENT_MESSAGES = 10
FORBIDDEN_LABEL_TERMS = {
    "可能",
    "推测",
    "耐受",
    "致敏",
    "风险等级",
    "潜在",
    "意味着",
}
FORBIDDEN_SUGGESTION_TERMS = {
    "停用",
    "清水",
    "清洁",
    "就医",
    "用药",
    "涂抹",
    "退款",
    "赔偿",
    "批准",
}


@dataclass(frozen=True)
class ServiceLabel:
    title: str
    content: str
    priority: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ServiceSuggestion:
    action: str
    content: str
    basis_ids: tuple[str, ...]


@dataclass(frozen=True)
class ServiceInsight:
    service_focus: str
    labels: tuple[ServiceLabel, ...]
    service_suggestions: tuple[ServiceSuggestion, ...]
    latency_ms: int
    knowledge_sources: list[dict[str, Any]] = None  # type: ignore[assignment]
    # Agent 模式工具调用指标（被动模式下为空）
    tool_calls: int = 0
    tool_trace: list[dict[str, Any]] | None = None  # type: ignore[assignment]
    # Agent 模式标识（便于 UI 区分两种洞察来源）
    agent_mode: bool = False
    # Agent 模式 token 用量 / 运行 ID / 是否触发强制收尾（成本与可观测性）
    agent_token_usage: dict[str, int] | None = None  # type: ignore[assignment]
    agent_run_id: str = ""
    forced_finalization: bool = False

    def __post_init__(self) -> None:
        if self.knowledge_sources is None:
            object.__setattr__(self, "knowledge_sources", [])
        if self.tool_trace is None:
            object.__setattr__(self, "tool_trace", [])
        if self.agent_token_usage is None:
            object.__setattr__(self, "agent_token_usage", {})


class ServiceInsightGenerator:
    """Generate a compact, evidence-bound service view in one LLM call.

    上下文工程集成：BM25 检索相关记忆 + 长对话摘要 + token 软预算。

    支持两种工作模式：
    - 被动模式（默认）：BM25 检索记忆 + 知识库注入，LLM 一次调用产出洞察。
    - Agent 模式（use_agent_mode=True）：LLM 决策调用工具（search_knowledge_base、
      get_user_profile 等），按需主动检索，工具调用循环上限 3 轮。
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        use_agent_mode: bool = False,
        user_id: str | None = None,
    ) -> None:
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get(
                "DASHSCOPE_API_KEY", ""
            )
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
        self.use_agent_mode = use_agent_mode
        self.user_id = user_id
        self._summarizer = None  # lazy init to avoid LLM client at construction
        self._knowledge_retriever = None  # lazy init
        self._agent_registry = None  # Agent 模式下的工具注册表
        self._trace_store = None  # Agent 轨迹落盘存储（lazy init）

    def generate(
        self,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> ServiceInsight:
        # Agent 模式分流
        if self.use_agent_mode:
            return self._generate_with_agent(messages, memories)

        from context_engine import BM25Retriever, ConversationSummarizer, TokenCounter
        from hybrid_retriever import HybridRetriever

        token_counter = TokenCounter()

        # 1. 长对话摘要
        if self._summarizer is None:
            self._summarizer = ConversationSummarizer()
        summary, recent_messages, was_summarized = self._summarizer.summarize_if_needed(messages)

        numbered_messages = _number_recent_messages(recent_messages)

        # 2. BM25 检索相关记忆（而非全量注入）
        query = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        bm25 = BM25Retriever(memories)
        retrieved = bm25.retrieve(query, top_k=15)
        retrieved_ids = {id(r.event) for r in retrieved}
        # 补充未命中的最近记忆（最多 5 条）
        recent_memory_ids = {id(m) for m in memories[:5]}
        extra = [m for m in memories[:5] if id(m) not in retrieved_ids]
        selected_memories = [r.event for r in retrieved] + extra
        memory_sources = _prepare_memories(selected_memories)

        # 3. 知识库检索 FAQ/SOP（混合检索：BM25 + 向量 RRF 融合）
        if self._knowledge_retriever is None:
            self._knowledge_retriever = HybridRetriever()
        kb_results = self._knowledge_retriever.search(query, doc_type="faq", top_k=3)
        kb_results += self._knowledge_retriever.search(query, doc_type="policy", top_k=2)
        knowledge_sources = [
            {"doc_id": d.doc_id, "doc_type": d.doc_type, "title": d.title, "content": d.content}
            for d in kb_results
        ]

        valid_ids = {
            item["id"] for item in numbered_messages
        } | {item["id"] for item in memory_sources}

        prompt = {
            "recent_conversation": numbered_messages,
            "complete_user_memory": memory_sources,
        }
        if summary:
            prompt["conversation_summary"] = summary
        if knowledge_sources:
            prompt["knowledge_base_sop"] = knowledge_sources

        # 4. 软 token 预算检查
        prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        prompt_tokens = token_counter.count(prompt_json)
        if prompt_tokens > 4000:
            # 超预算：先裁 memory_sources，再裁 knowledge_sources
            while memory_sources and prompt_tokens > 4000:
                memory_sources.pop()
                prompt["complete_user_memory"] = memory_sources
                prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
                prompt_tokens = token_counter.count(prompt_json)
            while knowledge_sources and prompt_tokens > 4000:
                knowledge_sources.pop()
                prompt["knowledge_base_sop"] = knowledge_sources
                prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
                prompt_tokens = token_counter.count(prompt_json)

        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt_json,
                },
            ],
        )
        latency_ms = round((perf_counter() - started) * 1000)
        payload = json.loads(response.choices[0].message.content or "{}")
        insight = _validate_insight(payload, valid_ids, numbered_messages, latency_ms)
        # 存储检索来源供 UI 展示（frozen dataclass 用 object.__setattr__）
        object.__setattr__(insight, "knowledge_sources", knowledge_sources)
        return insight

    def _generate_with_agent(
        self,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> ServiceInsight:
        """Agent 模式：LLM 主动调用工具，循环生成洞察。

        与被动模式相比：
        - 不主动注入知识库结果，让 LLM 通过 search_knowledge_base 按需检索
        - 不暴露完整记忆内容，只给 memory_index（key 列表），强制 LLM 调
          get_user_profile / get_user_events 读取实际值
        - 工具调用循环上限 3 轮，工具失败时降级到现有上下文给保守洞察
        """
        from agent_trace_store import AgentTraceStore
        from context_engine import ConversationSummarizer
        from tool_calling_agent import ToolCallingAgent
        from tool_registry import build_default_registry

        # 1. 长对话摘要（与被动模式一致，避免超长对话影响）
        if self._summarizer is None:
            self._summarizer = ConversationSummarizer()
        summary, recent_messages, was_summarized = self._summarizer.summarize_if_needed(messages)

        # 2. 懒初始化工具注册表（共享 SQLite + 知识库）
        if self._agent_registry is None:
            self._agent_registry = build_default_registry()

        # 3. 懒初始化轨迹存储（每次 run 落 JSONL，供失败回放）
        if self._trace_store is None:
            self._trace_store = AgentTraceStore()

        # 4. 运行 Agent 工具调用循环
        user_id = self.user_id or "default_user"
        agent = ToolCallingAgent(
            registry=self._agent_registry,
            user_id=user_id,
            client=self.client,
            model=self.model,
            trace_store=self._trace_store,
        )
        agent_result = agent.run(recent_messages, memories, conversation_summary=summary)

        # 5. 把 Agent 输出转成 ServiceInsight（复用 _validate_insight 校验）
        # Agent 模式下 valid_ids 扩展：除 msg_* 和 memory_*，还包括工具返回的 doc_id
        valid_ids = _collect_valid_ids(recent_messages, memories, agent_result)
        insight = _validate_insight(
            agent_result.raw_insight,
            valid_ids,
            _number_recent_messages(recent_messages),
            agent_result.total_latency_ms,
        )
        # 6. 把工具调用指标塞进 insight（frozen dataclass 用 object.__setattr__）
        object.__setattr__(insight, "tool_calls", agent_result.tool_calls)
        object.__setattr__(
            insight,
            "tool_trace",
            _serialize_tool_trace(agent_result.tool_trace),
        )
        object.__setattr__(insight, "agent_mode", True)
        object.__setattr__(insight, "agent_token_usage", agent_result.token_usage)
        object.__setattr__(insight, "agent_run_id", agent_result.run_id)
        object.__setattr__(insight, "forced_finalization", agent_result.forced_finalization)
        # 7. 从工具结果中提取知识库命中，复用 knowledge_sources 字段供 UI 展示
        kb_sources = _extract_kb_sources_from_trace(agent_result)
        object.__setattr__(insight, "knowledge_sources", kb_sources)
        return insight


# Compatibility alias for callers that still import the old class name.
LabelGenerator = ServiceInsightGenerator


def _collect_valid_ids(
    messages: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    agent_result: Any,
) -> set[str]:
    """Agent 模式下收集所有合法 source_ids（含工具返回的 doc_id）。

    LLM 在 Agent 模式下引用 source_ids 时可能用到：
    - msg_* : 来自对话消息编号
    - mem_* / memory_id : 来自 memory_index
    - doc_id / prod_id / ing_id : 来自工具调用返回的知识库文档
    - event_id : 来自工具调用返回的事件流（保留，便于扩展）
    """
    valid: set[str] = set()
    # 1. 消息编号
    for index, item in enumerate(messages, start=1):
        if str(item.get("content", "")).strip():
            valid.add(f"msg_{index}")
    # 2. memory_id
    for m in memories:
        mid = str(m.get("id", "")).strip()
        if mid:
            valid.add(mid)
    # 3. 工具返回结果中的 doc_id / event_id / profile_key
    # 同时收集 "doc_<id>" 前缀版本，防止 LLM 在 source_ids 中误加前缀
    for trace in agent_result.tool_trace:
        for execution in trace.executions:
            if not execution.ok or not isinstance(execution.result, (list, dict)):
                continue
            items = execution.result if isinstance(execution.result, list) else [execution.result]
            for item in items:
                if not isinstance(item, dict):
                    continue
                for key in ("doc_id", "event_id", "profile_key"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        valid.add(value)
                        # 兼容 LLM 加 doc_ 前缀的情况（如 doc_prod_001）
                        if key == "doc_id":
                            valid.add(f"doc_{value}")
    return valid


def _extract_kb_sources_from_trace(agent_result: Any) -> list[dict[str, Any]]:
    """从 Agent 工具调用轨迹中提取知识库命中（用于 UI 展示）。"""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in agent_result.tool_trace:
        for execution in trace.executions:
            if not execution.ok or execution.name != "search_knowledge_base":
                continue
            if not isinstance(execution.result, list):
                continue
            for item in execution.result:
                if not isinstance(item, dict):
                    continue
                doc_id = item.get("doc_id", "")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                sources.append({
                    "doc_id": doc_id,
                    "doc_type": item.get("doc_type", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                })
    return sources


def _serialize_tool_trace(trace_list: list[Any]) -> list[dict[str, Any]]:
    """把 ToolCallingTrace 列表转成可序列化的字典（供 UI 展示）。"""
    output: list[dict[str, Any]] = []
    for trace in trace_list:
        output.append({
            "iteration": trace.iteration,
            "latency_ms": trace.latency_ms,
            "decisions": trace.decisions,
            "executions": [e.to_display() for e in trace.executions],
        })
    return output


_SYSTEM_PROMPT = """你是美妆电商客服的洞察助手。根据当前对话和用户记忆，输出客服此刻需要的信息。

输出：service_focus（一句话）、labels（0-5个）、service_suggestions（0-2个）。

规则：
1. 只保留对当前回复有直接帮助的信息，按重要程度排列。当前困扰/需求/情绪优先于长期偏好。
2. priority：urgent（安全/强烈情绪/投诉）、high、normal。不夸大。
3. 标签引用1-3个 source_ids（msg_*或memory_id），只忠实压缩原文，不推断。禁用词：可能、潜在、意味着、耐受度、致敏、风险。
4. service_focus：一句话概括客服现在主要处理什么。
5. 建议 action 只能选：empathize、clarify、confirm_need、avoid_repeat、safety_attention、follow_up。只写给客服的沟通指导，不指导消费者停用/就医/用药，不承诺退款赔偿。
6. 无可靠依据时返回空数组。

JSON格式：
{"service_focus":"...","labels":[{"title":"...","content":"...","priority":"urgent|high|normal","source_ids":["..."]}],"service_suggestions":[{"action":"...","content":"...","basis_ids":["..."]}]}
"""


def _number_recent_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [
        {
            "id": f"msg_{index}",
            "role": str(item.get("role", "")),
            "content": str(item.get("content", "")).strip(),
        }
        for index, item in enumerate(messages, start=1)
        if str(item.get("content", "")).strip()
    ]
    return normalized[-MAX_RECENT_MESSAGES:]


def _prepare_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in memories:
        memory_id = str(item.get("id", "")).strip()
        text = str(item.get("memory", "")).strip()
        if not memory_id or not text or memory_id in seen:
            continue
        seen.add(memory_id)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        output.append(
            {
                "id": memory_id,
                "memory_key": str(metadata.get("memory_key", "")),
                "memory": text,
            }
        )
    return output


def _valid_source_ids(values: Any, valid_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(str(value) for value in values if str(value) in valid_ids))[:3]


def _validate_insight(
    payload: Any,
    valid_ids: set[str],
    recent_messages: list[dict[str, str]],
    latency_ms: int,
) -> ServiceInsight:
    if not isinstance(payload, dict):
        payload = {}
    focus = str(payload.get("service_focus", "")).strip()[:80]
    if not focus:
        last_customer = next(
            (item["content"] for item in reversed(recent_messages) if item["role"] == "user"),
            "等待识别当前服务重点",
        )
        focus = last_customer[:80]

    labels: list[ServiceLabel] = []
    seen_labels: set[tuple[str, str]] = set()
    raw_labels = payload.get("labels", [])
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()[:24]
            content = str(item.get("content", "")).strip()[:160]
            priority = str(item.get("priority", "")).strip()
            source_ids = _valid_source_ids(item.get("source_ids"), valid_ids)
            signature = (title, content)
            if (
                not title
                or not content
                or priority not in ALLOWED_PRIORITIES
                or not source_ids
                or any(term in content for term in FORBIDDEN_LABEL_TERMS)
                or signature in seen_labels
            ):
                continue
            seen_labels.add(signature)
            labels.append(ServiceLabel(title, content, priority, source_ids))
    labels.sort(key=lambda item: PRIORITY_RANK[item.priority])

    suggestions: list[ServiceSuggestion] = []
    raw_suggestions = payload.get("service_suggestions", [])
    if isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            content = str(item.get("content", "")).strip()[:160]
            basis_ids = _valid_source_ids(item.get("basis_ids"), valid_ids)
            if (
                action not in ALLOWED_ACTIONS
                or not content
                or not basis_ids
                or any(term in content for term in FORBIDDEN_SUGGESTION_TERMS)
            ):
                continue
            suggestions.append(ServiceSuggestion(action, content, basis_ids))

    return ServiceInsight(
        service_focus=focus,
        labels=tuple(labels[:MAX_LABELS]),
        service_suggestions=tuple(suggestions[:MAX_SUGGESTIONS]),
        latency_ms=latency_ms,
    )
