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


class ServiceInsightGenerator:
    """Generate a compact, evidence-bound service view in one LLM call."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get(
                "DASHSCOPE_API_KEY", ""
            )
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")

    def generate(
        self,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> ServiceInsight:
        recent_messages = _number_recent_messages(messages)
        memory_sources = _prepare_memories(memories)
        valid_ids = {
            item["id"] for item in recent_messages
        } | {item["id"] for item in memory_sources}

        prompt = {
            "recent_conversation": recent_messages,
            "complete_user_memory": memory_sources,
        }
        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        )
        latency_ms = round((perf_counter() - started) * 1000)
        payload = json.loads(response.choices[0].message.content or "{}")
        return _validate_insight(payload, valid_ids, recent_messages, latency_ms)


# Compatibility alias for callers that still import the old class name.
LabelGenerator = ServiceInsightGenerator


_SYSTEM_PROMPT = """你是一线美妆电商客服的服务洞察助手。你的任务不是写入记忆，而是根据当前对话和完整用户记忆，决定客服此刻最应该看到什么。

要求：
1. 一次性输出当前服务焦点、0至5个动态标签、0至2条服务建议。
2. 标签的类别、标题和内容可自由决定；只保留对当前回复有直接帮助的信息，并按重要程度排列。当前消息里的明确困扰、需求或强烈情绪通常应先于长期偏好；弱相关信息不展示。
3. priority只能是urgent、high、normal。身体安全、强烈情绪、投诉升级等可设为urgent；普通相关事实不得夸大。
4. 每个标签必须引用1至3个真实source_ids，只能使用输入中的msg_*或memory_id。标签内容只能忠实压缩来源原文，不得添加“可能、潜在、意味着、耐受度、致敏、风险”等推断、因果或诊断。例如记忆只有“用户是敏感肌”，标签只能写“用户自述敏感肌”，不能写“耐受度较低”。
5. 当前消息中的事实可以直接生成标签，不要求它已经写入长期记忆。长期事实只能来自complete_user_memory。
6. service_focus用一句简短中文概括客服现在主要处理什么；它是临时会话摘要，不是长期事实。
7. 服务建议只用于沟通辅助，action只能是：empathize、clarify、confirm_need、avoid_repeat、safety_attention、follow_up。
8. 每条建议必须引用1至3个真实basis_ids。建议的对象是客服，只能写“先回应用户的什么感受”“先询问什么信息”“先确认什么诉求”“关注什么变化”。不得直接指导消费者停用、清洁、就医或用药；不得诊断、归因、承诺退款赔偿、批准业务操作或断言商品质量。
9. 没有可靠依据时，labels或service_suggestions返回空数组。不要为了凑数展示无关记忆。
10. 只输出JSON，格式：
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
