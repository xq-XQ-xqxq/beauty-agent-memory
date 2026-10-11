"""Context Engineering module for the AI Agent memory system.

4 个组件：
1. TokenCounter — tiktoken 精确计 token，驱动软预算裁剪
2. BM25Retriever — jieba 分词 + BM25 检索，按相关性从用户 Event 中选 top-K
3. ConversationSummarizer — LLM 摘要长对话前期轮次，压缩上下文
4. ContextAssembler — 按优先级组装最终上下文，超软预算时从尾部裁剪

优先级（高→低）：
  当前对话 > Profile > BM25相关Event > 最近Event > 更早Event
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import jieba
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv(Path(__file__).parent / ".env")

# jieba 静默加载
jieba.setLogLevel(20)


# ---------------------------------------------------------------------------
# 1. Token Counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """tiktoken 精确计 token（cl100k_base 兼容 Qwen 系列）。"""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._encoder = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            total += self.count(msg.get("content", ""))
        return total


# ---------------------------------------------------------------------------
# 2. BM25 Retriever
# ---------------------------------------------------------------------------

@dataclass
class RetrievedEvent:
    event: dict[str, Any]
    score: float
    source: str  # "bm25" | "recent"


class BM25Retriever:
    """BM25 事件检索器：jieba 分词 + BM25Okapi。

    用法：用当前对话内容做 query，从用户全部 Event 中检索 top-K 相关事件。
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self._corpus = [self._tokenize(self._event_text(e)) for e in events]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    @staticmethod
    def _event_text(item: dict[str, Any]) -> str:
        """展平为可检索文本，兼容 Event 和 Memory 格式。"""
        parts: list[str] = []
        for key in ("event_type", "memory", "description"):
            val = item.get(key)
            if val:
                parts.append(str(val))
        fields = item.get("fields")
        if isinstance(fields, dict):
            parts.extend(str(v) for v in fields.values() if v)
        evidence = item.get("evidence", [])
        if isinstance(evidence, list):
            parts.extend(evidence)
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 分词，过滤单字噪声。"""
        return [t.strip() for t in jieba.cut(text) if t.strip() and len(t.strip()) > 1]

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedEvent]:
        if not self._bm25:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[RetrievedEvent] = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append(RetrievedEvent(
                    event=self._events[idx],
                    score=float(score),
                    source="bm25",
                ))
        return results


# ---------------------------------------------------------------------------
# 3. Conversation Summarizer
# ---------------------------------------------------------------------------

class ConversationSummarizer:
    """LLM 摘要长对话前期轮次。

    对话 > recent_count 轮时，把前 N 轮摘要成 ≤200 字，后 recent_count 轮保留原文。
    按 (conversation_id, message_hash) 缓存，同一对话不重复摘要。
    """

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
        self._cache: dict[str, str] = {}
        self.last_latency_ms: int = 0
        self.last_usage: dict[str, int] = {}

    def _cache_key(self, conversation_id: str, messages: list[dict]) -> str:
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return f"{conversation_id}:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def summarize_if_needed(
        self,
        messages: list[dict[str, str]],
        conversation_id: str = "",
        recent_count: int = 10,
    ) -> tuple[str, list[dict[str, str]], bool]:
        """返回 (summary, recent_messages, was_summarized)。

        - len(messages) <= recent_count → ("", messages, False)
        - len(messages) > recent_count → (summary, messages[-recent_count:], True)
        """
        if len(messages) <= recent_count:
            return "", messages, False

        old_messages = messages[:-recent_count]
        recent = messages[-recent_count:]

        key = self._cache_key(conversation_id, old_messages)
        if key in self._cache:
            return self._cache[key], recent, True

        summary = self._llm_summarize(old_messages)
        self._cache[key] = summary
        return summary, recent, True

    def _llm_summarize(self, messages: list[dict[str, str]]) -> str:
        conversation = "\n".join(
            f"[{m.get('role', '')}] {m.get('content', '')}" for m in messages
        )
        system = (
            "你是对话摘要助手。将客服对话压缩成 200 字以内的摘要，保留："
            "讨论的产品、用户的问题/症状、已达成的共识、用户的情绪状态。"
            "不要添加任何推断或诊断。只输出摘要文本。"
        )
        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": conversation},
            ],
        )
        self.last_latency_ms = round((perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        } if usage else {}
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 4. Context Assembler
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    """组装完成的上下文，供 LLM prompt 使用。"""

    # 四段式上下文文本
    conversation_text: str  # 编号对话（或摘要 + 最近轮次）
    profile_text: str  # 紧凑 Profile
    events_text: str  # 检索 + 最近 Event 合并
    knowledge_text: str  # 知识库检索结果（产品档案 + 成分百科）

    # 诊断元数据
    total_tokens: int
    token_budget: int
    was_truncated: bool
    conversation_summarized: bool
    bm25_event_count: int
    recent_event_count: int
    summary_latency_ms: int = 0
    summary_tokens: int = 0
    knowledge_doc_count: int = 0


class ContextAssembler:
    """按优先级组装 LLM 上下文，超软预算时从尾部裁剪。

    优先级（高→低）：
    1. 当前对话（或摘要 + 最近轮次）— 必须完整
    2. Profile（全量，通常很小）— 优先保留
    3. BM25 相关 Event — 按分数排序
    4. 最近 Event（不在 BM25 结果中的）— 按时间排序
    5. 裁剪时从 4 → 3 依次删除
    """

    SOFT_BUDGET = 6000  # tokens，软限制

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        summarizer: ConversationSummarizer | None = None,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.summarizer = summarizer or ConversationSummarizer()

    def assemble(
        self,
        messages: list[dict[str, str]],
        profile: dict[str, dict[str, Any]],
        events: list[dict[str, Any]],
        conversation_id: str = "",
        recent_event_count: int = 5,
        bm25_top_k: int = 10,
        knowledge_docs: list[dict[str, Any]] | None = None,
    ) -> AssembledContext:
        # Step 1: 长对话压缩
        summary, recent_messages, was_summarized = self.summarizer.summarize_if_needed(
            messages, conversation_id
        )
        conversation_text = self._format_conversation(recent_messages, summary)
        conv_tokens = self.token_counter.count(conversation_text)

        # Step 2: Profile（紧凑，全量）
        profile_text = self._format_profile(profile)
        profile_tokens = self.token_counter.count(profile_text)

        # Step 3: Event 检索（BM25 + 最近补充）
        query = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        retriever = BM25Retriever(events)
        bm25_results = retriever.retrieve(query, top_k=bm25_top_k)

        bm25_ids = {r.event.get("event_id") for r in bm25_results}
        recent_events = [
            e for e in events[:recent_event_count]
            if e.get("event_id") not in bm25_ids
        ]
        merged_events = [r.event for r in bm25_results] + recent_events

        # Step 4: 知识库检索结果（外部注入）
        knowledge_text = self._format_knowledge(knowledge_docs or [])
        knowledge_tokens = self.token_counter.count(knowledge_text)

        # Step 5: 软预算裁剪
        events_text = self._format_events(merged_events)
        events_tokens = self.token_counter.count(events_text)

        total_tokens = conv_tokens + profile_tokens + events_tokens + knowledge_tokens
        was_truncated = False

        if total_tokens > self.SOFT_BUDGET:
            excess = total_tokens - self.SOFT_BUDGET
            # 优先裁剪 Event（知识库文档更精确，优先保留）
            while merged_events and events_tokens > excess:
                merged_events.pop()
                events_text = self._format_events(merged_events)
                events_tokens = self.token_counter.count(events_text)
                total_tokens = conv_tokens + profile_tokens + events_tokens + knowledge_tokens
                was_truncated = True

        return AssembledContext(
            conversation_text=conversation_text,
            profile_text=profile_text,
            events_text=events_text,
            knowledge_text=knowledge_text,
            total_tokens=total_tokens,
            token_budget=self.SOFT_BUDGET,
            was_truncated=was_truncated,
            conversation_summarized=was_summarized,
            bm25_event_count=len(bm25_results),
            recent_event_count=len(merged_events) - len(bm25_results) if not was_truncated else 0,
            summary_latency_ms=self.summarizer.last_latency_ms,
            summary_tokens=self.summarizer.last_usage.get("total_tokens", 0),
            knowledge_doc_count=len(knowledge_docs or []),
        )

    # ------------------------------------------------------------------
    # 格式化方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation(
        messages: list[dict[str, str]],
        summary: str = "",
    ) -> str:
        parts: list[str] = []
        if summary:
            parts.append(f"[前期对话摘要]\n{summary}\n")
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"[{i}][{role}] {content}")
        return "\n".join(parts)

    @staticmethod
    def _format_profile(profile: dict[str, dict[str, Any]]) -> str:
        if not profile:
            return "（无）"
        lines: list[str] = []
        for key, entry in sorted(profile.items()):
            value = entry.get("value", {})
            # SET 类型 profile 的 value 是 list（如 [{"ingredient": "..."}]），
            # SINGLE 类型是 dict。两种都要处理。
            if isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict):
                        items.append("{" + "; ".join(
                            f"{k}={v}" for k, v in item.items() if v not in (None, "", [])
                        ) + "}")
                    else:
                        items.append(str(item))
                body = ", ".join(items)
            elif isinstance(value, dict):
                body = "; ".join(f"{k}={v}" for k, v in value.items() if v not in (None, "", []))
            else:
                body = str(value)
            lines.append(f"- {key}: {body}")
        return "\n".join(lines)

    @staticmethod
    def _format_events(events: list[dict[str, Any]]) -> str:
        if not events:
            return "（无）"
        lines: list[str] = []
        for event in events:
            etype = event.get("event_type", "")
            desc = event.get("description", "")
            lines.append(f"- [{etype}] {desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_knowledge(docs: list[dict[str, Any]]) -> str:
        if not docs:
            return ""
        lines: list[str] = []
        for doc in docs:
            dtype = doc.get("doc_type", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            # 截断过长内容（避免单条文档吃掉太多 token）
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"- [{dtype}] {title}: {content}")
        return "\n".join(lines)
