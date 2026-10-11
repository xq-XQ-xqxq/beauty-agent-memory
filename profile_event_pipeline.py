"""Profile+Event dual-channel extraction pipeline.

提取流程：
- Step 0 工具查历史：从 ProfileEventStore 读取该用户已有的 Profile + Event，
  作为 LLM 的上下文，让 LLM 只提取增量。
- Step 1 LLM 抽取：LLM 输出半结构化 JSON（自然语言值，如"敏感"而非"sensitive"），
  不含 operation_hint，不参与 ADD/UPDATE/DELETE 决策。
- Step 2 程序归一化：枚举码归一化（"敏感"→"sensitive"）、必填字段校验、
  按编号读取 evidence 原文（不让 LLM 复制原文）。
- Step 3 程序去重：Profile 100% 程序决策；Event 90% 程序 + 10% LLM 二分类。
- Step 4 程序执行：根据去重结果调用 store 的 ADD/UPDATE。

LLM 与工具分工原则：
- LLM 负责：事件理解、自然语言→半结构化转换、签名相同时的语义二分类。
- 程序负责：枚举归一化、必填校验、evidence 原文读取、精确/签名去重、ADD/UPDATE 执行。
- LLM 不负责：决策最终写入操作；真实数据查询与计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from context_engine import ContextAssembler, TokenCounter
from hybrid_retriever import HybridRetriever
from knowledge_retriever import RetrievedDoc
from memory_lifecycle import (
    aggregate_events_to_profile,
    apply_sentiment_ttl,
    detect_profile_conflict,
)
from profile_event_dedup import (
    EventDecision,
    ExistingEvent,
    ProfileDecision,
    SameDifferentJudge,
    decide_event_add,
    decide_profile_update,
    merge_set_item,
)
from profile_event_schema import (
    PROFILE_SPEC_MAP,
    EVENT_SPEC_MAP,
    get_spec,
    is_event_key,
    is_profile_key,
    normalize_value,
    schema_prompt,
    validate_value,
)
from profile_event_store import ProfileEventStore, render_event_text, render_profile_text

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Idempotency (mirrors processed_turn_store semantics)
# ---------------------------------------------------------------------------

def canonical_turn(messages: list[dict[str, Any]]) -> str:
    normalized = [
        {"role": str(x.get("role", "")), "content": str(x.get("content", "")).strip()}
        for x in messages
    ]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def turn_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_turn(messages).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Extracted item dataclasses (LLM-facing, pre-normalization)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawProfileUpdate:
    profile_key: str
    fields: dict[str, Any]
    evidence_message_ids: tuple[int, ...]
    description: str = ""


@dataclass(frozen=True)
class RawNewEvent:
    event_type: str
    fields: dict[str, Any]
    evidence_message_ids: tuple[int, ...]
    description: str = ""
    occurred_at: str = ""


@dataclass(frozen=True)
class ExtractedItems:
    profile_updates: tuple[RawProfileUpdate, ...]
    new_events: tuple[RawNewEvent, ...]
    raw_llm_payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Compact schema prompt (avoids overwhelming LLM with full enum lists)
# ---------------------------------------------------------------------------

def _compact_schema_prompt() -> str:
    """Compact schema: show key, required fields, and a one-line description.

    Full enum normalization happens in Step 2 (program-side), so the LLM only
    needs to know the key names and what fields to fill — not the complete
    enum value lists.  This keeps the system prompt short and focused.
    """
    from profile_event_schema import PROFILE_SPEC_MAP, EVENT_SPEC_MAP

    lines = ["Profile 通道（长期可变画像，可覆盖更新）："]
    for spec in PROFILE_SPEC_MAP.values():
        fields = ", ".join(spec.required_fields)
        lines.append(f"- {spec.key} | 必填:{fields} | {spec.description}")

    lines.append("")
    lines.append("Event 通道（不可变事件流，按时间追加）：")
    for spec in EVENT_SPEC_MAP.values():
        fields = ", ".join(spec.required_fields)
        lines.append(f"- {spec.key} | 必填:{fields} | {spec.description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: LLM extraction
# ---------------------------------------------------------------------------

class IncrementalExtractor:
    """LLM extracts only incremental facts given assembled context.

    上下文工程集成点：使用 ContextAssembler 替代原来的全量注入。
    - 长对话自动摘要（>10 轮时触发 LLM 摘要）
    - Event 通过 BM25 检索 top-K 相关事件（而非全量注入）
    - 软 token 预算管理（超 6000 token 时从尾部裁剪）
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        context_assembler: ContextAssembler | None = None,
        knowledge_retriever: HybridRetriever | None = None,
    ) -> None:
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
        self.context_assembler = context_assembler or ContextAssembler()
        self.knowledge_retriever = knowledge_retriever or HybridRetriever()

    def extract(
        self,
        messages: list[dict[str, Any]],
        existing_profile: dict[str, dict[str, Any]],
        existing_events: list[dict[str, Any]],
        conversation_id: str = "",
    ) -> tuple[ExtractedItems, dict[str, Any]]:
        system_prompt = self._system_prompt()

        # 知识库检索：从对话中提取产品/成分关键词，检索知识库
        query = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        kb_results = self.knowledge_retriever.search(query, doc_type="product", top_k=3)
        kb_results += self.knowledge_retriever.search(query, doc_type="ingredient", top_k=3)
        knowledge_docs = [
            {"doc_type": d.doc_type, "title": d.title, "content": d.content}
            for d in kb_results
        ]

        # 上下文工程：用 ContextAssembler 组装 user prompt（含知识库）
        ctx = self.context_assembler.assemble(
            messages=messages,
            profile=existing_profile,
            events=existing_events,
            conversation_id=conversation_id,
            knowledge_docs=knowledge_docs,
        )
        user_prompt = (
            f"已存在的 Profile：\n{ctx.profile_text}\n\n"
            f"已存在的 Events（BM25 检索 {ctx.bm25_event_count} 条 + 最近补充）：\n{ctx.events_text}\n\n"
        )
        if ctx.knowledge_text:
            user_prompt += f"知识库参考（产品档案 + 成分百科）：\n{ctx.knowledge_text}\n\n"
        user_prompt += f"请从以下编号对话中提取增量事实：\n\n{ctx.conversation_text}"

        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = round((perf_counter() - started) * 1000)
        payload = json.loads(response.choices[0].message.content or "{}")
        usage = self._extract_usage(response)
        return self._parse_payload(payload, messages), {
            "latency_ms": latency_ms,
            "usage": usage,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            # 上下文工程指标
            "context_tokens": ctx.total_tokens,
            "context_truncated": ctx.was_truncated,
            "conversation_summarized": ctx.conversation_summarized,
            "bm25_event_count": ctx.bm25_event_count,
            "recent_event_count": ctx.recent_event_count,
            "summary_latency_ms": ctx.summary_latency_ms,
            "summary_tokens": ctx.summary_tokens,
            "knowledge_doc_count": ctx.knowledge_doc_count,
        }

    def _extract_usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return f"""你是美妆电商客服的用户记忆抽取 Agent。把对话中对后续服务有价值的事实转成 JSON。

存储分两个通道：
1. Profile（长期画像）：肤质、敏感度、成分过敏等稳定属性，可被新事实覆盖
2. Event（事件流）：用户的经历、需求、困扰、不满、服务共识、情绪等，按时间追加

可用的 key 与必填字段：

{_compact_schema_prompt()}

抽取规则：
1. 只提取对话原文明确支持的事实。不提取问候、订单金额、物流工单、模板话术。
2. "已存在"的事实不要重复，只提取新增的。
3. 枚举字段（情绪、肤质、目的等）用中文自然描述即可，程序会自动归一化。
4. evidence_message_ids 引用对话编号（从1开始），1-3 条。不要复制原文。
5. 不输出 ADD/UPDATE/DELETE 决策，由程序完成。
6. 产品咨询也是有效记忆：送礼/选品/赠品/成分/预算等均需提取。示例：
   - 用户问"有适合送礼的套装吗"→ new_events: need.information(topic="套装推荐", question="送礼套装")
   - 用户说"送人的，想买个套装"→ new_events: purchase.context(purpose="gift")
   - 客服推荐方案→ new_events: service.plan_response(plan="...", subject="...", response="接受/考虑中/拒绝")
   - 客服承诺赠品→ new_events: service.commitment(owner="merchant", action="送...", subject="赠品")
7. "孕妇能用吗"不代表用户怀孕；客服推测不能成为用户事实。

仅输出 JSON：
{{"profile_updates": [{{"profile_key": "…", "fields": {{…}}, "evidence_message_ids": [1]}}], "new_events": [{{"event_type": "…", "fields": {{…}}, "evidence_message_ids": [1]}}]}}
"""

    # ------------------------------------------------------------------
    # Payload parsing (light parse only; normalization happens in Step 2)
    # ------------------------------------------------------------------

    def _parse_payload(self, payload: Any, messages: list[dict[str, Any]]) -> ExtractedItems:
        if not isinstance(payload, dict):
            payload = {}

        raw_profiles: list[RawProfileUpdate] = []
        for item in payload.get("profile_updates", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("profile_key", "")).strip()
            if not key:
                continue
            raw_profiles.append(RawProfileUpdate(
                profile_key=key,
                fields=item.get("fields", {}) if isinstance(item.get("fields"), dict) else {},
                evidence_message_ids=_parse_message_ids(item.get("evidence_message_ids"), len(messages)),
                description=str(item.get("description", "")).strip(),
            ))

        raw_events: list[RawNewEvent] = []
        for item in payload.get("new_events", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("event_type", "")).strip()
            if not key:
                continue
            raw_events.append(RawNewEvent(
                event_type=key,
                fields=item.get("fields", {}) if isinstance(item.get("fields"), dict) else {},
                evidence_message_ids=_parse_message_ids(item.get("evidence_message_ids"), len(messages)),
                description=str(item.get("description", "")).strip(),
                occurred_at=str(item.get("occurred_at", "")).strip(),
            ))

        return ExtractedItems(
            profile_updates=tuple(raw_profiles),
            new_events=tuple(raw_events),
            raw_llm_payload=payload if isinstance(payload, dict) else {},
        )


def _parse_message_ids(raw: Any, message_count: int) -> tuple[int, ...]:
    if isinstance(raw, (int, str)):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    output: list[int] = []
    for value in raw:
        try:
            message_id = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= message_id <= message_count and message_id not in output:
            output.append(message_id)
    return tuple(output)


# ---------------------------------------------------------------------------
# Step 2 + 3 + 4: normalize → dedup → execute
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    profile_decisions: tuple[dict[str, Any], ...]
    event_decisions: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    idempotent_skip: bool
    source_turn_hash: str
    extracted_items: dict[str, Any]
    latency_ms: int = 0
    token_usage: dict[str, int] = None  # type: ignore[assignment]
    llm_judge_calls: int = 0
    llm_judge_latency_ms: int = 0
    llm_judge_token_usage: dict[str, int] = None  # type: ignore[assignment]
    # 上下文工程指标
    context_tokens: int = 0
    context_truncated: bool = False
    conversation_summarized: bool = False
    bm25_event_count: int = 0
    summary_latency_ms: int = 0
    summary_tokens: int = 0
    # 记忆生命周期指标
    sentiment_reset: bool = False
    events_tagged: int = 0
    aggregation_count: int = 0
    # 知识库检索指标
    knowledge_doc_count: int = 0

    def __post_init__(self) -> None:
        if self.token_usage is None:
            object.__setattr__(self, "token_usage", {})
        if self.llm_judge_token_usage is None:
            object.__setattr__(self, "llm_judge_token_usage", {})


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileEventPipeline:
    """Orchestrates Step 0 → Step 4 over a ProfileEventStore."""

    def __init__(
        self,
        store: ProfileEventStore,
        extractor: IncrementalExtractor | None = None,
        judge: SameDifferentJudge | None = None,
        processed_turns_path: Path | str | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or IncrementalExtractor()
        self.judge = judge or SameDifferentJudge()
        # Reuse the existing ProcessedTurnStore for idempotency.
        from processed_turn_store import ProcessedTurnStore
        if processed_turns_path is None:
            processed_turns_path = Path(store.path).parent / "profile_event_processed_turns.sqlite3"
        self.processed_turn_store = ProcessedTurnStore(Path(processed_turns_path))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_turn(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        conversation_id: str | None = None,
        event_id_hint: str | None = None,
        scene_major: str | None = None,
        scene_minor: str | None = None,
        occurred_at: str | None = None,
    ) -> PipelineResult:
        if not messages:
            raise ValueError("messages must not be empty")

        source_hash = turn_hash(messages)
        previous = self.processed_turn_store.get(user_id, source_hash)
        if previous is not None:
            return PipelineResult(
                profile_decisions=(),
                event_decisions=(),
                rejected=(),
                idempotent_skip=True,
                source_turn_hash=source_hash,
                extracted_items={"previous_result": previous},
            )

        # Step 0a: 记忆生命周期 — 情绪 TTL + 解决标签。
        ttl_result = apply_sentiment_ttl(
            self.store, user_id, conversation_id or "",
            is_new_conversation=conversation_id is not None,
        )

        # Step 0: query history.
        existing_profile = self.store.get_all_profile(user_id)
        existing_events = self.store.get_events(user_id)  # flat list for BM25

        # Step 1: LLM extract increment (with ContextAssembler).
        self._judge_calls = 0
        self._judge_latency_ms = 0
        self._judge_token_usage: dict[str, int] = {}
        extracted, extract_stats = self.extractor.extract(
            messages, existing_profile, existing_events, conversation_id or "",
        )

        # Step 2 + 3 + 4: normalize → validate → dedup → execute for each item.
        profile_results, profile_rejects = self._process_profile_updates(
            extracted.profile_updates, messages, user_id, source_hash, conversation_id,
            scene_major, scene_minor, occurred_at,
        )
        event_results, event_rejects = self._process_new_events(
            extracted.new_events, messages, user_id, source_hash, conversation_id,
            scene_major, scene_minor, occurred_at,
        )

        # Step 5: 记忆生命周期 — Event→Profile 聚合。
        agg_result = aggregate_events_to_profile(self.store, user_id)
        for candidate in agg_result.new_profile_entries:
            pkey = candidate["profile_key"]
            pvalue = candidate["value"]
            # 归一化 + 校验
            normalized, _ = normalize_value(pkey, pvalue)
            errors = validate_value(pkey, normalized)
            if not errors:
                spec = PROFILE_SPEC_MAP[pkey]
                existing = self.store.get_profile(user_id, pkey)
                if spec.cardinality == "multi":
                    merged = merge_set_item(pkey, normalized, existing["value"] if existing else None)
                    self.store.upsert_profile(
                        user_id=user_id, profile_key=pkey,
                        value=merged, evidence=[],
                        source_turn_hash=source_hash,
                    )
                else:
                    self.store.upsert_profile(
                        user_id=user_id, profile_key=pkey,
                        value=normalized, evidence=[],
                        source_turn_hash=source_hash,
                    )
                profile_results.append({
                    "profile_key": pkey,
                    "operation": "AGGREGATE",
                    "value": normalized,
                    "reason": f"Event 聚合触发：{candidate['trigger']}（{candidate['evidence_count']} 条证据）",
                    "used_llm": False,
                })

        result = PipelineResult(
            profile_decisions=tuple(profile_results),
            event_decisions=tuple(event_results),
            rejected=tuple(profile_rejects + event_rejects),
            idempotent_skip=False,
            source_turn_hash=source_hash,
            extracted_items={
                "raw_profile_updates": [self._raw_profile_to_dict(p) for p in extracted.profile_updates],
                "raw_new_events": [self._raw_event_to_dict(e) for e in extracted.new_events],
            },
            latency_ms=extract_stats.get("latency_ms", 0),
            token_usage=extract_stats.get("usage", {}),
            llm_judge_calls=self._judge_calls,
            llm_judge_latency_ms=self._judge_latency_ms,
            llm_judge_token_usage=dict(self._judge_token_usage),
            # 上下文工程指标
            context_tokens=extract_stats.get("context_tokens", 0),
            context_truncated=extract_stats.get("context_truncated", False),
            conversation_summarized=extract_stats.get("conversation_summarized", False),
            bm25_event_count=extract_stats.get("bm25_event_count", 0),
            summary_latency_ms=extract_stats.get("summary_latency_ms", 0),
            summary_tokens=extract_stats.get("summary_tokens", 0),
            # 记忆生命周期指标
            sentiment_reset=ttl_result.sentiment_reset,
            events_tagged=ttl_result.events_tagged,
            aggregation_count=len(agg_result.new_profile_entries),
            # 知识库检索指标
            knowledge_doc_count=extract_stats.get("knowledge_doc_count", 0),
        )
        self.processed_turn_store.record(user_id, source_hash, {
            "profile_decisions": profile_results,
            "event_decisions": event_results,
            "rejected": profile_rejects + event_rejects,
        })
        return result

    # ------------------------------------------------------------------
    # Profile processing
    # ------------------------------------------------------------------

    def _process_profile_updates(
        self,
        updates: tuple[RawProfileUpdate, ...],
        messages: list[dict[str, Any]],
        user_id: str,
        source_hash: str,
        conversation_id: str | None,
        scene_major: str | None,
        scene_minor: str | None,
        occurred_at: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        rejects: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()

        for raw in updates:
            if not is_profile_key(raw.profile_key):
                rejects.append(self._reject(raw.profile_key, raw.fields, raw.evidence_message_ids, "非 Profile 通道 key", messages))
                continue

            # Step 2: normalize + validate.
            normalized, warnings = normalize_value(raw.profile_key, raw.fields)
            errors = validate_value(raw.profile_key, normalized)
            evidence = self._read_evidence(raw.evidence_message_ids, messages)
            if not evidence:
                errors.append("缺少来源消息或编号越界")
            if errors:
                rejects.append(self._reject(raw.profile_key, normalized, raw.evidence_message_ids, "；".join(errors), messages, warnings))
                continue

            # Step 3: dedup (100% program).
            existing = self.store.get_profile(user_id, raw.profile_key)
            spec = PROFILE_SPEC_MAP[raw.profile_key]
            if spec.cardinality == "multi":
                # SET: first write wraps the item in a list; later writes merge.
                if existing is None:
                    merged_value = [normalized]
                    already_present = False
                    reason = "SET 首次写入"
                else:
                    merged_value = merge_set_item(raw.profile_key, normalized, existing["value"])
                    new_sig = match_signature_safe(raw.profile_key, normalized)
                    already_present = any(
                        isinstance(item, dict) and match_signature_safe(raw.profile_key, item) == new_sig
                        for item in merged_value[:-1]
                    )
                    reason = "SET 追加新 item"
                if already_present:
                    results.append(self._decision_dict(
                        raw.profile_key, normalized, evidence, "NOOP",
                        "SET 中已有同签名 item", used_llm=False,
                    ))
                    continue
                decision_value = merged_value
                operation = "UPDATE"
            else:
                # SINGLE: compare full value.
                decision = decide_profile_update(raw.profile_key, normalized, list(evidence), existing)
                if decision.operation == "NOOP":
                    results.append(self._decision_dict(
                        raw.profile_key, normalized, evidence, "NOOP",
                        decision.reason, used_llm=False,
                    ))
                    continue
                decision_value = normalized
                reason = decision.reason
                operation = "UPDATE"

            batch_key = f"profile::{raw.profile_key}::{json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"
            if batch_key in seen_in_batch:
                results.append(self._decision_dict(
                    raw.profile_key, normalized, evidence, "NOOP",
                    "本批次重复提取", used_llm=False,
                ))
                continue
            seen_in_batch.add(batch_key)

            # Step 3b: 记忆生命周期 — 冲突检测 + 归档（仅 SINGLE）。
            # 注意：不重复添加 decision，只改写 reason 标记冲突。
            conflict_reason = ""
            if spec.cardinality != "multi" and existing is not None:
                conflict_result = detect_profile_conflict(
                    self.store, user_id, raw.profile_key, normalized,
                )
                if conflict_result.has_conflict:
                    conflict_reason = f" | ⚠️ 冲突：{conflict_result.old_value} → {conflict_result.new_value}"

            # Step 4: execute UPDATE.
            self.store.upsert_profile(
                user_id=user_id,
                profile_key=raw.profile_key,
                value=decision_value,
                evidence=list(evidence),
                source_event_id=None,
                source_turn_hash=source_hash,
            )
            results.append(self._decision_dict(
                raw.profile_key, normalized, evidence, operation, reason + conflict_reason,
                used_llm=False, stored_value=decision_value,
            ))
        return results, rejects

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_new_events(
        self,
        events: tuple[RawNewEvent, ...],
        messages: list[dict[str, Any]],
        user_id: str,
        source_hash: str,
        conversation_id: str | None,
        scene_major: str | None,
        scene_minor: str | None,
        occurred_at: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        rejects: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()

        for raw in events:
            if not is_event_key(raw.event_type):
                rejects.append(self._reject(raw.event_type, raw.fields, raw.evidence_message_ids, "非 Event 通道 key", messages))
                continue

            # Step 2: normalize + validate.
            normalized, warnings = normalize_value(raw.event_type, raw.fields)
            # Auto-fill observed_at for TRANSIENT events; LLM lacks wall-clock time.
            if "observed_at" in (EVENT_SPEC_MAP[raw.event_type].required_fields) and not normalized.get("observed_at"):
                normalized["observed_at"] = occurred_at or _utcnow_iso()
            errors = validate_value(raw.event_type, normalized)
            evidence = self._read_evidence(raw.evidence_message_ids, messages)
            if not evidence:
                errors.append("缺少来源消息或编号越界")
            if errors:
                rejects.append(self._reject(raw.event_type, normalized, raw.evidence_message_ids, "；".join(errors), messages, warnings))
                continue

            # Step 3: dedup (program-first, LLM only when signature matches but fields differ).
            existing_rows = self.store.get_events(user_id, event_type=raw.event_type)
            existing = [
                ExistingEvent(
                    event_id=row["event_id"],
                    fields=row["fields"],
                    evidence=list(row.get("evidence", [])),
                )
                for row in existing_rows
            ]
            decision = decide_event_add(
                event_type=raw.event_type,
                new_fields=normalized,
                new_evidence=list(evidence),
                existing_events=existing,
                llm_judge=self.judge,
            )
            if decision.used_llm:
                self._judge_calls += 1
                self._judge_latency_ms += getattr(self.judge, "last_latency_ms", 0)
                last_usage = getattr(self.judge, "last_usage", {}) or {}
                for key, value in last_usage.items():
                    self._judge_token_usage[key] = self._judge_token_usage.get(key, 0) + value

            batch_key = f"event::{raw.event_type}::{json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"
            if decision.operation == "ADD" and batch_key in seen_in_batch:
                results.append(self._decision_dict(
                    raw.event_type, normalized, evidence, "NOOP",
                    "本批次重复提取", used_llm=False,
                ))
                continue
            seen_in_batch.add(batch_key)

            if decision.operation == "NOOP":
                results.append(self._decision_dict(
                    raw.event_type, normalized, evidence, "NOOP",
                    decision.reason, used_llm=decision.used_llm,
                    matched_event_id=decision.matched_event_id,
                ))
                continue

            # Step 4: execute ADD.
            event_id = self.store.add_event(
                user_id=user_id,
                event_type=raw.event_type,
                fields=normalized,
                evidence=list(evidence),
                description=raw.description,
                occurred_at=raw.occurred_at or occurred_at,
                conversation_id=conversation_id,
                scene_major=scene_major,
                scene_minor=scene_minor,
                source_turn_hash=source_hash,
            )
            results.append(self._decision_dict(
                raw.event_type, normalized, evidence, "ADD",
                decision.reason, used_llm=decision.used_llm,
                stored_event_id=event_id,
            ))
        return results, rejects

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_evidence(
        self, message_ids: tuple[int, ...], messages: list[dict[str, Any]],
    ) -> tuple[str, ...]:
        return tuple(
            str(messages[i - 1].get("content", ""))
            for i in message_ids
            if 1 <= i <= len(messages) and messages[i - 1].get("content")
        )

    def _reject(
        self,
        key: str,
        value: dict[str, Any],
        message_ids: tuple[int, ...],
        reason: str,
        messages: list[dict[str, Any]],
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        evidence = self._read_evidence(message_ids, messages)
        return {
            "key": key,
            "value": value,
            "evidence": list(evidence),
            "evidence_message_ids": list(message_ids),
            "reasons": ([reason] + (warnings or [])),
        }

    def _decision_dict(
        self,
        key: str,
        value: dict[str, Any],
        evidence: tuple[str, ...],
        operation: str,
        reason: str,
        used_llm: bool = False,
        matched_event_id: str | None = None,
        stored_value: dict[str, Any] | None = None,
        stored_event_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "key": key,
            "value": value,
            "stored_value": stored_value,
            "stored_event_id": stored_event_id,
            "matched_event_id": matched_event_id,
            "evidence": list(evidence),
            "operation": operation,
            "reason": reason,
            "used_llm": used_llm,
        }

    def _raw_profile_to_dict(self, item: RawProfileUpdate) -> dict[str, Any]:
        return {
            "profile_key": item.profile_key,
            "fields": item.fields,
            "evidence_message_ids": list(item.evidence_message_ids),
            "description": item.description,
        }

    def _raw_event_to_dict(self, item: RawNewEvent) -> dict[str, Any]:
        return {
            "event_type": item.event_type,
            "fields": item.fields,
            "evidence_message_ids": list(item.evidence_message_ids),
            "description": item.description,
            "occurred_at": item.occurred_at,
        }


def match_signature_safe(memory_key: str, value: dict[str, Any]) -> tuple[str, ...]:
    """Local import to avoid circular dependencies."""

    from profile_event_schema import match_signature as _impl
    return _impl(memory_key, value)
