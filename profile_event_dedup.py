"""Deduplication for the Profile+Event dual-channel pipeline.

去重策略与 LLM 分工（呼应设计原则）：
- Profile 通道（长期稳定属性）：
  * SINGLE 类型：精确匹配 value 完全相同 → 跳过；否则覆盖更新。100% 程序决策。
  * SET 类型：item 签名已在集合中 → 跳过；否则追加到集合。100% 程序决策。
- Event 通道（不可变事件流）：
  * 精确匹配 fields 完全相同 → 跳过。
  * 签名匹配不同 → 一定是新事件，直接 ADD。
  * 签名相同但 fields 不同 → 调 LLM 判断 same/different（仅此分支使用 LLM）。
    - same → 跳过（视为同事件重复叙述）
    - different → ADD 为新事件
- LLM 仅输出 "same" / "different" 二分类，不参与 ADD/UPDATE/DELETE 决策；
  程序根据二分类结果执行对应写入操作。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from memory_schema import MemoryKeySpec

from profile_event_schema import (
    get_spec,
    is_profile_key,
    match_signature,
    normalize_value,
    value_fingerprint,
)

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfileDecision:
    operation: str  # "UPDATE" | "NOOP"
    profile_key: str
    value: dict[str, Any]
    evidence: list[str]
    reason: str


@dataclass(frozen=True)
class EventDecision:
    operation: str  # "ADD" | "NOOP"
    event_type: str
    fields: dict[str, Any]
    evidence: list[str]
    reason: str
    matched_event_id: str | None = None  # 仅 NOOP 时填充，便于追溯
    used_llm: bool = False


# ---------------------------------------------------------------------------
# Profile dedup (100% program)
# ---------------------------------------------------------------------------

def decide_profile_update(
    profile_key: str,
    new_value: dict[str, Any],
    new_evidence: list[str],
    existing: dict[str, Any] | None,
) -> ProfileDecision:
    """Decide whether to overwrite the Profile slot.

    - SINGLE 类型：value 完全相同 → NOOP；否则 UPDATE（覆盖）。
    - SET 类型：merge 模式由调用方负责，本函数只判断新 item 是否已在集合中。
    """

    spec = get_spec(profile_key)
    if spec is None:
        return ProfileDecision("NOOP", profile_key, new_value, new_evidence, "未知 profile_key")
    if existing is None:
        return ProfileDecision("UPDATE", profile_key, new_value, new_evidence, "槽位首次写入")

    if spec.cardinality == "single":
        if _normalize_eq(new_value, existing["value"]):
            return ProfileDecision("NOOP", profile_key, new_value, new_evidence, "已有相同 Profile 值")
        return ProfileDecision("UPDATE", profile_key, new_value, new_evidence, "用户纠正/更新了 Profile 值")

    # SET 类型：调用方传入的 new_value 是单个 item 的 fields；按签名判断是否已在集合中。
    existing_items = existing["value"] if isinstance(existing["value"], list) else [existing["value"]]
    new_sig = match_signature(profile_key, new_value)
    for item in existing_items:
        if not isinstance(item, dict):
            continue
        if match_signature(profile_key, item) == new_sig:
            return ProfileDecision("NOOP", profile_key, new_value, new_evidence, "集合中已有同签名 item")
    return ProfileDecision("UPDATE", profile_key, new_value, new_evidence, "集合追加新 item")


def merge_set_item(
    profile_key: str,
    new_value: dict[str, Any],
    existing_value: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Merge a new item into a SET profile slot, returning the new stored value list."""

    items: list[dict[str, Any]] = []
    if existing_value is None:
        pass
    elif isinstance(existing_value, list):
        items = [dict(x) for x in existing_value if isinstance(x, dict)]
    elif isinstance(existing_value, dict):
        # 旧数据可能误存为单个 dict；兼容为单元素列表。
        items = [existing_value]
    items.append(dict(new_value))
    return items


# ---------------------------------------------------------------------------
# Event dedup (90% program + 10% LLM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExistingEvent:
    event_id: str
    fields: dict[str, Any]
    evidence: list[str]


def decide_event_add(
    event_type: str,
    new_fields: dict[str, Any],
    new_evidence: list[str],
    existing_events: list[ExistingEvent],
    llm_judge: "SameDifferentJudge | None" = None,
) -> EventDecision:
    """Decide whether to append a new event.

    1. Exact match (fields identical) → NOOP.
    2. Signature differs from all existing → ADD.
    3. Signature same but fields differ → LLM judges same/different.
       - same → NOOP
       - different → ADD
    """

    new_sig = match_signature(event_type, new_fields)
    new_fp = value_fingerprint(event_type, new_fields)

    same_signature_candidates: list[ExistingEvent] = []
    for existing in existing_events:
        if value_fingerprint(event_type, existing.fields) == new_fp:
            return EventDecision(
                "NOOP", event_type, new_fields, new_evidence,
                "已有完全相同 fields 的事件",
                matched_event_id=existing.event_id,
                used_llm=False,
            )
        if match_signature(event_type, existing.fields) == new_sig:
            same_signature_candidates.append(existing)

    if not same_signature_candidates:
        return EventDecision(
            "ADD", event_type, new_fields, new_evidence,
            "签名与所有现有事件不同，判定为新事件",
            used_llm=False,
        )

    # 签名相同但 fields 不同 → 交由 LLM 二分类。
    if llm_judge is None:
        # 未注入 judge 时，保守起见按 ADD 处理（避免漏写）。
        return EventDecision(
            "ADD", event_type, new_fields, new_evidence,
            "签名相同但 fields 不同，未注入 LLM judge，保守追加",
            used_llm=False,
        )

    judgment = llm_judge.judge(event_type, new_fields, new_evidence, same_signature_candidates)
    if judgment == "same":
        return EventDecision(
            "NOOP", event_type, new_fields, new_evidence,
            f"LLM 判定与已有事件 {same_signature_candidates[0].event_id} 同事件重复叙述",
            matched_event_id=same_signature_candidates[0].event_id,
            used_llm=True,
        )
    return EventDecision(
        "ADD", event_type, new_fields, new_evidence,
        "LLM 判定签名相同但语义不同，作为新事件追加",
        used_llm=True,
    )


def _normalize_eq(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two dicts after re-normalizing their enum values."""

    if not a and not b:
        return True
    return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# LLM same/different judge (only used when signature matches but fields differ)
# ---------------------------------------------------------------------------

class SameDifferentJudge:
    """LLM binary classifier: 'same' or 'different'.

    只做二分类，不做 ADD/UPDATE/DELETE 决策。决策由程序根据返回值执行。
    设计动机：LLM 擅长语义比较，但易在多分类中幻觉；强制二分类可显著降低误判率。
    """

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
        # Last-call diagnostics, exposed for pipeline-level instrumentation.
        self.last_usage: dict[str, int] = {}
        self.last_latency_ms: int = 0

    def _client_obj(self) -> Any:
        if self._client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def judge(
        self,
        event_type: str,
        new_fields: dict[str, Any],
        new_evidence: list[str],
        candidates: list[ExistingEvent],
    ) -> str:
        from time import perf_counter

        spec = get_spec(event_type)
        match_fields = spec.match_fields if spec else ()
        compact = [
            {
                "event_id": c.event_id,
                "fields": c.fields,
                "evidence": c.evidence,
            }
            for c in candidates[:5]  # 限制上下文长度
        ]

        system_prompt = (
            "你是事件去重判断 Agent。新事件和已有候选项属于同一 event_type 且 match_fields 签名相同。\n"
            "你只输出一个二分类结果：\n"
            "- same：新事件和候选项描述的是同一次经历或同一事实的不同叙述，应去重跳过；\n"
            "- different：新事件是一次新的、独立的事件或事实，应追加。\n"
            "判断依据：核心实体、时间、地点、动作、是否同一对象同一过程。\n"
            "禁止输出 ADD/UPDATE/DELETE；禁止返回理由；只输出 JSON：{\"verdict\":\"same|different\"}"
        )

        user_prompt = json.dumps({
            "event_type": event_type,
            "match_fields": list(match_fields),
            "new_fields": new_fields,
            "new_evidence": new_evidence,
            "candidates": compact,
        }, ensure_ascii=False, separators=(",", ":"))

        started = perf_counter()
        response = self._client_obj().chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        self.last_latency_ms = round((perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        } if usage else {}
        payload = json.loads(response.choices[0].message.content or "{}")
        verdict = str(payload.get("verdict", "")).strip().lower()
        return verdict if verdict in {"same", "different"} else "different"
