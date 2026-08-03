"""Choose ADD, UPDATE, DELETE or NOOP for validated structured facts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from evidence_pipeline import CandidateMemory
from memory_schema import get_memory_spec, match_signature

load_dotenv(Path(__file__).parent / ".env")


@dataclass(frozen=True)
class MemoryDecision:
    operation: str
    target_id: str | None
    reason: str


def memory_value(item: dict[str, Any]) -> dict[str, Any]:
    raw = (item.get("metadata") or {}).get("memory_value")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class MemoryDecisionAgent:
    """Use exact slot rules first and LLM comparison only when semantics matter."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        # Create the client lazily: SINGLE/TRANSIENT and exact matches need no LLM call.
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")

    def _client(self) -> Any:
        if self.client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            self.client = OpenAI(api_key=api_key, base_url=os.environ.get("MEM0_LLM_BASE_URL"))
        return self.client

    def decide(self, candidate: CandidateMemory, existing: list[dict[str, Any]]) -> MemoryDecision:
        spec = get_memory_spec(candidate.memory_key)
        if spec is None:
            raise ValueError(f"Unknown memory_key: {candidate.memory_key}")

        exact = next((item for item in existing if memory_value(item) == candidate.value), None)
        if exact and candidate.operation_hint != "DELETE":
            return MemoryDecision("NOOP", str(exact["id"]), "结构化事实完全相同")

        if not existing:
            operation = "NOOP" if candidate.operation_hint == "DELETE" else "ADD"
            return MemoryDecision(operation, None, "不存在可删除记忆" if operation == "NOOP" else "该槽位尚无记忆")

        # match_fields only narrows candidates; it never performs the mutation itself.
        signature = match_signature(candidate.memory_key, candidate.value)
        narrowed = [item for item in existing if match_signature(candidate.memory_key, memory_value(item)) == signature]
        comparison_set = narrowed or existing
        return self._llm_decide(candidate, comparison_set[:20])

    def _llm_decide(self, candidate: CandidateMemory, existing: list[dict[str, Any]]) -> MemoryDecision:
        compact = [{"id": str(x.get("id", "")), "value": memory_value(x)} for x in existing if x.get("id")]
        system_prompt = """你是结构化用户记忆的记忆决策Agent。候选和旧记忆属于同一 memory_key。

选择一个操作：
- ADD：新的独立事实或新经历。
- UPDATE：同一对象、事项或经历被纠正、补充或推进；必须返回旧 memory_id。
- DELETE：用户明确撤回某条旧事实且没有替代值；必须返回旧 memory_id。
- NOOP：语义已存在、证据不足或没有可删除目标。

不要因为主题相似就合并两次独立经历。match_fields 只用于缩小候选范围，最终按完整语义判断。
只输出 JSON：{"operation":"ADD|UPDATE|DELETE|NOOP","target_id":null,"reason":"一句话理由"}
"""

        user_prompt = f"""memory_key: {candidate.memory_key}
operation_hint: {candidate.operation_hint}
新事实: {json.dumps(candidate.value, ensure_ascii=False)}
新事实来源原文: {json.dumps(candidate.evidence, ensure_ascii=False)}
候选旧记忆: {json.dumps(compact, ensure_ascii=False)}"""
        response = self._client().chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        operation = str(payload.get("operation", "NOOP")).strip().upper()
        target_id = str(payload.get("target_id") or "").strip() or None
        valid_ids = {x["id"] for x in compact}
        if operation not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            operation = "NOOP"
        if operation in {"UPDATE", "DELETE"} and target_id not in valid_ids:
            return MemoryDecision("NOOP", None, "模型没有返回合法的目标 memory_id")
        if operation == "ADD":
            target_id = None
        return MemoryDecision(operation, target_id, str(payload.get("reason", "")).strip())


# Backward-compatible name for callers created before the two-agent workflow was formalized.
MemoryDecisionEngine = MemoryDecisionAgent
