"""Evidence-grounded extraction into the closed structured-memory schema."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from memory_schema import ALLOWED_OPERATION_HINTS, format_memory_text, schema_prompt, validate_value

load_dotenv(Path(__file__).parent / ".env")

SENSITIVE_PATTERNS = [
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{15,19}(?!\d)"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]
EVIDENCE_LINE_PREFIX = re.compile(r"^\[\d+]\[(?:user|assistant)]\s*")


@dataclass(frozen=True)
class CandidateMemory:
    memory_key: str
    value: dict[str, Any]
    evidence: tuple[str, ...]
    operation_hint: str = "UPSERT"
    evidence_message_ids: tuple[int, ...] = ()

    @property
    def memory(self) -> str:
        return format_memory_text(self.memory_key, self.value)


@dataclass(frozen=True)
class ValidationResult:
    accepted: tuple[CandidateMemory, ...]
    rejected: tuple[dict[str, Any], ...]


def canonical_turn(messages: list[dict[str, Any]]) -> str:
    normalized = [{"role": str(x.get("role", "")), "content": str(x.get("content", "")).strip()} for x in messages]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def turn_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_turn(messages).encode("utf-8")).hexdigest()


class FactExtractionAgent:
    """First mapping: extract evidence-backed facts; never decide DB mutations."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")

    def extract(self, messages: list[dict[str, Any]]) -> list[CandidateMemory]:
        numbered = "\n".join(f"[{i}][{x.get('role', '')}] {x.get('content', '')}" for i, x in enumerate(messages, 1))
        system_prompt = f"""你是美妆电商客服用户记忆的事实提取Agent。你只提取事实，不决定数据库操作。

只能使用以下固定 memory_key，严禁创建新 key：
{schema_prompt()}

规则：
1. 尽量提取对后续人工服务有用且能被逐字原文支持的事实；问候、模板话术、订单金额、编号、物流和工单状态不提取。
2. value 必须是符合对应 key 的 JSON 对象。常见枚举之外的细节放 detail，不得捏造字段值。
3. evidence_message_ids 必须是1至3个消息编号。不要复制或改写原文，程序会根据编号读取原始消息。
4. 用户明确撤回旧事实且没有给替代值时 operation_hint=DELETE；其他情况均为 UPSERT。纠正时只提取新事实。
5. “孕妇能用吗”不代表用户怀孕；“适合干皮吗”不代表用户是干皮；客服推测不能成为用户事实。
6. 产品使用后反应只记录时间顺序和用户表达，不作医学诊断或因果判断。
7. 未解决事项、风险、优先级、等待时长和动态标签均不提取。
8. 找不到合法 memory_key 时直接过滤。
9. 只输出 JSON：{{"candidates":[{{"memory_key":"...","value":{{}},"operation_hint":"UPSERT|DELETE","evidence_message_ids":[1]}}]}}

"""
        user_prompt = f"请从以下编号对话中提取结构化事实：\n\n{numbered}"
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        items = payload if isinstance(payload, list) else payload.get("candidates", [])
        output: list[CandidateMemory] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            raw_ids = item.get("evidence_message_ids", [])
            if isinstance(raw_ids, (int, str)):
                raw_ids = [raw_ids]
            message_ids: list[int] = []
            for raw_id in raw_ids if isinstance(raw_ids, list) else []:
                try:
                    message_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if 1 <= message_id <= len(messages) and message_id not in message_ids:
                    message_ids.append(message_id)
            # Evidence text is copied by the program, never by the model.
            evidence = tuple(str(messages[i - 1].get("content", "")) for i in message_ids)
            value = item.get("value", {})
            output.append(CandidateMemory(
                memory_key=str(item.get("memory_key", "")).strip(),
                value=value if isinstance(value, dict) else {},
                evidence=tuple(text for text in evidence if text),
                operation_hint=str(item.get("operation_hint", "UPSERT")).strip().upper(),
                evidence_message_ids=tuple(message_ids),
            ))
        return output


# Backward-compatible name used by the existing Streamlit service and tests.
EvidenceExtractor = FactExtractionAgent


def validate_candidates(candidates: list[CandidateMemory], messages: list[dict[str, Any]]) -> ValidationResult:
    source_text = "\n".join(str(x.get("content", "")) for x in messages)
    accepted: list[CandidateMemory] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        reasons = validate_value(candidate.memory_key, candidate.value)
        if candidate.operation_hint not in ALLOWED_OPERATION_HINTS:
            reasons.append("operation_hint 不合法")
        if not candidate.evidence_message_ids:
            reasons.append("缺少来源消息编号")
        elif any(not 1 <= message_id <= len(messages) for message_id in candidate.evidence_message_ids):
            reasons.append("来源消息编号不存在")
        else:
            expected = tuple(str(messages[i - 1].get("content", "")) for i in candidate.evidence_message_ids)
            if candidate.evidence != expected:
                reasons.append("证据内容不是程序读取的原始消息")
        serialized = json.dumps(candidate.value, ensure_ascii=False, sort_keys=True)
        if any(pattern.search(serialized) for pattern in SENSITIVE_PATTERNS):
            reasons.append("疑似包含敏感标识符")
        fingerprint = f"{candidate.memory_key}:{candidate.operation_hint}:{serialized}"
        if fingerprint in seen:
            reasons.append("本批次重复记忆")
        if reasons:
            rejected.append({"memory_key": candidate.memory_key, "value": candidate.value, "evidence": list(candidate.evidence), "reasons": reasons})
        else:
            seen.add(fingerprint)
            accepted.append(candidate)
    return ValidationResult(tuple(accepted), tuple(rejected))
