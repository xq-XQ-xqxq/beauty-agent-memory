"""Closed, versioned schema for structured customer memories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class MemoryKeySpec:
    key: str
    dimension: str
    description: str
    memory_type: str
    scope: str
    required_fields: tuple[str, ...]
    match_fields: tuple[str, ...]
    enums: dict[str, tuple[str, ...]]

    @property
    def cardinality(self) -> str:  # Compatibility with the earlier prototype.
        return "single" if self.memory_type in {"SINGLE", "TRANSIENT"} else "multi"


def _spec(
    key: str,
    dimension: str,
    description: str,
    memory_type: str,
    scope: str,
    required: tuple[str, ...],
    match: tuple[str, ...] = (),
    **enums: tuple[str, ...],
) -> MemoryKeySpec:
    return MemoryKeySpec(key, dimension, description, memory_type, scope, required, match, enums)


MEMORY_KEY_SPECS = (
    _spec("body.skin_type", "身体状态", "用户明确自述的当前肤质", "SINGLE", "长期", ("skin_type",), skin_type=("dry", "oily", "combination", "normal", "unknown")),
    _spec("body.skin_sensitivity", "身体状态", "用户明确自述的皮肤敏感情况", "SINGLE", "长期", ("level",), level=("sensitive", "not_sensitive", "uncertain")),
    _spec("body.appearance_trait", "身体状态", "肤色、唇纹等外观特征", "SET", "长期", ("trait",), ("trait", "body_area")),
    _spec("body.baseline_condition", "身体状态", "长期或反复存在的皮肤、头皮或头发基础状态", "SET", "长期", ("condition",), ("condition", "body_area")),
    _spec("body.special_status", "身体状态", "孕期、哺乳期等用户明确自述的特殊状态", "SET", "长期", ("status",), ("status",), status=("pregnant", "breastfeeding", "preparing_pregnancy", "minor", "other")),
    _spec("body.ingredient_constraint", "身体状态", "过敏、避用、慎用或不喜欢的成分", "SET", "长期", ("ingredient", "constraint_type"), ("ingredient",), constraint_type=("allergy", "avoid", "caution", "dislike")),
    _spec("experience.product_reaction", "产品经历", "具体产品使用后的身体反应，不判断医学因果", "EPISODE", "长期", ("product", "reaction"), ("product", "body_area", "onset"), severity=("mild", "moderate", "severe", "unknown"), current_change=("worsened", "improved", "disappeared", "unchanged", "unknown")),
    _spec("experience.product_result", "产品经历", "用户实际感受到的产品效果", "EPISODE", "长期", ("product", "result"), ("product", "result", "occurred_at"), polarity=("positive", "negative", "mixed", "neutral")),
    _spec("preference.product_attribute", "偏好", "质地、香味、色号、妆效、配方等偏好", "SET", "长期", ("attribute", "preference", "value"), ("attribute", "context"), preference=("like", "dislike", "prefer", "avoid")),
    _spec("purchase.context", "购买背景", "自用、送礼、场合、预算与时间要求", "ACTIVE_ITEM", "当前事项", ("purpose",), ("purpose", "recipient", "occasion"), purpose=("self_use", "gift", "travel", "occasion", "stock_up", "other")),
    _spec("need.product_goal", "需求", "希望产品达到的效果或使用目标", "SET", "长期", ("goal",), ("goal", "body_area")),
    _spec("concern.observed_problem", "困扰", "用户已经观察到的商品、服务或身体问题", "ACTIVE_ITEM", "当前事项", ("subject", "problem"), ("subject", "problem")),
    _spec("concern.core_worry", "困扰", "用户担心发生的结果及原因", "ACTIVE_ITEM", "当前事项", ("subject", "worry"), ("subject", "worry")),
    _spec("need.decision_support", "需求", "用户需要客服帮助完成的选择", "ACTIVE_ITEM", "当前事项", ("decision",), ("decision", "options")),
    _spec("need.information", "需求", "用户希望确认的信息", "ACTIVE_ITEM", "当前事项", ("topic", "question"), ("topic", "subject")),
    _spec("need.service_action", "需求", "用户希望商家采取的服务动作", "ACTIVE_ITEM", "当前事项", ("action_type", "subject", "request"), ("action_type", "subject"), action_type=("refund", "return", "exchange", "reship", "investigate", "explain", "apology", "callback", "other")),
    _spec("sentiment.current", "情绪", "用户当前情绪，仅短期有效", "TRANSIENT", "短期", ("emotion", "intensity", "observed_at"), emotion=("angry", "anxious", "disappointed", "impatient", "sad", "calm", "satisfied", "other"), intensity=("low", "medium", "high")),
    _spec("dissatisfaction.reason", "不满", "用户明确表达的不满对象、原因与影响", "ACTIVE_ITEM", "当前事项", ("target", "reason"), ("target", "reason")),
    _spec("dissatisfaction.escalation_intent", "不满", "用户明确表达的投诉或升级意图", "ACTIVE_ITEM", "当前事项", ("intent", "target"), ("intent", "target"), intent=("complaint", "public_exposure", "platform_escalation", "regulatory_complaint", "legal_action", "leave", "other")),
    _spec("satisfaction.positive_feedback", "反馈", "用户对商品或服务的正面反馈", "EPISODE", "长期", ("target", "feedback"), ("target", "occurred_at")),
    _spec("service.evidence_submitted", "服务共识", "用户已经提交的图片、视频或说明材料", "ACTIVE_ITEM", "当前事项", ("evidence_type", "subject"), ("evidence_type", "subject")),
    _spec("service.plan_response", "服务共识", "用户对客服方案的接受、拒绝或附条件回应", "ACTIVE_ITEM", "当前事项", ("plan", "subject", "response"), ("plan", "subject"), response=("accepted", "rejected", "conditional", "considering")),
    _spec("service.commitment", "服务共识", "客服、用户或平台明确承诺的动作与时限", "ACTIVE_ITEM", "当前事项", ("owner", "action", "subject"), ("owner", "action", "subject"), owner=("support", "user", "merchant", "platform")),
    _spec("service.user_resolution", "服务共识", "用户明确确认的解决状态，不等同于工单状态", "ACTIVE_ITEM", "当前事项", ("subject", "resolution", "statement"), ("subject",), resolution=("resolved", "partially_resolved", "unresolved")),
)

MEMORY_KEYS = {spec.key: spec for spec in MEMORY_KEY_SPECS}
ALLOWED_OPERATION_HINTS = {"UPSERT", "DELETE"}


def get_memory_spec(memory_key: str) -> MemoryKeySpec | None:
    return MEMORY_KEYS.get(memory_key)


def validate_value(memory_key: str, value: Any) -> list[str]:
    spec = get_memory_spec(memory_key)
    if spec is None:
        return ["memory_key 不在固定 Schema 中"]
    if not isinstance(value, dict):
        return ["value 必须是 JSON 对象"]
    errors: list[str] = []
    for field in spec.required_fields:
        if value.get(field) in (None, "", []):
            errors.append(f"缺少必填字段：{field}")
    for field, allowed in spec.enums.items():
        actual = value.get(field)
        if actual not in (None, "") and actual not in allowed:
            errors.append(f"字段 {field} 的值不在允许范围内")
    return errors


def match_signature(memory_key: str, value: dict[str, Any]) -> tuple[str, ...]:
    spec = MEMORY_KEYS[memory_key]
    return tuple(str(value.get(field, "")).strip().lower() for field in spec.match_fields)


def format_memory_text(memory_key: str, value: dict[str, Any]) -> str:
    spec = MEMORY_KEYS[memory_key]
    body = "；".join(f"{key}={val}" for key, val in value.items() if val not in (None, "", []))
    return f"[{spec.dimension}][{memory_key}][{spec.scope}] {body}"


def schema_prompt() -> str:
    lines = []
    for spec in MEMORY_KEY_SPECS:
        fields = ", ".join(spec.required_fields)
        enums = json.dumps(spec.enums, ensure_ascii=False)
        lines.append(f"- {spec.key} | {spec.memory_type} | 必填:{fields} | 枚举:{enums} | {spec.description}")
    return "\n".join(lines)
