"""Profile + Event dual-channel schema (alternative to memory_schema.MEMORY_KEYS).

设计原则：
- Profile（可变画像，按 key 覆盖）+ Event（不可变事件流，按时间追加）双通道，而非单一记忆列表。
- LLM 只做抽取、不做 ADD/UPDATE/DELETE 决策，只输出半结构化 fields（自然语言值）。
- 程序负责枚举归一化、必填校验、去重和决策执行。

字段定义沿用 memory_schema.MEMORY_KEY_SPECS 的 24 个 key 以保证语义对齐，
但根据 scope / memory_type 分流到 Profile 或 Event 通道。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memory_schema import MEMORY_KEY_SPECS, MemoryKeySpec


# ---------------------------------------------------------------------------
# Profile vs Event classification
# ---------------------------------------------------------------------------

# Profile 通道：长期稳定属性，可被新事实覆盖更新。
# SINGLE 类型 → 单值覆盖；SET 类型 → 集合追加/移除。
PROFILE_KEYS: tuple[str, ...] = (
    "body.skin_type",
    "body.skin_sensitivity",
    "body.appearance_trait",
    "body.baseline_condition",
    "body.special_status",
    "body.ingredient_constraint",
    "preference.product_attribute",
    "need.product_goal",
)

# Event 通道：不可变事件流，按 timestamp 追加。
# EPISODE / ACTIVE_ITEM / TRANSIENT 全部进入事件流；TRANSIENT 通过 occurred_at 自然过期。
EVENT_KEYS: tuple[str, ...] = (
    "experience.product_reaction",
    "experience.product_result",
    "purchase.context",
    "concern.observed_problem",
    "concern.core_worry",
    "need.decision_support",
    "need.information",
    "need.service_action",
    "sentiment.current",
    "dissatisfaction.reason",
    "dissatisfaction.escalation_intent",
    "satisfaction.positive_feedback",
    "service.evidence_submitted",
    "service.plan_response",
    "service.commitment",
    "service.user_resolution",
)


def _get_spec(key: str) -> MemoryKeySpec:
    """Resolve a key to its MemoryKeySpec via MEMORY_KEY_SPECS."""

    for spec in MEMORY_KEY_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown memory_key: {key}")


PROFILE_SPEC_MAP: dict[str, MemoryKeySpec] = {k: _get_spec(k) for k in PROFILE_KEYS}
EVENT_SPEC_MAP: dict[str, MemoryKeySpec] = {k: _get_spec(k) for k in EVENT_KEYS}


def is_profile_key(memory_key: str) -> bool:
    return memory_key in PROFILE_SPEC_MAP


def is_event_key(memory_key: str) -> bool:
    return memory_key in EVENT_SPEC_MAP


def get_spec(memory_key: str) -> MemoryKeySpec | None:
    return PROFILE_SPEC_MAP.get(memory_key) or EVENT_SPEC_MAP.get(memory_key)


# ---------------------------------------------------------------------------
# Natural-language → enum normalization
# ---------------------------------------------------------------------------

# LLM 输出半结构化值时允许使用中文自然语言，程序统一归一化到枚举码。
# 设计动机：LLM 擅长语义理解，但易在枚举码上幻觉；让 LLM 输出自然语义，
# 由程序做确定性映射，兼顾灵活性与可控性。
ENUM_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "skin_type": {
        "dry": ("干性", "干皮", "干燥", "干", "dry"),
        "oily": ("油性", "油皮", "出油", "油", "oily"),
        "combination": ("混合性", "混合", "混油", "混干", "混合性敏感肌", "混敏皮", "combination", "combo"),
        "normal": ("中性", "正常", "普通", "normal"),
        "unknown": ("不清楚", "未知", "不确定", "unknown"),
    },
    "level": {
        "sensitive": ("敏感", "敏感肌", "易敏感", "sensitive"),
        "not_sensitive": ("不敏感", "耐受", "not_sensitive"),
        "uncertain": ("不清楚", "未知", "不确定", "uncertain"),
    },
    "status": {
        "pregnant": ("孕期", "怀孕", "孕妇", "pregnant"),
        "breastfeeding": ("哺乳期", "哺乳", "母乳", "breastfeeding"),
        "preparing_pregnancy": ("备孕", "preparing_pregnancy"),
        "minor": ("未成年", "未成年人", "小孩", "minor"),
        "other": ("其他", "other"),
    },
    "constraint_type": {
        "allergy": ("过敏", "过敏原", "allergy", "过敏成分"),
        "avoid": ("避用", "忌用", "不用", "avoid", "不使用"),
        "caution": ("慎用", "caution"),
        "dislike": ("不喜欢", "反感", "dislike"),
    },
    "severity": {
        "mild": ("轻度", "轻微", "mild"),
        "moderate": ("中度", "moderate"),
        "severe": ("重度", "严重", "severe"),
        "unknown": ("不清楚", "未知", "unknown"),
    },
    "current_change": {
        "worsened": ("加重", "恶化", "worsened"),
        "improved": ("好转", "改善", "improved"),
        "disappeared": ("消失", "消退", "disappeared"),
        "unchanged": ("不变", "无变化", "unchanged"),
        "unknown": ("不清楚", "未知", "unknown"),
    },
    "polarity": {
        "positive": ("正面", "好", "满意", "positive"),
        "negative": ("负面", "不好", "不满意", "negative"),
        "mixed": ("混合", "褒贬不一", "mixed"),
        "neutral": ("中性", "一般", "neutral"),
    },
    "preference": {
        "like": ("喜欢", "偏好", "like"),
        "dislike": ("不喜欢", "反感", "dislike"),
        "prefer": ("更倾向", "prefer", "倾向"),
        "avoid": ("避用", "不选", "avoid"),
    },
    "purpose": {
        "self_use": ("自用", "自己用", "自己使用", "self_use"),
        "gift": ("送礼", "送人", "赠礼", "送朋友", "送家人", "gift"),
        "travel": ("出差", "旅行", "出行", "旅游", "travel"),
        "occasion": ("场合", "活动", "婚礼", "聚会", "occasion"),
        "stock_up": ("囤货", "备货", "囤", "stock_up"),
        "other": ("其他", "other"),
    },
    "action_type": {
        "refund": ("退款", "退钱", "退", "退货退款", "refund"),
        "return": ("退货", "return"),
        "exchange": ("换货", "换", "exchange"),
        "reship": ("补发", "重发", "reship"),
        "investigate": ("核实", "调查", "investigate"),
        "explain": ("解释", "说明", "explain"),
        "apology": ("道歉", "致歉", "apology"),
        "callback": ("回访", "回电", "callback"),
        "other": ("其他", "其他操作", "other"),
    },
    "emotion": {
        "angry": ("愤怒", "生气", "气愤", "发火", "angry"),
        "anxious": ("焦虑", "着急", "焦急", "急", "焦虑不安", "anxious"),
        "disappointed": ("失望", "遗憾", "disappointed"),
        "impatient": ("不耐烦", "急躁", "烦躁", "impatient"),
        "sad": ("难过", "伤心", "沮丧", "sad"),
        "calm": ("平静", "冷静", "稳定", "calm"),
        "satisfied": ("满意", "开心", "satisfied"),
        "other": ("其他", "其他情绪", "other"),
    },
    "intensity": {
        "low": ("低", "轻微", "轻度", "low"),
        "medium": ("中", "中等", "中度", "medium"),
        "high": ("高", "强烈", "严重", "high"),
    },
    "intent": {
        "complaint": ("投诉", "complaint"),
        "public_exposure": ("曝光", "公开", "public_exposure"),
        "platform_escalation": ("平台升级", "platform_escalation"),
        "regulatory_complaint": ("12315", "监管", "regulatory_complaint"),
        "legal_action": ("法律", "起诉", "legal_action"),
        "leave": ("流失", "不再购买", "表达购买意愿下降", "leave"),
        "other": ("其他", "其他意图", "other"),
    },
    "response": {
        "accepted": ("接受", "同意", "accepted"),
        "rejected": ("拒绝", "不接受", "rejected"),
        "conditional": ("有条件", "附条件", "conditional"),
        "considering": ("考虑中", "considering"),
    },
    "owner": {
        "support": ("客服", "support"),
        "user": ("用户", "买家", "user"),
        "merchant": ("商家", "merchant"),
        "platform": ("平台", "platform"),
    },
    "resolution": {
        "resolved": ("已解决", "解决", "resolved"),
        "partially_resolved": ("部分解决", "partially_resolved"),
        "unresolved": ("未解决", "unresolved"),
    },
}


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _build_alias_index() -> dict[str, dict[str, str]]:
    """Flatten ENUM_ALIASES into {field_name: {alias: canonical_code}}."""

    index: dict[str, dict[str, str]] = {}
    for field, mapping in ENUM_ALIASES.items():
        flat: dict[str, str] = {}
        for canonical, aliases in mapping.items():
            for alias in aliases:
                flat[_norm(alias)] = canonical
            flat[_norm(canonical)] = canonical
        index[field] = flat
    return index


_ALIAS_INDEX = _build_alias_index()


def normalize_value(memory_key: str, value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize natural-language enum values to canonical codes.

    Returns (normalized_value, warnings).  When exact alias lookup fails, tries
    a substring-based fallback: if any known alias is contained in the input,
    the matching canonical code is used.  Unknown values are kept as-is so that
    downstream review / manual inspection can still see them.
    """

    spec = get_spec(memory_key)
    if spec is None or not isinstance(value, dict):
        return dict(value), []

    normalized: dict[str, Any] = {}
    warnings: list[str] = []
    for field, raw in value.items():
        if field not in spec.enums or raw in (None, ""):
            normalized[field] = raw
            continue
        canonical = _ALIAS_INDEX.get(field, {}).get(_norm(raw))
        if canonical:
            normalized[field] = canonical
            continue
        # Fuzzy fallback: check if any known alias is a substring of the input
        # (or vice versa).  This catches composite LLM outputs like "混合性敏感肌".
        canonical = _fuzzy_match(field, str(raw))
        if canonical:
            normalized[field] = canonical
        else:
            normalized[field] = raw
            warnings.append(f"字段 {field} 的值 '{raw}' 未匹配到枚举码")
    # Pass through non-enum fields unchanged.
    for field, raw in value.items():
        if field not in spec.enums:
            normalized.setdefault(field, raw)
    return normalized, warnings


def _fuzzy_match(field: str, raw: str) -> str | None:
    """Substring-based enum matching for composite LLM outputs.

    Strategy:
    1. Input contains a known alias → match to that alias's canonical code.
       Example: "混合性敏感肌" contains "混合性" → combination.
    2. A known alias contains the input → match to that alias's canonical code.
       Example: "油" is contained in "油皮" → oily (ambiguous, but prefers
       longer aliases first).
    """

    raw_lower = _norm(raw)
    if not raw_lower:
        return None
    field_index = _ALIAS_INDEX.get(field, {})
    # Build (alias, canonical) pairs sorted by alias length (longest first)
    # so that "混合性敏感肌" matches before "敏感".
    pairs = sorted(
        ((alias, canonical) for alias, canonical in field_index.items()),
        key=lambda p: len(p[0]),
        reverse=True,
    )
    for alias, canonical in pairs:
        if alias and (alias in raw_lower or raw_lower in alias):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Required-field validation (post-normalization)
# ---------------------------------------------------------------------------

def validate_value(memory_key: str, value: dict[str, Any]) -> list[str]:
    """Validate required fields and enum membership after normalization."""

    spec = get_spec(memory_key)
    if spec is None:
        return [f"memory_key 不在 Profile/Event Schema 中：{memory_key}"]
    if not isinstance(value, dict):
        return ["value 必须是 JSON 对象"]

    errors: list[str] = []
    for field in spec.required_fields:
        if value.get(field) in (None, "", []):
            errors.append(f"缺少必填字段：{field}")
    for field, allowed in spec.enums.items():
        actual = value.get(field)
        if actual not in (None, "") and actual not in allowed:
            errors.append(f"字段 {field} 的值 '{actual}' 不在允许范围内：{list(allowed)}")
    return errors


# ---------------------------------------------------------------------------
# Dedup signature
# ---------------------------------------------------------------------------

def match_signature(memory_key: str, value: dict[str, Any]) -> tuple[str, ...]:
    """Build a dedup signature from match_fields (post-normalization values)."""

    spec = get_spec(memory_key)
    if spec is None:
        return ()
    return tuple(_norm(value.get(field, "")) for field in spec.match_fields)


def value_fingerprint(memory_key: str, value: dict[str, Any]) -> str:
    """Stable JSON fingerprint used for exact-match dedup."""

    spec = get_spec(memory_key)
    payload = {field: value.get(field) for field in spec.required_fields} if spec else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def profile_prompt() -> str:
    lines = ["Profile 通道（长期可变画像，可覆盖更新）："]
    for spec in PROFILE_SPEC_MAP.values():
        fields = ", ".join(spec.required_fields)
        enums = json.dumps(spec.enums, ensure_ascii=False)
        lines.append(f"- {spec.key} | {spec.memory_type} | 必填:{fields} | 枚举:{enums} | {spec.description}")
    return "\n".join(lines)


def event_prompt() -> str:
    lines = ["Event 通道（不可变事件流，按时间追加）："]
    for spec in EVENT_SPEC_MAP.values():
        fields = ", ".join(spec.required_fields)
        enums = json.dumps(spec.enums, ensure_ascii=False)
        lines.append(f"- {spec.key} | {spec.memory_type} | 必填:{fields} | 枚举:{enums} | {spec.description}")
    return "\n".join(lines)


def schema_prompt() -> str:
    return profile_prompt() + "\n\n" + event_prompt()
