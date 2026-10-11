"""用户记忆生命周期管理。

3 个核心机制：
1. 情绪 TTL：新会话重置情绪为平静 + 历史产品经历打解决标签
2. Event→Profile 聚合：同成分反复过敏 → 自动写入 Profile 成分约束
3. 冲突检测：SINGLE Profile 枚举值变化 → 记录冲突供 UI 标注
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profile_event_schema import (
    PROFILE_SPEC_MAP,
    EVENT_SPEC_MAP,
    is_profile_key,
    is_event_key,
    normalize_value,
    validate_value,
)
from profile_event_store import ProfileEventStore


# ---------------------------------------------------------------------------
# 1. 情绪 TTL + 解决标签
# ---------------------------------------------------------------------------

@dataclass
class TTLResult:
    sentiment_reset: bool       # 是否重置了情绪
    events_tagged: int          # 标记了多少条历史事件
    resolution_status: str      # 当前会话的解决状态


def apply_sentiment_ttl(
    store: ProfileEventStore,
    user_id: str,
    conversation_id: str,
    is_new_conversation: bool,
) -> TTLResult:
    """新会话开始时：
    - 情绪重置为平静（旧 sentiment.current 不注入上下文）
    - 检查上一会话是否已解决 → 给历史事件打标签

    解决状态判定规则：
    - 上一会话有 service.user_resolution(resolution="resolved") → resolved
    - 上一会话有 dissatisfaction.escalation_intent → unresolved
    - 都没有 → unresolved（保守策略，保证不犯大错）
    """
    events_tagged = 0
    resolution_status = "unresolved"

    if is_new_conversation:
        # 查找上一会话的解决状态
        all_events = store.get_events(user_id)
        for event in reversed(all_events):
            etype = event.get("event_type", "")
            if etype == "service.user_resolution":
                fields = event.get("fields", {})
                if fields.get("resolution") == "resolved":
                    resolution_status = "resolved"
                break
            if etype == "dissatisfaction.escalation_intent":
                resolution_status = "unresolved"
                break

        # 给上一会话的事件批量打标签
        for event in all_events:
            conv_id = event.get("conversation_id")
            if conv_id and conv_id != conversation_id:
                if not event.get("resolution_status"):
                    store.tag_event_resolution(event["event_id"], resolution_status)
                    events_tagged += 1

    return TTLResult(
        sentiment_reset=is_new_conversation,
        events_tagged=events_tagged,
        resolution_status=resolution_status,
    )


def get_emotional_weight(event: dict[str, Any]) -> str:
    """根据解决状态返回情绪权重，供上下文注入时参考。

    - resolved → "低"（问题已解决，用户大概率平静）
    - unresolved → "高"（问题未解决，用户可能仍然不满）
    - 未标记 → "中"（默认）
    """
    status = event.get("resolution_status")
    if status == "resolved":
        return "低"
    if status == "unresolved":
        return "高"
    return "中"


# ---------------------------------------------------------------------------
# 2. Event → Profile 聚合
# ---------------------------------------------------------------------------

@dataclass
class AggregationResult:
    new_profile_entries: list[dict[str, Any]]  # 自动生成的 Profile 候选项
    trigger: str  # "ingredient_allergy" | "product_goal" | "recurring_issue"


def aggregate_events_to_profile(
    store: ProfileEventStore,
    user_id: str,
) -> AggregationResult:
    """扫描用户全部 Event，检测可聚合模式，生成 Profile 候选项。

    规则（纯程序，不需要 LLM）：
    1. 同一成分出现 ≥2 次 experience.product_reaction → ingredient_constraint(allergy)
    2. 同一 goal 出现 ≥3 次 need.information → need.product_goal
    3. 同一产品出现 ≥2 次 concern.observed_problem → 标记为 recurring（仅返回建议）
    """
    events = store.get_events(user_id)
    existing_profile = store.get_all_profile(user_id)
    candidates: list[dict[str, Any]] = []

    # 规则 1：成分过敏聚合
    ingredient_reactions: dict[str, int] = {}
    for event in events:
        if event.get("event_type") == "experience.product_reaction":
            fields = event.get("fields", {})
            ingredient = fields.get("ingredient") or fields.get("product")
            if ingredient:
                ingredient_reactions[ingredient] = ingredient_reactions.get(ingredient, 0) + 1

    existing_constraints = set()
    constraint_entry = existing_profile.get("body.ingredient_constraint", {})
    if constraint_entry:
        # 兼容两种存储格式：SET 类型可能存为 list 或 {"items": [...]} dict
        raw_value = constraint_entry.get("value", {})
        if isinstance(raw_value, dict):
            items = raw_value.get("items", [])
        elif isinstance(raw_value, list):
            items = raw_value
        else:
            items = []
        for item in items:
            if isinstance(item, dict):
                existing_constraints.add(item.get("ingredient"))

    for ingredient, count in ingredient_reactions.items():
        if count >= 2 and ingredient not in existing_constraints:
            candidates.append({
                "profile_key": "body.ingredient_constraint",
                "value": {"ingredient": ingredient, "constraint_type": "allergy"},
                "trigger": "ingredient_allergy",
                "evidence_count": count,
            })

    # 规则 2：产品目标聚合
    goal_mentions: dict[str, int] = {}
    for event in events:
        if event.get("event_type") == "need.information":
            fields = event.get("fields", {})
            topic = fields.get("topic") or fields.get("information_type")
            if topic:
                goal_mentions[topic] = goal_mentions.get(topic, 0) + 1

    existing_goals = set()
    goal_entry = existing_profile.get("need.product_goal", {})
    if goal_entry:
        for item in goal_entry.get("value", {}).get("items", []):
            existing_goals.add(item.get("goal"))

    for goal, count in goal_mentions.items():
        if count >= 3 and goal not in existing_goals:
            candidates.append({
                "profile_key": "need.product_goal",
                "value": {"goal": goal},
                "trigger": "product_goal",
                "evidence_count": count,
            })

    trigger = candidates[0]["trigger"] if candidates else "none"
    return AggregationResult(new_profile_entries=candidates, trigger=trigger)


# ---------------------------------------------------------------------------
# 3. 冲突检测
# ---------------------------------------------------------------------------

@dataclass
class ConflictResult:
    has_conflict: bool
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    profile_key: str


def detect_profile_conflict(
    store: ProfileEventStore,
    user_id: str,
    profile_key: str,
    new_value: dict[str, Any],
) -> ConflictResult:
    """检测 SINGLE Profile 枚举值是否与旧值冲突。

    判定规则：
    - 取旧 Profile 的核心枚举字段（如 skin_type, skin_sensitivity）
    - 如果旧值和新值不同 → 记录冲突
    - SET 类型不检测（追加语义，不冲突）
    """
    spec = PROFILE_SPEC_MAP.get(profile_key)
    if spec is None or spec.memory_type != "SINGLE":
        return ConflictResult(False, None, None, profile_key)

    old_entry = store.get_profile(user_id, profile_key)
    if old_entry is None:
        return ConflictResult(False, None, None, profile_key)

    old_value = old_entry.get("value", {})

    # 比较核心字段值（用 required_fields 而非 match_fields，后者可能为空）
    cmp_keys = spec.required_fields or spec.match_fields
    old_core = {k: v for k, v in old_value.items() if k in cmp_keys}
    new_core = {k: v for k, v in new_value.items() if k in cmp_keys}

    if old_core and new_core and old_core != new_core:
        # 记录冲突
        store.record_conflict(
            user_id=user_id,
            profile_key=profile_key,
            old_value=old_value,
            new_value=new_value,
            old_evidence=old_entry.get("evidence", []),
            new_evidence=[],
        )
        # 归档旧值
        store.archive_profile(
            user_id=user_id,
            profile_key=profile_key,
            value=old_value,
            evidence=old_entry.get("evidence", []),
        )
        return ConflictResult(True, old_value, new_value, profile_key)

    return ConflictResult(False, old_value, new_value, profile_key)
