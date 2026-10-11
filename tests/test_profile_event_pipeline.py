"""Unit tests for the Profile+Event dual-channel pipeline.

覆盖点：
- Schema：枚举归一化、必填校验、签名计算
- Profile 去重：SINGLE 完全相同 NOOP / 不同 UPDATE；SET 签名相同 NOOP / 不同 UPDATE
- Event 去重：精确匹配 NOOP / 签名不同 ADD / 签名相同 fields 不同 → LLM same/different
- Pipeline 端到端：Profile 写回、Event 追加、幂等跳过、缺少 evidence 拒绝
- LLM judge：FakeSameDifferentJudge 模拟 same/different 二分类
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from profile_event_dedup import (
    ExistingEvent,
    ProfileDecision,
    SameDifferentJudge,
    decide_event_add,
    decide_profile_update,
    merge_set_item,
)
from profile_event_pipeline import (
    IncrementalExtractor,
    ProfileEventPipeline,
    RawNewEvent,
    RawProfileUpdate,
)
from profile_event_schema import (
    PROFILE_KEYS,
    EVENT_KEYS,
    match_signature,
    normalize_value,
    validate_value,
)
from profile_event_store import ProfileEventStore


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_profile_and_event_partition_covers_all_keys():
    """Profile + Event 应该是 memory_schema 全部 24 个 key 的不相交并集。"""

    assert set(PROFILE_KEYS).isdisjoint(set(EVENT_KEYS))
    assert len(PROFILE_KEYS) + len(EVENT_KEYS) == 24


def test_normalize_maps_natural_language_to_enum_codes():
    normalized, warnings = normalize_value("body.skin_type", {"skin_type": "干性"})
    assert normalized == {"skin_type": "dry"}
    assert warnings == []

    normalized, _ = normalize_value(
        "body.skin_sensitivity", {"level": "敏感"}
    )
    assert normalized == {"level": "sensitive"}

    normalized, warnings = normalize_value(
        "body.ingredient_constraint",
        {"ingredient": "酒精", "constraint_type": "过敏"},
    )
    assert normalized == {"ingredient": "酒精", "constraint_type": "allergy"}
    assert warnings == []


def test_normalize_keeps_unknown_values_with_warning():
    normalized, warnings = normalize_value(
        "body.skin_type", {"skin_type": "外星皮"}
    )
    assert normalized == {"skin_type": "外星皮"}
    assert any("未匹配到枚举码" in w for w in warnings)


def test_validate_value_rejects_missing_required_field():
    errors = validate_value("body.skin_type", {})
    assert errors == ["缺少必填字段：skin_type"]


def test_validate_value_rejects_unknown_enum_after_normalization():
    # Normalization keeps unknown alias as-is; validation flags it.
    errors = validate_value("body.skin_type", {"skin_type": "外星皮"})
    assert any("不在允许范围内" in e for e in errors)


def test_match_signature_is_lowercase_stable():
    sig_a = match_signature("experience.product_reaction", {"product": "B5 面膜", "reaction": "泛红", "body_area": "额头"})
    sig_b = match_signature("experience.product_reaction", {"product": "b5 面膜", "reaction": "泛红", "body_area": "额头"})
    assert sig_a == sig_b


# ---------------------------------------------------------------------------
# Profile dedup tests
# ---------------------------------------------------------------------------

def test_profile_single_first_write_updates():
    decision = decide_profile_update(
        "body.skin_type",
        {"skin_type": "dry"},
        ["我是干皮"],
        existing=None,
    )
    assert decision.operation == "UPDATE"
    assert "首次写入" in decision.reason


def test_profile_single_identical_value_is_noop():
    existing = {"value": {"skin_type": "dry"}, "evidence": ["我是干皮"]}
    decision = decide_profile_update(
        "body.skin_type",
        {"skin_type": "dry"},
        ["我是干皮"],
        existing=existing,
    )
    assert decision.operation == "NOOP"
    assert "已有相同 Profile 值" in decision.reason


def test_profile_single_correction_updates():
    existing = {"value": {"skin_type": "oily"}, "evidence": ["我是油皮"]}
    decision = decide_profile_update(
        "body.skin_type",
        {"skin_type": "dry"},
        ["说错了，其实是干皮"],
        existing=existing,
    )
    assert decision.operation == "UPDATE"
    assert "纠正" in decision.reason or "更新" in decision.reason


def test_profile_set_dedup_by_signature():
    # existing with one item already having same signature (ingredient 酒精).
    existing = {"value": [{"ingredient": "酒精", "constraint_type": "allergy"}], "evidence": []}
    decision = decide_profile_update(
        "body.ingredient_constraint",
        {"ingredient": "酒精", "constraint_type": "avoid"},
        ["我对酒精需要避用"],
        existing=existing,
    )
    # signature (ingredient) already in set → NOOP
    assert decision.operation == "NOOP"
    assert "同签名" in decision.reason


def test_profile_set_merges_new_item():
    existing_value = [{"ingredient": "酒精", "constraint_type": "allergy"}]
    merged = merge_set_item(
        "body.ingredient_constraint",
        {"ingredient": "香精", "constraint_type": "avoid"},
        existing_value,
    )
    assert len(merged) == 2
    assert merged[1]["ingredient"] == "香精"


# ---------------------------------------------------------------------------
# Event dedup tests
# ---------------------------------------------------------------------------

def test_event_exact_match_is_noop_without_llm():
    existing = [ExistingEvent(
        event_id="e1",
        fields={"product": "B5面膜", "reaction": "全脸泛红"},
        evidence=["用了B5面膜三小时全脸泛红"],
    )]
    decision = decide_event_add(
        event_type="experience.product_reaction",
        new_fields={"product": "B5面膜", "reaction": "全脸泛红"},
        new_evidence=["用了B5面膜三小时全脸泛红"],
        existing_events=existing,
        llm_judge=None,
    )
    assert decision.operation == "NOOP"
    assert decision.used_llm is False
    assert decision.matched_event_id == "e1"


def test_event_signature_differs_is_add_without_llm():
    existing = [ExistingEvent(
        event_id="e1",
        fields={"product": "B5面膜", "reaction": "全脸泛红"},
        evidence=["用了B5面膜三小时全脸泛红"],
    )]
    # Different product → different signature (product is in match_fields).
    decision = decide_event_add(
        event_type="experience.product_reaction",
        new_fields={"product": "VC精华", "reaction": "全脸泛红"},
        new_evidence=["用了VC精华后泛红"],
        existing_events=existing,
        llm_judge=None,
    )
    assert decision.operation == "ADD"
    assert decision.used_llm is False


class FakeJudge(SameDifferentJudge):
    """Override judge() with a stub result, no LLM call."""

    def __init__(self, verdict: str) -> None:
        super().__init__(client=None)
        self._verdict = verdict
        self.calls = 0

    def judge(self, event_type, new_fields, new_evidence, candidates):
        self.calls += 1
        return self._verdict


def test_event_signature_same_fields_differ_invokes_llm_same_verdict():
    existing = [ExistingEvent(
        event_id="e1",
        fields={"product": "B5面膜", "reaction": "额头泛红"},
        evidence=["额头泛红"],
    )]
    # Same signature (product=B5面膜) but different fields (reaction differs).
    judge = FakeJudge("same")
    decision = decide_event_add(
        event_type="experience.product_reaction",
        new_fields={"product": "B5面膜", "reaction": "脸颊泛红"},
        new_evidence=["脸颊也红了"],
        existing_events=existing,
        llm_judge=judge,
    )
    assert judge.calls == 1
    assert decision.operation == "NOOP"
    assert decision.used_llm is True
    assert decision.matched_event_id == "e1"


def test_event_signature_same_fields_differ_invokes_llm_different_verdict():
    existing = [ExistingEvent(
        event_id="e1",
        fields={"product": "B5面膜", "reaction": "额头泛红"},
        evidence=["额头泛红"],
    )]
    judge = FakeJudge("different")
    decision = decide_event_add(
        event_type="experience.product_reaction",
        new_fields={"product": "B5面膜", "reaction": "脸颊泛红"},
        new_evidence=["脸颊也红了"],
        existing_events=existing,
        llm_judge=judge,
    )
    assert judge.calls == 1
    assert decision.operation == "ADD"
    assert decision.used_llm is True


def test_event_signature_same_fields_differ_without_judge_falls_back_to_add():
    existing = [ExistingEvent(
        event_id="e1",
        fields={"product": "B5面膜", "reaction": "额头泛红"},
        evidence=["额头泛红"],
    )]
    decision = decide_event_add(
        event_type="experience.product_reaction",
        new_fields={"product": "B5面膜", "reaction": "脸颊泛红"},
        new_evidence=["脸颊也红了"],
        existing_events=existing,
        llm_judge=None,
    )
    # No judge → conservative ADD (avoid dropping real events).
    assert decision.operation == "ADD"
    assert decision.used_llm is False


# ---------------------------------------------------------------------------
# Pipeline end-to-end tests (with fakes)
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Records calls and returns a canned LLM payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeJudgeForPipeline(SameDifferentJudge):
    """Pipeline injects this; returns a preset verdict without LLM."""

    def __init__(self, verdict: str = "different") -> None:
        super().__init__(client=None)
        self._verdict = verdict
        self.calls = 0

    def judge(self, event_type, new_fields, new_evidence, candidates):
        self.calls += 1
        return self._verdict


@pytest.fixture()
def temp_store(tmp_path: Path) -> ProfileEventStore:
    db_path = tmp_path / "test_profile_event.sqlite3"
    return ProfileEventStore(db_path)


def _make_pipeline(
    store: ProfileEventStore,
    llm_payload: dict[str, Any],
    judge_verdict: str = "different",
) -> tuple[ProfileEventPipeline, FakeLLMClient, FakeJudgeForPipeline]:
    client = FakeLLMClient(llm_payload)
    extractor = IncrementalExtractor(client=client)
    judge = FakeJudgeForPipeline(verdict=judge_verdict)
    pipeline = ProfileEventPipeline(
        store=store,
        extractor=extractor,
        judge=judge,
        processed_turns_path=Path(store.path).parent / "test_processed.sqlite3",
    )
    return pipeline, client, judge


def test_pipeline_writes_profile_and_event_and_records_evidence(temp_store: ProfileEventStore):
    llm_payload = {
        "profile_updates": [
            {"profile_key": "body.skin_type", "fields": {"skin_type": "干性"}, "evidence_message_ids": [1]},
        ],
        "new_events": [
            {
                "event_type": "experience.product_reaction",
                "fields": {"product": "B5面膜", "reaction": "全脸泛红"},
                "description": "用户使用B5面膜后泛红",
                "evidence_message_ids": [2],
            },
        ],
    }
    pipeline, _, _ = _make_pipeline(temp_store, llm_payload)
    messages = [
        {"role": "user", "content": "我是干皮"},
        {"role": "user", "content": "用了B5面膜之后全脸泛红"},
    ]
    result = pipeline.process_turn(messages, user_id="u1", conversation_id="c1")

    assert result.idempotent_skip is False
    assert len(result.profile_decisions) == 1
    assert result.profile_decisions[0]["operation"] == "UPDATE"
    assert result.profile_decisions[0]["value"] == {"skin_type": "dry"}
    assert result.profile_decisions[0]["evidence"] == ["我是干皮"]
    assert len(result.event_decisions) == 1
    assert result.event_decisions[0]["operation"] == "ADD"
    assert result.event_decisions[0]["value"]["product"] == "B5面膜"

    profile = temp_store.get_profile("u1", "body.skin_type")
    assert profile["value"] == {"skin_type": "dry"}
    assert profile["evidence"] == ["我是干皮"]

    events = temp_store.get_events("u1")
    assert len(events) == 1
    assert events[0]["event_type"] == "experience.product_reaction"
    assert events[0]["evidence"] == ["用了B5面膜之后全脸泛红"]


def test_pipeline_idempotent_skip_on_same_messages(temp_store: ProfileEventStore):
    llm_payload = {
        "profile_updates": [
            {"profile_key": "body.skin_type", "fields": {"skin_type": "干性"}, "evidence_message_ids": [1]},
        ],
        "new_events": [],
    }
    pipeline, client, _ = _make_pipeline(temp_store, llm_payload)
    messages = [{"role": "user", "content": "我是干皮"}]
    first = pipeline.process_turn(messages, user_id="u2", conversation_id="c2")
    second = pipeline.process_turn(messages, user_id="u2", conversation_id="c2")

    assert first.idempotent_skip is False
    assert second.idempotent_skip is True
    assert second.profile_decisions == ()
    assert second.event_decisions == ()
    # Second call must not invoke the LLM again.
    assert len(client.calls) == 1


def test_pipeline_rejects_missing_evidence_message_ids(temp_store: ProfileEventStore):
    llm_payload = {
        "profile_updates": [
            {"profile_key": "body.skin_type", "fields": {"skin_type": "干性"}, "evidence_message_ids": []},
        ],
        "new_events": [],
    }
    pipeline, _, _ = _make_pipeline(temp_store, llm_payload)
    messages = [{"role": "user", "content": "我是干皮"}]
    result = pipeline.process_turn(messages, user_id="u3", conversation_id="c3")
    assert result.rejected
    assert any("缺少来源消息" in r["reasons"][0] for r in result.rejected)
    # Profile key is body.skin_type (Profile channel) → store should be empty.
    assert temp_store.get_profile("u3", "body.skin_type") is None


def test_pipeline_rejects_unknown_profile_key(temp_store: ProfileEventStore):
    llm_payload = {
        "profile_updates": [
            {"profile_key": "service.commitment", "fields": {"owner": "客服"}, "evidence_message_ids": [1]},
        ],
        "new_events": [],
    }
    pipeline, _, _ = _make_pipeline(temp_store, llm_payload)
    messages = [{"role": "user", "content": "客服承诺补发"}]
    result = pipeline.process_turn(messages, user_id="u4", conversation_id="c4")
    # service.commitment is an Event key, not a Profile key → rejected.
    assert result.rejected
    assert temp_store.get_profile("u4", "service.commitment") is None


def test_pipeline_event_dedup_skips_exact_match(temp_store: ProfileEventStore):
    # First turn writes the event.
    first_payload = {
        "profile_updates": [],
        "new_events": [
            {
                "event_type": "experience.product_reaction",
                "fields": {"product": "B5面膜", "reaction": "全脸泛红"},
                "evidence_message_ids": [1],
            },
        ],
    }
    pipeline, client, _ = _make_pipeline(temp_store, first_payload)
    messages = [{"role": "user", "content": "用了B5面膜后全脸泛红"}]
    first = pipeline.process_turn(messages, user_id="u5", conversation_id="c5")
    assert first.event_decisions[0]["operation"] == "ADD"

    # Second turn: different messages (different source_turn_hash) → pipeline runs again,
    # but Step 0 reads the existing event and dedup detects exact match → NOOP.
    client.payload = first_payload  # Same payload again.
    second_messages = [{"role": "user", "content": "再补一句，用了B5面膜后全脸泛红"}]
    second = pipeline.process_turn(second_messages, user_id="u5", conversation_id="c5b")
    assert second.event_decisions[0]["operation"] == "NOOP"
    assert second.event_decisions[0]["used_llm"] is False
    # Still one event in store.
    events = temp_store.get_events("u5")
    assert len(events) == 1


def test_pipeline_event_dedup_signature_match_invokes_judge(temp_store: ProfileEventStore):
    # Pre-seed an existing event.
    temp_store.add_event(
        user_id="u6",
        event_type="experience.product_reaction",
        fields={"product": "B5面膜", "reaction": "额头泛红"},
        evidence=["额头泛红"],
    )
    # New event has same signature (product=B5面膜) but different reaction.
    payload = {
        "profile_updates": [],
        "new_events": [
            {
                "event_type": "experience.product_reaction",
                "fields": {"product": "B5面膜", "reaction": "脸颊泛红"},
                "evidence_message_ids": [1],
            },
        ],
    }
    pipeline, _, judge = _make_pipeline(temp_store, payload, judge_verdict="same")
    messages = [{"role": "user", "content": "脸颊也泛红了"}]
    result = pipeline.process_turn(messages, user_id="u6", conversation_id="c6")
    assert judge.calls == 1
    assert result.event_decisions[0]["operation"] == "NOOP"
    assert result.event_decisions[0]["used_llm"] is True
    # No new event appended.
    assert len(temp_store.get_events("u6")) == 1


def test_pipeline_set_profile_merges_multiple_distinct_ingredients(temp_store: ProfileEventStore):
    # Turn 1: user is allergic to 酒精.
    payload_1 = {
        "profile_updates": [
            {"profile_key": "body.ingredient_constraint", "fields": {"ingredient": "酒精", "constraint_type": "过敏"}, "evidence_message_ids": [1]},
        ],
        "new_events": [],
    }
    pipeline, client, _ = _make_pipeline(temp_store, payload_1)
    pipeline.process_turn(
        [{"role": "user", "content": "我对酒精过敏"}],
        user_id="u7", conversation_id="c7",
    )

    # Turn 2: different messages → different source_turn_hash → pipeline runs again.
    client.payload = {
        "profile_updates": [
            {"profile_key": "body.ingredient_constraint", "fields": {"ingredient": "香精", "constraint_type": "避用"}, "evidence_message_ids": [1]},
        ],
        "new_events": [],
    }
    pipeline.process_turn(
        [{"role": "user", "content": "香精我也想避用"}],
        user_id="u7", conversation_id="c7b",
    )

    profile = temp_store.get_profile("u7", "body.ingredient_constraint")
    assert isinstance(profile["value"], list)
    assert len(profile["value"]) == 2
    ingredients = {item["ingredient"] for item in profile["value"]}
    assert ingredients == {"酒精", "香精"}


def test_pipeline_set_profile_skips_duplicate_signature(temp_store: ProfileEventStore):
    # Pre-seed 酒精 allergy.
    temp_store.upsert_profile(
        user_id="u8",
        profile_key="body.ingredient_constraint",
        value=[{"ingredient": "酒精", "constraint_type": "allergy"}],
        evidence=["我对酒精过敏"],
    )
    # New turn says user is allergic to 酒精 again (same signature).
    payload = {
        "profile_updates": [
            {"profile_key": "body.ingredient_constraint", "fields": {"ingredient": "酒精", "constraint_type": "过敏"}, "evidence_message_ids": [1]},
        ],
        "new_events": [],
    }
    pipeline, _, _ = _make_pipeline(temp_store, payload)
    result = pipeline.process_turn(
        [{"role": "user", "content": "我对酒精过敏"}],
        user_id="u8", conversation_id="c8",
    )
    assert result.profile_decisions[0]["operation"] == "NOOP"
    profile = temp_store.get_profile("u8", "body.ingredient_constraint")
    assert len(profile["value"]) == 1  # Did not append duplicate.
