import json
from types import SimpleNamespace

from evidence_pipeline import CandidateMemory, EvidenceExtractor, validate_candidates
from memory_decision import MemoryDecisionEngine
from memory_service import CustomerMemoryService, MemoryContext
from memory_schema import MEMORY_KEYS, validate_value


class FakeMemory:
    def __init__(self):
        self.items = []
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.search_calls = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        item = {"id": f"m{len(self.items) + 1}", "user_id": kwargs["user_id"], "memory": kwargs["messages"][0]["content"], "metadata": kwargs["metadata"]}
        self.items.append(item)
        return {"results": [item]}

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        for item in self.items:
            if item["id"] == kwargs["memory_id"]:
                item["memory"] = kwargs["text"]
                item["metadata"] = kwargs["metadata"]
        return {"results": []}

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        self.items = [x for x in self.items if x["id"] != kwargs["memory_id"]]

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        filters = filters or {}
        result = []
        for item in self.items:
            payload = {**item.get("metadata", {}), "user_id": item.get("user_id", "u1")}
            if all(payload.get(k) == v for k, v in filters.items()):
                result.append(item)
        return {"results": result[:top_k]}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": self.items[: kwargs.get("top_k", 5)]}


class FakeExtractor:
    def __init__(self, candidates):
        self.candidates = candidates

    def extract(self, messages):
        return self.candidates


class FakeProcessedTurnStore:
    def __init__(self):
        self.items = {}

    def get(self, user_id, source_turn_hash):
        return self.items.get((user_id, source_turn_hash))

    def record(self, user_id, source_turn_hash, result):
        self.items[(user_id, source_turn_hash)] = result


class DecisionClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def context():
    return MemoryContext("u1", "c1", "event-provenance-only", "售后", "咨询", "2026-08-02T00:00:00")


def candidate(key, value, quote):
    return CandidateMemory(key, value, (quote,), evidence_message_ids=(1,))


def make_service(memory, store=None):
    return CustomerMemoryService(
        memory=memory,
        processed_turn_store=store or FakeProcessedTurnStore(),
    )


def test_schema_has_24_closed_keys():
    assert len(MEMORY_KEYS) == 24
    assert "service.open_issue" not in MEMORY_KEYS


def test_schema_rejects_missing_required_field_and_unknown_enum():
    assert validate_value("body.skin_type", {}) == ["缺少必填字段：skin_type"]
    assert validate_value("body.skin_type", {"skin_type": "沙漠皮"})


def test_evidence_gate_rejects_question_as_missing_schema_value():
    result = validate_candidates(
        [candidate("body.skin_type", {}, "这款适合干皮吗？")],
        [{"role": "user", "content": "这款适合干皮吗？"}],
    )
    assert not result.accepted


def test_extraction_agent_returns_message_id_and_program_copies_original_text():
    original = "不好意思，我前面说错了，我其实是干性皮肤。"
    client = DecisionClient({
        "candidates": [{
            "memory_key": "body.skin_type",
            "value": {"skin_type": "dry"},
            "operation_hint": "UPSERT",
            "evidence_message_ids": [1],
        }]
    })
    result = EvidenceExtractor(client=client).extract([{"role": "user", "content": original}])
    assert result[0].evidence_message_ids == (1,)
    assert result[0].evidence == (original,)


def test_single_slot_correction_updates_instead_of_adding_duplicate():
    memory = FakeMemory()
    service = make_service(memory)
    first = candidate("body.skin_type", {"skin_type": "oily"}, "我是油性皮肤")
    second = candidate("body.skin_type", {"skin_type": "dry"}, "我说错了，我是干性皮肤")

    # First fact has no old candidate, so the program safely short-circuits to ADD.
    service.add_validated_turn(
        [{"role": "user", "content": "我是油性皮肤"}], context(),
        extractor=FakeExtractor([first]), decision_engine=MemoryDecisionEngine(client=DecisionClient({})),
    )
    decision_client = DecisionClient({
        "operation": "UPDATE",
        "target_id": "m1",
        "reason": "用户明确纠正了此前的肤质自述",
    })
    result = service.add_validated_turn(
        [{"role": "user", "content": "我说错了，我是干性皮肤"}], context(),
        extractor=FakeExtractor([second]), decision_engine=MemoryDecisionEngine(client=decision_client),
    )

    assert len(memory.items) == 1
    assert len(memory.update_calls) == 1
    assert result["decisions"][0]["operation"] == "UPDATE"
    assert len(decision_client.calls) == 1
    value = json.loads(memory.items[0]["metadata"]["memory_value"])
    assert value == {"skin_type": "dry"}


def test_idempotent_turn_is_noop_before_extraction():
    memory = FakeMemory()
    store = FakeProcessedTurnStore()
    service = make_service(memory, store)
    fact = candidate("body.skin_type", {"skin_type": "dry"}, "我是干性皮肤")
    messages = [{"role": "user", "content": "我是干性皮肤"}]
    client = DecisionClient({})
    service.add_validated_turn(messages, context(), extractor=FakeExtractor([fact]), decision_engine=MemoryDecisionEngine(client=client))
    result = service.add_validated_turn(messages, context(), extractor=FakeExtractor([fact]), decision_engine=MemoryDecisionEngine(client=client))
    assert result["idempotent_skip"] is True
    assert len(memory.add_calls) == 1


def test_exact_existing_fact_is_program_noop_without_decision_agent():
    memory = FakeMemory()
    service = make_service(memory)
    old_metadata = {
        **context().metadata(),
        "memory_key": "body.skin_type",
        "memory_value": json.dumps({"skin_type": "dry"}),
    }
    memory.items.append({"id": "m1", "memory": "dry", "metadata": old_metadata})
    client = DecisionClient({"operation": "DELETE", "target_id": "m1"})
    decision = MemoryDecisionEngine(client=client).decide(
        candidate("body.skin_type", {"skin_type": "dry"}, "我是干性皮肤"), memory.items
    )
    assert decision.operation == "NOOP"
    assert client.calls == []


def test_decision_agent_can_delete_an_explicitly_withdrawn_fact():
    memory = FakeMemory()
    memory.items.append({
        "id": "m1",
        "user_id": "u1",
        "memory": "avoid alcohol",
        "metadata": {
            **context().metadata(),
            "memory_key": "body.ingredient_constraint",
            "memory_value": json.dumps({"ingredient": "酒精", "constraint_type": "avoid"}),
        },
    })
    service = make_service(memory)
    withdrawn = CandidateMemory(
        "body.ingredient_constraint",
        {"ingredient": "酒精", "constraint_type": "avoid"},
        ("我撤回之前关于避用酒精的说法。",),
        operation_hint="DELETE",
        evidence_message_ids=(1,),
    )
    client = DecisionClient({"operation": "DELETE", "target_id": "m1", "reason": "用户明确撤回旧事实"})
    result = service.add_validated_turn(
        [{"role": "user", "content": "我撤回之前关于避用酒精的说法。"}],
        context(),
        extractor=FakeExtractor([withdrawn]),
        decision_engine=MemoryDecisionEngine(client=client),
    )
    assert result["decisions"][0]["operation"] == "DELETE"
    assert memory.items == []


def test_retrieval_is_user_scoped_not_event_scoped():
    memory = FakeMemory()
    service = make_service(memory)
    service.search_event_memories("现在怎么办", context(), top_k=2)
    assert memory.search_calls
    assert all(call["filters"] == {"user_id": "u1"} for call in memory.search_calls)
