"""Mem0-backed semantic memory service for the customer-support prototype."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

from custom_instructions import CUSTOM_INSTRUCTIONS
from evidence_pipeline import FactExtractionAgent, turn_hash, validate_candidates
from memory_decision import MemoryDecisionAgent
from memory_schema import MEMORY_KEYS
from processed_turn_store import ProcessedTurnStore

try:
    from dotenv import load_dotenv
except ImportError:  # Allows dry-run/static use before dependencies are installed.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent / ".env")


DIMENSION_QUERIES = {
    "concern": "这个用户在当前事件中遇到了什么困扰、实际影响和核心担忧？",
    "physical_condition": "这个用户有哪些与当前商品或问题相关的身体状态、肤质、症状或处置情况？",
    "need": "这个用户在当前事件中希望获得什么信息、行动或最终结果？",
    "dissatisfaction": "这个用户在当前事件中对什么不满，原因是什么，是否表达了投诉或残留顾虑？",
    "service_consensus": "当前事件中用户与客服已经确认、接受、拒绝或承诺了什么？",
}

DIMENSION_PREFIXES = {
    "concern": "[困扰]",
    "physical_condition": "[身体状态]",
    "need": "[需求]",
    "dissatisfaction": "[不满]",
    "service_consensus": "[服务共识]",
}

LONG_TERM_QUERY = "这个用户有哪些当前问题可能需要考虑的长期肤质、历史不良反应、明确偏好或使用关注？"


@dataclass(frozen=True)
class MemoryContext:
    user_id: str
    conversation_id: str
    event_id: str
    scene_major: str
    scene_minor: str
    occurred_at: str

    def metadata(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "event_id": self.event_id,
            "scene_major": self.scene_major,
            "scene_minor": self.scene_minor,
            "occurred_at": self.occurred_at,
            "prompt_version": "mem0-v1",
        }


def build_mem0_config() -> dict[str, Any]:
    """Build a Mem0 OSS config from environment variables.

    The OpenAI-compatible fields allow Alibaba/Qwen-compatible endpoints without
    hard-coding a contest environment or committing credentials.
    """

    llm_model = os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
    llm_key = (
        os.environ.get("MEM0_LLM_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    llm_base_url = os.environ.get("MEM0_LLM_BASE_URL")

    embed_model = os.environ.get("MEM0_EMBED_MODEL", "text-embedding-v4")
    embed_key = os.environ.get("MEM0_EMBED_API_KEY") or llm_key
    embed_base_url = os.environ.get("MEM0_EMBED_BASE_URL") or llm_base_url

    llm_config: dict[str, Any] = {
        "model": llm_model,
        "api_key": llm_key,
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    if llm_base_url:
        llm_config["openai_base_url"] = llm_base_url

    embed_config: dict[str, Any] = {
        "model": embed_model,
        "api_key": embed_key,
        "embedding_dims": 1024,
    }
    if embed_base_url:
        embed_config["openai_base_url"] = embed_base_url

    data_dir = Path(os.environ.get("MEM0_DATA_DIR", Path(__file__).parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    collection_name = os.environ.get(
        "MEM0_COLLECTION_NAME", "customer_structured_memories_v1_1024"
    )

    return {
        "llm": {"provider": "openai", "config": llm_config},
        "embedder": {"provider": "openai", "config": embed_config},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(data_dir / "qdrant"),
                "collection_name": collection_name,
                "embedding_model_dims": 1024,
            },
        },
        "custom_instructions": CUSTOM_INSTRUCTIONS,
    }


def _prepare_runtime_dirs() -> Path:
    """Keep Mem0 telemetry/history files inside the portable project folder."""

    data_dir = Path(os.environ.get("MEM0_DATA_DIR", Path(__file__).parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    mem0_home = data_dir / "mem0_home"
    mem0_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MEM0_DIR", str(mem0_home))
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    return data_dir


class CustomerMemoryService:
    """Thin domain wrapper around Mem0's native add/search pipeline."""

    def __init__(self, memory: Any | None = None, processed_turn_store: Any | None = None) -> None:
        data_dir = _prepare_runtime_dirs()
        if memory is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise RuntimeError(
                    "Mem0 is not installed. Run: pip install -r requirements.txt"
                ) from exc
            memory = Memory.from_config(build_mem0_config())
        self.memory = memory
        self.processed_turn_store = processed_turn_store or ProcessedTurnStore(
            data_dir / "processed_turns.sqlite3"
        )

    def add_turn(self, messages: list[dict[str, Any]], context: MemoryContext) -> dict[str, Any]:
        """Submit one complete support turn to Mem0's native extraction pipeline."""

        if not messages:
            raise ValueError("messages must not be empty")
        return self.memory.add(
            messages=messages,
            user_id=context.user_id,
            metadata=context.metadata(),
            infer=True,
        )

    def add_validated_turn(
        self,
        messages: list[dict[str, Any]],
        context: MemoryContext,
        extractor: FactExtractionAgent | None = None,
        decision_engine: MemoryDecisionAgent | None = None,
    ) -> dict[str, Any]:
        """Run the fixed two-agent chain, validate decisions, then mutate Mem0."""

        if not messages:
            raise ValueError("messages must not be empty")
        source_turn_hash = turn_hash(messages)
        # Idempotency is independent from mutable memories and survives UPDATE/DELETE.
        previous_result = self.processed_turn_store.get(context.user_id, source_turn_hash)
        if previous_result is not None:
            return {
                "results": [],
                "rejected": [],
                "idempotent_skip": True,
                "source_turn_hash": source_turn_hash,
                "previous_result": previous_result,
            }

        extractor = extractor or FactExtractionAgent()
        decision_engine = decision_engine or MemoryDecisionAgent()
        validation = validate_candidates(extractor.extract(messages), messages)
        written: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for candidate in validation.accepted:
            old_result = self.memory.get_all(
                filters={"user_id": context.user_id, "memory_key": candidate.memory_key},
                top_k=100,
            )
            # Defense in depth: never trust the vector-store filter as the only
            # tenant boundary. A mismatched or unscoped record cannot be compared,
            # updated, deleted, or exposed to another user.
            existing_for_key = _user_scoped_items(old_result, context.user_id)
            decision = decision_engine.decide(candidate, existing_for_key)
            metadata = {
                **context.metadata(),
                "source_turn_hash": source_turn_hash,
                "memory_key": candidate.memory_key,
                "memory_type": MEMORY_KEYS[candidate.memory_key].memory_type,
                "memory_value": json.dumps(candidate.value, ensure_ascii=False, sort_keys=True),
                "evidence": json.dumps(list(candidate.evidence), ensure_ascii=False),
                "write_pipeline": "structured-memory-v1",
            }
            if decision.operation == "ADD":
                result = self.memory.add(
                    messages=[{"role": "user", "content": candidate.memory}],
                    user_id=context.user_id,
                    metadata=metadata,
                    infer=False,
                )
                written.extend(_result_items(result))
            elif decision.operation == "UPDATE" and decision.target_id:
                result = self.memory.update(
                    memory_id=decision.target_id,
                    text=candidate.memory,
                    metadata=metadata,
                )
                written.extend(_result_items(result) or [{"id": decision.target_id, "memory": candidate.memory}])
            elif decision.operation == "DELETE" and decision.target_id:
                self.memory.delete(memory_id=decision.target_id)
            decisions.append({
                "memory_key": candidate.memory_key,
                "value": candidate.value,
                "operation": decision.operation,
                "target_id": decision.target_id,
                "reason": decision.reason,
            })
        output = {
            "results": written,
            "decisions": decisions,
            "rejected": list(validation.rejected),
            "idempotent_skip": False,
            "source_turn_hash": source_turn_hash,
        }
        self.processed_turn_store.record(context.user_id, source_turn_hash, output)
        return output

    def search_event_memories(
        self,
        current_message: str,
        context: MemoryContext,
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve five semantic dimensions for the active event."""

        # Event ID is provenance only. Relevant unresolved history may cross sessions.
        filters = {"user_id": context.user_id}
        output: dict[str, list[dict[str, Any]]] = {}
        for dimension, base_query in DIMENSION_QUERIES.items():
            query = f"当前用户消息：{current_message}\n检索目标：{base_query}"
            result = self.memory.search(query=query, filters=filters, top_k=max(top_k * 3, 10))
            output[dimension] = _filter_by_prefix(
                _result_items(result), DIMENSION_PREFIXES[dimension]
            )[:top_k]
        return output

    def search_long_term_memories(
        self,
        current_message: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve user-level memories relevant to the current message."""

        query = f"当前用户消息：{current_message}\n检索目标：{LONG_TERM_QUERY}"
        result = self.memory.search(
            query=query,
            filters={"user_id": user_id},
            top_k=top_k,
        )
        return _filter_by_prefix(_result_items(result), "[长期]")

    def build_label_context(
        self,
        current_message: str,
        context: MemoryContext,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return deduplicated retrieval context for a downstream label LLM."""

        dimensions = self.search_event_memories(current_message, context, top_k=top_k)
        long_term = self.search_long_term_memories(current_message, context.user_id, top_k=top_k)
        all_items = [item for items in dimensions.values() for item in items] + long_term
        return {
            "current_message": current_message,
            "event_id": context.event_id,
            "dimensions": dimensions,
            "long_term": long_term,
            "deduplicated_memories": _deduplicate(all_items),
        }

    def get_user_memories(self, user_id: str) -> list[dict[str, Any]]:
        """Read all active structured memories for one user without vector search."""

        result = self.memory.get_all(filters={"user_id": user_id}, top_k=100)
        return sorted(
            _user_scoped_items(result, user_id),
            key=lambda item: (
                str((item.get("metadata") or {}).get("memory_key", "")),
                str(item.get("id", "")),
            ),
        )

    def close(self) -> None:
        """Release Mem0/vector-store resources deterministically on Windows."""

        closed_clients: set[int] = set()
        for name in ("vector_store", "_entity_store", "_telemetry_vector_store"):
            store = getattr(self.memory, name, None)
            client = getattr(store, "client", None)
            if client is None or id(client) in closed_clients:
                continue
            client.close()
            closed_clients.add(id(client))
        close = getattr(self.memory, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "CustomerMemoryService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _result_items(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = result.get("results", [])
        return [item for item in items if isinstance(item, dict)]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _user_scoped_items(result: Any, user_id: str) -> list[dict[str, Any]]:
    """Enforce the user boundary again after reading from the backing store."""

    expected = str(user_id).strip()
    output: list[dict[str, Any]] = []
    for item in _result_items(result):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        actual = str(item.get("user_id") or metadata.get("user_id") or "").strip()
        if actual == expected:
            output.append(item)
    return output


def _deduplicate(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("id") or item.get("memory") or item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _filter_by_prefix(items: Iterable[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    return [item for item in items if marker in str(item.get("memory", ""))]
