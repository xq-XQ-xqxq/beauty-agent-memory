"""Minimal Mem0 v1 demo.

Run `python demo.py --dry-run` without API access to inspect the configured
prompt, metadata, and retrieval questions. Run `python demo.py` after filling
`.env` to exercise the real Mem0 pipeline.
"""

from __future__ import annotations

import argparse
import json

from custom_instructions import CUSTOM_INSTRUCTIONS
from memory_service import CustomerMemoryService, DIMENSION_QUERIES, MemoryContext


CONTEXT = MemoryContext(
    user_id="demo_customer_001",
    conversation_id="S00058",
    event_id="EVT_S00058_ADVERSE_REACTION",
    scene_major="不良反应",
    scene_minor="泛红刺痒",
    occurred_at="2026-05-08T10:00:00",
)

MESSAGES = [
    {
        "role": "user",
        "content": "我是混合敏感肌，昨晚用了B5面膜，三个小时后额头又红又痒。我想退掉，麻烦尽快处理。",
    },
    {
        "role": "assistant",
        "content": "建议先停止使用。可以为您办理使用不适退货，专员会在24小时内回访。",
    },
    {
        "role": "user",
        "content": "好，我先停用，等你们回访。",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({
            "metadata": CONTEXT.metadata(),
            "messages": MESSAGES,
            "dimension_queries": DIMENSION_QUERIES,
            "custom_instructions_preview": CUSTOM_INSTRUCTIONS[:500] + "...",
        }, ensure_ascii=False, indent=2))
        return

    with CustomerMemoryService() as service:
        add_result = service.add_turn(MESSAGES, CONTEXT)
        print("ADD RESULT")
        print(json.dumps(add_result, ensure_ascii=False, indent=2, default=str))

        current_message = "这个面膜我还能继续用吗？"
        label_context = service.build_label_context(current_message, CONTEXT)
        print("LABEL CONTEXT")
        print(json.dumps(label_context, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
