"""Run a reproducible Mem0 smoke test against representative official chats."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from memory_service import CustomerMemoryService, MemoryContext


PROJECT_DIR = Path(__file__).resolve().parent
WORKBOOK = PROJECT_DIR.parent / "赛题 1：数据共情者-业务数据.xlsx"
EXTRACTED_INPUT = PROJECT_DIR / "data" / "official_selected_chats.json"
RUN_NAME = os.environ.get("MEM0_TEST_RUN_NAME", "v1")
REPORT = PROJECT_DIR / f"Mem0主办方数据批量测试_{RUN_NAME}.md"
RAW_RESULT = PROJECT_DIR / "data" / f"official_batch_test_results_{RUN_NAME}.json"
SELECTED_SESSIONS = [
    value.strip()
    for value in os.environ.get(
        "MEM0_TEST_SESSIONS", "S00002,S00003,S00006,S00010,S00024"
    ).split(",")
    if value.strip()
]


def as_text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def load_sessions() -> list[dict[str, Any]]:
    if EXTRACTED_INPUT.exists():
        rows = json.loads(EXTRACTED_INPUT.read_text(encoding="utf-8"))
    else:
        import pandas as pd

        frame = pd.read_excel(WORKBOOK, sheet_name="聊天记录")
        rows = frame[frame["会话ID"].isin(SELECTED_SESSIONS)].to_dict(orient="records")
    rows.sort(key=lambda row: (str(row["会话ID"]), int(row["消息序号"])))
    sessions: list[dict[str, Any]] = []
    grouped = {session_id: [] for session_id in SELECTED_SESSIONS}
    for row in rows:
        grouped.setdefault(str(row["会话ID"]), []).append(row)
    for session_id in SELECTED_SESSIONS:
        group = grouped.get(session_id, [])
        if not group:
            continue
        first = group[0]
        messages = []
        for row in group:
            role_name = as_text(row["角色"])
            if role_name not in {"买家", "客服"}:
                continue
            content = as_text(row["message_text"]) or as_text(row["chat_content"])
            if content:
                messages.append({
                    "role": "user" if role_name == "买家" else "assistant",
                    "content": content,
                })
        sessions.append({
            "session_id": str(session_id),
            "buyer": as_text(first["买家昵称"]),
            "scene_major": as_text(first["scene_major"]),
            "scene_minor": as_text(first["scene_minor"]),
            "occurred_at": as_text(first["发送时间"]).replace(" ", "T"),
            "messages": messages,
        })
    return sessions


def memories(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    return [
        str(item.get("memory", ""))
        for item in result.get("results", [])
        if isinstance(item, dict) and item.get("memory")
    ]


def render_report(records: list[dict[str, Any]]) -> str:
    extracted = sum(len(row["memories"]) for row in records)
    duplicate_writes = sum(len(row["duplicate_memories"]) for row in records)
    lines = [
        "# Mem0 主办方数据批量测试 v1",
        "",
        "## 测试结论",
        "",
        f"- 测试会话：{len(records)} 个",
        f"- 首次写入提取记忆：{extracted} 条",
        f"- 相同会话重复写入新增记忆：{duplicate_writes} 条（理想值为 0）",
        f"- 证据门控拒绝候选：{sum(len(row['rejected']) for row in records)} 条",
        "- 数据源：主办方《赛题 1：数据共情者-业务数据.xlsx》聊天记录表",
        "",
        "## 分场景结果",
        "",
    ]
    for row in records:
        lines.extend([
            f"### {row['session_id']}｜{row['scene_major']}－{row['scene_minor']}",
            "",
            f"对话消息数：{row['message_count']}。首次提取 {len(row['memories'])} 条；重复写入新增 {len(row['duplicate_memories'])} 条。",
            "",
            "提取记忆：",
            "",
        ])
        lines.extend(f"- {item}" for item in row["memories"])
        if not row["memories"]:
            lines.append("- （无）")
        lines.extend(["", "被证据门控拒绝：", ""])
        if row["rejected"]:
            for item in row["rejected"]:
                lines.append(f"- {item.get('memory', '')}｜原因：{'、'.join(item.get('reasons', []))}")
        else:
            lines.append("- （无）")
        lines.extend(["", "五维检索：", ""])
        for dimension, items in row["dimensions"].items():
            texts = [str(item.get("memory", "")) for item in items]
            lines.append(f"- `{dimension}`：" + ("；".join(texts) if texts else "（空）"))
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def main() -> None:
    os.environ.setdefault("MEM0_COLLECTION_NAME", "official_batch_test_v1_1024")
    sessions = load_sessions()
    records: list[dict[str, Any]] = []
    with CustomerMemoryService() as service:
        for session in sessions:
            context = MemoryContext(
                user_id=f"official_{session['buyer']}",
                conversation_id=session["session_id"],
                event_id=f"EVT_{session['session_id']}",
                scene_major=session["scene_major"],
                scene_minor=session["scene_minor"],
                occurred_at=session["occurred_at"],
            )
            first = service.add_validated_turn(session["messages"], context)
            duplicate = service.add_validated_turn(session["messages"], context)
            current_message = next(
                msg["content"] for msg in reversed(session["messages"])
                if msg["role"] == "user"
            )
            retrieved = service.build_label_context(current_message, context, top_k=5)
            records.append({
                **{key: session[key] for key in (
                    "session_id", "buyer", "scene_major", "scene_minor"
                )},
                "message_count": len(session["messages"]),
                "memories": memories(first),
                "duplicate_memories": memories(duplicate),
                "duplicate_idempotent_skip": duplicate.get("idempotent_skip", False),
                "rejected": first.get("rejected", []),
                "dimensions": retrieved["dimensions"],
                "long_term": retrieved["long_term"],
            })
            print(
                f"{session['session_id']}: extracted={len(memories(first))}, "
                f"duplicate={len(memories(duplicate))}",
                flush=True,
            )
    RAW_RESULT.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESULT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render_report(records), encoding="utf-8")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
