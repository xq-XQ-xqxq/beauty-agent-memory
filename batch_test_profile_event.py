"""Run the Profile+Event pipeline against the same official chats used by
batch_test_official_data.py, then emit a side-by-side comparison report.

用法：
- 与 batch_test_official_data.py 共享数据源（Excel 或提取后的 JSON）。
- 选用环境变量 MEM0_TEST_SESSIONS 指定会话（默认与原脚本一致）。
- 输出：
  * data/official_batch_test_results_profile_event_{RUN_NAME}.json
  * Mem0主办方数据批量测试_新方案_{RUN_NAME}.md
- 不依赖 Mem0 / Qdrant / 向量数据库；只需要 OpenAI 兼容 LLM endpoint。
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

from profile_event_pipeline import ProfileEventPipeline
from profile_event_schema import EVENT_SPEC_MAP, PROFILE_SPEC_MAP, is_event_key, is_profile_key
from profile_event_store import ProfileEventStore, render_event_text, render_profile_text


PROJECT_DIR = Path(__file__).resolve().parent
WORKBOOK = PROJECT_DIR.parent / "赛题 1：数据共情者-业务数据.xlsx"
EXTRACTED_INPUT = PROJECT_DIR / "data" / "official_selected_chats.json"
RUN_NAME = os.environ.get("MEM0_TEST_RUN_NAME", "pe_v1")
REPORT = PROJECT_DIR / f"Mem0主办方数据批量测试_新方案_{RUN_NAME}.md"
RAW_RESULT = PROJECT_DIR / "data" / f"official_batch_test_results_profile_event_{RUN_NAME}.json"
DB_PATH = PROJECT_DIR / "data" / f"profile_event_store_{RUN_NAME}.sqlite3"
SELECTED_SESSIONS = [
    value.strip()
    for value in os.environ.get(
        "MEM0_TEST_SESSIONS", "S00002,S00003,S00006,S00010,S00024"
    ).split(",")
    if value.strip()
]


# ---------------------------------------------------------------------------
# Data loading (mirror of batch_test_official_data.py)
# ---------------------------------------------------------------------------

def as_text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def load_sessions() -> list[dict[str, Any]]:
    if EXTRACTED_INPUT.exists():
        rows = json.loads(EXTRACTED_INPUT.read_text(encoding="utf-8"))
    elif WORKBOOK.exists():
        import pandas as pd
        frame = pd.read_excel(WORKBOOK, sheet_name="聊天记录")
        rows = frame[frame["会话ID"].isin(SELECTED_SESSIONS)].to_dict(orient="records")
    else:
        # Neither data source is available; let main() emit a helpful message.
        return []
    rows.sort(key=lambda row: (str(row["会话ID"]), int(row["消息序号"])))
    grouped: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in SELECTED_SESSIONS}
    for row in rows:
        grouped.setdefault(str(row["会话ID"]), []).append(row)
    sessions: list[dict[str, Any]] = []
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
            "session_id": session_id,
            "buyer": as_text(first["买家昵称"]),
            "scene_major": as_text(first["scene_major"]),
            "scene_minor": as_text(first["scene_minor"]),
            "occurred_at": as_text(first["发送时间"]).replace(" ", "T"),
            "messages": messages,
        })
    return sessions


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(store: ProfileEventStore, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Per-run processed_turns ledger so re-runs with a new RUN_NAME are not short-circuited
    # by idempotency records left over from a previous RUN_NAME.
    processed_turns_path = Path(store.path).parent / f"profile_event_processed_turns_{RUN_NAME}.sqlite3"
    if processed_turns_path.exists():
        processed_turns_path.unlink()
    pipeline = ProfileEventPipeline(
        store=store,
        processed_turns_path=processed_turns_path,
    )
    records: list[dict[str, Any]] = []
    for session in sessions:
        user_id = f"official_{session['buyer']}"
        # Reset per-user state so each session is reproducible in isolation.
        store.reset_user(user_id)
        first = pipeline.process_turn(
            messages=session["messages"],
            user_id=user_id,
            conversation_id=session["session_id"],
            scene_major=session["scene_major"],
            scene_minor=session["scene_minor"],
            occurred_at=session["occurred_at"],
        )
        # Re-run the same turn to verify idempotency (mirrors v1 batch test).
        duplicate = pipeline.process_turn(
            messages=session["messages"],
            user_id=user_id,
            conversation_id=session["session_id"],
            scene_major=session["scene_major"],
            scene_minor=session["scene_minor"],
            occurred_at=session["occurred_at"],
        )
        profile_state = store.get_all_profile(user_id)
        events_state = store.get_events(user_id)
        records.append({
            "session_id": session["session_id"],
            "buyer": session["buyer"],
            "scene_major": session["scene_major"],
            "scene_minor": session["scene_minor"],
            "message_count": len(session["messages"]),
            "first_profile_decisions": first.profile_decisions,
            "first_event_decisions": first.event_decisions,
            "first_rejected": list(first.rejected),
            "duplicate_idempotent_skip": duplicate.idempotent_skip,
            "duplicate_profile_decisions": duplicate.profile_decisions,
            "duplicate_event_decisions": duplicate.event_decisions,
            "latency_ms": first.latency_ms,
            "token_usage": dict(first.token_usage),
            "llm_judge_calls": first.llm_judge_calls,
            "llm_judge_latency_ms": first.llm_judge_latency_ms,
            "llm_judge_token_usage": dict(first.llm_judge_token_usage),
            "final_profile": profile_state,
            "final_events": [
                {
                    "event_id": e["event_id"],
                    "event_type": e["event_type"],
                    "occurred_at": e["occurred_at"],
                    "description": e["description"],
                    "fields": e["fields"],
                    "evidence": e["evidence"],
                }
                for e in events_state
            ],
        })
        print(
            f"{session['session_id']}: profile={len(first.profile_decisions)} "
            f"events={len(first.event_decisions)} rejected={len(first.rejected)} "
            f"latency={first.latency_ms}ms tokens={first.token_usage.get('total_tokens', 0)} "
            f"judge_calls={first.llm_judge_calls} idempotent={duplicate.idempotent_skip}",
            flush=True,
        )
    return records


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _counter(decisions: list[dict[str, Any]]) -> Counter:
    return Counter(d.get("operation", "NOOP") for d in decisions)


def render_report(records: list[dict[str, Any]]) -> str:
    profile_total = sum(len(r["first_profile_decisions"]) for r in records)
    event_total = sum(len(r["first_event_decisions"]) for r in records)
    rejected_total = sum(len(r["first_rejected"]) for r in records)
    duplicate_profile = sum(len(r["duplicate_profile_decisions"]) for r in records)
    duplicate_event = sum(len(r["duplicate_event_decisions"]) for r in records)
    all_idempotent = all(r["duplicate_idempotent_skip"] for r in records)

    profile_op_counts: Counter = Counter()
    event_op_counts: Counter = Counter()
    llm_used = 0
    total_extract_latency = 0
    total_extract_tokens = 0
    total_judge_latency = 0
    total_judge_tokens = 0
    total_judge_calls = 0
    for record in records:
        profile_op_counts.update(_counter(record["first_profile_decisions"]))
        event_op_counts.update(_counter(record["first_event_decisions"]))
        llm_used += sum(1 for d in record["first_event_decisions"] if d.get("used_llm"))
        total_extract_latency += record.get("latency_ms", 0)
        total_extract_tokens += record.get("token_usage", {}).get("total_tokens", 0)
        total_judge_latency += record.get("llm_judge_latency_ms", 0)
        total_judge_tokens += record.get("llm_judge_token_usage", {}).get("total_tokens", 0)
        total_judge_calls += record.get("llm_judge_calls", 0)

    lines = [
        f"# Mem0 主办方数据批量测试 - 新方案（Profile+Event 双通道）{RUN_NAME}",
        "",
        "## 测试结论",
        "",
        f"- 测试会话：{len(records)} 个",
        f"- Profile 通道决策：{profile_total} 条（UPDATE {profile_op_counts.get('UPDATE', 0)}，"
        f"NOOP {profile_op_counts.get('NOOP', 0)}）",
        f"- Event 通道决策：{event_total} 条（ADD {event_op_counts.get('ADD', 0)}，"
        f"NOOP {event_op_counts.get('NOOP', 0)}）",
        f"- 证据/校验拒绝候选：{rejected_total} 条",
        f"- LLM same/different 判定调用：{total_judge_calls} 次（仅在签名相同 fields 不同时触发）",
        f"- 相同会话重复写入新增决策：profile {duplicate_profile} + event {duplicate_event}（理想值为 0）",
        f"- 幂等跳过：{'全部跳过' if all_idempotent else '存在未跳过项'}",
        "- 数据源：主办方《赛题 1：数据共情者-业务数据.xlsx》聊天记录表",
        "",
        "## 性能 / Token 使用（5 会话合计）",
        "",
        f"- 抽取 LLM 调用：5 次（每个会话 1 次），总耗时 {total_extract_latency} ms，平均 {total_extract_latency // max(len(records), 1)} ms/次",
        f"- 抽取 LLM Token：合计 {total_extract_tokens}（含 system+user prompt 与 completion）",
        f"- 同异判断 LLM 调用：{total_judge_calls} 次，总耗时 {total_judge_latency} ms，Token {total_judge_tokens}",
        f"- LLM 总调用量：{5 + total_judge_calls} 次（现有方案等价场景约需 5 次抽取 + {profile_total + event_total} 次决策 = {5 + profile_total + event_total} 次）",
        f"- LLM 总 Token：{total_extract_tokens + total_judge_tokens}",
        "",
        "## 与现有方案的关键差异",
        "",
        "- 存储结构：现有方案单一记忆列表（向量库）；新方案 Profile 表（按 key 覆盖）+ Event 表（按时间线追加，SQLite）。",
        "- LLM 决策范围：现有方案 LLM 输出 ADD/UPDATE/DELETE/NOOP；新方案 LLM 仅输出 same/different 二分类，决策由程序执行。",
        "- LLM 调用量：现有方案每条候选都过决策 LLM；新方案仅签名相同但 fields 不同时调用 LLM。",
        "- 历史感知：现有方案抽取时不带历史；新方案 Step 0 先查 Profile+Event，让 LLM 只提取增量。",
        "- 枚举处理：现有方案 LLM 必须输出枚举码；新方案 LLM 输出自然语言值，程序归一化。",
        "",
        "## 分场景结果",
        "",
    ]

    for record in records:
        lines.extend([
            f"### {record['session_id']}｜{record['scene_major']}－{record['scene_minor']}",
            "",
            f"对话消息数：{record['message_count']}。"
            f"Profile 决策 {len(record['first_profile_decisions'])} 条；"
            f"Event 决策 {len(record['first_event_decisions'])} 条；"
            f"拒绝 {len(record['first_rejected'])} 条。",
            "",
            "**Profile 通道决策：**",
            "",
        ])
        if record["first_profile_decisions"]:
            for d in record["first_profile_decisions"]:
                lines.append(
                    f"- `{d['operation']}` {d['key']} → "
                    f"{json.dumps(d.get('value', {}), ensure_ascii=False)} "
                    f"｜ {d['reason']}"
                )
        else:
            lines.append("- （无）")
        lines.extend(["", "**Event 通道决策：**", ""])
        if record["first_event_decisions"]:
            for d in record["first_event_decisions"]:
                llm_flag = "（LLM 判定）" if d.get("used_llm") else ""
                lines.append(
                    f"- `{d['operation']}`{llm_flag} {d['key']} → "
                    f"{json.dumps(d.get('value', {}), ensure_ascii=False)} "
                    f"｜ {d['reason']}"
                )
        else:
            lines.append("- （无）")
        lines.extend(["", "**被拒绝候选：**", ""])
        if record["first_rejected"]:
            for r in record["first_rejected"]:
                lines.append(
                    f"- {r.get('key', '')}：{json.dumps(r.get('value', {}), ensure_ascii=False)} "
                    f"｜ {'、'.join(r.get('reasons', []))}"
                )
        else:
            lines.append("- （无）")
        lines.extend(["", "**最终 Profile 状态：**", ""])
        if record["final_profile"]:
            for key, entry in sorted(record["final_profile"].items()):
                lines.append(
                    f"- {render_profile_text(key, entry.get('value', {}))} "
                    f"｜ evidence: {entry.get('evidence', [])}"
                )
        else:
            lines.append("- （无）")
        lines.extend(["", "**最终 Event 时间线：**", ""])
        if record["final_events"]:
            for event in record["final_events"]:
                lines.append(f"- {render_event_text(event)}")
        else:
            lines.append("- （无）")
        lines.extend(["", "---", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sessions = load_sessions()
    if not sessions:
        print(
            f"No sessions loaded. Put data at {EXTRACTED_INPUT} or {WORKBOOK}, "
            "or set MEM0_TEST_SESSIONS to match your workbook.",
            flush=True,
        )
        return

    if DB_PATH.exists():
        DB_PATH.unlink()
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with ProfileEventStore(DB_PATH) as store:
        records = run_pipeline(store, sessions)

    RAW_RESULT.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESULT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render_report(records), encoding="utf-8")
    print(f"Report: {REPORT}")
    print(f"Raw:    {RAW_RESULT}")


if __name__ == "__main__":
    main()
