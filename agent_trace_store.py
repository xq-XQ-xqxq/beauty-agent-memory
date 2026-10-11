"""Agent 运行轨迹持久化与回放（可观测性）。

设计：
- 每次 ToolCallingAgent.run 完整落一条 JSONL 记录，包含：
  run_id / 时间 / user_id / model / 原始输入 / 完整 chat_messages（可回放）/
  每步工具决策与执行 / 耗时 / token / 是否强制收尾 / 最终洞察。
- 按天分文件：data/agent_traces/traces_YYYYMMDD.jsonl，追加写入。
- 给定失败案例的 run_id，load_run / replay_run 可还原当时的完整消息序列，
  用于离线复盘和把失败案例加进评估集。

面试考点：
- 为什么要落盘而不是只打日志：结构化轨迹可被程序分析（统计步数/token/错误率），
  而日志只能人肉 grep。
- 为什么存完整 chat_messages：agent 的 bug 经常依赖"第 N 轮喂回去的 tool 结果长什么样"
  才能复现，只存工具名参数不足以回放。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRACES_DIR = Path(__file__).parent / "data" / "agent_traces"


class AgentTraceStore:
    """JSONL 轨迹存储。"""

    def __init__(self, traces_dir: Path | str | None = None) -> None:
        self._dir = Path(traces_dir) if traces_dir else DEFAULT_TRACES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save_run(
        self,
        *,
        run_id: str,
        user_id: str,
        model: str,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        chat_messages: list[dict[str, Any]],
        result: Any,
    ) -> Path:
        """持久化一次完整 agent 运行。返回写入的文件路径。"""
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user_id": user_id,
            "model": model,
            "input": {
                "messages": messages,
                "memory_index": [
                    {
                        "memory_id": str(m.get("id", "")),
                        "memory_key": str(m.get("metadata", {}).get("memory_key", "")),
                    }
                    for m in (memories or [])[:20]
                ],
            },
            # 完整对话序列（system/user/assistant/tool），回放的关键
            "chat_messages": chat_messages,
            "trace": result.to_display(),
            "raw_insight": result.raw_insight,
        }

        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._dir / f"traces_{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------------
    # 读取 / 回放
    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        """按 run_id 加载完整轨迹记录。"""
        for path in sorted(self._dir.glob("traces_*.jsonl"), reverse=True):
            for line in self._iter_lines(path):
                if line.get("run_id") == run_id:
                    return line
        return None

    def replay_run(self, run_id: str) -> dict[str, Any] | None:
        """回放指定 run：返回完整记录（含 chat_messages），供离线复盘。"""
        return self.load_run(run_id)

    def list_runs(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """列出最近的运行摘要（不含完整消息体）。"""
        summaries: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("traces_*.jsonl"), reverse=True):
            for line in self._iter_lines(path):
                if user_id and line.get("user_id") != user_id:
                    continue
                trace = line.get("trace", {})
                summaries.append({
                    "run_id": line.get("run_id", ""),
                    "timestamp": line.get("timestamp", ""),
                    "user_id": line.get("user_id", ""),
                    "model": line.get("model", ""),
                    "llm_calls": trace.get("llm_calls", 0),
                    "tool_calls": trace.get("tool_calls", 0),
                    "total_latency_ms": trace.get("total_latency_ms", 0),
                    "token_usage": trace.get("token_usage", {}),
                    "forced_finalization": trace.get("forced_finalization", False),
                    "has_insight": bool(line.get("raw_insight")),
                })
                if len(summaries) >= limit:
                    return summaries
        return summaries

    def iter_errors(self, limit: int = 20) -> list[dict[str, Any]]:
        """筛选出有工具错误或触发强制收尾的运行（失败案例快速定位）。"""
        error_runs: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("traces_*.jsonl"), reverse=True):
            for line in self._iter_lines(path):
                trace = line.get("trace", {})
                has_error = any(
                    execution.get("error")
                    for iteration in trace.get("traces", [])
                    for execution in iteration.get("executions", [])
                )
                if has_error or trace.get("forced_finalization"):
                    error_runs.append({
                        "run_id": line.get("run_id", ""),
                        "timestamp": line.get("timestamp", ""),
                        "user_id": line.get("user_id", ""),
                        "forced_finalization": trace.get("forced_finalization", False),
                        "tool_errors": [
                            {
                                "name": execution.get("name"),
                                "error": execution.get("error"),
                            }
                            for iteration in trace.get("traces", [])
                            for execution in iteration.get("executions", [])
                            if execution.get("error")
                        ],
                    })
                    if len(error_runs) >= limit:
                        return error_runs
        return error_runs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_lines(path: Path):
        """逐行读取 JSONL，跳过损坏行。"""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def format_replay(record: dict[str, Any]) -> str:
    """把回放记录格式化为可读文本（用于人工复盘 / 打印）。"""
    lines: list[str] = []
    lines.append(
        f"=== Run {record.get('run_id')} | user={record.get('user_id')} "
        f"| model={record.get('model')} | {record.get('timestamp')} ==="
    )
    for i, msg in enumerate(record.get("chat_messages", []), start=1):
        role = msg.get("role", "?")
        if role == "tool":
            lines.append(f"[{i}] tool({msg.get('name')}) → {str(msg.get('content'))[:200]}")
        elif msg.get("tool_calls"):
            names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg["tool_calls"]
            ]
            lines.append(f"[{i}] assistant 调用工具: {names}")
        else:
            lines.append(f"[{i}] {role}: {str(msg.get('content', ''))[:200]}")
    lines.append(f"--- 最终洞察: {json.dumps(record.get('raw_insight', {}), ensure_ascii=False)}")
    return "\n".join(lines)
