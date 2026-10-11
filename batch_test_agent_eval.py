"""Agent 端到端评估管道：工具选择 / 参数 / 任务完成率 / 步数 / 成本。

两种运行模式（同一份 15 条黄金集 eval/agent_eval_golden.json）：

1. 确定性模式（默认，零 token）：
   决策方是一个透明的关键词策略基线 PolicyPlanner（不是 LLM），
   通过脚本化 FakeClient 驱动【真实的 ToolCallingAgent + 真实工具执行层】，
   用于：① CI 回归（框架/提示词改动后指标可复现）② 给真实 LLM 提供对照基线。
   面试叙事："规则基线完成率 X%，qwen-plus 达到 Y%，差距主要在 Z 类场景。"

2. 真实模式（--real，消耗 token）：
   qwen-plus 通过 OpenAI 兼容接口真实运行 agent loop，
   额外统计延迟和 token；每次运行轨迹落 data/agent_eval_traces/ 可回放。

指标：
- 任务完成率：工具集/禁用工单/参数/焦点/JSON 结构全部通过
- 工具选择召回率：期望工具被执行的比例
- 工具选择精确率：执行的工具中期望工具占比（惩罚多调）
- 违禁工具误触率：forbidden_tools 被执行的用例比例（工单护栏）
- 参数断言通过率：equals / contains
- 焦点命中率：service_focus 命中期许关键词
- 平均 LLM 步数 / 平均延迟 / 平均 token（仅真实模式有意义）

运行：
    python batch_test_agent_eval.py            # 确定性基线（零 token）
    python batch_test_agent_eval.py --real     # 真实 qwen-plus 评估
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from knowledge_retriever import RetrievedDoc
from profile_event_store import ProfileEventStore
from service_ticket_store import ServiceTicketStore
from tool_calling_agent import ToolCallingAgent
from tool_registry import ToolRegistry

ROOT = Path(__file__).parent
GOLDEN_PATH = ROOT / "eval" / "agent_eval_golden.json"
REPORT_PATH = ROOT / "data" / "agent_eval_report.json"
HISTORY_DIR = ROOT / "data" / "agent_eval_history"


# ---------------------------------------------------------------------------
# 确定性模式：Fake 检索器（不调 embedding API，保证零 token）
# ---------------------------------------------------------------------------

class _FakeRetriever:
    """鸭子类型 retriever：返回固定文档，仅供确定性基线走真实工具执行层。"""

    _DOCS = {
        "product": ("prod_eval_001", "product", "评测用产品文档"),
        "ingredient": ("ing_eval_001", "ingredient", "评测用成分文档"),
        "faq": ("faq_eval_001", "faq", "评测用 FAQ 文档"),
        "policy": ("pol_eval_001", "policy", "评测用政策文档"),
    }

    def search(self, query, doc_type=None, top_k=5):
        if not query or not str(query).strip():
            return []
        doc_id, dtype, title = self._DOCS.get(doc_type or "product", self._DOCS["product"])
        return [RetrievedDoc(
            doc_id=doc_id,
            doc_type=dtype,
            title=title,
            content="评测占位内容",
            metadata={},
            score=1.0,
        )]


# ---------------------------------------------------------------------------
# 确定性策略基线：透明关键词规则（这是被对比的 baseline，不是被测对象）
# ---------------------------------------------------------------------------

_INGREDIENTS = ["水杨酸", "烟酰胺", "酒精", "玻尿酸", "神经酰胺", "A醇", "视黄醇"]
_PRODUCT_NOUNS = ["精华", "面霜", "面膜", "水乳"]
_PRODUCT_ACTIONS = ["推荐", "功效", "有货", "适合", "含量", "起皮", "保湿"]
_POLICY_HINTS = ["七天无理由", "退换货规则", "无理由", "政策", "规则"]
_HISTORY_HINTS = ["投诉过", "查一下记录", "历史记录", "上个月", "之前的记录"]
_COMPLAINT_HINTS = ["垃圾", "气死", "生气", "差评", "愤怒"]
_SAFETY_HINTS = ["能用", "安全", "含量", "刺激", "过敏", "成分"]
_ALLERGY_FILING_HINTS = ["用了", "你们", "严重", "烂脸", "又红又痒", "泛红", "投诉"]
_GREETING_HINTS = ["你好", "您好", "天气", "在吗", "早上好", "晚上好"]


def policy_plan(user_id: str, messages: list[dict], memories: list[dict]) -> tuple[list[dict], str]:
    """根据对话内容确定性地产出工具调用计划和 service_focus。

    返回 (tool_calls, focus)。这是规则基线：简单、可解释、零成本。
    """
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    calls: list[dict] = []
    focus_parts: list[str] = []

    wants_history = any(h in text for h in _HISTORY_HINTS)
    wants_complaint = any(h in text for h in _COMPLAINT_HINTS)
    wants_policy = any(h in text for h in _POLICY_HINTS)
    wants_product = (
        any(n in text for n in _PRODUCT_NOUNS)
        and any(a in text for a in _PRODUCT_ACTIONS)
        and not wants_policy
    )
    ingredient = next((i for i in _INGREDIENTS if i in text), None)
    wants_safety = ingredient is not None and any(h in text for h in _SAFETY_HINTS)

    # 工单：明确的售后诉求（唯一写操作，最后执行）
    ticket_type = None
    if "退货" in text or "退了" in text:
        ticket_type = "return"
    elif "换货" in text or "换一瓶" in text or "换一款" in text:
        ticket_type = "exchange"
    elif "退款" in text:
        ticket_type = "refund"
    elif ("过敏" in text or "烂脸" in text) and any(h in text for h in _ALLERGY_FILING_HINTS):
        ticket_type = "allergy_case"

    # 组装（只读在前，写草稿在后）
    if wants_safety and not memories:
        calls.append({"name": "get_user_profile", "arguments": {"user_id": user_id}})
    if wants_policy:
        calls.append({
            "name": "search_knowledge_base",
            "arguments": {"query": "七天无理由退换货政策", "doc_type": "policy", "top_k": 3},
        })
    if wants_product:
        calls.append({
            "name": "search_knowledge_base",
            "arguments": {"query": "美妆产品推荐", "doc_type": "product", "top_k": 3},
        })
    # 明确查历史投诉，或情绪投诉但未提售后诉求（查历史避免重复冲突）
    if wants_history and not ticket_type:
        calls.append({
            "name": "get_user_events",
            "arguments": {
                "user_id": user_id,
                "event_type": "dissatisfaction.escalation_intent",
                "limit": 5,
            },
        })
    elif wants_complaint and not ticket_type:
        calls.append({
            "name": "get_user_events",
            "arguments": {
                "user_id": user_id,
                "event_type": "dissatisfaction.reason",
                "limit": 5,
            },
        })
    if wants_safety:
        calls.append({
            "name": "check_ingredient_safety",
            "arguments": {"ingredient_name": ingredient},
        })
    if ticket_type:
        calls.append({
            "name": "create_aftersales_ticket_draft",
            "arguments": {
                "user_id": user_id,
                "ticket_type": ticket_type,
                "reason": f"用户明确提出售后诉求（{ticket_type}）",
                "product_name": next((n for n in _PRODUCT_NOUNS if n in text), ""),
            },
        })

    # focus（与真实 LLM 输出同样按关键词评分）
    focus_map = {
        "return": "用户明确申请退货，已生成工单草稿待人工确认",
        "exchange": "用户明确申请换货，已生成工单草稿待人工确认",
        "refund": "用户明确要求退款，已生成工单草稿待人工确认",
        "allergy_case": "用户反馈使用后过敏，已生成过敏工单待人工确认",
    }
    if ticket_type:
        focus_parts.append(focus_map[ticket_type])
    if wants_safety:
        focus_parts.append("用户咨询成分安全性与适用性")
    if wants_policy:
        focus_parts.append("用户咨询售后退换政策")
    if wants_history:
        focus_parts.append("用户查询历史投诉记录")
    if any(h in text for h in _COMPLAINT_HINTS):
        focus_parts.append("用户情绪不满，需要安抚")
    if wants_product and not ticket_type:
        focus_parts.append("用户咨询产品信息与推荐")
    if not calls and any(h in text for h in _GREETING_HINTS):
        focus_parts.append("日常问候，暂无服务诉求")
    if not focus_parts:
        focus_parts.append("用户咨询，待进一步确认需求")

    return calls, "；".join(focus_parts)


class _ScriptedCompletions:
    """按脚本返回：第一步工具调用（若有），第二步最终 JSON。usage 全 0（零 token）。"""

    def __init__(self, steps: list[dict]):
        self._steps = steps
        self.call_kwargs: list[dict] = []

    def create(self, **kwargs):
        self.call_kwargs.append(kwargs)
        idx = min(len(self.call_kwargs) - 1, len(self._steps) - 1)
        step = self._steps[idx]

        if step["type"] == "tool":
            tool_calls = [
                SimpleNamespace(
                    id=f"policy_call_{idx}_{i}",
                    function=SimpleNamespace(
                        name=c["name"],
                        arguments=json.dumps(c["arguments"], ensure_ascii=False),
                    ),
                )
                for i, c in enumerate(step["calls"])
            ]
            message = SimpleNamespace(content=None, tool_calls=tool_calls)
        else:
            message = SimpleNamespace(content=json.dumps(step["insight"], ensure_ascii=False),
                                      tool_calls=None)

        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _ScriptedClient:
    def __init__(self, calls: list[dict], insight: dict):
        steps = [{"type": "tool", "calls": calls}] if calls else []
        steps.append({"type": "final", "insight": insight})
        self.completions = _ScriptedCompletions(steps)
        self.chat = SimpleNamespace(completions=self.completions)


def _build_policy_insight(focus: str, calls: list[dict], n_messages: int) -> dict:
    """策略基线最终输出：结构与真实 LLM 完全一致，走同一套评分。"""
    urgent = any(c["arguments"].get("ticket_type") == "allergy_case" for c in calls)
    labels = []
    if focus:
        labels.append({
            "title": "当前服务焦点",
            "content": focus,
            "priority": "urgent" if urgent else "normal",
            "source_ids": [f"msg_{n_messages}"],
        })
    return {"service_focus": focus, "labels": labels, "service_suggestions": []}


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------

def score_case(case: dict, result: Any, run_error: str | None = None) -> dict[str, Any]:
    """单条用例打分。"""
    executed = [
        e.name
        for t in result.tool_trace
        for e in t.executions
    ] if result else []
    executed_args = {
        e.name: e.arguments
        for t in result.tool_trace
        for e in t.executions
    } if result else {}
    executed_set = set(executed)
    expected = set(case.get("expected_tools", []))
    forbidden = set(case.get("forbidden_tools", []))

    # 工具召回 / 精确
    hits = expected & executed_set
    tool_recall = len(hits) / len(expected) if expected else 1.0
    tool_precision = len(hits) / len(executed_set) if executed_set else (1.0 if not expected else 0.0)
    forbidden_hit = sorted(forbidden & executed_set)

    # 参数断言
    param_checks: list[dict] = []
    for spec in case.get("expected_params", []):
        tool = spec["tool"]
        args = executed_args.get(tool)
        passed = False
        actual: Any = None
        if args is not None:
            actual = args.get(spec["arg"], "")
            if "equals" in spec:
                passed = actual == spec["equals"]
            elif "contains" in spec:
                passed = spec["contains"] in str(actual)
            elif "in" in spec:
                passed = actual in spec["in"]
        expect_desc = (
            spec.get("equals")
            or (f"contains:{spec.get('contains')}" if "contains" in spec else None)
            or (f"in:{spec.get('in')}" if "in" in spec else None)
        )
        param_checks.append({
            "tool": tool,
            "arg": spec["arg"],
            "expect": expect_desc,
            "actual": actual,
            "passed": passed,
        })
    params_pass = all(p["passed"] for p in param_checks)

    # 输出质量（结构 + 焦点关键词）
    raw = result.raw_insight if result else {}
    parse_pass = (
        isinstance(raw, dict)
        and isinstance(raw.get("service_focus"), str)
        and bool(raw["service_focus"].strip())
        and isinstance(raw.get("labels"), list)
    )
    focus = raw.get("service_focus", "") if isinstance(raw, dict) else ""
    focus_keywords = case.get("expected_focus_keywords", [])
    focus_pass = any(k in focus for k in focus_keywords) if focus_keywords else parse_pass

    task_complete = (
        run_error is None
        and tool_recall == 1.0
        and not forbidden_hit
        and params_pass
        and focus_pass
        and parse_pass
    )

    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "executed_tools": executed,
        "tool_recall": tool_recall,
        "tool_precision": tool_precision,
        "forbidden_hit": forbidden_hit,
        "param_checks": param_checks,
        "params_pass": params_pass,
        "parse_pass": parse_pass,
        "focus": focus,
        "focus_pass": focus_pass,
        "task_complete": task_complete,
        "llm_calls": result.llm_calls if result else 0,
        "latency_ms": result.total_latency_ms if result else 0,
        "token_usage": result.token_usage if result else {},
        "forced_finalization": result.forced_finalization if result else False,
        "guardrails": list(result.guardrail_hits) if result else [],
        "run_error": run_error,
    }


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

def run_deterministic(cases: list[dict]) -> list[dict]:
    """零 token：策略基线 + FakeClient，驱动真实 agent 和真实工具层。"""
    tmp = tempfile.mkdtemp(prefix="agent_eval_det_")
    registry = ToolRegistry(
        store=ProfileEventStore(Path(tmp) / "profile.db"),
        retriever=_FakeRetriever(),
        ticket_store=ServiceTicketStore(Path(tmp) / "tickets.db"),
    )
    scored: list[dict] = []
    for case in cases:
        user_id = f"eval_{case['case_id']}"
        messages = case["messages"]
        memories = case.get("memories", [])
        calls, focus = policy_plan(user_id, messages, memories)
        insight = _build_policy_insight(focus, calls, len(messages))
        client = _ScriptedClient(calls, insight)
        agent = ToolCallingAgent(
            registry=registry,
            user_id=user_id,
            client=client,
            model="policy-baseline(zero-token)",
        )
        result = agent.run(messages, memories)
        scored.append(score_case(case, result))
    return scored


def run_real(cases: list[dict], model: str, runs: int = 1) -> list[list[dict]]:
    """真实 LLM 模式：qwen-plus 跑完整 agent loop；runs>1 时重复跑多轮量化方差。"""
    from openai import OpenAI

    from agent_trace_store import AgentTraceStore
    from hybrid_retriever import HybridRetriever

    api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("MEM0_LLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 评估用独立临时 DB（工单不污染应用库），知识库检索复用已有索引（只读）
    tmp = tempfile.mkdtemp(prefix="agent_eval_real_")
    registry = ToolRegistry(
        store=ProfileEventStore(Path(tmp) / "profile.db"),
        retriever=HybridRetriever(),
        ticket_store=ServiceTicketStore(Path(tmp) / "tickets.db"),
    )
    trace_store = AgentTraceStore(ROOT / "data" / "agent_eval_traces")

    all_runs: list[list[dict]] = []
    for run_idx in range(1, runs + 1):
        print(f"--- 真实评估第 {run_idx}/{runs} 轮 ---")
        scored: list[dict] = []
        for case in cases:
            user_id = f"eval_{case['case_id']}_r{run_idx}"
            agent = ToolCallingAgent(
                registry=registry,
                user_id=user_id,
                client=client,
                model=model,
                trace_store=trace_store,
            )
            run_error = None
            result = None
            try:
                result = agent.run(case["messages"], case.get("memories", []))
            except Exception as exc:  # 单条失败不影响整批评估
                run_error = f"{type(exc).__name__}: {exc}"
            scored.append(score_case(case, result, run_error))
            time.sleep(1)  # 保守限速
        run_summary = summarize(scored)
        print(f"    本轮完成率: {run_summary['task_completion_rate']:.1%}")
        all_runs.append(scored)
    return all_runs


# ---------------------------------------------------------------------------
# 汇总 / 输出
# ---------------------------------------------------------------------------

def summarize(scored: list[dict]) -> dict[str, float]:
    n = len(scored)
    return {
        "case_count": n,
        "task_completion_rate": sum(s["task_complete"] for s in scored) / n,
        "tool_selection_recall": sum(s["tool_recall"] for s in scored) / n,
        "tool_selection_precision": sum(s["tool_precision"] for s in scored) / n,
        "forbidden_trigger_rate": sum(bool(s["forbidden_hit"]) for s in scored) / n,
        "param_pass_rate": (
            sum(
                all(p["passed"] for p in s["param_checks"])
                if s["param_checks"] else 1.0
                for s in scored
            ) / n
        ),
        "focus_hit_rate": sum(s["focus_pass"] for s in scored) / n,
        "parse_pass_rate": sum(s["parse_pass"] for s in scored) / n,
        "forced_finalization_rate": sum(s["forced_finalization"] for s in scored) / n,
        "guardrail_trigger_rate": sum(bool(s["guardrails"]) for s in scored) / n,
        "avg_llm_calls": sum(s["llm_calls"] for s in scored) / n,
        "avg_latency_ms": sum(s["latency_ms"] for s in scored) / n,
        "avg_total_tokens": sum(s["token_usage"].get("total_tokens", 0) for s in scored) / n,
        "error_count": sum(bool(s["run_error"]) for s in scored),
        # 全部用例异常 = 基础设施故障（欠费/密钥失效/网络），不是模型能力回退
        "infra_failure": all(bool(s["run_error"]) for s in scored),
    }


_METRIC_ROWS = [
    ("任务完成率", "task_completion_rate", "{:.1%}"),
    ("工具选择召回率", "tool_selection_recall", "{:.1%}"),
    ("工具选择精确率", "tool_selection_precision", "{:.1%}"),
    ("违禁工具误触率", "forbidden_trigger_rate", "{:.1%}"),
    ("参数断言通过率", "param_pass_rate", "{:.1%}"),
    ("焦点命中率", "focus_hit_rate", "{:.1%}"),
    ("洞察可解析率", "parse_pass_rate", "{:.1%}"),
    ("强制收尾触发率", "forced_finalization_rate", "{:.1%}"),
    ("护栏触发率", "guardrail_trigger_rate", "{:.1%}"),
    ("平均 LLM 步数", "avg_llm_calls", "{:.2f}"),
    ("平均延迟(ms)", "avg_latency_ms", "{:.0f}"),
    ("平均总 token", "avg_total_tokens", "{:.0f}"),
    ("运行异常数", "error_count", "{:.0f}"),
]


def aggregate_runs(all_runs: list[list[dict]]) -> tuple[dict, list[dict]]:
    """多轮聚合：指标取均值，完成率给 min/max，逐用例给跨轮稳定性。"""
    run_summaries = [summarize(run) for run in all_runs]
    keys = [k for k in run_summaries[0] if k != "case_count"]
    agg = {
        "case_count": run_summaries[0]["case_count"],
        "runs": len(all_runs),
        "task_completion_min": min(s["task_completion_rate"] for s in run_summaries),
        "task_completion_max": max(s["task_completion_rate"] for s in run_summaries),
    }
    for key in keys:
        agg[key] = sum(s[key] for s in run_summaries) / len(run_summaries)
    agg["infra_failure"] = all(s["infra_failure"] for s in run_summaries)

    # 逐用例跨轮稳定性
    stability: list[dict] = []
    for i, case_result in enumerate(all_runs[-1]):
        completes = [run[i]["task_complete"] for run in all_runs]
        stability.append({
            "case_id": case_result["case_id"],
            "scenario": case_result["scenario"],
            "complete_runs": sum(completes),
            "total_runs": len(all_runs),
            "stability_rate": sum(completes) / len(all_runs),
            "guardrail_runs": sum(bool(run[i]["guardrails"]) for run in all_runs),
            "last_tools": case_result["executed_tools"],
        })
    return agg, stability


def print_report(
    scored: list[dict],
    summary: dict,
    mode: str,
    model: str,
    stability: list[dict] | None = None,
) -> None:
    runs = summary.get("runs", 1)
    title = "确定性策略基线（零 token）" if mode == "deterministic" else f"真实模型 {model}"
    print("\n" + "=" * 100)
    print(f"Agent 端到端评估 | {title} | 用例 {summary['case_count']} 条 | 重复 {runs} 轮")
    print("=" * 100)
    if summary.get("infra_failure"):
        print("⚠⚠ 所有用例均运行异常：这是基础设施故障（账户欠费/密钥失效/网络），")
        print("   本轮 0% 不代表模型能力回退，请勿用于 prompt/模型对比。")
    header = (
        f"{'case_id':<10} {'稳定性':>7} {'工具召回':>8} {'参数':>6} "
        f"{'焦点':>6} {'步数':>4} {'护栏':>5}  末次实际执行"
    )
    print(header)
    print("-" * 100)
    stable_by_id = {s["case_id"]: s for s in (stability or [])}
    for s in scored:
        actual = ",".join(s["executed_tools"]) or "(无)"
        if s["forbidden_hit"]:
            actual += f"  ⚠误触:{s['forbidden_hit']}"
        if runs > 1:
            st = stable_by_id[s["case_id"]]
            stability_txt = f"{st['complete_runs']}/{st['total_runs']}"
        else:
            stability_txt = "✓" if s["task_complete"] else "✗"
        print(
            f"{s['case_id']:<10} "
            f"{stability_txt:>7} "
            f"{s['tool_recall']:>8.2f} "
            f"{('✓' if s['params_pass'] else '✗'):>6} "
            f"{('✓' if s['focus_pass'] else '✗'):>6} "
            f"{s['llm_calls']:>4} "
            f"{len(s['guardrails']):>5}  {actual}"
        )
        if s["run_error"]:
            print(f"            ERROR: {s['run_error']}")
        for p in s["param_checks"]:
            if not p["passed"]:
                print(f"            参数断言失败: {p['tool']}.{p['arg']} "
                      f"期望 {p['expect']} 实际 {p['actual']!r}")

    if runs > 1:
        print("\n逐用例跨轮稳定性（完成轮数/总轮数；<100% 表示该场景路由不稳定）：")
        for st in stability or []:
            if st["stability_rate"] < 1.0:
                print(f"  {st['case_id']}: {st['complete_runs']}/{st['total_runs']}"
                      f"  {st['scenario'][:36]}")

    print("\n" + "-" * 100)
    print("汇总指标（多轮时为均值）：")
    for label, key, fmt in _METRIC_ROWS:
        print(f"  {label:<18}: {fmt.format(summary[key])}")
    if runs > 1:
        print(f"  {'完成率区间':<18}: {summary['task_completion_min']:.1%}"
              f" ~ {summary['task_completion_max']:.1%}")


def print_delta(previous: dict, current: dict) -> None:
    """与上一次同模式报告对比（支撑'改 prompt 前后对比'叙事）。"""
    if not previous or previous.get("mode") != current.get("mode"):
        return
    if current.get("summary", {}).get("infra_failure"):
        print("\n" + "-" * 100)
        print("本轮为基础设施故障，跳过指标对比（上一份有效报告保留在原处，不被覆盖）。")
        return
    old = previous.get("summary", {})
    new = current.get("summary", {})
    print("\n" + "-" * 100)
    print(f"与上次同模式报告对比（{previous.get('timestamp', '?')}）：")
    for key in ["task_completion_rate", "tool_selection_recall", "param_pass_rate",
                "focus_hit_rate", "guardrail_trigger_rate", "avg_llm_calls", "avg_total_tokens"]:
        if key in old and key in new:
            delta = new[key] - old[key]
            arrow = "→" if abs(delta) < 1e-9 else ("↑" if delta > 0 else "↓")
            pct_keys = {"task_completion_rate", "tool_selection_recall", "param_pass_rate",
                        "focus_hit_rate", "guardrail_trigger_rate"}
            if key in pct_keys:
                print(f"  {key:<26}: {old[key]:.1%} {arrow} {new[key]:.1%} ({delta:+.1%})")
            else:
                print(f"  {key:<26}: {old[key]:.2f} {arrow} {new[key]:.2f} ({delta:+.2f})")


def save_report(report: dict) -> None:
    """保存报告：每次归档一份带时间戳的历史；故障轮不覆盖最新有效报告。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    compact_ts = re.sub(r"[^0-9T]", "", report["timestamp"])[:15]
    archive_path = HISTORY_DIR / f"{report['mode']}_{compact_ts}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告归档：{archive_path}")

    if report["summary"].get("infra_failure"):
        print(f"⚠ 基础设施故障轮，最新有效报告保留不动：{REPORT_PATH}")
        return
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"最新有效报告：{REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 端到端评估管道")
    parser.add_argument("--real", action="store_true", help="使用真实 qwen-plus（默认确定性零 token）")
    parser.add_argument("--model", default=None, help="真实模式模型名（默认读 MEM0_LLM_MODEL）")
    parser.add_argument("--runs", type=int, default=1,
                        help="真实模式重复轮数（>1 时报告均值与逐用例稳定性，量化模型波动）")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = golden["cases"]

    mode = "real" if args.real else "deterministic"
    model = args.model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
    runs = max(1, args.runs) if args.real else 1

    started = time.time()
    stability = None
    if args.real:
        all_runs = run_real(cases, model, runs=runs)
        summary, stability = aggregate_runs(all_runs)
        scored = all_runs[-1]
        run_summaries = [
            {k: v for k, v in s.items()}
            for s in (summarize(r) for r in all_runs)
        ]
    else:
        scored = run_deterministic(cases)
        summary = summarize(scored)
        run_summaries = [summary]
    elapsed = round(time.time() - started, 1)
    summary["wall_clock_sec"] = elapsed

    print_report(scored, summary, mode, model, stability)

    previous = None
    if REPORT_PATH.exists():
        try:
            previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
    if previous:
        print_delta(previous, {"mode": mode, "summary": summary})

    save_report({
        "mode": mode,
        "model": model if args.real else "policy-baseline(zero-token)",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_path": str(GOLDEN_PATH),
        "wall_clock_sec": elapsed,
        "summary": summary,
        "run_summaries": run_summaries,
        "case_stability": stability,
        "cases": scored,
    })


if __name__ == "__main__":
    main()
