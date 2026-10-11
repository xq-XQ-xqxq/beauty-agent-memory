"""Agent Loop 测试：用脚本化 Fake LLM 验证循环骨架（不调真实 API）。

覆盖：
- 无工具直接作答
- 单轮工具调用 → 结果回填 → 最终作答（消息角色/tool_call_id）
- MAX_ITER 耗尽时强制收尾（旧实现会静默返回空洞察）
- 工具失败时错误结构化回填
- token 用量聚合
- 轨迹落盘钩子
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_calling_agent import MAX_ITER, ToolCallingAgent
from tool_registry import ToolCallResult


# ---------------------------------------------------------------------------
# Fake LLM：按脚本逐步返回 tool_calls 或最终 JSON
# ---------------------------------------------------------------------------

class FakeCompletions:
    """脚本序列：每一步是 {"tool": [(name, args), ...]} 或 {"final": insight_dict}。"""

    def __init__(self, steps: list[dict]):
        self._steps = steps
        self.call_kwargs: list[dict] = []

    def create(self, **kwargs):
        self.call_kwargs.append(kwargs)
        idx = len(self.call_kwargs) - 1
        step = self._steps[min(idx, len(self._steps) - 1)]

        if "tool" in step:
            tool_calls = []
            for i, (name, arguments) in enumerate(step["tool"]):
                tool_calls.append(SimpleNamespace(
                    id=f"call_{idx}_{i}",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                ))
            message = SimpleNamespace(content=None, tool_calls=tool_calls)
        else:
            message = SimpleNamespace(
                content=json.dumps(step["final"], ensure_ascii=False),
                tool_calls=None,
            )

        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    def __init__(self, steps):
        self.completions = FakeCompletions(steps)
        self.chat = SimpleNamespace(completions=self.completions)


# ---------------------------------------------------------------------------
# Fake Registry：记录调用、可按工具名返回错误
# ---------------------------------------------------------------------------

CANNED_RESULTS = {
    "search_knowledge_base": [
        {"doc_id": "prod_003", "doc_type": "product", "title": "水杨酸清洁面膜", "content": "...", "score": 1.0}
    ],
    "get_user_profile": [
        {"profile_key": "body.skin_type", "value": {"skin_type": "oily"}, "render": "肤质：油性"}
    ],
    "get_user_events": [],
    "check_ingredient_safety": {"ingredient": "水杨酸", "constraints": {"should_avoid": True}},
    "create_aftersales_ticket_draft": {"ticket_id": "TKT_test00000001", "status": "pending"},
}


class FakeRegistry:
    def __init__(self, fail_tools: set[str] | None = None):
        self.fail_tools = fail_tools or set()
        self.executed: list[dict] = []
        self.tool_names = list(CANNED_RESULTS.keys())

    def openai_schemas(self):
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in CANNED_RESULTS
        ]

    def execute_batch(self, calls):
        results = []
        for call in calls:
            self.executed.append(call)
            name = call.get("name", "")
            if name in self.fail_tools:
                results.append(ToolCallResult(
                    name=name,
                    arguments=call.get("arguments", {}),
                    result=None,
                    latency_ms=1,
                    error="模拟工具错误：检索服务暂时不可用",
                ))
            else:
                results.append(ToolCallResult(
                    name=name,
                    arguments=call.get("arguments", {}),
                    result=CANNED_RESULTS.get(name),
                    latency_ms=2,
                ))
        return results


class FakeTraceStore:
    def __init__(self):
        self.records: list[dict] = []

    def save_run(self, **kwargs):
        self.records.append(kwargs)


def _final_insight(focus="测试焦点", source_ids=None):
    return {
        "service_focus": focus,
        "labels": [
            {
                "title": "测试标签",
                "content": "测试内容",
                "priority": "normal",
                "source_ids": source_ids or ["msg_1"],
            }
        ],
        "service_suggestions": [],
    }


def _make_agent(steps, registry=None, trace_store=None):
    registry = registry or FakeRegistry()
    return ToolCallingAgent(
        registry=registry,
        user_id="u_test",
        client=FakeClient(steps),
        model="fake-model",
        trace_store=trace_store,
    ), registry


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_direct_final_answer_without_tools():
    """模型不调工具时：1 次 LLM 调用直接出洞察，token 被采集。"""
    agent, _ = _make_agent([{"final": _final_insight()}])
    result = agent.run(
        messages=[{"role": "user", "content": "你好"}],
        memories=[],
    )

    assert result.llm_calls == 1
    assert result.tool_calls == 0
    assert result.forced_finalization is False
    assert result.raw_insight["service_focus"] == "测试焦点"
    assert result.token_usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert result.run_id  # 非空


def test_single_tool_loop_backfills_results():
    """单轮工具：工具被执行、结果以 role=tool + tool_call_id 回填。"""
    trace_store = FakeTraceStore()
    steps = [
        {"tool": [("search_knowledge_base", {"query": "祛痘产品", "doc_type": "product"})]},
        {"final": _final_insight(source_ids=["prod_003"])},
    ]
    agent, registry = _make_agent(steps, trace_store=trace_store)
    result = agent.run(
        messages=[{"role": "user", "content": "长痘用什么"}],
        memories=[],
    )

    # 工具确实执行，参数透传正确
    assert len(registry.executed) == 1
    assert registry.executed[0]["name"] == "search_knowledge_base"
    assert registry.executed[0]["arguments"]["query"] == "祛痘产品"
    assert result.tool_calls == 1
    assert result.llm_calls == 2

    # 回填消息角色正确：assistant(tool_calls) + tool(tool_call_id)
    chat_messages = trace_store.records[0]["chat_messages"]
    assistant_msgs = [m for m in chat_messages if m["role"] == "assistant" and m.get("tool_calls")]
    tool_msgs = [m for m in chat_messages if m["role"] == "tool"]
    assert len(assistant_msgs) == 1
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == assistant_msgs[0]["tool_calls"][0]["id"]
    assert tool_msgs[0]["name"] == "search_knowledge_base"
    # token 聚合：2 次调用
    assert result.token_usage["total_tokens"] == 240


def test_forced_finalization_when_model_keeps_calling_tools():
    """3 轮工具后模型仍请求工具（不收敛）：自然收尾失败 → 无工具强制收尾一次。

    旧实现此时直接 parse 空 content，静默返回空洞察。
    """
    steps = [
        {"tool": [("search_knowledge_base", {"query": "q1"})]},
        {"tool": [("search_knowledge_base", {"query": "q2"})]},
        {"tool": [("get_user_profile", {"user_id": "u_test"})]},
        {"tool": [("get_user_events", {"user_id": "u_test"})]},  # 第 4 次仍要调工具
        {"final": _final_insight(focus="基于已有结果的最终洞察")},
    ]
    agent, _ = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "连续追问"}],
        memories=[],
    )

    assert result.forced_finalization is True
    # 3 个工具轮 + 1 次预算耗尽探测 + 1 次强制收尾
    assert result.llm_calls == MAX_ITER + 2
    assert result.tool_trace[-1].forced_finalization is True
    # 最关键：洞察不是空的
    assert result.raw_insight.get("service_focus") == "基于已有结果的最终洞察"

    # 强制收尾调用不带 tools 参数（脚本 fake 的最后一次调用）
    last_call = agent.client.completions.call_kwargs[-1]
    assert "tools" not in last_call
    # 且追加了"禁止再调用工具"的系统消息
    system_msgs = [m for m in last_call["messages"] if m["role"] == "system"]
    assert any("上限" in m["content"] for m in system_msgs)


def test_natural_finish_after_budget_round_succeeds():
    """3 轮工具后模型自然给出最终答案：不需要强制收尾。"""
    steps = [
        {"tool": [("search_knowledge_base", {"query": "q1"})]},
        {"tool": [("search_knowledge_base", {"query": "q2"})]},
        {"tool": [("get_user_profile", {"user_id": "u_test"})]},
        {"final": _final_insight(focus="自然收尾")},
    ]
    agent, _ = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "连续追问"}],
        memories=[],
    )
    assert result.forced_finalization is False
    assert result.llm_calls == MAX_ITER + 1
    assert result.raw_insight["service_focus"] == "自然收尾"


def test_tool_error_is_backfilled_to_model():
    """工具报错：错误结构化回填（role=tool, ok=false），模型仍可降级作答。"""
    trace_store = FakeTraceStore()
    registry = FakeRegistry(fail_tools={"search_knowledge_base"})
    steps = [
        {"tool": [("search_knowledge_base", {"query": "异常查询"})]},
        {"final": _final_insight(focus="工具失败后的保守洞察")},
    ]
    agent = ToolCallingAgent(
        registry=registry,
        user_id="u_test",
        client=FakeClient(steps),
        model="fake-model",
        trace_store=trace_store,
    )
    result = agent.run(
        messages=[{"role": "user", "content": "触发异常工具"}],
        memories=[],
    )

    # trace 中记录了错误
    execution = result.tool_trace[0].executions[0]
    assert not execution.ok
    assert "模拟工具错误" in execution.error

    # 回填给模型的 tool 消息包含结构化错误，模型仍给出了最终答案
    tool_msg = [m for m in trace_store.records[0]["chat_messages"] if m["role"] == "tool"][0]
    assert '"ok": false' in tool_msg["content"] or '"ok":False' in tool_msg["content"]
    assert result.raw_insight["service_focus"] == "工具失败后的保守洞察"


def test_trace_persisted_with_full_messages_and_result(tmp_path):
    """轨迹通过真实 AgentTraceStore 落盘，并可按 run_id 回放。"""
    from agent_trace_store import AgentTraceStore, format_replay

    trace_store = AgentTraceStore(tmp_path / "traces")
    agent, _ = _make_agent([{"final": _final_insight()}], trace_store=trace_store)
    result = agent.run(
        messages=[{"role": "user", "content": "持久化测试"}],
        memories=[{"id": "m1", "metadata": {"memory_key": "body.skin_type"}}],
    )

    # 按 run_id 回放
    record = trace_store.load_run(result.run_id)
    assert record is not None
    assert record["run_id"] == result.run_id
    assert record["user_id"] == "u_test"
    assert record["model"] == "fake-model"
    assert record["chat_messages"][0]["role"] == "system"
    assert record["input"]["memory_index"][0]["memory_key"] == "body.skin_type"
    assert record["raw_insight"]["service_focus"] == "测试焦点"

    # list_runs 摘要
    summaries = trace_store.list_runs()
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == result.run_id
    assert summaries[0]["token_usage"]["total_tokens"] == 120

    # 回放文本可生成
    replay_text = format_replay(record)
    assert "system" in replay_text and "测试焦点" in replay_text


def test_trace_store_flags_error_runs(tmp_path):
    """iter_errors 能定位工具出错的运行。"""
    from agent_trace_store import AgentTraceStore

    trace_store = AgentTraceStore(tmp_path / "traces")
    registry = FakeRegistry(fail_tools={"search_knowledge_base"})
    agent = ToolCallingAgent(
        registry=registry,
        user_id="u_err",
        client=FakeClient([
            {"tool": [("search_knowledge_base", {"query": "x"})]},
            {"final": _final_insight()},
        ]),
        model="fake-model",
        trace_store=trace_store,
    )
    result = agent.run(messages=[{"role": "user", "content": "err"}], memories=[])
    errors = trace_store.iter_errors()
    assert len(errors) == 1
    assert errors[0]["run_id"] == result.run_id
    assert errors[0]["tool_errors"][0]["name"] == "search_knowledge_base"


def test_multiple_tool_calls_in_one_iteration():
    """一轮内并行调用多个工具：都被执行，结果都回填。"""
    steps = [
        {"tool": [
            ("get_user_profile", {"user_id": "u_test"}),
            ("search_knowledge_base", {"query": "敏感肌", "doc_type": "product"}),
        ]},
        {"final": _final_insight()},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我是敏感肌推荐产品"}],
        memories=[],
    )

    assert result.tool_calls == 2
    executed_names = [c["name"] for c in registry.executed]
    assert executed_names == ["get_user_profile", "search_knowledge_base"]


def test_safe_parse_insight_tolerates_fences_and_prose():
    """最终 JSON 被 markdown 围栏包裹或前后带解释文字时仍能解析。"""
    from tool_calling_agent import _safe_parse_insight

    fenced = '```json\n{"service_focus": "测试", "labels": []}\n```'
    assert _safe_parse_insight(fenced)["service_focus"] == "测试"

    with_prose = '好的，结果如下：\n{"service_focus": "测试2", "labels": []}\n以上。'
    assert _safe_parse_insight(with_prose)["service_focus"] == "测试2"

    assert _safe_parse_insight(None) == {}
    assert _safe_parse_insight("not json at all") == {}


def test_final_assistant_message_is_in_replay(tmp_path):
    """最终 assistant 输出原文也进入 chat_messages，回放时可定位解析失败。"""
    from agent_trace_store import AgentTraceStore

    trace_store = AgentTraceStore(tmp_path / "traces")
    agent, _ = _make_agent([{"final": _final_insight()}], trace_store=trace_store)
    result = agent.run(messages=[{"role": "user", "content": "x"}], memories=[])
    record = trace_store.load_run(result.run_id)
    assert record["chat_messages"][-1]["role"] == "assistant"
    assert "测试焦点" in record["chat_messages"][-1]["content"]


def test_aftersales_gate_forces_reconsideration():
    """模型识别到退货却漏建工单时：护栏触发复议一轮，模型补建工单。"""
    steps = [
        {"final": _final_insight(focus="用户要退货但我漏了工单")},  # 草率收尾
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "return", "reason": "我要退货"})]},
        {"final": _final_insight(focus="用户明确申请退货，已生成工单草稿待人工确认")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我要退货，用了之后脸泛红"}],
        memories=[],
    )

    assert "aftersales_intent_gate" in result.guardrail_hits
    assert len(registry.executed) == 1
    assert registry.executed[0]["name"] == "create_aftersales_ticket_draft"
    assert result.llm_calls == 3
    assert result.raw_insight["service_focus"].startswith("用户明确申请退货")


def test_aftersales_gate_not_triggered_for_policy_inquiry():
    """名词性提及（退货政策）不触发护栏。"""
    agent, registry = _make_agent([{"final": _final_insight(focus="用户咨询退货政策")}])
    result = agent.run(
        messages=[{"role": "user", "content": "想问下你们的退货政策是什么？"}],
        memories=[],
    )
    assert result.guardrail_hits == ()
    assert registry.executed == []


def test_aftersales_gate_fires_at_most_twice():
    """护栏最多复议 2 次（逐级升级）：模型始终不建单时，第三次放行，不死循环。"""
    from tool_calling_agent import MAX_GATE_FIRES

    steps = [
        {"final": _final_insight(focus="坚持不建单1")},
        {"final": _final_insight(focus="坚持不建单2")},
        {"final": _final_insight(focus="仍然不建单")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我要退款，赶紧处理"}],
        memories=[],
    )
    gate_traces = [t for t in result.tool_trace if t.guardrail]
    assert len(gate_traces) == MAX_GATE_FIRES == 2
    assert result.tool_trace[-1].guardrail == ""  # 第 3 次不再拦截
    assert registry.executed == []
    assert result.raw_insight["service_focus"] == "仍然不建单"
    assert result.llm_calls == 3


def test_aftersales_gate_skipped_when_ticket_tool_absent():
    """未注册工单工具时（工具白名单中没有），护栏不触发。"""
    steps = [{"final": _final_insight(focus="无工单环境")}]
    registry = FakeRegistry()
    registry.tool_names = ["search_knowledge_base"]  # 模拟无 ticket_store 的部署
    agent = ToolCallingAgent(
        registry=registry,
        user_id="u_test",
        client=FakeClient(steps),
        model="fake-model",
    )
    result = agent.run(
        messages=[{"role": "user", "content": "我要退货"}],
        memories=[],
    )
    assert result.guardrail_hits == ()


def test_safe_parse_insight_two_json_blocks_picks_final():
    """模型输出两份 JSON（草稿 + 围栏终稿）时，解析最后一份结构完整的。"""
    from tool_calling_agent import _safe_parse_insight

    two_blocks = (
        '{"service_focus": "草稿", "labels": []}\n\n'
        '```json\n{"service_focus": "终稿", "labels": []}\n```'
    )
    assert _safe_parse_insight(two_blocks)["service_focus"] == "终稿"


def test_safe_parse_insight_two_concatenated_json_objects():
    """模型输出两份拼接的 JSON（无围栏，空行分隔）时，取第一份完整对象。

    真实评估中 qwen-plus 偶发：先输出单行压缩 JSON，再输出格式化 JSON。
    """
    from tool_calling_agent import _safe_parse_insight

    two_objects = (
        '{"service_focus":"第一份","labels":[],"service_suggestions":[]}\n\n'
        '{\n  "service_focus": "第二份",\n  "labels": []\n}'
    )
    result = _safe_parse_insight(two_objects)
    assert result["service_focus"] == "第一份"


def test_safe_parse_insight_string_with_braces_does_not_confuse_extractor():
    """JSON 字符串值里包含 { } 时，平衡括号提取器不误判对象结束。"""
    from tool_calling_agent import _safe_parse_insight

    content = '{"service_focus":"值里有 { 大括号 }","labels":[]}'
    result = _safe_parse_insight(content)
    assert result["service_focus"] == "值里有 { 大括号 }"


def test_ticket_type_mismatch_is_rejected_then_retried():
    """用户明确说退货但模型建 allergy_case：执行前拦截，结构化驳回，模型改回 return。"""
    steps = [
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "allergy_case", "reason": "泛红"})]},
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "return", "reason": "泛红；用户选择退货"})]},
        {"final": _final_insight(focus="已建退货工单草稿")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我要退货，昨天用了你们的面膜之后脸一直泛红"}],
        memories=[],
    )

    # 只执行了第二次（类型正确的）调用
    assert len(registry.executed) == 1
    assert registry.executed[0]["arguments"]["ticket_type"] == "return"
    # 第一次的结构化驳回留在轨迹里
    first_exec = result.tool_trace[0].executions[0]
    assert first_exec.ok is False
    assert "前置校验拦截" in first_exec.error
    assert "ticket_type=return" in first_exec.error
    assert result.llm_calls == 3
    # 驳回轮不占工具预算，没有触发强制收尾
    assert result.forced_finalization is False


def test_ticket_type_check_rejects_only_once():
    """同一种错配只驳回一次：模型坚持 allergy_case 时放行（避免死循环，人工审核兜底）。"""
    steps = [
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "allergy_case", "reason": "泛红"})]},
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "allergy_case", "reason": "泛红"})]},
        {"final": _final_insight(focus="坚持建过敏单")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我要退货，脸泛红"}],
        memories=[],
    )
    assert len(registry.executed) == 1
    assert registry.executed[0]["arguments"]["ticket_type"] == "allergy_case"
    assert result.tool_trace[0].executions[0].ok is False
    assert result.tool_trace[1].executions[0].ok is True


def test_ticket_type_allows_allergy_case_without_explicit_channel():
    """用户只投诉过敏症状、未选择退换渠道时，allergy_case 是合理推断，直接执行。"""
    steps = [
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "allergy_case",
                    "reason": "烂脸、又红又痒"})]},
        {"final": _final_insight(focus="过敏投诉工单")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "用了你们的产品脸都烂了，过敏很严重，又红又痒"}],
        memories=[],
    )
    assert len(registry.executed) == 1
    assert result.tool_trace[0].executions[0].ok is True


def test_ticket_type_check_keeps_alignment_with_parallel_calls():
    """同一轮并行调用：工单被驳回、其他工具正常执行，tool 回填顺序与 id 不错位。

    模型下一轮按驳回提示用 return 重建，流程正常收尾（护栏门不重复触发）。
    """
    steps = [
        {"tool": [
            ("create_aftersales_ticket_draft",
             {"user_id": "u_test", "ticket_type": "allergy_case", "reason": "泛红"}),
            ("search_knowledge_base", {"query": "退货政策", "doc_type": "policy"}),
        ]},
        {"tool": [("create_aftersales_ticket_draft",
                   {"user_id": "u_test", "ticket_type": "return", "reason": "泛红；用户选择退货"})]},
        {"final": _final_insight(focus="已建退货工单草稿")},
    ]
    agent, registry = _make_agent(steps)
    result = agent.run(
        messages=[{"role": "user", "content": "我要退货，脸泛红，顺便问下政策"}],
        memories=[],
    )
    # 检索 + 第二次类型正确的工单真正执行（被驳回的第一次不落库）
    assert len(registry.executed) == 2
    assert registry.executed[0]["name"] == "search_knowledge_base"
    assert registry.executed[1]["name"] == "create_aftersales_ticket_draft"
    assert registry.executed[1]["arguments"]["ticket_type"] == "return"
    execs = result.tool_trace[0].executions
    assert execs[0].name == "create_aftersales_ticket_draft"
    assert execs[0].ok is False
    assert execs[1].name == "search_knowledge_base"
    assert execs[1].ok is True
    assert result.guardrail_hits == ()
