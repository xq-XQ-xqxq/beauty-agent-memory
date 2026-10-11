"""工具调用 Agent（LLM 决策 + 工具执行循环）。

工作流（Qwen 兼容 OpenAI Function Calling 协议）：
1. LLM 第 1 次调用（含工具 schema）→ 模型决定调用哪些工具、用什么参数
2. 程序执行所有工具调用，把结果回填给 LLM
3. LLM 第 2 次调用（带工具结果）→ 生成最终的 service insight
   - 若第 2 次仍请求工具，则继续循环（上限 MAX_ITER=3，防失控）

设计要点：
- 工具调用循环可控：MAX_ITER=3，每次循环 1 次 LLM 调用 + N 个并行工具执行
- 工具失败不中断流程：错误信息回填给 LLM，让模型自己决定降级方案
- 工具调用轨迹全程记录（含参数、结果、耗时），供 UI 展示
- 复用 ServiceInsightGenerator 的 _validate_insight 校验，保持输出一致性
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tool_registry import ToolCallResult, ToolRegistry

load_dotenv(Path(__file__).parent / ".env")


MAX_ITER = 3  # 工具调用循环上限（防失控；通常 1 次足够，2 次为补充，3 次为兜底）
MAX_OUTPUT_TOKENS = 8192  # 显式输出上限：复杂标签场景可达 4500+ token，截断会损坏 JSON

# 售后意图兜底门（aftersales_intent_gate）：
# 当模型在最终答案里识别出售后诉求却没有建工单时，强制"复议"一轮。
# 只匹配【明确的施为性表达】，不匹配"退货政策/退换货规则"等名词性提及。
_AFTERSALES_INTENT_PATTERNS = [
    (re.compile(r"我要退货|想退货|申请退货|帮我.{0,6}退货|退了吧|我退了"), "return"),
    (re.compile(r"我要换货|想换货|申请换货|帮我.{0,6}换货|换一瓶|换一款"), "exchange"),
    (re.compile(r"给我退款|我要退款|直接.{0,6}退款|申请退款"), "refund"),
    (re.compile(r"烂脸|过敏很严重|又红又痒"), "allergy_case"),
]

GUARDRAIL_AFTERSALES_GATE = "aftersales_intent_gate"
MAX_GATE_FIRES = 2  # 售后兜底门最多复议次数（逐级升级；真实评估发现 1 次常被模型无视）


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallingTrace:
    """单轮工具调用决策的完整轨迹。"""

    iteration: int
    decisions: list[dict[str, Any]]  # LLM 决定的 tool calls（原始格式）
    executions: list[ToolCallResult]  # 实际执行结果
    latency_ms: int
    token_usage: dict[str, int] = field(default_factory=dict)  # 本轮 LLM token
    forced_finalization: bool = False  # 本轮是否为"超限后强制收尾"调用
    guardrail: str = ""  # 本轮触发的护栏名称（空串表示无）


@dataclass(frozen=True)
class AgentRunResult:
    """Agent 一次运行的完整输出（含洞察 + 工具轨迹）。"""

    raw_insight: dict[str, Any]
    tool_trace: list[ToolCallingTrace]
    total_latency_ms: int
    llm_calls: int
    tool_calls: int
    token_usage: dict[str, int] = field(default_factory=dict)
    forced_finalization: bool = False  # 是否触发了 MAX_ITER 强制收尾
    run_id: str = ""
    guardrail_hits: tuple[str, ...] = ()  # 本次运行触发的护栏名称（如 aftersales_intent_gate）

    @property
    def has_tool_calls(self) -> bool:
        return self.tool_calls > 0

    def all_tool_results(self) -> list[ToolCallResult]:
        results: list[ToolCallResult] = []
        for trace in self.tool_trace:
            results.extend(trace.executions)
        return results

    def to_display(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_latency_ms": self.total_latency_ms,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "token_usage": self.token_usage,
            "forced_finalization": self.forced_finalization,
            "iterations": len(self.tool_trace),
            "guardrail_hits": list(self.guardrail_hits),
            "traces": [
                {
                    "iteration": t.iteration,
                    "latency_ms": t.latency_ms,
                    "token_usage": t.token_usage,
                    "forced_finalization": t.forced_finalization,
                    "guardrail": t.guardrail,
                    "decisions": t.decisions,
                    "executions": [e.to_display() for e in t.executions],
                }
                for t in self.tool_trace
            ],
        }


# ---------------------------------------------------------------------------
# Tool calling agent
# ---------------------------------------------------------------------------

class ToolCallingAgent:
    """LLM 工具调用 Agent。

    用法：
        registry = ToolRegistry(store=store, retriever=retriever)
        agent = ToolCallingAgent(registry=registry, user_id="u123")
        result = agent.run(messages, memories)
        insight = result.raw_insight
        trace = result.tool_trace
    """

    def __init__(
        self,
        registry: ToolRegistry,
        user_id: str,
        client: Any | None = None,
        model: str | None = None,
        trace_store: Any | None = None,
    ) -> None:
        self.registry = registry
        self.user_id = user_id
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get(
                "DASHSCOPE_API_KEY", ""
            )
            base_url = os.environ.get("MEM0_LLM_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model or os.environ.get("MEM0_LLM_MODEL", "qwen-plus")
        # 轨迹落盘存储（可选）：传入 AgentTraceStore 后每次 run 都会持久化完整消息序列
        self.trace_store = trace_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        conversation_summary: str | None = None,
    ) -> AgentRunResult:
        """运行工具调用循环，返回洞察 + 工具轨迹。"""
        started_total = perf_counter()

        # 准备初始消息（系统提示 + 上下文）
        context_prompt = self._build_context_prompt(messages, memories, conversation_summary)
        chat_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context_prompt},
        ]

        tool_schemas = self.registry.openai_schemas()
        trace: list[ToolCallingTrace] = []
        llm_calls = 0
        token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        run_id = uuid.uuid4().hex[:16]

        def _accumulate_usage(resp: Any) -> dict[str, int]:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return {}
            piece = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            for key, value in piece.items():
                token_totals[key] = token_totals.get(key, 0) + value
            return piece

        # 工具调用循环
        # 注意：不强制 response_format=json_object，否则与 tool_calls 协议冲突，
        # LLM 会把"调用工具"写进 suggestions 文本里而非真正发起 tool_calls。
        # 最终洞察的 JSON 格式由系统提示约束 + 兜底解析保证。
        message = None
        gate_fires = 0
        iteration = 0
        tool_rounds_used = 0  # 实际执行工具的轮次（护栏复议轮不占用该预算）
        rejected_ticket_keys: set[str] = set()  # 已驳回过的工单类型错配（只拦一次，防死循环）
        while True:
            iteration += 1
            started_iter = perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                tools=tool_schemas,
                tool_choice="auto",
                messages=chat_messages,
            )
            llm_calls += 1
            iter_usage = _accumulate_usage(response)
            message = response.choices[0].message

            # 没有 tool_calls → 检查护栏，否则视为最终答案
            if not message.tool_calls:
                # 护栏：售后意图兜底门（最多复议 MAX_GATE_FIRES 次，逐级升级）
                ticket_type = (
                    self._detect_aftersales_intent(chat_messages)
                    if gate_fires < MAX_GATE_FIRES
                    else None
                )
                if ticket_type:
                    gate_fires += 1
                    trace.append(ToolCallingTrace(
                        iteration=iteration,
                        decisions=[],
                        executions=[],
                        latency_ms=round((perf_counter() - started_iter) * 1000),
                        token_usage=iter_usage,
                        guardrail=GUARDRAIL_AFTERSALES_GATE,
                    ))
                    # 保留模型这轮"草率收尾"的原文，再要求复议（便于回放复盘）
                    chat_messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                    })
                    if gate_fires == 1:
                        gate_content = (
                            f"【护栏 {GUARDRAIL_AFTERSALES_GATE}】用户消息中包含明确的施为性售后诉求"
                            f"（疑似 ticket_type={ticket_type}），但你尚未成功调用 "
                            "create_aftersales_ticket_draft。这不是政策咨询（政策咨询不会触发本护栏）。"
                            "请立即调用 create_aftersales_ticket_draft 创建待人工审核的草稿，"
                            "然后再输出最终 JSON。"
                        )
                    else:
                        gate_content = (
                            f"【护栏 {GUARDRAIL_AFTERSALES_GATE} 第 {gate_fires} 次复议】"
                            "用户的售后诉求仍未建工单。禁止在未调用 "
                            f"create_aftersales_ticket_draft（ticket_type={ticket_type}）的情况下"
                            "输出最终 JSON。请现在只发起该工具调用。"
                        )
                    chat_messages.append({"role": "system", "content": gate_content})
                    continue

                trace.append(ToolCallingTrace(
                    iteration=iteration,
                    decisions=[],
                    executions=[],
                    latency_ms=round((perf_counter() - started_iter) * 1000),
                    token_usage=iter_usage,
                ))
                break

            # 工具轮次预算耗尽：不再执行，留给循环后的强制收尾
            if tool_rounds_used >= MAX_ITER:
                # 记录模型在预算耗尽时仍想调用的工具（只记录不执行，便于复盘）
                exhausted_decisions = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": _safe_parse_arguments(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]
                trace.append(ToolCallingTrace(
                    iteration=iteration,
                    decisions=exhausted_decisions,
                    executions=[],
                    latency_ms=round((perf_counter() - started_iter) * 1000),
                    token_usage=iter_usage,
                ))
                break

            # 解析 LLM 决定的工具调用
            decisions = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": _safe_parse_arguments(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

            # 执行工具调用（工单类型前置语义校验不通过的，不执行、结构化驳回让模型重试）
            executions: list[ToolCallResult | None] = [None] * len(decisions)
            batch_requests: list[tuple[int, dict[str, Any]]] = []
            for idx, decision in enumerate(decisions):
                semantic_error = self._check_ticket_type(
                    chat_messages, decision, rejected_ticket_keys
                )
                if semantic_error is not None:
                    executions[idx] = ToolCallResult(
                        name=decision["name"],
                        arguments=decision["arguments"],
                        result=None,
                        latency_ms=0,
                        error=semantic_error,
                    )
                else:
                    batch_requests.append(
                        (idx, {"name": decision["name"], "arguments": decision["arguments"]})
                    )
            if batch_requests:
                tool_rounds_used += 1
                batch_results = self.registry.execute_batch(
                    [request for _, request in batch_requests]
                )
                for (idx, _), batch_result in zip(batch_requests, batch_results):
                    executions[idx] = batch_result
            executions = [e for e in executions if e is not None]

            iter_latency = round((perf_counter() - started_iter) * 1000)
            trace.append(ToolCallingTrace(
                iteration=iteration,
                decisions=decisions,
                executions=executions,
                latency_ms=iter_latency,
                token_usage=iter_usage,
            ))

            # 把 assistant 的 tool_calls 消息 + 工具结果回填给 LLM
            chat_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })
            for decision, execution in zip(decisions, executions):
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": decision["id"],
                    "name": decision["name"],
                    "content": _format_tool_result(execution),
                })

        # 停止条件兜底：循环耗尽时模型仍在请求工具（异常/不收敛情况）。
        # 不能直接 parse 空 content（那会静默返回空洞察），而是追加明确指令、
        # 不带 tools 再调用一次，强制模型基于已有工具结果输出最终 JSON。
        forced_finalization = False
        if message is not None and message.tool_calls:
            forced_finalization = True
            chat_messages.append({
                "role": "system",
                "content": (
                    f"工具调用轮次已达上限（{MAX_ITER} 轮）。禁止再调用任何工具，"
                    "请立即基于以上已返回的工具结果输出最终洞察 JSON，"
                    "格式必须与系统提示一致，信息不足的字段返回空数组。"
                ),
            })
            started_final = perf_counter()
            final_response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=chat_messages,
            )
            llm_calls += 1
            final_usage = _accumulate_usage(final_response)
            message = final_response.choices[0].message
            trace.append(ToolCallingTrace(
                iteration=MAX_ITER + 1,
                decisions=[],
                executions=[],
                latency_ms=round((perf_counter() - started_final) * 1000),
                token_usage=final_usage,
                forced_finalization=True,
            ))

        # 解析最终洞察
        raw_insight = _safe_parse_insight(message.content if message else None)

        # 把最终 assistant 消息补进轨迹（回放失败案例时需要看到模型最终输出原文）
        if message is not None:
            chat_messages.append({
                "role": "assistant",
                "content": message.content or "",
            })

        total_latency_ms = round((perf_counter() - started_total) * 1000)
        tool_call_count = sum(len(t.executions) for t in trace)

        result = AgentRunResult(
            raw_insight=raw_insight,
            tool_trace=trace,
            total_latency_ms=total_latency_ms,
            llm_calls=llm_calls,
            tool_calls=tool_call_count,
            token_usage=token_totals,
            forced_finalization=forced_finalization,
            run_id=run_id,
            guardrail_hits=tuple(dict.fromkeys(
                t.guardrail for t in trace if t.guardrail
            )),
        )

        # 轨迹持久化（完整消息序列 + 指标），供失败案例回放
        if self.trace_store is not None:
            try:
                self.trace_store.save_run(
                    run_id=run_id,
                    user_id=self.user_id,
                    model=self.model,
                    messages=messages,
                    memories=memories,
                    chat_messages=chat_messages,
                    result=result,
                )
            except Exception as exc:
                # 落盘失败不影响主流程
                print(f"[ToolCallingAgent] trace 落盘失败: {type(exc).__name__}: {exc}")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_ticket_type(
        self,
        chat_messages: list[dict[str, Any]],
        decision: dict[str, Any],
        rejected_keys: set[str],
    ) -> str | None:
        """工单类型执行前语义校验：用户明确选择的渠道优先于模型推断。

        真实评估发现：用户说"我要退货 + 泛红症状"时，模型 3/3 次把工单建成
        allergy_case（症状显著导致忽略用户明确选择），提示词规则无效。
        因此在工具执行层加确定性兜底：

        - 只认 return/exchange/refund 的施为性表达（"我要退货/想换一瓶/直接退款"）
        - 过敏症状短语不参与（用户没选渠道时，模型判 allergy_case 是合理推断）
        - 同一种错配只驳回一次；模型坚持时放行，避免死循环（最终由人工审核兜底）
        """
        if decision.get("name") != "create_aftersales_ticket_draft":
            return None
        chosen = str(decision.get("arguments", {}).get("ticket_type", ""))
        user_text = "\n".join(
            str(m.get("content", ""))
            for m in chat_messages
            if m.get("role") == "user"
        )
        expected = None
        for pattern, ticket_type in _AFTERSALES_INTENT_PATTERNS:
            if ticket_type in {"return", "exchange", "refund"} and pattern.search(user_text):
                expected = ticket_type
                break
        if expected is None or chosen == expected:
            return None
        dedupe_key = f"{chosen}->{expected}"
        if dedupe_key in rejected_keys:
            return None
        rejected_keys.add(dedupe_key)
        return (
            f"【前置校验拦截】用户消息中包含明确的 {expected} 诉求（施为性表达），"
            f"但本次 ticket_type={chosen or '（空）'} 与用户明确选择不一致。"
            f"请以用户明确选择为准，使用 ticket_type={expected} 重新调用 "
            f"create_aftersales_ticket_draft；泛红/刺痛等症状请写入 reason 字段，"
            f"不要据此改变工单类型。"
        )

    def _detect_aftersales_intent(self, chat_messages: list[dict[str, Any]]) -> str | None:
        """售后意图兜底门：检测"明确施为性售后诉求"。

        - 只扫描 user 消息（系统提示里的规则文本不会误伤）
        - 只匹配"我要退货/换一瓶/给我退款/烂脸"等明确表达，
          不匹配"退货政策/退换货规则"等名词性提及
        - 已建过工单、或工单工具未注册时不触发
        """
        if "create_aftersales_ticket_draft" not in self.registry.tool_names:
            return None
        # 已有"成功创建"的工单才静默；被前置校验驳回（ok=false）的不算建单
        for m in chat_messages:
            if m.get("role") == "tool" and m.get("name") == "create_aftersales_ticket_draft":
                try:
                    payload = json.loads(str(m.get("content", "")))
                    if isinstance(payload, dict) and payload.get("ok") is True:
                        return None
                except (json.JSONDecodeError, TypeError):
                    continue
        user_text = "\n".join(
            str(m.get("content", "")) for m in chat_messages if m.get("role") == "user"
        )
        for pattern, ticket_type in _AFTERSALES_INTENT_PATTERNS:
            if pattern.search(user_text):
                return ticket_type
        return None

    def _build_context_prompt(
        self,
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        conversation_summary: str | None,
    ) -> str:
        """组装给 LLM 的上下文（对话 + 记忆索引，不含知识库）。

        与被动注入不同：Agent 模式只给 LLM 提供「记忆索引」而非完整记忆，
        让 LLM 通过 get_user_profile / get_user_events 工具按需检索。
        """
        # 1. 当前对话（最近 10 条）
        recent = [
            {"id": f"msg_{i}", "role": m.get("role", ""), "content": m.get("content", "")}
            for i, m in enumerate(messages[-10:], start=1)
            if m.get("content", "").strip()
        ]

        # 2. 记忆索引（只暴露 key 列表，不暴露内容；让 LLM 主动调工具取实际值）
        # 设计动机：被动注入完整记忆会让 LLM "懒"得不调工具；
        # 只给 key 列表，LLM 必须通过 get_user_profile / get_user_events 读取详情。
        memory_index = [
            {
                "memory_id": str(m.get("id", "")),
                "memory_key": str(m.get("metadata", {}).get("memory_key", "")),
            }
            for m in memories[:20]  # 最多展示 20 条索引
        ]

        prompt = {
            "current_user_id": self.user_id,
            "recent_conversation": recent,
            "memory_index": memory_index,  # 只给预览，详细内容用工具取
            "available_tools_hint": _TOOL_HINT,
        }
        if conversation_summary:
            prompt["conversation_summary"] = conversation_summary

        return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """你是美妆电商客服的洞察助手（Agent 模式）。

工作方式：你不能凭对话内容直接编写产品/成分/政策结论。你必须**先调用工具取证，再基于工具返回的事实输出洞察**。
仅凭对话就能直接作答的场景只有：纯问候闲聊、要求用户澄清信息、安抚情绪。其余服务场景一律先调工具。

⚠ 关键约束：memory_index 只列出了 memory_id 和 memory_key（如 body.skin_type、experience.product_reaction），**不包含实际值**。
要读取用户的实际肤质、敏感度、历史事件等，必须主动调用 get_user_profile 或 get_user_events。禁止根据 memory_key 名称猜测值。

## 强制路由表（命中意图必须调用对应工具，不允许跳过）

| 用户意图（识别到即触发） | 必须调用的工具 |
|---|---|
| 询问具体产品（功效/适合肤质/有没有货/推荐产品） | search_knowledge_base，doc_type=product |
| 询问售后政策（七天无理由/退换货规则/运费/时效） | search_knowledge_base，doc_type=policy |
| 询问某成分能不能用/刺激吗/含量多少/安全吗 | check_ingredient_safety（**不得用 search_knowledge_base 代替**），且 memory_index 无肤质时先调 get_user_profile |
| 需要用户肤质/敏感度等长期属性 | get_user_profile |
| 查询历史投诉/历史记录，或用户情绪激动地抱怨 | get_user_events（event_type 必须取工具 schema 中的合法枚举值） |
| 明确提出退货 / 换货 / 退款 | create_aftersales_ticket_draft，ticket_type 分别为 return / exchange / refund |
| 明确投诉使用产品后过敏、烂脸、红痒等 | create_aftersales_ticket_draft，ticket_type=allergy_case |

ticket_type 判定优先级：用户**明确说出**"退货/换货/退款"时，以用户选择的类型为准（即使同时描述了泛红、刺痛等症状，仍归 return/exchange/refund）；
allergy_case 仅用于用户没有明确选择退换/退款、只投诉使用后过敏症状的情况。

一条消息可同时命中多个意图，应在同一轮并行发起多个工具调用；但**同一轮禁止用相同参数重复调用同一个工具**。
不允许出现"知道该调工具但跳过、直接写结论"的行为。

## 工具返回后，整合信息生成最终洞察
- service_focus：一句话概括客服现在主要处理什么
- labels：0-5 个标签，每个含 title/content/priority/source_ids
- service_suggestions：0-2 个建议，每个含 action/content/basis_ids

## 规则
1. 只保留对当前回复有直接帮助的信息，按重要程度排列。当前困扰/需求/情绪优先于长期偏好。
2. priority：urgent（安全/强烈情绪/投诉）、high、normal。不夸大。
3. 【反幻觉】source_ids 只能引用真实出现在你上下文中的 ID：
   - msg_1, msg_2...：来自对话消息编号
   - prod_001 / ing_001 / faq_001 / policy_001...：**必须是 search_knowledge_base 实际返回的 doc_id**，禁止凭产品名猜测或编造
   - memory_id：memory_index 中的 memory_id 字段值
   - EVT_xxx：必须是 get_user_events 实际返回的 event_id
   - body.skin_type, body.sensitivity...：必须是 get_user_profile 实际返回的 profile_key
   - TKT_xxx：必须是 create_aftersales_ticket_draft 实际返回的 ticket_id
   直接使用工具返回的 ID 字符串，不加前缀。工具没有返回的内容，标签里不得声称已核实。
   禁用词：可能、潜在、意味着、耐受度、致敏、风险。
4. service_focus：一句话概括客服现在主要处理什么。
5. 建议 action 只能选：empathize、clarify、confirm_need、avoid_repeat、safety_attention、follow_up。不指导消费者停用/就医/用药，不承诺退款赔偿。
6. 无可靠依据时返回空数组。
7. 工具失败时降级：基于现有上下文给出保守洞察，不要编造工具未返回的信息，并可换一种方式重试一次。
8. 售后工单是唯一写操作：只能创建待人工审核的草稿，你无权批准/驳回。
   草稿创建后在标签中标注"已生成工单草稿待人工确认"，不得承诺退款金额、时效或处理结果。
   用户只是询问政策、抱怨但未明确提出诉求时，不要建工单。

最终自检（输出 JSON 前必须逐条确认）：
- 对话是否命中强制路由表？命中了却没调对应工具 → 立刻先调工具，不要输出 JSON。
- labels 中的每个 doc_id / EVT_xxx / TKT_xxx 是否都有工具返回记录？没有就删掉该标签或改用 msg_x。

JSON 输出格式（仅在工具取证完成后输出）：
{"service_focus":"...","labels":[{"title":"...","content":"...","priority":"urgent|high|normal","source_ids":["..."]}],"service_suggestions":[{"action":"...","content":"...","basis_ids":["..."]}]}
"""

_TOOL_HINT = """可用工具说明（按强制路由表使用，不是"可选"）：
- search_knowledge_base(query, doc_type?, top_k?): 检索产品(product)/成分(ingredient)/FAQ(faq)/政策(policy)文档
- get_user_profile(user_id, profile_key?): 读取用户画像（肤质、敏感度、偏好、目标）
- get_user_events(user_id, event_type?, limit?): 读取用户事件流；查投诉/升级意图用 dissatisfaction.escalation_intent，查不满原因用 dissatisfaction.reason
- check_ingredient_safety(ingredient_name, skin_type?): 查成分安全性，叠加肤质约束（成分能不能用/刺激吗/含量多少，必须走这个工具）
- create_aftersales_ticket_draft(user_id, ticket_type, reason, ...): 【写】创建售后工单草稿（return 退货 / exchange 换货 / refund 退款 / allergy_case 过敏投诉），待人工审核

再次强调：
- 产品/推荐问题 → search_knowledge_base(doc_type=product)；政策问题 → doc_type=policy
- 成分安全问题 → check_ingredient_safety（不要用 search_knowledge_base 代替），无肤质记忆先 get_user_profile
- 查历史投诉/强烈抱怨 → get_user_events
- 明确退货/换货/退款/使用后过敏 → create_aftersales_ticket_draft；仅咨询政策、单纯抱怨不调
- 只有纯问候/闲聊/请求澄清可以不调任何工具
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse_arguments(arguments: str | None) -> dict[str, Any]:
    """解析 LLM 输出的工具参数（容错）。"""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _safe_parse_insight(content: str | None) -> dict[str, Any]:
    """解析 LLM 最终输出的 JSON（容错）。

    LLM 常见偏离：
    - 包 markdown 代码围栏、JSON 前后带解释文字
    - 先输出一份草稿 JSON，再用 ```json 围栏输出一份格式化终稿
    - 输出两份拼接的 JSON 对象（单行压缩版 + 格式化版，中间用空行分隔）
    """
    if not content:
        return {}
    text = content.strip()

    candidates: list[str] = []
    # 1. 所有围栏块（模型输出多份 JSON 时，最后一份通常是终稿，倒序尝试）
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.extend(b.strip() for b in reversed(fenced_blocks) if b.strip())
    # 2. 整体去掉单层围栏
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        candidates.append(stripped)
    # 3. 原文整体
    candidates.append(text)
    # 4. 第一个完整 JSON 对象（处理"两份 JSON 拼接"的情况，取第一份）
    first_obj = _extract_first_json_object(text)
    if first_obj is not None:
        candidates.append(first_obj)
    # 5. 退一步：截取第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    fallback: dict[str, Any] = {}
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            # 优先返回结构完整的洞察；结构残缺的留作兜底
            if "service_focus" in parsed:
                return parsed
            if not fallback:
                fallback = parsed
    return fallback


def _extract_first_json_object(text: str) -> str | None:
    """从文本中提取第一个完整的 JSON 对象（平衡大括号）。

    用于处理模型输出两份拼接 JSON 的情况：只取第一份，
    避免"首个 { 到末个 }"把两份都包进来导致解析失败。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _format_tool_result(result: ToolCallResult) -> str:
    """把工具结果格式化为 LLM 可读的字符串（含错误信息）。"""
    if not result.ok:
        return json.dumps(
            {"error": result.error, "ok": False},
            ensure_ascii=False,
        )
    return json.dumps(
        {"ok": True, "data": result.result},
        ensure_ascii=False,
    )
