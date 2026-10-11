"""Three-column customer-support workbench with Profile+Event v5 pipeline."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st

from label_engine import ServiceInsight, ServiceInsightGenerator, ServiceLabel, ServiceSuggestion
from profile_event_pipeline import ProfileEventPipeline
from profile_event_store import ProfileEventStore, render_event_text, render_profile_text
from security_guard import SecurityGuard


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    error_log = os.path.join(os.path.dirname(__file__), "data", "app_errors.log")
    os.makedirs(os.path.dirname(error_log), exist_ok=True)
    file_handler = logging.FileHandler(error_log, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)


# 全局安全守卫单例（PII 检测 + Prompt 注入检测）
_SECURITY_GUARD: SecurityGuard | None = None


def _guard() -> SecurityGuard:
    global _SECURITY_GUARD
    if _SECURITY_GUARD is None:
        _SECURITY_GUARD = SecurityGuard()
    return _SECURITY_GUARD


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="共情记忆工作台",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CUSTOM_CSS = """
<style>
/* 全局字体 */
.stApp { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }

/* 顶部标题渐变 */
.app-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0;
}
.app-subtitle {
    color: #888;
    font-size: 0.85rem;
    margin-top: -4px;
}

/* 卡片阴影 */
.stContainer:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
    border-radius: 12px;
}

/* Profile 卡片 */
.profile-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
}
.profile-card .key {
    color: #4f46e5;
    font-weight: 600;
    font-size: 0.8rem;
}
.profile-card .key .raw {
    float: right;
    color: #a5b4fc;
    font-weight: 400;
    font-size: 0.66rem;
    font-family: ui-monospace, Consolas, monospace;
}
.profile-card .val {
    color: #1e293b;
    font-size: 0.85rem;
}

/* Event 时间线 */
.event-card {
    border-left: 3px solid #f59e0b;
    padding: 6px 10px;
    margin-bottom: 4px;
    background: #fffbeb;
    border-radius: 0 8px 8px 0;
}
.event-card .type {
    color: #d97706;
    font-weight: 600;
    font-size: 0.75rem;
}
.event-card .desc {
    color: #475569;
    font-size: 0.82rem;
}

/* 性能指标 */
.metric-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
}
.metric-speed { background: #dcfce7; color: #166534; }
.metric-token { background: #dbeafe; color: #1e40af; }
.metric-llm   { background: #fef3c7; color: #92400e; }

/* 分析按钮居中 */
.analyze-btn { display: flex; justify-content: center; margin: 8px 0; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "user", "content": "你好，我用了面膜以后全脸发红，还起了小疹子，有点痒。"},
    {"role": "assistant", "content": "方便说下多久出现不适，以及您的肤质吗？"},
    {"role": "user", "content": "大概用了3小时后开始红的，我是混合敏感肌。这个还能继续用吗？"},
]


def new_session_record(user_id: str, conversation_id: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "title": "新会话",
        "messages": [],
        "analysis": None,
        "insight": None,
        "write_result": None,
        "processing_error": None,
        "event_id": f"EVT_{conversation_id}",
        "scene_major": "通用咨询",
        "scene_minor": "待识别",
        "agent_mode": False,  # 工具调用（Function Calling）开关
        "security_scans": [],  # 每条消息的安全扫描结果（PII / 注入）
    }


def initialize_state() -> None:
    if "sessions" not in st.session_state:
        default = new_session_record("demo_customer_001", "DEMO_001")
        default["messages"] = list(st.session_state.get("messages", []))
        default["analysis"] = st.session_state.get("analysis")
        default["insight"] = st.session_state.get("insight")
        default["write_result"] = st.session_state.get("write_result")
        default["processing_error"] = st.session_state.get("processing_error")
        st.session_state.sessions = {default["conversation_id"]: default}
    st.session_state.setdefault("active_conversation_id", next(iter(st.session_state.sessions)))
    st.session_state.setdefault("show_new_session_form", False)


def active_session() -> dict[str, Any]:
    active_id = st.session_state.active_conversation_id
    return st.session_state.sessions[active_id]


def save_active_session(session: dict[str, Any]) -> None:
    st.session_state.sessions[session["conversation_id"]] = session


# ---------------------------------------------------------------------------
# v5 Profile+Event pipeline (replaces old CustomerMemoryService)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(os.path.dirname(__file__)) / "data"
_DB_PATH = _DATA_DIR / "profile_event.db"


@st.cache_resource
def pe_store() -> ProfileEventStore:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return ProfileEventStore(_DB_PATH)


@st.cache_resource
def pe_pipeline() -> ProfileEventPipeline:
    return ProfileEventPipeline(store=pe_store())


def current_context() -> dict[str, str]:
    session = active_session()
    return {
        "user_id": session["user_id"],
        "conversation_id": session["conversation_id"],
        "scene_major": session["scene_major"],
        "scene_minor": session["scene_minor"],
        "occurred_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def create_session(user_id: str) -> None:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    conversation_id = f"CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    session = new_session_record(user_id, conversation_id)
    st.session_state.sessions[conversation_id] = session
    st.session_state.active_conversation_id = conversation_id


def show_new_session_form() -> None:
    st.session_state.show_new_session_form = True


def select_session(conversation_id: str) -> None:
    st.session_state.active_conversation_id = conversation_id


def clear_current_session() -> None:
    session = active_session()
    session.update(
        messages=[], security_scans=[], analysis=None, insight=None, write_result=None,
        processing_error=None, title="新会话",
    )
    save_active_session(session)


def reset_user_data() -> None:
    """Clear Profile+Event data for the current user (for testing)."""
    session = active_session()
    store = pe_store()
    store.reset_user(session["user_id"])
    session.update(analysis=None, insight=None, write_result=None, processing_error=None)
    save_active_session(session)


def load_sample() -> None:
    session = active_session()
    # 对示例消息也执行安全扫描
    guard = _guard()
    safe_messages = []
    security_scans = []
    for item in SAMPLE_MESSAGES:
        scan = guard.scan(item["content"])
        safe_messages.append({"role": item["role"], "content": scan.redacted_text})
        security_scans.append({
            "role": item["role"],
            "pii_count": len(scan.pii_matches),
            "pii_types": [m.label for m in scan.pii_matches],
            "pii_details": [
                {"type": m.pii_type, "label": m.label, "masked": m.masked}
                for m in scan.pii_matches
            ],
            "injection_count": len(scan.injection_matches),
            "injection_severities": [m.severity for m in scan.injection_matches],
            "injection_details": [
                {"name": m.pattern_name, "label": m.label, "severity": m.severity}
                for m in scan.injection_matches
            ],
        })
    session.update(
        messages=safe_messages,
        security_scans=security_scans,
        analysis=None, insight=None, write_result=None, processing_error=None,
        title="面膜使用后不适", scene_major="不良反应", scene_minor="泛红刺痒",
    )
    save_active_session(session)


def append_message(role: str, content: str) -> None:
    content = content.strip()
    if not content:
        return
    session = active_session()
    # 安全扫描：PII 检测 + 脱敏 + Prompt 注入检测
    scan = _guard().scan(content)
    safe_content = scan.redacted_text
    session["messages"].append({"role": role, "content": safe_content})
    session["security_scans"].append({
        "role": role,
        "pii_count": len(scan.pii_matches),
        "pii_types": [m.label for m in scan.pii_matches],
        "pii_details": [
            {"type": m.pii_type, "label": m.label, "masked": m.masked}
            for m in scan.pii_matches
        ],
        "injection_count": len(scan.injection_matches),
        "injection_severities": [m.severity for m in scan.injection_matches],
        "injection_details": [
            {"name": m.pattern_name, "label": m.label, "severity": m.severity}
            for m in scan.injection_matches
        ],
    })
    session["analysis"] = None
    session["insight"] = None
    session["processing_error"] = None
    if session["title"] == "新会话" and role == "user":
        session["title"] = safe_content[:14] + ("..." if len(safe_content) > 14 else "")
    save_active_session(session)


# ---------------------------------------------------------------------------
# Core analysis (v5 pipeline + ServiceInsight)
# ---------------------------------------------------------------------------

def _profile_event_to_memories(
    profile: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Profile+Event data to flat memory list for ServiceInsightGenerator."""
    memories: list[dict[str, Any]] = []
    for key, entry in profile.items():
        memories.append({
            "id": f"profile_{key}",
            "memory": render_profile_text(key, entry.get("value", {})),
            "metadata": {"memory_key": key},
        })
    for event in events:
        memories.append({
            "id": event["event_id"],
            "memory": render_event_text(event),
            "metadata": {"memory_key": event["event_type"]},
        })
    return memories


def analyze_conversation() -> None:
    session = active_session()
    session["processing_error"] = None
    stage = "初始化"
    try:
        ctx = current_context()
        pipeline = pe_pipeline()
        store = pe_store()
        started_total = perf_counter()

        # Step 1: v5 Profile+Event pipeline (1-2 LLM calls, ~3-6s)
        stage = "记忆提取（v5 Profile+Event）"
        result = pipeline.process_turn(
            messages=session["messages"],
            user_id=ctx["user_id"],
            conversation_id=ctx["conversation_id"],
            scene_major=ctx["scene_major"],
            scene_minor=ctx["scene_minor"],
            occurred_at=ctx["occurred_at"],
        )
        total_latency = round((perf_counter() - started_total) * 1000)

        # Step 2: Read Profile + Event + Conflicts from SQLite
        stage = "读取用户记忆"
        profile = store.get_all_profile(ctx["user_id"])
        events = store.get_events(ctx["user_id"])
        conflicts = store.get_conflicts(ctx["user_id"], unresolved_only=True)

        # Step 3: Service insight (1 LLM call, ~2-4s; Agent 模式 2-4 LLM 调用 ~5-10s)
        stage = "服务洞察映射"
        memories = _profile_event_to_memories(profile, events)
        generator = ServiceInsightGenerator(
            use_agent_mode=session.get("agent_mode", False),
            user_id=ctx["user_id"],
        )
        insight = generator.generate(session["messages"], memories)

        # Store results
        all_decisions = list(result.profile_decisions) + list(result.event_decisions)
        session["write_result"] = {
            "decisions": all_decisions,
            "rejected": list(result.rejected),
            "idempotent_skip": result.idempotent_skip,
            "pipeline_latency_ms": result.latency_ms,
            "pipeline_tokens": result.token_usage,
            "llm_judge_calls": result.llm_judge_calls,
            "total_latency_ms": total_latency,
            "insight_latency_ms": insight.latency_ms,
            # 上下文工程指标
            "context_tokens": result.context_tokens,
            "context_truncated": result.context_truncated,
            "conversation_summarized": result.conversation_summarized,
            "bm25_event_count": result.bm25_event_count,
            "summary_latency_ms": result.summary_latency_ms,
            "summary_tokens": result.summary_tokens,
            # 记忆生命周期指标
            "sentiment_reset": result.sentiment_reset,
            "events_tagged": result.events_tagged,
            "aggregation_count": result.aggregation_count,
            "conflicts": conflicts,
            # 知识库检索指标
            "knowledge_doc_count": result.knowledge_doc_count,
            "knowledge_sources": insight.knowledge_sources or [],
            # 工具调用（Function Calling）指标
            "agent_mode": bool(getattr(insight, "agent_mode", False)),
            "tool_calls": int(getattr(insight, "tool_calls", 0)),
            "tool_trace": list(getattr(insight, "tool_trace", []) or []),
        }
        session["analysis"] = {"profile": profile, "events": events}
        session["insight"] = insight
    except Exception as exc:
        LOGGER.exception("AI memory pipeline failed at stage=%s", stage)
        error_name = type(exc).__name__
        if error_name in {"APIConnectionError", "AuthenticationError", "OpenAIError"}:
            message = "AI 服务暂时无法连接，请确认网络与 DASHSCOPE_API_KEY 后重试。"
        elif isinstance(exc, (ValueError, TypeError, AttributeError, KeyError)):
            message = "AI 已返回结果，但结果格式暂时无法解析，请重试。"
        else:
            message = "记忆处理链路执行失败，请重试或查看本地日志。"
        # 包含错误详情便于调试（开发期）
        error_detail = str(exc)[:200] if str(exc) else error_name
        session["processing_error"] = f"{message}（阶段：{stage}；{error_name}；{error_detail}）"
    save_active_session(session)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_service_label(label: ServiceLabel) -> None:
    color = {"urgent": "red", "high": "orange", "normal": "blue"}[label.priority]
    priority_text = {"urgent": "立即关注", "high": "优先关注", "normal": "相关信息"}[label.priority]
    with st.container(border=True):
        st.badge(priority_text, color=color)
        st.markdown(f"**{label.title}**")
        st.caption(label.content)


def render_service_suggestion(suggestion: ServiceSuggestion) -> None:
    action_labels = {
        "empathize": "回应情绪", "clarify": "补充确认", "confirm_need": "确认诉求",
        "avoid_repeat": "避免重复询问", "safety_attention": "安全关注", "follow_up": "持续跟进",
    }
    with st.container(border=True):
        st.badge(action_labels[suggestion.action], icon=":material/assistant:", color="violet")
        st.caption(suggestion.content)


def _render_metric_badge(label: str, value: str, cls: str) -> str:
    return f'<span class="metric-badge {cls}">{label}: {value}</span>'


# 记忆键 / 字段 / 枚举值的中文显示映射（仅用于 UI 渲染，不影响存储与提示词）
PROFILE_KEY_TITLES: dict[str, str] = {
    "body.skin_type": "肤质",
    "body.skin_sensitivity": "皮肤敏感度",
    "body.appearance_trait": "外观特征",
    "body.baseline_condition": "基础皮肤状态",
    "body.special_status": "特殊状态",
    "body.ingredient_constraint": "成分禁忌",
    "experience.product_reaction": "产品使用反应",
    "experience.product_result": "产品使用效果",
    "preference.product_attribute": "产品偏好",
    "purchase.context": "购买背景",
    "need.product_goal": "使用目标",
    "need.decision_support": "决策支持需求",
    "need.information": "信息确认需求",
    "need.service_action": "服务诉求",
    "concern.observed_problem": "已观察问题",
    "concern.core_worry": "核心担忧",
    "sentiment.current": "当前情绪",
    "dissatisfaction.reason": "不满原因",
    "dissatisfaction.escalation_intent": "升级/投诉意向",
    "satisfaction.positive_feedback": "正面反馈",
    "service.evidence_submitted": "已提交材料",
    "service.plan_response": "方案回应",
    "service.commitment": "承诺事项",
    "service.user_resolution": "用户确认的解决状态",
}

FIELD_LABELS: dict[str, str] = {
    "skin_type": "肤质", "level": "敏感程度", "trait": "特征", "body_area": "部位",
    "condition": "状态", "status": "状态", "ingredient": "成分", "constraint_type": "约束",
    "product": "产品", "reaction": "反应", "severity": "严重程度", "current_change": "当前变化",
    "onset": "出现时机", "result": "效果", "polarity": "倾向", "attribute": "属性",
    "preference": "偏好", "value": "内容", "purpose": "用途", "recipient": "收礼人",
    "occasion": "场合", "goal": "目标", "subject": "对象", "problem": "问题",
    "worry": "担忧", "decision": "决策", "options": "选项", "topic": "主题",
    "question": "问题", "request": "诉求", "action_type": "动作", "emotion": "情绪",
    "intensity": "强度", "target": "对象", "reason": "原因", "intent": "意图",
    "feedback": "反馈", "evidence_type": "材料", "plan": "方案", "response": "回应",
    "owner": "责任方", "action": "动作", "resolution": "解决状态", "statement": "表述",
}

VALUE_LABELS: dict[str, str] = {
    "dry": "干性", "oily": "油性", "combination": "混合性", "normal": "中性",
    "sensitive": "敏感", "not_sensitive": "不敏感", "uncertain": "不确定",
    "allergy": "过敏", "avoid": "避用", "caution": "慎用",
    "pregnant": "孕期", "breastfeeding": "哺乳期", "preparing_pregnancy": "备孕",
    "minor": "未成年", "other": "其他",
    "mild": "轻度", "moderate": "中度", "severe": "重度", "unknown": "未知",
    "worsened": "加重", "improved": "好转", "disappeared": "消失", "unchanged": "未变化",
    "positive": "正面", "negative": "负面", "mixed": "好坏参半", "neutral": "中性",
    "like": "喜欢", "prefer": "偏好",
    "self_use": "自用", "gift": "送礼", "travel": "旅行", "occasion": "特定场合", "stock_up": "囤货",
    "refund": "退款", "return": "退货", "exchange": "换货", "reship": "补发",
    "investigate": "调查核实", "explain": "解释说明", "apology": "致歉", "callback": "回电",
    "angry": "愤怒", "anxious": "焦虑", "disappointed": "失望", "impatient": "着急",
    "sad": "伤心", "calm": "平静", "satisfied": "满意",
    "low": "低", "medium": "中", "high": "高",
    "complaint": "投诉", "public_exposure": "公开曝光", "platform_escalation": "平台投诉",
    "regulatory_complaint": "监管投诉", "legal_action": "法律途径", "leave": "流失",
    "accepted": "接受", "rejected": "拒绝", "conditional": "有条件接受", "considering": "考虑中",
    "resolved": "已解决", "partially_resolved": "部分解决", "unresolved": "未解决",
    "support": "客服", "user": "用户", "merchant": "商家", "platform": "平台",
}


def _zh_value(value: Any) -> str:
    """把枚举值翻译成中文，未知值原样返回。"""
    if isinstance(value, list):
        return "、".join(_zh_value(item) for item in value)
    text = str(value)
    return VALUE_LABELS.get(text, text)


def _zh_fields_body(fields: dict[str, Any]) -> str:
    """把字段字典渲染成 "中文字段：中文值；..." 的紧凑正文。"""
    parts = [
        f"{FIELD_LABELS.get(k, k)}：{_zh_value(v)}"
        for k, v in fields.items()
        if v not in (None, "", [])
    ]
    return "；".join(parts)


def _render_profile_card(key: str, entry: dict[str, Any]) -> str:
    title = PROFILE_KEY_TITLES.get(key, key)
    value = entry.get("value", {})
    if isinstance(value, dict):
        body = _zh_fields_body(value)
    else:  # SET 类型初始值可能是 list
        body = "；".join(_zh_value(v) for v in value if v not in (None, "", []))
    return (
        f'<div class="profile-card">'
        f'<div class="key">{title}<span class="raw">{key}</span></div>'
        f'<div class="val">{body or "（空）"}</div>'
        f'</div>'
    )


# 流水线在 LLM 未提供描述时生成的机器格式："[维度][key][范围] field=值；..."
_MACHINE_DESC_RE = re.compile(r"^\[([^\]]*)\]\[([^\]]*)\]\[([^\]]*)\]\s*(.*)$", re.DOTALL)


def _render_event_card(event: dict[str, Any]) -> str:
    etype = event.get("event_type", "")
    title = PROFILE_KEY_TITLES.get(etype, etype)
    occurred = event.get("occurred_at") or event.get("created_at", "")
    occurred_short = str(occurred)[:10] if occurred else ""
    desc = str(event.get("description") or "").strip()
    machine = _MACHINE_DESC_RE.match(desc)
    if machine:  # 机器格式 → 解析字段后用中文重排
        fields: dict[str, Any] = {}
        for chunk in machine.group(4).split("；"):
            if "=" in chunk:
                fk, _, fv = chunk.partition("=")
                fields[fk.strip()] = fv.strip()
        desc = _zh_fields_body(fields) or machine.group(4)
    elif not desc:
        desc = _zh_fields_body(event.get("fields") or {})
    return (
        f'<div class="event-card">'
        f'<div class="type">{title} <small style="color:#999">{occurred_short}</small></div>'
        f'<div class="desc">{desc}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_conversation_list() -> None:
    with st.container(border=True, height=760):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("会话")
            st.button("新建", icon=":material/add_comment:", type="primary",
                      key="open_create_session", on_click=show_new_session_form)
        if st.session_state.show_new_session_form:
            with st.form("create_session_form", border=True):
                user_id = st.text_input("用户 ID", placeholder="例如：customer_10086",
                                        key="new_session_user_id")
                submitted = st.form_submit_button("创建并进入", type="primary",
                                                   icon=":material/add_comment:", width="stretch")
            if submitted:
                if not user_id.strip():
                    st.error("请输入 user_id。")
                else:
                    create_session(user_id)
                    st.session_state.show_new_session_form = False
                    st.rerun()
        st.caption(f"共 {len(st.session_state.sessions)} 个会话")
        st.divider()
        for conversation_id, session in reversed(list(st.session_state.sessions.items())):
            selected = conversation_id == st.session_state.active_conversation_id
            if st.button(
                session["title"], key=f"session_{conversation_id}",
                type="primary" if selected else "secondary",
                icon=":material/chat_bubble:" if selected else ":material/chat:",
                width="stretch", on_click=select_session, args=(conversation_id,),
            ):
                pass
            st.caption(f"{session['user_id']} · {len(session['messages'])} 条消息")
        st.divider()
        with st.expander("演示工具", icon=":material/science:"):
            st.button("载入示例", icon=":material/science:", on_click=load_sample,
                      key="load_sample_active", width="stretch")
            st.button("清空当前会话", icon=":material/delete:", on_click=clear_current_session,
                      key="clear_active_session", width="stretch")
            st.button("重置用户数据", icon=":material/restart_alt:", on_click=reset_user_data,
                      key="reset_user_data", width="stretch")


def render_chat_panel(session: dict[str, Any]) -> None:
    with st.container(border=True, height=760):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(session["title"])
            st.badge(session["user_id"], icon=":material/person:", color="gray")
        st.caption(f"会话 ID：{session['conversation_id']}")
        chat_area = st.container(height=545)
        with chat_area:
            if not session["messages"]:
                st.info("开始一段新对话。可在下方选择由客户或客服发送消息。", icon=":material/chat:")
            for message in session["messages"]:
                avatar = ":material/person:" if message["role"] == "user" else ":material/support_agent:"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])

        sender_key = f"sender_{session['conversation_id']}"
        st.session_state.setdefault(sender_key, "客户")
        with st.form(f"message_form_{session['conversation_id']}", border=False):
            sender_col, input_col, send_col = st.columns(
                [1.4, 5, 0.9], vertical_alignment="bottom", gap="small")
            with sender_col:
                st.segmented_control("发言人", ["客户", "客服"], key=sender_key,
                                    selection_mode="single")
            with input_col:
                content = st.text_input("消息", placeholder="输入消息……",
                                         label_visibility="collapsed")
            with send_col:
                submitted = st.form_submit_button("发送", type="primary",
                                                   icon=":material/send:", width="stretch")
        if submitted and content.strip():
            role = "user" if st.session_state[sender_key] == "客户" else "assistant"
            append_message(role, content)
            st.rerun()


def render_insight_panel(session: dict[str, Any]) -> None:
    with st.container(border=True, height=760):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("服务洞察")
            st.button("分析", type="primary", icon=":material/psychology:",
                      disabled=not bool(session["messages"]),
                      on_click=analyze_conversation,
                      key=f"analyze_{session['conversation_id']}")
        # Agent 模式开关（Function Calling）
        session["agent_mode"] = st.toggle(
            "Agent 模式（主动工具调用）",
            value=bool(session.get("agent_mode", False)),
            key=f"agent_mode_{session['conversation_id']}",
            help="开启后 LLM 可主动调用 4 个工具：search_knowledge_base / get_user_profile / get_user_events / check_ingredient_safety。耗时约多 5-8s。",
        )

        if not session["analysis"]:
            st.caption("分析当前对话并更新用户记忆，随后生成动态服务标签。")
            # 安全防护状态（分析前也展示）
            pre_scans = session.get("security_scans", [])
            pre_pii = sum(s.get("pii_count", 0) for s in pre_scans)
            pre_inj = sum(s.get("injection_count", 0) for s in pre_scans)
            if pre_pii > 0 or pre_inj > 0:
                parts = []
                if pre_pii:
                    parts.append(f"PII 脱敏 {pre_pii}")
                if pre_inj:
                    parts.append(f"注入拦截 {pre_inj}")
                st.caption(f"🛡️ 安全防护已启用：{' · '.join(parts)}")
            else:
                st.caption("🛡️ 安全防护已启用（PII 检测 + 注入拦截）")
            if session["processing_error"]:
                st.error(session["processing_error"], icon=":material/cloud_off:")
            return

        # Performance metrics
        wr = session["write_result"] or {}
        total_ms = wr.get("total_latency_ms", 0)
        pipe_ms = wr.get("pipeline_latency_ms", 0)
        insight_ms = wr.get("insight_latency_ms", 0)
        tokens = wr.get("pipeline_tokens", {})
        total_tokens = tokens.get("total_tokens", 0)
        judge_calls = wr.get("llm_judge_calls", 0)
        # Agent 模式下洞察阶段会有多次 LLM 调用（决策 + 最终生成）
        agent_mode_on = wr.get("agent_mode", False)
        tool_trace = wr.get("tool_trace", []) or []
        tool_calls_count = wr.get("tool_calls", 0)
        if agent_mode_on:
            # 洞察阶段 = len(tool_trace) 次 LLM 调用（最后一次也计为 1）
            insight_llm_calls = max(1, len(tool_trace))
        else:
            insight_llm_calls = 1
        total_llm = 1 + judge_calls + insight_llm_calls  # extract + judge + insight

        metrics_html = (
            _render_metric_badge("耗时", f"{total_ms / 1000:.1f}s", "metric-speed")
            + _render_metric_badge("Token", str(total_tokens), "metric-token")
            + _render_metric_badge("LLM调用", str(total_llm), "metric-llm")
        )
        if agent_mode_on:
            metrics_html += _render_metric_badge("工具调用", str(tool_calls_count), "metric-token")
        st.markdown(metrics_html, unsafe_allow_html=True)

        # 状态徽章（单行排列，避免纵向堆叠）
        with st.container(horizontal=True):
            if wr.get("idempotent_skip"):
                st.badge("相同对话已幂等跳过", icon=":material/check_circle:", color="green")
            else:
                decisions = wr.get("decisions", [])
                operation_counts = {
                    op: sum(1 for item in decisions if item.get("operation") == op)
                    for op in ("ADD", "UPDATE", "DELETE", "NOOP")
                }
                summary = " · ".join(
                    f"{op} {count}" for op, count in operation_counts.items() if count
                ) or "没有可写入事实"
                st.badge(summary, icon=":material/database:", color="blue")
            security_scans_badge = session.get("security_scans", [])
            total_pii_badge = sum(s.get("pii_count", 0) for s in security_scans_badge)
            total_inj_badge = sum(s.get("injection_count", 0) for s in security_scans_badge)
            if total_pii_badge > 0:
                st.badge(f"PII 脱敏 {total_pii_badge}", icon=":material/shield:", color="orange")
            if total_inj_badge > 0:
                has_high = any(
                    sev == "high"
                    for s in security_scans_badge
                    for sev in s.get("injection_severities", [])
                )
                st.badge(
                    f"注入拦截 {total_inj_badge}",
                    icon=":material/warning:",
                    color="red" if has_high else "orange",
                )

        insight: ServiceInsight | None = session["insight"]
        profile = session["analysis"].get("profile", {})
        events = session["analysis"].get("events", [])
        conflicts = wr.get("conflicts", [])

        tab_view, tab_memory, tab_diag = st.tabs([
            "服务洞察",
            f"用户记忆（Profile {len(profile)} · Event {len(events)}）",
            "运行详情",
        ])

        # ── Tab 1：服务洞察（焦点 / 标签 / 建议）─────────────────────────
        with tab_view:
            if insight:
                st.markdown("**服务焦点**")
                st.info(insight.service_focus, icon=":material/center_focus_strong:")
                if insight.labels:
                    st.markdown("**动态服务标签**")
                    for label in insight.labels:
                        render_service_label(label)
                else:
                    st.caption("当前没有生成可展示的服务标签。")
                if insight.service_suggestions:
                    st.markdown("**AI 服务建议**")
                    for suggestion in insight.service_suggestions:
                        render_service_suggestion(suggestion)
            else:
                st.caption("当前没有生成可展示的服务洞察。")

        # ── Tab 2：用户记忆（冲突 / Profile / Event）────────────────────
        with tab_memory:
            for c in conflicts:
                old_val = c.get("old_value", {})
                new_val = c.get("new_value", {})
                st.warning(
                    f"⚠️ **{c['profile_key']}** 冲突："
                    f"旧值 {json.dumps(old_val, ensure_ascii=False)} → "
                    f"新值 {json.dumps(new_val, ensure_ascii=False)}",
                    icon=":material/warning:",
                )
            st.markdown(f"**Profile 画像**（{len(profile)}）")
            if profile:
                profile_html = "".join(
                    _render_profile_card(k, v) for k, v in sorted(profile.items())
                )
                st.markdown(profile_html, unsafe_allow_html=True)
            else:
                st.caption("暂无 Profile 数据，点击「分析」后自动写入。")
            resolved_count = sum(1 for e in events if e.get("resolution_status") == "resolved")
            unresolved_count = sum(1 for e in events if e.get("resolution_status") == "unresolved")
            st.markdown(f"**Event 事件流**（{len(events)}）")
            if events:
                with st.container(horizontal=True):
                    if resolved_count:
                        st.badge(f"已解决 {resolved_count}", color="green")
                    if unresolved_count:
                        st.badge(f"未解决 {unresolved_count}", color="red")
                event_html = "".join(_render_event_card(e) for e in events)
                st.markdown(event_html, unsafe_allow_html=True)
            else:
                st.caption("暂无 Event 数据，点击「分析」后自动写入。")

        # ── Tab 3：运行详情（诊断信息全部收进折叠区）────────────────────
        with tab_diag:
            ctx_tokens = wr.get("context_tokens", 0)
            ctx_truncated = wr.get("context_truncated", False)
            conv_summarized = wr.get("conversation_summarized", False)
            bm25_count = wr.get("bm25_event_count", 0)
            summary_ms = wr.get("summary_latency_ms", 0)
            summary_tk = wr.get("summary_tokens", 0)
            sentiment_reset = wr.get("sentiment_reset", False)
            events_tagged = wr.get("events_tagged", 0)
            agg_count = wr.get("aggregation_count", 0)
            with st.expander("运行指标（性能 / 上下文 / 生命周期）", icon=":material/speed:"):
                st.caption(
                    f"流水线耗时：{pipe_ms} ms · 洞察映射：{insight_ms} ms · 总耗时：{total_ms} ms"
                )
                st.caption(f"提取 Token：{tokens}")
                if agent_mode_on:
                    st.caption(
                        f"LLM 调用：提取 1 + 去重判断 {judge_calls} + 洞察(Agent) {insight_llm_calls} = {total_llm}"
                    )
                    st.caption(f"工具调用：{tool_calls_count} 次（并行执行，总耗时计入洞察）")
                else:
                    st.caption(f"LLM 调用：提取 1 + 去重判断 {judge_calls} + 洞察 1 = {total_llm}")
                st.divider()
                st.caption(f"上下文 Token：{ctx_tokens} / 6000（软限制）")
                if ctx_truncated:
                    st.caption("⚠ 触发裁剪：已从尾部移除低相关 Event")
                else:
                    st.caption("✓ 未触发裁剪")
                st.caption(f"BM25 检索：命中 {bm25_count} 条相关 Event")
                if conv_summarized:
                    st.caption(f"对话摘要：已触发（{summary_ms} ms, {summary_tk} tokens）")
                else:
                    st.caption("对话摘要：未触发（≤10 轮）")
                st.divider()
                if sentiment_reset:
                    st.caption("情绪 TTL：已重置为平静（新会话）")
                else:
                    st.caption("情绪 TTL：未触发")
                st.caption(f"解决标签：标记了 {events_tagged} 条历史事件")
                if agg_count > 0:
                    st.caption(f"Event→Profile 聚合：自动生成 {agg_count} 条 Profile 候选")
                else:
                    st.caption("Event→Profile 聚合：无触发")
                if len(conflicts) > 0:
                    st.caption(f"⚠ 记忆冲突：检测到 {len(conflicts)} 条未解决冲突")
                else:
                    st.caption("记忆冲突：无")

            if agent_mode_on and tool_trace:
                with st.expander(f"工具调用轨迹 ({tool_calls_count} 次调用)", icon=":material/smart_toy:"):
                    for trace in tool_trace:
                        iter_n = trace.get("iteration", 0)
                        iter_ms = trace.get("latency_ms", 0)
                        decisions = trace.get("decisions", [])
                        executions = trace.get("executions", [])
                        if not decisions and not executions:
                            st.caption(f"第 {iter_n} 轮：LLM 直接生成洞察（{iter_ms} ms）")
                            continue
                        st.caption(f"第 {iter_n} 轮：决策 {len(decisions)} 个工具 · {iter_ms} ms")
                        for decision in decisions:
                            tool_name = decision.get("name", "")
                            tool_args = decision.get("arguments", {})
                            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                            if len(args_str) > 80:
                                args_str = args_str[:80] + "..."
                            st.caption(f"  → `{tool_name}`({args_str})")
                        for execution in executions:
                            exec_name = execution.get("name", "")
                            exec_ms = execution.get("latency_ms", 0)
                            exec_ok = execution.get("ok", False)
                            if exec_ok:
                                result_str = execution.get("result", "")
                                if len(result_str) > 100:
                                    result_str = result_str[:100] + "..."
                                st.caption(f"    ↳ {exec_name} ✓ {exec_ms}ms · {result_str}")
                            else:
                                err = execution.get("error", "")
                                st.caption(f"    ↳ {exec_name} ✗ {exec_ms}ms · {err}")

            kb_count = wr.get("knowledge_doc_count", 0)
            kb_sources = wr.get("knowledge_sources", [])
            with st.expander(f"知识库检索 ({kb_count + len(kb_sources)} 条命中)", icon=":material/menu_book:"):
                if kb_sources:
                    for src in kb_sources:
                        dtype = src.get("doc_type", "")
                        title = src.get("title", "")
                        content = src.get("content", "")
                        if len(content) > 120:
                            content = content[:120] + "..."
                        st.caption(f"[{dtype}] **{title}**")
                        st.caption(f"  {content}")
                else:
                    st.caption("未检索到相关知识库文档")

            security_scans = session.get("security_scans", [])
            total_pii = sum(s.get("pii_count", 0) for s in security_scans)
            total_injection = sum(s.get("injection_count", 0) for s in security_scans)
            high_risk = sum(
                1 for s in security_scans
                for sev in s.get("injection_severities", [])
                if sev == "high"
            )
            panel_title = f"安全防护 (PII {total_pii} · 注入 {total_injection})"
            with st.expander(panel_title, icon=":material/shield:"):
                if total_pii == 0 and total_injection == 0:
                    st.caption("✓ 本会话未检测到 PII 或注入风险")
                else:
                    # PII 汇总
                    if total_pii > 0:
                        st.caption(f"**PII 检测**：共 {total_pii} 条已脱敏")
                        for i, scan in enumerate(security_scans):
                            if scan.get("pii_count", 0) == 0:
                                continue
                            role_label = "客户" if scan.get("role") == "user" else "客服"
                            pii_str = "、".join(scan.get("pii_types", []))
                            st.caption(f"  消息{i + 1}（{role_label}）：{pii_str}")
                            for detail in scan.get("pii_details", []):
                                st.caption(f"    → {detail['label']}：{detail['masked']}")
                    # 注入汇总
                    if total_injection > 0:
                        risk_emoji = "🔴" if high_risk > 0 else "🟡"
                        st.caption(f"{risk_emoji} **Prompt 注入**：共 {total_injection} 条"
                                   + (f"（含 {high_risk} 条高风险）" if high_risk else ""))
                        for i, scan in enumerate(security_scans):
                            if scan.get("injection_count", 0) == 0:
                                continue
                            role_label = "客户" if scan.get("role") == "user" else "客服"
                            for detail in scan.get("injection_details", []):
                                sev_color = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(
                                    detail["severity"], "⚪"
                                )
                                st.caption(
                                    f"  消息{i + 1}（{role_label}）：{sev_color} {detail['label']}"
                                )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

initialize_state()

st.markdown('<div class="app-title">共情记忆工作台</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Profile+Event 双通道记忆 · Agent 工具调用 · 服务洞察</div>', unsafe_allow_html=True)
st.divider()

session = active_session()
conversation_col, chat_col, insight_col = st.columns([1.05, 2.5, 1.65], gap="small")

with conversation_col:
    render_conversation_list()
with chat_col:
    render_chat_panel(session)
with insight_col:
    render_insight_panel(session)
