"""Three-column customer-support workbench for the Mem0 prototype."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Any

import streamlit as st

from label_engine import ServiceInsight, ServiceInsightGenerator, ServiceLabel, ServiceSuggestion
from memory_service import CustomerMemoryService, MemoryContext


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    error_log = os.path.join(os.path.dirname(__file__), "data", "app_errors.log")
    os.makedirs(os.path.dirname(error_log), exist_ok=True)
    file_handler = logging.FileHandler(error_log, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)


st.set_page_config(
    page_title="共情记忆工作台",
    page_icon=":material/support_agent:",
    layout="wide",
)


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
    }


def initialize_state() -> None:
    if "sessions" not in st.session_state:
        default = new_session_record("demo_customer_001", "DEMO_001")
        # Preserve the old single-session demo state across a hot reload.
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


@st.cache_resource
def memory_service() -> CustomerMemoryService:
    os.environ.setdefault("MEM0_COLLECTION_NAME", "customer_workbench_demo_v1_1024")
    return CustomerMemoryService()


def current_context() -> MemoryContext:
    session = active_session()
    return MemoryContext(
        user_id=session["user_id"],
        conversation_id=session["conversation_id"],
        event_id=session["event_id"],
        scene_major=session["scene_major"],
        scene_minor=session["scene_minor"],
        occurred_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


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
        messages=[],
        analysis=None,
        insight=None,
        write_result=None,
        processing_error=None,
        title="新会话",
    )
    save_active_session(session)


def load_sample() -> None:
    session = active_session()
    session.update(
        messages=[dict(item) for item in SAMPLE_MESSAGES],
        analysis=None,
        insight=None,
        write_result=None,
        processing_error=None,
        title="面膜使用后不适",
        scene_major="不良反应",
        scene_minor="泛红刺痒",
    )
    save_active_session(session)


def append_message(role: str, content: str) -> None:
    content = content.strip()
    if not content:
        return
    session = active_session()
    session["messages"].append({"role": role, "content": content})
    session["analysis"] = None
    session["insight"] = None
    session["processing_error"] = None
    if session["title"] == "新会话" and role == "user":
        session["title"] = content[:14] + ("…" if len(content) > 14 else "")
    save_active_session(session)


def analyze_conversation() -> None:
    session = active_session()
    session["processing_error"] = None
    stage = "初始化记忆服务"
    try:
        context = current_context()
        service = memory_service()
        stage = "第一映射与证据门控"
        write_result = service.add_validated_turn(session["messages"], context)
        session["write_result"] = write_result
        stage = "读取完整用户记忆"
        all_memories = service.get_user_memories(context.user_id)
        session["analysis"] = {"all_memories": all_memories}
        stage = "第二次服务洞察映射"
        session["insight"] = ServiceInsightGenerator().generate(
            session["messages"],
            all_memories,
        )
    except Exception as exc:
        LOGGER.exception("AI memory pipeline failed at stage=%s", stage)
        error_name = type(exc).__name__
        if error_name in {"APIConnectionError", "AuthenticationError", "OpenAIError"}:
            message = "AI 服务暂时无法连接，请确认网络与 DASHSCOPE_API_KEY 后重试。"
        elif isinstance(exc, (ValueError, TypeError, AttributeError, KeyError)):
            message = "AI 已返回结果，但结果格式暂时无法解析，请重试。"
        else:
            message = "记忆处理链路执行失败，请重试或查看本地日志。"
        session["processing_error"] = f"{message}（阶段：{stage}；{error_name}）"
    save_active_session(session)


def memory_text(item: dict[str, Any]) -> str:
    return str(item.get("memory", "")).strip()


def render_service_label(label: ServiceLabel) -> None:
    color = {"urgent": "red", "high": "orange", "normal": "blue"}[label.priority]
    priority_text = {"urgent": "立即关注", "high": "优先关注", "normal": "相关信息"}[
        label.priority
    ]
    with st.container(border=True):
        st.badge(priority_text, color=color)
        st.markdown(f"#### {label.title}")
        st.write(label.content)


def render_service_suggestion(suggestion: ServiceSuggestion) -> None:
    action_labels = {
        "empathize": "回应情绪",
        "clarify": "补充确认",
        "confirm_need": "确认诉求",
        "avoid_repeat": "避免重复询问",
        "safety_attention": "安全关注",
        "follow_up": "持续跟进",
    }
    with st.container(border=True):
        st.badge(
            action_labels[suggestion.action],
            icon=":material/assistant:",
            color="violet",
        )
        st.write(suggestion.content)


def render_conversation_list() -> None:
    with st.container(border=True, height=760):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("会话")
            st.button(
                "新建",
                icon=":material/add_comment:",
                type="primary",
                key="open_create_session",
                on_click=show_new_session_form,
            )
        if st.session_state.show_new_session_form:
            with st.form("create_session_form", border=True):
                user_id = st.text_input(
                    "用户 ID",
                    placeholder="例如：customer_10086",
                    key="new_session_user_id",
                )
                submitted = st.form_submit_button(
                    "创建并进入",
                    type="primary",
                    icon=":material/add_comment:",
                    width="stretch",
                )
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
                session["title"],
                key=f"session_{conversation_id}",
                type="primary" if selected else "secondary",
                icon=":material/chat_bubble:" if selected else ":material/chat:",
                width="stretch",
                on_click=select_session,
                args=(conversation_id,),
            ):
                pass
            st.caption(f"{session['user_id']} · {len(session['messages'])} 条消息")
        st.divider()
        with st.expander("演示工具", icon=":material/science:"):
            st.button(
                "载入示例",
                icon=":material/science:",
                on_click=load_sample,
                key="load_sample_active",
                width="stretch",
            )
            st.button(
                "清空当前会话",
                icon=":material/delete:",
                on_click=clear_current_session,
                key="clear_active_session",
                width="stretch",
            )


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
                [1.4, 5, 0.9],
                vertical_alignment="bottom",
                gap="small",
            )
            with sender_col:
                st.segmented_control(
                    "发言人",
                    ["客户", "客服"],
                    key=sender_key,
                    selection_mode="single",
                )
            with input_col:
                content = st.text_input(
                    "消息",
                    placeholder="输入消息……",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button(
                    "发送",
                    type="primary",
                    icon=":material/send:",
                    width="stretch",
                )
        if submitted and content.strip():
            role = "user" if st.session_state[sender_key] == "客户" else "assistant"
            append_message(role, content)
            st.rerun()


def render_insight_panel(session: dict[str, Any]) -> None:
    with st.container(border=True, height=760):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("服务洞察插件")
            st.button(
                "分析",
                type="primary",
                icon=":material/psychology:",
                disabled=not bool(session["messages"]),
                on_click=analyze_conversation,
                key=f"analyze_{session['conversation_id']}",
            )

        if not session["analysis"]:
            st.caption("分析当前对话并更新用户记忆，随后生成动态服务标签。")
            if session["processing_error"]:
                st.error(session["processing_error"], icon=":material/cloud_off:")
            return

        write_result = session["write_result"] or {}
        if write_result.get("idempotent_skip"):
            st.badge("相同对话已幂等跳过", icon=":material/check_circle:", color="green")
        else:
            decisions = write_result.get("decisions", [])
            operation_counts = {
                operation: sum(1 for item in decisions if item.get("operation") == operation)
                for operation in ("ADD", "UPDATE", "DELETE", "NOOP")
            }
            summary = " · ".join(
                f"{operation} {count}"
                for operation, count in operation_counts.items()
                if count
            ) or "没有可写入事实"
            st.badge(summary, icon=":material/database:", color="blue")

        insight: ServiceInsight | None = session["insight"]
        if insight:
            st.markdown("**当前服务焦点**")
            st.info(insight.service_focus, icon=":material/center_focus_strong:")

        if insight and insight.labels:
            st.markdown("**动态服务标签**")
            for service_label in insight.labels:
                render_service_label(service_label)
        else:
            st.caption("当前没有生成可展示的服务标签。")

        if insight and insight.service_suggestions:
            st.markdown("**AI 服务建议**")
            for suggestion in insight.service_suggestions:
                render_service_suggestion(suggestion)

        all_memories = session["analysis"].get("all_memories", [])
        with st.expander("查看标签依据", icon=":material/fact_check:"):
            if all_memories:
                st.markdown(f"**完整用户记忆** · {len(all_memories)}")
                for item in all_memories:
                    metadata = item.get("metadata") or {}
                    memory_key = metadata.get("memory_key", "未分类")
                    st.info(
                        f"{memory_key} · {memory_text(item)}",
                        icon=":material/person_search:",
                    )
            else:
                st.caption("该用户当前没有长期记忆；标签可直接引用近期对话。")
            if insight:
                st.caption(f"第二次映射耗时：{insight.latency_ms} ms")

initialize_state()

st.title("共情记忆客服工作台")
st.caption("会话列表 · 人工聊天 · AI 服务洞察")

session = active_session()
conversation_col, chat_col, insight_col = st.columns([1.05, 2.5, 1.65], gap="small")

with conversation_col:
    render_conversation_list()
with chat_col:
    render_chat_panel(session)
with insight_col:
    render_insight_panel(session)
