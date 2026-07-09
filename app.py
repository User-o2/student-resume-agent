"""Streamlit 浏览器入口，用于多轮生成学生简历。"""

from __future__ import annotations

import json

import streamlit as st

from app.agent import ResumeAgentService
from app.schema import ResumeState
from app.tools import STAGE_LABELS, check_missing_fields


def init_session_state() -> None:
    """初始化 Streamlit 会话状态。

    Args:
        无。

    Returns:
        None。
    """

    if "resume_state" not in st.session_state:
        st.session_state.resume_state = ResumeState()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "你好，我会按新版标准模板帮你生成学生简历。请先告诉我目标岗位、目标行业、期望城市，以及姓名、电话、邮箱和籍贯。",
            }
        ]
    if "resume_markdown" not in st.session_state:
        st.session_state.resume_markdown = ""
    if "output_path" not in st.session_state:
        st.session_state.output_path = ""
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []
    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = ""


@st.cache_resource(show_spinner=False)
def get_agent_service(use_llm: bool) -> ResumeAgentService:
    """获取缓存的 Agent 服务。

    Args:
        use_llm: 是否启用 LLM 抽取与润色。

    Returns:
        简历 Agent 服务实例。
    """

    return ResumeAgentService(use_llm=use_llm)


def reset_session() -> None:
    """重置当前简历生成会话。

    Args:
        无。

    Returns:
        None。
    """

    st.session_state.resume_state = ResumeState()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "已重置。请先告诉我目标岗位、目标行业、期望城市，以及姓名、电话、邮箱和籍贯。",
        }
    ]
    st.session_state.resume_markdown = ""
    st.session_state.output_path = ""
    st.session_state.agent_trace = []
    st.session_state.pending_user_input = ""


def render_sidebar(use_llm: bool) -> None:
    """渲染侧边栏状态面板。

    Args:
        use_llm: 是否启用 LLM。

    Returns:
        None。
    """

    state = st.session_state.resume_state
    report = check_missing_fields(state)

    st.sidebar.header("状态")
    st.sidebar.caption(f"阶段：{STAGE_LABELS.get(state.current_stage, state.current_stage)}")
    st.sidebar.caption(f"LLM：{'启用' if use_llm else '关闭'}")

    if report["missing_fields"]:
        st.sidebar.subheader("待补充")
        for item in report["missing_fields"][:8]:
            st.sidebar.write(f"- {item}")
    if report["quality_questions"]:
        st.sidebar.subheader("质量追问")
        for item in report["quality_questions"][:3]:
            st.sidebar.write(f"- {item}")
    if report.get("optional_suggestions"):
        st.sidebar.subheader("建议优化")
        for item in report["optional_suggestions"][:3]:
            st.sidebar.write(f"- {item}")

    with st.sidebar.expander("结构化状态", expanded=False):
        st.json(json.loads(state.model_dump_json()))

    with st.sidebar.expander("Agent 工具轨迹", expanded=False):
        if st.session_state.agent_trace:
            for trace_item in st.session_state.agent_trace:
                st.write(f"- {trace_item}")
        else:
            st.caption("暂无工具调用记录")

    if st.sidebar.button("重置会话", use_container_width=True):
        reset_session()
        st.rerun()


def render_chat_messages() -> None:
    """渲染历史聊天消息。

    Args:
        无。

    Returns:
        None。
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def enqueue_user_message(user_input: str) -> None:
    """先记录用户消息并等待下一轮处理。

    Args:
        user_input: 用户输入。

    Returns:
        None。
    """

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_user_input = user_input


def process_pending_message(use_llm: bool) -> None:
    """处理等待中的用户消息并追加 Agent 回复。

    Args:
        use_llm: 是否启用 LLM。

    Returns:
        None。
    """

    user_input = st.session_state.pending_user_input
    if not user_input:
        return

    st.session_state.pending_user_input = ""
    service = get_agent_service(use_llm)
    with st.spinner("处理中..."):
        result = service.handle_message(user_input, st.session_state.resume_state)
    st.session_state.resume_state = result.state
    st.session_state.messages.append({"role": "assistant", "content": result.assistant_message})
    if result.resume_markdown:
        st.session_state.resume_markdown = result.resume_markdown
        st.session_state.output_path = result.output_path
    st.session_state.agent_trace = result.agent_trace


def render_resume_result() -> None:
    """渲染生成后的 Markdown 下载区域。

    Args:
        无。

    Returns:
        None。
    """

    if not st.session_state.resume_markdown:
        return

    st.divider()
    st.subheader("Markdown 文件")
    st.caption(st.session_state.output_path)
    st.download_button(
        label="下载 Markdown",
        data=st.session_state.resume_markdown,
        file_name="student_resume.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main() -> None:
    """运行 Streamlit 应用。

    Args:
        无。

    Returns:
        None。
    """

    st.set_page_config(page_title="学生简历生成智能体", layout="wide")
    init_session_state()

    use_llm = st.sidebar.toggle("启用 LLM 抽取与润色", value=True)
    render_sidebar(use_llm)

    st.title("学生简历生成智能体")
    render_chat_messages()

    if st.session_state.pending_user_input:
        process_pending_message(use_llm)
        st.rerun()

    if st.sidebar.button("生成简历", type="primary", use_container_width=True):
        enqueue_user_message("生成简历")
        st.rerun()

    user_input = st.chat_input("输入本轮补充的信息")
    if user_input:
        enqueue_user_message(user_input)
        st.rerun()

    render_resume_result()


if __name__ == "__main__":
    main()
