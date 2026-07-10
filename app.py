"""Streamlit 浏览器入口，用于多轮生成学生简历。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.agent import ResumeAgentService
from app.config import OUTPUTS_DIR
from app.schema import ResumeState
from app.tools import STAGE_LABELS, check_missing_fields, export_resume_to_word


CONVERSATION_DIR = OUTPUTS_DIR / "conversations"


def new_session_log_path() -> str:
    """生成当前会话日志文件路径。

    Args:
        无。

    Returns:
        会话日志文件路径字符串。
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(CONVERSATION_DIR / f"session_{timestamp}.json")


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
                "content": "你好，我是简历生成智能体。我可以帮你梳理经历、优化表达，最终生成一份专业的求职简历！",
            }
        ]
    if "resume_markdown" not in st.session_state:
        st.session_state.resume_markdown = ""
    if "word_document" not in st.session_state:
        st.session_state.word_document = b""
    if "word_output_path" not in st.session_state:
        st.session_state.word_output_path = ""
    if "score_markdown" not in st.session_state:
        st.session_state.score_markdown = ""
    if "output_path" not in st.session_state:
        st.session_state.output_path = ""
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []
    if "agent_trace_history" not in st.session_state:
        st.session_state.agent_trace_history = []
    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = ""
    if "session_log_path" not in st.session_state:
        st.session_state.session_log_path = new_session_log_path()


def save_conversation_log(event: str = "turn_complete") -> None:
    """保存当前 Streamlit 会话的完整对话日志。

    Args:
        event: 触发保存的事件名称。

    Returns:
        None。
    """

    CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(st.session_state.session_log_path)
    payload = {
        "event": event,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
        "resume_state": st.session_state.resume_state.model_dump(),
        "agent_trace": st.session_state.agent_trace,
        "agent_trace_history": st.session_state.agent_trace_history,
        "output_path": st.session_state.output_path,
        "has_resume_markdown": bool(st.session_state.resume_markdown),
        "word_output_path": st.session_state.word_output_path,
        "score_markdown": st.session_state.score_markdown,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    st.session_state.word_document = b""
    st.session_state.word_output_path = ""
    st.session_state.score_markdown = ""
    st.session_state.output_path = ""
    st.session_state.agent_trace = []
    st.session_state.agent_trace_history = []
    st.session_state.pending_user_input = ""
    st.session_state.session_log_path = new_session_log_path()
    save_conversation_log("reset")


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

    with st.sidebar.expander("结构化状态", expanded=True):
        st.json(json.loads(state.model_dump_json()))

    with st.sidebar.expander("Agent 工具轨迹", expanded=True):
        if st.session_state.agent_trace_history:
            for turn_trace in st.session_state.agent_trace_history:
                st.caption(f"第 {turn_trace['turn']} 轮：{turn_trace['user_input']}")
                for trace_item in turn_trace["trace"]:
                    st.write(f"- {trace_item}")
        else:
            st.caption("暂无工具调用记录")


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
        st.session_state.word_document = b""
        st.session_state.word_output_path = ""
    st.session_state.agent_trace = result.agent_trace
    st.session_state.agent_trace_history.append(
        {
            "turn": len([message for message in st.session_state.messages if message["role"] == "user"]),
            "user_input": user_input,
            "trace": result.agent_trace,
        }
    )
    save_conversation_log("turn_complete")


def render_existing_resume_upload(use_llm: bool) -> None:
    """渲染已有简历上传与优化区域。

    Args:
        use_llm: 是否启用 LLM 解析与润色。

    Returns:
        None。
    """

    with st.expander("上传已有简历并优化", expanded=False):
        uploaded_file = st.file_uploader(
            "上传 Markdown 简历",
            type=["md", "txt"],
            accept_multiple_files=False,
        )
        if uploaded_file is None:
            st.caption("支持上传已有 Markdown 简历，系统会解析、去重、润色并套用当前模板。")
            return

        st.caption(f"已选择：{uploaded_file.name}")
        if not st.button("解析并优化已有简历", use_container_width=True):
            return

        try:
            resume_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("文件编码无法识别，请上传 UTF-8 编码的 Markdown 文本。")
            return

        output_path = OUTPUTS_DIR / f"optimized_{Path(uploaded_file.name).stem}_{datetime.now():%Y%m%d_%H%M%S}.md"
        service = get_agent_service(use_llm)
        with st.spinner("正在解析并优化已有简历..."):
            result = service.optimize_existing_resume(resume_text, output_path=output_path)

        st.session_state.resume_state = result.state
        st.session_state.resume_markdown = result.markdown
        st.session_state.output_path = result.output_path
        st.session_state.word_document = b""
        st.session_state.word_output_path = ""
        st.session_state.agent_trace = result.agent_trace
        st.session_state.agent_trace_history.append(
            {
                "turn": len([message for message in st.session_state.messages if message["role"] == "user"]) + 1,
                "user_input": f"上传已有简历：{uploaded_file.name}",
                "trace": result.agent_trace,
            }
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"已解析并优化已有简历：{result.output_path}\n\n{result.markdown}",
            }
        )
        save_conversation_log("optimize_existing_resume")
        st.rerun()


def render_resume_score_panel(use_llm: bool) -> None:
    """渲染简历评分区域。

    Args:
        use_llm: 是否启用 LLM 评分。

    Returns:
        None。
    """

    with st.expander("简历评分", expanded=False):
        state = st.session_state.resume_state
        default_target = state.job_intention.target_position
        target_position = st.text_input(
            "评分目标岗位",
            value=default_target,
            placeholder="例如：人工智能算法实习生、机械设计助理工程师",
        )
        st.caption("评分包含完整度、岗位匹配度和表达规范性。完整度由代码计算，匹配度与表达由 LLM 评估。")

        if st.button("开始评分", use_container_width=True):
            service = get_agent_service(use_llm)
            with st.spinner("正在生成评分报告..."):
                result = service.score_resume(state, target_position=target_position)

            st.session_state.score_markdown = result.markdown
            st.session_state.agent_trace = result.agent_trace
            st.session_state.agent_trace_history.append(
                {
                    "turn": len([message for message in st.session_state.messages if message["role"] == "user"]) + 1,
                    "user_input": f"简历评分：{target_position or default_target or '未指定岗位'}",
                    "trace": result.agent_trace,
                }
            )
            save_conversation_log("score_resume")
            st.rerun()

        if st.session_state.score_markdown:
            st.markdown(st.session_state.score_markdown)


def render_resume_result() -> None:
    """渲染生成后的 Markdown 与 Word 导出区域。

    Args:
        无。

    Returns:
        None。
    """

    if not st.session_state.resume_markdown:
        return

    st.divider()
    st.subheader("导出简历")
    st.caption(st.session_state.output_path)
    markdown_column, word_column = st.columns(2)
    with markdown_column:
        st.download_button(
            label="下载 Markdown",
            data=st.session_state.resume_markdown,
            file_name="student_resume.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with word_column:
        if not st.session_state.word_document:
            if st.button("导出为 Word", use_container_width=True):
                source_path = Path(st.session_state.output_path)
                word_path = source_path.with_suffix(".docx") if source_path.suffix else OUTPUTS_DIR / "student_resume.docx"
                try:
                    result = export_resume_to_word(st.session_state.resume_markdown, output_path=word_path)
                except (OSError, ValueError) as error:
                    st.error(f"Word 导出失败：{error}")
                else:
                    st.session_state.word_document = result["docx_bytes"]
                    st.session_state.word_output_path = result["output_path"]
                    st.session_state.agent_trace = ["调用工具：export_resume_to_word"]
                    st.session_state.agent_trace_history.append(
                        {
                            "turn": len([message for message in st.session_state.messages if message["role"] == "user"]) + 1,
                            "user_input": "导出为 Word",
                            "trace": st.session_state.agent_trace,
                        }
                    )
                    save_conversation_log("export_word")
                    st.rerun()
        else:
            st.download_button(
                label="下载 Word",
                data=st.session_state.word_document,
                file_name=Path(st.session_state.word_output_path).name or "student_resume.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            st.caption(st.session_state.word_output_path)


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

    render_existing_resume_upload(use_llm)
    render_resume_score_panel(use_llm)

    user_input = st.chat_input("输入本轮补充的信息")
    if user_input:
        enqueue_user_message(user_input)
        st.rerun()

    render_resume_result()


if __name__ == "__main__":
    main()
