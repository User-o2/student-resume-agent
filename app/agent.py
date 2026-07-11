"""学生简历生成智能体的 LLM 主导编排逻辑。"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.agent_models import (
    AgentTurnResult,
    ExistingResumeOptimizationResult,
    ResumeImportResult,
    ResumePolishResult,
    ResumeScoreReport,
    ResumeScoreResult,
    ResumeTurnDecision,
)
from app.config import AppConfig, load_config
from app.llm import (
    build_chat_model,
    build_import_agent,
    build_langchain_agent,
    build_polish_agent,
    build_score_agent,
)
from app.prompts import (
    FINAL_POLISH_SYSTEM_PROMPT,
    FINAL_POLISH_USER_PROMPT,
    IMPORT_RESUME_SYSTEM_PROMPT,
    IMPORT_RESUME_USER_PROMPT,
    RESUME_SCORE_SYSTEM_PROMPT,
    RESUME_SCORE_USER_PROMPT,
    TURN_DECISION_SYSTEM_PROMPT,
    TURN_DECISION_USER_PROMPT,
)
from app.resume_parser import parse_existing_resume as _parse_existing_resume_fallback
from app.resume_parser import split_items as _split_items
from app.resume_scoring import build_fallback_score_report as _fallback_score_report
from app.resume_scoring import calculate_completeness_score as _calculate_completeness_score
from app.resume_scoring import normalize_score_report as _normalize_score_report
from app.resume_scoring import score_report_to_markdown as _score_report_to_markdown
from app.schema import ResumeState
from app.tools import (
    collect_resume_info_tool,
    coerce_resume_state,
    fill_resume_template_tool,
    parse_json_object,
    polish_state_experiences,
    validate_resume_state_tool,
)


GENERATE_KEYWORDS = ("生成简历", "输出简历", "导出简历", "完成简历")
GREETING_KEYWORDS = {"你好", "您好", "hi", "hello", "嗨", "在吗", "开始"}


def _contains_generate_intent(text: str) -> bool:
    """判断用户是否明确要求生成简历。

    Args:
        text: 用户输入。

    Returns:
        是否为生成意图。
    """

    normalized = re.sub(r"[\s，,。.!！?？~～]+", "", text.strip().lower())
    return normalized in {"生成", "输出", "导出", "完成", "生成简历", "输出简历", "导出简历", "完成简历"} or any(
        keyword.lower() in text.lower() for keyword in GENERATE_KEYWORDS
    )


def _is_greeting_only(text: str) -> bool:
    """判断用户输入是否只是寒暄或开始对话。

    Args:
        text: 用户输入。

    Returns:
        是否为纯寒暄输入。
    """

    normalized = re.sub(r"[\s，,。.!！?？~～]+", "", text.strip().lower())
    return normalized in GREETING_KEYWORDS


def _compact_patch(value: Any, key: str = "") -> Any:
    """移除补丁中的空值和状态元数据。

    Args:
        value: 待压缩的值。
        key: 当前字段名。

    Returns:
        压缩后的值。
    """

    if key in {"current_stage", "created_at", "updated_at"}:
        return None
    if isinstance(value, BaseModel):
        return _compact_patch(value.model_dump(), key)
    if isinstance(value, dict):
        compacted = {item_key: _compact_patch(item_value, item_key) for item_key, item_value in value.items()}
        return {item_key: item_value for item_key, item_value in compacted.items() if item_value not in (None, "", [], {})}
    if isinstance(value, list):
        compacted_list = [_compact_patch(item) for item in value]
        return [item for item in compacted_list if item not in (None, "", [], {})]
    return value if value not in (None, "", [], {}) else None


def _basic_fallback_extract(text: str, state: ResumeState) -> dict[str, Any]:
    """在 LLM 不可用时执行最小规则兜底抽取。

    Args:
        text: 用户输入。
        state: 当前简历状态。

    Returns:
        ResumeState 增量补丁。
    """

    update: dict[str, Any] = {}
    basic: dict[str, Any] = {}
    job: dict[str, Any] = {}
    education: dict[str, Any] = {}
    skills: dict[str, Any] = {}

    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    if email_match:
        basic["email"] = email_match.group(0)

    phone_match = re.search(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}[-\s]?\d{7,8})(?!\d)", text)
    if phone_match:
        basic["phone"] = phone_match.group(0)

    labeled_patterns = {
        "name": r"(?:姓名|我叫)[:：]?\s*([\u4e00-\u9fa5A-Za-z·]{2,20})",
        "native_place": r"(?:籍贯|家乡)[:：]?\s*([^，,。；;\n]+)",
        "target_position": r"目标岗位[:：]?\s*([^，,。；;\n]+)",
        "target_industry": r"目标行业[:：]?\s*([^，,。；;\n]+)",
        "expected_city": r"期望城市[:：]?\s*([^，,。；;\n]+)",
        "school": r"(?:学校|院校)[:：]?\s*([^，,。；;\n]+)",
        "college": r"学院[:：]?\s*([^，,。；;\n]+)",
        "major": r"专业(?:是|为|[:：])\s*([^，,。；;\n]+)",
        "gpa_or_rank": r"(?:专业排名|成绩排名|排名|GPA|绩点)[:：]?\s*([^，,。；;\n]+)",
        "english_level": r"(?:英语水平|英语)[:：]?\s*([^，,。；;\n]+)",
    }
    matched: dict[str, str] = {}
    for key, pattern in labeled_patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matched[key] = match.group(1).strip()

    for key in ("name", "native_place"):
        if matched.get(key):
            basic[key] = matched[key]
    for key in ("target_position", "target_industry", "expected_city"):
        if matched.get(key):
            job[key] = matched[key]

    if not job.get("target_position"):
        position_match = re.search(r"(?:投递|应聘|申请|想投|想做)\s*([^，,。；;\n]+)", text)
        if position_match:
            job["target_position"] = position_match.group(1).strip()

    for key in ("school", "college", "major", "gpa_or_rank", "english_level"):
        if matched.get(key):
            education[key] = matched[key]

    university_match = re.search(r"([\u4e00-\u9fa5A-Za-z]+大学)(?!生)", text)
    if university_match and not education.get("school") and not state.education.school:
        education["school"] = university_match.group(1)

    college_match = re.search(r"([\u4e00-\u9fa5A-Za-z]+学院)", text)
    if college_match and not education.get("college"):
        education["college"] = college_match.group(1)

    course_match = re.search(r"(?:主修课程|核心课程|课程)[:：]?\s*([^。；;\n]+)", text)
    if course_match:
        education["courses"] = _split_items(course_match.group(1))[:8]

    english_matches = re.findall(r"(?:CET-?\s*[46]\s*\d{0,3}分?|英语[四六四6]级\s*\d{0,3}分?)", text, flags=re.IGNORECASE)
    if english_matches and not education.get("english_level"):
        education["english_level"] = " | ".join(item.strip() for item in english_matches)

    tech_match = re.search(r"(?:技术栈|技能|熟悉|掌握)[:：]?\s*([^。；;\n]+)", text, flags=re.IGNORECASE)
    if tech_match:
        skill_items = _split_items(tech_match.group(1))[:12]
        languages = {"Python", "Java", "C++", "JavaScript", "TypeScript", "Go"}
        skills["programming_languages"] = [item for item in skill_items if item in languages]
        skills["tools"] = [item for item in skill_items if item not in languages]

    self_evaluation_mode = state.current_stage == "self_evaluation" and not re.search(
        r"(项目经历|项目名称|项目[:：]|获奖|奖学金|证书)",
        text,
    )
    if (
        not self_evaluation_mode
        and re.search(r"(项目|平台|系统|网站|小程序|应用|模型|课题|实践)", text)
        and not re.search(r"(竞赛|获奖|奖学金|证书)", text)
    ):
        project_title = ""
        title_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+(?:平台|系统|网站|小程序|应用|模型|课题|实践|项目))", text)
        if title_match:
            project_title = title_match.group(1)
        update["projects"] = [
            {
                "title": project_title,
                "raw_description": text.strip(),
                "responsibilities": [text.strip()] if re.search(r"负责|开发|实现|完成|参与|设计|搭建", text) else [],
                "results": [text.strip()] if re.search(r"\d|%|提升|准确率|排名|响应", text) else [],
            }
        ]

    if not self_evaluation_mode and re.search(r"(获奖|奖学金|证书|一等奖|二等奖|三等奖|国家级|省级|校级)", text):
        name_match = re.search(r"([^（(：:，,。；;\n]+(?:竞赛|比赛|奖学金|证书|奖))", text)
        update["awards"] = [
            {
                "name": name_match.group(1).strip() if name_match else text.strip()[:40],
                "description": text.strip(),
                "highlights": [text.strip()],
            }
        ]

    if self_evaluation_mode and not update.get("projects") and not update.get("awards"):
        update["self_evaluation"] = text.strip()

    if basic:
        update["basic_info"] = basic
    if job:
        update["job_intention"] = job
    if education:
        update["education"] = education
    if any(skills.values()):
        update["skills"] = skills
    return _compact_patch(update) or {}


def _sync_stage(state: ResumeState, report: dict[str, Any]) -> ResumeState:
    """根据底线校验报告更新 UI 展示阶段。

    Args:
        state: 当前简历状态。
        report: 底线校验报告。

    Returns:
        更新阶段后的简历状态。
    """

    missing_fields = [str(item) for item in report.get("missing_fields", [])]
    if any(item.startswith("个人信息：") for item in missing_fields):
        state.current_stage = "personal_info"
    elif any(item.startswith("教育背景：") for item in missing_fields):
        state.current_stage = "education"
    elif any(item.startswith("项目经历：") for item in missing_fields):
        state.current_stage = "projects"
    elif any(item.startswith("竞赛获奖：") for item in missing_fields):
        state.current_stage = "awards"
    elif any(item.startswith("自我评价：") for item in missing_fields):
        state.current_stage = "self_evaluation"
    else:
        state.current_stage = "ready"
    state.touch()
    return state


def _build_fallback_question(report: dict[str, Any]) -> str:
    """根据底线校验报告生成兜底追问。

    Args:
        report: 底线校验报告。

    Returns:
        面向用户的追问文本。
    """

    missing_fields = [str(item) for item in report.get("missing_fields", [])]
    if not missing_fields:
        return "必要信息已完整。回复“生成简历”即可输出 Markdown 简历，也可以继续补充想强调的内容。"

    grouped: dict[str, list[str]] = {}
    for item in missing_fields:
        section, _, field_name = item.partition("：")
        grouped.setdefault(section or "待补充", []).append(field_name or item)
    section, fields = next(iter(grouped.items()))
    focus = "、".join(fields[:3])
    return f"还需要补充{section}中的{focus}。你可以直接用自然语言描述，不需要按表单填写。"


def _decision_message(decision: ResumeTurnDecision, report: dict[str, Any]) -> str:
    """选择 LLM 决策中的回复文本。

    Args:
        decision: LLM 单轮决策。
        report: 底线校验报告。

    Returns:
        面向用户的回复文本。
    """

    if decision.assistant_message.strip():
        return decision.assistant_message.strip()
    if decision.followup_questions:
        return " ".join(question.strip() for question in decision.followup_questions if question.strip())
    return _build_fallback_question(report)


def _extract_structured_response(raw_result: dict[str, Any], model_type: type[BaseModel]) -> BaseModel | None:
    """从 LangChain Agent 返回值中提取结构化响应。

    Args:
        raw_result: Agent invoke 返回值。
        model_type: 期望的 Pydantic 模型类型。

    Returns:
        结构化响应对象；不存在或类型不匹配时返回 None。
    """

    if not isinstance(raw_result, dict):
        return None
    structured = raw_result.get("structured_response")
    if isinstance(structured, model_type):
        return structured
    if isinstance(structured, dict):
        try:
            return model_type.model_validate(structured)
        except ValueError:
            return None
    return None


def _collect_with_tool(
    state: ResumeState,
    updates: dict[str, Any],
    trace: list[str],
) -> ResumeState:
    """通过 LangChain Tool 合并本轮结构化信息。

    Args:
        state: 当前简历状态。
        updates: 本轮字段补丁。
        trace: Agent 执行轨迹。

    Returns:
        合并后的简历状态。
    """

    state_json = collect_resume_info_tool.invoke(
        {
            "current_state_json": state.model_dump_json(ensure_ascii=False),
            "update_json": json.dumps(updates, ensure_ascii=False),
        }
    )
    trace.append("调用工具：collect_resume_info")
    return coerce_resume_state(state_json)


def _validate_with_tool(state: ResumeState, trace: list[str]) -> dict[str, Any]:
    """通过 LangChain Tool 执行简历底线校验。

    Args:
        state: 当前简历状态。
        trace: Agent 执行轨迹。

    Returns:
        缺失字段、格式错误和生成状态报告。
    """

    report_json = validate_resume_state_tool.invoke(
        {"current_state_json": state.model_dump_json(ensure_ascii=False)}
    )
    trace.append("调用工具：validate_resume_state")
    return json.loads(report_json)


def _fill_template_with_tool(
    state: ResumeState,
    trace: list[str],
    output_path: Path | str | None = None,
) -> dict[str, str]:
    """通过 LangChain Tool 渲染并保存 Markdown 简历。

    Args:
        state: 待渲染的完整简历状态。
        trace: Agent 执行轨迹。
        output_path: 可选的 Markdown 输出路径。

    Returns:
        Markdown 内容和输出路径。
    """

    result_json = fill_resume_template_tool.invoke(
        {
            "current_state_json": state.model_dump_json(ensure_ascii=False),
            "output_path": str(output_path) if output_path else "",
        }
    )
    result = json.loads(result_json)
    trace.append("调用工具：fill_resume_template")
    return {
        "markdown": str(result["markdown"]),
        "output_path": str(result["output_path"]),
    }


class ResumeAgentService:
    """面向 UI 和脚本的简历 Agent 服务。

    ResumeState 负责跨轮保存简历业务事实，主对话 Agent 的 checkpointer
    仅负责理解代词、省略和上下文衔接；导入、润色和评分均为一次性任务。
    """

    def __init__(
        self,
        use_llm: bool = True,
        config: AppConfig | None = None,
        use_agent_driver: bool = True,
    ) -> None:
        """初始化简历 Agent 服务。

        Args:
            use_llm: 是否启用 LLM 抽取、追问与润色。
            config: 可选应用配置。
            use_agent_driver: 是否使用 LangChain 结构化 Agent 驱动主流程。

        Returns:
            None。
        """

        self.config = config or load_config()
        self.llm = build_chat_model(self.config) if use_llm else None
        self.turn_agent = build_langchain_agent(self.llm) if use_agent_driver else None
        self.polish_agent = build_polish_agent(self.llm) if use_agent_driver else None
        self.import_agent = build_import_agent(self.llm) if use_agent_driver else None
        self.score_agent = build_score_agent(self.llm) if use_agent_driver else None
        self.thread_id = str(uuid.uuid4())

    def handle_message(
        self,
        user_input: str,
        state: ResumeState | dict[str, Any] | str | None,
    ) -> AgentTurnResult:
        """处理单轮用户消息。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            单轮处理结果。
        """

        resume_state = coerce_resume_state(state)
        trace: list[str] = []
        initial_report = _validate_with_tool(resume_state, trace)

        decision = self._decide_with_llm(user_input, resume_state, initial_report)
        if decision is None:
            decision = self._fallback_decision(user_input, resume_state, initial_report)
            trace.append("fallback: minimal_rules")
        else:
            trace.append("LLM 结构化决策：ResumeTurnDecision")

        patch = _compact_patch(decision.patch) or {}
        if patch:
            resume_state = _collect_with_tool(resume_state, patch, trace)
            report = _validate_with_tool(resume_state, trace)
        else:
            report = initial_report

        resume_state = _sync_stage(resume_state, report)
        report["current_stage"] = resume_state.current_stage

        should_generate = _contains_generate_intent(user_input) or decision.intent == "generate_resume"
        if should_generate:
            if not report["is_ready"]:
                message = f"现在还不能生成完整简历，仍需补充：{'；'.join(report['missing_fields'])}\n\n{_decision_message(decision, report)}"
                return AgentTurnResult(message, resume_state, report, agent_trace=trace)

            polished_state = self._polish_state_before_generation(resume_state, trace)
            polished_report = _validate_with_tool(polished_state, trace)
            if not polished_report["is_ready"]:
                message = f"生成前校验发现还缺少：{'；'.join(polished_report['missing_fields'])}\n\n{_build_fallback_question(polished_report)}"
                return AgentTurnResult(message, polished_state, polished_report, agent_trace=trace)

            result = _fill_template_with_tool(polished_state, trace)
            message = f"已生成 Markdown 简历：{result['output_path']}\n\n{result['markdown']}"
            return AgentTurnResult(
                assistant_message=message,
                state=polished_state,
                missing_report=polished_report,
                resume_markdown=result["markdown"],
                output_path=result["output_path"],
                agent_trace=trace,
            )

        message = _decision_message(decision, report)
        if report["is_ready"] and "生成简历" not in message:
            message = f"{message}\n\n必要信息已完整，可以回复“生成简历”输出 Markdown 简历。"
        return AgentTurnResult(message, resume_state, report, agent_trace=trace)

    def optimize_existing_resume(
        self,
        markdown_text: str,
        output_path: Path | str | None = None,
    ) -> ExistingResumeOptimizationResult:
        """解析并优化已有 Markdown 简历。

        Args:
            markdown_text: 用户上传或传入的 Markdown 简历文本。
            output_path: 可选的优化结果输出路径。

        Returns:
            已有简历优化结果。

        Raises:
            ValueError: 简历文本为空。
        """

        if not markdown_text.strip():
            raise ValueError("已有简历内容为空，无法解析优化。")

        trace: list[str] = []
        import_result = self._import_resume_with_llm(markdown_text, trace)
        if import_result is None:
            parsed_state = _parse_existing_resume_fallback(markdown_text)
            summary = "已使用规则兜底解析 Markdown 简历。"
            trace.append("fallback: parse_existing_resume")
        else:
            parsed_state = import_result.state
            summary = import_result.summary or "已使用 LLM 解析已有 Markdown 简历。"

        parsed_report = _validate_with_tool(parsed_state, trace)
        parsed_state = _sync_stage(parsed_state, parsed_report)

        optimized_state = self._polish_state_before_generation(parsed_state, trace)
        optimized_report = _validate_with_tool(optimized_state, trace)
        optimized_state = _sync_stage(optimized_state, optimized_report)
        result = _fill_template_with_tool(optimized_state, trace, output_path=output_path)

        return ExistingResumeOptimizationResult(
            state=optimized_state,
            markdown=result["markdown"],
            output_path=result["output_path"],
            summary=summary,
            missing_report=optimized_report,
            agent_trace=trace,
        )

    def score_resume(
        self,
        state: ResumeState | dict[str, Any] | str | None,
        target_position: str = "",
        source_markdown: str = "",
    ) -> ResumeScoreResult:
        """对结构化简历状态进行评分。

        Args:
            state: 待评分的结构化简历状态。
            target_position: 可选目标岗位；为空时使用状态中的目标岗位。
            source_markdown: 可选的原始 Markdown 简历，用于表达规范性评估。

        Returns:
            简历评分结果。
        """

        resume_state = coerce_resume_state(state)
        trace: list[str] = []
        validation_report = _validate_with_tool(resume_state, trace)
        completeness_score = _calculate_completeness_score(resume_state, validation_report)
        target = target_position.strip() or resume_state.job_intention.target_position or "学生求职/实习"

        score_report = self._score_resume_with_llm(
            resume_state,
            validation_report,
            completeness_score,
            target,
            source_markdown,
            trace,
        )
        if score_report is None:
            score_report = _fallback_score_report(resume_state, completeness_score, validation_report)
            trace.append("fallback: score_resume")
        else:
            score_report = _normalize_score_report(score_report, completeness_score)

        return ResumeScoreResult(
            report=score_report,
            markdown=_score_report_to_markdown(score_report),
            agent_trace=trace,
        )

    def score_existing_resume(
        self,
        markdown_text: str,
        target_position: str = "",
    ) -> ResumeScoreResult:
        """解析上传的 Markdown 简历并生成评分报告。

        Args:
            markdown_text: 待评分的 Markdown 简历内容。
            target_position: 可选评分目标岗位；为空时使用简历中的求职意向。

        Returns:
            基于上传简历生成的评分报告。

        Raises:
            ValueError: 上传内容为空时抛出。
        """

        if not markdown_text.strip():
            raise ValueError("上传的 Markdown 简历为空，无法评分。")

        trace: list[str] = []
        import_result = self._import_resume_with_llm(markdown_text, trace)
        if import_result is None:
            parsed_state = _parse_existing_resume_fallback(markdown_text)
            trace.append("fallback: parse_existing_resume")
        else:
            parsed_state = import_result.state

        score_result = self.score_resume(
            parsed_state,
            target_position=target_position,
            source_markdown=markdown_text,
        )
        score_result.agent_trace = trace + score_result.agent_trace
        return score_result

    def _decide_with_llm(
        self,
        user_input: str,
        state: ResumeState,
        report: dict[str, Any],
    ) -> ResumeTurnDecision | None:
        """调用 LLM 生成单轮结构化决策。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。
            report: 当前底线校验报告。

        Returns:
            LLM 决策；失败时返回 None。
        """

        if self.turn_agent is not None:
            prompt = TURN_DECISION_USER_PROMPT.format(
                state_json=state.model_dump_json(ensure_ascii=False),
                validation_report=json.dumps(report, ensure_ascii=False),
                user_input=user_input,
            )
            try:
                raw_result = self.turn_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": f"{self.thread_id}:turn"},
                        "recursion_limit": 6,
                    },
                )
                structured = _extract_structured_response(raw_result, ResumeTurnDecision)
                if isinstance(structured, ResumeTurnDecision):
                    return structured
            except Exception:
                pass

        if self.llm is None:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TURN_DECISION_SYSTEM_PROMPT),
                ("human", TURN_DECISION_USER_PROMPT),
            ]
        )
        try:
            response = (prompt | self.llm).invoke(
                {
                    "state_json": state.model_dump_json(ensure_ascii=False),
                    "validation_report": json.dumps(report, ensure_ascii=False),
                    "user_input": user_input,
                }
            )
            content = getattr(response, "content", str(response))
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            parsed = parse_json_object(str(content))
            return ResumeTurnDecision.model_validate(parsed) if parsed else None
        except Exception:
            return None

    def _import_resume_with_llm(
        self,
        markdown_text: str,
        trace: list[str],
    ) -> ResumeImportResult | None:
        """调用 LLM 解析已有 Markdown 简历。

        Args:
            markdown_text: 已有 Markdown 简历文本。
            trace: 执行轨迹列表。

        Returns:
            解析结果；失败时返回 None。
        """

        if self.import_agent is not None:
            prompt = IMPORT_RESUME_USER_PROMPT.format(resume_markdown=markdown_text)
            try:
                raw_result = self.import_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": f"{self.thread_id}:import:{uuid.uuid4()}"},
                        "recursion_limit": 6,
                    },
                )
                structured = _extract_structured_response(raw_result, ResumeImportResult)
                if isinstance(structured, ResumeImportResult):
                    trace.append("LLM 结构化解析：ResumeImportResult")
                    return structured
            except Exception:
                pass

        if self.llm is None:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", IMPORT_RESUME_SYSTEM_PROMPT),
                ("human", IMPORT_RESUME_USER_PROMPT),
            ]
        )
        try:
            response = (prompt | self.llm).invoke({"resume_markdown": markdown_text})
            content = getattr(response, "content", str(response))
            parsed = parse_json_object(str(content))
            if parsed:
                trace.append("LLM JSON 解析：ResumeImportResult")
                return ResumeImportResult.model_validate(parsed)
        except Exception:
            return None
        return None

    def _score_resume_with_llm(
        self,
        state: ResumeState,
        validation_report: dict[str, Any],
        completeness_score: int,
        target_position: str,
        source_markdown: str,
        trace: list[str],
    ) -> ResumeScoreReport | None:
        """调用 LLM 生成简历评分报告。

        Args:
            state: 当前简历状态。
            validation_report: 底线校验报告。
            completeness_score: 代码计算的完整度分。
            target_position: 评分使用的目标岗位。
            source_markdown: 上传的原始 Markdown 简历内容。
            trace: 执行轨迹列表。

        Returns:
            评分报告；失败时返回 None。
        """

        prompt_payload = {
            "source_markdown": source_markdown.strip() or "未提供原始 Markdown，仅依据结构化状态评分。",
            "state_json": state.model_dump_json(ensure_ascii=False),
            "validation_report": json.dumps(validation_report, ensure_ascii=False),
            "completeness_score": str(completeness_score),
            "target_position": target_position,
        }

        if self.score_agent is not None:
            prompt = RESUME_SCORE_USER_PROMPT.format(**prompt_payload)
            try:
                raw_result = self.score_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": f"{self.thread_id}:score:{uuid.uuid4()}"},
                        "recursion_limit": 6,
                    },
                )
                structured = _extract_structured_response(raw_result, ResumeScoreReport)
                if isinstance(structured, ResumeScoreReport):
                    trace.append("LLM 结构化评分：ResumeScoreReport")
                    return structured
            except Exception:
                pass

        if self.llm is None:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESUME_SCORE_SYSTEM_PROMPT),
                ("human", RESUME_SCORE_USER_PROMPT),
            ]
        )
        try:
            response = (prompt | self.llm).invoke(prompt_payload)
            content = getattr(response, "content", str(response))
            parsed = parse_json_object(str(content))
            if parsed:
                trace.append("LLM JSON 评分：ResumeScoreReport")
                return ResumeScoreReport.model_validate(parsed)
        except Exception:
            return None
        return None

    def _fallback_decision(
        self,
        user_input: str,
        state: ResumeState,
        report: dict[str, Any],
    ) -> ResumeTurnDecision:
        """生成无 LLM 时的最小兜底决策。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。
            report: 当前底线校验报告。

        Returns:
            兜底单轮决策。
        """

        if _is_greeting_only(user_input):
            return ResumeTurnDecision(
                intent="greeting",
                assistant_message="你好，我会通过多轮对话帮你生成学生简历。你可以先自然描述求职方向、基本信息、教育背景或项目经历。",
            )

        patch = _basic_fallback_extract(user_input, state) if user_input.strip() else {}
        if _contains_generate_intent(user_input):
            return ResumeTurnDecision(intent="generate_resume", patch=patch, assistant_message=_build_fallback_question(report))
        return ResumeTurnDecision(intent="collect_info", patch=patch, assistant_message=_build_fallback_question(report))

    def _polish_state_before_generation(self, state: ResumeState, trace: list[str]) -> ResumeState:
        """在模板填充前执行 LLM 结构化清洗。

        Args:
            state: 待生成简历的状态。
            trace: 本轮轨迹列表。

        Returns:
            清洗后的简历状态。
        """

        if self.polish_agent is not None:
            prompt = FINAL_POLISH_USER_PROMPT.format(state_json=state.model_dump_json(ensure_ascii=False))
            try:
                raw_result = self.polish_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": f"{self.thread_id}:polish:{uuid.uuid4()}"},
                        "recursion_limit": 6,
                    },
                )
                structured = _extract_structured_response(raw_result, ResumePolishResult)
                if isinstance(structured, ResumePolishResult):
                    trace.append("LLM 结构化清洗：ResumePolishResult")
                    polished_state = structured.state.model_copy(deep=True)
                    polished_state.touch()
                    return polished_state
            except Exception:
                pass

        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", FINAL_POLISH_SYSTEM_PROMPT),
                    ("human", FINAL_POLISH_USER_PROMPT),
                ]
            )
            try:
                response = (prompt | self.llm).invoke({"state_json": state.model_dump_json(ensure_ascii=False)})
                content = getattr(response, "content", str(response))
                parsed = parse_json_object(str(content))
                if parsed:
                    result = ResumePolishResult.model_validate(parsed)
                    trace.append("LLM JSON 清洗：ResumePolishResult")
                    polished_state = result.state.model_copy(deep=True)
                    polished_state.touch()
                    return polished_state
            except Exception:
                pass

        trace.append("fallback: polish_state_experiences")
        return polish_state_experiences(state, self.llm, force=True)
