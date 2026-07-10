"""学生简历生成智能体的 LLM 主导编排逻辑。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.config import AppConfig, load_config
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
from app.schema import ResumeState
from app.tools import (
    check_missing_fields,
    collect_resume_info,
    coerce_resume_state,
    fill_resume_template,
    parse_json_object,
    polish_state_experiences,
)


GENERATE_KEYWORDS = ("生成简历", "输出简历", "导出简历", "完成简历")
GREETING_KEYWORDS = {"你好", "您好", "hi", "hello", "嗨", "在吗", "开始"}


class ResumeTurnDecision(BaseModel):
    """LLM 对单轮用户输入的结构化决策。"""

    intent: Literal["collect_info", "generate_resume", "greeting", "chat"] = Field(
        default="collect_info",
        description="本轮用户意图。",
    )
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="从用户输入中抽取出的 ResumeState 增量补丁，只包含用户明确提供的信息。",
    )
    assistant_message: str = Field(
        default="",
        description="本轮要回复给用户的自然语言消息。",
    )
    followup_questions: list[str] = Field(
        default_factory=list,
        description="下一步建议追问的问题，每轮最多 3 个。",
    )
    ready_to_generate_reason: str = Field(
        default="",
        description="如果认为信息完整或用户请求生成，说明理由；不完整时留空。",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM 对本轮抽取和回复规划的置信度。",
    )


class ResumePolishResult(BaseModel):
    """LLM 对生成前简历状态的结构化清洗结果。"""

    state: ResumeState = Field(description="去重、润色后的简历状态。")
    summary: str = Field(default="", description="本次清洗的简短说明。")


class ResumeImportResult(BaseModel):
    """LLM 对已有 Markdown 简历的结构化解析结果。"""

    state: ResumeState = Field(description="从已有简历解析出的简历状态。")
    summary: str = Field(default="", description="已有简历内容概述。")
    optimization_notes: list[str] = Field(default_factory=list, description="解析阶段发现的可优化问题。")


class ResumeScoreReport(BaseModel):
    """简历评分结构化报告。"""

    completeness_score: int = Field(default=0, ge=0, le=100, description="完整度评分。")
    match_score: int = Field(default=0, ge=0, le=100, description="岗位匹配度评分。")
    expression_score: int = Field(default=0, ge=0, le=100, description="表达规范性评分。")
    total_score: int = Field(default=0, ge=0, le=100, description="综合评分。")
    strengths: list[str] = Field(default_factory=list, description="简历优势。")
    weaknesses: list[str] = Field(default_factory=list, description="主要问题。")
    suggestions: list[str] = Field(default_factory=list, description="优化建议。")
    summary: str = Field(default="", description="评分总结。")


@dataclass
class AgentTurnResult:
    """单轮对话处理结果。

    Args:
        assistant_message: 返回给用户的消息。
        state: 更新后的简历状态。
        missing_report: 缺失字段与质量检查报告。
        resume_markdown: 生成的 Markdown 简历内容。
        output_path: 生成文件路径。
        agent_trace: 本轮关键步骤轨迹。
    """

    assistant_message: str
    state: ResumeState
    missing_report: dict[str, Any]
    resume_markdown: str = ""
    output_path: str = ""
    agent_trace: list[str] = field(default_factory=list)


@dataclass
class ExistingResumeOptimizationResult:
    """已有简历解析与优化结果。

    Args:
        state: 优化后的结构化简历状态。
        markdown: 优化后的 Markdown 简历。
        output_path: 输出文件路径。
        summary: 解析与优化摘要。
        missing_report: 底线校验报告。
        agent_trace: 执行轨迹。
    """

    state: ResumeState
    markdown: str
    output_path: str
    summary: str
    missing_report: dict[str, Any]
    agent_trace: list[str] = field(default_factory=list)


@dataclass
class ResumeScoreResult:
    """简历评分结果。

    Args:
        report: 结构化评分报告。
        markdown: Markdown 格式评分报告。
        agent_trace: 执行轨迹。
    """

    report: ResumeScoreReport
    markdown: str
    agent_trace: list[str] = field(default_factory=list)


def build_chat_model(config: AppConfig | None = None) -> BaseChatModel | None:
    """构建 OpenAI 兼容的 LangChain 聊天模型。

    Args:
        config: 应用配置；为空时自动从 `.env` 加载。

    Returns:
        聊天模型实例；缺少 API Key 时返回 None。
    """

    app_config = config or load_config()
    if not app_config.api_key:
        return None

    http_client = httpx.Client(
        verify=app_config.ssl_verify,
        timeout=30,
        trust_env=True,
    )

    return ChatOpenAI(
        model=app_config.model,
        api_key=app_config.api_key,
        base_url=app_config.base_url,
        temperature=app_config.temperature,
        timeout=30,
        max_retries=2,
        extra_body={"enable_thinking": app_config.enable_thinking},
        http_client=http_client,
        http_socket_options=(),
    )


def build_langchain_agent(llm: BaseChatModel | None = None) -> Any | None:
    """构建结构化单轮决策 Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        使用 `ResumeTurnDecision` 作为结构化输出的 Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=TURN_DECISION_SYSTEM_PROMPT,
        response_format=ToolStrategy(ResumeTurnDecision),
        checkpointer=InMemorySaver(),
    )


def _build_polish_agent(llm: BaseChatModel | None = None) -> Any | None:
    """构建生成前简历清洗 Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        使用 `ResumePolishResult` 作为结构化输出的 Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=FINAL_POLISH_SYSTEM_PROMPT,
        response_format=ToolStrategy(ResumePolishResult),
        checkpointer=InMemorySaver(),
    )


def _build_import_agent(llm: BaseChatModel | None = None) -> Any | None:
    """构建已有简历解析 Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        使用 `ResumeImportResult` 作为结构化输出的 Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=IMPORT_RESUME_SYSTEM_PROMPT,
        response_format=ToolStrategy(ResumeImportResult),
        checkpointer=InMemorySaver(),
    )


def _build_score_agent(llm: BaseChatModel | None = None) -> Any | None:
    """构建简历评分 Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        使用 `ResumeScoreReport` 作为结构化输出的 Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=RESUME_SCORE_SYSTEM_PROMPT,
        response_format=ToolStrategy(ResumeScoreReport),
        checkpointer=InMemorySaver(),
    )


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


def _split_items(text: str) -> list[str]:
    """按常见中英文分隔符拆分条目。

    Args:
        text: 原始文本。

    Returns:
        清洗后的条目列表。
    """

    parts = re.split(r"[、,，;；/|\n]+", text)
    return [part.strip(" ：:。.") for part in parts if part.strip(" ：:。.")]


def _strip_markdown_label(line: str, label: str) -> str:
    """从 Markdown 列表行中去掉加粗标签。

    Args:
        line: Markdown 行文本。
        label: 标签名称。

    Returns:
        标签后的字段内容。
    """

    pattern = rf"^\s*-\s*\*\*{re.escape(label)}\*\*[:：]\s*(.+?)\s*$"
    match = re.search(pattern, line)
    return match.group(1).strip() if match else ""


def _extract_markdown_section(markdown_text: str, heading: str) -> str:
    """提取指定二级标题下的 Markdown 内容。

    Args:
        markdown_text: 完整 Markdown 文本。
        heading: 二级标题名称。

    Returns:
        标题下方内容；不存在时返回空字符串。
    """

    pattern = rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_titled_markdown_blocks(section_text: str) -> list[tuple[str, list[str]]]:
    """解析由加粗标题和 bullet 组成的 Markdown 块。

    Args:
        section_text: Markdown 小节文本。

    Returns:
        标题与 bullet 列表。
    """

    blocks: list[tuple[str, list[str]]] = []
    current_title = ""
    current_items: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        title_match = re.fullmatch(r"\*\*(.+?)\*\*", line)
        if title_match:
            if current_title:
                blocks.append((current_title, current_items))
            current_title = title_match.group(1).strip()
            current_items = []
            continue
        if line.startswith("-"):
            current_items.append(line.lstrip("-").strip())
    if current_title:
        blocks.append((current_title, current_items))
    return blocks


def _parse_existing_resume_fallback(markdown_text: str) -> ResumeState:
    """使用轻量规则解析标准 Markdown 简历。

    Args:
        markdown_text: 已有 Markdown 简历。

    Returns:
        解析出的简历状态。
    """

    state = ResumeState()
    update: dict[str, Any] = {}
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    title_match = re.search(r"^#\s+(.+?)\s*$", markdown_text, flags=re.MULTILINE)
    if title_match:
        update.setdefault("basic_info", {})["name"] = title_match.group(1).strip()

    for line in lines:
        job_text = _strip_markdown_label(line, "求职意向")
        if job_text:
            parts = [part.strip() for part in job_text.split("|")]
            update["job_intention"] = {
                "target_position": parts[0] if len(parts) > 0 else "",
                "target_industry": parts[1] if len(parts) > 1 else "",
                "expected_city": parts[2] if len(parts) > 2 else "",
            }
        for label, field_name in {"电话": "phone", "邮箱": "email", "籍贯": "native_place"}.items():
            value = _strip_markdown_label(line, label)
            if value:
                update.setdefault("basic_info", {})[field_name] = value

    education_text = _extract_markdown_section(markdown_text, "教育背景")
    education_update: dict[str, Any] = {}
    if education_text:
        education_lines = [line.strip() for line in education_text.splitlines() if line.strip()]
        header = next((line for line in education_lines if not line.startswith("-")), "")
        header_match = re.match(r"(.+?)\s+(.+?)\s+(.+?)专业", header)
        if header_match:
            school, college, major = [item.strip() for item in header_match.groups()]
            education_update.update({"school": school, "college": college, "major": major})
            update.setdefault("basic_info", {}).update({"university": school, "major": major})
        for line in education_lines:
            for label, field_name in {
                "专业排名": "gpa_or_rank",
                "英语水平": "english_level",
            }.items():
                value = _strip_markdown_label(line, label)
                if value:
                    education_update[field_name] = value
            courses = _strip_markdown_label(line, "核心课程")
            if courses:
                education_update["courses"] = _split_items(courses)
            tech_stack = _strip_markdown_label(line, "技术栈")
            if tech_stack:
                skill_items = _split_items(tech_stack)
                languages = {"Python", "Java", "C++", "JavaScript", "TypeScript", "Go"}
                update["skills"] = {
                    "programming_languages": [item for item in skill_items if item in languages],
                    "tools": [item for item in skill_items if item not in languages],
                }
    if education_update:
        update["education"] = education_update

    project_blocks = _parse_titled_markdown_blocks(_extract_markdown_section(markdown_text, "项目经历"))
    if project_blocks:
        update["projects"] = [
            {
                "title": title,
                "raw_description": "；".join(items),
                "responsibilities": items,
                "results": [item for item in items if re.search(r"\d|%|提升|准确率|响应|完成|排名|F1", item, flags=re.IGNORECASE)],
            }
            for title, items in project_blocks
        ]

    award_blocks = _parse_titled_markdown_blocks(_extract_markdown_section(markdown_text, "竞赛获奖"))
    if award_blocks:
        update["awards"] = [
            {
                "name": title,
                "description": "；".join(items),
                "highlights": items,
            }
            for title, items in award_blocks
        ]

    self_text = _extract_markdown_section(markdown_text, "自我评价")
    if self_text:
        update["self_evaluation"] = "；".join(
            line.lstrip("-").strip()
            for line in self_text.splitlines()
            if line.strip().lstrip("-").strip()
        )

    return collect_resume_info(state, update)


def _calculate_completeness_score(state: ResumeState, report: dict[str, Any]) -> int:
    """计算简历完整度评分。

    Args:
        state: 当前简历状态。
        report: 底线校验报告。

    Returns:
        0-100 的完整度分数。
    """

    checks = [
        bool(state.basic_info.name),
        bool(state.basic_info.phone),
        bool(state.basic_info.email),
        bool(state.basic_info.native_place),
        bool(state.job_intention.target_position),
        bool(state.job_intention.target_industry),
        bool(state.job_intention.expected_city),
        bool(state.education.school or state.basic_info.university),
        bool(state.education.college),
        bool(state.education.major or state.basic_info.major),
        bool(state.education.courses),
        bool(state.education.gpa_or_rank),
        bool(state.education.english_level),
        bool(state.skills.programming_languages or state.skills.tools or state.skills.professional_skills),
        bool(state.projects),
        any(project.technologies for project in state.projects),
        any(project.responsibilities or project.polished_bullets for project in state.projects),
        any(project.results or project.polished_bullets for project in state.projects),
        bool(state.awards),
        bool(state.self_evaluation),
    ]
    base_score = round(sum(checks) / len(checks) * 100)
    penalty = min(len(report.get("validation_errors", [])) * 8, 20)
    return max(0, min(100, base_score - penalty))


def _normalize_score_report(report: ResumeScoreReport, completeness_score: int) -> ResumeScoreReport:
    """规整评分报告并重算综合分。

    Args:
        report: 原始评分报告。
        completeness_score: 代码计算的完整度分。

    Returns:
        规整后的评分报告。
    """

    normalized = report.model_copy(deep=True)
    normalized.completeness_score = completeness_score
    normalized.match_score = max(0, min(100, normalized.match_score))
    normalized.expression_score = max(0, min(100, normalized.expression_score))
    normalized.total_score = round(
        normalized.completeness_score * 0.35
        + normalized.match_score * 0.35
        + normalized.expression_score * 0.30
    )
    return normalized


def _fallback_score_report(state: ResumeState, completeness_score: int, report: dict[str, Any]) -> ResumeScoreReport:
    """生成无 LLM 时的兜底评分报告。

    Args:
        state: 当前简历状态。
        completeness_score: 代码计算的完整度分。
        report: 底线校验报告。

    Returns:
        兜底评分报告。
    """

    match_score = 70
    if state.job_intention.target_position and state.projects:
        match_score += 10
    if state.skills.programming_languages or state.skills.tools or state.skills.professional_skills:
        match_score += 8
    if state.awards:
        match_score += 5
    expression_score = 72
    if any(project.polished_bullets for project in state.projects):
        expression_score += 12
    if report.get("quality_questions"):
        expression_score -= min(len(report["quality_questions"]) * 5, 15)

    weaknesses = list(report.get("missing_fields", []))[:3] or list(report.get("quality_questions", []))[:3]
    if not weaknesses:
        weaknesses = ["可继续补充更多量化成果或第二段项目经历，增强竞争力。"]

    return _normalize_score_report(
        ResumeScoreReport(
            completeness_score=completeness_score,
            match_score=max(0, min(100, match_score)),
            expression_score=max(0, min(100, expression_score)),
            strengths=[
                "简历已覆盖教育背景、项目经历、技能与获奖等核心模块。",
                "结构化信息可直接用于模板生成和后续岗位定制。",
            ],
            weaknesses=weaknesses,
            suggestions=[
                "优先补充项目中的个人职责、技术方法和量化成果。",
                "将项目 bullet 改写为完整简历句，避免短语堆叠。",
                "围绕目标岗位补充更匹配的技能关键词。",
            ],
            summary="已基于结构完整度和现有字段生成兜底评分。",
        ),
        completeness_score,
    )


def _score_report_to_markdown(report: ResumeScoreReport) -> str:
    """将评分报告格式化为 Markdown。

    Args:
        report: 结构化评分报告。

    Returns:
        Markdown 评分报告。
    """

    def list_block(items: list[str]) -> str:
        """格式化列表字段。

        Args:
            items: 文本列表。

        Returns:
            Markdown 列表文本。
        """

        return "\n".join(f"- {item}" for item in items) if items else "- 暂无"

    return "\n".join(
        [
            "## 简历评分报告",
            "",
            f"- **综合评分**：{report.total_score}/100",
            f"- **完整度**：{report.completeness_score}/100",
            f"- **岗位匹配度**：{report.match_score}/100",
            f"- **表达规范性**：{report.expression_score}/100",
            "",
            "### 优势",
            list_block(report.strengths),
            "",
            "### 主要问题",
            list_block(report.weaknesses),
            "",
            "### 优化建议",
            list_block(report.suggestions),
            "",
            "### 总结",
            report.summary or "暂无总结。",
        ]
    )


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
    if matched.get("major"):
        basic["major"] = matched["major"]
    if matched.get("school"):
        basic["university"] = matched["school"]

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
    if university_match and not education.get("school") and not (state.education.school or state.basic_info.university):
        education["school"] = university_match.group(1)
        basic.setdefault("university", university_match.group(1))

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


class ResumeAgentService:
    """面向 UI 和脚本的简历 Agent 服务。"""

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
        self.polish_agent = _build_polish_agent(self.llm) if use_agent_driver else None
        self.import_agent = _build_import_agent(self.llm) if use_agent_driver else None
        self.score_agent = _build_score_agent(self.llm) if use_agent_driver else None
        self.thread_id = str(uuid.uuid4())
        self.use_agent_driver = use_agent_driver
        self.recent_turns: list[dict[str, str]] = []

    def extract_update(self, user_input: str, state: ResumeState) -> dict[str, Any]:
        """抽取用户本轮提供的简历字段更新。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            字段更新字典。
        """

        decision = self._decide_with_llm(user_input, state, check_missing_fields(state))
        if decision is not None:
            return _compact_patch(decision.patch) or {}
        return _basic_fallback_extract(user_input, state)

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
        initial_report = check_missing_fields(resume_state)
        trace: list[str] = []

        decision = self._decide_with_llm(user_input, resume_state, initial_report)
        if decision is None:
            decision = self._fallback_decision(user_input, resume_state, initial_report)
            trace.append("fallback: minimal_rules")
        else:
            trace.append("LLM 结构化决策：ResumeTurnDecision")

        patch = _compact_patch(decision.patch) or {}
        if patch:
            resume_state = collect_resume_info(resume_state, patch)
            trace.append("调用工具：collect_resume_info")

        report = check_missing_fields(resume_state)
        trace.append("调用工具：validate_resume_state")
        resume_state = _sync_stage(resume_state, report)
        report = check_missing_fields(resume_state)

        should_generate = _contains_generate_intent(user_input) or decision.intent == "generate_resume"
        if should_generate:
            if not report["is_ready"]:
                message = f"现在还不能生成完整简历，仍需补充：{'；'.join(report['missing_fields'])}\n\n{_decision_message(decision, report)}"
                self._remember_turn(user_input, message)
                return AgentTurnResult(message, resume_state, report, agent_trace=trace)

            polished_state = self._polish_state_before_generation(resume_state, trace)
            polished_report = check_missing_fields(polished_state)
            if not polished_report["is_ready"]:
                message = f"生成前校验发现还缺少：{'；'.join(polished_report['missing_fields'])}\n\n{_build_fallback_question(polished_report)}"
                self._remember_turn(user_input, message)
                return AgentTurnResult(message, polished_state, polished_report, agent_trace=trace)

            result = fill_resume_template(polished_state)
            trace.append("调用工具：fill_resume_template")
            message = f"已生成 Markdown 简历：{result['output_path']}\n\n{result['markdown']}"
            self._remember_turn(user_input, message)
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
        self._remember_turn(user_input, message)
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

        parsed_report = check_missing_fields(parsed_state)
        parsed_state = _sync_stage(parsed_state, parsed_report)
        trace.append("调用工具：validate_resume_state")

        optimized_state = self._polish_state_before_generation(parsed_state, trace)
        optimized_report = check_missing_fields(optimized_state)
        optimized_state = _sync_stage(optimized_state, optimized_report)
        result = fill_resume_template(optimized_state, output_path=output_path)
        trace.append("调用工具：fill_resume_template")

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
        validation_report = check_missing_fields(resume_state)
        completeness_score = _calculate_completeness_score(resume_state, validation_report)
        target = target_position.strip() or resume_state.job_intention.target_position or "学生求职/实习"
        trace: list[str] = ["调用工具：validate_resume_state"]

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
                recent_turns=json.dumps(self.recent_turns[-6:], ensure_ascii=False),
                user_input=user_input,
            )
            try:
                raw_result = self.turn_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": f"{self.thread_id}:turn:{uuid.uuid4()}"},
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
                    "recent_turns": json.dumps(self.recent_turns[-6:], ensure_ascii=False),
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

    def _remember_turn(self, user_input: str, assistant_message: str) -> None:
        """记录最近对话摘要供下一轮 LLM 使用。

        Args:
            user_input: 用户输入。
            assistant_message: 助手回复。

        Returns:
            None。
        """

        self.recent_turns.append(
            {
                "user": user_input,
                "assistant": assistant_message[:600],
            }
        )
        self.recent_turns = self.recent_turns[-8:]

    def _read_generated_markdown(self, output_path: str) -> str:
        """从输出文件读取 Markdown 简历。

        Args:
            output_path: Markdown 文件路径。

        Returns:
            Markdown 文本；读取失败时返回空字符串。
        """

        try:
            return Path(output_path).read_text(encoding="utf-8")
        except OSError:
            return ""
