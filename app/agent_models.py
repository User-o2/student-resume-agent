"""Agent 结构化输出与服务返回值模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schema import ResumeState


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


class ResumePolishResult(BaseModel):
    """LLM 对生成前简历状态的结构化清洗结果。"""

    state: ResumeState = Field(description="去重、润色后的简历状态。")


class ResumeImportResult(BaseModel):
    """LLM 对已有 Markdown 简历的结构化解析结果。"""

    state: ResumeState = Field(description="从已有简历解析出的简历状态。")
    summary: str = Field(default="", description="已有简历内容概述。")


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
