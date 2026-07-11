"""简历确定性评分、兜底报告与 Markdown 格式化逻辑。"""

from __future__ import annotations

from typing import Any

from app.agent_models import ResumeScoreReport
from app.schema import ResumeState


# 综合分由代码统一重算，避免不同模型返回互相矛盾的 total_score。
COMPLETENESS_WEIGHT = 0.35
MATCH_WEIGHT = 0.35
EXPRESSION_WEIGHT = 0.30


def calculate_completeness_score(state: ResumeState, report: dict[str, Any]) -> int:
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
        bool(state.education.school),
        bool(state.education.college),
        bool(state.education.major),
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


def normalize_score_report(report: ResumeScoreReport, completeness_score: int) -> ResumeScoreReport:
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
        normalized.completeness_score * COMPLETENESS_WEIGHT
        + normalized.match_score * MATCH_WEIGHT
        + normalized.expression_score * EXPRESSION_WEIGHT
    )
    return normalized


def build_fallback_score_report(
    state: ResumeState,
    completeness_score: int,
    report: dict[str, Any],
) -> ResumeScoreReport:
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

    return normalize_score_report(
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


def score_report_to_markdown(report: ResumeScoreReport) -> str:
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
