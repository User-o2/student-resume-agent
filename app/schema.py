"""简历结构化状态的数据模型定义。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def now_text() -> str:
    """生成秒级 ISO 时间文本。

    Args:
        无。

    Returns:
        当前时间字符串。
    """

    return datetime.now().isoformat(timespec="seconds")


class BasicInfo(BaseModel):
    """学生基本信息。"""

    name: str = ""
    university: str = ""
    major: str = ""
    grade: str = ""
    phone: str = ""
    email: str = ""


class JobIntention(BaseModel):
    """求职意向信息。"""

    target_position: str = ""
    target_industry: str = ""
    expected_city: str = ""


class Education(BaseModel):
    """教育背景信息。"""

    school: str = ""
    major: str = ""
    courses: list[str] = Field(default_factory=list)
    gpa_or_rank: str = ""


class Experience(BaseModel):
    """项目、实习或实践经历。"""

    title: str = ""
    organization: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    technologies: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    raw_description: str = ""
    polished_bullets: list[str] = Field(default_factory=list)


class Skills(BaseModel):
    """技能特长信息。"""

    programming_languages: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    professional_skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class Award(BaseModel):
    """荣誉、奖项或证书信息。"""

    name: str = ""
    date: str = ""
    level: str = ""
    description: str = ""


class ResumeState(BaseModel):
    """简历生成过程中的结构化状态。"""

    basic_info: BasicInfo = Field(default_factory=BasicInfo)
    job_intention: JobIntention = Field(default_factory=JobIntention)
    education: Education = Field(default_factory=Education)
    projects: list[Experience] = Field(default_factory=list)
    internships: list[Experience] = Field(default_factory=list)
    internship_note: str = ""
    skills: Skills = Field(default_factory=Skills)
    awards: list[Award] = Field(default_factory=list)
    self_evaluation: str = ""
    current_stage: str = "job_intention"
    created_at: str = Field(default_factory=now_text)
    updated_at: str = Field(default_factory=now_text)

    def touch(self) -> None:
        """刷新状态更新时间。

        Args:
            无。

        Returns:
            None。
        """

        self.updated_at = now_text()

    def target_summary(self) -> str:
        """生成求职意向摘要。

        Args:
            无。

        Returns:
            求职方向摘要文本。
        """

        parts = [
            self.job_intention.target_position,
            self.job_intention.target_industry,
            self.job_intention.expected_city,
        ]
        return " / ".join(part for part in parts if part) or "待补充"

    def to_public_dict(self) -> dict[str, Any]:
        """导出适合 UI 展示和工具传递的字典。

        Args:
            无。

        Returns:
            简历状态字典。
        """

        return self.model_dump()
