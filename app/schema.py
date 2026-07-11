"""简历结构化状态的数据模型定义。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


def now_text() -> str:
    """生成秒级 ISO 时间文本。

    Args:
        无。

    Returns:
        当前时间字符串。
    """

    return datetime.now().isoformat(timespec="seconds")


def migrate_legacy_resume_data(data: Any, prefer_legacy: bool = False) -> Any:
    """将旧版基本信息中的学校、专业迁移到教育背景。

    Args:
        data: 待迁移的 ResumeState 字典或其他输入。
        prefer_legacy: 旧字段是否代表本轮修正并覆盖已有教育字段。

    Returns:
        不含重复学校、专业字段的新数据；非字典输入原样返回。
    """

    if not isinstance(data, Mapping):
        return data

    payload = dict(data)
    basic_info = dict(payload.get("basic_info") or {})
    education = dict(payload.get("education") or {})
    legacy_school = basic_info.pop("university", "")
    legacy_major = basic_info.pop("major", "")

    if legacy_school and (prefer_legacy or not education.get("school")):
        education["school"] = legacy_school
    if legacy_major and (prefer_legacy or not education.get("major")):
        education["major"] = legacy_major

    payload["basic_info"] = basic_info
    if education or "education" in payload:
        payload["education"] = education
    return payload


class BasicInfo(BaseModel):
    """学生基本信息。"""

    name: str = ""
    grade: str = ""
    phone: str = ""
    email: str = ""
    native_place: str = ""


class JobIntention(BaseModel):
    """求职意向信息。"""

    target_position: str = ""
    target_industry: str = ""
    expected_city: str = ""


class Education(BaseModel):
    """教育背景信息。"""

    school: str = ""
    college: str = ""
    major: str = ""
    courses: list[str] = Field(default_factory=list)
    gpa_or_rank: str = ""
    english_level: str = ""


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
    highlights: list[str] = Field(default_factory=list)


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
    current_stage: str = "personal_info"
    created_at: str = Field(default_factory=now_text)
    updated_at: str = Field(default_factory=now_text)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        """兼容读取仍含 basic_info.university/major 的历史状态。

        Args:
            data: Pydantic 校验前的原始状态。

        Returns:
            学校、专业已迁移到 education 的状态数据。
        """

        return migrate_legacy_resume_data(data)

    def touch(self) -> None:
        """刷新状态更新时间。

        Args:
            无。

        Returns:
            None。
        """

        self.updated_at = now_text()
