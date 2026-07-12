"""已有 Markdown 简历的确定性解析与文本拆分逻辑。"""

from __future__ import annotations

import re
from typing import Any

from app.resume import collect_resume_info
from app.schema import ResumeState


def split_items(text: str) -> list[str]:
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

    # 非贪婪匹配到下一个二级标题，避免当前章节吞掉后续简历内容
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
            # 遇到下一条加粗标题时，先提交上一块已累计的 bullet
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


def _parse_experience_section(markdown_text: str, heading: str) -> list[dict[str, Any]]:
    """将项目或实习 Markdown 小节解析为经历补丁。

    Args:
        markdown_text: 完整 Markdown 简历。
        heading: 待解析的二级标题。

    Returns:
        可写入 ResumeState 的经历字典列表。
    """

    blocks = _parse_titled_markdown_blocks(_extract_markdown_section(markdown_text, heading))
    return [
        {
            "title": title,
            "raw_description": "；".join(items),
            "responsibilities": items,
            # 带数字、比例或效果词的 bullet 更可能是成果，规则解析时单独标记
            "results": [
                item
                for item in items
                if re.search(r"\d|%|提升|准确率|响应|完成|排名|F1", item, flags=re.IGNORECASE)
            ],
        }
        for title, items in blocks
    ]


def parse_existing_resume(markdown_text: str) -> ResumeState:
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
                education_update["courses"] = split_items(courses)
            tech_stack = _strip_markdown_label(line, "技术栈")
            if tech_stack:
                skill_items = split_items(tech_stack)
                languages = {"Python", "Java", "C++", "JavaScript", "TypeScript", "Go"}
                update["skills"] = {
                    "programming_languages": [item for item in skill_items if item in languages],
                    "tools": [item for item in skill_items if item not in languages],
                }
    if education_update:
        update["education"] = education_update

    projects = _parse_experience_section(markdown_text, "项目经历")
    if projects:
        update["projects"] = projects

    internships = _parse_experience_section(markdown_text, "实习经历")
    if internships:
        update["internships"] = internships

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

    # 复用统一合并入口完成字段规整和 Pydantic 校验，保证导入与对话采集口径一致
    return collect_resume_info(state, update)
