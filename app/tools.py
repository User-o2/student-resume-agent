"""简历智能体的核心工具函数与 LangChain Tool 封装。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from app.config import DEFAULT_TEMPLATE_PATH, OUTPUTS_DIR
from app.prompts import POLISH_EXPERIENCE_PROMPT
from app.schema import Award, Experience, ResumeState


STAGE_LABELS = {
    "job_intention": "求职意向",
    "basic_education": "基本信息与教育背景",
    "projects": "项目经历",
    "internships": "实习或实践经历",
    "skills_awards": "技能与荣誉奖项",
    "self_evaluation": "自我评价",
    "ready": "可生成简历",
}


def parse_json_object(text: str) -> dict[str, Any]:
    """从文本中解析 JSON 对象。

    Args:
        text: 待解析文本。

    Returns:
        JSON 字典；解析失败时返回空字典。
    """

    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def coerce_resume_state(state: ResumeState | Mapping[str, Any] | str | None) -> ResumeState:
    """将不同输入形式转换为 ResumeState。

    Args:
        state: 简历状态对象、字典、JSON 字符串或 None。

    Returns:
        标准化后的简历状态对象。
    """

    if isinstance(state, ResumeState):
        return state.model_copy(deep=True)
    if isinstance(state, str):
        data = parse_json_object(state)
        return ResumeState.model_validate(data or {})
    if isinstance(state, Mapping):
        return ResumeState.model_validate(dict(state))
    return ResumeState()


def _is_blank(value: Any) -> bool:
    """判断字段值是否为空。

    Args:
        value: 任意字段值。

    Returns:
        是否为空。
    """

    return value is None or value == "" or value == [] or value == {}


def _unique_extend(base: list[Any], patch: list[Any]) -> list[Any]:
    """合并列表并保持顺序去重。

    Args:
        base: 原列表。
        patch: 新列表。

    Returns:
        合并后的列表。
    """

    result = list(base)
    for item in patch:
        if _is_blank(item):
            continue
        if item not in result:
            result.append(item)
    return result


def _merge_dict(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并字典。

    Args:
        base: 原始字典。
        patch: 更新字典。

    Returns:
        合并后的字典。
    """

    for key, value in patch.items():
        if _is_blank(value):
            continue

        if isinstance(base.get(key), dict) and isinstance(value, Mapping):
            base[key] = _merge_dict(dict(base[key]), value)
        elif isinstance(base.get(key), list) and isinstance(value, list):
            base[key] = _unique_extend(list(base[key]), value)
        else:
            base[key] = value
    return base


def _record_key(record: Mapping[str, Any], fallback_key: str) -> str:
    """提取经历或奖项的合并键。

    Args:
        record: 记录字典。
        fallback_key: 默认键名。

    Returns:
        可用于匹配的键值。
    """

    return str(record.get(fallback_key) or record.get("title") or record.get("name") or "").strip()


def _merge_records(
    base: list[dict[str, Any]],
    patch: list[Mapping[str, Any]],
    fallback_key: str,
) -> list[dict[str, Any]]:
    """按标题或名称合并列表记录。

    Args:
        base: 原始记录列表。
        patch: 新记录列表。
        fallback_key: 用于匹配的默认键名。

    Returns:
        合并后的记录列表。
    """

    result = [dict(item) for item in base]
    for item in patch:
        if not isinstance(item, Mapping) or _is_blank(item):
            continue
        item_key = _record_key(item, fallback_key)
        match_index = None
        for index, existed in enumerate(result):
            if item_key and _record_key(existed, fallback_key) == item_key:
                match_index = index
                break
        if match_index is None:
            result.append(dict(item))
        else:
            result[match_index] = _merge_dict(result[match_index], item)
    return result


def collect_resume_info(
    state: ResumeState | Mapping[str, Any] | str | None,
    updates: Mapping[str, Any] | str,
) -> ResumeState:
    """整理并更新用户已提供的简历字段。

    Args:
        state: 当前简历状态。
        updates: 本轮用户输入抽取出的字段更新。

    Returns:
        更新后的简历状态。
    """

    resume_state = coerce_resume_state(state)
    patch = parse_json_object(updates) if isinstance(updates, str) else dict(updates)
    if not patch:
        return resume_state

    data = resume_state.model_dump()
    for key in ("projects", "internships"):
        if isinstance(patch.get(key), list):
            data[key] = _merge_records(data.get(key, []), patch[key], "title")
            patch = {item_key: item_value for item_key, item_value in patch.items() if item_key != key}
    if isinstance(patch.get("awards"), list):
        data["awards"] = _merge_records(data.get("awards", []), patch["awards"], "name")
        patch = {item_key: item_value for item_key, item_value in patch.items() if item_key != "awards"}

    data = _merge_dict(data, patch)

    if data["basic_info"].get("university") and not data["education"].get("school"):
        data["education"]["school"] = data["basic_info"]["university"]
    if data["basic_info"].get("major") and not data["education"].get("major"):
        data["education"]["major"] = data["basic_info"]["major"]

    updated_state = ResumeState.model_validate(data)
    updated_state.touch()
    return updated_state


def _has_any_skill(state: ResumeState) -> bool:
    """判断是否已经采集到至少一项技能。

    Args:
        state: 当前简历状态。

    Returns:
        是否存在技能信息。
    """

    return any(
        [
            state.skills.programming_languages,
            state.skills.tools,
            state.skills.professional_skills,
            state.skills.languages,
        ]
    )


def _experience_quality_questions(experience: Experience, category: str) -> list[str]:
    """生成经历完整度追问。

    Args:
        experience: 项目或实习经历。
        category: 经历类别。

    Returns:
        需要追问的问题列表。
    """

    questions: list[str] = []
    title = experience.title or f"这段{category}"
    if not experience.technologies:
        questions.append(f"{title}还缺少技术方法或工具，请补充使用的框架、模型、数据库或软件。")
    if not experience.responsibilities:
        questions.append(f"{title}还缺少个人职责，请补充你具体负责的模块或工作内容。")
    if not experience.results:
        questions.append(f"{title}还缺少项目成果，请补充效果、指标、排名、上线情况或可验证结果。")
    return questions


def check_missing_fields(state: ResumeState | Mapping[str, Any] | str | None) -> dict[str, Any]:
    """检查必要字段缺失与经历质量问题。

    Args:
        state: 当前简历状态。

    Returns:
        包含缺失字段、质量追问和是否可生成的报告。
    """

    resume_state = coerce_resume_state(state)
    missing_fields: list[str] = []
    quality_questions: list[str] = []
    optional_suggestions: list[str] = []

    if not resume_state.job_intention.target_position:
        missing_fields.append("求职意向：目标岗位")
    if not resume_state.job_intention.expected_city:
        missing_fields.append("求职意向：期望城市")
    if not resume_state.basic_info.name:
        missing_fields.append("基本信息：姓名")
    if not resume_state.basic_info.university:
        missing_fields.append("基本信息：学校")
    if not resume_state.basic_info.major:
        missing_fields.append("基本信息：专业")
    if not resume_state.basic_info.email:
        missing_fields.append("基本信息：邮箱")
    if not resume_state.education.courses and not resume_state.education.gpa_or_rank:
        missing_fields.append("教育背景：主修课程或成绩排名")
    if not resume_state.projects:
        missing_fields.append("项目经历：至少 1 段项目")
    else:
        quality_questions.extend(_experience_quality_questions(resume_state.projects[0], "项目经历"))
    if not _has_any_skill(resume_state):
        missing_fields.append("技能特长：至少 1 类技能")
    if not resume_state.self_evaluation:
        missing_fields.append("自我评价：个人优势与发展方向")
    if not resume_state.internships and not resume_state.internship_note:
        optional_suggestions.append("可补充实习、课程实践、社团实践或说明暂无正式实习。")
    if not resume_state.awards:
        optional_suggestions.append("可补充竞赛奖项、奖学金、证书或说明暂无。")

    return {
        "is_ready": not missing_fields and not quality_questions,
        "missing_fields": missing_fields,
        "quality_questions": quality_questions,
        "optional_suggestions": optional_suggestions,
        "current_stage": resume_state.current_stage,
    }


def _clean_markdown_list(items: list[str]) -> list[str]:
    """清洗 Markdown 列表条目。

    Args:
        items: 原始条目列表。

    Returns:
        清洗后的条目列表。
    """

    cleaned: list[str] = []
    for item in items:
        text = str(item).strip().lstrip("-").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def polish_experience(
    raw_text: str,
    target_position: str = "",
    llm: BaseChatModel | None = None,
) -> list[str]:
    """将口语化经历润色为简历要点。

    Args:
        raw_text: 原始经历描述。
        target_position: 目标岗位。
        llm: 可选的聊天模型实例。

    Returns:
        简历化表达的要点列表。
    """

    text = raw_text.strip()
    if not text:
        return []

    if llm is not None:
        prompt = POLISH_EXPERIENCE_PROMPT.format(
            target_position=target_position or "学生求职/实习",
            raw_text=text,
        )
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            bullets = _clean_markdown_list(content.splitlines())
            if bullets:
                return bullets[:3]
        except Exception:
            pass

    normalized = re.sub(r"\s+", " ", text).strip("，。；; ")
    normalized = normalized.replace("我主要", "主要").replace("我负责", "负责")
    if not normalized.startswith(("参与", "负责", "主导", "协助", "完成")):
        normalized = f"参与{normalized}"

    bullets = [f"{normalized}，沉淀了与{target_position or '目标岗位'}相关的实践经验。"]
    if not re.search(r"\d|%|提升|准确率|排名|上线|用户|效率", text):
        bullets.append("建议继续补充量化成果，例如准确率、性能提升、用户规模、完成模块数量或竞赛排名。")
    return bullets


def polish_state_experiences(
    state: ResumeState | Mapping[str, Any] | str | None,
    llm: BaseChatModel | None = None,
) -> ResumeState:
    """批量润色状态中的项目与实习经历。

    Args:
        state: 当前简历状态。
        llm: 可选的聊天模型实例。

    Returns:
        润色后的简历状态。
    """

    resume_state = coerce_resume_state(state)
    target = resume_state.job_intention.target_position

    for group in (resume_state.projects, resume_state.internships):
        for experience in group:
            if experience.polished_bullets:
                continue
            if llm is None and (experience.responsibilities or experience.results):
                experience.polished_bullets = _clean_markdown_list(
                    experience.responsibilities + experience.results
                )
                continue
            raw_parts = [
                experience.raw_description,
                "；".join(experience.responsibilities),
                "；".join(experience.results),
            ]
            raw_text = "；".join(part for part in raw_parts if part)
            experience.polished_bullets = polish_experience(raw_text, target, llm)
    resume_state.touch()
    return resume_state


def _join_or_default(items: list[str], default: str = "待补充") -> str:
    """将列表拼接为中文顿号分隔文本。

    Args:
        items: 字符串列表。
        default: 空列表时的默认文本。

    Returns:
        拼接后的文本。
    """

    return "、".join(items) if items else default


def _format_experiences(experiences: list[Experience], empty_text: str) -> str:
    """格式化项目或实习经历为 Markdown。

    Args:
        experiences: 经历列表。
        empty_text: 空列表时的展示文本。

    Returns:
        Markdown 文本。
    """

    if not experiences:
        return empty_text

    blocks: list[str] = []
    for experience in experiences:
        title = experience.title or "未命名经历"
        time_range = " - ".join(part for part in [experience.start_date, experience.end_date] if part)
        meta_parts = [part for part in [experience.organization, experience.role, time_range] if part]
        lines = [f"### {title}"]
        if meta_parts:
            lines.append(f"**{' | '.join(meta_parts)}**")
        if experience.technologies:
            lines.append(f"**技术/工具：** {_join_or_default(experience.technologies)}")
        bullets = experience.polished_bullets or (experience.responsibilities + experience.results)
        if not bullets and experience.raw_description:
            bullets = polish_experience(experience.raw_description)
        lines.extend(f"- {item}" for item in _clean_markdown_list(bullets))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_awards(awards: list[Award]) -> str:
    """格式化荣誉奖项为 Markdown。

    Args:
        awards: 奖项列表。

    Returns:
        Markdown 文本。
    """

    if not awards:
        return "- 待补充"

    lines: list[str] = []
    for award in awards:
        details = " | ".join(part for part in [award.date, award.level, award.description] if part)
        suffix = f"（{details}）" if details else ""
        lines.append(f"- {award.name or '未命名奖项'}{suffix}")
    return "\n".join(lines)


def build_template_context(state: ResumeState | Mapping[str, Any] | str | None) -> dict[str, str]:
    """构建 Jinja2 简历模板上下文。

    Args:
        state: 当前简历状态。

    Returns:
        模板上下文字典。
    """

    resume_state = coerce_resume_state(state)
    basic = resume_state.basic_info
    job = resume_state.job_intention
    education = resume_state.education
    skills = resume_state.skills

    education_lines = [
        f"- **学校专业：** {education.school or basic.university or '待补充'} - {education.major or basic.major or '待补充'}",
        f"- **年级：** {basic.grade or '待补充'}",
        f"- **主修课程：** {_join_or_default(education.courses)}",
        f"- **成绩/排名：** {education.gpa_or_rank or '待补充'}",
    ]

    skill_lines = [
        f"- **编程语言：** {_join_or_default(skills.programming_languages)}",
        f"- **工具软件：** {_join_or_default(skills.tools)}",
        f"- **专业技能：** {_join_or_default(skills.professional_skills)}",
        f"- **语言能力：** {_join_or_default(skills.languages)}",
    ]

    return {
        "name": basic.name or "姓名待补充",
        "job_intention": " / ".join(
            part for part in [job.target_position, job.target_industry, job.expected_city] if part
        )
        or "待补充",
        "phone": basic.phone or "待补充",
        "email": basic.email or "待补充",
        "university": basic.university or education.school or "待补充",
        "major": basic.major or education.major or "待补充",
        "education": "\n".join(education_lines),
        "projects": _format_experiences(resume_state.projects, "- 待补充"),
        "internships": _format_experiences(
            resume_state.internships,
            f"- {resume_state.internship_note}" if resume_state.internship_note else "- 待补充",
        ),
        "skills": "\n".join(skill_lines),
        "awards": _format_awards(resume_state.awards),
        "self_evaluation": resume_state.self_evaluation or "待补充",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def fill_resume_template(
    state: ResumeState | Mapping[str, Any] | str | None,
    template_path: Path | str = DEFAULT_TEMPLATE_PATH,
    output_path: Path | str | None = None,
) -> dict[str, str]:
    """将结构化简历状态填入 Markdown 模板并保存。

    Args:
        state: 当前简历状态。
        template_path: Markdown 模板路径。
        output_path: 可选输出路径。

    Returns:
        包含 Markdown 内容和输出路径的字典。
    """

    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"简历模板不存在：{template_file}")

    env = Environment(
        loader=FileSystemLoader(str(template_file.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_file.name)
    markdown = template.render(**build_template_context(state)).strip() + "\n"

    destination = Path(output_path) if output_path else OUTPUTS_DIR / f"resume_{datetime.now():%Y%m%d_%H%M%S}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")

    return {"markdown": markdown, "output_path": str(destination)}


@tool
def collect_resume_info_tool(current_state_json: str, update_json: str) -> str:
    """整理并更新用户已提供的简历字段。

    Args:
        current_state_json: 当前简历状态 JSON。
        update_json: 本轮字段更新 JSON。

    Returns:
        更新后的简历状态 JSON。
    """

    state = collect_resume_info(current_state_json, update_json)
    return state.model_dump_json(ensure_ascii=False)


@tool
def check_missing_fields_tool(current_state_json: str) -> str:
    """检查简历必要字段与经历质量。

    Args:
        current_state_json: 当前简历状态 JSON。

    Returns:
        缺失字段与质量检查 JSON 报告。
    """

    report = check_missing_fields(current_state_json)
    return json.dumps(report, ensure_ascii=False)


@tool
def polish_experience_tool(raw_text: str, target_position: str = "") -> str:
    """对项目或实习经历进行简历化润色。

    Args:
        raw_text: 原始经历描述。
        target_position: 目标岗位。

    Returns:
        Markdown 要点文本。
    """

    return "\n".join(f"- {item}" for item in polish_experience(raw_text, target_position))


@tool
def fill_resume_template_tool(current_state_json: str) -> str:
    """将结构化简历状态填入模板并保存 Markdown 简历。

    Args:
        current_state_json: 当前简历状态 JSON。

    Returns:
        包含 Markdown 内容和输出路径的 JSON。
    """

    result = fill_resume_template(current_state_json)
    return json.dumps(result, ensure_ascii=False)


RESUME_TOOLS = [
    collect_resume_info_tool,
    check_missing_fields_tool,
    polish_experience_tool,
    fill_resume_template_tool,
]
