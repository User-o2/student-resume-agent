"""LangChain Tool 适配层与稳定的简历领域函数导出。"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.resume import (
    STAGE_LABELS,
    check_missing_fields,
    coerce_resume_state,
    collect_resume_info,
    export_resume_to_word,
    fill_resume_template,
    parse_json_object,
    polish_experience,
    polish_state_experiences,
)


@tool
def collect_resume_info_tool(current_state_json: str, update_json: str) -> str:
    """整理并更新用户已提供的简历字段。

    Args:
        current_state_json: 当前简历状态 JSON。
        update_json: 本轮字段更新 JSON。

    Returns:
        更新后的简历状态 JSON。
    """

    # Tool 边界统一使用 JSON，避免 LangChain 将复杂 Pydantic 对象序列化成不稳定参数
    state = collect_resume_info(current_state_json, update_json)
    return state.model_dump_json(ensure_ascii=False)


@tool
def validate_resume_state_tool(current_state_json: str) -> str:
    """执行模板生成前的底线校验。

    Args:
        current_state_json: 当前简历状态 JSON。

    Returns:
        缺失字段、格式错误与质量建议 JSON 报告。
    """

    report = check_missing_fields(current_state_json)
    return json.dumps(report, ensure_ascii=False)


@tool
def fill_resume_template_tool(current_state_json: str, output_path: str = "") -> str:
    """将结构化简历状态填入模板并保存 Markdown 简历。

    Args:
        current_state_json: 当前简历状态 JSON。
        output_path: 可选的 Markdown 输出路径。

    Returns:
        包含 Markdown 内容和输出路径的 JSON。
    """

    result = fill_resume_template(
        current_state_json,
        output_path=output_path or None,
    )
    return json.dumps(result, ensure_ascii=False)


__all__ = [
    "STAGE_LABELS",
    "check_missing_fields",
    "coerce_resume_state",
    "collect_resume_info",
    "collect_resume_info_tool",
    "export_resume_to_word",
    "fill_resume_template",
    "fill_resume_template_tool",
    "parse_json_object",
    "polish_experience",
    "polish_state_experiences",
    "validate_resume_state_tool",
]
