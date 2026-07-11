"""聊天模型与各类结构化 LangChain Agent 的构建逻辑。"""

from __future__ import annotations

from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.agent_models import (
    ResumeImportResult,
    ResumePolishResult,
    ResumeScoreReport,
    ResumeTurnDecision,
)
from app.config import AppConfig, load_config
from app.prompts import (
    FINAL_POLISH_SYSTEM_PROMPT,
    IMPORT_RESUME_SYSTEM_PROMPT,
    RESUME_SCORE_SYSTEM_PROMPT,
    TURN_DECISION_SYSTEM_PROMPT,
)


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
    """构建带会话记忆的结构化单轮决策 Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        使用 `ResumeTurnDecision` 作为结构化输出的 Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    # Checkpointer 只保存对话语境；姓名、经历等业务事实始终以每轮传入的
    # ResumeState 为准，避免再维护一份手工对话摘要作为事实来源。
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=TURN_DECISION_SYSTEM_PROMPT,
        response_format=ToolStrategy(ResumeTurnDecision),
        checkpointer=InMemorySaver(),
    )


def build_polish_agent(llm: BaseChatModel | None = None) -> Any | None:
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
    )


def build_import_agent(llm: BaseChatModel | None = None) -> Any | None:
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
    )


def build_score_agent(llm: BaseChatModel | None = None) -> Any | None:
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
    )
