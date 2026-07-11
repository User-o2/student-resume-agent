"""项目配置与路径管理模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TEMPLATE_PATH = DATA_DIR / "resume_template.md"


@dataclass(frozen=True)
class AppConfig:
    """应用运行配置。

    Args:
        api_key: 模型服务 API Key。
        base_url: OpenAI 兼容接口的基础地址。
        model: 模型名称。
    """

    api_key: str | None
    base_url: str | None
    model: str | None


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """将 Chat Completions 完整地址规整为 OpenAI SDK 需要的 base_url。

    Args:
        base_url: `.env` 中读取到的接口地址。

    Returns:
        规整后的基础地址；如果输入为空则返回 None。
    """

    if not base_url:
        return None

    normalized = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized.rstrip("/") or None


def load_config(env_path: Path | None = None) -> AppConfig:
    """从 `.env` 加载项目约定的三个模型接口变量。

    Args:
        env_path: 可选的 `.env` 文件路径。

    Returns:
        应用配置对象。
    """

    load_dotenv(env_path or PROJECT_ROOT / ".env")

    api_key = (os.getenv("office_api_key") or "").strip() or None
    base_url = (os.getenv("office_base_url") or "").strip() or None
    model = (os.getenv("office_model") or "").strip() or None

    return AppConfig(
        api_key=api_key,
        base_url=normalize_openai_base_url(base_url),
        model=model,
    )


def ensure_project_dirs() -> None:
    """确保运行所需的数据与输出目录存在。

    Args:
        无。

    Returns:
        None。
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
