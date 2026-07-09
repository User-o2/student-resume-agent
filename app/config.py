"""项目配置与路径管理模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TEMPLATE_PATH = DATA_DIR / "resume_template.md"
DEFAULT_MODEL = "qwen3.6-35b-a3b"
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class AppConfig:
    """应用运行配置。

    Args:
        api_key: 模型服务 API Key。
        base_url: OpenAI 兼容接口的基础地址。
        model: 模型名称。
        provider: 当前使用的模型服务来源。
        temperature: 生成温度。
        enable_thinking: 是否启用模型思考模式，本项目默认关闭。
        ssl_verify: 是否启用 HTTPS 证书校验。
    """

    api_key: str | None
    base_url: str | None
    model: str = DEFAULT_MODEL
    provider: str = "legacy"
    temperature: float = 0.2
    enable_thinking: bool = False
    ssl_verify: bool = True


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


def parse_bool_env(value: str | None) -> bool | None:
    """解析环境变量中的布尔值。

    Args:
        value: 环境变量原始字符串。

    Returns:
        解析后的布尔值；未设置或无法识别时返回 None。
    """

    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def resolve_ssl_verify(base_url: str | None, override: bool | None = None) -> bool:
    """确定当前 API 地址是否启用 HTTPS 证书校验。

    Args:
        base_url: OpenAI 兼容接口基础地址。
        override: 来自环境变量的手动覆盖值。

    Returns:
        是否启用证书校验。
    """

    if override is not None:
        return override

    hostname = urlparse(base_url or "").hostname or ""
    # 老师提供的统一域名包含下划线，Python/OpenSSL 的 hostname 校验不接受
    # `*.zlth.cn` 匹配这种主机名，因此这里仅对该类地址自动关闭校验。
    if "_" in hostname:
        return False
    return True


def first_env(*names: str) -> str | None:
    """按优先级读取第一个非空环境变量。

    Args:
        *names: 候选环境变量名称，顺序越靠前优先级越高。

    Returns:
        第一个非空变量值；均未设置时返回 None。
    """

    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def load_config(env_path: Path | None = None) -> AppConfig:
    """从环境变量与 `.env` 文件加载应用配置。

    Args:
        env_path: 可选的 `.env` 文件路径。

    Returns:
        应用配置对象。
    """

    load_dotenv(env_path or PROJECT_ROOT / ".env")

    official_api_key = first_env(
        "office_api_key",
        "official_api_key",
        "DASHSCOPE_API_KEY",
        "ALIYUN_API_KEY",
        "OPENAI_OFFICIAL_API_KEY",
    )
    official_base_url = first_env(
        "office_base_url",
        "official_base_url",
        "DASHSCOPE_BASE_URL",
        "ALIYUN_BASE_URL",
        "OPENAI_OFFICIAL_BASE_URL",
    )
    official_model = first_env(
        "office_model",
        "official_model",
        "DASHSCOPE_MODEL",
        "ALIYUN_MODEL",
        "OPENAI_OFFICIAL_MODEL",
    )

    if official_api_key:
        api_key = official_api_key
        base_url = official_base_url or DEFAULT_ALIYUN_BASE_URL
        model = official_model or os.getenv("model") or os.getenv("MODEL") or DEFAULT_MODEL
        provider = "aliyun_official"
    else:
        api_key = os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("model") or os.getenv("MODEL") or DEFAULT_MODEL
        provider = "legacy"

    normalized_base_url = normalize_openai_base_url(base_url)
    ssl_verify_override = parse_bool_env(os.getenv("ssl_verify") or os.getenv("SSL_VERIFY"))

    return AppConfig(
        api_key=api_key,
        base_url=normalized_base_url,
        model=model,
        provider=provider,
        ssl_verify=resolve_ssl_verify(normalized_base_url, ssl_verify_override),
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
