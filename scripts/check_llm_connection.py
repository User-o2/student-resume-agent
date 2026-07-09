"""检查 `.env` 中模型服务是否可以通过 LangChain 联网调用。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import build_chat_model
from app.config import load_config


def main() -> None:
    """执行一次最小化 LLM 连通性检查。

    Args:
        无。

    Returns:
        None。
    """

    config = load_config()
    print(f"base_url: {config.base_url}")
    print(f"model: {config.model}")
    print(f"ssl_verify: {config.ssl_verify}")
    print(f"has_api_key: {bool(config.api_key)}")

    llm = build_chat_model(config)
    if llm is None:
        raise RuntimeError("未读取到 API Key，请检查 .env 中的 api_key。")

    response = llm.invoke("请只回复 OK")
    content = getattr(response, "content", response)
    print(f"response: {str(content).strip()}")


if __name__ == "__main__":
    main()
