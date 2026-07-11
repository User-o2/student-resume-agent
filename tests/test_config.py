"""配置解析逻辑的单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig, load_config, normalize_openai_base_url
from app.llm import build_chat_model, resolve_ssl_verify


class ConfigTestCase(unittest.TestCase):
    """测试模型 API 配置解析行为。"""

    def test_normalize_openai_base_url_removes_chat_completions_suffix(self) -> None:
        """验证完整 Chat Completions 地址会规整为 SDK 基础地址。"""

        base_url = normalize_openai_base_url("https://api_2604_w5t3.zlth.cn/v1/chat/completions")

        self.assertEqual(base_url, "https://api_2604_w5t3.zlth.cn/v1")

    def test_resolve_ssl_verify_disables_underscore_hostname_by_default(self) -> None:
        """验证带下划线的统一域名默认关闭 Python TLS hostname 校验。"""

        ssl_verify = resolve_ssl_verify("https://api_2604_w5t3.zlth.cn/v1")

        self.assertFalse(ssl_verify)

    def test_resolve_ssl_verify_keeps_normal_hostname_verified(self) -> None:
        """验证普通域名默认保持证书校验。"""

        ssl_verify = resolve_ssl_verify("https://api.example.com/v1")

        self.assertTrue(ssl_verify)

    def test_load_config_reads_only_office_settings(self) -> None:
        """验证配置只读取项目约定的三个 office 变量。"""

        env_text = "\n".join(
            [
                "base_url=https://api_2604_w5t3.zlth.cn/v1/chat/completions",
                "api_key=legacy-key",
                "model=legacy-model",
                "OPENAI_BASE_URL=https://legacy.example.com/v1",
                "OPENAI_API_KEY=openai-key",
                "office_base_url=https://office.example.com/v1/chat/completions",
                "office_api_key=office-key",
                "office_model=office-model",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(env_path)

        self.assertEqual(config.api_key, "office-key")
        self.assertEqual(config.base_url, "https://office.example.com/v1")
        self.assertEqual(config.model, "office-model")

    def test_load_config_does_not_fallback_to_legacy_settings(self) -> None:
        """验证缺少 office 变量时不会读取旧变量或设置默认值。"""

        env_text = "\n".join(
            [
                "base_url=https://api_2604_w5t3.zlth.cn/v1/chat/completions",
                "api_key=legacy-key",
                "model=legacy-model",
                "OPENAI_API_KEY=openai-key",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(env_path)

        self.assertIsNone(config.base_url)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.model)

    def test_build_chat_model_requires_all_three_office_settings(self) -> None:
        """验证任一 office 配置缺失时不会创建不完整的模型客户端。"""

        configs = [
            AppConfig(api_key=None, base_url="https://api.example.com/v1", model="office-model"),
            AppConfig(api_key="office-key", base_url=None, model="office-model"),
            AppConfig(api_key="office-key", base_url="https://api.example.com/v1", model=None),
        ]

        for config in configs:
            with self.subTest(config=config):
                self.assertIsNone(build_chat_model(config))


if __name__ == "__main__":
    unittest.main()
