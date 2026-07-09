"""配置解析逻辑的单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import DEFAULT_ALIYUN_BASE_URL, load_config, normalize_openai_base_url, parse_bool_env, resolve_ssl_verify


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

    def test_parse_bool_env_accepts_common_values(self) -> None:
        """验证布尔环境变量常见写法。"""

        self.assertTrue(parse_bool_env("true"))
        self.assertFalse(parse_bool_env("false"))
        self.assertIsNone(parse_bool_env("unknown"))

    def test_load_config_prefers_official_aliyun_settings(self) -> None:
        """验证官方阿里云配置会覆盖旧的统一 API 配置。"""

        env_text = "\n".join(
            [
                "base_url=https://api_2604_w5t3.zlth.cn/v1/chat/completions",
                "api_key=legacy-key",
                "model=legacy-model",
                "office_base_url=https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                "office_api_key=official-key",
                "office_model=official-model",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(env_path)

        self.assertEqual(config.provider, "aliyun_official")
        self.assertEqual(config.api_key, "official-key")
        self.assertEqual(config.base_url, "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.model, "official-model")
        self.assertTrue(config.ssl_verify)

    def test_load_config_uses_official_default_base_url_without_explicit_url(self) -> None:
        """验证只配置官方 Key 时不会回退到旧的统一 API 地址。"""

        env_text = "\n".join(
            [
                "base_url=https://api_2604_w5t3.zlth.cn/v1/chat/completions",
                "api_key=legacy-key",
                "office_api_key=official-key",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(env_path)

        self.assertEqual(config.provider, "aliyun_official")
        self.assertEqual(config.base_url, DEFAULT_ALIYUN_BASE_URL)
        self.assertEqual(config.api_key, "official-key")
        self.assertTrue(config.ssl_verify)


if __name__ == "__main__":
    unittest.main()
