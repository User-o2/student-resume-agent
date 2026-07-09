"""配置解析逻辑的单元测试。"""

from __future__ import annotations

import unittest

from app.config import normalize_openai_base_url, parse_bool_env, resolve_ssl_verify


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


if __name__ == "__main__":
    unittest.main()
