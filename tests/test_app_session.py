"""Streamlit 会话级 Agent 服务的隔离测试。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_streamlit_app_module() -> ModuleType:
    """以独立模块名加载根目录 Streamlit 入口。

    Args:
        无。

    Returns:
        已加载的 Streamlit 应用模块。
    """

    spec = importlib.util.spec_from_file_location("streamlit_app_for_test", PROJECT_ROOT / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 Streamlit 应用模块。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgentServiceSessionTestCase(unittest.TestCase):
    """验证 Agent 服务只在当前 Streamlit 会话内复用。"""

    def test_agent_service_is_reused_only_inside_current_session(self) -> None:
        """验证同一会话复用服务，不同会话创建独立服务。"""

        web_app = load_streamlit_app_module()
        first_session = SimpleNamespace(agent_service=None, agent_service_use_llm=None)
        second_session = SimpleNamespace(agent_service=None, agent_service_use_llm=None)
        first_service = object()
        second_service = object()

        with patch.object(web_app, "ResumeAgentService", side_effect=[first_service, second_service]) as factory:
            with patch.object(web_app.st, "session_state", first_session):
                self.assertIs(web_app.get_agent_service(True), first_service)
                self.assertIs(web_app.get_agent_service(True), first_service)
            with patch.object(web_app.st, "session_state", second_session):
                self.assertIs(web_app.get_agent_service(True), second_service)

        self.assertEqual(factory.call_count, 2)

    def test_switching_llm_mode_rebuilds_session_service(self) -> None:
        """验证切换 LLM 模式时重建服务并隔离旧消息记忆。"""

        web_app = load_streamlit_app_module()
        session = SimpleNamespace(agent_service=None, agent_service_use_llm=None)
        llm_service = object()
        offline_service = object()

        with patch.object(web_app, "ResumeAgentService", side_effect=[llm_service, offline_service]) as factory:
            with patch.object(web_app.st, "session_state", session):
                self.assertIs(web_app.get_agent_service(True), llm_service)
                self.assertIs(web_app.get_agent_service(False), offline_service)

        self.assertEqual(factory.call_args_list[0].kwargs, {"use_llm": True})
        self.assertEqual(factory.call_args_list[1].kwargs, {"use_llm": False})


if __name__ == "__main__":
    unittest.main()
