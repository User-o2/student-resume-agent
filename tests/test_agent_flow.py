"""离线多轮对话流程测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.agent import ResumeAgentService
from app.schema import ResumeState


class ResumeAgentFlowTestCase(unittest.TestCase):
    """测试简历 Agent 的多轮阶段流转。"""

    def test_offline_multi_turn_flow_reaches_generation(self) -> None:
        """验证规则兜底模式可以完成 5 轮以上采集并进入生成。"""

        service = ResumeAgentService(use_llm=False)
        state = ResumeState()
        turns = [
            "我想投 Python 后端实习，目标行业互联网，期望城市杭州。我叫李明，电话13800000001，邮箱 liming@example.com，籍贯山东省济南市",
            "浙江工业大学计算机学院，专业是计算机科学与技术，主修课程 数据结构、操作系统，专业排名前15%，CET-6 510，技术栈 Python、MySQL、Git、Docker",
            "校园二手交易平台项目，用 Python、FastAPI、MySQL，负责后端接口开发，完成用户认证和订单管理，完成 18 个接口",
            "获得校级程序设计竞赛二等奖，负责后端题解整理和核心算法实现，团队排名前10%",
            "我计算机基础扎实，关注后端工程化和性能优化，希望继续提升复杂业务开发能力。",
        ]

        result = None
        for turn in turns:
            result = service.handle_message(turn, state)
            state = result.state

        self.assertIsNotNone(result)
        self.assertEqual(state.current_stage, "ready")
        self.assertTrue(state.projects)
        self.assertEqual(state.basic_info.major, "计算机科学与技术")

        with patch(
            "app.agent.fill_resume_template",
            return_value={"markdown": "# 李明\n\n## 项目经历\n校园二手交易平台\n", "output_path": "/tmp/resume.md"},
        ):
            generated = service.handle_message("生成简历", state)

        self.assertIn("# 李明", generated.resume_markdown)
        self.assertEqual(generated.output_path, "/tmp/resume.md")


if __name__ == "__main__":
    unittest.main()
