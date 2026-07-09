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
            "我想投 Python 后端实习，互联网行业，杭州",
            "我叫李明，浙江工业大学计算机科学与技术专业大三，邮箱 liming@example.com，主修课程 数据结构、操作系统，GPA 3.7",
            "校园二手交易平台项目，用 Python、FastAPI、MySQL，负责后端接口开发，完成用户认证和订单管理，完成 18 个接口",
            "暂无正式实习",
            "技能 Python、MySQL、Git、Docker，英语六级，获得校级程序设计竞赛二等奖",
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
