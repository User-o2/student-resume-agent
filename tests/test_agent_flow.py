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

    def test_offline_flow_waits_for_explicit_generate_intent(self) -> None:
        """验证信息完整后不会自动生成，必须等待用户明确要求生成简历。"""

        service = ResumeAgentService(use_llm=False, use_agent_driver=False)
        state = ResumeState()
        turns = [
            "目标岗位：人工智能算法实习生 目标行业：互联网 期望城市：北京 姓名：张明 电话：13800001234 邮箱：zhangming@edu.cn 籍贯：江苏南京",
            "学校：南京大学；学院：人工智能学院；专业：人工智能",
            "专业排名：前15%（GPA 3.7/4.0） 英语水平：英语四级 560分，英语六级 558分 核心课程：机器学习、深度学习、数据结构与算法、概率论与数理统计、线性代数",
            "技术栈：Python、TensorFlow、Keras、Flask、CNN",
            "项目经历：基于轻量级CNN的手写数字识别系统。负责数据清洗与增强，使用TensorFlow/Keras搭建MobileNetV2变体，最终在测试集上准确率99.1%；另封装为Flask接口，支持图片上传识别，响应时间低于200ms",
            "全国大学生数学建模竞赛（省级一等奖）：负责构建基于CNN的图像分类模型，运用数据增强提升泛化能力，最终排名全省前8%。",
            "对计算机视觉与模型轻量化有强烈兴趣，持续关注前沿论文并尝试复现。具备工程落地意识，能独立完成从数据预处理到模型部署的完整流程。",
        ]

        result = None
        for turn in turns:
            result = service.handle_message(turn, state)
            state = result.state

        self.assertIsNotNone(result)
        self.assertEqual(state.current_stage, "ready")
        self.assertEqual(result.output_path, "")
        self.assertEqual(result.resume_markdown, "")
        self.assertIn("生成简历", result.assistant_message)

    def test_offline_extraction_does_not_pollute_sections(self) -> None:
        """验证教育、竞赛和自评文本不会被误抽到项目或教育字段。"""

        service = ResumeAgentService(use_llm=False, use_agent_driver=False)
        state = ResumeState()

        for turn in [
            "目标岗位：人工智能算法实习生 目标行业：互联网 期望城市：北京 姓名：张明 电话：13800001234 邮箱：zhangming@edu.cn 籍贯：江苏南京",
            "学校：南京大学；学院：人工智能学院；专业：人工智能",
            "专业排名：前15%（GPA 3.7/4.0） 英语水平：英语四级 560分，英语六级 558分 核心课程：机器学习、深度学习、数据结构与算法、概率论与数理统计、线性代数",
            "全国大学生数学建模竞赛（省级一等奖）：负责构建基于CNN的图像分类模型，运用数据增强提升泛化能力，最终排名全省前8%。",
            "校级优秀学生奖学金（二等奖，2025年）：连续两学期综合测评专业前12%，用于表彰学业与科研实践综合表现。",
            "对计算机视觉与模型轻量化有强烈兴趣，善于团队协作与技术文档撰写，在竞赛和项目中能清晰表达方案逻辑。",
        ]:
            result = service.handle_message(turn, state)
            state = result.state

        self.assertEqual(state.education.school, "南京大学")
        self.assertEqual(state.education.major, "人工智能")
        self.assertEqual(state.projects, [])
        self.assertEqual([award.name for award in state.awards], ["全国大学生数学建模竞赛", "校级优秀学生奖学金"])


if __name__ == "__main__":
    unittest.main()
