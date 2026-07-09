"""离线多轮对话流程测试。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.agent import ResumeAgentService, ResumeTurnDecision
from app.schema import ResumeState
from app.tools import collect_resume_info


class FakeDecisionService(ResumeAgentService):
    """使用预设结构化决策模拟 LLM 的测试服务。"""

    def __init__(self, decisions: list[ResumeTurnDecision], polished_state: ResumeState | None = None) -> None:
        """初始化假 LLM 服务。

        Args:
            decisions: 依次返回的结构化单轮决策。
            polished_state: 可选的生成前清洗状态。

        Returns:
            None。
        """

        super().__init__(use_llm=False, use_agent_driver=False)
        self.decisions = list(decisions)
        self.polished_state = polished_state

    def _decide_with_llm(
        self,
        user_input: str,
        state: ResumeState,
        report: dict,
    ) -> ResumeTurnDecision | None:
        """返回预设的结构化 LLM 决策。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。
            report: 当前校验报告。

        Returns:
            预设决策；没有预设时返回 None。
        """

        if not self.decisions:
            return None
        return self.decisions.pop(0)

    def _polish_state_before_generation(self, state: ResumeState, trace: list[str]) -> ResumeState:
        """返回预设的生成前清洗状态。

        Args:
            state: 待清洗状态。
            trace: 本轮轨迹列表。

        Returns:
            预设清洗状态或原状态。
        """

        trace.append("LLM 结构化清洗：ResumePolishResult")
        return self.polished_state or state


def build_ready_state() -> ResumeState:
    """构造可生成简历的完整状态。

    Args:
        无。

    Returns:
        完整的简历状态。
    """

    return collect_resume_info(
        ResumeState(),
        {
            "basic_info": {
                "name": "张明",
                "university": "南京大学",
                "major": "人工智能",
                "phone": "13800001234",
                "email": "zhangming@edu.cn",
                "native_place": "江苏南京",
            },
            "job_intention": {
                "target_position": "人工智能算法实习生",
                "target_industry": "互联网",
                "expected_city": "北京",
            },
            "education": {
                "school": "南京大学",
                "college": "人工智能学院",
                "major": "人工智能",
                "courses": ["机器学习", "深度学习"],
                "gpa_or_rank": "专业前15%",
                "english_level": "CET-6 540",
            },
            "projects": [
                {
                    "title": "手写数字识别系统",
                    "technologies": ["Python", "PyTorch"],
                    "responsibilities": ["负责模型训练"],
                    "results": ["测试集准确率 99.1%"],
                }
            ],
            "skills": {"programming_languages": ["Python"], "tools": ["PyTorch"]},
            "awards": [{"name": "数学建模竞赛省级一等奖", "description": "团队排名前8%"}],
            "self_evaluation": "对计算机视觉有兴趣，具备模型训练实践经验。",
        },
    )


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

    def test_llm_decision_extracts_unlabeled_skill_list(self) -> None:
        """验证 LLM 主链路可以抽取没有“技术栈”标签的技能列表。"""

        decision = ResumeTurnDecision(
            intent="collect_info",
            patch={"skills": {"programming_languages": ["Python", "C++"], "tools": ["PyTorch", "Linux", "Git"]}},
            assistant_message="这些技能我已经记录了。请继续补充项目经历。",
        )
        service = FakeDecisionService([decision])

        result = service.handle_message("Python, C++, PyTorch, Linux, Git", ResumeState())

        self.assertEqual(result.state.skills.programming_languages, ["Python", "C++"])
        self.assertIn("PyTorch", result.state.skills.tools)
        self.assertIn("LLM 结构化决策：ResumeTurnDecision", result.agent_trace)

    def test_llm_decision_extracts_cross_section_natural_text(self) -> None:
        """验证 LLM 主链路可以从自然长文本跨模块抽取信息。"""

        decision = ResumeTurnDecision(
            intent="collect_info",
            patch={
                "basic_info": {"name": "王欣", "phone": "13800000002", "email": "wangxin@example.com"},
                "job_intention": {"target_position": "机器学习实习", "target_industry": "人工智能", "expected_city": "上海"},
                "education": {"school": "南京理工大学", "college": "人工智能学院", "major": "人工智能"},
                "projects": [
                    {
                        "title": "垃圾分类图像识别模型",
                        "technologies": ["PyTorch", "ResNet"],
                        "responsibilities": ["负责数据清洗和模型训练"],
                        "results": ["验证集准确率 91.3%"],
                    }
                ],
            },
            assistant_message="已记录你的基本信息、求职方向和项目经历。还需要补充成绩排名、英语水平、核心课程、奖项和自我评价。",
        )
        service = FakeDecisionService([decision])

        result = service.handle_message("我叫王欣，想去上海做机器学习实习，也做过垃圾分类识别。", ResumeState())

        self.assertEqual(result.state.basic_info.name, "王欣")
        self.assertEqual(result.state.job_intention.expected_city, "上海")
        self.assertEqual(result.state.projects[0].title, "垃圾分类图像识别模型")

    def test_generation_request_is_blocked_by_validation(self) -> None:
        """验证 LLM 想生成时仍会被底线校验拦截。"""

        partial_state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {"name": "李明", "email": "liming@example.com"},
                "job_intention": {"target_position": "Python 后端实习"},
            },
        )
        decision = ResumeTurnDecision(
            intent="generate_resume",
            patch={},
            assistant_message="我准备生成简历。",
        )
        service = FakeDecisionService([decision])

        result = service.handle_message("生成简历", partial_state)

        self.assertEqual(result.output_path, "")
        self.assertFalse(result.missing_report["is_ready"])
        self.assertIn("现在还不能生成完整简历", result.assistant_message)

    def test_generation_uses_polished_state_before_template_fill(self) -> None:
        """验证生成前会先使用 LLM 清洗后的状态再填充模板。"""

        ready_state = build_ready_state()
        polished_state = ready_state.model_copy(deep=True)
        polished_state.projects[0].responsibilities = ["负责模型训练", "负责模型训练"]
        polished_state.projects[0].results = ["测试集准确率 99.1%"]
        polished_state.projects[0].polished_bullets = [
            "负责手写数字识别模型训练与评估，完成数据预处理、模型调参与结果分析",
            "在测试集上取得 99.1% 准确率，并整理实验结论支撑后续优化",
        ]
        decision = ResumeTurnDecision(intent="generate_resume", patch={}, assistant_message="开始生成简历。")
        service = FakeDecisionService([decision], polished_state=polished_state)

        def fake_fill(state: ResumeState) -> dict[str, str]:
            """模拟模板填充并断言使用了清洗后的项目要点。

            Args:
                state: 传入模板填充函数的简历状态。

            Returns:
                模拟的 Markdown 内容和输出路径。
            """

            self.assertEqual(state.projects[0].polished_bullets, polished_state.projects[0].polished_bullets)
            markdown = "\n".join(
                [
                    "# 张明",
                    "## 项目经历",
                    "- 负责手写数字识别模型训练与评估，完成数据预处理、模型调参与结果分析",
                    "- 在测试集上取得 99.1% 准确率，并整理实验结论支撑后续优化",
                ]
            )
            return {"markdown": markdown, "output_path": "/tmp/resume.md"}

        with patch("app.agent.fill_resume_template", side_effect=fake_fill):
            result = service.handle_message("生成简历", ready_state)

        self.assertNotIn("待补充", result.resume_markdown)
        self.assertEqual(result.resume_markdown.count("负责手写数字识别模型训练"), 1)

    def test_optimize_existing_resume_offline_writes_markdown(self) -> None:
        """验证已有 Markdown 简历可以离线解析并输出优化文件。"""

        markdown = """# 张明

- **求职意向**：人工智能算法实习生 | 互联网 | 北京
- **电话**：13800001234
- **邮箱**：zhangming@edu.cn
- **籍贯**：江苏南京

## 教育背景
南京大学 人工智能学院 人工智能专业
- **专业排名**：前15%
- **英语水平**：CET-6 540
- **核心课程**：机器学习，深度学习
- **技术栈**：Python, PyTorch, Flask

## 项目经历
**图像识别系统**
- 负责数据清洗
- 准确率99.1%

## 竞赛获奖
**数学建模竞赛省级一等奖**
- 团队排名前8%

## 自我评价
- 对计算机视觉有兴趣
"""
        service = ResumeAgentService(use_llm=False, use_agent_driver=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "optimized.md"
            result = service.optimize_existing_resume(markdown, output_path=output_path)

        self.assertEqual(result.state.basic_info.name, "张明")
        self.assertIn("图像识别系统", result.markdown)
        self.assertTrue(Path(result.output_path).name.endswith("optimized.md"))


if __name__ == "__main__":
    unittest.main()
