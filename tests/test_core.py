"""核心工具函数的单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.schema import ResumeState
from app.tools import (
    check_missing_fields,
    collect_resume_info,
    fill_resume_template,
    polish_experience,
)


class ResumeToolTestCase(unittest.TestCase):
    """测试简历工具函数的确定性行为。"""

    def test_collect_resume_info_merges_nested_fields(self) -> None:
        """验证嵌套字段和列表记录可以正确合并。"""

        state = ResumeState()
        state = collect_resume_info(
            state,
            {
                "basic_info": {"name": "李明", "university": "浙江工业大学"},
                "projects": [{"title": "校园平台", "technologies": ["Python"]}],
            },
        )
        state = collect_resume_info(
            state,
            {
                "basic_info": {"email": "liming@example.com"},
                "projects": [{"title": "校园平台", "results": ["完成 18 个接口"]}],
            },
        )

        self.assertEqual(state.basic_info.name, "李明")
        self.assertEqual(state.basic_info.email, "liming@example.com")
        self.assertEqual(len(state.projects), 1)
        self.assertEqual(state.projects[0].technologies, ["Python"])
        self.assertEqual(state.projects[0].results, ["完成 18 个接口"])

    def test_check_missing_fields_reports_project_quality_without_blocking_generation(self) -> None:
        """验证项目质量追问不会阻塞已满足模板必填板块的简历生成。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {
                    "name": "李明",
                    "university": "浙江工业大学",
                    "major": "计算机科学与技术",
                    "phone": "13800000001",
                    "email": "liming@example.com",
                    "native_place": "山东省济南市",
                },
                "job_intention": {
                    "target_position": "Python 后端实习",
                    "target_industry": "互联网",
                    "expected_city": "杭州",
                },
                "education": {
                    "school": "浙江工业大学",
                    "college": "计算机学院",
                    "major": "计算机科学与技术",
                    "courses": ["数据结构"],
                    "gpa_or_rank": "专业前 15%",
                    "english_level": "CET-6 510",
                },
                "projects": [{"title": "图像识别项目", "raw_description": "我做过一个图像识别项目"}],
                "skills": {"programming_languages": ["Python"]},
                "awards": [{"name": "校级程序设计竞赛二等奖", "description": "团队排名前 10%"}],
                "self_evaluation": "学习能力强，关注后端开发方向。",
            },
        )

        report = check_missing_fields(state)
        self.assertTrue(report["is_ready"])
        self.assertTrue(report["quality_questions"])

    def test_check_missing_fields_blocks_only_required_fields(self) -> None:
        """验证新版模板所有必填板块缺失时会阻塞生成。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {
                    "name": "李明",
                    "university": "浙江工业大学",
                    "major": "计算机科学与技术",
                },
                "job_intention": {"target_position": "Python 后端实习"},
            },
        )

        report = check_missing_fields(state)

        self.assertFalse(report["is_ready"])
        self.assertIn("个人信息：目标行业", report["missing_fields"])
        self.assertIn("个人信息：电话", report["missing_fields"])
        self.assertIn("个人信息：邮箱", report["missing_fields"])
        self.assertIn("个人信息：籍贯", report["missing_fields"])
        self.assertIn("教育背景：学院", report["missing_fields"])
        self.assertIn("项目经历：至少 1 段项目经历", report["missing_fields"])
        self.assertIn("竞赛获奖：至少 1 项竞赛、奖学金或证书", report["missing_fields"])
        self.assertIn("自我评价：2-3 条个人优势", report["missing_fields"])

    def test_collect_resume_info_merges_untitled_project_supplements(self) -> None:
        """验证无标题项目补充会合并到上一段项目。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "projects": [
                    {
                        "technologies": ["Flask"],
                        "responsibilities": ["负责智能体路由模块"],
                    }
                ]
            },
        )
        state = collect_resume_info(
            state,
            {
                "projects": [
                    {
                        "technologies": ["Redis", "Celery"],
                        "results": ["接口吞吐量提升约 40%"],
                    }
                ]
            },
        )

        self.assertEqual(len(state.projects), 1)
        self.assertEqual(state.projects[0].technologies, ["Flask", "Redis", "Celery"])
        self.assertEqual(state.projects[0].results, ["接口吞吐量提升约 40%"])

    def test_check_missing_fields_uses_best_meaningful_project(self) -> None:
        """验证空项目不会导致完整项目被误判为缺技术。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {
                    "name": "小明",
                    "university": "天津科技大学",
                    "major": "人工智能",
                    "phone": "15208898345",
                    "email": "27485938@qq.com",
                    "native_place": "天津市",
                },
                "job_intention": {
                    "target_position": "Python 后端实习",
                    "target_industry": "互联网",
                    "expected_city": "杭州",
                },
                "education": {
                    "school": "天津科技大学",
                    "college": "人工智能学院",
                    "major": "人工智能",
                    "courses": ["机器学习"],
                    "gpa_or_rank": "3/105",
                    "english_level": "CET-6 500",
                },
                "projects": [
                    {},
                    {
                        "title": "智能体路由系统",
                        "technologies": ["Flask", "Redis", "Celery"],
                        "responsibilities": ["负责智能体路由与对话管理模块"],
                        "results": ["接口吞吐量提升约 40%"],
                    },
                ],
                "skills": {"programming_languages": ["Python"]},
                "awards": [{"name": "校级程序设计竞赛二等奖", "description": "团队排名前 10%"}],
                "self_evaluation": "有较好的开发经验，认真学习新技术。",
            },
        )

        report = check_missing_fields(state)

        self.assertTrue(report["is_ready"])
        self.assertEqual(report["quality_questions"], [])

    def test_fill_resume_template_writes_markdown(self) -> None:
        """验证模板填充可以写出 Markdown 文件。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {
                    "name": "李明",
                    "university": "浙江工业大学",
                    "major": "计算机科学与技术",
                    "phone": "13800000001",
                    "email": "liming@example.com",
                    "native_place": "山东省济南市",
                },
                "job_intention": {
                    "target_position": "Python 后端实习",
                    "target_industry": "互联网",
                    "expected_city": "杭州",
                },
                "education": {
                    "school": "浙江工业大学",
                    "college": "计算机学院",
                    "major": "计算机科学与技术",
                    "courses": ["数据结构"],
                    "gpa_or_rank": "专业前 15%",
                    "english_level": "CET-6 510",
                },
                "projects": [
                    {
                        "title": "校园平台",
                        "technologies": ["Python"],
                        "responsibilities": ["负责后端接口开发"],
                        "results": ["完成 18 个接口"],
                    }
                ],
                "skills": {"programming_languages": ["Python"]},
                "awards": [{"name": "校级程序设计竞赛二等奖", "description": "团队排名前 10%"}],
                "self_evaluation": "具备后端开发实践经验。",
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "resume.md"
            result = fill_resume_template(state, output_path=output_path)

        self.assertIn("# 李明", result["markdown"])
        self.assertIn("校园平台", result["markdown"])
        self.assertIn("## 竞赛获奖", result["markdown"])
        self.assertTrue(Path(result["output_path"]).name.endswith("resume.md"))

    def test_polish_experience_fallback_returns_bullets(self) -> None:
        """验证无 LLM 时经历润色仍能返回简历要点。"""

        bullets = polish_experience("我做过校园网站，主要写后端接口", "Python 后端实习")

        self.assertGreaterEqual(len(bullets), 1)
        self.assertIn("Python 后端实习", bullets[0])


if __name__ == "__main__":
    unittest.main()
