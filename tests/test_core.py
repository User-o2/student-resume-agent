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

    def test_check_missing_fields_reports_project_quality(self) -> None:
        """验证项目描述过于简单时会触发质量追问。"""

        state = collect_resume_info(
            ResumeState(),
            {
                "basic_info": {
                    "name": "李明",
                    "university": "浙江工业大学",
                    "major": "计算机科学与技术",
                    "email": "liming@example.com",
                },
                "job_intention": {"target_position": "Python 后端实习", "expected_city": "杭州"},
                "education": {"courses": ["数据结构"]},
                "projects": [{"title": "图像识别项目", "raw_description": "我做过一个图像识别项目"}],
                "skills": {"programming_languages": ["Python"]},
                "self_evaluation": "学习能力强，关注后端开发方向。",
            },
        )

        report = check_missing_fields(state)
        self.assertFalse(report["is_ready"])
        self.assertTrue(report["quality_questions"])

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
                    "email": "27485938@qq.com",
                },
                "job_intention": {"target_position": "Python 后端实习", "expected_city": "杭州"},
                "education": {"courses": ["机器学习"], "gpa_or_rank": "3/105"},
                "projects": [
                    {},
                    {
                        "technologies": ["Flask", "Redis", "Celery"],
                        "responsibilities": ["负责智能体路由与对话管理模块"],
                        "results": ["接口吞吐量提升约 40%"],
                    },
                ],
                "skills": {"programming_languages": ["Python"]},
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
                    "email": "liming@example.com",
                },
                "job_intention": {"target_position": "Python 后端实习", "expected_city": "杭州"},
                "education": {"courses": ["数据结构"]},
                "projects": [
                    {
                        "title": "校园平台",
                        "technologies": ["Python"],
                        "responsibilities": ["负责后端接口开发"],
                        "results": ["完成 18 个接口"],
                    }
                ],
                "skills": {"programming_languages": ["Python"]},
                "self_evaluation": "具备后端开发实践经验。",
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "resume.md"
            result = fill_resume_template(state, output_path=output_path)

        self.assertIn("# 李明", result["markdown"])
        self.assertIn("校园平台", result["markdown"])
        self.assertTrue(Path(result["output_path"]).name.endswith("resume.md"))

    def test_polish_experience_fallback_returns_bullets(self) -> None:
        """验证无 LLM 时经历润色仍能返回简历要点。"""

        bullets = polish_experience("我做过校园网站，主要写后端接口", "Python 后端实习")

        self.assertGreaterEqual(len(bullets), 1)
        self.assertIn("Python 后端实习", bullets[0])


if __name__ == "__main__":
    unittest.main()
