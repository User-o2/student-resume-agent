"""根据示例 JSON 生成两份 Markdown 简历。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EXAMPLES_DIR, OUTPUTS_DIR, ensure_project_dirs
from app.schema import ResumeState
from app.tools import fill_resume_template, polish_state_experiences


def generate_case(case_file: Path, output_file: Path) -> None:
    """生成单个示例案例的 Markdown 简历。

    Args:
        case_file: 示例 JSON 文件路径。
        output_file: 输出 Markdown 文件路径。

    Returns:
        None。
    """

    data = json.loads(case_file.read_text(encoding="utf-8"))
    state = ResumeState.model_validate(data)
    polished_state = polish_state_experiences(state)
    fill_resume_template(polished_state, output_path=output_file)


def main() -> None:
    """批量生成示例简历。

    Args:
        无。

    Returns:
        None。
    """

    ensure_project_dirs()
    generate_case(EXAMPLES_DIR / "student_case_1.json", OUTPUTS_DIR / "resume_case_1.md")
    generate_case(EXAMPLES_DIR / "student_case_2.json", OUTPUTS_DIR / "resume_case_2.md")


if __name__ == "__main__":
    main()
