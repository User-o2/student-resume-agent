"""解析并优化已有 Markdown 简历。"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import ResumeAgentService
from app.config import OUTPUTS_DIR, load_config


def main() -> None:
    """运行已有简历解析与优化命令。

    Args:
        无。

    Returns:
        None。
    """

    parser = ArgumentParser(description="解析并优化已有 Markdown 简历。")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/resume_to_optimize/resume_1.md",
        help="待优化 Markdown 简历路径。",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选输出 Markdown 路径；默认写入 outputs/optimized_<文件名>_<时间>.md。",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="关闭 LLM，仅使用规则兜底解析和确定性润色。",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"待优化简历不存在：{input_path}")

    output_path = Path(args.output) if args.output else OUTPUTS_DIR / f"optimized_{input_path.stem}.md"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    config = load_config()
    service = ResumeAgentService(use_llm=not args.no_llm, config=config)
    markdown_text = input_path.read_text(encoding="utf-8")
    result = service.optimize_existing_resume(markdown_text, output_path=output_path)

    print(f"input_path: {input_path}")
    print(f"output_path: {result.output_path}")
    print(f"summary: {result.summary}")
    print(f"is_ready: {result.missing_report['is_ready']}")
    print(f"trace: {result.agent_trace}")
    print("\n## 优化结果预览")
    print("\n".join(result.markdown.splitlines()[:40]))


if __name__ == "__main__":
    main()
