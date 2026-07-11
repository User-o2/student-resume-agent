"""检查 LangChain Agent 是否真实进入简历主流程。"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import ResumeAgentService
from app.config import EXAMPLES_DIR, load_config
from app.schema import ResumeState


def print_turn_result(title: str, result) -> None:
    """打印单轮 Agent 处理结果。

    Args:
        title: 场景标题。
        result: AgentTurnResult 实例。

    Returns:
        None。
    """

    print(f"\n## {title}")
    print(f"stage: {result.state.current_stage}")
    print(f"target_position: {result.state.job_intention.target_position}")
    print(f"expected_city: {result.state.job_intention.expected_city}")
    print(f"trace: {result.agent_trace}")
    print(f"message: {result.assistant_message[:300]}")
    if result.output_path:
        print(f"output_path: {result.output_path}")


def main() -> None:
    """运行 Agent 主流程联网验收。

    Args:
        无。

    Returns:
        None。
    """

    parser = ArgumentParser(description="检查 LangChain Agent 是否真实进入简历主流程。")
    parser.add_argument(
        "--full",
        action="store_true",
        help="额外验证完整简历生成；该模式会多发一次真实模型请求，可能耗时更长。",
    )
    args = parser.parse_args()

    config = load_config()
    print(f"provider: {config.provider}")
    print(f"base_url: {config.base_url}")
    print(f"model: {config.model}")
    print(f"ssl_verify: {config.ssl_verify}")

    service = ResumeAgentService(config=config, use_llm=True, use_agent_driver=True)

    first_start = time.perf_counter()
    first_turn = service.handle_message(
        "我想投 Python 后端实习，互联网行业，杭州",
        ResumeState(),
    )
    print(f"elapsed_seconds: {time.perf_counter() - first_start:.2f}")
    print_turn_result("采集求职意向", first_turn)

    required_tool_traces = {
        "调用工具：collect_resume_info",
        "调用工具：validate_resume_state",
    }
    missing_tool_traces = required_tool_traces.difference(first_turn.agent_trace)
    if missing_tool_traces:
        raise RuntimeError(f"采集场景缺少真实 LangChain Tool 调用：{sorted(missing_tool_traces)}")
    if not first_turn.state.job_intention.target_position:
        raise RuntimeError("Agent 没有成功更新求职意向。")

    if not args.full:
        print("\n快速 Agent 验收通过。如需额外验证完整生成，运行：python scripts/check_agent_driver.py --full")
        return

    case_data = json.loads((EXAMPLES_DIR / "student_case_1.json").read_text(encoding="utf-8"))
    ready_state = ResumeState.model_validate(case_data)
    generate_start = time.perf_counter()
    generate_turn = service.handle_message("生成简历", ready_state)
    print(f"elapsed_seconds: {time.perf_counter() - generate_start:.2f}")
    print_turn_result("生成完整简历", generate_turn)

    if not generate_turn.output_path:
        raise RuntimeError("生成场景没有输出 Markdown 文件。")


if __name__ == "__main__":
    main()
