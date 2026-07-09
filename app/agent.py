"""学生简历生成智能体的编排逻辑。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppConfig, load_config
from app.prompts import AGENT_DRIVER_PROMPT, AGENT_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT
from app.schema import ResumeState
from app.tools import (
    RESUME_TOOLS,
    check_missing_fields,
    collect_resume_info,
    coerce_resume_state,
    fill_resume_template,
    parse_json_object,
    polish_state_experiences,
)


GENERATE_KEYWORDS = ("生成简历", "输出简历", "导出简历", "完成简历")
GREETING_KEYWORDS = {"你好", "您好", "hi", "hello", "嗨", "在吗", "开始"}
CITY_KEYWORDS = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "武汉",
    "西安",
    "重庆",
    "天津",
    "长沙",
    "厦门",
    "青岛",
    "远程",
)
ROLE_KEYWORDS = (
    "Python 后端实习",
    "后端开发实习",
    "算法实习",
    "机器学习实习",
    "数据分析实习",
    "前端开发实习",
    "Java 后端实习",
    "产品经理实习",
    "测试开发实习",
)
TECH_KEYWORDS = (
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "TypeScript",
    "Go",
    "Flask",
    "Django",
    "FastAPI",
    "Spring Boot",
    "Vue",
    "React",
    "MySQL",
    "PostgreSQL",
    "Redis",
    "Docker",
    "Linux",
    "PyTorch",
    "TensorFlow",
    "OpenCV",
    "YOLO",
    "ResNet",
    "CNN",
    "Pandas",
    "NumPy",
    "Git",
    "Streamlit",
)
GRADE_KEYWORDS = ("大一", "大二", "大三", "大四", "研一", "研二", "研三", "本科", "硕士", "博士")


@dataclass
class AgentTurnResult:
    """单轮对话处理结果。

    Args:
        assistant_message: 返回给用户的消息。
        state: 更新后的简历状态。
        missing_report: 缺失字段与质量检查报告。
        resume_markdown: 生成的 Markdown 简历内容。
        output_path: 生成文件路径。
    """

    assistant_message: str
    state: ResumeState
    missing_report: dict[str, Any]
    resume_markdown: str = ""
    output_path: str = ""
    agent_trace: list[str] = field(default_factory=list)


def build_chat_model(config: AppConfig | None = None) -> BaseChatModel | None:
    """构建 OpenAI 兼容的 LangChain 聊天模型。

    Args:
        config: 应用配置；为空时自动从 `.env` 加载。

    Returns:
        聊天模型实例；缺少 API Key 时返回 None。
    """

    app_config = config or load_config()
    if not app_config.api_key:
        return None

    http_client = httpx.Client(
        verify=app_config.ssl_verify,
        timeout=30,
        trust_env=True,
    )

    return ChatOpenAI(
        model=app_config.model,
        api_key=app_config.api_key,
        base_url=app_config.base_url,
        temperature=app_config.temperature,
        timeout=30,
        max_retries=2,
        extra_body={"enable_thinking": app_config.enable_thinking},
        http_client=http_client,
        http_socket_options=(),
    )


def build_langchain_agent(llm: BaseChatModel | None = None) -> Any | None:
    """构建带工具和内存检查点的 LangChain Agent。

    Args:
        llm: 聊天模型实例。

    Returns:
        LangChain Agent；模型不可用时返回 None。
    """

    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=RESUME_TOOLS,
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def _contains_generate_intent(text: str) -> bool:
    """判断用户是否明确要求生成简历。

    Args:
        text: 用户输入。

    Returns:
        是否为生成意图。
    """

    normalized = text.strip()
    return normalized in {"生成", "输出", "导出"} or any(keyword in normalized for keyword in GENERATE_KEYWORDS)


def _is_greeting_only(text: str) -> bool:
    """判断用户输入是否只是寒暄或开始对话。

    Args:
        text: 用户输入。

    Returns:
        是否为纯寒暄输入。
    """

    normalized = re.sub(r"[\s，,。.!！?？~～]+", "", text.strip().lower())
    return normalized in GREETING_KEYWORDS


def _find_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    """从文本中匹配关键词。

    Args:
        text: 用户输入文本。
        keywords: 候选关键词。

    Returns:
        命中的关键词列表。
    """

    lower_text = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lower_text]


def _split_cn_items(text: str) -> list[str]:
    """按常见中文分隔符拆分条目。

    Args:
        text: 原始文本。

    Returns:
        清洗后的条目列表。
    """

    parts = re.split(r"[、,，;；/|\n]+", text)
    return [part.strip(" ：:。.") for part in parts if part.strip(" ：:。.")]


def _merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """合并两个抽取补丁。

    Args:
        base: 已有补丁。
        patch: 新补丁。

    Returns:
        合并后的补丁。
    """

    if not patch:
        return base
    merged = collect_resume_info(base, patch).model_dump()
    return _compact_patch(merged)


def _compact_patch(value: Any, key: str = "") -> Any:
    """移除补丁中的空值和状态元数据。

    Args:
        value: 待压缩的值。
        key: 当前字段名。

    Returns:
        压缩后的值。
    """

    if key in {"current_stage", "created_at", "updated_at"}:
        return None
    if isinstance(value, dict):
        compacted = {
            item_key: _compact_patch(item_value, item_key)
            for item_key, item_value in value.items()
        }
        return {item_key: item_value for item_key, item_value in compacted.items() if item_value not in (None, "", [], {})}
    if isinstance(value, list):
        compacted_list = [_compact_patch(item) for item in value]
        return [item for item in compacted_list if item not in (None, "", [], {})]
    return value if value not in (None, "", [], {}) else None


def _extract_common_fields(text: str) -> dict[str, Any]:
    """抽取不依赖当前阶段的通用字段。

    Args:
        text: 用户输入。

    Returns:
        字段更新字典。
    """

    update: dict[str, Any] = {}
    basic: dict[str, Any] = {}
    education: dict[str, Any] = {}

    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    if email_match:
        basic["email"] = email_match.group(0)

    phone_match = re.search(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}[-\s]?\d{7,8})(?!\d)", text)
    if phone_match:
        basic["phone"] = phone_match.group(0)

    name_match = re.search(r"(?:我叫|姓名(?:是|为|[:：])?)([\u4e00-\u9fa5A-Za-z·]{2,20})", text)
    if name_match:
        basic["name"] = name_match.group(1).strip()

    native_match = re.search(r"(?:籍贯|家乡)(?:是|为|[:：])?([\u4e00-\u9fa5]{2,20}(?:省|市|自治区|特别行政区)?[\u4e00-\u9fa5]{0,20}(?:市|县|区)?)", text)
    if native_match:
        basic["native_place"] = native_match.group(1).strip()

    university_match = re.search(r"([\u4e00-\u9fa5A-Za-z]+大学)", text)
    if university_match:
        basic["university"] = university_match.group(1)
        education["school"] = university_match.group(1)

    college_match = re.search(r"([\u4e00-\u9fa5A-Za-z]+学院)", text)
    if college_match:
        college = college_match.group(1)
        if not university_match:
            school_hint = re.search(r"(?:学校|院校|毕业院校)(?:是|为|[:：])?([\u4e00-\u9fa5A-Za-z]+学院)", text)
            if school_hint:
                basic["university"] = school_hint.group(1)
                education["school"] = school_hint.group(1)
            else:
                education["college"] = college
        elif college != university_match.group(1):
            education["college"] = college

    major_match = re.search(r"专业(?:是|为|[:：])([\u4e00-\u9fa5A-Za-z0-9+\- ]{2,30})", text)
    if major_match:
        major = major_match.group(1).strip(" ，,。.")
        basic["major"] = major
        education["major"] = major
    else:
        suffix_major_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9+\-]{2,30}专业)", text)
        if suffix_major_match:
            major = suffix_major_match.group(1).removesuffix("专业")
            university = basic.get("university")
            if university and major.startswith(university):
                major = major[len(university) :]
            basic["major"] = major
            education["major"] = major

    for grade in GRADE_KEYWORDS:
        if grade in text:
            basic["grade"] = grade
            break

    course_match = re.search(r"(?:主修课程|课程)(?:包括|有|是|为|[:：])?(.+)", text)
    if course_match:
        education["courses"] = _split_cn_items(course_match.group(1))[:8]

    rank_match = re.search(r"(GPA[:：]?\s*[\d.]+|排名[:：]?\s*前?\s*\d+%?|绩点[:：]?\s*[\d.]+|成绩[:：]?\s*[^，,。]+)", text)
    if rank_match:
        education["gpa_or_rank"] = rank_match.group(1).strip()

    english_matches = re.findall(
        r"(?:CET-?\s*[46]\s*\d{0,3}|英语[四六四6]级\s*\d{0,3}|雅思\s*\d(?:\.\d)?|托福\s*\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if english_matches:
        education["english_level"] = " | ".join(item.strip() for item in english_matches)

    if basic:
        update["basic_info"] = basic
    if education:
        update["education"] = education
    return update


def _extract_job_intention(text: str) -> dict[str, Any]:
    """抽取求职意向字段。

    Args:
        text: 用户输入。

    Returns:
        求职意向更新字典。
    """

    job: dict[str, Any] = {}
    cities = _find_keywords(text, CITY_KEYWORDS)
    roles = _find_keywords(text, ROLE_KEYWORDS)

    if roles:
        job["target_position"] = roles[0]
    else:
        position_match = re.search(r"(?:投递|投|应聘|申请|想做|目标岗位(?:是|为)?)([^，,。；;]+)", text)
        if position_match:
            job["target_position"] = position_match.group(1).strip()

    if cities:
        job["expected_city"] = cities[0]
    industry_match = re.search(r"(互联网|人工智能|金融科技|教育科技|制造业|游戏|电商)", text)
    if industry_match:
        job["target_industry"] = industry_match.group(1)
    return {"job_intention": job} if job else {}


def _extract_experience(text: str, category: str) -> dict[str, Any]:
    """抽取项目或实习实践经历。

    Args:
        text: 用户输入。
        category: `projects` 或 `internships`。

    Returns:
        经历更新字典。
    """

    if category == "internships" and re.search(r"(没有|暂无|无).{0,6}(实习|实践)", text):
        return {"internship_note": "暂无正式实习经历，可使用课程实践、竞赛经历或项目经历补充。"}

    title_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+(?:平台|系统|网站|小程序|项目|模型|算法|应用|课题|实践))", text)
    technologies = _find_keywords(text, TECH_KEYWORDS)
    result_clauses = [
        clause.strip()
        for clause in re.split(r"[。；;\n]", text)
        if re.search(r"\d|%|提升|准确率|召回率|上线|用户|排名|效率|降低|优化", clause)
    ]
    responsibility_clauses = [
        clause.strip()
        for clause in re.split(r"[。；;\n]", text)
        if re.search(r"负责|开发|实现|设计|搭建|完成|参与|优化|训练|部署", clause)
    ]

    item: dict[str, Any] = {
        "title": title_match.group(1) if title_match else "",
        "technologies": technologies,
        "responsibilities": responsibility_clauses[:4] or [text.strip()],
        "results": result_clauses[:3],
        "raw_description": text.strip(),
    }

    if category == "internships":
        org_match = re.search(r"在([^，,。；;]{2,30})(?:担任|做|实习|实践)", text)
        role_match = re.search(r"(?:担任|岗位(?:是|为)?)([^，,。；;]{2,20})", text)
        if org_match:
            item["organization"] = org_match.group(1).strip()
        if role_match:
            item["role"] = role_match.group(1).strip()

    return {category: [item]}


def _extract_skills_and_awards(text: str) -> dict[str, Any]:
    """抽取技能与荣誉奖项。

    Args:
        text: 用户输入。

    Returns:
        技能和奖项更新字典。
    """

    update: dict[str, Any] = {}
    technologies = _find_keywords(text, TECH_KEYWORDS)
    programming = [item for item in technologies if item in {"Python", "Java", "C++", "JavaScript", "TypeScript", "Go"}]
    tools = [item for item in technologies if item not in set(programming)]

    skills: dict[str, Any] = {}
    if programming:
        skills["programming_languages"] = programming
    if tools:
        skills["tools"] = tools
    if any(keyword in text for keyword in ("数据分析", "机器学习", "后端开发", "接口设计", "数据库", "深度学习")):
        skills["professional_skills"] = _split_cn_items(text)[:8]
    language_match = re.search(r"(英语[四六六四]级|CET-?[46]|雅思\d(?:\.\d)?|托福\d+)", text, flags=re.IGNORECASE)
    if language_match:
        skills["languages"] = [language_match.group(1)]
    if skills:
        update["skills"] = skills

    award_clauses = [
        clause.strip()
        for clause in re.split(r"[。；;\n]", text)
        if any(keyword in clause for keyword in ("奖", "证书", "竞赛", "奖学金"))
    ]
    if award_clauses:
        awards: list[dict[str, Any]] = []
        for clause in award_clauses[:5]:
            name_match = re.search(
                r"([\u4e00-\u9fa5A-Za-z0-9 \-]+(?:竞赛|比赛|奖学金|证书|奖)[\u4e00-\u9fa5A-Za-z0-9 \-]*(?:一等奖|二等奖|三等奖|Top\s*\d+%|前\s*\d+%)?)",
                clause,
                flags=re.IGNORECASE,
            )
            date_match = re.search(r"(20\d{2}年?)", clause)
            level_match = re.search(r"(国家级|省级|校级|院级|Top\s*\d+%|前\s*\d+%)", clause, flags=re.IGNORECASE)
            awards.append(
                {
                    "name": (name_match.group(1).strip() if name_match else clause[:50]),
                    "date": date_match.group(1) if date_match else "",
                    "level": level_match.group(1) if level_match else "",
                    "description": clause,
                    "highlights": [clause],
                }
            )
        update["awards"] = awards
    return update


def _heuristic_extract_update(text: str, state: ResumeState) -> dict[str, Any]:
    """在 LLM 不可用时根据当前阶段进行规则抽取。

    Args:
        text: 用户输入。
        state: 当前简历状态。

    Returns:
        字段更新字典。
    """

    update = _extract_common_fields(text)
    stage = state.current_stage

    if stage in {"personal_info", "job_intention"}:
        update = _merge_patch(update, _extract_job_intention(text))
    elif stage in {"education", "basic_education"}:
        update = _merge_patch(update, _extract_skills_and_awards(text))
    elif stage == "projects":
        update = _merge_patch(update, _extract_experience(text, "projects"))
    elif stage in {"awards", "skills_awards"}:
        update = _merge_patch(update, _extract_skills_and_awards(text))
    elif stage == "self_evaluation":
        update["self_evaluation"] = text.strip()
    else:
        update = _merge_patch(update, _extract_job_intention(text))
        update = _merge_patch(update, _extract_skills_and_awards(text))

    if any(keyword in text for keyword in ("岗位", "投", "应聘", "实习", "城市")):
        update = _merge_patch(update, _extract_job_intention(text))
    if any(keyword in text for keyword in ("项目", "平台", "系统", "模型", "算法", "网站", "小程序", "应用")):
        update = _merge_patch(update, _extract_experience(text, "projects"))
    if any(keyword in text for keyword in ("竞赛", "比赛", "获奖", "奖学金", "证书")):
        update = _merge_patch(update, _extract_skills_and_awards(text))
    if any(keyword in text for keyword in ("实践", "课题", "社团")) and stage not in {"personal_info", "job_intention"}:
        update = _merge_patch(update, _extract_experience(text, "projects"))
    return _compact_patch(parse_json_object(json.dumps(update, ensure_ascii=False))) or {}


def _llm_extract_update(text: str, state: ResumeState, llm: BaseChatModel | None) -> dict[str, Any]:
    """使用 LLM 抽取用户本轮提供的结构化信息。

    Args:
        text: 用户输入。
        state: 当前简历状态。
        llm: 聊天模型实例。

    Returns:
        字段更新字典。
    """

    if llm is None:
        return {}

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            (
                "human",
                "当前阶段：{stage}\n当前状态 JSON：{state_json}\n用户本轮输入：{user_input}",
            ),
        ]
    )
    try:
        response = (prompt | llm).invoke(
            {
                "stage": state.current_stage,
                "state_json": state.model_dump_json(ensure_ascii=False),
                "user_input": text,
            }
        )
        content = getattr(response, "content", str(response))
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return parse_json_object(str(content))
    except Exception:
        return {}


def _has_missing_prefix(report: dict[str, Any], prefix: str) -> bool:
    """判断缺失报告中是否存在指定板块缺口。

    Args:
        report: 缺失字段报告。
        prefix: 缺失字段前缀。

    Returns:
        是否存在指定板块缺失。
    """

    return any(str(item).startswith(prefix) for item in report.get("missing_fields", []))


def _sync_stage(state: ResumeState, previous_stage: str, report: dict[str, Any]) -> ResumeState:
    """根据状态完整度同步当前对话阶段。

    Args:
        state: 当前简历状态。
        previous_stage: 更新前阶段。
        report: 缺失字段报告。

    Returns:
        更新阶段后的简历状态。
    """

    if _has_missing_prefix(report, "个人信息："):
        state.current_stage = "personal_info"
    elif _has_missing_prefix(report, "教育背景："):
        state.current_stage = "education"
    elif _has_missing_prefix(report, "项目经历："):
        state.current_stage = "projects"
    elif _has_missing_prefix(report, "竞赛获奖："):
        state.current_stage = "awards"
    elif _has_missing_prefix(report, "自我评价："):
        state.current_stage = "self_evaluation"
    else:
        state.current_stage = "ready"
    state.touch()
    return state


def build_followup_question(state: ResumeState, report: dict[str, Any]) -> str:
    """根据阶段和缺失报告生成追问。

    Args:
        state: 当前简历状态。
        report: 缺失字段报告。

    Returns:
        下一轮对用户的追问文本。
    """

    missing_fields = report.get("missing_fields", [])
    optional_suggestions = report.get("optional_suggestions", [])
    quality_questions = report.get("quality_questions", [])

    if state.current_stage in {"personal_info", "job_intention"}:
        personal_missing = [
            item.replace("个人信息：", "")
            for item in missing_fields
            if item.startswith("个人信息：")
        ]
        focus = "、".join(personal_missing[:3]) if personal_missing else "目标岗位、目标行业、期望城市、姓名、电话、邮箱、籍贯"
        return f"请先补充个人信息：{focus}。这些信息会出现在简历顶部。"
    if state.current_stage in {"education", "basic_education"}:
        education_missing = [
            item.replace("教育背景：", "")
            for item in missing_fields
            if item.startswith("教育背景：")
        ]
        focus = "、".join(education_missing[:3]) if education_missing else "学校、学院、专业、专业排名、英语水平、核心课程、技术栈"
        return f"请补充教育背景：{focus}。"
    if state.current_stage == "projects":
        if quality_questions:
            return quality_questions[0]
        return "请补充至少 1 段项目经历，包含项目名称和 2-3 条项目要点，建议覆盖技术方法、个人职责和量化结果。"
    if state.current_stage in {"awards", "skills_awards"}:
        return "请补充至少 1 项竞赛获奖、奖学金或证书，包含奖项名称和 1-2 条说明，例如负责内容、技术方法、排名或成果。"
    if state.current_stage == "self_evaluation":
        return "请补充自我评价，建议 2-3 条，说明技术兴趣、实践能力、协作表达或职业方向。"
    if quality_questions:
        return f"必要信息已完整，可以回复“生成简历”。如果想继续优化，建议补充：{quality_questions[0]}"
    if optional_suggestions:
        return f"必要信息已完整，可以回复“生成简历”。如果想继续优化，建议先补充：{optional_suggestions[0]}"
    return "必要信息已完整。回复“生成简历”即可输出 Markdown 简历，也可以继续补充需要强调的内容。"


class ResumeAgentService:
    """面向 UI 和脚本的简历 Agent 服务。"""

    def __init__(
        self,
        use_llm: bool = True,
        config: AppConfig | None = None,
        use_agent_driver: bool = True,
    ) -> None:
        """初始化简历 Agent 服务。

        Args:
            use_llm: 是否启用 LLM 抽取与润色。
            config: 可选应用配置。
            use_agent_driver: 是否使用 LangChain Agent 驱动主流程。

        Returns:
            None。
        """

        self.config = config or load_config()
        self.llm = build_chat_model(self.config) if use_llm else None
        self.langchain_agent = build_langchain_agent(self.llm)
        self.thread_id = str(uuid.uuid4())
        self.use_agent_driver = use_agent_driver

    def extract_update(self, user_input: str, state: ResumeState) -> dict[str, Any]:
        """抽取用户本轮提供的简历字段更新。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            字段更新字典。
        """

        llm_update = _llm_extract_update(user_input, state, self.llm)
        heuristic_update = _heuristic_extract_update(user_input, state)
        return _merge_patch(heuristic_update, _compact_patch(llm_update) or {})

    def handle_message(
        self,
        user_input: str,
        state: ResumeState | dict[str, Any] | str | None,
    ) -> AgentTurnResult:
        """处理单轮用户消息。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            单轮处理结果。
        """

        resume_state = coerce_resume_state(state)
        if _is_greeting_only(user_input):
            report = check_missing_fields(resume_state)
            return AgentTurnResult(
                assistant_message=(
                    "你好，我会通过多轮对话帮你生成学生简历。"
                    "请先告诉我这份简历的目标岗位、目标行业、期望城市，以及姓名、电话、邮箱和籍贯。"
                ),
                state=resume_state,
                missing_report=report,
                agent_trace=["fast_path: greeting"],
            )

        if self.use_agent_driver and self.langchain_agent is not None:
            agent_result = self._handle_message_with_agent(user_input, resume_state)
            if agent_result is not None:
                return agent_result

        return self._handle_message_controlled(user_input, resume_state)

    def _handle_message_controlled(
        self,
        user_input: str,
        state: ResumeState | dict[str, Any] | str | None,
    ) -> AgentTurnResult:
        """使用确定性编排处理单轮消息。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            单轮处理结果。
        """

        resume_state = coerce_resume_state(state)
        previous_stage = resume_state.current_stage

        if user_input.strip():
            update = self.extract_update(user_input, resume_state)
            resume_state = collect_resume_info(resume_state, update)

        report = check_missing_fields(resume_state)
        resume_state = _sync_stage(resume_state, previous_stage, report)
        report = check_missing_fields(resume_state)

        if _contains_generate_intent(user_input):
            if not report["is_ready"]:
                missing_text = "；".join(report["missing_fields"])
                message = f"现在还不能生成完整简历，仍需补充：{missing_text}\n\n{build_followup_question(resume_state, report)}"
                return AgentTurnResult(message, resume_state, report, agent_trace=["fallback: controlled_flow"])

            polished_state = polish_state_experiences(resume_state, self.llm)
            result = fill_resume_template(polished_state)
            message = f"已生成 Markdown 简历：{result['output_path']}\n\n{result['markdown']}"
            return AgentTurnResult(
                assistant_message=message,
                state=polished_state,
                missing_report=report,
                resume_markdown=result["markdown"],
                output_path=result["output_path"],
                agent_trace=["fallback: controlled_flow"],
            )

        return AgentTurnResult(
            assistant_message=build_followup_question(resume_state, report),
            state=resume_state,
            missing_report=report,
            agent_trace=["fallback: controlled_flow"],
        )

    def _handle_message_with_agent(self, user_input: str, state: ResumeState) -> AgentTurnResult | None:
        """使用 LangChain Agent 主导处理单轮消息。

        Args:
            user_input: 用户输入。
            state: 当前简历状态。

        Returns:
            Agent 处理结果；失败时返回 None 以便兜底。
        """

        if self.langchain_agent is None:
            return None

        prompt = AGENT_DRIVER_PROMPT.format(
            state_json=state.model_dump_json(ensure_ascii=False),
            current_stage=state.current_stage,
            user_input=user_input,
        )

        try:
            raw_result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={
                    "configurable": {"thread_id": f"{self.thread_id}:{uuid.uuid4()}"},
                    "recursion_limit": 8,
                },
            )
        except Exception:
            return None

        payload, tool_state, tool_report, tool_resume, trace = self._parse_agent_result(raw_result)
        if not trace:
            return None
        payload_state = payload.get("state_json") if isinstance(payload, dict) else ""
        updated_state = coerce_resume_state(payload_state or tool_state or state)
        previous_stage = state.current_stage
        report = tool_report or check_missing_fields(updated_state)
        updated_state = _sync_stage(updated_state, previous_stage, report)
        report = check_missing_fields(updated_state)

        generated_markdown = ""
        output_path = ""
        if tool_resume:
            generated_markdown = str(tool_resume.get("markdown", ""))
            output_path = str(tool_resume.get("output_path", ""))
        elif payload.get("resume_markdown"):
            generated_markdown = str(payload.get("resume_markdown", ""))
            output_path = str(payload.get("output_path", ""))

        if output_path and not generated_markdown:
            generated_markdown = self._read_generated_markdown(output_path)

        if _contains_generate_intent(user_input) and report["is_ready"] and not output_path:
            result = fill_resume_template(updated_state)
            generated_markdown = result["markdown"]
            output_path = result["output_path"]
            trace.append("服务端补偿执行：fill_resume_template")

        assistant_message = str(payload.get("assistant_message") or "").strip()
        if not generated_markdown:
            assistant_message = build_followup_question(updated_state, report)
        elif not assistant_message:
            assistant_message = build_followup_question(updated_state, report)

        if generated_markdown and output_path:
            assistant_message = f"已生成 Markdown 简历：{output_path}\n\n{generated_markdown}"

        return AgentTurnResult(
            assistant_message=assistant_message,
            state=updated_state,
            missing_report=report,
            resume_markdown=generated_markdown,
            output_path=output_path,
            agent_trace=trace,
        )

    def _read_generated_markdown(self, output_path: str) -> str:
        """从输出文件读取 Agent 生成的 Markdown 简历。

        Args:
            output_path: Markdown 文件路径。

        Returns:
            Markdown 文本；读取失败时返回空字符串。
        """

        try:
            return Path(output_path).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _parse_agent_result(
        self,
        raw_result: dict[str, Any],
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], list[str]]:
        """解析 LangChain Agent 输出、工具返回和调用轨迹。

        Args:
            raw_result: Agent invoke 返回值。

        Returns:
            最终 JSON、工具状态 JSON、缺失报告、生成结果、调用轨迹。
        """

        messages = raw_result.get("messages", []) if isinstance(raw_result, dict) else []
        last_human_index = -1
        for index, message in enumerate(messages):
            if message.__class__.__name__ == "HumanMessage":
                last_human_index = index
        messages = messages[last_human_index + 1 :]
        final_payload: dict[str, Any] = {}
        tool_state = ""
        tool_report: dict[str, Any] = {}
        tool_resume: dict[str, Any] = {}
        trace: list[str] = []

        for message in messages:
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name", "unknown_tool")
                args = call.get("args", {})
                trace.append(f"调用工具：{name}，参数：{list(args.keys())}")

            content = getattr(message, "content", "")
            if not content:
                continue

            parsed = parse_json_object(str(content))
            if not parsed:
                continue

            if {"basic_info", "job_intention", "education"}.issubset(parsed.keys()):
                tool_state = json.dumps(parsed, ensure_ascii=False)
            elif {"missing_fields", "quality_questions", "is_ready"}.issubset(parsed.keys()):
                tool_report = parsed
            elif {"markdown", "output_path"}.issubset(parsed.keys()):
                tool_resume = parsed
            else:
                final_payload = parsed

        return final_payload, tool_state, tool_report, tool_resume, trace
