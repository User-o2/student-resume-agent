# 学生简历生成智能体

一个基于 Python、LangChain 和 Streamlit 的学生求职简历应用。它通过多轮对话或上传已有 Markdown 简历收集信息，以 LLM 的结构化理解和润色为主、代码校验与模板渲染为辅，生成可直接下载的 Markdown 与 Word 简历。

默认模型为 `qwen3.6-35b-a3b`，运行时关闭 thinking 模式。

## 功能

### 多轮对话生成简历

- 支持自然语言补充求职意向、个人信息、教育背景、技能、项目/实习、竞赛获奖和自我评价。
- LLM 每轮返回结构化决策：识别用户意图、抽取本轮信息补丁、生成自然回复与追问建议。
- 简历状态统一保存为 `ResumeState`，对话中补充的信息会持续合并，不依赖关键词阶段流程。
- 在用户明确发送“生成简历”后，系统执行最终清洗：去重项目和奖项要点、生成正式完整的简历 bullet，并按目标岗位调整表达。
- 代码负责联系方式格式、模板必填字段和项目/奖项最低展示要求等底线校验；校验未通过时不会生成简历。

### 模板与导出

- 使用 `data/resume_template.md` 作为 Jinja2 模板，生成包含个人信息、教育背景、项目经历、竞赛获奖和自我评价的 Markdown 简历。
- 生成结果自动写入 `outputs/`，页面中可直接下载 Markdown。
- 已生成 Markdown 后可导出 `.docx` 文件，并下载 Word 简历。

### 上传已有简历并优化

- 支持上传 UTF-8 编码的 Markdown 或文本简历。
- LLM 将已有简历解析为 `ResumeState`，保留原有事实，再执行去重、项目要点润色和统一模板改写。
- 优化后的 Markdown 简历会写入 `outputs/`，并可继续导出 Word。

### 上传简历评分

- 支持上传 Markdown 简历进行独立评分，不会读取或修改当前对话中的简历草稿。
- 评分报告包括完整度、岗位匹配度、表达规范性、综合评分、优势、主要问题和可执行的优化建议。
- 完整度由代码根据结构化字段计算；岗位匹配度和表达规范性由 LLM 结合原始 Markdown、解析状态和目标岗位评估。

### 对话与执行记录

- 侧边栏展示当前结构化简历状态、待补充字段与完整会话中的执行轨迹。
- 对话、状态、输出路径和评分报告以 JSON 保存到 `outputs/conversations/`，方便复盘演示过程。

## 工作流程

```text
用户对话 / 上传 Markdown 简历
            ↓
LLM 结构化理解或解析
            ↓
ResumeState 状态合并与底线校验
            ↓
LLM 最终润色或结构化评分
            ↓
Jinja2 渲染 Markdown / 生成评分报告
            ↓
下载 Markdown 或导出 Word
```

## 项目结构

```text
.
├── app.py                         # Streamlit 应用入口与页面状态管理
├── app/
│   ├── agent.py                   # LLM 结构化 Agent、简历生成、优化和评分服务
│   ├── config.py                  # 环境变量、模型与输出目录配置
│   ├── prompts.py                 # 对话决策、最终润色、导入和评分 Prompt
│   ├── schema.py                  # ResumeState 及其嵌套数据模型
│   └── tools.py                   # 状态合并、校验、模板填充和 Word 导出函数
├── data/
│   ├── resume_template.md         # Jinja2 Markdown 简历模板
│   ├── resume_to_optimize/        # 待优化或评分的 Markdown 简历示例
│   ├── resume_case/               # 简历案例
│   └── examples/                  # 结构化学生案例数据
├── outputs/
│   ├── conversations/             # 对话与执行记录
│   └── *.md / *.docx              # 生成、优化和导出的简历
├── scripts/
│   ├── generate_examples.py       # 生成内置案例简历
│   └── optimize_existing_resume.py # 命令行优化 Markdown 简历
├── tests/
│   ├── test_agent_flow.py         # Agent 主流程测试
│   ├── test_core.py               # 状态、模板与 Word 导出测试
│   └── test_config.py             # 配置加载测试
├── requirements.txt
└── README.md
```

## 环境配置

### 运行环境

- Python 3.11+
- Miniconda 环境：`langchain`
- 可访问 OpenAI 兼容接口的 Qwen API Key

项目依赖写在 `requirements.txt` 中，包含 LangChain、Streamlit、Jinja2 和 `python-docx`。

```bash
source ~/miniconda3/bin/activate
conda activate langchain
python -m pip install -r requirements.txt
```

### 配置模型接口

在项目根目录创建 `.env`。推荐使用 OpenAI 兼容接口配置：

```dotenv
office_base_url=BASE_URL
office_api_key=你的_API_Key
office_model=qwen3.6-35b-a3b
```

## 启动应用

```bash
source ~/miniconda3/bin/activate
conda activate langchain
streamlit run app.py
```

默认在浏览器打开 `http://localhost:8501`。

## 使用方式

### 对话生成

1. 在聊天框自然描述求职方向、教育背景、项目经历等信息。
2. 根据智能体的追问补充缺失内容。
3. 信息完整后发送“生成简历”。
4. 在“导出简历”区域下载 Markdown，或点击“导出为 Word”生成并下载 `.docx`。

### 已有简历优化

1. 展开“上传已有简历并优化”。
2. 上传 Markdown 或文本简历。
3. 点击“解析并优化已有简历”。
4. 查看统一模板改写后的结果，并按需下载 Markdown 或 Word。

### 简历评分

1. 展开“上传简历评分”。
2. 上传待评分的 Markdown 或文本简历。
3. 可选填写评分目标岗位；留空时使用简历中的求职意向。
4. 点击“开始评分”，查看完整度、匹配度、表达规范性和优化建议。

## 命令行工具

生成内置案例简历：

```bash
source ~/miniconda3/bin/activate
conda activate langchain
python scripts/generate_examples.py
```

优化已有 Markdown 简历：

```bash
source ~/miniconda3/bin/activate
conda activate langchain
python scripts/optimize_existing_resume.py \
  data/resume_to_optimize/resume_1.md \
  --output outputs/optimized_resume_1.md
```

## 测试

```bash
source ~/miniconda3/bin/activate
conda activate langchain
python -m unittest discover -s tests
```
