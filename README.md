# 学生简历生成智能体

基于 Python、LangChain 和 Streamlit 的学生简历生成 Agent。项目第一阶段聚焦基础闭环：多轮信息采集、缺失字段追问、经历润色、Markdown 模板填充和结果保存。

## 功能范围

- 多轮对话采集求职意向、基本信息、教育背景、项目经历、实习实践、技能奖项和自我评价。
- 使用结构化 `ResumeState` 保存简历状态，避免只依赖聊天历史。
- 提供 4 个核心工具：`collect_resume_info`、`check_missing_fields`、`polish_experience`、`fill_resume_template`。
- 注册 LangChain Tools，并通过 `create_agent` 构建 Agent。
- 使用 Jinja2 模板生成 Markdown 简历，结果保存到 `outputs/`。
- 内置两个学生案例，便于验收演示。

## 项目结构

```text
.
├── app.py
├── app/
│   ├── agent.py
│   ├── config.py
│   ├── prompts.py
│   ├── schema.py
│   └── tools.py
├── data/
│   ├── resume_template.md
│   └── examples/
├── outputs/
├── scripts/
│   └── generate_examples.py
├── tests/
│   └── test_core.py
├── requirements.txt
└── README.md
```

## 环境配置

`.env` 使用以下字段：

```dotenv
base_url=https://example.com/v1/chat/completions
api_key=你的_API_Key
model=qwen3.6-35b-a3b
# 可选：手动控制 HTTPS 证书校验；不设置时会对包含下划线的统一域名自动关闭
ssl_verify=false
```

代码会把 `/chat/completions` 后缀规整为 OpenAI SDK 需要的 `base_url`，并通过 `extra_body={"enable_thinking": false}` 关闭模型 thinking 模式。老师提供的统一域名包含下划线时，Python/OpenSSL 无法通过 wildcard 证书的 hostname 校验，代码会自动关闭该地址的 HTTPS hostname 校验以保证本地验收可联网调用。

## 联网调用检查

```bash
source ~/miniconda3/bin/activate && conda activate langchain
python scripts/check_llm_connection.py
```

## 运行

```bash
source ~/miniconda3/bin/activate && conda activate langchain
streamlit run app.py
```

## 生成示例简历

```bash
source ~/miniconda3/bin/activate && conda activate langchain
python scripts/generate_examples.py
```

输出文件：

- `outputs/resume_case_1.md`
- `outputs/resume_case_2.md`

## 验证

```bash
source ~/miniconda3/bin/activate && conda activate langchain
python -m unittest discover -s tests
```

## 当前阶段未纳入范围

- 上传已有简历解析。
- PDF/Word 导出。
- 照片排版。
- 多版本简历自动对比。

这些属于后续加分功能，建议在基础闭环稳定后继续扩展。
