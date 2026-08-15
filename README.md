# FinSight

FinSight 是一个面向企业财报的可验证经营分析 Agent。它将财务问题拆成可审计的有限步骤：

`INGESTED → PLANNED → RETRIEVED → CALCULATED → VERIFIED → DELIVERED`

如果关键证据缺失、计算程序不安全或验证失败，流程进入 `REFUSED`，不会输出确定性结论。

## 核心能力

- FinQA 数据读取和表格行序列化；
- 无外部依赖的 BM25 检索及 Recall@K/MRR 评测；
- FinQA 风格白名单计算 DSL，禁止执行任意 Python；
- 证据覆盖验证、计算轨迹和结构化报告；
- 3 个正常业务案例与 1 个拒答案例；
- Streamlit 实时 Agent 单入口；
- 稳定事件轨迹、确定性 Reward 和四面板审计评测。
- 表格字段增强检索和可审计的 Observation/上下文预算。
- 用户可在界面选择 DeepSeek、OpenAI 或其他 OpenAI-compatible 模型，并填写 Base URL、模型 ID 和 API Key；
- 实时 LLM Actor 动态选择动作并生成计算 DSL；
- LLM 与隐藏标准公式、标准答案隔离，所有动作经过 Schema、环境 Guard 和验证器。

## 快速开始

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m finsight.evaluation --data data/raw/finqa/dev.json --limit 100
uv run streamlit run app/streamlit_app.py
```

启动页面后，在“01 · 现场分析”选择模型服务商并填写 Base URL、模型 ID 和 API Key。
API Key 只保存在当前 Streamlit 会话内存中，不写入本地配置、运行报告或下载文件；页面只有
在点击“运行 FinSight Agent”后才调用接口，不会在刷新时自动产生费用。

当前产品支持 Chat Completions 兼容协议：

- DeepSeek：`https://api.deepseek.com`，可选 `deepseek-v4-flash` 或 `deepseek-v4-pro`；
- OpenAI：`https://api.openai.com/v1`，模型 ID 按账号实际可用模型填写；
- 其他兼容服务：手动填写 Base URL 和模型 ID。

接口参数以 [DeepSeek 官方文档](https://api-docs.deepseek.com/api/create-chat-completion)和
[OpenAI 官方文档](https://platform.openai.com/docs/api-reference/chat/create)为准。原生
Anthropic Messages、Gemini GenerateContent 等不同协议需要单独适配器，不能直接填写到兼容入口。
完整参数、安全边界与适配范围见 [`docs/MODEL_CONNECTIONS.md`](docs/MODEL_CONNECTIONS.md)。

如果本机已有兼容环境，也可以：

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m finsight.evaluation --data data/raw/finqa/dev.json --limit 100
streamlit run app/streamlit_app.py
```

Streamlit 页面包含四个区域：

1. 现场分析：配置并运行实时 LLM Agent；
2. 运行轨迹：查看搜索词、动作、证据、公式、工具回执和验证结果；
3. 效果评测：展示本次实时运行和离线检索指标；
4. 项目说明：说明模型、工具、数据、合规和技术边界。

确定性工作流仅保留在工程测试中，用于底层工具链检查、错误定位、自动化测试和版本回归，
不在产品网页中展示。

现场验收的样例输入、预期输出、完整动作轨迹与截图清单见
[`docs/WEB_VALIDATION_GUIDE.md`](docs/WEB_VALIDATION_GUIDE.md)。

## 训练数据准备

冻结互斥任务划分：

```bash
PYTHONPATH=src python -m finsight.training.cli splits
```

构建并真实回放一批 Oracle Action-only SFT 轨迹：

```bash
PYTHONPATH=src python -m finsight.training.cli oracle \
  --partition sft_train \
  --limit 100 \
  --output-dir artifacts/oracle_sft_sample_v1
```

Oracle 轨迹只用于建立数据格式和确定性验收基线，不作为模型泛化能力成绩。

离线验证教师采集器和断点续跑：

```bash
PYTHONPATH=src python -m finsight.training.cli teacher \
  --partition sft_train \
  --provider scripted \
  --limit 10 \
  --output-dir artifacts/teacher_collection_smoke_v1
```

真实 OpenAI-compatible 教师配置及数据发送边界见
[`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md)。

校验并冻结教师轨迹为 Action-only SFT 数据：

```bash
PYTHONPATH=src python -m finsight.training.cli sft-data \
  --train-source outputs/teacher_train_v1/accepted.jsonl \
  --validation-source outputs/teacher_validation_v1/accepted.jsonl \
  --output-dir outputs/sft_data_v1
```

该步骤会复算行哈希、检查动作与工具回执、拒绝上下文超限和任务泄漏，并记录本机训练
依赖与加速器状态。Oracle 数据默认禁止进入正式训练数据；只有管线冒烟时才能显式添加
`--allow-oracle`。

## 数据目录

FinQA 数据来自其[官方仓库](https://github.com/czyssrs/FinQA)，采用 MIT License。原始文件不作为本项目原创内容；训练、验证、测试集合必须严格隔离。业务演示数据位于 `data/business_eval/cases.json`，目前为合成的非敏感数据，便于稳定演示流程。

## 当前边界

本版本完成单主控 LLM Actor、BM25/Table-aware BM25、确定性计算与验证闭环。向量检索、
Reranker、PDF/OCR 泛化、LoRA SFT、完整 E7–E8 训练实验和容器化属于后续迭代范围。

系统仅供经营分析辅助，不构成投资、授信、审计或风控决策。
