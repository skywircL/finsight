# FinSight 可视化网页验证指南

## 1. 验证目标

本轮网页验证要证明实时 LLM 确实参与了 Agent 决策，而不是只展示一个最终数字。每次运行至少保存以下证据：

1. 用户选择的服务商和模型；
2. LLM 生成的搜索词；
3. 检索返回、打开和选择的 evidence_id；
4. LLM 生成的受限计算公式；
5. 确定性计算工具的执行结果；
6. Verifier 的证据覆盖与验证回执；
7. 最终 `DELIVERED` 或 `REFUSED` 状态；
8. 下载的完整运行 JSON。

当前网页使用四组预解析样例资料，不支持上传任意财报或脱离当前资料包自由提问。测试时应从页面提供的任务中选择，不要把样例输入框理解为通用财报聊天框。

## 2. 启动与模型配置

启动：

```bash
streamlit run app/streamlit_app.py
```

打开 `01 · 现场分析`，完成以下配置：

### DeepSeek 示例

- 模型服务商：`DeepSeek`；
- Base URL：`https://api.deepseek.com`；
- 模型：优先用 `deepseek-v4-flash`，需要更强推理时用 `deepseek-v4-pro`；
- API Key：填写新生成的有效密钥；
- 高级设置：第一次测试建议关闭 Thinking，减少延迟与费用；
- 点击：`运行 FinSight Agent`。

### OpenAI 示例

- 模型服务商：`OpenAI`；
- Base URL：`https://api.openai.com/v1`；
- 模型 ID：填写账号实际可用、支持 Chat Completions 的模型；
- API Key：填写对应账号的有效密钥；
- 点击：`运行 FinSight Agent`。

API Key 只应通过页面密码框填写。不要把密钥写入截图、录屏、下载 JSON、文档或聊天消息。

## 3. 样例一：收入增长分析——验证正常交付

### 页面操作

1. 进入 `01 · 现场分析`；
2. 选择 `收入增长分析`；
3. 页面显示样例输入：

```text
2023 revenue compared with 2022 revenue percentage change
```

4. 展开“查看本任务资料与边界”，确认资料中包含：

```text
[page=42] [table=Income statement] [unit=million CNY]
metric=revenue 2023=1250 2022=1100
```

5. 填写模型接口并点击 `运行 FinSight Agent`；
6. 运行完成后查看本页报告；
7. 打开 `02 · 运行轨迹`，逐项检查动作和环境回执；
8. 打开 `03 · 效果评测`，检查本次运行终局与证据覆盖；
9. 回到 `02 · 运行轨迹` 下载运行 JSON。

### 预期 Agent 流程

模型生成的具体搜索词和公式写法可以不同，但逻辑上应完成：

```text
search
→ open_evidence(income_statement)
→ select_evidence([income_statement])
→ emit_program
→ calculate
→ verify
→ deliver
```

搜索词应包含收入指标和年份，例如：

```text
2023 2022 revenue percentage change
```

模型生成的公式可以是以下任一等价 DSL：

```text
multiply(divide(subtract(1250,1100),1100),const_100)
```

或：

```text
divide(multiply(subtract(1250,1100),const_100),1100)
```

确定性工具应执行：

```text
1250 - 1100 = 150
150 / 1100 = 0.1363636...
0.1363636... × 100 = 13.63636...
```

### 预期输出

- 终局：`DELIVERED`；
- 结论：约 `13.64%`；
- 证据：包含 `income_statement`；
- 证据页码：`42`；
- 验证结果：通过；
- 程序安全：通过；
- 报告含义：2023 年收入相对 2022 年增长约 13.64%；该结果不代表利润、现金流或整体经营质量同步增长。

### 需要保存的运行证据

- 截图 A：`01 · 现场分析`中的模型、结论、API 调用次数和耗时；
- 截图 B：`02 · 运行轨迹`中的 `search` 参数，证明搜索词由 LLM 产生；
- 截图 C：`open_evidence`、`select_evidence` 和 `income_statement` 原文；
- 截图 D：`emit_program`、`calculate` 和 `verify` 回执；
- 文件：`finsight-revenue_growth-run.json`。

## 4. 样例二：存货周转率核查——验证缺证拒答

### 页面操作

1. 回到 `01 · 现场分析`；
2. 选择 `存货周转率核查`；
3. 页面显示样例输入：

```text
2023 inventory turnover based on cost of sales and average inventory
```

4. 展开资料，确认当前只能看到：

```text
[page=44] cost of sales 2023=730
[page=102] Inventories are measured at the lower of cost and net realisable value.
```

资料中没有平均存货、期初存货或期末存货数值。

5. 使用同一模型配置运行 Agent；
6. 打开 `02 · 运行轨迹`，确认模型搜索过平均存货相关证据；
7. 查看最终拒答原因并下载 JSON。

### 预期 Agent 流程

允许模型使用不同搜索词或多次搜索，但最终不应伪造平均存货：

```text
search("2023 average inventory inventory turnover")
→ 可选的 open_evidence
→ 可选的再次 search
→ abstain(reason_code="missing_evidence")
```

### 预期输出

- 终局：`REFUSED`；
- 分析结论：不输出具体周转率；
- 缺失证据：`inventory_table`；
- 拒答原因：缺少平均存货或期初/期末存货数值；
- 不应出现 `730 / 125`、`5.84 次`等使用隐藏预设数据产生的结果；
- 不应进入 `deliver`。

### 需要保存的运行证据

- 截图 A：`01 · 现场分析`中的“安全拒答”；
- 截图 B：`02 · 运行轨迹`中的搜索词和已经打开的证据；
- 截图 C：`abstain(missing_evidence)`与最终 `REFUSED`；
- 截图 D：证据页中的 `inventory_table` 缺失提示；
- 文件：`finsight-missing_inventory-run.json`。

## 5. 可选补充样例

### 营业利润变化分析

输入：

```text
2023 operating profit compared with 2022 operating profit percentage change
```

资料：2023 年营业利润 562，2022 年为 491，单位百万元。

预期结果：

```text
(562 - 491) / 491 × 100 = 14.4603...% ≈ 14.46%
```

### 经营现金流变化分析

输入：

```text
2023 operating cash flow compared with 2022 operating cash flow percentage change
```

资料：2023 年经营现金流 310，2022 年为 400，单位百万元。

预期结果：

```text
(310 - 400) / 400 × 100 = -22.5%
```

业务解释必须指出：收入增长但经营现金流下降时，需要继续核查应收账款、回款和存货占用，不能只根据收入增长判断经营质量。

## 6. 运行失败时如何判断

| 页面表现 | 可能原因 | 处理方式 |
|---|---|---|
| 401/鉴权失败 | API Key 无效或已撤销 | 在服务商控制台生成新密钥后重新填写 |
| 404/接口不存在 | Base URL 或模型 ID 错误 | 核对官方文档；Base URL 可填根地址或完整 `/chat/completions` 地址 |
| JSON/Schema 多次失败 | 模型未稳定输出动作 JSON | 查看 Schema 修复次数；换用更强模型或开启 JSON mode |
| 达到最大步数 | 模型重复搜索、重复验证或没有及时拒答 | 在运行轨迹中定位重复动作，保存 JSON 作为失败案例 |
| `REFUSED` 出现在正常样例 | 模型没有完整选择证据或公式未通过 | 查看 `verify` 回执，确认缺失的是证据、程序还是数值 |
| 缺证样例却输出数字 | 模型或环境边界失效 | 立即保存轨迹；该结果不合格，不用于演示 |

## 7. 一次合格验收的文件清单

```text
evidence/
├── 01-revenue-live-result.png
├── 02-revenue-action-trace.png
├── 03-revenue-evidence-formula-verify.png
├── 04-inventory-refusal.png
├── 05-inventory-missing-evidence.png
├── finsight-revenue_growth-run.json
└── finsight-missing_inventory-run.json
```

正常案例与拒答案例都通过后，才能说明 FinSight 同时具备“完成任务”和“守住边界”两类 Agent 能力。
