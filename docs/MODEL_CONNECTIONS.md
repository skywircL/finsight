# FinSight 模型连接说明

## 产品支持范围

FinSight 主界面始终运行实时 LLM Agent。用户在当前 Streamlit 会话中选择服务商并填写：

- Base URL；
- 模型 ID；
- API Key；
- 可选的高级参数。

当前适配器支持 OpenAI-compatible **Chat Completions** 协议。DeepSeek 和 OpenAI 使用预设入口；其他兼容服务允许手动填写。原生 Anthropic Messages、Gemini GenerateContent 等不同协议需要新增独立适配器，不能假设与 Chat Completions 完全兼容。

## 通用请求

```http
POST {BASE_URL}/chat/completions
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

基础请求体：

```json
{
  "model": "用户选择的模型 ID",
  "messages": [
    {"role": "system", "content": "FinSight 动作策略提示词"},
    {"role": "user", "content": "当前问题与环境 Observation"}
  ],
  "response_format": {"type": "json_object"}
}
```

模型输出必须是单个 FinSight 动作 JSON。系统会再次执行 Schema 校验，不会因为服务商声称支持 JSON mode 就跳过验证。

## DeepSeek

- Base URL：`https://api.deepseek.com`；
- 当前预设模型：`deepseek-v4-flash`、`deepseek-v4-pro`；
- 鉴权：`Authorization: Bearer <API_KEY>`；
- 结构化输出：`response_format={"type":"json_object"}`；
- 可选思考参数：`thinking.type=enabled|disabled`；启用时发送 `reasoning_effort=high`。

接口依据：[DeepSeek Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)和[DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)。

## OpenAI

- Base URL：`https://api.openai.com/v1`；
- 模型 ID：按用户账号通过 Models API 可见的实际模型填写；
- 鉴权：`Authorization: Bearer <API_KEY>`；
- 当前适配器调用 `/chat/completions`；
- 默认发送 JSON object response format；为兼容推理模型，不主动发送 `temperature`。

接口依据：[OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat/create)和[OpenAI Models API](https://platform.openai.com/docs/api-reference/models/list)。

## 其他 OpenAI-compatible 服务

用户可填写 API 根地址，或直接填写以 `/chat/completions` 结尾的完整地址。高级设置允许：

- 开关 `response_format=json_object`；
- 设置 `temperature`；
- 设置单次请求超时。

“OpenAI-compatible”只代表请求和响应结构相近，不代表所有服务都支持相同参数。如果服务商不支持 JSON mode，可在高级设置关闭，由提示词和 FinSight Schema 校验继续约束动作。

## 密钥与数据边界

- API Key 使用密码输入框，只保存在当前 Streamlit 会话内存；
- 不写入 `.env`、日志、运行轨迹、下载 JSON 或项目文档；
- FinSight 只向用户填写的 Base URL 发送当前任务问题、可见 Observation 和工具回执；
- 不向模型发送隐藏标准证据 ID、预置公式或标准答案；
- 用户应确认所选服务商的数据保留、跨境传输、费用和使用条款。

## 确定性工作流的位置

确定性工作流不属于产品模型连接能力，也不会出现在主运行入口。它只保留在“03 · 效果评测”的开发对照区域，用于：

- API 不可用时检查底层工具链；
- 区分模型策略错误与工具错误；
- 自动化测试和版本回归；
- 比较固定流程与模型自主决策。
