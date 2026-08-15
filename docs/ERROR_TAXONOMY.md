# FinSight 错误分类 v1

轨迹错误由代码根据真实事件、终止状态和确定性验收结果分类，不使用 LLM Judge。

## 分类维度

| Category | 典型错误 |
|---|---|
| `task` | task ID 不一致、当前不支持非数值答案 |
| `protocol` | 非法 JSON、未知动作、参数不合法、Guard 拒绝 |
| `retrieval` | Gold 证据没有完整召回 |
| `evidence` | 证据已出现但没有被选择 |
| `program` | 程序缺失或不安全 |
| `calculation` | 计算执行异常 |
| `verification` | 结果与标注答案不一致 |
| `termination` | 未验证交付、过早拒答、循环、最大步数 |
| `context` | Observation 或累计上下文溢出 |
| `infrastructure` | 教师接口、环境服务或采集流程异常 |

## 主错误优先级

同一轨迹可能同时出现“未交付、答案不匹配、证据不完整”。系统保留全部次级错误，但按
稳定优先级选择一个 `primary_error`：基础设施和任务边界优先，其次是上下文、循环、
协议、计算、程序、证据、答案和终止。这样跨实验比较不会因字典顺序或模型判断变化。

每个 Diagnosis 包含：

- taxonomy 版本；
- `valid`；
- `primary_error`；
- `secondary_errors`；
- `primary_category`；
- 对应的 `failing_event_ids`。

错误分类只用于诊断和数据验收，不与 Reward 强行合成为一个总分。
