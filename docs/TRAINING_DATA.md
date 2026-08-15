# FinSight 训练数据协议 v1

## 冻结划分

任务划分由 `sha256(seed + task_id)` 稳定排序后按固定数量切片，任务 ID 两两互斥：

| Partition | 数量 | 来源 | 用途 |
|---|---:|---|---|
| `sft_train` | 4000 | FinQA train | Action-only SFT 候选池 |
| `sft_validation` | 500 | FinQA train | SFT 验证 |
| `grpo_train` | 1500 | FinQA train | 在线 Rollout 候选池 |
| `grpo_validation` | 251 | FinQA train | GRPO 验证 |
| `checkpoint_selection` | 883 | FinQA dev | 统一 checkpoint 选择 |
| `frozen_test` | 1147 | FinQA test | 最终盲测，不参与训练和调参 |

冻结文件和 SHA-256 位于 `data/splits/finqa_v1/manifest.json`。读取时会重新检查文件
行数、task ID 唯一性和 SHA-256；任何手工修改都会失败关闭。

## Oracle Gold Replay

Oracle 构建器使用 Gold evidence 和 Gold program 提议动作，但每一步都必须在
`FinSightEnvironment` 中真实执行：

1. 使用原问题搜索；
2. 对遗漏 Gold evidence 进行细化搜索；
3. 只能打开最新搜索结果中的证据；
4. 只能选择已经打开的证据；
5. 程序必须通过白名单计算器；
6. 执行结果必须与标注答案在 FinQA 舍入容差内一致；
7. Gold evidence coverage 必须为 100%；
8. `verify` 通过后才允许 `deliver`。

Oracle 数据用来验证环境、消息格式和验收器，不代表模型能力。正式教师模型只能看到
Actor-visible Observation，不能读取 Gold evidence、Gold program 或 Gold answer；
它产生的动作仍使用同一环境和验收器。

## Action-only SFT 行格式

训练行使用 `finsight-action-sft-v1`：

- `system`、`user`、`tool` 消息均为 `trainable=false`；
- 只有包含结构化动作的 `assistant` 消息为 `trainable=true`；
- 元数据记录 builder、环境版本、数据规范化版本、事件数和行哈希；
- 训练行不包含隐藏 Gold answer 或 Gold evidence 列表。

当前 100 条 `sft_train` 小样中，98 条通过真实执行验收；2 条为 yes/no 比较题，超出
当前数值型经营分析边界，因此以 `unsupported_non_numeric_answer` 明确拒绝。

## 真实教师采集接口

教师策略通过 OpenAI-compatible `POST /chat/completions` 每轮生成一个 JSON 动作。
每个动作经过相同 Schema、Action Guard、环境执行和确定性验收；接口返回的自然语言
解释或额外字段不会进入训练集。

配置：

```bash
export OPENAI_BASE_URL=https://provider.example/v1
export OPENAI_API_KEY=...
export TEACHER_MODEL=your-teacher-model

PYTHONPATH=src python -m finsight.training.cli teacher \
  --partition sft_train \
  --provider openai \
  --limit 10 \
  --output-dir outputs/teacher_collection_v1
```

如果 Codex 桌面任务无法继承另一个终端的环境变量，可以在项目根目录手工创建已被
`.gitignore` 排除的 `.env.teacher.local`：

```text
OPENAI_BASE_URL=https://provider.example/v1
OPENAI_API_KEY=在本机填写，不要发到聊天或提交仓库
TEACHER_MODEL=your-teacher-model
```

然后为采集命令增加 `--config-file .env.teacher.local`。配置文件只允许这三个键；程序
不会打印密钥，也不会把密钥复制进任何轨迹或 manifest。

该命令会向外部服务发送 FinQA 问题及 Actor-visible Observation，可能产生费用。因此
正式运行前必须确认数据发送范围和接口计费。本项目不会把 Gold evidence、Gold program
或 Gold answer 发送给教师。

采集以 `raw.jsonl` 为可断点续跑的事实来源。`--limit` 表示本次新增尝试数，已有
`task_id` 会跳过。派生文件包括：

- `accepted.jsonl`：通过确定性验收的 Action-only SFT 行；
- `rejected.jsonl`：任务 ID、主错误和拒绝原因；
- `summary.json`：接受率与主错误分布。

无需网络的完整性测试可以使用 `--provider scripted`。脚本教师使用 Gold 生成动作，
只用于验证采集器和断点续跑，不能作为真实教师成绩。

## SFT 数据冻结与 Action-only 编码

教师采集完成后，先执行不可绕过的数据门禁：

```bash
PYTHONPATH=src python -m finsight.training.cli sft-data \
  --train-source outputs/teacher_train_v1/accepted.jsonl \
  --validation-source outputs/teacher_validation_v1/accepted.jsonl \
  --output-dir outputs/sft_data_v1
```

门禁会逐行完成以下检查：

- Schema、消息顺序、`trainable` 标记和最终 `done=true`；
- 每个 Assistant JSON 动作可解析，且下一条 Tool 回执名称与动作一致；
- 事件数与动作数一致，重新计算的行 SHA-256 与元数据一致；
- 估算上下文不超过阈值，train/validation `task_id` 不重叠；
- Oracle builder 默认拒绝进入正式数据，必须显式使用 `--allow-oracle` 才能做管线冒烟。

冻结目录包含规范化的 `train.jsonl`、`validation.jsonl` 和 `manifest.json`，其中记录源文件
及输出文件哈希、动作分布、上下文长度分位数、教师模型、本机依赖和加速器状态。

`encode_action_only` 将 System、User、Tool 以及角色分隔符全部标为 `labels=-100`，只保留
Assistant 动作 JSON 和对应 EOS 参与损失。轨迹一旦超过模型真实 `max_length` 会失败关闭，
不会静默截断 Action/Observation 对。

当前 Oracle 管线冒烟工件为 `artifacts/sft_pipeline_smoke_v1`：train 98 条、validation
97 条、任务重叠为 0，最大保守估算上下文分别为 3034 和 3371 token。该工件的
`formal_training_data_ready=false`，不能替代真实教师数据。

## 首批真实教师试采结论

使用 `finsight-teacher-prompt-v2-dsl-recovery` 对冻结 `sft_train` 的前 5 个任务完成了
真实教师试采，工件位于 `outputs/teacher_deepseek_pilot_v3`：

- 2 条是无失败事件、可直接进入 SFT 的干净轨迹，严格接受率 40%；
- 另外 2 条最终答案正确、证据覆盖 100%、成功 deliver，但早期发生过参数 Schema 错误，
  因而继续按严格规则剔除；端到端任务成功率为 80%；
- 1 条达到 28 步仍缺一条 Gold text evidence，且百分比变化方向错误，没有交付；
- 全部 5 条共 84 个环境动作、23 次搜索和 7 次程序尝试。

试采暴露的首要问题不是接口稳定性，而是动作参数格式和验证失败后的检索效率。因此教师
接口现已在动作进入环境前执行完整 Schema 校验；格式错误会在有限次数内要求教师自修复，
不会污染环境轨迹。`audit-teacher` 会分别报告任务成功率与干净 SFT 接受率：

```bash
PYTHONPATH=src python -m finsight.training.cli audit-teacher \
  --partition sft_train \
  --output-dir outputs/teacher_deepseek_pilot_v3
```

真实试采仍不足以确定正式训练配方。下一批建议在新目录采集 20 条，目标为严格干净轨迹
接受率至少 70%，同时观察 Schema 自修复次数、检索步数 P95 和证据缺失分布。
