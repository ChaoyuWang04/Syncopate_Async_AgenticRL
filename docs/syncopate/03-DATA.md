# Syncopate · 数据

> 本文是 v16 数据的唯一现行说明：数据怎么产生、如何切分、经过哪些门禁、产物在哪里。
> 数据相关工作排期只看 [01-TASKS.md](01-TASKS.md)。

## 1. 当前状态

| 产物 | 当前事实 | 证据 |
|---|---|---|
| case 库 | 2030 个 case | [local_gen_sha.json](../../_audit/v16/local_gen_sha.json) |
| 三桶切分 | EVAL 401 / SFT 597 / RL 1032 | 同上；本机与 Modal 三份 SHA-256 逐一相同 |
| SFT 数据 | train 1222 行、18 桶 | Modal `/vol/data/sft/v16`；run28 建库审计 |
| 数据质量 | 现行题库门禁、出厂体检、prompt 预算和三桶隔离全绿 | Modal `/vol/_audit/v16/` |
| 人工检查 | 六族样本已经抽看；正式 candidate 前仍按 TASKS 冻结最终带宽和新增门禁 | [01-TASKS.md](01-TASKS.md#t2--candidate-训练与质量验收) |

run28 的“全绿”指当次预注册的 report 模式全部通过。六桶份额带宽仍是报告项，
不能把“报告出来”误写成“严格阈值已经最终批准”。

## 2. 数据生命周期

固定入口 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh) 中，数据部分按以下顺序运行：

| 阶段 | 做什么 | 正式实现 | 主要产物 |
|---|---|---|---|
| cases | 按 v16 规格生成 case、gold 和 verifier | [syncopate/cli.py](../../syncopate/cli.py)、[authoring](../../syncopate/authoring) | `data/batches/v16` |
| menus | 为 verifier 和 routing 计算 case 工具菜单 | [tool_menus.py](../../syncopate/pipeline/tool_menus.py) | case 内的 `tool_menu` |
| split | 生成互斥的 EVAL/SFT/RL 三桶 | [split.py](../../syncopate/pipeline/split.py) | `data/splits/v16` |
| gates | 运行 D1–D11 与 L1/L2 | [data_gates.py](../../syncopate/pipeline/data_gates.py) | 门禁输出 |
| supply | 在调用教师前核对每类底题供给 | [supply_gate.py](../../syncopate/pipeline/supply_gate.py) | 供给判定 |
| rl-data | 渲染 RL train/val 并复核隔离 | [build_dataset.py](../../syncopate/pipeline/build_dataset.py) | `data/rl/v16` |
| teacher | 启动统一教师端点 | 模型来源见 [model_paths.py](../../syncopate/core/model_paths.py) | OpenAI 兼容端点 |
| sft-data | 构造多轮、CoT 和终答，完成全部出厂检查 | [build_sft.py](../../syncopate/pipeline/build_sft.py) | `data/sft/v16`、画廊和审计 |

云上编排只负责调用固定阶段，不另写一套数据逻辑。

## 3. 三桶隔离

三桶的职责：

- **EVAL**：冻结评测，只用于比较模型，不进入训练。
- **SFT**：监督学习和教师扩写的底题来源。
- **RL**：强化学习 rollout 的底题来源。

隔离依靠硬机制：

1. [split.py](../../syncopate/pipeline/split.py) 在源头生成互斥清单。
2. 派生样本记录 `case_id` 和 `source_case_ids`。
3. 正式写盘走带隔离断言的出口。
4. 写盘后由 [split_isolation.py](../../syncopate/pipeline/split_isolation.py) 独立复核。
5. L1 检查“题面原文 + 答案”是否跨桶；L2 对需要读取世界状态的模板检查“题面句式 + 答案”是否跨桶。

任何一层跳过，都不能声称数据隔离通过。

## 4. 数据门禁

### 题库门禁

| 组 | 防什么 |
|---|---|
| D1–D4 | 句式太少、近义改写、全局塌缩或只换词不换表达结构 |
| D5 | 题面句式泄露隐藏档位，让模型不用读取工具结果就能猜答案 |
| D6 | 实体取值过少，模型记住 ID |
| D7 | 工具菜单恒定 |
| D8 | gold 工具轨迹单一 |
| D9–D10 | 行为或结局缺失 |
| D11 | 题面长度本身泄露路由 |
| L1–L2 | 同一底题或同形孪生题跨越 EVAL/SFT/RL |

阈值只在 [data_gates.py](../../syncopate/pipeline/data_gates.py) 定义。新增门禁必须说明它防哪一种具体失效，以及阈值从哪里来。

### SFT 出厂检查

SFT 数据还必须通过：

- 训练与 Runtime 的消息、历史、工具菜单、观测和 session 信令同形。
- `input_ids` 与 `loss_mask` 长度一致，监督段非空，终答完整。
- 不含占位教师文本、旧版本物料、OOV 工具或预设答案泄漏。
- prompt 和整条轨迹不超过统一预算；不允许静默截断。
- 三桶隔离复核再次通过。
- 画廊可供人工逐条查看。

相关实现集中在 [syncopate/pipeline](../../syncopate/pipeline)，其中多轮渲染的唯一出口是
[sft_replay.py](../../syncopate/pipeline/sft_replay.py)。

## 5. v15 协议与 v16 数据的关系

- v16 数据使用 v15 消息协议。
- v15 下训练和评测 prompt 使用全量工具菜单；case 的 `tool_menu` 仍供 verifier 和 routing 使用。
- `account_id` 等运行态身份不进入题面或模型输出，由 Runtime 按当前租户注入。
- 机器字段通过 `session.report`，给用户看的终答保持自然语言。
- 每个 assistant 轮的 thinking 形状由共享渲染路径生成，不能在独立脚本中手搓。

## 6. 复现与证据

- 数据版本、默认目录和切分算法：[split.py](../../syncopate/pipeline/split.py)
- v16 规格：[configs/buckets/v16.yaml](../../configs/buckets/v16.yaml)
- 本机确定性基线：[_audit/v16/local_gen_sha.json](../../_audit/v16/local_gen_sha.json)
- 云上正式数据与建库审计：`/vol/data/sft/v16`、`/vol/_audit/v16/`
- 历次建库失败、旧门禁和修复过程：
  [26 号归档](../archive/syncopate/pre-consolidation-v16/26-repair-rulers-and-data.md)

本机 dry-run 产物只证明构造路径能运行；带占位文本的 dry 数据永远不能成为训练或候选产物。
