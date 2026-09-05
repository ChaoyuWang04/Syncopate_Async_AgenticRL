# Syncopate · 训练与评测

> 本文是 SFT、Exam、RL、OPD 和模型晋级规则的唯一现行说明。
> 当前施工顺序只看 [01-TASKS.md](01-TASKS.md)。

## 1. 当前状态

| 环节 | 状态 | 结论边界 |
|---|---|---|
| SFT | 本轮 v16 30-step smoke 健康通过 | adapter、选点和 310/310 层合并均已验；不是候选模型 |
| Exam | 运行链通过、质量 WARN | 40/40 记录齐；6 个判卷失败、1 个终答机器壳待修 |
| RL | 本轮双卡 2-step smoke 健康通过、质量 WARN | loss、梯度、权重同步、step-2 checkpoint 与 350 组 LoRA 导出通过；每步 25% rollout 撞响应上限 |
| OPD | 本轮 1-update smoke 健康通过、质量待解 | 84 个有效蒸馏 token、有限 KL、final 和 completion marker 通过；短评测工具错误多且多样性 5/8 |
| 全链 smoke | 机械链路通过 | `pipeline_ok=true`、`all_passed=false`；B02 是分段修复证据链，不是固定源码性能 baseline |
| candidate | 未开始 | 只能在全链 smoke 和候选门槛冻结后开始 |

## 2. 正式训练链

固定入口是 [scripts/v16_pipeline.sh](../../scripts/v16_pipeline.sh)。训练部分包含：

| 阶段 | 输入 | 输出 | 正式实现 |
|---|---|---|---|
| sft-train | `data/sft/v16/train.parquet` | SFT LoRA checkpoints | [sft.py](../../syncopate/train/sft.py) |
| sft-eval | 每个候选 adapter + 冻结 EVAL | entropy 与本地评测 JSON | [entropy.py](../../syncopate/train/entropy.py)、[eval_local.py](../../syncopate/train/eval_local.py) |
| sft-select | 候选评测结果 | `SELECTED` | [select_sft_ckpt.py](../../syncopate/train/select_sft_ckpt.py) |
| merge | 学生底座 + 选中 SFT adapter | 合并后的 SFT 模型 | [merge_adapter.py](../../syncopate/train/merge_adapter.py) |
| exam | 合并模型 | 考场记录、判卷、分诊 | [scripts/v16/exam_chain.sh](../../scripts/v16/exam_chain.sh)、[evaluation](../../syncopate/evaluation) |
| rl-train | 合并后的 SFT 模型 + RL 数据 | verl checkpoint | [launch_rl_v1.py](../../syncopate/train/launch_rl_v1.py)；动态分池实验才调用 [main_ppo_pool.py](../../syncopate/train/main_ppo_pool.py) |
| rl-adapter | 最新 RL actor checkpoint | PEFT LoRA adapter | [ckpt_to_adapter.py](../../syncopate/train/ckpt_to_adapter.py)、[lora_adapter_check.py](../../syncopate/train/lora_adapter_check.py) |
| rl-eval | 同一个合并底座 + RL adapter | 冻结评测 JSON | [eval_local.py](../../syncopate/train/eval_local.py) |
| opd-train | 本轮合并 SFT 底座 + RL adapter + 教师 | OPD adapter | [opd.py](../../syncopate/train/opd.py)、[opd_render.py](../../syncopate/train/opd_render.py) |
| opd-eval | 与 OPD 训练一致的底座 + OPD adapter | 冻结评测 JSON | [eval_local.py](../../syncopate/train/eval_local.py) |

正确的产物关系是：

```text
v16 SFT 数据
  → SFT adapter
  → 选点
  → 合并 SFT 模型
  → RL checkpoint
  → RL adapter
  → OPD adapter
```

`all` 每次生成独立 run id，训练产物和 `_audit/v16/runs/<run-id>/manifest.json` 同轮绑定。
后一阶段必须记录并核对自己读取的上游路径和模型身份。仅仅看到旧目录存在不算接线成功。

## 3. Smoke 与 Candidate

### Smoke

Smoke 用来回答：

- 代码能否启动并正常退出；
- loss、梯度、权重变化和关键数值是否有限；
- checkpoint、合并模型和 adapter 能否被下一阶段加载；
- 数据、模型和消息契约是否真的经过目标路径；
- 分布式、权重同步和蒸馏机制是否生效。

Smoke 不回答“模型质量是否足够”。短步数或小样本分数只能用于诊断。

固定入口默认 `smoke/observe`：质量或暂未冻结的读数缺口记 WARN 后可以继续收集后段证据；
模型/数据身份错误、越桶、NaN、程序异常、坏 checkpoint、OPD 0 次真实更新属于 FATAL，立即停止。
当前代码已强制 RL 读取本轮合并 SFT，OPD 读取本轮 RL adapter；B02 已在 B200 证明该产物关系真实成立。现在 T1 负责质量收口与固定源码 clean smoke。

### Candidate

Candidate 用来回答模型能否晋级。开始前必须冻结：

- 数据版本、六桶带宽与严格门禁；
- 模型、预算、采样、训练 profile 和随机种子；
- SFT 选点、Exam、RL、OPD 的停止和晋级规则；
- 冻结 EVAL、配对比较方式和证据目录；
- 失败后的回退路径。

训练进程跑完不等于 candidate 通过。只能按预注册判据给出“晋级、回退、无结论”之一。

## 4. 共同训练契约

### 同形

- SFT parquet 已经预分词，`input_ids` 和 `loss_mask` 来自与 RL 相同的 gold 回放路径。
- prompt、会话历史、全量工具菜单、工具观测、session 信令和终答必须与 Runtime 同形。
- 工具返回不参与语言模型 loss；只有模型自己产生的监督 token 参与。
- 超过统一长度预算时硬报错，不允许静默截断。
- v15 默认 think-on；每个 assistant 轮都使用共享模板形状。
- Qwen think-on 的 `<think>` 开标签由 generation prompt 写入，completion 常从思考正文开始再输出 `</think>`；共享解析器必须以 `implicit_think_open=True` 处理。没有闭标签时整段仍是未完成思考，不能当可见终答。Runtime、Exam、RL、OPD 和 eval 不得各写一份解析规则。

### 单一常量

- 模型与分词器：[model_paths.py](../../syncopate/core/model_paths.py)
- thinking、长度和采样：[rollout_budget.py](../../syncopate/train/rollout_budget.py)
- 数据版本与默认目录：[split.py](../../syncopate/pipeline/split.py)
- 消息和行为契约：[contract.py](../../syncopate/core/contract.py)
- RL profile：[launch_rl_v1.py](../../syncopate/train/launch_rl_v1.py)

任何实验覆盖这些默认值时，都必须说明目的、判据和审计路径；固定管线本身不维护第二份常量。

## 5. 各阶段必须证明什么

### SFT

- 数据没有零监督行或超预算行。
- 可训练参数只落在登记的 LoRA 目标上。
- loss 和 grad norm 有限，adapter 权重确实变化。
- 评测、选点与合并读取本轮相同输出目录。
- 合并模型相对底座确实包含增量。
- GPU 数必须显式；当前固定 runbook 仍是 1×B200，目标默认形态是 `DP=2`。用户已免除 1 卡与 2 卡 DP 的速度 A/B，但改默认前仍要用当前源码通过双卡梯度/权重一致性 smoke。B04 改为在同样 2×B200、相同有效 batch 和每次更新 token 下比较 `DP=2` 与 `TP=2`；当前 TP 尚未实现，不能把文档计划当成已接通能力。
- 稀疏词表投影只计算监督位置的 logits，已有朴素完整 logits 的 loss 对拍；它可能绕开标准 forward、并行 hook 或 fused CE，因此还要在 B10 重测梯度、显存和端到端收益。

当前真实冒烟证据：本轮 run `b02_20260905a` 使用 1222 行 v16 训练集、单卡 B200、30 步；
训练用时 329s，val loss `0.5670 → 0.3083`，`ΔW=0.5426%`，峰值 80.6GB。
修复后的 8×2 评测平均 reward 0.374、行为 16/16、多样性 7/8，但有 19 次工具错误；
这些只是短跑诊断。证据见 [B02 报告](../../_audit/infra/B02/REPORT.md)。

### Exam

- 数据库、Redis、模型端点、API/worker、考场执行、判卷和分诊全部完成。
- 每轮模型身份与 adapter 身份可追溯。
- Smoke 结果只诊断链路；candidate 才按冻结门槛判质量。

当前 B02：运行、模型身份和 40/40 记录均完整；6/40 判卷失败、1/40 终答含机器语法，
因此是 operational PASS、quality WARN。

### RL

- smoke 和 candidate 都从各自本轮合并 SFT 模型开始，不从裸学生底座开始。
- 默认使用 verl 官方均匀采样。动态分池只在显式 `--dynamic-pool` 实验臂启用；它根据组内 reward 方差降权当前无梯度信号的题，不按静态难度删除简单题，质量收益尚未在新栈证明。
- 多轮 agent loop、梯度、权重同步和保存路径都有判据行；动态分池臂另要求池判据行。
- RL checkpoint 转出的 adapter 必须能加载，并在相同底座上评测。
- 更新次数、数据量和训练完成条件是不同概念；停止与晋级按预注册规则。
- 响应上限必须单独量。B02 两个 step 都有 2/8 rollout 达到 12,288 token；原始轨迹显示未闭合或重复思考。这是质量 WARN，不是程序健康失败，但 candidate 必须阻止。

### OPD

- 全链学生必须使用本轮 RL adapter；缺 RL 产物就失败。SFT-only 只能另开诊断臂，不能作为 fallback。
- 教师和学生词表必须一致。
- `max_steps` 只数真实 optimizer update，`attempted/skipped/real` 分开记录；未达到登记的真实步数就失败且不写 `final`。
- 必须出现有效蒸馏 token、有限 KL 和零掩码正对照，不能靠大量跳步得到绿色退出码。
- OPD adapter 与评测底座必须和训练时一致。

当前 B02：2 次尝试中第 1 次没有有效自然语言 token，按规则跳过；第 2 次得到 84 个有效 token，
完成 1 次真实更新，chat KL/token 0.3345。机制通过；一更新评测不能证明 OPD 带来质量收益。

## 6. 健康、比较与晋级

运行中先看系统是否真实工作，再看 loss：

1. 入口、数据、模型、profile、rank 和判据行是否正确。
2. loss、grad norm、KL、reward、权重变化是否有限且非静止。
3. 采样、截断、零梯度、有效蒸馏步和工具调用构成是否健康。
4. checkpoint、adapter 和合并模型是否能被独立加载。
5. 冻结 EVAL 上做同起点、同预算、同采样的配对比较。
6. 用任务级“变好、变差、没动”计数决定晋级，不用单一 loss 或步数代替业务质量。

评测两端如果不是同一底座、同一 case 集合或同一采样契约，就不能做配对结论。

## 7. 旧数字与证据纪律

早期 RL 跑曾受到梯度同步和 rollout 权重同步问题污染。完整名单仍保存在
[21-invalidated-numbers.md](../archive/syncopate/pre-consolidation-v16/21-invalidated-numbers.md)，
旧过程保存在 [历史归档](../archive/syncopate/pre-consolidation-v16/README.md)。

规则只有三条：

- 旧数字不能因为代码后来修好而自动恢复有效。
- 新结论必须来自修复后的新运行，并带数据、模型、配置和产物身份。
- 硬件吞吐、训练质量和机制正确性是三类证据，不能互相替代。

## 8. 当前证据位置

- 数据和确定性：`_audit/v16/` 与 Modal `/vol/_audit/v16/`
- 当前完整 smoke 报告：`_audit/infra/B02/REPORT.md`
- 本轮逐 run 证据：`_audit/v16/runs/b02_20260905a/` 与 Modal 同路径
- SFT：`checkpoints/sft/v16_smoke_b02_20260905a`
- 合并模型：`models/Qwen3.6-35B-A3B-sft-v16_smoke_b02_20260905a`
- RL：`checkpoints/grpo/v16_smoke_b02_20260905a`；adapter 为 `models/adapters/rl_v16_smoke_b02_20260905a/lora_adapter`
- OPD：`checkpoints/opd/v16_smoke_b02_20260905a/final`
- 最新详细施工快照：[26](../archive/syncopate/pre-consolidation-v16/26-repair-rulers-and-data.md)、
  [31](../archive/syncopate/pre-consolidation-v16/31-modal-and-new-stack.md)

这些历史快照用于追证据，不用于决定下一步；下一步只看 TASKS。
