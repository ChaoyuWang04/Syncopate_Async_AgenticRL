# 提交包⑤ · verl：LoRA 训练的 checkpoint 无条件全量落盘（E29）

> **状态：`DRAFT` —— 问题描述 + 修改范围已就位，等 Claude 做上游考据后成稿。**
> ⛔ 未经考据不许提交（见 [`../README.md`](../README.md) §4-①）。原文如下：
> 完整 PR 由 upstream 同事调研后成稿——
> 提交前必查上游 main 是否已有同类功能/在途 PR（我们钉死的是 **verl 0.8.0**）。
> 目标仓库：**verl-project/verl**（feature PR，非 bug report——功能缺失，不是行为错误）。

## 一句话

`FSDPCheckpointManager.save_checkpoint` 对 PEFT/LoRA 训练**无条件保存全量 `state_dict`**：
每 rank 8.5 GB × world_size，其中 **97% 是与基座逐字节相同的冻结权重**；`checkpoint_contents`
只能整类开关 model/optimizer/extra，**没有「model 只存可训练部分」这一档**。LoRA RL 下这意味着
每次存档搬运 ~27 GB（3 rank）纯冗余字节。我们的补丁实测：**单次 save 7.91→0.83 s（9.5×）、
ckpt 体积 12×**，断点续跑与下游链路无损。

## 问题定位（verl 0.8.0）

| 位置 | 现状 |
|---|---|
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py::save_checkpoint`（~L220） | `model_state_dict = self.model.state_dict()` 无条件全量；LoRA 意识仅限旁存 `lora_train_meta.json`（r/alpha 元数据，说明**上游已知这是 LoRA 训练**，但保存路径没有分叉） |
| 同文件 `::load_checkpoint`（~L139） | `self.model.load_state_dict(model_state_dict)` 全量严格匹配——只存 LoRA 的话 load 端也要动 |
| `megatron_checkpoint_manager.py` | ⚠️ 未核查，同事调研时看一眼是否同样形态 |

## 为什么上游应该要（普适性论证）

- 任何「LoRA + 定期 ckpt」的 RL/SFT 用户都在付这笔钱：字节 ~34×冗余、时间与字节量近线性
  （我们实测盘直写 3.9 GB/s 时 save 仍要 7.9 s——大头在序列化 + D2H，不是盘）。
- 高频 ckpt（如 staleness 研究、细粒度选点）在全量落盘下不可行，LoRA-only 下变成秒级。
- 上游自己存了 `lora_train_meta.json` ⇒ 检测条件（`peft_config` 存在）现成。

## 建议的修改范围（PR 形态）

1. **配置**：`checkpoint.save_contents` 增加一档（如 `model_trainable_only: bool` 或
   `model: "lora"`），默认关（行为不变，向后兼容）。
2. **save 端**：开启且 `unwrap_model.peft_config` 存在时，模型分片按可训练键过滤
   （我们用 `"lora_" in k`；上游或许想按 `requires_grad` 名单更普适）。**无 PEFT 时忽略该选项
   并告警**——绝不能把全参模型静默存成空字典。
3. **load 端**：检测 lora-only 分片（或读同目录 marker）⇒ 用「当前模型（基座已由初始化就位）
   state_dict 更新 lora 键后整载」的合成路径——**不依赖 FSDP sharded `load_state_dict` 的
   `strict=False` 语义**（我们没有验证过那条路，合成路径实测可靠）。
4. **守卫**（强烈建议一并进 PR）：load 完把内存中的 lora 键**逐位**比回 ckpt，不一致硬失败
   （加载错权重继续训比崩溃贵；成本一次 `state_dict()`，仅 resume 时发生）。
5. optimizer/extra 分片**不动**（optimizer 状态本来就只含可训练参数）。

## 我们的参考实现与验证（可整体搬给 PR）

| 物料 | 位置 |
|---|---|
| 补丁本体（save 过滤 + load 合成 + 逐位校验，~90 行） | `syncopate/train/verl_patches.py::_patch_lora_only_ckpt` 及 `filter_lora_state` / `merge_lora_into` / `assert_loaded_matches` |
| 单测 6 条（过滤/合成/结构变更硬失败/逐位校验/幂等） | `tests/train/test_ckpt_lora_only.py` |
| 实测报告（A/B 同尺子 + 续跑 + 下游链路） | `docs/infra_exp/E29-ckpt-lora-only.md` |
| 实跑日志 | `logs/e29_ckpt_{off,on,resume}.log` · `logs/e29v_*.log` |

**数字（Qwen3-4B + LoRA r32，fully_async 2 trainer + 1 rollout，仅差开关的同尺子 A/B）**：
单次 save **7.91 → 0.83 s**；模型分片 **8.5 GB → 252 MB**/rank（504/903 键）；ckpt 目录
**18 → 1.5 GB**；`--resume auto` 续跑：合成加载 + optimizer 接续 + **load 后 504 键逐位校验通过**；
PEFT adapter 提取链路无损。

## 注意事项（同事调研清单）

- [ ] 上游 main / 在途 PR 是否已有同类功能（关键词：lora checkpoint, save adapter, peft save）
- [ ] `megatron_checkpoint_manager` 是否同形态、要不要一并覆盖
- [ ] PEFT 键名带 `.default.`（adapter name）——过滤按子串没问题，文档里说清
- [ ] `should_save_hf_model`（huggingface/ 目录）路径与本改动正交，确认不受影响
- [ ] world_size 变化时的 resume 语义（全量路径同样有此限制，说明即可，不扩 scope）
