# 提交包 · verl disaggregated 下 LoRA adapter 从不同步（E22）

> **状态：`READY` —— 材料齐备（源码树验证 ✅ 2026-08-19），等 Chaoyu 点头后提交。** 目标 **verl-project/verl**。
> 与包①（[`../OPEN-verl-fsdp-size-1/`](../OPEN-verl-fsdp-size-1/)）同批提交：同一天发现、同一形状 ——
> **配置意图正确，静默走进错误分支，所有指标正常**。

## 一句话

`fully_async` / `one_step_off`（disaggregated）+ LoRA（`merge=False` 默认）下，
每次权重同步推给 rollout 的都是**冻结基座**，adapter 一个字节不传
⇒ rollout 全程用起点策略 π₀ 采样，RL 回路静默断开。
上游注释自己写着 *"kept for **Phase 2 adapter path**"* —— 本包就是那个 Phase 2。

## 文件清单

| 文件 | 是什么 |
|---|---|
| [`submission-EN.md`](submission-EN.md) | **issue + PR 英文正文**（GitHub 直接粘贴）+ 注意事项 |
| [`verl-engine-workers-two-phase.patch`](verl-engine-workers-two-phase.patch) | trainer 侧：两段式协议（基于 main，避开 delta_sharded 分支） |
| [`verl-checkpoint-engine-worker-peft.patch`](verl-checkpoint-engine-worker-peft.patch) | rollout 侧：自描述 peek + `peft_config` 就地重建（带 `wire_format` 守卫） |
| [`test_disaggregated_lora_sync.py`](test_disaggregated_lora_sync.py) | PR 测试草稿（mock 型，无 GPU：两段式序列 + peek 注解 + 顺序转发） |
| [`analysis.md`](analysis.md) | 中文分析与证据链（根因五站 / 顶包表 / merge=True 为何不配当 workaround） |

## 修法要点（为什么是这个形状）

```
零 wire 改动     不新增序列化任何东西 ⇒ named_tensors 的后端（nccl/nixl/mooncake/kimi）全覆盖
自描述判别       adapter 推送 100% 是 lora_ 张量、基座推送 0 个 ⇒ peek 第一个名字即可，两侧零协调
照抄上游语义     base_sync_done 初始化 = "dummy" not in load_format（colocate 同款）
                 ⇒ 真实权重加载时第一次同步就只推 adapter，8.4 GB 基座一次都不用推
三类不受影响     全参 / merge=True / dummy 首推基座 —— 载荷里没有 lora_ ⇒ 走今天的原路
delta_sharded    peek 守在 wire_format=="named_tensors" 上 ⇒ delta 引擎自管的路不碰
```

## 证据（仓库内产物）

```
logs/e22_verl_fix_20260819.log                源码树验证跑（monkeypatch 关闭，diff 是唯一机制）
docs/infra_exp/E22-lora-never-synced.md       完整实验记录（含 §6.5 三层数值验证）
scripts/probe_weight_sync_payload.py          离线分支复现（两分支对照表）
monkeypatch 等价实现                           syncopate/train/verl_patches.py::_patch_lora_adapter_sync
                                              （60 步真实训练 + v_numeric 独立复现全绿）
```

## 提交前最后一眼

- [x] 源码树验证 ✅：5 步 / 3 次同步全 adapter（252 MiB×3，基座 0 次）· kl 6.4e-05/2.9e-04 贴地板
      · exit 0 · 三 rank 504/504 逐位相同 · mock 测试 3/3 过 —— `logs/e22_verl_fix_20260819.log`
- [ ] 查 verl CONTRIBUTING（DCO / pre-commit / 测试目录惯例）
- [ ] tag CODEOWNERS 里的 LoRA 维护者（HollowMan6，#7436 刚加）
