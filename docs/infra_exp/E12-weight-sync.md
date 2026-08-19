# E12 · 权重同步的代价与根因

> ⛔⛔ **2026-08-18 口径更正（[`E22`](E22-lora-never-synced.md)）**：本报告整份建立在
> 「稳态只推 **132 MB** LoRA」这个前提上 —— 而那是**算出来的**（66M×2B），**从没量过**。
> 实测：每次同步推的是 **399 个张量 / 8,414.1 MiB 的完整基座**，LoRA 一个字节都没推。
> 🆕 **2026-08-18 晚续**：修法① 落地后（E22 §6.4）按**真实载荷**重测 ——
> `param_sync` 稳态 **0.974 s**（占一步 **0.8%**），首次推基座 13.3 s（一次性）。
> ⇒ **本报告的核心谜题（"99.9% 不是传输"）随前提一起消失**：它是在"误以为只推 132 MB、
> 实际推 8.4 GB"之下算出来的。**按真实载荷，权重同步就是 1 秒的事。**
> ⇒ **计时数据保留，分析要重写**：§253 记的那条反常「与数据量无关（8 GB 与 132 MB 同耗时）」
> **正是这个 bug 在敲门** —— 因为一直都是 8 GB。⇒ 见 `00-INFRA-HANDOFF §5` 的 R4。

> 状态：🟡（E12-a 读码 + 日志分析完成；分步计时未做）　最后更新：2026-08-14

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | **B** · 「权重同步的根因与优化」 |
| **需求从哪来** | 主线实测：`update_weights` 恒定且与 attention / 打包 / 卡数**全无关** |
| **问题** | 这笔时间到底花在哪？是传输吗？ |
| **答案** | ⛔⛔ **本报告原来的答案（"既不是传输也不是编排，99.94% 花在处理 132 MB LoRA 上"）已作废** —— 它整份建立在「稳态只推 132 MB」这个**从没量过**的前提上。**实测每次推的是 8,414 MiB 的冻结基座**（[E22 §3.2](E22-lora-never-synced.md)）。<br>✅ **现在的答案**：修法① 之后 `param_sync` 稳态 **0.974 s**、占一步 **0.8%**；而调大 `sync_every` 省下的钱**根本不在权重同步里** —— **96% 在 `gen`（trainer 等样本）上**（[E08 §5.5](E08-async-rl.md)） |
| **本报告还剩什么** | ✅ §1–§4.4 的**原始计时数据**（它们量的是真实发生的时间，仍然有效）<br>⛔ §4.5–§4.7 的全部**分析**已移至 [`../syncopate/21-invalidated-numbers.md §5.3`](../syncopate/21-invalidated-numbers.md) |
| **下一步** | 无 —— **这条线已结案且靶子已转移**：`param_sync` 只占 0.8%，优化它最多值 0.8%；钱在「同步不打断 rollout」（handoff §5.1 第 5 项）
---

## 1 · 问题与预测

**问题**：`update_weights` / `param_sync` 的时间花在哪？

**★ 预测（写死于 `TRACK-B` §0.5，跑之前）**：

- **P-B2**：大头**不在传输、不在计算**，而在**分配/同步/串行握手**
  （bucket 2048MB×双缓冲每次 alloc/free，或遍历全参）。⛔ 原文此处写的「而非 132MB LoRA」前提错误 —— 实际推的是 8.4 GB 冻结基座。
- **P-B3**：查因后可降到 < 3 s（36% → <10%）。

## 2 · 环境指纹

```
日期        2026-08-14（M7 全程，05:25–09:09，224 分钟）
运行        fully_async · 3 trainer + 1 rollout · Qwen3-4B-sft-v11-e1 + LoRA r32
配置        --sync-every 4 · bypass_mode=False(decoupled) · partial_rollout=True
            --dynamic-bsz True --max-token-len-per-gpu 16384 · gpu_util 0.75
            checkpoint_engine backend=nccl · update_weights_bucket_megabytes=2048
框架        verl 0.8.0 / torch 2.9.0+cu128 / vllm 0.12.0 / flash_attn 2.8.3
原始日志    logs/m7_v11e1_fullyasync.log（147 global steps / 38 次 param_sync / 37 条 step 计时）
被读代码    verl/checkpoint_engine/base.py（update_weights 八步 / build_process_group）
            verl/checkpoint_engine/nccl_checkpoint_engine.py（prepare / finalize / init_process_group）
            verl/workers/engine/fsdp/transformer_impl.py:794（get_per_tensor_param）
```

⚠️ 本节全部来自**已完成运行的日志 + 读码**，未额外占用 GPU。

## 3 · 方法

1. 从日志抽出全部 38 次 `param_sync` 耗时，按 `param_version` 分首次/稳态；
2. **两点反解**：首次推「基座 + LoRA」，稳态只推 LoRA（`base_sync_done` 分支），
   数据量相差约 60×，用两个时间点解出「固定开销 + 传输」两项；
3. 读 `update_weights` 的调用链，列出每次同步真正执行的步骤；
4. 逐条排除嫌疑（`rebuild_group` / sleep-wake / 是否遍历全参）。

## 4 · 数据

### 4.1 param_sync 耗时

| | 值 |
|---|---|
| 首次（v0，推基座 ~8 GB + LoRA） | **103.3 s** |
| 第二次（v1） | 63.6 s |
| **稳态（v2–v37，只推 LoRA 132 MB）** | **中位 56.0 · 均值 55.8 · min 51.2 · max 59.7 · 标准差 1.79（3.2%）** |
| 稳态合计（36 次） | 33.4 分钟 |

**两点反解**：
```
103.3 = 固定 + 8.13 GB / BW
 55.8 = 固定 + 0.132 GB / BW
⇒ ΔT = 47.5 s 对应 Δ数据 8.0 GB ⇒ 有效带宽 ≈ 168 MB/s
⇒ 传输分量 ≈ 0.132 GB / 168 MB/s ≈ 0.79 s
⇒ ★ 固定开销 ≈ 55.0 s，占稳态 param_sync 的 98.6%
```

⇒ **数据量降 60×，时间只降 1.85×** —— 时间不在传输上，铁证。
标准差仅 3.2% 也印证：这是**固定成本操作**，与传输内容量几乎无关。

### 4.2 一次同步真正做的 8 件事（`checkpoint_engine/base.py` update_weights）

```
1  abort_replicas()              打断并保存所有未完成请求（partial rollout）
2  建临时 RayWorkerGroup
3  release_kv_cache_replicas()   释放 vLLM 的 KV cache
4  build_process_group(rollout)  → prepare(): 每个 worker 分配 2×2048 MB buffer
5  trainer.update_weights + rollout.update_weights   ← ★ 只有这一步在传数据
6  finalize() on 每个 worker     释放 buffer + torch.cuda.empty_cache()
7  resume_kv_cache_replicas()    重建 KV cache
8  resume_generation_replicas()  恢复未完成请求
```

### 4.3 已排除的嫌疑

| 嫌疑 | 结论 |
|---|---|
| 每次重建 NCCL 通信组 | ❌ **排除**：`rebuild_group` 默认 **False**（`nccl_checkpoint_engine.py:119`），组不销毁 |
| sleep/wake 搬 7.6 GB | ❌ **排除**：`free_cache_engine=False` 生效，全程日志只出现 3 次 sleep（非每次同步） |
| 遍历全部 4B 参数而非 LoRA | ❌ **排除**：`get_per_tensor_param(base_sync_done=...)` → `collect_lora_params(...)`，首次之后只收 LoRA |

### 4.4 ★ 意外收获：日志里有完整的步骤分解

> ⚠️⚠️ **口径（2026-08-14 补正，之前漏了）**：每条 timing 行覆盖**恰好 4 个 global step**
> —— `training/global_step` 序列为 3, 7, 11, …, 147，**相邻差全为 4**，共 37 行 = 148 步。
> ⇒ **下表是「每 4 个 global step 的和」，不是「每步」。百分比不受影响**（同窗口内的比值），
> **但绝对秒数要除以 4 才是每步**。`param_sync` 例外：它每 4 步只发生一次，
> 所以 55.8 s 就是**一次同步的真实时长**，摊到每步是 13.9 s。
> ⇒ 每 global step：update_actor 24.5 · old_log_prob 19.1 · param_sync(摊) 13.9 ·
> ref 9.8 · gen 4.7 · adv 2.4 · **合计 74.1 s**。

| 项 | 中位 (s) | 均值 (s) | 占 step |
|---|---|---|---|
| **update_actor** | 98.4 | **98.1** | **33.1%** |
| **old_log_prob** | 77.0 | **76.3** | **25.7%** |
| **param_sync** | 56.0 | **55.8** | **18.8%** |
| ref | 39.4 | 39.2 | 13.2% |
| gen | 17.9 | 18.6 | 6.3% |
| adv | 9.5 | 9.6 | 3.2% |
| **step 合计** | 298.7 | **296.4** | 100% |

★ **分项合计 297.7 s vs step 296.4 s —— 账几乎完全对上，没有黑洞。**
（每条 step 行覆盖 4 个 global step + 1 次同步，`--sync-every 4`。）

## §4.5–§4.7 —— **已作废，原文移至** [`../syncopate/21-invalidated-numbers.md §5`](../syncopate/21-invalidated-numbers.md)

> **为什么作废**：整段建立在「稳态只推 132 MB LoRA」这个**从没量过**的前提上——实测每次推的是 **8,414 MiB 冻结基座**。修法① 之后 param_sync 稳态 **0.974 s**，而 sync_every 省下的钱**根本不在权重同步里**（gen 占降幅 96%）
> **被谁推翻**：E22 §6.4.3 / E08 §5.5

