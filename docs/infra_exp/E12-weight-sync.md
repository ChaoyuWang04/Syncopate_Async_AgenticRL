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
| **需求从哪来** | 主线实测：`update_weights` 恒定且与 attention / 打包 / 卡数**全无关**；LoRA 仅 132 MB，6.4 GB/s 上应为 0.02 s。M7 fully_async 实测稳态 **55.8 s** |
| **问题** | 这笔时间到底花在哪？是传输吗？ |
| **答案** | **既不是传输，也不是编排。** 分步实测：编排 8 步里的 5 步合计 **0.038 s（0.06%）**，**99.94% 在第 5 步「传输 + 两侧 update_weights」**；而两点反解说真正的数据传输只有 **~0.8 s（1.4%）** ⇒ **约 59 秒花在「处理/装载 132 MB LoRA」上** |
| **信心** | 高（两条独立证据：37 次同步的两点反解 + 独占跑的分步计时；两者互相印证） |
| **推翻了什么** | ①「权重同步 13.3 s」→ fully_async 下 55.8 s；② 注释「decoupled +6–10 s」→ 实测 25.7%；③ **我自己的「KV cache 拆建是大头」**→ 实测 0.06%；④ **我自己的「build_process_group 占 82%」**→ 第二次同步就掉到 0.023 s（§6） |
| **下一步** | 拆第 5 步：trainer 侧（取参数+send）vs rollout 侧（recv+装进 vLLM）。**探针已加好**，待下一跑 |

---

## 1 · 问题与预测

**问题**：`update_weights` / `param_sync` 的时间花在哪？

**★ 预测（写死于 `TRACK-B` §0.5，跑之前）**：

- **P-B2**：大头**不在传输、不在计算**，而在**分配/同步/串行握手**
  （bucket 2048MB×双缓冲每次 alloc/free，或遍历全参而非 132MB LoRA）。
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

## 4.5 ★★ E12-b 分步计时（2026-08-14，独占，`SYNCOPATE_SYNC_TIMING=1`）

台架：`verl_patches._patch_sync_step_timing`（可选补丁，挂 `setup_worker`——
`CheckpointEngineManager` 活在 **FullyAsyncTrainer 这个 Ray actor** 里，driver 侧打无效）。
判据行 `[verl-patch] 权重同步分步计时已启用` 已在日志确认。
运行：`logs/e12b_synctiming.log`，配置与 M7 完全一致，`--steps 12`。

### 第一次同步（param_version 0，推基座 + LoRA）

| 步骤 | 耗时 | 占比 |
|---|---|---|
| 1 `abort_replicas` | **0.008 s** | 0.0% |
| 3 `release_kv_cache_replicas` | **0.003 s** | 0.0% |
| 4 `build_process_group` | **45.967 s** | **42.6%** |
| 7 `resume_kv_cache_replicas` | **0.002 s** | 0.0% |
| 8 `resume_generation_replicas` | **0.002 s** | 0.0% |
| — 未计量（2 建 WorkerGroup + **5 传输** + 6 finalize） | 62.05 s | 57.4% |
| **param_sync 总计** | **108.034 s** | 100% |

⇒ **⛔ 推翻了 §4.2 之后写下的主假设**（见 §6-③）：
「KV cache 拆了重建 + partial rollout 中断恢复是大头」——
**这四步加起来只有 0.015 秒**，完全不是。

### ★★★ 第二次同步（param_version 1，稳态：只推 LoRA 132 MB）——**结案**

| 步骤 | 耗时 | 占比 |
|---|---|---|
| 1 `abort_replicas` | 0.008 s | 0.01% |
| 3 `release_kv_cache_replicas` | 0.004 s | 0.01% |
| 4 `build_process_group` | **0.023 s** | 0.04% |
| 7 `resume_kv_cache_replicas` | 0.002 s | 0.00% |
| 8 `resume_generation_replicas` | 0.001 s | 0.00% |
| **已计量小计** | **0.038 s** | **0.06%** |
| **未计量 = 第 5 步（传输 + 两侧 update_weights）** | **59.76 s** | **99.94%** |
| **param_sync 总计** | **59.797 s** | 100% |

★ **`build_process_group` 从 45.967 s 掉到 0.023 s（2000×）** ——
它是**一次性**开销（首次建 NCCL 组），`rebuild_group=False` 完全按设计工作。
⇒ 我在第一次同步后写下的「它占稳态 82%」的推测**当场被第二次数据推翻**（见 §6-④）。

### 4.5.1 首次 `build_process_group` 的 50 s 是什么（`logs/e12c_step5.log`）

再拆一层（探针打在 `NCCLCheckpointEngine` 上）：

```
engine.prepare(分配 2×2048MB buffer)      0.133 s   ← 0.3%
engine.init_process_group(rank=0)        50.083 s
engine.init_process_group(rank=1)        50.095 s
build_process_group 合计                 50.239 s
```

⇒ **首次开销 99.7% 是 `init_process_group`，即 NCCL 通信组的建立**，
和 buffer 分配无关（0.133 s，与空卡微基准 2.6 ms 同量级，说明"在被 vLLM 占满的卡上分配 4 GB 很贵"
这个猜测也**不成立**）。
⇒ 且它是**一次性**的（第二次起 0.023 s）⇒ **不在优化的关键路径上**，如实记录、不再追。

### 4.5.2 ★ 第 5 步拆成 trainer 侧 / rollout 侧（`logs/e12c_step5b.log`）

探针打在 `ActorRolloutRefWorker.update_weights`（trainer 侧）与
`CheckpointEngineWorker.update_weights`（rollout 侧）。
⚠️ 两者**并发执行**（`ray.get(trainer.update_weights(...) + rollout.update_weights(...))`），
**不能相加**。

**首次同步（param_version 0，推基座 8 GB + LoRA）**：

| 步骤 | 耗时 |
|---|---|
| `build_process_group`（其中 `init_process_group` 53.87） | 54.087 s |
| **trainer 侧：取参数 + send** | **67.086 s** |
| **rollout 侧：recv + 装进 vLLM** | **70.978 s** |
| 其余编排（abort / KV 拆建 / 恢复）合计 | 0.014 s |
| **param_sync 总计** | **125.103 s** |

⇒ **rollout 侧（71.0）≈ trainer 侧（67.1）+ 4 s**，而两者并发
⇒ **rollout 大部分时间在等着收，瓶颈在 trainer 侧。**
⇒ 首次推 8 GB 用 67 s ⇒ 有效带宽 ≈ **120 MB/s**，与 §4.1 两点反解出的 168 MB/s 同量级
（口径不同：那里是端到端反解，这里是单侧墙钟）。

### 4.5.3 ★★★ 稳态：trainer 侧的成本**与传输量完全无关**

```
首次同步（推 8 GB 基座 + LoRA）   trainer侧 取参数+send  67.086 s
第二次同步（只推 132 MB LoRA）    trainer侧 取参数+send  69.887 s
```

**数据量差 60×，耗时反而略涨。** ⇒ **成本不在 send，在"取参数"那一步。**

### 4.5.4 根因：`layered_summon` —— 为分片 FSDP 设计的路径，被用在了不分片的机器上

`ActorRolloutRefWorker.update_weights` → `get_per_tensor_param` → `collect_lora_params`。
我们**已经开着** `layered_summon=True`（`launch_rl.py`，跟 LoRA 配置一起写死、无注释），
所以走的是 `fsdp_utils.layered_summon_lora_params`：

```python
for prefix in prefix_list:
    for name, submodule in __prefix_submodules(fsdp_module, prefix):   # ← 36 层
        if fsdp_version(submodule) > 0:
            with FSDP.summon_full_params(submodule, writeback=False):
                sub = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                ...拷 LoRA 到 CPU...
            get_torch_device().empty_cache()        # ← ★ 每层都调一次
```

**每步付 36 × (summon + 该层全量 `state_dict()` 物化 + `empty_cache()`)。**

★ **它是为「真正分片的 FSDP」设计的**——分片时一次性 gather 整个模型会爆显存，
所以宁可逐层。**但我们跑的是 `--fsdp-size 1`（DDP，不分片）**：
参数本来就在每张卡上完整存着，整体 summon 近乎免费，
而逐层路径把一个免费操作拆成 36 份收费操作。

⇒ **这是今天第三次遇到同一个形状**（另两次：E13 存整个模型只为取 3% 的 LoRA；
E11 对全部 token 算 logprob 只为用 4%）：
**「机制是对的，但它假设的运行条件和我们的不一样。」**

### 4.5.5 ⛔ A/B 结果：`layered_summon` **不是**原因，预测被推翻

`--layered-summon False`（`logs/e12d_nolayered.log`，只变这一个变量，
判据已核：日志里 `'layered_summon': False` ×2，基线是 True）：

| | 基线 True | 对照 False |
|---|---|---|
| 首次同步 trainer 侧 | 67.086 s | 58.775 s |
| **稳态 trainer 侧** | 55.76 / 66.24 / 58.40（均值 **60.13**） | **61.443** |

⇒ **对照值正好落在基线的波动区间内，无改善。**

> **原猜想**：`layered_summon` 逐层 summon（36×(summon + state_dict + empty_cache)）
> 是那 60 s 的主因；关掉它 trainer 侧应降到 **< 10 s**。
> **实测**：60.13 → 61.443，**没动**。
> **推翻后**：这反而是一次**高价值的排除**——
> `layered_summon=False` 走的是 `summon_full_params(整个模型)` 这条**完全不同**的取参数路径，
> **两条路径都是 ~60 s** ⇒ **成本不在「取参数」，在两条路径共有的部分。**
> 结合"与数据量无关"（8 GB 与 132 MB 同耗时），**只剩 `send_weights`**。
> **教训**：**A/B 的负结果能一次砍掉一整条分支**——
> 比"再读一遍代码找可疑处"有效得多，因为读码只能生成假设，不能排除假设。

### 4.5.6 ⚠️ 顺带炸出一个真实脆弱点：`gpu_util 0.75` 也是贴着墙的

`layered_summon=False` 那一跑在**第 4 次同步 OOM**（数据已够，A/B 结论不受影响）：

```
rollout 卡（31.37 GB）
  vLLM 进程                    24.65 GB
  CheckpointEngineWorker        4.71 GB   （自身 ~2.71 + 已分配的一个 2 GB bucket）
  ────────────────────────────────────
  已用                         29.36 GB
  剩余                          1.99 GB
  要再分配                      2.00 GB   ← 差 0.01 GB
```

⇒ **和文档里记的 `gpu_util 0.85` 那次 OOM 是同一个形状**（当时"差 0.13 GB"），
只是这次发生在 **0.75** 上 ⇒ **0.75 也不是安全边际，前几跑只是运气好擦过去了。**

**根因**：`prepare()` 在**每个 worker** 上分配 **send_buf + recv_buf 各 2048 MB = 4 GB**，
其中一份就住在被 vLLM 占了 24.65 GB 的 rollout 卡上。

⛔ **要更正的旧结论**：`gpu_util 0.75` 被记成"安全值"（记忆 + 分布式文档 §7.3）。
**正确的说法是：0.75 只是勉强够，实测会在第 4 次同步 OOM。
真正的解法不是压 `gpu_util`，是调小 bucket** ——
那 4 GB 里绝大部分根本用不上（实际只推 132 MB）。

★ **由此，下一个 A/B 一箭双雕**：调小 `--weight-sync-bucket-mb`
① 若耗时按比例下降 ⇒ 坐实"广播整个定长 buffer"的猜想；
② 无论如何都解除这个 OOM 脆弱点（2048→512 MB 可给 rollout 卡腾出 3 GB）。

⬜ **下一层探针已加**：单独给 `NCCLCheckpointEngine.send_weights` 计时，
把 60 s 切成「取参数」与「发」两半。⚠️ 新的可疑点：
`prepare()` 分配的是 **2048 MB 定长 buffer**，而 `collective.broadcast(self.bucket, ...)`
广播的是**整个 buffer**，与实际装了多少无关 —— 这与"耗时和数据量无关"的观测吻合。
若成立，`--weight-sync-bucket-mb` 调小应按比例降耗时，是下一个一行开关的 A/B。

### ⇒ E12 的答案（终）

```
稳态权重同步 59.8 s
  ├─ 编排（abort / KV cache 拆建 / 建组 / 恢复）  0.038 s   ← 0.06%，全部免费
  └─ 第 5 步：传输 + 两侧 update_weights        59.76 s   ← 99.94%
       └─ 其中真正的数据传输（两点反解）           ~0.8 s   ← 1.4%
```

**⇒ 约 59 秒花在「处理/装载 132 MB 的 LoRA」上，既不是编排，也不是传输。**
下一层（trainer 侧取参数+send vs rollout 侧 recv+装进 vLLM）的探针已加好，待下一跑。

## 4.6 🐟 2026-08-17 补：主线 v13-e1 跑里白捡的观测 —— **稳态 55.8 s 不是普适常数**

> 来源：主线 2026-08-17 16:20 那一跑（fully_async 3+1，v13 数据），**零成本旁观，未占用 GPU**。
> ⚠️⚠️ **口径污染，必须写在最前面**：那一跑被 `nsys profile --delay 900 --duration 180` 包着，
> 而且主线在 16:52 把它**弃掉重跑了**（`checkpoints/grpo/m7b_v13e1_nsys_aborted`）。
> 采样窗口（16:35:46–16:38:46）晚于本节用到的 6 条 timing 行，profiler 开销**应该**很小 ——
> 但「应该」不是判据。⇒ **同样的数要在 16:52 起的干净跑上复测**，复测前本节只作方向性证据。
> 尺子：`scripts/parse_fully_async_timing.py`（新建，已把「timing 行覆盖 4 个 global step」固化进工具）
> 数据：`logs/e12d_v13e1_timing.json`

这一跑显式传了 `update_weights_bucket_megabytes=512`（本报告的 55.8 s 是 **2048** 下量的）：

```
param_sync   13.34 → 10.32 → 9.22 → 8.39 → 8.12 → 8.47 → 7.96 s     稳态中位 8.43 s
对照本报告   稳态 55.8 s（bucket 2048）                               ⇒ 差 6.6×
```

⇒ **这与 §8-2 的候选修法「buffer 能否常驻复用而不是每次 alloc/free」指向同一件事**：
`bucket ÷4 ⇒ param_sync ÷6.6`，**耗时随 bucket 尺寸走，而不是随实际传输量（恒定 132 MB）走**
—— 正是「广播/处理的是整个定长 buffer」这个猜想预测的形状。

⚠️⚠️ **这不是同尺子 A/B，不许当结论用。** 同时变了至少五个变量：
数据版本（v11→v13）、底座（sft-v11-e1→v13-e1）、`dynamic_bsz`（True→False）、
`gpu_memory_utilization`（0.75）、`rollout.n=8`。
⇒ **它只把 B2 从「25 分钟去验证一个猜想」改成「25 分钟去补一个干净分母」**，B2 不能取消。

★ 同一批 timing 行还顺手更新了占空比的构成（每行覆盖 4 个 global step，已折算）：

| 项 | 本次实测（占步） | 本报告 §4.4 记录（M7） |
|---|---|---|
| update_actor | **54.0%** | 33.1% |
| old_log_prob | 16.7% | 25.7% |
| ref | 14.2% | 13.2% |
| **⇒ 三次前向合计** | **84.9%** | **72.0%** |
| gen | 9.0% | 6.3% |
| **param_sync** | **6.5%** | **18.8%** |
| step（每 global step） | 32.7 s | 74.1 s |

⇒ ★★ **两个方向相反的位移，都直接改队列的排序**：
① 三次前向 72% → **84.9%**（B12 / E17 更该打）；
② 权重同步 18.8% → **6.5%**（B1 的可回收空间缩水到原来的 1/3）。

⚠️ 同样受上面那五个变量污染，**方向可信、数值待同尺子复核**（B2 补分母、B3 补三模式）。

★ **教训（本报告要补的一条）**：§6 已经写了「同一个名字的指标在不同模式下不是同一件事」，
这次是它的**下一层** —— **同一个模式、同一个指标，换一个默认参数（bucket）就差 6.6 倍。**
⇒ 全局常量表里的每个数，除了记「哪个模式」，还要记**「当时那几个旋钮拧在哪」**。

## 4.7 ⛔ B2 实测：**bucket 不是原因** —— §4.6 的推论被推翻（2026-08-17 19:37）

同尺子 A/B（fully_async 3+1，12 步，**只改 bucket 这一个变量**，其余全锁死）：

| bucket | param_sync 各次 | **稳态中位** | 每 global step | 三次前向占步 |
|---|---|---|---|---|
| **512** | 12.77 / 10.21 / 8.63 / 9.04 | **9.04 s** | 32.57 s | 83.1% |
| **2048** | 13.70 / 10.85 / 8.79 / 9.73 | **9.73 s** | 32.88 s | 82.9% |

⇒ **差 7.7%，不是 6.6×。**

> **原猜想**（§4.6，由主线那一跑推出来的）：`bucket ÷4 ⇒ param_sync ÷6.6`，
> 即「广播/处理的是整个定长 buffer」。
> **实测**：只改 bucket，两档几乎一样。
> **推翻后**：**bucket 不解释 55.8 → 9 s 这个 6 倍。** §4.6 那段推论作废，
> 但**它推出来的那个「现象」仍然成立**：param_sync 今天就是 ~9 s，而本报告记录的是 55.8 s。
> **教训**：**同一次观测里的两件事要分开** ——
> 「param_sync 现在只有 9 秒」是**观测**（可信）；
> 「因为 bucket 小了」是**归因**（当时就标了「不是同尺子 A/B」，现在被证伪）。
> ⚠️ 而我**用这个归因改过队列排序**（B1 降级）。**排序的方向仍然对**
> （9 s 就是 9 s，B1 的可回收空间确实只剩 1/3），**但理由错了** ——
> ⇒ 纪律：**带着「未验证归因」去改优先级是可以的，但要在队列里标明它是待验的**，
> 否则验证结果回不来的时候，没人知道该回去改哪一条。

### 4.7.1 ⇒ 新问题（比原来那个更值钱）：**那 6 倍到底是谁干的**

已锁死不是：bucket（本节）、`layered_summon`（§4.5.5 的 A/B 两档都 ~60 s）。
仍在候选里的变量（M7 那次 vs 今天）：

```
数据版本      v11 → v13（序列长度分布不同 ⇒ 同步前要排空的在飞请求量不同）
底座          sft-v11-e1 → sft-v13-e1
dynamic_bsz   True → False
E13 的修复     ✅ **已查（零 GPU，git log -S）**：`if param.requires_grad` 落在
              **commit 743a6e0 · 08-16 12:50**，而 M7 跑在 **08-14 05:25–09:09**
              ⇒ **M7 那次跑的时候这个修复还不存在**，是一条**确凿的差异**。
              ⚠️ 但它**未必够解释 6 倍**：E13 自报每步省 4.34→0.083 s，
              4 步也就 ~17 s，而缺口是 ~46 s ⇒ **它像是其中一块，不像是全部**。
              而且它patch 的是 `fsdp2_sharded_save_to_cpu`（proximal anchor 快照），
              与 param_sync 的 `get_per_tensor_param` **是两条路径** —— 除非其中一趟
              搬运正好落在同步窗口里。**这一步只能靠实验分**
verl 版本      两次都记的 0.8.0（未复核）
```

★ **为什么值得查**：一个**没人刻意优化过的组件，悄悄快了 6 倍**。
要么是真收益（那要知道是谁给的，别哪天改回去）、要么 E12 的招牌数字 55.8 s 有问题
（那 TRACK-B 的兑现物之一要重写）。**两种情况都必须知道。**
⇒ 已排 **B15**：先查 E13 落地时间与 M7 的先后（**零 GPU，git log 就能定**），
再按需做一次 v11 + dynamic_bsz=True 的复现跑。

## 5 · 结论

1. **权重同步的时间 98.6% 不是传输**，而是同步前后那七个步骤的固定开销。
   ⇒ P-B2 **方向对了**（不在传输不在计算），但**具体机制猜错了一半**：
   不是 NCCL 建组、不是遍历全参，而是 **buffer 分配/释放 + `empty_cache()` + KV cache 拆建 + partial rollout 的中断/恢复**。
2. **瓶颈整体在训练侧的 logprob 计算**：三次前向 `update_actor` + `old_log_prob` + `ref`
   = 213.6 s = **72%**；而 rollout 生成只占 **6.3%** —— 异步确实把生成藏起来了，这是它该有的样子。
3. ⇒ **E11（稀疏 logprob）的价值被抬高**：那三项**全部**在算 logprob，且**全部**对整条打包序列算
   （88.3% 是 prompt）。E11 一次命中三项。
   ⚠️ **但别把 E11 的「24×」读成端到端 24×**：省的只是 lm_head 那一层（约占参数量 10%），
   36 层前向对 prompt 仍要算。粗估端到端收益 ≈ 10% × 72% ≈ **7%**，需 E01 确认。

## 5.1 ★★ 一个必须写死的区分：这笔钱**不是**「免传输策略」治的那笔

本报告的结论极易被误读成「免传输策略没用」。**两笔钱完全无关**：

| | 权重同步（本报告） | 训练侧分片通信（Track A 的免传输策略治的） |
|---|---|---|
| 是什么 | trainer → rollout 推新权重 | FSDP 每层 all-gather 冻结基座 |
| 多久一次 | 每 `sync_every`=4 步一次 | **每个 micro-batch 一次** |
| 多大 | LoRA **132 MB** | 约 **90 GB** |
| 实测结论 | **98.6% 不是传输**（本报告） | **传输就是要命的那一项**：3 卡 FULL_SHARD 1182 s/步 vs 单卡 198 s ⇒ 多给卡慢 6 倍；换 DDP 立刻 3.00× 线性 |

⇒ **E07 的 4bit 量化复制（配置 C）不受本报告影响**，它治的是右边那列。

⇒ **但本报告确实枪毙了一个想法**：「把 LoRA 权重再压缩让同步更快」——
传输只占 0.8 s，压到 0 也只省 1.4%。**优化方向要从「传得更快」改成「让停工—复工的仪式更便宜」。**
这两个方向的工作内容完全不同：不查就动手，很可能去调 bucket 大小 / 换传输后端 / 压权重
——**全都在动那 0.8 秒。**

## 6 · ⛔ 推翻了什么

> **原猜想 ①**：权重同步 **13.3 s**（占步 27–36%），来自 one_step_off。
> **实测**：fully_async 稳态 **55.8 s**。**⇒ 这个数不能跨模式引用。**
> **推翻后**：全局常量表要按模式分别记。
> **教训**：**「同一个名字的指标，在不同模式下不是同一件事。」** `update_weights`（one_step_off）
> 和 `param_sync`（fully_async）走的是不同代码路径、包含的步骤也不同。

> **原猜想 ②**（`launch_rl.py` 的注释）：decoupled 模式「每步多一次 actor 前向算 old_log_prob
> （**约 +6–10 s**）。**值得。**」
> **实测**：`old_log_prob` = **76.3 s，占 25.7%**。**低估 8–13 倍。**
> **推翻后**：「值得」这个判断是基于 6–10 s 做的，代价要如实标价：
> 76.3 s × 37 步 = **47 分钟，占全程 224 分钟的 21%**。
>
> ⛔ **但「换来的是 ESS 刹车」这个说法本身也是错的，我一度也这么写过（已更正）。**
> 读码查实（`experimental/separation/ray_trainer.py:503-530` + `config/algorithm.py:123-131`）：
> ```
> bypass  (2 policies)  old_log_prob := rollout_log_prob，走 compute_policy_loss_bypass_mode()
> decoupled(3 policies) 重算 old_log_prob 作为**proximal anchor π_old**，走标准 PPO loss + IS 权重修正
>                       源码注释：「π_old computed once per data batch,
>                                  serves as stable reference during mini-batch updates」
> ```
> ⇒ **`old_log_prob` 不是指标，是损失函数的一部分。** 它就是 AReaL 的 π_prox ——
> 「用 π_prox 而不是 π_behav 算 IS 比率，防止当前策略被陈旧的低质策略拽偏」。
> AReaL 的消融：decoupled 下 η≤4 安全，**朴素 PPO 超过 η=1 就崩**。
> ⇒ **这 76.3 s 买的是「异步在高陈旧度下还能训」这件事本身，ESS 指标只是副产品。**
>
> ⇒ ⛔ **因此「ESS 降频采样（每 N 步算一次 old_log_prob）」不可行**——
> 那不是「少测几次」，那是**在两个不同的目标函数之间来回横跳**
> （decoupled PPO ↔ bypass PPO）。**别做。**
>
> ⇒ ✅ **真正可查的优化方向**：`old_log_prob`（76.3 s）约是 `ref`（39.2 s）的 **1.95×**，
> 而两者**都是同一批数据上的纯前向**。差这一倍是从哪来的？
> （old_log_prob 顺带算 entropy，但 `use_fused_kernels=True` 下 entropy 应该几乎免费。）
> **若两者本该相当，则每步有约 37 s 是没有解释的** → 挂到 E01 一起查。
> ⇒ 另外 **E11（稀疏 logprob）直接命中它**：old_log_prob 是纯 logprob 前向，正是 E11 优化的形状。
> **教训**：**注释里的估算数字会被后人当实测引用。** 估算就要标「估算」，
> 并在拿到实测后回填——这条已经在 E11 犯过一次（配置上限当实际值）。

> **原猜想 ③（读码之后、分步计时之前）**：55 s 的固定开销大头是
> **KV cache 拆了重建 + partial rollout 的中断/恢复 + `empty_cache()`**。
> **实测**：这四步加起来 **0.015–0.038 s**，占比 **0.06%**。`empty_cache` 微基准 12.2 ms。
> **推翻后**：编排**全部免费**，钱在第 5 步的**权重处理**上。
> **教训**：**「读码列出来的步骤多」不等于「那些步骤贵」。** 我从调用链上数出 8 步，
> 就默认成本分散在这 8 步里——实际是 7 步几乎为 0、1 步占 99.94%。
> ⇒ **列出候选之后要逐个称重，不能按"看起来重"排序。**

> **原猜想 ④（第一次同步数据出来之后，当场写下的）**：
> `build_process_group` 46 s ⇒「若每次都这样，它占稳态同步的 ~82%」。
> **实测**：第二次同步 **0.023 s**，掉了 2000 倍。
> **推翻后**：它是**一次性**建组开销，`rebuild_group=False` 按设计工作。
> **教训**：**n=1 的推断连一次都撑不过。** 这条猜想从写下到被推翻只隔了约 6 分钟——
> 便宜是因为它写在报告里而不是写进了代码。**先记录假设、再等第二个数据点，成本极低。**

## 7 · 踩的坑

| 症状 | 根因 | 修法 |
|---|---|---|
| 日志里 `timing_s/timing_s/param_sync` 键名重复 | verl 拼前缀时重复加了一层 | 解析时按 `timing_s/timing_s/param_sync` 匹配 |
| 一开始以为 fully_async 每 step 74 s 比 one_step_off 32.6 s 慢一倍多 | **基线不可比**：one_step_off 那个数是 `bypass_mode=True` 跑的（不含 old_log_prob 的 76.3 s） | **不下这个结论**；需要 E08-b 的同机分母 |
| ⛔ **给第 5 步加探针，整跑崩了**：`AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'` | `update_weights` 上有 verl 的 `@register(...)`，它把 `{dispatch_mode, execute_mode, blocking}` 挂在函数的 **`MAGIC_ATTR = "attrs_3141562937"`** 上（`decorator.py:440`），**`RayWorkerGroup` 靠扫描这个属性自动生成组级方法**。我用裸的 `async def` 换掉方法，把元数据丢了 ⇒ 组级方法直接不存在 | 包装时 `functools.wraps` **并把 `MAGIC_ATTR` 原样复制过去**；拿不到该属性就**跳过探针**而不是硬装。★ 教训：**包装带装饰器的方法，必须把装饰器挂的属性一起带过去**——"看起来只是加一层计时"是最容易忽略这点的场景 |

## 8 · 下一步 / 衍生问题

0. 🆕 **B2 的目标已改写（2026-08-17，见 §4.6）**：bucket 512 下主线实测 param_sync 稳态
   **8.43 s**，而本报告的 55.8 s 是 bucket 2048 下量的。⇒ B2 不再是「验证一个猜想」，
   而是**在同尺子下补一个干净分母**（其余变量全锁死，只改 bucket）。
   ⚠️ 在 B2 出结果之前，**别再引用「权重同步占步 18.8%」** —— 它绑死在 bucket 2048 上。
1. **E12-b · 给 8 个步骤分别加计时**（🔴 需一次独占 GPU 跑）——
   确认 55.0 s 的固定开销里，`empty_cache()` / KV cache 拆建 / partial rollout 恢复 各占多少。
2. **候选修法（等 ① 定位后再动，别猜着改）**：
   - `finalize()` 里的 `torch.cuda.empty_cache()` 是否必要？buffer 能否**常驻复用**而不是每次 alloc/free？
   - KV cache 是否必须整体释放？（权重变了旧 KV 确实失效，但**释放显存**和**作废缓存**是两件事）
   - 调大 `--sync-every`（现在 4）——**但这会加大 staleness**，是取舍不是优化。
3. **重新评估 decoupled 的性价比**（§6-②）：ESS 刹车能否降频。
4. **E08-b · colocate 同机基线**——现在有两个模式的数了，就差同机分母。
