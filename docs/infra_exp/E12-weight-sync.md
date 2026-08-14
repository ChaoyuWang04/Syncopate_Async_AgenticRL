# E12 · 权重同步的代价与根因

> 状态：🟡（E12-a 读码 + 日志分析完成；分步计时未做）　最后更新：2026-08-14

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | **B** · 「权重同步的根因与优化」 |
| **需求从哪来** | 主线实测：`update_weights` 恒定且与 attention / 打包 / 卡数**全无关**；LoRA 仅 132 MB，6.4 GB/s 上应为 0.02 s。M7 fully_async 实测稳态 **55.8 s** |
| **问题** | 这笔时间到底花在哪？是传输吗？ |
| **答案** | **不是传输。反解出固定开销 ≈ 55.0 s、传输 ≈ 0.8 s ⇒ 约 98.6% 与传输无关。** 根因不是单点，是每次同步要走完 **8 个步骤**，真正传数据的只有第 5 步 |
| **信心** | 高（37 次同步、标准差仅 1.79 s = 3.2%；首次 vs 稳态的数据量差 60× 而时间只差 1.85×，构成两点反解） |
| **推翻了什么** | ① 「权重同步 13.3 s」→ fully_async 下是 **55.8 s**；② `launch_rl` 注释「decoupled 代价 +6–10 s」→ 实测 **76.3 s**（§6） |
| **下一步** | 给 8 个步骤分别加计时（需一次独占 GPU 跑），确认是 `empty_cache` / KV cache 重建 / partial rollout 恢复 哪一个 |

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

### 4.4 ★ 意外收获：日志里有完整的步骤分解（37 步均值）

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
> **推翻后**：「值得」这个判断是基于 6–10 s 做的，**必须重新算这笔账**——
> 76.3 s 换来的是 `rollout_corr/*` 那套 ESS 刹车指标。刹车仍然要，但代价要如实标价。
> ⇒ 可考虑的折中：ESS 是否可以**降频采样**（比如每 N 步算一次 old_log_prob），
> 而不是每步都算。**做之前先设计对照。**
> **教训**：**注释里的估算数字会被后人当实测引用。** 估算就要标「估算」，
> 并在拿到实测后回填——这条已经在 E11 犯过一次（配置上限当实际值）。

## 7 · 踩的坑

| 症状 | 根因 | 修法 |
|---|---|---|
| 日志里 `timing_s/timing_s/param_sync` 键名重复 | verl 拼前缀时重复加了一层 | 解析时按 `timing_s/timing_s/param_sync` 匹配 |
| 一开始以为 fully_async 每 step 74 s 比 one_step_off 32.6 s 慢一倍多 | **基线不可比**：one_step_off 那个数是 `bypass_mode=True` 跑的（不含 old_log_prob 的 76.3 s） | **不下这个结论**；需要 E08-b 的同机分母 |

## 8 · 下一步 / 衍生问题

1. **E12-b · 给 8 个步骤分别加计时**（🔴 需一次独占 GPU 跑）——
   确认 55.0 s 的固定开销里，`empty_cache()` / KV cache 拆建 / partial rollout 恢复 各占多少。
2. **候选修法（等 ① 定位后再动，别猜着改）**：
   - `finalize()` 里的 `torch.cuda.empty_cache()` 是否必要？buffer 能否**常驻复用**而不是每次 alloc/free？
   - KV cache 是否必须整体释放？（权重变了旧 KV 确实失效，但**释放显存**和**作废缓存**是两件事）
   - 调大 `--sync-every`（现在 4）——**但这会加大 staleness**，是取舍不是优化。
3. **重新评估 decoupled 的性价比**（§6-②）：ESS 刹车能否降频。
4. **E08-b · colocate 同机基线**——现在有两个模式的数了，就差同机分母。
