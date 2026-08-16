# Infra 线交接（独立于主线训练）

> 更新于 **2026-08-14 晚（收工）**。给下一个上下文窗口。
> **分工**：主线训练（数据/SFT/RL/RAG/Runtime 里程碑）看 `../syncopate/05-handoff.md`；
> **本文档只管 infra 线**——多卡并行、异步 RL、通信、kernel、框架/模型选型。
> 按 Chaoyu 的约定：**短，只保证下一个窗口能接上**；细节全部指向对应文档。

---

## 0 · 三十秒读懂

infra 线的目标：做出**两个有真实需求支撑、可验证的简历项目**。判据是一句话：

> **先有被测量出来的需求，才有优化目标。** 答不上「服务哪条 track 的哪条兑现物」的实验，一律停放。

实验以 **E 编号报告**组织（编号是身份、永不重排），track 是叠加的索引视图。

```
TRACK-A-hardware-kernel.md   负载形状 × 硬件拓扑 决定该写什么算子   ← ⚠️ 偏薄，明天的重心
TRACK-B-framework-async.md   agentic RL 训练系统的框架级改造        ← ✅ 已够撑一个项目
```

## 1 · ★★ 2026-08-14 的头号结论（三个数，互相印证）

```
① 用 4 张卡只换来 1.59× 加速     colocate 1卡 117.8 s/步 vs fully_async 3+1 74.1 s/步   [E08 §4.6]
② 整机占空比只有 31%             trainer 空闲 54–57%，rollout 空闲 82.5% @ 47.7 W       [E08 §4.5.1]
③ 权重同步 59.8 s 里 99.94%      不是传输（0.8 s）、不是编排（0.038 s）                 [E12 §4.5]
   在「处理 132 MB 的 LoRA」上
```

**①从时间维度量，②从空间维度量，③指出其中一大块在哪 —— 三者说的是同一件事。**
⇒ **在动任何算子之前，先把这 69% 的闲置搞清楚。**
对照量级：Track A 全套自写 kernel 的端到端收益是 **4.3%**（E11 实测后主动降级）。

## 2 · 已定决策（别再重新讨论）

| 决策 | 结论 | 详见 |
|---|---|---|
| 框架 | **verl 不换** | E07 §1 |
| 训练侧并行 | **DDP 必选**（`--fsdp-size 1`）。首步 FULL_SHARD×3 1182 s vs 单卡 198 s = **5.97×** | E02 |
| attention | `flash_attention_2` 默认（真轮子 2.8.3，`/workspace/wheels/`） | E02 §2 |
| dynamic_bsz | **默认 True**（FA2 下 ÷1.37；符号由 attention 决定） | README §6 |
| **MoE 模型** | 🆕 ~~GLM-4.7-Flash~~ → **`Qwen3-30B-A3B-Instruct-2507`**（已下载 57 GB）。GLM 的 `Glm4MoeLiteForCausalLM` **当前栈不支持**，要 transformers 5.0rc | **E07 §4.5.1** |
| **MoE 的 LoRA** | 🆕 **绝不能用 `all-linear`**（98.7% 的 Linear 在专家里 ⇒ 参数 26×、张量 74×、每步同步 3.39 GB）。用「注意力+router」30.1 M | **E07 §4.5.3** |
| E11 稀疏 logprob | 🔻 **降级，不写 kernel**（端到端仅 4.3%，切片就有 4.0%） | E11 §6-③ |

## 3 · 今天落地的改动（都在 `syncopate/train/`）

| 改动 | 效果 | 守护 |
|---|---|---|
| **`verl_patches.ddp_save_to_cpu` 加 `if param.requires_grad`** | `old_log_prob/ref` 比值 **1.941 → 1.069**，超额开销消掉 93%（≈8.5 s/步） | 3 条测试，含「全参微调时自动退回全量」 |
| `launch_rl` 新增 `--layered-summon` | 从写死改成显式参数（A/B 已证明它不是瓶颈） | 代码处写清了为什么可疑 |
| `launch_rl` 新增 `--target-modules` | dense 默认不变；**MoE 必须显式传** | 代码处写清了 26× 的账 |
| `verl_patches._patch_sync_step_timing` | 可选探针（`SYNCOPATE_SYNC_TIMING=1`），已含 `send_weights` 一层 | 判据行 + 保留 dispatch 元数据 |
| `launch_rl` 注释更正 | decoupled 的代价从「+6–10 s」订正为实测 25.7% | — |

## 4 · ⚠️ 两个已知的坑（明天会撞）

1. **`--weight-sync-bucket-mb 2048` 会 OOM。** 今天有**两跑**死在这（`e12d_nolayered`、`e08b_onestepoff`）：
   rollout 卡 vLLM 24.65 GB + CE worker 4.71 GB，剩 1.99 要 2.00，**差 0.01 GB**。
   ⇒ **`gpu_util 0.75` 不是安全值**；解法不是压 gpu_util，是**调小 bucket**（实际只推 132 MB）。
2. **`--save-freq 999` 挡不住收尾那次保存。** 每个短跑结束都落 **27 GB** ckpt。
   今天差点把 61 GB 的 MoE 下载挤爆。⇒ **计时/探针类短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**
   （`dispatched.jsonl` 和 `rollout_dumps` 要留）。

## 5 · 明天的队列（按简历贡献排序，已定）

```
🔴① bucket A/B（512 vs 2048 MB）   ~25 min  收尾 E12 + 解除 OOM 脆弱点 + 补上 one_step_off 那一格
🔴② E07 P2 探针                     ~30 min  verl 能否加载 Qwen3-30B-A3B + LoRA 前向一步
                                             ★ go/no-go，gate 住 Track A 后面几天的工作
🟠③ 长跑复核（~40 步）              ~60 min  让「整机 31%」和 E13 端到端有足够样本（现在 n 太小）
⚪④ E16 sm_120 能力探底              ~1 周    Track A 唯一的「硬手艺」产出，最健壮
```

**②排在③前面**：它 gate 的是天级工作量，③只是让已有结论更硬。
⚠️ **`bitsandbytes` 未装**，P6（4bit）要先 pip 装，sm_120 上有风险。

★ **明天的重心必须压到 Track A**——B 已经够了，A 只有一道兑现的菜（SFT 稀疏投影），
而它需要的东西（MoE、量化、kernel）成本最高、最该早开工。

## 6 · 新窗口阅读顺序

```
本文档 → TRACK-A / TRACK-B（看接哪条线）→ README.md（E 索引/模板/纪律）
→ E12（权重同步查因，方法论样板）→ E08（占空比与同机分母）→ E13（一行改动的全过程）
→ E07（MoE 决策 + 今天的三处更正）
→ ../syncopate/08-machine-and-environment.md（环境怎么跑起来）
→ ../ostinato-project-design-v0.2.md §4.0（被推翻的因果链，★ 必读）
```

⚠️ 关机重启后：`/workspace` 是网络盘会活着。记忆实体在 `.claude/memory/`（已进 git），
软链接没了就重建：
`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`。
Ray 不会自启，直接跑 launch_rl。
