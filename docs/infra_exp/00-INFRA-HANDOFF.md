# Infra 线交接（独立于主线训练）

> 更新于 **2026-08-17**。给下一个上下文窗口。
> **分工**：主线训练（数据/SFT/RL/RAG/Runtime 里程碑）看 `../syncopate/05-handoff.md`；
> **本文档只管 infra 线**——多卡并行、异步 RL、通信、kernel、框架/模型选型。
> 按 Chaoyu 的约定：**短，只保证下一个窗口能接上**；细节全部指向对应文档。
>
> ★ **焦点是怎么定下来的 → [`../focus-migration-2026-08.md`](../focus-migration-2026-08.md)**
> —— 唯一记录迁移历史的地方。其余文档只写当前焦点，不做新旧对照。

---

## 0 · 三十秒读懂

infra 线的目标：做出**两个有真实需求支撑、可验证的简历项目**。判据是一句话：

> **先有被测量出来的需求，才有优化目标。** 答不上「服务哪条 track 的哪条兑现物」的实验，一律停放。

实验以 **E 编号报告**组织（编号是身份、永不重排），track 是叠加的索引视图。

```
TRACK-A-hardware-kernel.md   负载形状 × 硬件拓扑 决定该写什么算子   ← ~30%，四条兑现物只落地一条
TRACK-B-framework-async.md   agentic RL 训练系统的框架级改造        ← ~50%，诊断够了但没有 after
NARRATIVE-AND-RESUME.md      🆕 对外怎么讲：完成态的故事线 + 简历（未测的留〔 〕）
```

⚠️ **2026-08-16 的重估**：B 那句「已经够撑一个项目」要加限定词——**够撑的是「诊断」，不是「成果」**。
手上的数几乎全是 before。⇒ **实验优先级从此只看「能不能把某个 before 变成 after」**（见 §5）。

## 1 · ★★ 头号诊断：闲置在哪

```
① 整机占空比约 31%          trainer 空闲 54–57%，rollout 空闲 82.5%      [E08]
② 权重同步里 ~99.9% 不是传输 传输 0.8 s、编排 0.038 s，其余是固定开销      [E12]
③ 训练侧三次前向占步 72%     update_actor + old_log_prob + ref 同一批数据 [E08 分解]
```

**①从空间维度量，②③指出大块在哪。** ⇒ **动任何算子之前，先把这 69% 的闲置搞清楚。**
对照量级：Track A 全套自写 kernel 的端到端收益是 **4.3%**（E11 实测后主动降级）。

⚠️ **绝对秒数与加速比正在重测**（队列 A6/A7）—— 上面三条的**构成比**是结论，
绝对值以重测为准。★ 占空比的**四个成因与「谁在打」**见 `TRACK-B §3.5.1`。

### 1.1 · 机器画像

```
4×RTX 5090 / sm_120，P2P 全关（NCCL 走 SHM/direct/direct）
2 socket EPYC 9V74，**2+2**：GPU0/1@node0、GPU2/3@node1；PCIe Gen5 x16
all-reduce busbw @256MB   组内 28.8 · 跨 socket 22.2 · 四卡 25.6 GB/s
  ⇒ 跨 socket 掉 22%，NUMA 绑定无效（UPI 跳的物理代价）
  ⇒ 换算 DDP 梯度 260MB ≈ 10.2 ms/步，跨 socket 净代价 1.2 ms/步 ＝ 一步的 0.004%
尺子 scripts/probe_allreduce_bw.py · 数据 logs/e00_allreduce_*.json
```

★ **2+2 拓扑让「放置」第一次有了意义**：当前 trainer=GPU0/1/2 跨了 socket
⇒ DDP 的 all-reduce 每步走 UPI。候选摆法（已并入 **B11**）：
`trainer 0,1 + rollout 2,3`（两侧都不跨）/ `trainer 2,3 + rollout 0,1`（对称性检验）。

## 2 · 已定决策（别再重新讨论）

| 决策 | 结论 | 详见 |
|---|---|---|
| 框架 | **verl 不换** | E07 §1 |
| 训练侧并行 | **DDP 必选**（`--fsdp-size 1`）：LoRA 每步只同步 260 MB。⚠️ 三档稳态对照待 **A6** | E02 |
| attention | `flash_attention_2` 默认 —— 🆕 **必须是官方 cu13torch2.9 轮子**（社区 cu128 那个**反向**是坏的，RL 会静默空转）。换轮子先跑 `scripts/check_flash_attn_backward.py` | 05-handoff §0.1 |
| dynamic_bsz | 代码默认 **False**。⚠️ **符号由 attention 决定**，当前机器 + FA2 下**未测** | README §6 |
| **MoE 模型** | 🆕 ~~GLM-4.7-Flash~~ → **`Qwen3-30B-A3B-Instruct-2507`**（已下载 57 GB）。GLM 的 `Glm4MoeLiteForCausalLM` **当前栈不支持**，要 transformers 5.0rc | **E07 §4.5.1** |
| **MoE 的 LoRA** | 🆕 **绝不能用 `all-linear`**（98.7% 的 Linear 在专家里 ⇒ 参数 26×、张量 74×、每步同步 3.39 GB）。用「注意力+router」30.1 M | **E07 §4.5.3** |
| E11 稀疏 logprob | 🔻 **降级，不写 kernel**（端到端仅 4.3%，切片就有 4.0%） | E11 §6-③ |

## 3 · 已落地的改动（2026-08-14，都在 `syncopate/train/`）

| 改动 | 效果 | 守护 |
|---|---|---|
| **`verl_patches.ddp_save_to_cpu` 加 `if param.requires_grad`** | `old_log_prob/ref` 比值 **1.941 → 1.069**，超额开销消掉 93%（≈8.5 s/步） | 3 条测试，含「全参微调时自动退回全量」 |
| `launch_rl` 新增 `--layered-summon` | 从写死改成显式参数（A/B 已证明它不是瓶颈） | 代码处写清了为什么可疑 |
| `launch_rl` 新增 `--target-modules` | dense 默认不变；**MoE 必须显式传** | 代码处写清了 26× 的账 |
| `verl_patches._patch_sync_step_timing` | 可选探针（`SYNCOPATE_SYNC_TIMING=1`），已含 `send_weights` 一层 | 判据行 + 保留 dispatch 元数据 |
| `launch_rl` 注释更正 | decoupled 的代价从「+6–10 s」订正为实测 25.7% | — |

## 4 · ⚠️ 两个已知的坑（下次开跑必撞）

1. **`--weight-sync-bucket-mb` 默认 2048 会 OOM**（⇒ 队列 B2 就是治它的）。
   CheckpointEngine 会在**目标卡上**分配一个 bucket 大小的暂存区（`nccl_checkpoint_engine.py:142`），
   而实际只推 132 MB。⇒ **`gpu_util` 不是安全阀**，解法是**调小 bucket**（短跑传 512）。
   ⚠️ 它对**所有模式**生效，colocate 同样会撞。
2. **`--save-freq 999` 挡不住收尾那次保存。** 每个短跑结束都落 **27 GB** ckpt。
   曾差点把 61 GB 的 MoE 下载挤爆。⇒ **计时/探针类短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**
   （`dispatched.jsonl` 和 `rollout_dumps` 要留）。

## 5 · 队列（🆕 2026-08-17 重排 —— 依据见 `../focus-migration-2026-08.md` §6）

★ **排序原则，两条，按顺序应用**：
1. **短探测优先，尤其是能防止长实验白做的**——一个 30 分钟的 go/no-go 挡住 2–3 天的工作量，
   它的期望收益比任何优化都高；
2. **然后看对两条 track 核心数字的贡献力度**——能把某个 before 变成 after 的排前面。

⇒ 欠的实验全表见 `TRACK-B §3.5`（B1–B12）与 `TRACK-A §7.5`（A1–A7）；
**完成态**的叙事与简历见 [`NARRATIVE-AND-RESUME.md`](NARRATIVE-AND-RESUME.md)。

### 第 0 轨 · 不吃 GPU，随时并行推进（别排队等卡）

```
🟢 B4写码  E08-c 仪器移到真下发点        0.5 天   差异化的核心（AReaL 明说没做的那格）
🟢 B6写码  动态分池 patch → setup_worker  —       ⚠️ 钩子已改成延迟 import，改动要沿用那个模式
🟢 E03成文 NCCL 调优                     —       ⚠️ 旋钮结论要按当前口径复述
```

### 第 1 批 · 短探测 / gate ✅ **2026-08-17 全部完成**（结果见 `README §7`）

```
✅ A6  E02 三档稳态      DDP 7.97s / ZeRO-2 3.42× / ZeRO-3 6.01×（同 3 卡、只改一个变量）
✅ A7  满载降频          4 卡单卡算力仅 −2.0% ⇒ 「污染所有对照」被证伪
✅ A1  MoE go/no-go      GO：4bit 15.6GB、LoRA 30.1M、前反向通过、梯度全有限
⛔ E04 rollout TP=2      实测净负（生成慢 20%）⇒ 再次停放，这次理由是实测
```

### 第 1.5 批 · 第 1 批产出的后续（🆕 按重要度排）

```
✅ A8  集合通信带宽的**分算子口径**        E00   **已完成 2026-08-17**，结果见 README §7.1
        ★★ 头号发现：**all_gather 在 3 卡上塌了** —— 2卡 51.0 / 4卡 37.9 / **3卡 3.2 GB/s**（差 12×），
        而 all_reduce / reduce_scatter / broadcast 在 3 卡上都正常。NCCL 在非 2 幂次 rank 上的
        算法退化（无 P2P 绕主机内存时尤其明显），**已被两个独立探针复现**。
        ⇒ **E00 的「卡间带宽」从此必须分算子 + 分卡数给**，一个数会错 12 倍。
        ⇒ DDP 不受影响（它走 all_reduce）；**踩坑的只有 FSDP/ZeRO 这类靠 all_gather 的**。
✅ A11 ZeRO-3 在 4 卡上重测              E02   **已完成**：3卡 6.02× → **4卡 1.54×**
        ⇒ 「分片慢 6 倍」大部分是「3 个 rank」的锅 ⇒ **筛子②的说法已改写**（README §7.2）
✅ A12 3 rank all_gather 退化的机理        E00   **已完成**：不是硬件限制，是**协议选择**。
        `NCCL_PROTO=LL128` 把 3 卡 all_gather 拉回 6.9×，实跑 ZeRO-3 **47.94→14.40 s（3.33×）**，
        比值 6.02×→**1.81×**。⚠️ 代价 all_reduce −30%/broadcast −41% ⇒ **不能全局开**，
        已写进 `launch_rl`：`fsdp_size>1` 时自动带上并打判据行。（README §7.3）
        ⚠️ 「验 5/6/7 rank」做不到：NCCL 禁止同卡多 rank（实测 invalid usage），
        rank 数被物理卡数卡死在 2/3/4，三个都已测。
🔴 A9🆕 4bit MoE 的加载路径                ~1 h     E07   ★ **A2 的前置**：bnb 逐层量化
        造成严重碎片（权重 13.32 GB，却 17.43 GB reserved 未分配 ⇒ OOM）。
        本次靠 `expandable_segments:True` 解，**但它在真训练路径上用不了**
        （与 vLLM colocate 内存池冲突）⇒ 必须换办法：分层加载 / CPU 侧先量化 / 预量化存盘
🟠 A13🆕 「为什么偏偏是 3 rank」最后一层    ~2 h   E18   §9.5：已排除缓冲区尺寸与环构造；
        剩下的假设是「all_gather 按 rank 数等分缓冲区，3 非 2 幂次 ⇒ 分块边界与 SHM 暂存区
        对齐不匹配」。要坐实需读 NCCL 源码的 chunk 划分逻辑，或用 ncu 看访存模式。
        ⚠️ 在它之前，E18 的结论只到「协议选型失误」，不到「因为对齐」。
🟠 B13🆕 ZeRO-3 惩罚的字节账核对            ~30 min  E02   把「100 GB/步 ÷ 3.2 GB/s ≈ 31 s」
        与实测 40 s 的 9 s 缺口坐实（A8 完成后做，顺带回答 ZeRO-2 的 3.42× 是否同源）
```

### 第 2 批 · 核心数字（贡献力度最大）

★ **B12 提到本批之首**：它是占空比里最大的一块（占步 72%）、**纯计算、不受互联换代影响**
⇒ 在所有「贡献力度」候选里，它是唯一一个价值只会随硬件变好而**上升**的。

```
🔴⑤ B12 训练侧三次前向的必要性      —        E17    占空比成因③，占步 **72%**，最大的一块。
                                                     门槛 A5；⚠️ old_log_prob 不能降频（已论证），
                                                     只能从 ref 和「共享前向」两个方向进
🐟⑥ A5  E01 一步的时间去哪了        —        E01    ★ B12 的门槛，挂在任意一跑上、几乎零成本
                                                     ⇒ 实际要先于 B12 完成
🔴⑦ B1  E12 最后一刀 + 真的做优化   1–2 天   E12-c  占步 18.8%。★ 带宽换代后传输占比更小
                                                     ⇒ 「不在传输上」这个结论**更硬**，目标未变
🔴⑧ B11 rollout 的配比与放置        2 跑     E08-e  🆕 **升级成「拓扑感知的放置」**：2+2 下
                                                     trainer 跨不跨 socket 第一次有了区别。
                                                     ⚠️ 必须在 B1 之后（时间构成会变）
🔴⑨ B10 陈旧度节流的代价曲线        3–4 跑   E08-d  占空比成因②，量级至今未知；顺带做掉 B7 的 η
🔴⑩ B5  任务级尺子（EVAL 128×8）    1 跑     —      一次性验收 E13+B1+B10/B11；自定纪律，至今没过
```

### 第 3 批 · Track A 的兑现

```
🟠⑪ A4  E11-b 切片对照组落地        ~1 天    E11-b  ①在 RL 侧唯一的 after，改动小、不受换代影响
🟠⑫ A2  E07 三摆法实测              2–3 天   E07    ★ 论证已改写成「数据量比值 470×，与带宽无关」
                                                     ⇒ 这条**不因换代而贬值**，只是被 A1 gate
🔴⑬ B2  缓冲区 512 vs 2048 A/B      ~25 min  E12-e  ⚠️ 它原本的「解 OOM」目的已达成（接线已修）
                                                     ⇒ 只剩验证「广播整个定长 buffer」这个猜想
🟠⑭ B3  三模式同尺子对照            ~30 min  E08-b  ⚠️ 三种模式现在都能跑了，只差**同尺子**跑一遍
```

### 第 4 批 · 长投入 / 收尾

```
🔴⑮ A3  E16 sm_120 能力探底         ~1 周    E16    ★ **价值相对上升**：Track A 唯一的硬手艺，
                                                     且**完全不依赖任何被换代影响的因果链**
⚪⑯ B6  动态分池接进 fully_async 的验证   —        —     写码见第 0 轨；⚠️ 顺序钉死在 B4 之后
⚪⑰ B8  rollout 长尾画像                  —        E10   异步收益理论上界（1.37–2.75×）从哪来
⚪⑱ B9  前缀缓存分片                      —        E09   最可能提上游 PR，不做也不影响这条线成立
⚪⑲ B7  一致性尺子 ESS/TIS + η 换算       —        E15   ⚠️ 已并进 B10 顺带做，此处仅留映射
⚪⑳ A10🆕 E14 执行层消泡                  —        E14   ★ **Track A 兑现物④，此前队列里根本没有它**
                                                          （nsys/ncu 定位 decode 空转，门槛 A5）
```

★ **相对上一版的四处变动**（依据见 `../focus-migration-2026-08.md` §6）：
① **A6 从第 3 批提到第 1 批之首** —— 它决定两条 track 共用的一句话还能不能说；
② **B12 从第 3 批提到第 2 批之首** —— 最大的一块，且不受互联换代影响；
③ **🆕 E04 探针进第 1 批** —— 停放理由过期，30 分钟买断一个新方向；
④ **B2/B3 降到第 3 批** —— 它们原本的「解 OOM / 补格子」目的已被本次修复达成。

⚠️ **`bitsandbytes` 未装**，4bit 相关探针（A2）要先装。
⚠️ **计时/探针类短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**（见 §4-2）。

★ **覆盖核对（2026-08-17）**：E00–E17 共 18 个编号，队列覆盖 16 个；
未进队列的两个是**显式停放**，理由在 `README §2.1`：**E05**（序列并行 —— 问题已被稀疏投影解决）、
**E06**（装更大模型 —— 已并入 E07，不单独复活）。⇒ **没有"因为优先级低而被悄悄漏掉"的实验。**

★ **完成度**：Track B ~50%（诊断 85 / 优化 15 / 验收 0）、Track A ~30%（论证 80 / 兑现 25 / 硬手艺 0）。
★ **队列编号（B*/A*）是执行顺序，E 编号是报告身份**——上面第三列就是映射，
新报告按 E 编号归档，**别用 B/A 编号建文件**（README §1 的规矩）。

## 6 · 新窗口阅读顺序

```
本文档 → TRACK-A / TRACK-B（看接哪条线）→ NARRATIVE-AND-RESUME.md（这些实验最后要变成什么）
→ README.md（E 索引/模板/纪律）
→ E12（权重同步查因，方法论样板）→ E08（占空比与同机分母）→ E13（一行改动的全过程）
→ E07（MoE 决策 + 三处更正）
→ ../syncopate/08-machine-and-environment.md（环境怎么跑起来）
→ ../ostinato-project-design-v0.2.md §4.0（被推翻的因果链，★ 必读）
```

⚠️ 关机重启后：`/workspace` 是网络盘会活着。记忆实体在 `.claude/memory/`（已进 git），
软链接没了就重建：
`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`。
Ray 不会自启，直接跑 launch_rl。
