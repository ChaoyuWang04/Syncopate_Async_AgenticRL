# Infra 线交接（独立于主线训练）

> 🔴🔴 **2026-08-17 换机器了 —— §1 那三个招牌数字全部作废，必须在新机器重测。**
> 新机仍是 4×5090、P2P 仍全关，但 **2+2 跨 socket + PCIe Gen5** ⇒ **卡间带宽 6.44 → 25.6 GB/s（四卡）**。
> ⇒ 首当其冲要复查 **E02「FSDP 慢 6 倍」**（因果就是 6.4 GB/s，带宽 ×4 后可能只剩 ~2×）＝ 队列 **A6**；
> **A7（E00 四卡曲线）今天已做掉一半**，数据在 `logs/e00_allreduce_*.json`、尺子 `scripts/probe_allreduce_bw.py`。
> ⚠️ 旧机器已不存在 ⇒ **换机器救不回基线**，只能重测。细节见 `../syncopate/05-handoff.md` §0.1。
>
> 🆕 **同日修好三个会静默毁掉训练的 bug**（详见 05-handoff §0.1）：
> ① flash-attn 轮子**反向**坏（前向全过、反向 nan 或**恒为 0**）⇒ RL 完全空转；
>    已换官方 cu13 轮子 + CUDA13 运行时，判据 `scripts/check_flash_attn_backward.py`
> ② 分卡模式 3 个 trainer rank **全挤 GPU0**（根因是我们自己的 `worker_process_setup_hook`
>    在 Ray 设 CVD 之前 import verl，把 CUDA 设备枚举固化了）
> ③ `--weight-sync-bucket-mb` 在 colocate 下**被静默忽略**（⚠️ 默认值仍是 2048，短跑要显式传）
>
> 更新于 **2026-08-16**。给下一个上下文窗口。
> **分工**：主线训练（数据/SFT/RL/RAG/Runtime 里程碑）看 `../syncopate/05-handoff.md`；
> **本文档只管 infra 线**——多卡并行、异步 RL、通信、kernel、框架/模型选型。
> 按 Chaoyu 的约定：**短，只保证下一个窗口能接上**；细节全部指向对应文档。

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
| attention | `flash_attention_2` 默认 —— 🆕 **必须是官方 cu13torch2.9 轮子**（社区 cu128 那个**反向**是坏的，RL 会静默空转）。换轮子先跑 `scripts/check_flash_attn_backward.py` | 05-handoff §0.1 |
| dynamic_bsz | ⚠️ 代码里**默认 False**（这行以前写「默认 True」，与代码不符）。符号由 attention 决定，新机器**未重测** | README §6 |
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

1. **`--weight-sync-bucket-mb 2048` 会 OOM。**（⇒ 队列第 1 批 B2 就是治它的）有**两跑**死在这（`e12d_nolayered`、`e08b_onestepoff`）：
   rollout 卡 vLLM 24.65 GB + CE worker 4.71 GB，剩 1.99 要 2.00，**差 0.01 GB**。
   ⇒ **`gpu_util 0.75` 不是安全值**；解法不是压 gpu_util，是**调小 bucket**（实际只推 132 MB）。
2. **`--save-freq 999` 挡不住收尾那次保存。** 每个短跑结束都落 **27 GB** ckpt。
   曾差点把 61 GB 的 MoE 下载挤爆。⇒ **计时/探针类短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**
   （`dispatched.jsonl` 和 `rollout_dumps` 要留）。

## 5 · 队列（🆕 2026-08-16 重排）

★ **排序原则，两条，按顺序应用**：
1. **短探测优先，尤其是能防止长实验白做的**——一个 30 分钟的 go/no-go 挡住 2–3 天的工作量，
   它的期望收益比任何优化都高；
2. **然后看对两条 track 核心数字的贡献力度**——能把某个 before 变成 after 的排前面。

⇒ 欠的实验全表见 `TRACK-B §3.5`（B1–B12）与 `TRACK-A §7.5`（A1–A7）；
**完成态**的叙事与简历见 [`NARRATIVE-AND-RESUME.md`](NARRATIVE-AND-RESUME.md)。

### 第 0 轨 · 不吃 GPU，随时并行推进（别排队等卡）

```
🟢 B4写码  E08-c 仪器移到真下发点        0.5 天   差异化的核心（AReaL 明说没做的那格）
🟢 B6写码  动态分池 patch → setup_worker  —       验证要卡，但码可以先写
🟢 E03成文 NCCL 调优（数据已有）          —       只差成文
```

### 第 1 批 · 短探测 / gate（合计约 2 小时，**必须最先**）

```
🔴① B2  缓冲区 512 vs 2048 A/B      ~25 min  E12-e  ★ 解掉 OOM 脆弱点 —— 它挡着 B3，
                                                     而且 B1 的跑大概率会撞同一堵墙
🔴② A1  E07 P2 探针                 ~30 min  E07-P2 ★ go/no-go：verl 能否加载 30B MoE + LoRA
                                                     前向一步。**它 gate 住 A2 的 2–3 天**
🔴③ A7  E00 满载降频 + 4卡曲线      ~1 h     E00    ★ 分母的分母。README §6 自己写着
                                                     「会污染所有对照」——不先做，后面每一次
                                                     多卡对照都可能是白测的
🔴④ B3  补上 one_step_off 那一格    ~30 min  E08-b  「三模式同尺子对照」这句话现在还是假的
                                                     （B2 一完成就能跑）
```

### 第 2 批 · 核心数字（贡献力度最大）

```
🔴⑤ B1  E12 最后一刀 + 真的做优化   1–2 天   E12-c  唯一能把占步 18.8% 变成收益的路径；
                                                     探针已写好，就差跑
🔴⑥ B10 陈旧度节流的代价曲线        3–4 跑   E08-d  占空比成因②，量级至今未知；
                                                     ★ 顺带一次做掉 B7 的 η 换算
🔴⑦ B11 rollout 的配比与放置        2 跑     E08-e  占空比成因④。⚠️ **必须排在 B1 之后**——
                                                     B1 会大幅改变时间构成，配比的最优点
                                                     会跟着变，先测就白测
🔴⑧ B5  任务级尺子（EVAL 128×8）    1 跑     —      一次性验收 E13 + B1 + B10/B11；
                                                     自己定的纪律，至今一次没过
```

### 第 3 批 · Track A 的兑现

```
🐟⑨ A5  E01 一步的时间去哪了        —        E01    ★ 挂在上面任意一跑上，几乎零成本；
                                                     它是 B12 的门槛，也是 P-A3 空着的原因
🟠⑩ A4  E11-b 切片对照组落地        ~1 天    E11-b  ①在 RL 侧唯一的 after，改动小
                                                     （顺带答「省下的能否换更大 token 预算」）
🟠⑪ A6  E02 补 FULL_SHARD 稳态      ~1 h     E02    让「慢 6 倍」不再只有首步支撑
🟠⑫ B12 训练侧三次前向的必要性      —        E17🆕  占空比成因③（占步 72%，最大的一块）。
                                                     门槛 A5；⚠️ old_log_prob 不能降频（已论证），
                                                     只能从 ref 和「共享前向」两个方向进
🟠⑬ A2  E07 三摆法实测              2–3 天   E07    把②从「推算」变成「实测」（被 A1 gate）
```

### 第 4 批 · 长投入 / 收尾

```
⚪⑭ A3  E16 sm_120 能力探底         ~1 周    E16    Track A 唯一的「硬手艺」，最健壮，
                                                     不 gate 任何人也不被 gate ⇒ 放最后但不能砍
⚪⑮ B6验证 / B8 长尾画像 / B9 前缀缓存分片
```

⚠️ **`bitsandbytes` 未装**，4bit 相关探针（A2）要先装。
⚠️ **计时/探针类短跑，跑完就删 `checkpoints/grpo/<exp>/global_step_*`**（见 §4-2）。

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
