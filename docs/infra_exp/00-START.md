# Infra · 00 · START — **文档在哪 · 现在在哪 · 下一步做什么**

> **新窗口的第一份文档，也是唯一一份「导航」。** 只做三件事：
> **① 指出每份文档放什么 ② 说清现在到哪一步 ③ 指向下一步。**
> ❌ 不记历史、不记决策过程 —— 那些在 [`02-DECISIONS.md`](02-DECISIONS.md)。
>
> ⚠️ **2026-08-19 重组**（与主线 `syncopate/00-START` 同一套约定）：
>
> ```
> 00-INFRA-HANDOFF.md  →  拆进 00（导航/守则）+ 01（队列）+ 02（决策/作废）
> ONBOARDING.md        →  并入本文（§1 读法 · §5 守则 · §6 跑前清单）
> TRACK-A / TRACK-B    →  合并成 TRACKS.md（只留兑现物与状态，叙事归 E 报告）
> E12-weight-sync.md   →  docs/archive/（它的核心谜题随前提一起消失，E22 §6.4）
> ```
>
> ## ★★★ 文档更新秩序（Chaoyu 2026-08-19 立，**两线通用**）
> ## 前科：主线 00 曾 916 行、20 曾 1269 行、19 份共 8334 行；本线 meta 层曾 2600 行
> ## —— 同一天两道压缩令。膨胀的文档不会有人来整理，只会被推倒重写。
>
> **默认动作是「就地改写」，不是「往后追加」。** 增量不是禁止的，但必须证明是必须的：
>
> ```
> ① 动笔前先通读目标文档（至少目标章节），回答四问：
>    · 已有哪一节在讲同一件事？   → 合并进去，不开新节
>    · 本次更新推翻了哪些旧行？   → 同一次编辑里删掉或改掉（不许新旧并存）
>    · 这份文档的单一用途还成立吗？→ 不成立就拆/搬，不硬塞
>    · 真的需要新开文档吗？       → 先问能不能进现有的某一份
> ② 允许追加的只有登记型内容：E 报告的新实验小节（编号是身份）·
>    01 的新任务行（办完删行）· 02 的新决策/作废行（理由变了就地改原行）
> ③ 状态、结论、数字一律就地改；被取代的叙述就地删 —— 历史去 git log，
>    「⛔ 作废但保留」只用于 02 §4 登记表与 E 报告的「推翻了什么」节
> ④ 每次收尾看一眼行数：比上次长而没有新结论 ⇒ 你在堆历史，回去合并
> ```
>
> 其余纪律：每份文档单一用途；结论只写权威处，其它地方放指针；办完的事从 01 删行。

---

## 0 · 三十秒读懂

```
这条线在做   两个有真实需求支撑、可验证的 infra 项目（多卡/异步 RL/通信/kernel）
判据一句话   先有被测量出来的需求，才有优化目标；答不上「服务哪条兑现物」的实验一律停放
现在的状态   异步 RL 已跑通且跑对（E21/E22 修复）；PG 让 trainer 快 2.31×（E26）；
             陈旧度的剂量条件第一次真正具备 —— 研究问题回到了可测的位置
和主线的关系 两条线共用一个 git 仓库；往来只写根目录 /MAINLINE-INFRA.md（⛔ 禁止信件文档）
```

## 1 · ★ 读什么，按这个顺序（约 20 分钟）

```
1  01-TASKS.md                      ★ 队列唯一来源 —— 下一步做什么、谁在打
2  本文 §5 守则 + §6 跑前清单        都是拿代价换来的
3  STORY-async-lora-weight-sync.md  ★★★ 异步 RL 为什么两个月没在学、怎么接通的
4  TRACKS.md                        两条线各要兑现什么、现在在哪
5  02-DECISIONS.md                  回查「当时为什么这么定 / 哪些数字作废」才读
6  README.md                        E 索引 / 编号规则 / 报告模板 / 全局常量与机器画像
```

## 2 · 文档地图

| 文档 | 放什么 | 什么时候读 |
|---|---|---|
| **01** | ★ 队列（唯一来源），办完删行 | 每天 |
| **02** | 已定决策 · 作废登记 · 已落地改动 · 排序原则沿革 | 回查/引用旧数字前 |
| TRACKS | A/B 两条线的兑现物与完成度 | 想「这条线还欠什么」 |
| README | E 编号索引 · 编号规则 · 报告模板 · **全局常量/机器画像** | 建新实验 / 查常量 |
| NARRATIVE-AND-RESUME | 对外的完成态故事 + 简历（未测留〔 〕） | 收尾填数 |
| PRIMER-precision-sm120 | 背景层：精度格式/硬件特性/软件栈/训推组合的通俗讲解 | 新窗口补背景 |
| STORY | E21+E22 的完整叙事（现象→误判→根因→修法） | 新窗口必读一次 |
| E01–E26 | ★ **证据层**：编号是身份、永不重排、不合并 | 引用结论时 |
| /MAINLINE-INFRA.md | 与主线往来的唯一文档（开着的事 + 双方现状） | 有交界动作前 |

## 3 · 现在到哪一步（2026-08-20）

```
✅ 正确性   E21（梯度不同步）E22（LoRA 没推给 rollout）已修，异步 RL 第一次真正在学
            两个补丁默认开、不许关：SYNCOPATE_FSDP_DDP_FIX · SYNCOPATE_LORA_ADAPTER_SYNC
✅ 吞吐     E26 PrefixGrouper：生产现状→PG **端到端 2.31×**；cand 实测 11.33 s/gstep
            （构成：update_actor 54.3% · gen 23.9% · olp 18.8% · sync 1.8% · save 0.6%）
✅ 候选     cand_v13r2_e1（PG+mb8+KL关+seq IS）400 步全绿：候选 RL-100 配对 +0.186（t≈16）
            ESS 中位 0.92/最低 0.816 · rollout_corr/kl 中位 4e-4 在地板 · abort=0
✅ 默认值   **PG 开（mb 联动 8）· KL 关**已切库默认（08-20，上一行的证据垫底）；
            B5/KL 多种子/token-seq 多种子/「步数太少」全部由 candidate 兜底结案或撤销
✅ IO/工具   E29 ckpt 只存 LoRA（save 9.5×·写盘 12×·逐位校验常驻）
✅ E14/R2   两批+五臂扫描全完：**当日同尺子 12.84→9.23 s/gstep（−28%）**
            = 同步频率 16+紧阈值（杀接力赛的是暂停频率非陈旧度）+ CUDA graph（微间隙
            32.4s>计算）+ 乒乓修理②③（sync 912→236，阶梯逐级命中）；
            等时论证成立（激进陈旧档墙钟对齐后 +0.171 全场最高）；
            ⚠️ defer 在 64 步单种子是刀锋态（剂量不单调）⇒ 晋级只认多种子
✅ 收官     E14 全闭环（08-21，§4.10/§5）：s16/0.1 三种子复核过 + graph 单变量精度闸过
            ⇒ **sync-every 16 + CUDA graph 已切库默认**；compile 判死入边界表
✅ 精度线   08-21 一天全测完：trainer 前向 FP8 朴素接线判死（E19 §7）；serving 五臂
            全曲线（E19 §8：fp8 KV=容量杠杆+50% · FP4 权重对 4B 判死）；A3 探底闭环
            （E16 §6/§7：Triton 仿真实锤 · 块缩放 MMA 峰值阶梯 1:2:4:8 ·
            ★传统 FP8 mma 半速 · 数值逐位验证）；背景层沉淀 PRIMER-precision-sm120
✅ 三裁定   08-21 Chaoyu 已定：**fp8 KV 切默认**（launch_rl + serving 端点，02 §1）·
            trainer FP8 融合栈**降独立线**（01 §4，与 MoE 同档）· **A3/TileLang = 队首**
⬜ 欠的     🔴 队首 = A3 TileLang 自有算子（01 §1-3，尺子已立 1026/2055）；
            CoT 训练支持（产品线）；⚠️ 判据③ kl 地板在 fp8 KV 下首跑要实测重标
完成度     ⚠️ 08-20 换标：A=框架/异步（原 B）· B=算子/硬件（原 A）——
            Track A：诊断完 + before→after（B12/E26）+ 候选闭环；Track B：落地一条半
```

## 4 · 接着做什么

**看 [`01-TASKS.md`](01-TASKS.md)，别在这里维护第二份。**

## 5 · ★★ 守则（只留仍然有效的；出处在括号里，细节去 E 报告）

**① 判据五律**（E26 §7 七形状 + 四解药）：只在**终态**读（退出码/timing 首行）；
配一个**已知必然发生**的对照计数；bf16 下比等价先立**噪声地板**；修完现象**一字不变 = 改错地方**；
「我假设 X」一律写成断言，判据行必须真的比、比不了就说比不了。

**② 接缝三律**（E26 §6.2，13 处接线 5 处报错在别人家）：报错位置≈误导，
在接缝**自己打判据行** + **降维成秒级复现**（⛔ 别用「起训练」当调试循环）；
补丁先问**装在哪个进程**（trainer driver ≠ WorkerDict）；
`setup_worker` 里**绝不 import 碰 CUDA 的模块**（早于 Ray 分卡 ⇒ 三 rank 挤一张卡）。

**③ FSDP1 绝不绕过根模块的 forward 调子模块**（E26 §6.3）：归约由根输出上的
pre-backward hook 驱动，绕过 = **梯度静默不跨 rank 归约且不报错**；
判据 = 三 rank 梯度范数逐位相同（`SYNCOPATE_DDP_PROBE=1`）。

**④ 口径三律**：fully_async 的 timing 行覆盖几个 global step **必须实测**
（`scripts/parse_fully_async_timing.py`，单行日志答不了 ⇒ E26 曾因此作废一个 2.12×）；
`--steps` 固定的是更新次数不是数据量（E20 §7.10）；
**同一个指标换个前提/默认值就不是同一件事**（param_sync 18.8% 的教训，02 §4）。

**⑤ 单因素复现不了 ≠ 复现不了**（E26）：先**全因素照抄真实跑**求复现，再**留一法**收敛
到最小组合（那次是 state_dict × accum≥2，单开全不显形）。

**⑥ 撞到「框架行为不符合预期」先花五分钟搜上游 issue**（E21/E22 都能搜到，能省两个月）。

**⑦ 跑与盯**：起跑前过 `scripts/gpu_gate.sh`（⛔ 门禁没过不许起，别用 `;` 放行）；
监视器必须带**进程退出兜底**（启动即死时所有训练 pattern 都不出现）；
每个短跑收尾会落 **27 GB ckpt**（`--save-freq 999` 挡不住），跑完就删
`global_step_*`（dispatched.jsonl / rollout_dumps 留；**要过任务尺子的 ckpt 必须留**）。

**⑧ 工具卫生**：⛔ **第三方工具绝不 `uv pip install` 进生产 venv**（08-21 事故：装 llmcompressor
把 torch 2.9→2.13 整栈静默重解析；恢复 = `uv sync --frozen --all-extras` + flash-attn 反向判据）
——一次性工具住隔离 venv（`/workspace/venvs/<tool>`），产物走文件交接；
`pkill -f` 会杀掉你自己（用 `[.]` 转义或按 PID）；
nsys 不在 PATH（`/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`）、
只能包住启动、中间文件吃几十 GB；换 flash-attn 轮子先跑
`scripts/check_flash_attn_backward.py`（有轮子**前向全对反向全错**）。

**⑨ 框架的 `UserWarning` 当判据读**（E21 的告警在自己日志里躺了两个月）；
注释里**事实与推断分开写**，推断句标 `[推断，未验证]`。

**⑩ 提交按路径 `git add <文件>`，绝不 `-a`**（两条线共用仓库）；
⛔ **绝不传 `--lora-merge`**（bf16 合并毁掉 adapter 一半作用，启动已拦）；
两个正确性补丁与 `mb=8`（PG 下）**别再动**。

**⑪ 更新文档 = 改写，不是追加** —— 全文见本文头部「文档更新秩序」（两线通用）。

## 6 · 跑任何东西之前

```bash
set -a; . /workspace/.env; set +a     # ★ 不做的话缓存全落 16 G overlay，会爆盘
bash scripts/gpu_gate.sh              # 三条判据全过才许起
```

**开跑模板与每跑必查四条**：

```bash
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
  .venv/bin/python -m syncopate.train.launch_rl --model models/Qwen3-4B-sft-v13-e1 \
  --lora-rank 32 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --train-file data/rl/v13/train.parquet --val-file data/rl/v13/val.parquet ...
```

```
① [lora-probe]   list_loras() 非空（step≥1）
② [sync-payload] 第 2 次同步起 lora_ > 0（首次推基座是设计行为）
③ rollout_corr/kl 每次同步后回落 ~3.4e-4（不回落 = 同步没生效）
④ prompt_length/clip_ratio 必须 0.0000（100% 截断曾翻案一整条归因链）
```

★ 完整的训练前自查清单（环境→检查器→起跑→10 分钟判据行→晋级）在
**主线 `../syncopate/06-rl-run-protocol.md §1`**，两线通用；上面四条是其中的跑中最小集。

⚠️ 关机重启后：`/workspace` 是网络盘会活着；记忆软链断了重建：
`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`

### ★ 换机器完全重建清单（2026-08-21 立，搬家用）

```
① git clone + uv sync --frozen --all-extras   ⚠️ 裸 uv sync 会卸掉 train extras（守则⑧事故）
② 手放 /workspace/.env（含密钥，不在任何仓库）→ set -a; . /workspace/.env; set +a
③ 必跑 scripts/check_flash_attn_backward.py（退出码 0 才算环境可用；反向恒 0 比 nan 更毒）
④ 重建记忆软链（上面那行）
⑤ 资产：HF SamWang0405/Syncopate-AgenticRL 下回 bases/（底座真身，不可再生——
   重合并实测 max|Δ|=4.9e-4 > RL 信号 1.3e-5，禁止用"重新 merge"替代）+ 所需 adapter
⑥ reference/（版权包）只能人工拷，不进任何云端
```

**新机器重画像（拓扑变了哪些数字要重测；决策大概率不翻——当年都是输数倍）**：

```
🔴 必重测（~2h 全有现成尺子）：集合通信带宽（probe_allreduce/collective_bw）·
   16B 对齐悬崖复现（probe_alignment_cliff，预期复现=给上游的跨机验证）·
   NCCL 旋钮（LL128 结论可能翻）· 满载降频（probe_power_throttle）· README 机器画像重写
🟡 探针大动才复核：A6 三档稳态（DDP必选）· E04 TP=2 ——输 3–6× 的比赛不因跑道好 30% 翻盘
🟢 不动：全部正确性/学习/质量结论 · E16/A3 单卡硅片数字 · E19-c serving 曲线 · 全部默认值
⚠️ 方向未知处：同 socket 消掉 UPI 跳但四卡挤一份内存带宽——通信数字可能反变差，先测再说
⚠️ 判据③ 的 kl 地板（3.4e-4）在 fp8 KV 默认下会略抬：新机器首跑实测重标，别拿旧地板判罪
⚠️ 若新卡不是 5090/sm_120：E16/A3 全部硬件结论不迁移，重新探底
```

## 7 · 最后一句

这条线最贵的失效从来不是「算错」，是**「机制在但没接上」且不报错**
（E21 空组归约 · E22 参数没传 · E26 根旁路竞态 —— 三次都是同一个形状）。
遇到「看起来没问题」，**去比两个应该相同的东西**（跨 rank 梯度、开关两态的 logprob、
推出去的 ‖W‖ 与磁盘），别去读代码。
