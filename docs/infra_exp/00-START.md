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
> ★ 维护纪律（跟主线学的，代价他们付过）：**每份文档单一用途；结论只写权威处，
> 其它地方放指针；过时就地删，历史去 git log；办完的事从 01 删行。**

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
| STORY | E21+E22 的完整叙事（现象→误判→根因→修法） | 新窗口必读一次 |
| E01–E26 | ★ **证据层**：编号是身份、永不重排、不合并 | 引用结论时 |
| /MAINLINE-INFRA.md | 与主线往来的唯一文档（开着的事 + 双方现状） | 有交界动作前 |

## 3 · 现在到哪一步（2026-08-19）

```
✅ 正确性   E21（梯度不同步）E22（LoRA 没推给 rollout）已修，异步 RL 第一次真正在学
            两个补丁默认开、不许关：SYNCOPATE_FSDP_DDP_FIX · SYNCOPATE_LORA_ADAPTER_SYNC
✅ 吞吐     E26 PrefixGrouper：生产现状→PG **端到端 2.31×**（34.52→14.94 s/gstep），
            生产配置定格 mb=8；gen 占步 12%→26% ⇒ 瓶颈开始向 rollout 移动
✅ 学习     RL 任务分 +0.101→+0.137（修截断后）；defer 崩塌翻案 = prompt 截断不是 reward
⬜ 欠的     B5 任务尺子（PG 默认开的门槛）· KL/IS 多种子 · 「步数太少」正式验证
完成度     Track B：诊断完 + 第一个 before→after（B12/E26）；Track A：四条兑现物落地一条半
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

**⑧ 工具卫生**：`pkill -f` 会杀掉你自己（用 `[.]` 转义或按 PID）；
nsys 不在 PATH（`/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`）、
只能包住启动、中间文件吃几十 GB；换 flash-attn 轮子先跑
`scripts/check_flash_attn_backward.py`（有轮子**前向全对反向全错**）。

**⑨ 框架的 `UserWarning` 当判据读**（E21 的告警在自己日志里躺了两个月）；
注释里**事实与推断分开写**，推断句标 `[推断，未验证]`。

**⑩ 提交按路径 `git add <文件>`，绝不 `-a`**（两条线共用仓库）；
⛔ **绝不传 `--lora-merge`**（bf16 合并毁掉 adapter 一半作用，启动已拦）；
两个正确性补丁与 `mb=8`（PG 下）**别再动**。

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

⚠️ 关机重启后：`/workspace` 是网络盘会活着；记忆软链断了重建：
`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`

## 7 · 最后一句

这条线最贵的失效从来不是「算错」，是**「机制在但没接上」且不报错**
（E21 空组归约 · E22 参数没传 · E26 根旁路竞态 —— 三次都是同一个形状）。
遇到「看起来没问题」，**去比两个应该相同的东西**（跨 rank 梯度、开关两态的 logprob、
推出去的 ‖W‖ 与磁盘），别去读代码。
