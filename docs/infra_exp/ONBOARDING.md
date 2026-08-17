# infra 线 · 新窗口对接

> **每次收尾时更新这份文件**（怎么更新见文末 §7）。新窗口开场只说一句
> 「读 `docs/infra_exp/ONBOARDING.md`」就够了。
>
> 最后更新：**2026-08-17**

---

你接手的是 **Syncopate 项目的 infra 线**（多卡 / 异步 RL / 通信 / kernel / 模型选型）。
这条线和**主线训练**是**两条独立的线、共用一个 git 仓库**，由不同窗口负责 ——
这一点很重要，见 §5-1。

---

## 1 · 按这个顺序读（约 20 分钟）

```
1. docs/infra_exp/00-INFRA-HANDOFF.md   ★ 先读。现在在哪 / §5 队列 = 下一步做什么
2. docs/focus-migration-2026-08.md      焦点是怎么定下来的（唯一记录迁移历史的地方；
                                        其余文档只写当前态、不做新旧对照）
3. docs/infra_exp/E18-rank3-allgather-collapse.md   ★★ 最近一次完整实验，也是方法论样板
4. docs/infra_exp/TRACK-A / TRACK-B     两条线各要兑现什么、现在在哪（A*/B* 全表）
5. docs/infra_exp/README.md             E 编号索引 / §6 全局常量 / §7 最近实测 / 报告模板
6. docs/syncopate/08-machine-and-environment.md   环境怎么跑起来（★ 尤其 §2.0 和 §2.2）
```

**记忆在 `.claude/memory/`（已进 git）**，`MEMORY.md` 是索引。软链断了就重建：

```bash
ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory \
      /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory
```

---

## 2 · 每个 shell 的第一件事

```bash
set -a; . /workspace/.env; set +a
```

`/` 只有 **16 GB overlay**，`/workspace` 才是 300 GB 持久卷。
`.env` 里那十几条重定向（`RAY_TMPDIR` / `TRITON_CACHE_DIR` / `PGDATA` /
`LD_LIBRARY_PATH` …）**少一条，Ray 一溢写就把盘撑爆**。

---

## 3 · 最近一轮做完了什么（2026-08-17 夜）

**白天**：第 1 批四项 + 四条追加全部完成，刨到根 —— **3 卡 ZeRO-3 慢 6.02× 的根因是
「每 rank 分块字节数不被 16 整除 ⇒ NCCL 的 Simple kernel 整段放弃向量化 ⇒ 掉 12×」**
（完整叙事在 [`E18`](E18-rank3-allgather-collapse.md)）。

**夜里**（主线让出卡之后跑的第 1.6 批，队列在 `scripts/run_batch2_gpu.sh`）：

```
✅ A14  真实 ZeRO-3 的 all_gather **335.5 GB 里 99.9% 的字节**错位（每 rank 只差 4 个字节）
        ⇒ E18 闭环。同尺子对 Simple 96.08 vs LL128 29.76 = 3.23×（此前 3.33×，比值复现）
⛔ B2   **推翻了我自己的归因**：只改 bucket，512→9.04s / 2048→9.73s，**差 7.7% 不是 6.6×**
        ⇒ 「param_sync 现在只有 9 秒」是观测（成立）、「因为 bucket」是归因（证伪）
        ⇒ 新问题 **B15**：那 6 倍（55.8→9.0）到底是谁干的
✅ B3   **三模式同尺子**（这句兑现物此前是假的）：colocate3 55.28 / one_step_off 36.43 /
        fully_async 33.25 s 每步 ⇒ 1.52× / 1.66×，落在 P-B1 的 1.37–2.75× 内
        ★ one_step_off 把生成藏得**更彻底**却仍慢 10% ⇒ **杠杆是同步频率，不是藏生成**
✅ E01  白捡主线的 nsys：trainer gemm 58%/elementwise 24%/attn 12%；**GEMM 全是 cutlass_80**
        （Ampere 代）；E13 的修复在 kernel 层被证实（每步 CPU 快照 8.31 GB → 0.26 GB）
✅ E03  NCCL 旋钮层结案：只有两个旋钮真有用，其余全部无效（那张「无效」表最有用）
✅ B4写码 下发记账三类事件 + 首次验证 576/576/0；顺带堵住 Pool.ingest 的静默污染
```

⇒ **占空比的两个数**（干净跑复核）：权重同步占步 **7.2%**（不再是 18.8%）、
三次前向 **83.1%**（不是 72%）⇒ **B12/E17 升、B1 降**。

## 4 · ★ 第一件事：**先确认卡是空的**，再按第 1.6 批的队列跑

⛔⛔ **训练是最高优先级，严禁抢卡**（用户 2026-08-17 明令）。主线 RL **和 eval 及其后续管线**
全部跑完之前，一个 🔴 实验都不许起。判据是**产物落地 + 进程退出 + 显存归还**，
不是日志里那句「完成」（`scripts/wait_for_gpu.sh` 开头记着这个坑）。

```bash
scripts/gpu_gate.sh          # 🆕 三条判据一起查：显存 / 训练进程 / 主线产物
```

卡空之后，**按 `00-INFRA-HANDOFF §5 第 1.6 批` 的 8 项按序跑**（队首就是 A14）：

```bash
bash scripts/run_batch2_gpu.sh          # 🆕 串行跑完，每项自带预测与判据
```

**队首 A14**（~20 分钟）是上一轮唯一没闭合的环 —— **机制已证实，但还没证明 verl 的
ZeRO-3 真的撞在上面**：抓所有 `AllGather: N Bytes`，统计 `%16 != 0` 的**按字节加权**占比。

⚠️ **必须按字节加权，不能按调用次数** —— 小张量再多也解释不了 6.02×。
⚠️🆕 **`--fsdp-size 3` 会让 `launch_rl` 自动加 `NCCL_PROTO=LL128`** ⇒ 那一跑**不再是**
47.94 s 那条基线。要复现基线必须显式压回 Simple（`run_batch2_gpu.sh` 里两跑都做了）。

```
占比高 ⇒ 因果链闭合，接着 A15（决定给 NCCL 还是 FSDP 提 upstream issue）
占比低 ⇒ 6.02× 另有原因，这条链还差一环
```

⛔ **A14 出结果之前，不要把「6.02× 由对齐造成」写成定论。**
E18 §11 结尾和记忆里都钉了这条限定。

★ 本批的大头是 **B12 / E17 · 训练侧三次前向**（🆕 v13 实测占步 **84.9%**，比原记录的 72% 还大），
门槛 **A5**（nsys 拆解）—— ⚠️ **nsys 只能包住启动、不能事后 attach**，所以它必须挂在**我们自己**
的某一跑上，错过就再等一轮。

---

## 5 · ⚠️ 五条会让你踩坑的

1. **提交时按路径 `git add <具体文件>`，绝不用 `-a`。**
   训练线那个窗口在同一个仓库里工作（当前有 `data/splits/v13/`、`_audit/v13_*` 等
   未提交产出）—— **别碰**。
2. **换 flash-attn 轮子后先跑 `scripts/check_flash_attn_backward.py`。**
   有个轮子**前向数值三项全过、反向全错**（nan 或恒为 0），会让 RL 完全空转且不报错。
   ⇒ 「import 成功 ≠ 契约满足」的下一层是「**前向对 ≠ 反向对**」。
3. **`--weight-sync-bucket-mb` 默认 2048 会 OOM**（所有模式，不只分卡），短跑显式传 512。
4. **`--save-freq 999` 挡不住收尾保存** —— 每个短跑结束落一个 **27 GB** ckpt。跑完就删
   `checkpoints/grpo/<exp>/global_step_*`（`dispatched.jsonl` 和 `rollout_dumps` 要留）。
5. **fully_async 的 timing 行覆盖 4 个 global step**，绝对秒数要 ÷4。
   ⇒ 🆕 别再靠人记：用 `scripts/parse_fully_async_timing.py`，它按 `global_step` 差分**实测**覆盖步数。
6. 🆕 **引用「权重同步占步 18.8% / 55.8 s」之前先看 bucket**。同为 fully_async 稳态，
   bucket 2048 是 55.8 s、bucket 512 是 **8.43 s**（差 6.6×，README §7.4）。
   ⇒ 「同一个指标换个模式就不是一件事」的**下一层**：同一个模式换个默认参数也不是。
7. 🆕 **`nsys` 装了但不在 PATH**：`/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`。
   ⚠️ 它**只能包住启动**，不能对已经在跑的 Ray 作业 attach ⇒ A5 必须提前规划进某一跑。
   （`py-spy` 在 `.venv/bin/`，那个**可以** attach，用于 Python 栈采样。）

---

## 6 · 三条纪律（这条线的立身之本）

1. **先写预测再跑。** 错了就在报告 §6 写「原猜想 / 实测 / 推翻后 / 教训」四段。
2. **每个加速比都要有同分母**，一次只变一个变量。
3. **只报吞吐不报任务精度的优化不算完成**（队列 B5 至今一次没过）。

★ 上一轮那条链上**错了三次、对了一次** —— 唯一对的那次是**基于测量**而不是推理的预测。
停在「NCCL 选错了」会得到一个**看起来完整、实际是错的**根因，而且导出错误建议（换协议），
真正的正解是补齐对齐。⇒ **多问一句「那它为什么选错」。**

---

## 7 · 现成的工具（别重写）

```
scripts/probe_allreduce_bw.py          卡间 all-reduce 带宽（E00 口径）
scripts/probe_collective_bw.py         算子 × 卡数 的二维带宽扫描
scripts/probe_collective_granularity.py 固定总字节、只改切分次数
scripts/probe_alignment_cliff.py       ★ 16 字节对齐悬崖
scripts/probe_power_throttle.py        满载功耗与降频
scripts/check_flash_attn_backward.py   ★ flash-attn **反向**数值判据
scripts/check_data_gates.py            数据门槛
scripts/parse_fully_async_timing.py 🆕 timing 行 → 每 global step 的口径（自动求覆盖步数）
scripts/analyze_drift.py            🆕 发出/完成/训练到 三段的漂移（⚠️ 跑完再读）
scripts/analyze_nsys_step.py        🆕 nsys sqlite → 按**进程**（不是 deviceId！）拆算子构成
scripts/gpu_gate.sh                 🆕 ★ 抢卡门禁：显存 + 训练进程 + 主线产物 三条一起查
scripts/run_batch2_gpu.sh           🆕 第 1.6 批 8 项串行跑（每项自带预测与判据）
scripts/wait_for_gpu.sh                等显存释放（教训：日志说完了 ≠ 资源还回来了）
logs/e00_*.json · logs/e02_*.json      以上所有原始数据
```

---

## 8 · 怎么维护这份文件（收尾时做）

每次收尾，**只改这四处**，别让它增量堆长：

| 改哪 | 改成什么 |
|---|---|
| §3 最近一轮做完了什么 | **整段换掉**，只留最新一轮。历史归 E 报告和 `focus-migration` |
| §4 第一件事 | 换成新的队首任务，附**可直接粘贴的命令**和**判据** |
| §5 踩坑 | 只留**仍然有效**的；修好的删掉（细节在 E 报告里） |
| §7 工具 | 新增探针时加一行 |

⇒ 目标是**新窗口 20 分钟内能接上手**，不是记录完整历史。
完整历史在 E 报告、`focus-migration-2026-08.md` 和 `.claude/memory/`。
