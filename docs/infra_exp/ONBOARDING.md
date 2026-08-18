# infra 线 · 新窗口对接

> **每次收尾时更新这份文件**（怎么更新见文末 §7）。新窗口开场只说一句
> 「读 `docs/infra_exp/ONBOARDING.md`」就够了。
>
> 最后更新：**2026-08-18**

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

## 3 · 最近一轮做完了什么（2026-08-18）

★★★ **头等大事：抓到并修好了一个静默的正确性 bug** —— [`E21`](E21-ddp-not-syncing.md)

```
现象   三个 trainer rank **各训各的 LoRA，梯度从没同步过** ⇒ 每次更新只用 1/3 的数据
判据   lora_B 是零初始化的 ⇒ step1 三个 rank 权重都是 0.000000（起点相同）
       而梯度范数不同（2.209e-05 / 2.565e-05）⇒ 只可能是没 all-reduce
根因   fsdp_size=1 ⇒ 网格 (3,1) ⇒ HYBRID_SHARD
       ⇒ PyTorch 见分片维=1 自动降级成 NO_SHARD，**却把归约留在那个大小为 1 的组上**
       ⇒ 空操作。**只打了一行 UserWarning，训练照常跑完、所有指标正常。**
修复   拦住退化网格的 FSDP 构造，改用 NO_SHARD + 默认进程组（默认开启）
复验   三个 rank 的梯度**逐位相同**（最小复现 + 真实训练双验证）
上游   两份草稿已成文 → docs/upstream/（PyTorch 那条 + verl 那条）
```

⚠️⚠️ **⇒ 此前所有位移 / ESS 的绝对值都在坏基线上**，重测队列见 `00-INFRA-HANDOFF §5`。

**同一轮的另外三条**：

```
E20  RL 学不动的两个独立原因：① 序列级 IS 在 694 token 上指数崩塌
     （chi2_seq 64.19 vs chi2_token 0.065，**差 989×**）② 一个 epoch 只更新 109 次
     ⇒ token 级 IS 实测把 ESS 从 0.449 修到 **1.000**、grad_norm 趋势反转，**零吞吐代价**
E19  FP8 在 sm_120 上是真的（真实形状 1.70–2.22×）；**ref 可以换、rollout 先别换**
     （FP8 的误差是 vLLM↔FSDP 数值地板的 316 倍，会直接喂大 E20 那个问题）
E01  一步的时间去哪了：三次前向占 kernel 时间 83.2%（与墙钟 83.1% 几乎相等）
     ⚠️ 但卡只忙 74.6–78.2%，**有 22–25% 的空档** —— 我曾说过头说成"卡是满的"
```

## 4 · ★ 第一件事：**先做管线排查，再谈重测**

E21 的教训是「**四处信号一处都没接住**」（那行 warning 每跑打两次，就在我们自己的日志里）。
所以下一步**不是**急着重跑实验，而是：

```bash
# ① 先看 handoff §5.1 的「排查清单」—— 还有哪些"从没验证过的前提"
# ② 每条写成**断言或探针**，不要用"读代码确认"
#    （E21 证明读代码会漏：我们读对了三句，错在第四句）
# ③ 再按 handoff §5 的重测队列 R1→R7 重跑
```

⚠️ **判据（重测时省一半力气用的）**：
**同一批里两臂都受同样影响的 A/B，比值仍可信；绝对值一律作废。**

⛔ **在排查完成之前，不要把任何数字写进默认配置。**

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
8. 🆕🔴🔴 **框架打出的 `UserWarning` 要当判据读，别只盯自己打的判据行。**
   E21 那个静默 bug 的告警**每跑打两次、一直在我们自己的日志里**，两个月没人看。
   ⇒ 训练起来后 `grep -c "UserWarning" <log>`，**新出现的必须有人看过**。
9. 🆕🔴 **「读码得出的事实」和「据此做的推断」要在注释里分开写。**
   E21 那段注释里三句事实 + 一句推断，排版完全一样 ⇒ 没人会去质疑第四句，而错的正是它。
   ⇒ 推断句标 `[推断，未验证]`。
10. 🆕 **写工具时把"我假设 X 成立"写成断言** —— 成本几乎为零。
   E21 就是被 `rl_ckpt_to_adapter.py` 里一句随手写的断言抓到的，
   **而四处显式信号都没抓住它**。
11. 🆕🔴 **nsys 的中间文件会吃掉几十 GB**：180 秒采样在 `/workspace/tmp/nvidia/nsight_systems/`
   下产生 **72 GB**（最终 `.nsys-rep` 只有 386 MB —— **差两个数量级**）。
   ⇒ 采样前先看 `df`；夜间无人值守时挂上 `scripts/disk_guard.sh`（低于 15 G 就杀 nsys 保队列）。
   ⚠️ 导出的 `.sqlite` 也有 **8.8 GB**，分析完就删。

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
scripts/analyze_nsys_step.py        🆕 nsys sqlite → 按**进程**拆算子构成 + NVTX 阶段归属 + 空泡分布
scripts/rl_ckpt_drift.py            🆕 位移 ‖ΔW_eff‖/‖W_base‖（⚠️ 只读 rank_0）
scripts/rl_ckpt_to_adapter.py       🆕 RL ckpt → PEFT adapter（★ 评测链路的缺口；E21 就是它抓到的）
scripts/repro_fsdp_hybrid_nosync.py 🆕 ★ E21 的最小复现，`REPRO_APPLY_FIX=1` 兼作修复验证
scripts/probe_sm120_fp8.py / probe_fp8_real_shapes.py / probe_fp8_logprob_error.py  🆕 FP8 三件套
scripts/probe_moe_4bit_load.py      🆕 4bit MoE 加载路径（碎片）
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
