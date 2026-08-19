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
3. docs/infra_exp/STORY-async-lora-weight-sync.md ★★★ **先读这份** —— 异步 RL 为什么两个月
                                        没在学、我们怎么接通的（现象→误判→根因→修法→结果）
4. docs/infra_exp/E18-rank3-allgather-collapse.md   ★★ 方法论样板
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

★★★ **抓到并修好了两个静默的正确性 bug。它们叠在一起，让整条 RL 回路是断的。**

**① [`E22`](E22-lora-never-synced.md) —— LoRA 从没被推给 rollout（影响面最大）**

```
现象   fully_async / one_step_off 下，每次权重同步推给 vLLM 的都是**冻结基座**
判据   推出去的 q_proj.base_layer.weight ‖W‖=75.377708
       与**磁盘上起点模型逐位相同**（4 跑 × 2 次同步全一致）；含 lora_ 的张量 **0 个**
根因   engine_workers.py:698 在 disaggregated 分支**只调一次** get_per_tensor_param() 且不传参
       ⇒ base_sync_done=False ⇒ collect_lora_params **显式跳过所有 lora_ 张量**
       （colocate 那条路调两次，先基座后 adapter，**是对的**）
后果   **rollout 永远用起点策略 π₀ 采样** ⇒ 我们从来没跑过一次正确的异步 RL
止血   --lora-merge（bf16 合并）⇒ ⛔ **已被 R0-b 否掉**：合并造成的 logprob 偏移中位
       1.717e-02 = **adapter 自身作用的 50%**，是引擎地板的 50×
修法   ✅ **我们自己把 verl 缺的那段管子接上了**（默认开启，SYNCOPATE_LORA_ADAPTER_SYNC）
       trainer 侧首次送基座、之后送 adapter；rollout 侧带上"这是 adapter"的标记
验证   list_loras() []→**[123]** · 载荷 8,414→**252 MiB** · kl 回到地板 ·
       **param_sync 6.25→0.974 s（6.4×）** · 60 步 0 错误
数值   V1 两侧 peft_config scaling **都是 2.0** · V3 log_ppl_diff 落在同版本地板 ~3.4e-4
       ⇒ **「推了 adapter」和「推对了」两条都验了**
```

⇒ ⭐ **异步 RL 第一次真正跑通。** 整条链（现象→误判→根因→修法→结果）见
[`STORY-async-lora-weight-sync.md`](STORY-async-lora-weight-sync.md)。

**② [`E21`](E21-ddp-not-syncing.md) —— 三个 trainer rank 的梯度从没同步过**

```
判据   lora_B 零初始化 ⇒ step1 三 rank 权重都是 0.000000，而梯度范数不同 ⇒ 没 all-reduce
根因   fsdp_size=1 ⇒ 网格 (3,1) ⇒ HYBRID_SHARD ⇒ PyTorch 见分片维=1 降级成 NO_SHARD，
       **却把归约留在那个大小为 1 的组上** ⇒ 空操作。只打一行 UserWarning
修复   拦住退化网格的 FSDP 构造（默认开启）。三 rank 梯度**逐位相同**
0-A    ✅ 又验了「合得对不对」：3 卡 = 1 卡，比值 **1.000000**（是求平均，口径正确）
       白捡：verl 那套「按全局 token 数归一 × dp_size」在**变长序列**下确实在保护我们
       （最常见的"本地平均"写法在不等量数据上错 26%，方向都不对）
上游   **三份**草稿已成文 → docs/upstream/（PyTorch 一条 + verl 两条）。**提交由另一位负责。**
🔻     PyTorch 那条**已知结局**：有人报过、也真的提到了 GitHub，但 **FSDP1 已进入最低限度维护**，
       不会被处理。换 FSDP2 不会遇到，但 FSDP2 另有**张量形态不一致**的问题
       ⇒ **Chaoyu 决定：管线刚跑通，先不换后端。**
       ⇒ **所以我们的退化网格补丁是长期方案，SYNCOPATE_FSDP_DDP_FIX 必须一直默认开。**
```

⚠️⚠️ **⇒ 作废清单与重跑队列见 `00-INFRA-HANDOFF §5`。一句话判据**：
**算了多少、搬了多少字节 → 不受影响；算得对不对、学到没有 → 全部作废。**

**同一轮的另外两条**（结论本身仍成立，但绝对值受上面影响）：

```
E20  RL 学不动：① 序列级 IS 在 694 token 上指数崩塌（chi2_seq 64.19 vs chi2_token 0.065）
     ② 一个 epoch 只更新 110 次。★ 这两条**数学结论**不受 E21/E22 影响，实测数字全部作废
E19  FP8 在 sm_120 上是真的（1.70–2.22×）；ref 可换、rollout 先别换
```

## 4 · ★ 第一件事：**R1 —— 在修好的异步基线上重测 E20**

2026-08-18 晚 **异步 RL 第一次真正跑通**（E22 §6.4）：我们自己补上了 verl 缺的那段管子，
adapter 现在真的会到达 vLLM（`list_loras()=[123]`），载荷 8,414→252 MiB，
`param_sync` 6.25→**0.974 s**。⇒ **三个前提（梯度同步 / 归约口径 / adapter 送达）全部就位。**

```bash
# R1：E20 全套重测（序列级 vs token 级 IS），**直接在 fully_async 上跑**
SYNCOPATE_LORA_ADAPTER_SYNC=1 SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708 \
  .venv/bin/python -m syncopate.train.launch_rl --model models/Qwen3-4B-sft-v13-e1 \
  --lora-rank 32 --rollout-is token --mode fully_async --trainer-gpus 3 --rollout-gpus 1 \
  --weight-sync-bucket-mb 512 --steps 60 --sync-every 4 ...
```

⛔ **每一跑必须带 `SYNCOPATE_LORA_ADAPTER_SYNC=1`** —— 不带就退回"推冻结基座"。
⛔ **不要用 `--lora-merge`**（R0-b：bf16 合并毁掉 adapter 一半的作用）。

★ **三条常驻判据，一跑完就查**：
```
① [lora-probe] list_loras() 必须非空（step≥1）    ⇒ adapter 真的在引擎里
② [sync-payload] 第 2 次同步起含 lora_ 张量 > 0    ⇒ 推的是 adapter 不是基座
③ rollout_corr/kl 每次同步后回落到地板 ~3.4e-4     ⇒ 策略真的被交付了
```
⚠️ **判据要有能力分辨**：跑太短（<8 次同步）时 ①②③ 里的 ③ 分辨不出来 ——
坏基线早期也贴着地板，两条曲线要到第 7 个版本才拉开 10×。

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
6. ⛔ **「权重同步占步 18.8% / 55.8 s」这组数已作废**（它们是在「每次推 8.4 GB 冻结基座」时量的，
   而当时以为只推 132 MB）。✅ 现值：修法① 之后 `param_sync` **0.974 s / 占一步 0.8%**。
   ⇒ 教训仍然有效且更狠：**同一个指标换个默认参数就不是一件事 —— 甚至换个"前提对不对"也不是。**
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
11. 🆕🔴🔴 **LoRA + 异步模式必须传 `--lora-merge`**（E22）。不传的话每次权重同步推的是
   **冻结基座**，rollout 的策略永远停在起点 —— 不报错、指标全正常。
   ⇒ 判据：`SYNCOPATE_SYNC_PAYLOAD=1` + `SYNCOPATE_SYNC_REF=<起点 ‖W‖>`，盯住层必须**逐次变化**。
12. 🆕🔴🔴 **每次权重同步之后，`rollout_corr/kl` 必须回落到首步那个数量级**（数值失配地板，
   实测 3.4e-4）。**不回落 = 同步没生效。**（主线 §11.2 给的判据，比我的载荷探针更省 ——
   它**只用已有指标、可以回溯地跑在旧日志上**。两条都留：载荷探针在第 1 次同步就报，
   kl 判据要到第 3 个版本才显形。）
13. 🆕🔴 **判据行的文案本身也是判据的一部分。** 写 0-B 探针时我**两次**打出
   「与磁盘起点相同」这句结论 —— 一次是判据根本没绑上（`‖W‖=None`），
   一次是结论**硬编码在文案里、压根没做比较**。
   ⇒ **判据必须真的比；比不了就说比不了，不许打结论。**
14. 🆕🔴🔴 **撞到"框架行为不符合预期"时，先花五分钟搜上游 issue / 论坛。**
   今天两个基石级 bug **都能搜到**：E21 有人在 PyTorch 论坛报过（维护者说 looks like a bug、
   要复现脚本，至今没人提）；E22 的限制在 `verl#2048` 里明写着（"LoRA 只支持同步 rollout"，
   关成 not planned）。⇒ **成本几乎为零，这次能省下的是两个月。**
15. 🆕 **nsys 的中间文件会吃掉几十 GB**：180 秒采样在 `/workspace/tmp/nvidia/nsight_systems/`
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
scripts/repro_ddp_reduce_convention.py 🆕 ★ 0-A：3 卡合起来的梯度 == 1 卡在全部数据上的梯度？
scripts/probe_weight_sync_payload.py 🆕 ★ 0-B：权重同步**推的是什么**（离线复现发送侧分支）
  ⇒ 配套运行时探针 `SYNCOPATE_SYNC_PAYLOAD=1`（在 verl_patches 里，判据带 SYNCOPATE_SYNC_REF）
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
