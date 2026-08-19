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

## 3 · 最近一轮做完了什么（2026-08-18 ~ 08-19）

**两件事**：白天修好了两个静默的正确性 bug，夜里用一条 9 小时队列（19 项）把受影响的实验全部重测。

### 3.1 两个 bug（都已修、已验证、默认开启）

```
E22  LoRA **从没被推给 rollout** —— 异步模式下每次同步推的是 8.4 GB 冻结基座
     ⇒ 生成数据的策略两个月停在起点，整条 RL 回路是断的
     ⇒ 我们**自己补上了 verl 缺的那段传参**（SYNCOPATE_LORA_ADAPTER_SYNC，**默认开**）
        验证：引擎里 adapter 从 [] 变 [123] · 载荷 8,414→252 MiB · param_sync 6.25→0.974 s
E21  三个 trainer rank 的**梯度从没同步过**（退化网格 ⇒ PyTorch 静默降级）
     ⇒ 已修（SYNCOPATE_FSDP_DDP_FIX，**默认开**）；0-A 又验了归约口径正确（3 卡 = 1 卡）
     🔻 上游**不会修**（FSDP1 已最低限度维护）⇒ **我们的补丁是长期方案**
```
⇒ 完整故事：[`STORY-async-lora-weight-sync.md`](STORY-async-lora-weight-sync.md)

### 3.2 夜跑 19 项的结论（**这些才是现在的事实**）

```
✅ B5 任务尺子**第一次通过**：+0.101（t=9.3，MDE 0.022），且**不是 reward hacking**
✅ E24 合并损失 −0.025 ⇒ 配对基线一律用 _audit/v13_sft_e1_merged.json
🔴 **否定结果**：token 级 vs 序列级 IS 在任务尺子上 **+0.000（MDE 0.016）**
   ⇒ **让 RL 从"学不动"变成"+0.101"的是 E21+E22 的修复，不是 IS 口径**
🔴🔴 **defer 崩塌**：lr 3e-5 该拒绝时 97%→83%；**lr 1e-4 →0%**，而总分仍 +0.063
   ⇒ 当前 reward 下 RL 会系统性学"不拒绝" ⇒ **reward 设计问题，归主线**
✅ B19/B10：sync_every 4→16（吞吐 +11.4%）· 陈旧阈值 0.1→0.5（陈旧样本 6×）
   两个旋钮在估计量上**都测不出代价**，但**都没过任务尺子**
★★ 那 11.4% **96% 省在 `gen`（trainer 等样本）上**，param_sync 只占 4%
   ⇒ **同步的真实代价是"打断 rollout"，不是"搬权重"** ⇒ **B1 彻底死了**
```

### 3.3 ★ 四条方法论（都是被自己的判据抓到的，比数字更值钱）

```
1. **总分连着两次盖住真实差异**（+0.101 很漂亮而 defer 97→83；总分打平而 defer 差 14 点）
   ⇒ 三计数比均值好，但**仍是"打包"判据**；要看的是"哪一类行为变了"
2. **训练分与任务分给出相反排序**：lr 1e-4 训练分更高，任务分显著更低（−0.039）
   ⇒ **训练时看到的分数不能用来选配置**
3. **「首值→末值」在噪声带宽 ≥ 变化量时会凭空造出趋势**（把 30 个点全打出来才发现）
4. **五条判据被证伪，三条是当晚自己写的** ⇒ **判据必须能对自己失败**
```

⚠️ **已作废的旧数字全部搬进** [`../syncopate/21-invalidated-numbers.md §5`](../syncopate/21-invalidated-numbers.md)，
实验报告正文只留一行指针。**归档只用于回溯当时怎么想，不许引用里面的数字。**

## 4 · ★ 第一件事：**队列已被夜跑改写，先读 handoff §5.1**

R1（重测 E20）、⑥（重基线）、B5（任务尺子）**都已做完**。新队首见
[`00-INFRA-HANDOFF §5.1`](00-INFRA-HANDOFF.md)，一句话版：

```
1 🔴🔴 reward 在教模型"不拒绝"          主线的活（今晚最严重的发现）
2 把 defer 双向率 / REJ 分 / fabricated_safety_line 提成**常驻判据**
3 token vs sequence **各跑 2–3 个种子**，判据用 defer 而不是总分
4 「更新次数」实验**重新设计**：固定跑满一个 epoch，不是固定步数
5 🆕 **让同步不打断 rollout**（双缓冲 / 加深队列）—— B1 死后钱在这里
6 调大 sync_every 之前**先过 B5**
```

**开跑的模板**（三条常驻判据靠这两个环境变量产出）：

```bash
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
  .venv/bin/python -m syncopate.train.launch_rl --model models/Qwen3-4B-sft-v13-e1 \
  --lora-rank 32 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 --sync-every 4 ...
```

★ **每跑完必查的三条**（缺一条就不要读结果）：
```
① [lora-probe]   list_loras() 非空（step≥1）  ⇒ adapter 真在引擎里
② [sync-payload] 第 2 次同步起含 lora_ > 0    ⇒ 推的是 adapter 不是基座
③ rollout_corr/kl 每次同步后回落到 ~3.4e-4    ⇒ 策略真的被交付了
⚠️ 跑太短（<8 次同步）时 ③ **分辨不出来** —— 坏基线早期也贴着地板
```

## 5 · ⚠️ 五条会让你踩坑的

1. **提交时按路径 `git add <具体文件>`，绝不用 `-a`。**
   训练线那个窗口在同一个仓库里工作（当前有 `data/splits/v13/`、`_audit/v13_*` 等
   未提交产出）—— **别碰**。
2. **换 flash-attn 轮子后先跑 `scripts/check_flash_attn_backward.py`。**
   有个轮子**前向数值三项全过、反向全错**（nan 或恒为 0），会让 RL 完全空转且不报错。
   ⇒ 「import 成功 ≠ 契约满足」的下一层是「**前向对 ≠ 反向对**」。
3. ~~`--weight-sync-bucket-mb` 默认 2048 会 OOM~~ ✅ **已修**：默认值已改成 512，不用再手动传。
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
11. ⛔⛔ **绝对不要传 `--lora-merge`**（这条 2026-08-18 曾写反过，务必看清）。
   bf16 合并会**毁掉 adapter 一半的作用**（logprob 偏移中位 1.717e-02 = adapter 自身作用的 50%，
   是引擎噪声地板的 50×，E22 §6.1）。现在 `launch_rl` **启动就报错拦住**。
   ✅ 正确做法：什么都不用传 —— `SYNCOPATE_LORA_ADAPTER_SYNC` **默认开**，会自动推 adapter。
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
15. 🆕🔴 **`--ppo-mini-batch-size × --rollout-n` 必须能被 trainer 卡数整除**。
   3 卡下可用值是 `[3,6,9,12,15]`。传 2 的话 verl 要**跑起来两分钟**才报 `16 % 3 != 0`；
   现在 `launch_rl` 启动秒炸并列出可用值。
16. 🆕🔴🔴 **`--steps` 在 `mini_batch` 变化时换了含义**：它固定的是"更新多少次"，
   **不是"训练多久"**。固定 `--steps` 时把 `mini_batch` 减半，只会让**每步数据减半**、
   更新次数不变（E20 §7.10 实测）。⇒ 要测"每个 epoch 更新几次"，**必须固定数据量而不是步数**。
17. 🆕🔴 **不要用 `pkill -f "xxx.sh"`** —— 它会把**你自己这条命令**也匹配上杀掉
   （命令行里含同样的字符串，08-18 实测中招）。⇒ 用 `[.]` 转义或直接按 PID。
18. 🆕 **nsys 的中间文件会吃掉几十 GB**：180 秒采样在 `/workspace/tmp/nvidia/nsight_systems/`
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
scripts/probe_merge_logprob_fidelity.py 🆕 ★ bf16 合并让策略偏了多少（同引擎比两份权重）
scripts/probe_pipeline_invariants_on_clean_runs.py 🆕 干净基线上复查管线不变量（不吃卡）
scripts/run_queue_9h.sh + run_queue_retry{2,3,4}.sh 🆕 ★ 会自己往下走的任务队列
  （每项自带：磁盘/显存前置检查 · 判据解析 · ckpt 策略 · 完成标记 logs/queue9h/T*.done）
探针开关（都在 verl_patches 里）：
  SYNCOPATE_LORA_ADAPTER_SYNC=1  **默认开** · E22 修法①（推 adapter）
  SYNCOPATE_FSDP_DDP_FIX=1       **默认开** · E21 修复（退化网格）
  SYNCOPATE_SYNC_PAYLOAD=1       每次同步打「张量数/字节/lora_ 个数/盯住层 ‖W‖」+ list_loras()
  SYNCOPATE_OPT_STEP_PROBE=1     **真实**优化器更新次数（不是 fit step / dump 数 / param_version）
  SYNCOPATE_DDP_PROBE=1          三个 rank 的梯度范数（E21 的判据）
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
