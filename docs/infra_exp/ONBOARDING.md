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
4. docs/infra_exp/INFRA-REPLY-TO-MAINLINE-2026-08-19.md  📮 与主线的往来（还开着哪一项）
5. docs/infra_exp/E18-rank3-allgather-collapse.md   ★★ 方法论样板
4. docs/infra_exp/TRACK-A / TRACK-B     两条线各要兑现什么、现在在哪（A*/B* 全表）
5. docs/infra_exp/E26-prefix-grouper.md ★★ **判据方法论集大成**（§7 判据失效七形状 · §6 十三处接线）
6. docs/infra_exp/README.md             E 编号索引 / §6 全局常量 / §7 最近实测 / 报告模板
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

## 3 · 最近一轮做完了什么（2026-08-19）

**一句话**：修好 prompt 截断之后**一条大归因被推翻**；KL 可以砍；PrefixGrouper 在微基准上
兑现 3.96×，但**真实训练集成没跑通**（13 处接线，卡在优化器 dtype）。

### 3.1 ⛔⛔ 头条：defer 崩塌是 **prompt 截断**，不是 reward（[E20 §7.12](E20-rl-not-learning.md)）

E17 A 臂与夜跑 `r1_tokenis` **同配置、只差长度预算**（3584/1536 → 5120/2048）：

```
                    3584（100% 截断）      5120（0% 截断）
该 defer            97% → **83%** 🟠      97% → **100%** ✅
REJ（8 条）         **−0.188** 🔴          **+0.203** ✅
fabricated_safety   **+3** 🔴              **−2** ✅
任务分              +0.101                 **+0.137**（t=11.6）
```

机制：左截断砍掉的正是「调查先于任何结论（含**拒绝**、反问）」（截断后存活 **0/659**）
⇒ **模型训练时从没见过"可以拒绝"这个选项，评测时却见得到。**
⇒ 主线 R-1 应从队首撤下（回信 [`INFRA-TO-MAINLINE-2026-08-19b.md`](INFRA-TO-MAINLINE-2026-08-19b.md)）。
★ **"行为异常"先查输入，再查激励。**

### 3.2 ✅ E17 KL 两臂：砍 KL 省 15.4%，任务分无代价（[E17 §9](E17-triple-forward.md)）

```
吞吐   step 135.92 → 115.03 s（−15.4%，正好等于 ref 的 20.93 s）；B 臂 timing_s/ref 出现 0 次
精度   A vs B 直接配对 −0.009 < MDE 0.015 ⇒ 没测出差异；defer 100% vs 100%；REJ 0.953 双同
🟠 唯一反向信号  fabricated_safety_line_cap A 6 → B 8（+2）⇒ 多种子时必须盯
⛔ 连带           E19「ref 走 FP8」失效（ref 都不算了）
⚠️ 默认值尚未改   等多种子证据链齐
```

### 3.3 🟡 E25 / E26：喂饱 GPU 的单位是 token，省时间只能靠「少算」

```
E25 ✅ 「trainer 没喂饱」**被证伪**：micro_batch 1→2 是**负收益**（定长 −1.0% / 变长 −6.3%，
       mb=4 OOM）；关 gradient_checkpointing 在 mb=1 就 OOM
       ★ 一条序列已 ~4850 token ⇒ GPU 早就吃饱；变长下 mb=1 等价于完美打包
E26 ✅ PrefixGrouper 微基准 **3.96×**（三次前向 10.211→2.577 s，显存 +3.6%），
       等价性 fp32 逐位过（四配置含组 8）
    🔴 **真实训练集成未完成** —— 16 次尝试 / 13 处接线 / 卡在 Adam dtype（E26 §6.3）
    ⛔ **端到端吞吐数一个都没有**，3.96× 只在微基准上成立
```

### 3.4 ★★★ 判据失效的**七种形状**（一天集齐，[E26 §7](E26-prefix-grouper.md)）

```
设计错  D1 太松 ⇒ 为错误的理由通过（"量墙钟"会把"没接上"读成"没收益"）
        D2 太严 ⇒ 把对的判成错的（bf16 下要求逐位相同；"组大小必须相等"误杀合法情形）
执行错  E1 读早了（"还没发生"与"不会发生"在日志里一样）
        E2 读错对象（新跑没起来，读的是旧日志）
        E3 读到文案（安装横幅里含判据行字符串）
        E4 覆盖面不足（只挂第一个参数的钩子）
        E5 打错目标（patch AdamW 而实际是 Adam ⇒ 零迹象）
★ 四个解药：① 只在**终态**读  ② 配一个**已知必然发生**的对照计数
           ③ **噪声地板**对照  ④ **一字不变检验**（修完现象完全没变 ⇒ 改错地方了，立刻回头）
```

### 3.5 ⚠️ 顺手发现并代修的主线阻断 bug

`--seed` 发出的 `actor_rollout_ref.rollout.val_kwargs.seed` **是不存在的键**
⇒ Hydra 拒绝 ⇒ **launch_rl 100% 启动即死**。已删（`data.seed` 保留）。
★ **新增的 config 覆盖必须起一次跑才算落地。**

## 4 · ★ 第一件事

**队列见 [`00-INFRA-HANDOFF §5.1`](00-INFRA-HANDOFF.md)。** 一句话版：

```
1 🔴 5120 下重测 lr 1e-4          4 卡 ~40 min —— R-1 的前提没了，重测前不许动 reward
2 🔴 E26 集成：脱 Ray 最小复现     秒级迭代打 dtype（⛔ 别再用"起训练"当调试循环）
3 🟠 KL 多种子（盯 fabricated_safety_line_cap）→ 过了才改默认值
4 🟠 token vs sequence 多种子      判据用 defer 而不是总分
```

**开跑模板**（三条常驻判据靠这两个环境变量产出）：

```bash
SYNCOPATE_SYNC_PAYLOAD=1 SYNCOPATE_SYNC_REF=75.377708 \
SYNCOPATE_SYNC_WATCH="model.layers.0.self_attn.q_proj.base_layer.weight" \
  .venv/bin/python -m syncopate.train.launch_rl --model models/Qwen3-4B-sft-v13-e1 \
  --lora-rank 32 --mode fully_async --trainer-gpus 3 --rollout-gpus 1 --steps 60 --sync-every 4 ...
```

★ **每跑必查四条**（前三条老的，第四条 2026-08-19 新增）：
```
① [lora-probe]   list_loras() 非空（step≥1）
② [sync-payload] 第 2 次同步起含 lora_ > 0
③ rollout_corr/kl 每次同步后回落到 ~3.4e-4
④ 🆕 prompt_length/clip_ratio **必须是 0.0000** —— 夜跑 19 项全在 1.0 下跑的（E20 §7.12）
⚠️ 判据一律**只在终态读**（退出码 / timing 首行），并配一个已知必然发生的对照计数
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

19. 🆕🔴🔴 **判据只在「终态」读，而且要配一个对照计数。**
   「还没发生」和「不会发生」在日志里**长得一模一样**。今天为此误判三次：读早了、
   读的是上一次的旧日志（mtime 是唯一证据）、grep 匹配到了安装横幅里含的判据行文案。
   ⇒ 终态 = 进程退出码 / `timing_s/step` 首行出现；对照计数 = 一个**已知必然发生**的打印
   （如 `组构成`、`判据A`）——「扫描 0 行」只有在对照计数 >0 时才等于「没找到问题」。
20. 🆕🔴 **改完之后现象「一字不变」⇒ 改错地方了，立刻回头。**
   不是"方向对但不够"。今天有两次顺着一个看似合理的假设连改两轮，而报错一个字符都没变。
21. 🆕🔴🔴 **`setup_worker` 里绝对不能 import 会碰 CUDA 的模块。**
   它靠 Ray 的 `worker_process_setup_hook` 执行，**早于 Ray 给进程分卡** ⇒ CUDA 上下文
   绑死 device 0 ⇒ `NCCL WARN Duplicate GPU detected`，三 rank 挤一张卡、训练起不来。
   实测：`verl.workers.engine.fsdp.transformer_impl` 会初始化 CUDA；
   `monkey_patch` / `modeling_utils` / `prefix_grouper` / `workers.utils.padding` **不会**。
   ⇒ 加 import 前先跑一行 `torch.cuda.is_initialized()` 对照。
22. 🆕 **多 actor 架构里，补丁要问「装在哪个进程」。**
   `_compute_old_log_prob` 跑在 **trainer driver**，模型构建跑在 **WorkerDict** ——
   装错进程的表现是「补丁装了、也打印了，但现象一字不变」。
23. 🆕 **接缝错误的报错位置几乎总在别人家里**（今天 5/13）：返回值顺序反了报在 verl 的
   postprocess、熵传 None 报在 padding.py、dtype 不一致报在 torch 的 Adam。
   ⇒ 有效手段只有两个：**接缝处自己打判据行** · **把问题降维成秒级观测**。
   ⛔ 「看报错位置推根因」今天十三次里错了至少五次。

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
  SYNCOPATE_PREFIX_GROUPER=1     🆕 **默认关**·E26 打包前向（⚠️ 真实训练集成未完成，见 E26 §6.3；
                                 里面暂留有 Adam 扫描的诊断代码，解决后要清）
  SYNCOPATE_SYNC_PAYLOAD=1       每次同步打「张量数/字节/lora_ 个数/盯住层 ‖W‖」+ list_loras()
  SYNCOPATE_OPT_STEP_PROBE=1     **真实**优化器更新次数（不是 fit step / dump 数 / param_version）
  SYNCOPATE_DDP_PROBE=1          三个 rank 的梯度范数（E21 的判据）
scripts/probe_weight_sync_payload.py 🆕 ★ 0-B：权重同步**推的是什么**（离线复现发送侧分支）
  ⇒ 配套运行时探针 `SYNCOPATE_SYNC_PAYLOAD=1`（在 verl_patches 里，判据带 SYNCOPATE_SYNC_REF）
scripts/probe_sm120_fp8.py / probe_fp8_real_shapes.py / probe_fp8_logprob_error.py  🆕 FP8 三件套
scripts/probe_moe_4bit_load.py      🆕 4bit MoE 加载路径（碎片）
scripts/check_prefix_grouper_equivalence.py 🆕 ★ PrefixGrouper 等价性（**fp32 + 噪声地板**，
  ⛔ 不要在 bf16 下判等价性：噪声地板 mean 1.28e-2 > 要抓的误差）
scripts/probe_prefix_grouper_speed.py      🆕 四臂吞吐（A 现状FA2 / B 同后端SDPA / C 打包SDPA / D 打包FA2）
scripts/probe_trainer_feed.py              🆕 E25：micro_batch × gradient_checkpointing（自带保真度自检）
docs/upstream/verl-prefix-grouper-not-wired/repro_prefix_grouper_wiring.py 🆕 零 GPU 三判据复现
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
