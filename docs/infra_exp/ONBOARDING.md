# infra 线 · 新窗口对接

> **每次收尾时更新这份文件**（怎么更新见文末 §7）。新窗口开场只说一句
> 「读 `docs/infra_exp/ONBOARDING.md`」就够了。
>
> 最后更新：**2026-08-19**

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
4. /MAINLINE-INFRA.md（仓库根目录）      📮 与主线往来的**唯一**文档（⛔ 铁律：禁止再写信件）
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

## 3 · 最近一轮做完了什么（2026-08-19 下午：E26 集成收口）

**一句话**：E26 的 Adam dtype 卡点用**脱 Ray 复现**（12 轮 × 30 s）定了案 —— 真身不是 dtype，
是**打包前向绕过了根 FSDP ⇒ 梯度归约竞态**（E21 形状的静默错误）；已修，冒烟四常驻判据全绿；
随后同尺子 A/B 四臂定案：**生产现状→PG 端到端 2.31×**（34.52→14.94 s/gstep）、最优 mb=8。
队列已按 Chaoyu 重排（lr 1e-4 降级——主因是步数太少不是 lr 低）。

### 3.1 ★★★ 根因与修法（全文 [E26 §6.3–6.4](E26-prefix-grouper.md)）

```
根因   _pg_forward 直接调基座 ⇒ 损失不流经根 FSDP 的输出 ⇒ 根的 pre-backward 永不触发
       ⇒ 归约的 final callback 没人排队 ⇒ fp32 all-reduce 挂在没人等的 CUDA stream 上
       ⇒ optimizer 读竞态快照：有时没归约（**不报错**！），有时抓到半途 fp32（Adam 报错）
显形   adapter 同步启动时的一次 state_dict() 让真根完成 lazy_init、子单元不再自封根
       ⇒ 最危险的格子是「state_dict × 单 micro-batch」：**梯度静默不同步、零报错**
修法   前向走根 FSDP（CausalLM forward 临时换回 HF 原版 + logits_to_keep=1）
       + hook 捕获 hidden + log_probs 加 0×根输出的锚
验收   脱 Ray：三 rank 梯度和与健康路径**逐位相同** · log_probs sum 分毫未动
       真实跑：判据A×12 · ddp-probe 跨 rank 逐位同 · list_loras=[123] · kl 4.4e-4 · clip_ratio 0.0
```

★ 方法论：**16 轮起训练没定位到的，秒级复现 12 轮定案** —— 单因素全不复现时，
用「全开组合 + 留一法」找**组合条件**（本例 = state_dict × accum≥2）。

### 3.2 上午那轮的三件大事（详情看各报告，此处只留指针）

```
⛔⛔ defer 崩塌翻案：是 prompt 截断不是 reward（E20 §7.12）⇒ 主线 R-1 已撤
✅  E17 KL 定案：砍 KL 省 15.4%、任务分无差异（E17 §9）⚠️ 默认值未改，等多种子
✅  E25 证伪「trainer 没喂饱」；E26 §7 判据失效七形状（D1/D2 + E1–E5 + 四解药）
```

### 3.3 ⚠️ 顺手修的三个启动类 bug（都属「默认值/文案指向另一件事」族）

```
① launch_rl --train-file/--val-file 默认写死 v3（目录已不存在）⇒ 已改成跟 DATA_VERSION 走
② launch_rl --help 崩溃：一处 help 文案里裸 %（argparse 会做 %-格式化）⇒ 已修
③ （上午修的）--seed 发不存在的键 val_kwargs.seed ⇒ 启动即死 ⇒ 已删
```

## 4 · ★ 第一件事

**队列见 [`00-INFRA-HANDOFF §5.1`](00-INFRA-HANDOFF.md)。** 一句话版：

```
0 ✅ E26 同尺子吞吐 A/B 已定（08-19 13:04）：生产现状→PG **2.31×**、PG 净效果 2.23×、
  mb1→mb8 仅 +3.8%、gen 占比 12%→26%（E26 §6.6，判据 logs/queue_e26ab/AB.done）
1 🟡 E26 B5 任务尺子（≥60 步 + 冻结 EVAL 配对）→ 过了才有资格谈 PG 默认开
2 🟠 KL 多种子（盯 fabricated_safety_line_cap）→ 过了才改默认值
3 🟠 token vs sequence 多种子      判据用 defer 而不是总分
4 🆕 E20 原因②「步数太少」正式验证  固定 epoch 而非步数、≥400 步、停由判据定（主线观点）
🔵 lr 1e-4 @5120 已降级为可选上限基线（Chaoyu 2026-08-19：真实训练不用 1e-4；
   脚本备好 scripts/run_e20h_lr1e4_5120.sh，不挡任何人）
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
24. 🆕🔴🔴 **FSDP1 下绝不能绕过根模块的 forward 去调子模块**（E26 §6.3，脱 Ray 实测）。
   归约/精度收尾由**根输出张量上的 pre-backward hook** 驱动；损失不流经根输出 ⇒ 整套
   post-backward 机制静默不跑 ⇒ **梯度可能不跨 rank 归约且不报错**（E21 形状）。
   更阴的是它有时"能跑对"：根没被 lazy_init 时子单元自封根 —— 而任何一次 state_dict()
   都会把这个巧合打破。⇒ 判据：三 rank 梯度范数逐位相同（SYNCOPATE_DDP_PROBE=1）。
25. 🆕 **单因素复现不了 ≠ 复现不了**：E26 那个竞态要「state_dict × ≥2 micro-batch」同时
   成立才显形。⇒ 先"全因素照抄真实跑"求复现，再**留一法**收敛到最小组合 —— 每轮 30 s。
26. 🆕 **盯长跑不能只 grep 训练中的信号**：启动即死时所有 pattern 都不出现，"安静"和
   "还在跑"长得一样（坑 19 的孪生）。⇒ 监视器必须带**进程退出兜底**（今天为此瞎等一轮）。

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
  SYNCOPATE_PREFIX_GROUPER=1     🆕 **默认关**·E26 打包前向（✅ 集成已通 2026-08-19，E26 §6.3–6.4；
                                 欠同尺子吞吐 A/B 与 B5，转正前默认保持关）
scripts/repro_pg_dtype.py       🆕 ★ E26 竞态的脱 Ray 复现（三臂 × 5 因素开关 × 4 探针，单轮 ~30 s；
                                 也是「FSDP 根旁路 ⇒ 归约竞态」的最小证据 + 修复回归测试）
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
