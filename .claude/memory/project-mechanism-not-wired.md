---
name: project-mechanism-not-wired
description: 这个项目最反复出现的失效形状是「机制建好了但没接上」，测试抓不到，只能靠日志判据和硬失败
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-20T03:41:46.081Z
---

**「我建了一个机制，然后假设它会自动生效。」** 这是 Syncopate 反复栽的同一个形状，
到 2026-08-14 已经累计十五次以上（cap 监视的工具不在菜单里、稀疏格子被取模削成 0 条、
verl 把日志级别硬编码成 WARN 导致统计看不到、动态分池的 monkeypatch 对另一个 trainer
无效、staleness 修正在 bypass 模式下不产出 ESS 指标……）。

**Why**：**测试抓不到这类问题** —— 单元测试验的是"机制本身对不对"，
不是"机制有没有被接上"。只有实跑才暴露。

**How to apply**：
1. **判据永远是日志，不是代码。** 每个靠 monkeypatch / 环境变量 / 配置生效的机制，
   都要在启动时打一行；`[pool] 动态分池启用` / `[rl] 模式=...` / `[verl-patch] ...`
   **没有这行就是没生效**。review 代码没用，去看那行日志。
2. **硬失败，不自动降级。** `--mode` 在卡数不够时直接报错而不是退回单卡 ——
   静默降级的表现是「跑起来了、但测的根本不是你以为的东西」。
3. **需要精确控制的量，别靠取模碰运气。**
4. 上游也会犯：verl 的 `train_batch_size` 在 fully_async 里强制为 0（而不是忽略），
   就是**逼你发现"你以为在控制的东西其实没接上"** —— 这是好设计，值得抄。

---

## ★ 2026-08-14 一天里撞到的三个新变种（都比"忘了接"更隐蔽）

**① 判据行打出来了 ≠ 补丁在需要它的进程里生效 —— 作用域。**
`verl_patches` 的 P2 打在 driver，`[verl-patch]` 那行照常打印，而断言在
`WorkerDict` 这个 **Ray actor 进程**里触发。修法是 `runtime_env.worker_process_setup_hook`。
⇒ **判据要连作用域一起看：是哪个 pid 打的这行？**

**② 判据行不能写断言，只能写观测。**
`main_ppo_pool` 打的是 `[pool] ⚠️ fully_async 不调 create_rl_sampler ⇒ 本轮不生效`。
**这句话本身是错的**：`fully_async_rollouter.py:464` 调了，import 还写在函数体内
（最适合 monkeypatch）。真因仍是作用域——rollouter 在另一个 worker 进程里。
⇒ 一行陈述"为什么不行"的日志，**把"其实能行、只是没接上"整个盖住了**，而且它长得像个合格判据。
⇒ 凡是说「上游不支持 X」，先查三层：**① 配置项在不在 ② 代码路径调不调 ③ 它在哪个进程里跑。**

**③ 短冒烟证明不了长跑的稳定性 —— 时间维度。**
`--sync-every 4` ⇒ 第一次 bucketed 权重同步在第 4 步，而冒烟只跑 3 步；
真正炸的是**第二次**同步（第 8 步）：rollout 卡可用显存随生成推进变紧，
第一次够用、第二次差 0.09 GB。⇒ 冒烟至少覆盖**两个**同步周期。

---

## ★ 2026-08-16 第四种形态：保底写在了**另一条代码分支**里

M8 新增的 POL / CONF 两个模板，**SFT 桶里各 0 条**。

根因：`split.py` 的稀有行为保底**只写在 `dead_grid` 分支里**，而 v12 走的是
`difficulty_proxy` 分支（v11 走 dead_grid，所以从没暴露）。后者按
(难度, gold 链长) 降序取前 20%，POL 的 gold 只有「一次检索 + 终答」，链最短
⇒ 整个模板被挤出桶外。

⚠️ **最刺的地方：保底机制早就写好了，注释里连"为什么需要它"都写清了**
（「defer 只有 4 条 ⇒ SFT 后从 77% 崩到 36%」）—— 它只是待在另一条 if 分支里。

**后果不是"少学一点"**：p=0 的格子 RL 永远够不着 ⇒ 没进 SFT 的模板，
冻结 EVAL 上必然量出"没学会" —— **而这个结论和 RAG 实现好坏完全无关**，
会把一个分桶 bug 误判成 RAG 设计问题，然后去改根本没错的地方。

⇒ **判据：凡是"保护性"的逻辑，先问它是不是所有代码路径共用一份。**
修法是提成一个函数（`_apply_sft_floors`），两条分支都调。

★ 但共用的是**函数**，不是**策略**：判据是「说不说得出为什么这里不需要保护」——
dead_grid 说得出（这个模板在 EVAL 上是活的），difficulty_proxy 一个理由都说不出。
第一版两条分支都开，当场撞坏一条既有测试（活格被硬拽进 SFT 桶）。

★★ **附带一条：保护参数的"上限"不能一刀切。** 我给保底统一加了「最多吃一半」，
**直接把 `defer` 从 12 削到 8** —— 那正是当年 defer 77%→36% 的那条线。
⇒ **一个参数同时服务两个目的时，先问它们要的是不是一回事**：
要**饱和**的（行为，学不会就是学不会）不能设上限；只要**在场**的（模板/结局）
必须设，否则把 RL 的口粮全吃光。

---

## ★★★ 2026-08-17 第五形态：**修坑的机制自己变成了更深的坑，而且静默**

分卡模式（one_step_off / fully_async）下 **3 个 trainer rank 全挤在物理 GPU0**，
第一次权重同步 OOM。colocate 完全正常。

根因是**我们自己加的 `worker_process_setup_hook`** —— 它当初正是为了修第①种形态
（「补丁打在 driver、断言在 worker」的作用域问题）才加的：

    Ray worker 进程刚起来 → 跑我们的钩子 → 钩子 import verl.utils.fsdp_utils
                          → **CUDA 设备枚举被固化**
    Ray 之后才给这个 actor 设 CUDA_VISIBLE_DEVICES
                          → 可见数量变了、**设备→物理卡的映射不变**
                          → 每个 worker 的 cuda:0 都指向物理 GPU0

最小复现（两行就能证伪，我却先绕了三圈）：

    设 CVD=2 再 import torch                        → 0xa1 ✅（物理 GPU2）
    先 setup_worker() 再设 CVD=2                     → 0x21 🔴（物理 GPU0）

⇒ **新纪律：进程启动钩子里只许做纯 Python 的事。任何可能碰 CUDA 的 import
都要延迟到设备确定之后**（修法：`_defer_until_imported`，用 meta_path finder
在 verl 自己 import 那一刻才打补丁；判据行照打，测试 3 条仍绿）。

### 这一轮里三个「判据本身不可信」的教训

1. **`/proc/<pid>/environ` 是 exec 时快照**，Ray 在运行时改的 `os.environ` 不反映进去。
   我据此断定「Ray 没设 CVD」——**错的**，Ray 设了（探针在进程内打出 CVD=0/2/3）。
   ⇒ 进程内的环境要在**进程内、运行时**打，别从外面读。
2. **`torch.cuda.is_initialized()` 是 False 不代表 CUDA 没被摸过** ——
   驱动层的枚举已经定了，PyTorch 的上下文还没建。这个判据漏掉了真正的损害。
3. **A/B 才定的位**：colocate 正常 × 分卡异常 ⇒ 差别只有那个钩子。
   在此之前我试过 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`，
   它让 trainer 分对了卡（所以"看起来像修好了"），但把 rollout 侧撞出
   `NCCL Duplicate GPU` —— **绕过症状的修法会把你带离根因。**

★ 同一轮还抓到一条真·没接上：**`--weight-sync-bucket-mb` 在 colocate 下被静默忽略**
（override 只写在分卡分支里，注释还断言「分卡模式独有的开销」——那句话是错的，
colocate 一样走 NCCL checkpoint engine）⇒ 吃 verl 默认 2048 MB，第一次同步就 OOM。

---

## ★★★ 2026-08-17 第六形态：**字段全程有效，只是没有消费者**

M9 符合性审计（`docs/syncopate/11-runtime-acceptance.md`）：`automation_tier='C'`
的 run **不走审批，写动作直接执行、run succeeded**。而 §3 里 C 档 = 不可逆/代价高。

链路是这样断的：

    API   pattern="^[ABCD]$"  校验 ✅
    schema CHECK (... IN ('A','B','C','D'))  约束 ✅
    落库   agent_runs.automation_tier = 'C'   ✅
    claim_run 的 RETURNING **没有这一列**      ← 断在这
    worker  DecisionContext(automation_tier=None)  硬编码
    ⇒ tier_c 触发器**在真实路径上永远不可达**

⚠️ **前五种形态至少有个"接"的动作**（接错了作用域、接在另一条分支、接的时机不对）；
这一种**连接的动作都不存在**，而且**两头的关卡都做得一丝不苟** —— 所以它长得特别像做完了。

⇒ **判据：一个字段被收下、校验、落库之后，去问"谁读它"。** 只有写没有读 = 没接上。
⇒ 同一轮还查出：六个降级触发器只有 2 个有生产者、`checkpoints`/`model_calls`
两张表**建了但没有写入路径**、`outcome_*` 除 schema 外零引用 —— **全是同一形态。**

★ 「不可达」这件事**没有行为可测** —— 你测不出一个永远不会发生的事件。
所以判据只能是**源码扫描**（`DecisionContext` 每个字段在 worker 里有没有被赋值），
不是行为测试。这不是偷懒，是这类缺陷唯一可测的形状。

★★ 配套手法：缺口用 **`xfail(strict=True)`** 钉住，不用注释。
现在不满足所以标 xfail；**修好那天会变成 XPASS = 失败**，逼下一个人回来翻标记。
⇒ **让缺口自己会喊**，这是 [[observed-needs-an-owner]] 在代码里的落法。

---

## ★★ 2026-08-17 第七形态：**一个默认值/判据看起来正常，实际指向另一件事**（且不报错）

同一天撞了三次，形状完全一样：

```
① 评测的基座默认成裸模型   RL 的 LoRA 是在 **SFT 合并模型**上训的，
                          eval 脚本 MODEL 默认 models/Qwen3-4B ⇒ 把 SFT 全丢了再评
                          症状：不报错，**分数全线暴跌**（CLAR 0.844→0.000）
                          看着像"RL 把模型训坏了" —— 差点写进结论
② `until [ -f 输出文件 ]`   那文件**上一版就存在** ⇒ 循环立刻通过
                          ⇒ 第二批评测当场起来抢同样四张卡、全部 OOM
③ 守卫匹配错进程名         `launch_rl` 会 exec 成 `main_ppo_pool`
                          ⇒ `pgrep -f train.launch_rl` 找不到 ⇒ 守卫立刻退出、形同虚设
```

⇒ **判据：凡是"默认值"和"存在性检查"，问一句「它指的是不是我以为的那个东西」。**
   前六种形态是"机制没接上"，这一种是**接上了、但接到了另一根线上**。

★ 配套：给错宁可报错。`eval_parallel.sh` 的基座改成 `${MODEL:?...}` 必填。

---

## ★★ 2026-08-20 第八形态：**测试全绿，但系统作为"服务"从没被起过**

runtime 195 条测试全绿、40 条验收判据核过 —— 但**第一次真的起进程跑 HTTP**
（2026-08-20）当场发现三件：① worker 没有进程入口（队列永远没有消费者）；
② `agent_loop` 写好带测试、worker 却还跑写死的三步计划；③ cancelled 各退出路径
不发终态 SSE 事件 ⇒ 客户端永远挂着（195 条进程内测试测不到"流不关"）。

⇒ **进程内测试验的是函数，不是部署形状**：入口、进程边界、连接生命周期
这些只在"真的把它起起来"时存在。⇒ 判据：每个服务组件问一句
「它的 main 在哪、谁真的起过它」。冒烟驱动 `scripts/runtime_smoke.py` 就是这次补的
（固定 query 集 + SSE 跟流带死线 —— 驱动器自己没有死线时，被它测的挂死 bug 会把它一起挂死）。

相关：[[feedback-measure-dont-infer]] [[machine-4x5090-constraints]] [[rl-step-size-is-lr-times-steps]]
[[blank-thresholds-are-not-passes]] [[clean-machine-only-gaps]] [[observed-needs-an-owner]]
[[sandbox-is-subset-of-runtime]]
