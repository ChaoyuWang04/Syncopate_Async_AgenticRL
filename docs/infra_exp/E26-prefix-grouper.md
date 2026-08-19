# E26 · PrefixGrouper：题面只算一次 —— **3.96×**（三次前向）

> 状态：✅ 等价性已过（fp32 逐位）　✅ 真实训练集成已通（§6.3–6.4）　✅ **同尺子吞吐 A/B 已定**（§6.6）　⬜ 未过任务尺子（B5）
> ★ **生产现状 → PG 端到端 2.31×**（34.52 → 14.94 s/global step，覆盖数实测）；PG 净效果 2.23×；
> 微基准 3.96× 兑现 ~70%。⛔ 早前"2.12×"作废（单行 timing 分母没实测过，§6.4）
> 尺子 `scripts/probe_prefix_grouper_speed.py` · 等价性 `scripts/check_prefix_grouper_equivalence.py`
> 补丁 `syncopate/train/verl_patches.py::_patch_prefix_grouper`（`SYNCOPATE_PREFIX_GROUPER=1`，默认关）

## 0 · 结论卡片

| | |
|---|---|
| **需求从哪来** | E25 证伪「trainer 没喂饱」⇒ 只剩「让它少算」。而 87% 的 token 是一组 8 条共享的题面 |
| **结果（微基准）** | **三次前向 3.96×**（10.211 s → 2.577 s），显存 11.88 → **12.31 GB**（+3.6%） |
| **结果（端到端实测）** | ✅ **2.31×**（34.52 → 14.94 s/gstep，§6.6）· PG 净效果 2.23× · 最优 mb=8（mb16 慢 5.7%） |
| **等价性** | ✅ fp32 + 噪声地板对照，四个配置（含组大小 8）全过；小配置**逐位相同** |
| **纸面上界** | 4.12×（token 数 38,800 → 9,428）⇒ **兑现了 96%** |

## 1 · 三臂 + 一臂（拆开"打包的收益"和"后端的代价"）

```
A 生产现状（逐条 ×8 · FA2）              10.211 s   11.88 GB
B 同后端对照（逐条 ×8 · SDPA）           35.396 s   22.02 GB   ← 只用来解释成因
C 打包（PrefixGrouper · SDPA）            8.400 s   25.84 GB
D 打包（PrefixGrouper · FA2）             2.577 s   12.31 GB   ← ★ 该报的是 A→D
```

★ **中间那一跑（C）差点让我们下错结论**：修好 FA2 之前只有 SDPA 能跑通，
A→C 的净收益只有 **1.22×**，而 A→D 是 **3.96×**。
⇒ **「唯一能跑通的配置」和「最优配置」不是一回事** —— 前者会让一个 4× 的优化看起来像 1.2×。

## 2 · 四处修复（三处是 verl 欠的，一处是我们独有的发现）

```
① apply_monkey_patch 不传 use_prefix_grouper ⇒ 注意力 patch 永不执行
② forward_micro_batch_with_prefix_grouper 零调用者 ⇒ 打包前向是死代码
③ 🔴 **suffix_mask 拿到的是梯度掩码** ⇒ 多轮下工具 observation 被静默删掉
   （verl 自带 tool_agent_loop.py:400 也是这个语义；上游 #7202 一字未改）
④ uid 在 non_tensor_batch，到不了 engine ⇒ 改用「同组 prompt 逐位相同」推组边界 + 断言
⑤ 🔴🔴 **因果掩码对齐**（本项目独有，不在 #7202 也不在上游情报里）
⑥ 🔴🔴 **position_ids 透传** ⇒ FA2 把它当变长序列的标记 ⇒ **非法访存**
```

### 2.1 ⑤ 因果对齐

PrefixGrouper 把 2D padding mask 当第 4 个位置参数传给 attn_func；suffix 子调用是
**q 长 R、k 长 P+R**，需要**右下对齐**的因果掩码。

```
eager / sdpa        构造**左上对齐** ⇒ 结构性错误（fp32 实测差 21 万倍）
flash_attention_2   2D mask 长度 ≠ query 长度 ⇒ 走 _upad_input 变长路径 ⇒ 错位
✅ 正解              **不传那个 mask**，让后端走纯 causal
                    实测 flash_attn(causal=True) **就是右下对齐**（与显式右下参考逐位相同）
```

### 2.2 ⑥ position_ids（FA2 崩溃的真正原因）

子调用收到的是**整条 grouped 序列**的 `position_ids`（长 9,428），而子调用形状是 `B=8, Lq=654`
⇒ HF 的 FA2 用它探测打包序列并切到变长路径 ⇒ **越界访存**。
⚠️ **`flash_attn_func` 本身在所有这些形状上都正常**（单独复现过五组）——
崩溃点报在 `batch_repeat_cat` 里，那是 CUDA 异步错误的**假位置**。
⇒ 旋转位置编码在进入注意力函数**之前**已经加过，丢掉是安全的。

## 3 · ⛔ 推翻了什么（一天里被同一个形状骗了五次）

每一次都有一个**能解释现象的机制**，每一次都是错的：

```
❌ verl 的 wrapper 把 attention_mask 吞掉了     —— PrefixGrouper 自己会传，wrapper 形状是对的
❌ 输出布局转置错了                              —— 文档明写：进 [B,H,T,D]、出 [B,T,H,D]，本来就不同
❌ position_ids 导致**数值**错误                 —— 去掉后数字一模一样（它导致的是**崩溃**，不是错值）
❌ prefix_grouper 包本身算错                     —— fp64 玩具测试 0.000e+00
❌ flash-attn 内核在这个形状上有 bug             —— 单独复现五组形状全部正常
```

## 4 · ★★ 最贵的一课：**一个判据可以严格到无法执行**

我最初写的是「等价判据 = logprob 逐位相同」。实测这个模型在 **bf16** 下的噪声地板
（同样内容、只把两条序列换个批组成）就有 **mean 1.28e-2 / max 1.0** ——
**比我要抓的错误还大。**

⇒ 我因此在噪声里追了三轮根因，每一轮都提出一个自洽的错误解释。
⇒ 换成 **fp32 + 噪声地板对照** 之后，答案一次就出来了。

★ **判据太松 ⇒ 错的东西通过；判据太严 ⇒ 对的东西被毙掉。两种都是「判据没写对」。**
★ 而真正把我救出来的动作只有一个：**加一个噪声地板对照**
（同样的数据、同样的数学、只改一个无关变量），它一次性回答了「我追的差异有没有超出噪声」。

## 5 · 还欠什么

```
✅ 在真实训练里跑通（2026-08-19，§6.3 根因 + §6.4 冒烟）
⬜ 同尺子吞吐 A/B（§6.4 的 2.12× 是 n=1 短跑，只定方向）
⬜ 过 B5 任务尺子（纪律：只报吞吐不报任务精度不算完成）
⬜ 组大小 ≠ 8、变长 response 下的显存与稳定性（冒烟里出现过组 [8] 与孤立混合，未系统测）
```
⚠️ **3.96× 是微基准数，不是端到端数。** 「约 2.98×」是外推；实测方向数 2.12×（§6.4，未转正）。

---

## 6 · 🔴 真实训练集成（2026-08-19 下午）：**微基准 ≠ 端到端** —— 16 次尝试、13 处接线

> 状态：**未完成**。打包路径已在真实 fully_async 里被走到（判据A 打印、组构成 `[8]`、
> 前缀只算一次、只投影回答位），前向/损失/反向全通，**卡在优化器**（§6.3）。

### 6.1 13 处接线问题全录（按遇到顺序；★ = 被判据主动抓住，其余靠崩溃暴露）

| # | 问题 | 报错位置 vs 根因 | 处置 |
|---|---|---|---|
| 1 | （主线）`val_kwargs.seed` 覆盖不存在的键 | Hydra 启动即死，100% 触发 | ✅ 删那一行（`data.seed` 保留） |
| 2 | `setup_worker` 里 import `transformer_impl` ⇒ **CUDA 提前初始化**，早于 Ray 分卡 | NCCL `Duplicate GPU`（三 rank 挤一张卡） | ✅ setup 期只改轻量 `monkey_patch`；重模块推迟到 `apply_monkey_patch` 被调时（届时已分卡） |
| 3 | 第一次修惰性导入**改错模块**（`prefix_grouper` 不碰 CUDA，`transformer_impl` 才碰） | 现象不变 | ✅ `torch.cuda.is_initialized()` 逐模块实测，一次定位 |
| 4 | `forward_step` 返回 `(output, loss)`，verl 契约是 **`(loss, output)`** | `Tensor.__contains__` 炸在 verl postprocess | ✅ 换序 |
| 5 | `micro_batch=1` ⇒ 无组可分 | ★ 判据行「投影 **31** 个位置」明显不对 | ✅ `--micro-batch-size = rollout_n` |
| 6 | verl 的 `use_prefix_grouper` 组感知划分**本身是坏的**（nested 假设） | `padding.py` `'NoneType'…is_nested` | ✅ **不开它**（#7202 修的正是这条，已被上游关闭） |
| 7 | `trainer.balance_batch` 按长度重排 ⇒ 组被打散 | ★ 断言「组大小 `[2,2,2,1,1]`」 | ✅ `balance_batch=False` |
| 8 | 我自己的断言「组大小必须相等」**过度保守**（合法但低效 ≠ 错） | 误杀合法情形 | ✅ 改为一次性「组构成/碎片化」报告 |
| 9 | 前缀掩码用 `ne(pad_token_id)`（pad id 猜错）⇒ **padding 被打包** | 判据行 grouped−投影 = 5120 = 补齐宽度 | ✅ 与 response 同口径：用 `attention_mask` 前半段 |
| 10 | 熵传 `None`，verl **无条件**对它做 padding | `padding.py` `is_nested` | ✅ 始终带出熵（`FusedLinearForPPO` 本来就算了） |
| 11 | 旁路 `no_padding_2_padding` **装错进程**（WorkerDict ≠ trainer driver） | 现象一字不变 | ✅ 移到 setup 期 patch 源模块（实测该模块不碰 CUDA） |
| 12 | verl 聚合只认 **jagged** 格式（稠密 `[bsz,R]` 被 `unbind` 后长度全错） | `assert sequence_offsets[-1]==values.shape[0]` | ✅ **按下游契约产出**：每序列全长向量、log_probs 左移一位放 `[L−R−1, L−1)`；旁路删除 |
| 13 | 优化器 dtype：`expected BFloat16 … got float` | `torch/optim/adam.py`（离根因最远的一次：真身是**绕过根 FSDP ⇒ 归约竞态**） | ✅ **已解**，§6.3 |

⇒ 分布：**主线 1 · verl 缺陷 3（#2/#6/#7）· 我们接错 9**。
⇒ **数学部分（等价性 / 因果对齐 / 掩码语义 / response-only 投影）全程零改动、零错误。**
⇒ 13 处里**只有 2 处**（#5/#7）是被判据主动抓住的，其余靠崩溃 —— 如果它们碰巧不崩，
   得到的就是一个跑得飞快、结果全错的训练。**这就是等价性判据必须排在吞吐之前的原因。**

### 6.2 ★★ 报错位置与根因的距离（5/13 报在别人家里）

`#4`→verl postprocess · `#10`→verl padding · `#2`→NCCL · `#13`→torch Adam ·
（微基准阶段的 position_ids →PG 的 `batch_repeat_cat`，CUDA 异步的假位置）。
**没有一次报在我们改的那几行上。** 接缝错误的形状：上游产出「形状对、语义错」的东西，
下游照单全收继续走，直到某层碰上处理不了的类型/长度/dtype 才炸 —— 中间没有契约检查。
⇒ 实测有效的手段只有两个：**接缝处自己打判据行** · **把问题降维成秒级观测**。

### 6.3 ✅ 已解（2026-08-19）：根因是**绕过了根 FSDP 的 forward**，dtype 只是竞态的一种表现

> 复现与全部证据：`scripts/repro_pg_dtype.py`（脱 Ray，单轮 ~30 s；三臂 × 5 因素开关 × 4 个探针）。
> **16 轮起训练没定位到的，秒级复现 12 轮定案。**

**当时的卡点记录**（保留原文，弯路也是证据）：

```
已确认   对照跑（补丁关）跑通：step 119.43 s / update_actor 59.72 s ⇒ 是我们的路径引入的
         ⚠️ 2026-08-19 晚更正：该对照实测是 **mb=1**（outputs/09-17-42 的 overrides），
         不是"同配置"——PG 开必须 mb=8、关了只能回 mb=1（E25：gc=on 时 mb=4 就 OOM）
         ⇒ 119.43 vs 56.44 的口径是「生产现状(mb1) vs PG(mb8)」，恰好是该报的那组
已排除   「把 log_probs 转 bf16」—— 方向反了：FusedLinearForPPO **契约就是返回 fp32**
         （源码 assert token_log_probs.dtype == torch.float32）；且改后现象一字不变
诊断三连败  钩子只挂第一个参数（E4 覆盖面）· patch 了 AdamW 而 verl 用 Adam（E5 打错目标）
           · 改对目标后扫描仍 0 行（原因未明，判据A/组构成同期各打 2 次）
```

**根因（因果链，每一环有探针实测）**：

```
_pg_forward 直接调 _base_model(model)(...) ⇒ 绕过根 FSDP 模块
⇒ FSDP1 把 pre-backward hook 注册在**根 forward 的输出张量**上，损失不流经它 ⇒ 永不触发
⇒ `_post_backward_final_callback` 只由根的 pre-backward 排队 ⇒ 没人排队（探针实测：排队 0 次）
⇒ 归约（fp32 all-reduce）+ 转回 bf16 全部跑在 `_post_backward_stream` 上，默认流从不等它
⇒ **optimizer 读到竞态快照**：
     有时没归约 —— 三 rank 梯度和各不相同（E21 形状的静默错误，**不报错**）
     有时抓到半途的 fp32 —— Adam `expected BFloat16 for 'end'`（幸运的金丝雀）
```

**为什么此前「看起来能跑」**：根没被 lazy_init 时，被直接调用的子单元会**各自自封 `_is_root`**、
各自排队 final callback ⇒ 碰巧正确。而 **adapter 同步在启动时的一次 `state_dict()` 会把真根
init 掉**、子单元从此不再自封 ⇒ 真实跑里每次更新都在竞态里。
最小充分条件实测 = `state_dict 一次 × ≥2 个 micro-batch`（正是真实跑的形状）。
⚠️ **最危险的格子是 `state_dict × 1 个 micro-batch`：不报错、梯度静默不同步。**

**修法**（都在 `_patch_prefix_grouper` 里，三件一起）：

```
① 前向走**根 FSDP 模块**；CausalLM 的 class forward 临时换回 HF 原版
   （verl fused 版不透传 kwargs、prefix_grouper 会被吞；HF 原版 logits_to_keep=1 只投影 1 个位置）
② hidden 用 forward hook 从基座捕获（仍不用 output_hidden_states）
③ `log_probs += 0×根输出` 的锚 ⇒ 根的 pre-backward 在 backward 一开始就触发
```

**修复验收（脱 Ray）**：最忠实组合（tie+rmpad+varlen+accum2+statedict、3 rank）下：
final callback 每次 backward 排队/执行各 1 · 三 rank 梯度和**与健康路径逐位相同**
（0.08351319283246994）· dtype 全 bf16 · AdamW OK · log_probs sum 分毫未动
（−1789.591431 —— 修的是接线，没动数学）。真实训练验收见 §6.4。

⚠️ 复现脚本自己也踩过一次「覆盖被顶掉」：补丁闭包的 `_pending` 在第一次 forward_step
调用时重赋 `pg_forward`，把试验臂的覆盖静默顶回去 ⇒ **前四轮 rootfwd 臂结果全部作废**。
解法 = 覆盖后加 `_ran` 断言（解药②「对照计数」的又一次兑现）。

### 6.4 ✅ 真实训练冒烟（2026-08-19 12:03，fully_async 3+1，与崩溃跑同配置）

> 日志 `logs/e26/pg_fix_smoke.log` · ckpt 已删（探针短跑），`dispatched.jsonl`/`rollout_dumps` 保留

```
判据A     [prefix-grouper] 打包前向已生效 ×12（组构成 [8]，真实题面 grouped (1,4181)→投影 248）
崩溃点    上次必崩的 optimizer step：通过（expected dtype / RuntimeError 0 次）
梯度同步  ddp-probe：rank0 与 rank2 梯度范数逐位相同（2.448827e-03）—— E21 判据在真实跑复验
四常驻    ① lora-probe step≥1 list_loras()=[123] ✅ ② 第 2 次同步 504 个 lora_ / 252 MiB ✅
          ③ rollout_corr/kl 4.42e-4（≈3.4e-4 地板）✅ ④ prompt_length/clip_ratio 0.0 ✅
```

**第一组端到端数字** —— ⛔⛔ **2026-08-19 晚作废**：ctrl_off 与冒烟**各只有 1 行 timing**，
解析器差分不了覆盖数、按默认 1 计 ⇒ **119.43 / 56.44 两个分母都没被实测过**（坑 5 的变体：
「每行覆盖几个 global step」单行日志根本答不了）⇒ 由它们算的 2.12×/2.87× **不可引用**。

✅ 正式数字来自同尺子 A/B（`scripts/run_e26_ab.sh`，20 global steps/臂 ⇒ 5 行 timing，
覆盖数**实测**=4）：结果见 §6.6。已知一角：**on_mb8 = 14.94 s/global step**（update_actor 6.69 · gen 3.91）。

⚠️ 口径说明一：A/B 两臂 micro-batch 不同**不是混杂变量，是各自的可行最优**——
PG 必须 mb=8（组要同批），但打包去重后显存 ≈ mb2（trainer 卡实测 ~10 GB）；
关 PG 时 mb=8 会 OOM（E25：gc=on mb=4 就炸）⇒ mb=1 就是它的生产形态。

⚠️ 口径说明二（主线曾在 MAINLINE-INFRA 提出、Chaoyu 2026-08-19 裁定）：主线的采样器
改动（排除上一批）恰好落在 ON 臂与 OFF 臂之间 ⇒ 两臂抽到的题不同批。**裁定不重跑**：
采样器只改「抽到哪些题」，对吞吐对比统计等价（同数据集、20 步平均）；
⛔ 但**正确性/学习类对比不许跨这条代码边**——B5 必须两臂同代码跑。

### 6.6 ✅ 同尺子吞吐 A/B（2026-08-19 13:04，`scripts/run_e26_ab.sh`，判据 `logs/queue_e26ab/AB.done`）

三臂 × 20 global steps × seed 42 × sync-every 4；覆盖数**实测**=4（每臂 5 行 timing）；
三臂 clip_ratio 全 0、判据A on=12 / off=0、无真报错（各臂 2 条均为关闭期 resource_tracker 噪声）。

| 每 global step | **on_mb8（PG）** | off_mb8 | off_mb1（生产现状） |
|---|---|---|---|
| step | **14.94 s** | 33.26 s | 34.52 s |
| update_actor | 6.69 | 19.08 | 19.90 |
| old_log_prob | 2.33 | 6.22 | 6.24 |
| ref | 2.01 | 5.16 | 5.37 |
| gen（等样本） | 3.91（26.2%） | 4.11（12.3%） | 4.28（12.4%） |
| 三次前向占步 | **73.9%** | 91.6% | 91.3% |

```
★ 生产现状 → PG：端到端 **2.31×**（34.52 → 14.94）
★ PG 净效果（同 mb8）：**2.23×**；三次前向 2.79×（微基准 3.96× 在真实系统兑现 ~70%）
★ mb1 → mb8（不开 PG）只值 +3.8% ⇒ E25「喂饱单位是 token」再次验证；
  ⚠️ off_mb8 没有 OOM（显存 29–31/32 GB 贴顶跑完）—— E25 定长探针的 mb=4 OOM
  没有迁移到变长真实批，但这个余量薄到不能当生产配置
★ gen 占比 12% → 26%：trainer 加速后瓶颈向 rollout 移动 ⇒ 陈旧度的剂量条件第一次真正具备
```

### 6.6.1 追加臂 on_mb16（两组/micro-batch，同 seed 同 20 步）

```
原猜想   与 on_mb8 差 ±5% 以内（打包后 ~9400 token/前向已喂饱 GPU）
实测     step 15.79 s/gstep = 比 on_mb8 **慢 5.7%**（update_actor 7.60 vs 6.69）
         组构成 [8,8] ×12 次 ⇒ 两组打包机制本身工作正常
结论     方向对、幅度压线偏慢 ⇒ **PG 的最优配置就是 mb=8（一组一批）**；
         慢的来源与 E25 同族：token 已饱和 + 组间补齐到同宽的白算
```

### 6.7 还欠的一步（验收顺序的 ④）

```
⬜ B5 任务尺子（≥60 步 + 冻结 EVAL 配对）—— 纪律：只报吞吐不报任务精度不算完成
```

⚠️ 顺手修掉的两个启动失败（都不是本补丁的问题，但都挡了冒烟）：
`--ppo-mini-batch-size` 不传时默认 2 被整除守卫拦（行为正确，参数要记得传）；
`--train-file/--val-file` **默认值写死 v3（目录已不存在）→ 已改成跟 `DATA_VERSION` 走**；
`launch_rl --help` 崩在一处裸 `%`（argparse 格式化）→ 已修。

## 7 · ★★★ 判据失效的七种形状（今天一天集齐：2 种设计错 + 5 种执行错）

**设计错（判据本身不对）：**
```
D1 太松 ⇒ 为错误的理由通过    「打开开关量墙钟」会把"没接上"读成"没收益"（§5.9 上游考古的核心）
D2 太严 ⇒ 把对的判成错的      bf16 下要求逐位相同（噪声地板 mean 1.28e-2 > 要抓的误差）；
                              「组大小必须相等」把合法但低效的情形误杀（#8）
```
**执行错（判据是对的，读的方式不对）：**
```
E1 读早了      「还没发生」和「不会发生」在日志里长得一样 ⇒ 只在**终态**读
E2 读错对象    新跑根本没起来，读的是旧日志（mtime 是唯一证据）
E3 读到文案    安装横幅里含判据行字符串 ⇒ grep 误报（ONBOARDING 坑 #13 同族）
E4 覆盖面不足  只挂第一个参数的钩子 ⇒「第一个匹配」≠「没有不匹配」
E5 打错目标    patch AdamW 而实际是 Adam ⇒ 零迹象；输出 0 行会被读成"没找到问题"
```
**四个解药（今天实测有效）：**
```
① 终态判据：退出码 / timing 首行出现，不读中间态
② 对照计数：同时打一个**已知必然发生**的计数（组构成/判据A），把"没测到"与"没发生"分开
③ 噪声地板：同数据同数学、只改一个无关变量，回答"差异是否超出噪声"
④ 一字不变检验：修复后现象**完全不变** ⇒ 改的东西不在因果链上，立即回头
```
