# E26 · PrefixGrouper：题面只算一次 —— **3.96×**（三次前向）

> 状态：✅ 等价性已过（fp32 逐位）+ 微基准吞吐已测　🔴 **真实训练集成未完成**（§6：13 处接线，卡在优化器 dtype）　⬜ 未过任务尺子
> ⛔ **端到端吞吐数一个都没有** —— 3.96× 只在微基准上成立
> 尺子 `scripts/probe_prefix_grouper_speed.py` · 等价性 `scripts/check_prefix_grouper_equivalence.py`
> 补丁 `syncopate/train/verl_patches.py::_patch_prefix_grouper`（`SYNCOPATE_PREFIX_GROUPER=1`，默认关）

## 0 · 结论卡片

| | |
|---|---|
| **需求从哪来** | E25 证伪「trainer 没喂饱」⇒ 只剩「让它少算」。而 87% 的 token 是一组 8 条共享的题面 |
| **结果** | **三次前向 3.96×**（10.211 s → 2.577 s），显存 11.88 → **12.31 GB**（+3.6%） |
| **端到端外推** | 三次前向占步 88.9% ⇒ `1/(0.111+0.889/3.96)` = **约 2.98×** |
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
⬜ 在真实训练里跑通（本报告是隔离的微基准：无 FSDP / 无 Ray / 无 DDP）
⬜ 过 B5 任务尺子（纪律：只报吞吐不报任务精度不算完成）
⬜ 组大小 ≠ 8、变长 response 下的显存与稳定性
```
⚠️ **3.96× 是微基准数，不是端到端数。** 那个「约 2.98×」是外推，不是实测。

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
| 13 | 优化器 dtype：`expected BFloat16 … got float` | `torch/optim/adam.py`（离根因最远的一次） | 🔴 **未解**，§6.3 |

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

### 6.3 🔴 当前卡点：Adam `expected dtype BFloat16 for 'end' but got float`

```
已确认   对照跑（补丁关、同配置）跑通：step 119.43 s / update_actor 59.72 s ⇒ 是我们的路径引入的
已排除   「把 log_probs 转 bf16」—— 方向反了：FusedLinearForPPO **契约就是返回 fp32**
         （源码 assert token_log_probs.dtype == torch.float32）；且改后现象一字不变
诊断三连败  钩子只挂第一个参数（E4 覆盖面）· patch 了 AdamW 而 verl 用 Adam（E5 打错目标）
           · 改对目标后扫描仍 0 行（原因未明，判据A/组构成同期各打 2 次）
```
⇒ **下一步：脱 Ray 最小复现**（`scripts/repro_pg_dtype.py`）—— FSDP+LoRA+我们的前向，
   秒级迭代，直接打印全部 `(param, grad)` dtype。
⚠️ 那段 Adam 扫描的诊断代码**已清掉** —— 它从没跑起来过（先打错类：patch 了 AdamW，而 verl 用 Adam；改对之后仍 0 行、原因未明）。⇒ **不留「看起来在监控、其实什么都没做」的死代码**（那正是 D1 型判据的温床）。

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
