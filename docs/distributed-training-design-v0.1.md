# 4×5090 分布式训练实验设计 v0.1


> ⚠️ **本文件是设计记录，内含的实测数字属于当时的测量条件。**
> **当前有效的机器画像、带宽口径与焦点，一律以 `docs/infra_exp/00-INFRA-HANDOFF.md`
> 和 `docs/syncopate/08-machine-and-environment.md` 为准**；焦点是怎么定下来的见
> [`docs/focus-migration-2026-08.md`](focus-migration-2026-08.md)。
> 本文件不随硬件更新逐条改写 —— 设计过程本身是资产。

> 2026-08-13。**搬到 4×RTX 5090 服务器之后的施工蓝图。**
>
> 三份文档的分工（别混）：
> - `docs/syncopate/05-handoff.md` —— 业务/数据/训练主线的交接（infra 增量在 §17）
> - `docs/ostinato-project-design-v0.2.md` —— **单卡** infra 优化（前缀缓存 / 量化 / 手写算子）
> - **本文档** —— **多卡才能做的事**：异步 RL、并行策略、通信画像
>
> 上下文：单卡阶段的全部结论见 ostinato 文档；本文档只写「4 卡解锁了什么」。
>
> ---
>
> **v0.1.1 进度增补（2026-08-13 深夜，上机当天）**——这份文档写完当天就有四条要改：
> 1. ✅ **D-A1 答完了，而且比预期复杂**：P2P 不只是"关着"，**NCCL 还以为它开着**
>    ⇒ 必须显式 `NCCL_P2P_DISABLE=1`，否则 FSDP 第一次广播就炸（§2.0.1）。
> 2. 🆕 **分卡异步的接线已经写完**（`--mode one_step_off/fully_async`），
>    路上撞了四堵墙，其中两堵是 verl 的上游问题（§7）。
> 3. ⛔ **§6 的「分卡后 rollout 卡可以给 0.8+」被实测推翻**：rollout 卡上不是只有
>    vLLM，权重同步的 bucket 也在那儿（§7.3）。
> 4. 🔴 **D3（TP/PP）现在做不了**：verl 的 megatron/torchtitan 后端**一个都没装**，
>    这是文档写作时没查的前提（§4.3.1）。**实验计划要重排。**
>
> ★ **记账纪律**（和 ostinato 文档统一）：推翻的预期**不删**，就地写
> **原猜想 / 实测 / 推翻后 / 教训**四段。这份文档存在的意义有一半是记录
> **"上机之前我们以为会怎样"**——那部分只有不删才有价值。

---

## 0 · 三条纪律（从单卡阶段继承，代价已经付过）

1. **先测量后动手。** 单卡阶段最大的收获是一个**被自己的测量推翻的假设**
   （交接文档 §17.4：以为 KV 池太小导致缓存反复失效，实测命中率 96.7–97.5%、
   池子用不到一半、零 preemption）。**省掉了 1–2 周写一个解决不存在问题的 kernel。**
   ⇒ 4 卡上第一件事是 §2 的硬件画像，不是急着跑并行实验。

2. **每个加速比都要有同分母。** 用固定 trace 重放（同一批 case、同样的组结构），
   优化前后各跑三遍取中位。不用合成负载。

3. **任何 infra 改动都要过任务级尺子。** 冻结 EVAL 198 条 × 8 采样配对比较
   （MDE ≈ 0.044）+ cap 构成 + 决策位熵。**只报吞吐不报任务精度的优化不算完成。**

---

## 1 · 为什么这件事现在才能做

verl 的两条异步路径**都要求 rollout 和 training 在不同 GPU 上**：

```
one_step_off_policy/ray_trainer.py:89   assert not self.hybrid_engine
fully_async_policy                       trainer_pool 和 rollout 是两个独立资源池
```

⇒ 单卡上**只能跑 sync colocate**，第二研究目标（异步 agentic RL）一直卡着。
单卡阶段能做的只有离线合成：留着第 t−k 步的 ckpt，用第 t 步的 policy 重算同一串 token
的 logprob（`train/staleness.py`），已有第一个点 **ESS/N = 0.846、σ²(0) ≈ 2.0e-4/token**。

**4 卡是这条线从「离线合成」变成「真跑」的门槛。**

---

## 2 · 上机第一件事：硬件画像（半天，不跑训练）

⚠️ **消费级 Blackwell 和数据中心卡最不一样的地方全在这里**，而且它决定后面所有实验的解读。

### 2.0 ★ 已实测的部分（2026-08-13，搬完家当天）

```
4 × RTX 5090 32607 MiB / 575W / sm_120   驱动 570.195.03   CUDA 12.8   torch 2.9.0+cu128
CPU  2× EPYC 7543（120 核 / 2 NUMA）      RAM 944 GB
nvidia-smi topo -m : GPU0–3 两两 PHB，全部 NUMA 0 —— 四卡完全对称
torch.cuda.can_device_access_peer : 4×4 全 0
```

🔴 **D-A1 有答案了：P2P 全关。** 卡间通信一律经主机内存中转。

⇒ **两条预期要就地修正**：
1. **§2.2 和 §4.3 的「拓扑异构 ⇒ DeviceMesh 维度顺序对照」这个实验，在这台机器上
   前提不成立** —— 四卡对称、单 NUMA、没有快慢链路之分。要么换机器做，要么砍掉。
   （表 §3 里「拓扑异构（PIX vs SYS）」那一行同理。）
2. **D2 和 D3 的预期方向都被加强**：连 P2P 都没有 ⇒ TP 的每层 all-reduce 更贵，
   而 LoRA 下 DDP 只需同步 132 MB。**「DDP 显著赢 FSDP」现在是一个更强的预测，
   仍然要实测。**

### 2.0.1 🆕 P2P 的第二半答案：**硬件没有，但 NCCL 以为有**

D-A1 的答案不止「关着」这么简单。**`can_device_access_peer` 全 0 的同一台机器上，
NCCL 的自动探测仍然选了 P2P 路径**，第一次 FSDP 参数广播就炸：

```
transport/shm.cc:590 NCCL WARN Cuda failure 217 'peer access is not supported between these two devices'
torch.distributed.DistBackendError: ncclUnhandledCudaError  (NCCL 2.27.5)
  ← 栈：FSDP.__init__ → _init_param_handle_from_module → _sync_module_params_and_buffers
```

#### ⛔ 上面这一段的修法是错的（同日晚些时候被自己的测量推翻，按纪律不删）

**原猜想**：`can_device_access_peer` 全 0 ⇒ NCCL 探测选错了 P2P ⇒ 设 `NCCL_P2P_DISABLE=1`。
代码已经这么改了。

**实测**（把条件一个一个拆开测，而不是接着猜）：

```
① 裸 torchrun，每个进程看得见全部 4 张卡
     默认                  ✅ 通       ← ★ 关键：不设任何环境变量就能跑
② 模拟 Ray：每个进程 CUDA_VISIBLE_DEVICES 只给一张卡
     默认                  ❌ 217
     NCCL_P2P_DISABLE=1    ❌ 217      ← ★ 原修法**完全无效**
     NCCL_SHM_DISABLE=1    ✅ 通
     NCCL_CUMEM_ENABLE=0   ✅ 通
```

**推翻后的结论**：触发条件不是"P2P 关着"这**一个**条件，是
**P2P 缺失 × Ray 只给每个 worker 开放一张卡** 两个条件叠加 ——
进程根本看不见对端设备，SHM 传输要用的 **CUDA IPC** 就建立不起来。
`NCCL_P2P_DISABLE` 只关 P2P 传输，管不到 SHM 传输里的 IPC，所以无效。

**选哪个可行解，用带宽定**（2 卡 all-reduce bus bandwidth，实测）：

| 配置 | 1 MB | 8 MB | 64 MB | 256 MB |
|---|---|---|---|---|
| `NCCL_SHM_DISABLE=1`（退回 socket） | 1.11 GB/s | 1.89 | 2.05 | 2.09 GB/s |
| **`NCCL_CUMEM_ENABLE=0`**（保留 SHM，走老式 IPC） | ✅ 选它 | — | — | 当前口径见 `infra_exp/README.md` §6 |

⇒ `launch_rl.py` 多卡时设的是 **`NCCL_CUMEM_ENABLE=0`**，快 3 倍。

★ **6.4 GB/s 就是这台机器卡间通信的天花板**（无 NVLink、无 P2P、经主机中转）。
这是 D7 通信画像的第一个实测点，也是所有并行策略选择的定量依据：
**TP 每层两次 all-reduce 在这个带宽下大概率是负收益；LoRA 的 DDP 只同步 132 MB ≈ 20 ms**
（对照每步 91–99 秒 ⇒ 千分之二，可忽略）。

> **教训（比原来那条更值钱）**：
> 1. **"硬件能力"和"库以为的硬件能力"是两件事** —— 这条仍然成立。
> 2. ★ **但"我以为的根因"和"真根因"也是两件事。** 我从一个正确的观察
>    （P2P 全关）推出了一个错误的修法，而且**代码都改完了才去测**。
>    真正省时间的做法是：**把条件拆开，一次只变一个，让实测告诉你哪个是开关。**
>    四行探针脚本、两分钟，抵得上几轮"改一改再跑跑看"。
> 3. ⇒ 凡是"自动检测/自动选择"的东西，在异构或消费级硬件上都要当作待验证项；
>    而**凡是自己推出来的修法，在改代码之前先想想能不能用五分钟证伪它**。

⬜ **还没测的**：NCCL 带宽曲线（`nccl-tests`）、主机内存带宽、
**满载功耗与降频**（D-A2，2.3kW，会污染所有对照实验）。
⚠️ 还有一条新的：**PCIe 空闲时报 Gen1 x16（max Gen4 x16）**，需要在负载下复测 ——
这台是虚拟化环境，链路能力得实测而不是看 max。

### 2.1 必测清单

| 测什么 | 怎么测 | 为什么要紧 |
|---|---|---|
| **GPU 间 P2P 是否可用** | `cuda-samples/p2pBandwidthLatencyTest`，或 `torch.cuda.can_device_access_peer` | ⚠️ **GeForce 从 4090 起就关掉了 PCIe P2P**。5090 极可能同样，**必须实测确认**。没有 P2P ⇒ 所有卡间通信要经过**主机内存中转**，all-reduce 带宽掉一个量级 |
| **PCIe 拓扑与代数** | `nvidia-smi topo -m`、`lspci -vv` | 消费主板 4 卡通常**不是每张都 x16**（常见 x8/x8/x4/x4），而且可能跨 CPU/芯片组 ⇒ **卡与卡之间不对称**（表里的 PIX vs SYS） |
| **NCCL 实测带宽曲线** | `nccl-tests` 的 `all_reduce_perf`，消息大小从 1MB 扫到 1GB | 得到「消息大小 → 有效带宽」曲线。这是后面所有并行策略选择的**唯一定量依据** |
| **主机内存带宽** | `mbw` / STREAM | 没有 P2P 时它才是真瓶颈 |
| **功耗与降频** | `nvidia-smi -q -d POWER,TEMPERATURE` 长跑采样 | 4×575W = **2.3kW**。消费机箱/电源在满载时可能降频 ⇒ **这是一个会污染所有对照实验的隐藏变量**，必须先量出来 |

### 2.2 画像出来之后才能回答的问题

- 没有 NVLink（5090 确定没有）+ 可能没有 P2P ⇒ **TP 的每层 all-reduce 会不会贵到不可用**？
- 卡间不对称 ⇒ **DeviceMesh 的维度顺序**（哪两张卡做 TP、哪两张做 DP）会不会显著影响吞吐？
- **这正是表里「拓扑异构（PIX vs SYS）」那一行的实质**，也是 2 卡永远测不出来的东西。

---

## 3 · 为什么是 4 卡（不是 2 卡）

| 能力 | 2 卡 | 4 卡 |
|---|---|---|
| all-reduce 系数 $\frac{P-1}{P}$ | 0.5（**乐观特例**） | 0.75（接近渐进值 1） |
| NCCL 算法/channel 选择 | 不触发 | **触发**（Ring vs Tree 的分界在 P>2） |
| DeviceMesh 维度顺序对照 | **不可能**（只有一种排法） | ✓ |
| 拓扑异构（PIX vs SYS） | 单一链路 | ✓（同芯片组 vs 跨 CPU） |
| straggler / skew 分布 | 无统计意义（n=2） | 可扫分布 |
| PP 气泡 | 11%，噪声内 | **27%，可见** |
| 2D 组合（DP×TP 等） | 不存在 | ✓ |

**核心论点**：2 卡的每一个数都是**特例**——$\frac{P-1}{P}=0.5$ 是所有 P 里最乐观的，
NCCL 在 P=2 时走退化路径，PP 气泡被噪声淹没，2 个样本谈不上分布。
**4 卡是「能观察到普遍规律」的最小配置。**

### 3.1 PP 气泡为什么 4 卡才看得见

朴素 PP 的气泡占比 $= \frac{P-1}{m+P-1}$（$m$ = micro-batch 数）：

$$P=2,\ m=8:\ \frac{1}{9}=11\%\quad\text{(噪声内)}\qquad
P=4,\ m=8:\ \frac{3}{11}=27\%\quad\text{(明确可见)}$$

⇒ 只有 4 卡上「1F1B / interleaved 到底消掉了多少气泡」才是可测量的命题。

---

## 4 · 实验线 D1–D7

> 排序原则：**先做只有多卡能做、且和本项目研究目标直接相关的**（D1），
> 再做通用并行策略（D2–D4），最后是 agentic 特有的新问题（D5–D6）。

### 4.1 D1 · 异步 agentic RL（★ 第二研究目标，最高优先级）

**这是单卡阶段唯一完全做不了、而项目从一开始就想做的东西。**

三个模式的对照（同一份数据、同一个尺子）：

| 模式 | 资源摆法 | 已有基线 |
|---|---|---|
| **sync colocate** | 单卡（现状） | ✅ 已有完整基线，每步 91–99 秒 |
| **one_step_off_policy** | rollout 卡 / train 卡分离 | ⬜ |
| **fully_async_policy** | 两个独立资源池 | ⬜ |

**资源配比本身就是一个实验**：1 rollout + 3 train / 2+2 / 3+1。
agentic 负载的 rollout 很重（多轮 + 工具往返），**最优配比大概率不是 2+2**。

**要量的四个数**（尺子都已就位，见交接文档 §7）：

1. **吞吐**：每步墙钟、样本/秒 —— 分母是 sync colocate 的基线；
2. **staleness 分布**：异步下实际的 k 分布（单卡只能人为构造 k）；
3. **σ²(k) 真值 vs 离线合成值** —— 我们有 `staleness.py` 合成的曲线
   （ESS/N=0.846, σ²(0)≈2.0e-4/token），**真异步跑出来的曲线和它对不对得上，
   本身就是一个可发表级的验证**；
4. **分布漂移**：`record_dispatch`（下发过的）vs rollout dump（训练到的）。
   sync 下这个差**恒为 0**（barrier 保证），异步下短任务先回、长任务被切断 ⇒ 差就出来了。
   **而长链正是 agentic 的核心能力** ⇒ 漂移会不会系统性地丢掉长任务，是这条线最关键的问题。

**理论上界已经在单卡量到了**：同批里最慢/平均 = **1.37–2.75×** ⇒
sync barrier 浪费掉的就是这一块，也是异步收益的天花板。**4 卡上验证实际吃到了多少。**

### 4.2 D2 · 数据并行的分片策略（最便宜、最先出数）

4B + LoRA r32 **单卡放得下**（实测 actor 峰值 13.9 GB）⇒ **DDP 是可行的**，
于是 FSDP 的 all-gather 变成**纯开销**。这是一个很干净的对照：

| 策略 | 每步通信量 | 预期 |
|---|---|---|
| `FULL_SHARD`（现状） | 每层 all-gather 权重 + reduce-scatter 梯度 | 通信最重 |
| `SHARD_GRAD_OP` | 只分片梯度/优化器状态 | 中 |
| `NO_SHARD` / DDP | 只 all-reduce 梯度 | **LoRA 下梯度只有 66M ⇒ 通信极轻** |

★ **LoRA 让这个对照特别有意思**：可训练参数只占 1.64%，
**DDP 的 all-reduce 只需要同步 66M 参数（约 132 MB），而 FSDP 要 all-gather 全部 4B 权重。**
在没有 NVLink 的机器上，这个差距会被放大到极致。
⇒ **预期结论：本项目这个规模，DDP 应该显著赢 FSDP。** 但要实测。

### 4.3 D3 · 张量并行 / 流水并行（在无 NVLink 上做，结论才有信息量）

- **TP=2 / TP=4**：每层两次 all-reduce，通信频率最高。
  **没有 NVLink 的情况下 TP 大概率是负收益** —— 但「负多少、从哪个 batch size 开始
  转正」是有价值的曲线，而且是数据中心论文里看不到的。
- **PP=2 / PP=4**：测气泡占比与理论值 $\frac{P-1}{m+P-1}$ 的吻合度；
  micro-batch 数 $m$ 扫描；1F1B 调度实际消掉了多少。
- **2D 组合**（DP×TP、DP×PP）：**2 卡上根本不存在的维度**。
  重点是 **DeviceMesh 维度顺序** —— 在拓扑异构的机器上，
  「把 TP 放在快链路上、DP 放在慢链路上」应该显著优于反过来。**这是 §2 画像的直接应用。**

#### 4.3.1 🔴 前提没查：**训练侧 TP/PP 现在根本跑不了**（2026-08-13 查证）

写 D3 的时候默认了「verl 支持 TP/PP，配一下就能跑」。**查了才发现差一整个后端。**

verl 0.8 的训练侧 `strategy` 有五个：

```
fsdp ✅   megatron ❌   mindspeed ❌   torchtitan ❌   veomni ❌
```

而 venv 里的实际情况（`.venv/bin/python -c "import ..."`）：

```
torch 2.9.0+cu128 ✅    vllm 0.12.0 ✅    flash_attn = 我们自己的垫片 ✅
megatron ❌未装   torchtitan ❌未装   veomni ❌未装   transformer_engine ❌未装   apex ❌未装
```

⇒ **今天开箱能做的并行维度只有三个**：

| 维度 | 怎么开 | 状态 |
|---|---|---|
| **DP** | FSDP 三档（`FULL_SHARD` / `SHARD_GRAD_OP` / `NO_SHARD`≈DDP） | ✅ 立刻可做 = **D2** |
| **SP** | `ulysses_sequence_parallel_size`（FSDP 路径自带） | ✅ 可做 = **D4** |
| **推理侧 TP** | vLLM 的 `tensor_model_parallel_size`（rollout 卡上） | ✅ 可做，但**和训练侧 TP 是两回事** |
| 训练侧 **TP / PP** | 要 megatron（+ megatron-bridge）或 torchtitan | 🔴 **阻塞** |

⚠️ **两条路都有 sm_120 的风险，而且是我们见过的形状**：
- **Megatron** 要 Transformer Engine / apex ⇒ **和当初 flash-attn 是同一个坑**
  （消费级 Blackwell 上编译一两个小时且可能失败）。verl 有 `workers/config/megatron_peft.py`
  ⇒ LoRA 在 megatron 下**看起来**有支持，但没验过。
- **torchtitan** 是纯 PyTorch DTensor 路线，**大概率省事得多**，
  而且 verl 已经生成了 `_generated_ppo_torchtitan_trainer.yaml`（含 TP/PP/CP 配置项）。

⇒ **D3 拆成两步**：先做**可行性探针**（半天～一天，装得上/LoRA 能不能用/能不能跑一步），
探针通过再排实验。**不要把"装后端"和"做实验"记成同一件事**——
本项目已经栽过一次「以为是一个开关，其实是一整套 trainer」（§7.1）。

> **教训**：**写实验设计时，"框架支持 X" 要区分成三层**：
> ① 配置项存在 ② 代码路径存在 ③ **依赖装着、且在这块卡上能跑**。
> D3 当初只查到 ①。这和 ostinato §26.2.1「现成的用不了」是同一枚硬币的两面——
> 一次把不能用的当成能用，一次把能用的当成不能用。**都是没往下查一层。**

### 4.4 D4 · 序列并行（Ulysses）—— 和我们的长轨迹强相关

verl 有 `ulysses_sp_size`。agent 轨迹很长（M8 之后会更长），SP 把序列切开分给多卡。
通信模式是 **all-to-all**（比 all-reduce 更吃拓扑）⇒ 在无 NVLink 机器上尤其值得测。

★ 和单卡的稀疏投影/融合 kernel 是**互补**的：那两个减的是**显存**，SP 减的是**单卡序列长度**。
⇒ 值得测「多长的序列之后 SP 才划算」。

### 4.5 D5 · ★★ agentic 特有的新问题：rollout 分片会不会打碎前缀缓存

**这一条是本项目独有的角度，别的分布式论文不会碰。**

GRPO 一个 case 复制 8 份 rollout。如果这 8 份被**分派到不同的 rollout 副本**上：

```
同一副本   8 份共享一个 prefix cache 条目  ⇒ prefill 只算一次
分到 2 副本 每个副本各算一次              ⇒ prefill 算两次，缓存被打碎
```

单卡阶段实测**命中率 96.7–97.5%**（交接文档 §17.4），
**多副本之后这个数会掉多少，是一个从没人测过的问题。**

- 要测：副本数 1/2/4 时的命中率、prefill token 总量、TTFT；
- 如果掉得明显 ⇒ **「同组 rollout 路由到同一副本」是一个有实际价值的改进**
  （SGLang 有 cache-aware routing，verl 的 dispatch 未必做这件事）；
- ⇒ 这可能是本项目**最有希望提上游 PR 的一个点**，而且它把 ostinato 的前缀线和
  分布式线接在了一起。

### 4.6 D6 · rollout 长尾在数据并行下的放大效应

单卡实测：同批里**最慢/平均 = 1.37–2.75×**。
DP=4 时每个 rank 分到的 batch **token 总量不同** ⇒ all-reduce 要等最慢的 rank
⇒ **长尾在 DP 下被放大**（表里「EP skew / straggler」那一行的我们版本）。

- 要测：per-rank 每步耗时分布、等待时间占比；
- 缓解手段对照：按 token 数均衡分片（verl 的 `balance_batch`）、
  动态分池（本项目已有 `train/pool.py`）、APRIL 式超发早停；
- ⚠️ **超发早停有统计风险**：先完成的偏短 ⇒ advantage 的长度偏差。
  **动它等于动实验有效性，必须先设计好对照。**

### 4.7 D7 · 通信开销的直接画像（贯穿全部实验）

- nsys 时间线上把 **NCCL kernel 单独拉出来**：通信占每步的百分比；
- 计算/通信重叠率（FSDP 的 prefetch、梯度桶）；
- NCCL 调参对照：`NCCL_ALGO`（Ring/Tree）、`NCCL_P2P_DISABLE`、`NCCL_SHM_DISABLE`、
  `NCCL_MIN_NCHANNELS`、`NCCL_BUFFSIZE`；
- ★ **权重同步的代价**：异步模式下 trainer 每步要把权重推给 rollout 卡。
  **LoRA 下只有 132 MB**（verl 的 `TensorLoRARequest` 走 adapter-only，
  见 ostinato 文档 §25.1）⇒ 实测这笔传输在无 P2P 机器上要多久，会不会成为异步的新瓶颈。

---

## 5 · 里程碑

| # | 内容 | 验收 | 预估 |
|---|---|---|---|
| **D0** | §2 硬件画像 | P2P 有无 / 拓扑图 / NCCL 带宽曲线 / 功耗降频曲线 | 半天 |
| **D1** | 单卡基线在 4 卡上复现 | 同 trace、同尺子，先跑通 DP=4，确认任务分与单卡一致（EVAL 配对） | 1 天 |
| **D2** | D2 分片策略对照 | FULL_SHARD / SHARD_GRAD_OP / DDP 三方跑分 | 1–2 天 |
| **D3** | ★ 异步三模式（D1 实验线） | 吞吐 / staleness 分布 / **σ²(k) 真值 vs 合成值** / 分布漂移 | 1 周 |
| **D4** | TP/PP/SP 与 2D 组合 | 通信占比曲线 + DeviceMesh 顺序对照 | 1 周 |
| **D5** | ★ 前缀缓存分片效应 | 命中率 vs 副本数曲线；必要时实现同组路由 | 3–5 天 |
| **D6** | 长尾与均衡 | per-rank 耗时分布 + 三种缓解手段对照 | 3–5 天 |

**依赖**：D0 → D1 是硬前置；D2–D6 之间大体独立，可按兴趣调序。
**D3（异步）是研究价值最高的**，D5 是最可能产出上游贡献的。

### 5.1 🆕 状态盘点与重排（2026-08-13 深夜）

| # | 状态 | 变化 |
|---|---|---|
| **D0** 硬件画像 | 🟡 **做了一半** | P2P ✅（含 §2.0.1 那条 NCCL 的意外）、拓扑 ✅。**还差**：NCCL 带宽曲线、主机内存带宽、**满载功耗降频**（2.3kW，会污染所有对照实验）、PCIe 负载下复测 |
| **D1** 单卡基线在 4 卡复现 | ⬜ 未开始 | —— |
| **D2** 分片策略（FSDP 三档 vs DDP） | ⬜ **现在是最该先做的实验** | 依赖只有 FSDP，**没有被 §4.3.1 的后端问题阻塞**；预期最明确（LoRA 下 DDP 只同步 132 MB vs FSDP all-gather 全部 4B，无 P2P 会把差距放大） |
| **D3** 异步三模式 | 🟡 **接线完成，冒烟未通** | 见 §7.5。翻过三堵墙，卡在 rollout 卡显存预算 |
| **D4** TP/PP/SP + 2D | 🔴 **部分阻塞** | **SP（Ulysses）可做**；**训练侧 TP/PP 差一整个后端**（§4.3.1）⇒ 先做可行性探针 |
| **D5** 前缀缓存分片 | ⬜ 未开始 | 🆕 **权重上升**：ostinato §4.0 推翻了单卡的"容量红利"之后，**D5 是"容量/缓存"这条线唯一还活着的兑现场景** |
| **D6** 长尾与均衡 | ⬜ 未开始 | —— |

★ **重排后的顺序**：**D0 补齐（半天） → D2（最便宜、无阻塞） → D3 冒烟跑通 →
D5 → SP/TP 探针**。理由：
1. **D0 是所有加速比的分母**，尤其是 `NCCL_P2P_DISABLE=1` 之后**走 SHM 的真实带宽**
   —— 不量它，后面每一条"通信开销占比"都没有参照；
2. D2 无阻塞、预期最明确，**是验证 4 卡管线的最便宜路径**；
3. 🔴 **另外插一件本该更早做的事**：ostinato 的 **H0（nsys 拆 91–99 秒）**。
   单卡时代欠的账，但**在 4 卡上做反而更值** —— 可以直接做
   「colocate vs DP=4 vs 分卡异步」的**时间构成 diff**，一次拿到三份画像。

---

## 6 · 从单卡搬过来时会失效的东西

⚠️ **单卡阶段的显存调优几乎全部是 colocate 的产物，分卡之后语义都变了**：

| 单卡设置 | 4 卡上怎么办 |
|---|---|
| `--free-cache-engine False` | **不再需要**：rollout 和 trainer 不共卡，就没有 sleep/wake 争抢 |
| `--rollout-gpu-util 0.40` | ~~分卡后 rollout 卡上没人抢，**可以给到 0.8+**~~ ⛔ **已推翻，见 §7.3**：rollout 卡上还住着权重同步的 bucket。实测 0.85 在第一次权重同步时 OOM ⇒ 默认 **0.75** |
| `--max-num-seqs 32` | 按新的 KV 池重算 |
| `--param-offload False` | 重测（主机内存 30 GB 的约束在新机器上可能不同） |
| `--remove-padding True` / `--fused-kernels True` | ✅ **继续开**，它们省的是训练侧显存，和卡数无关 |
| `--micro-batch-size 1` | 显存宽松后可以加大，但**改了要同步调整才和单卡基线可比** |

⚠️ **有效 batch 的可比性**：DP=4 时每步样本数 ×4。
要和单卡基线比能力，**必须保持有效 batch 一致**（降 `train_batch_size` 或 grad-accum），
否则测的是「更大 batch」不是「并行策略」。

⚠️ **数据和记忆的搬迁**：数据 100% 脚本可复现（交接文档 §16.1）；
`reference/` 870MB 只能手动 scp；**Claude 的记忆不在 git 里**——
关键内容已经合并进交接文档 §13–§17 和本文档。

---

## 7 · 🆕 已落地：分卡异步的接线，与撞到的四堵墙（2026-08-13）

> 这一节记的是**从「设计文档说可以做」到「命令能跑起来」之间那段路**。
> 四堵墙里**两堵是 verl 的上游问题**，一堵是我们自己的想当然，一堵是硬件。
> 都不难，但**没有一堵是设计文档里预见到的** —— 这本身是这一节存在的理由。

### 7.1 墙①②：这不是一个开关，是**三套 trainer**

`--mode {colocate | one_step_off | fully_async}` 背后是三个**不同的 verl 入口**：

```
colocate      verl.trainer.main_ppo                                   rollout 和 train 同卡
one_step_off  verl.experimental.one_step_off_policy.main_ppo          分卡，落后一步
fully_async   verl.experimental.fully_async_policy.fully_async_main    分卡，两个独立池
```

不同的 hydra 根配置、不同的 worker 栈（`separation.engine_workers.DetachActorWorker`，
不是 colocate 那套 FSDP worker）、不同的 main。由 `main_ppo_pool` 按 `SYNCOPATE_RL_MODE` 分发。

**墙① · 上游 bug（one_step_off 独有）**：`one_step_off_ppo_trainer.yaml` 的
`hydra.searchpath` 写的是 `file://verl/trainer/config` —— **相对 CWD**，
只有在 verl 源码仓库根目录下跑才找得到，从我们自己的仓库根跑就
`Could not load 'ppo_trainer'`。同目录的 `fully_async_ppo_trainer.yaml` 写的是正确的
`pkg://verl.trainer.config` ⇒ **这是疏漏不是我们配错**。修法：命令行覆盖成 `pkg://`。
**（可提上游的第一个小 PR。）**

**墙② · 两个实验性入口必须当 `__main__` 跑，不能 import 了再调**：

```
Primary config module 'verl.experimental.one_step_off_policy.config' not found.
```

根因：`@hydra.main(config_path="config")` 解析路径的方式取决于 **task function 的
`__module__`** —— 是 `__main__` 就按**文件路径**找，否则按**模块**找（要求那个 config
目录是个包）。而实测：

```
verl/trainer/config/__init__.py                          有   ← 所以 colocate 直接 import 能跑
verl/experimental/one_step_off_policy/config/__init__.py  没有
verl/experimental/fully_async_policy/config/__init__.py   没有
```

⇒ 两个实验性配置目录**不是包**，verl 只考虑了 `python -m` 的用法。
修法：`runpy.run_module(..., run_name="__main__")`，等价于 `python -m`，
**且不用碰 site-packages**（那是旁路，会绕开主路径的守卫）。

### 7.2 动态分池：one_step_off ✅ / fully_async ❌（**说出来，别让人以为在生效**）

`create_rl_sampler` 的 monkeypatch 在 `one_step_off` 下**能生效**——但不是自动的：
`one_step_off_policy/main_ppo.py` 用的是 `from verl.trainer.main_ppo import create_rl_sampler`
（**导入时**绑名），按说改模块属性对它无效；**能生效是因为 runpy 那次导入发生在补丁之后**。

⚠️ **`fully_async` 根本不调 `create_rl_sampler`**（它自己排采样计划）
⇒ **动态分池在那个模式下不存在**，启动时打一行警告。**做三模式对照时别把两件事混在一起。**

★ 判据仍然是那一条：日志里有没有 `[pool] 动态分池启用`。
**11:21 那次失败的日志里有这行** ⇒ 接线是通的，挂在别处。

### 7.3 ⛔ 墙③：rollout 卡上**不是只有 vLLM**（推翻 §6 的「0.8+」）

> **原预期**（§6 表格）：`--rollout-gpu-util 0.40` 是和 actor 抢卡算出来的，
> **分卡后没人抢 ⇒ 可以给到 0.8+**。
>
> **实测**（one_step_off，trainer 3 卡 / rollout 1 卡，`--rollout-gpu-util 0.85`）：
> 模型加载完、vLLM 建池完、rollout 跑完，**死在第一次权重同步**：
> ```
> torch.OutOfMemoryError: Tried to allocate 2.00 GiB
>   ← verl/checkpoint_engine/nccl_checkpoint_engine.py:142  prepare()
> 账：vLLM 0.85 × 31.37 = 26.66 GB 常驻（进程实占 27.55 GB）
>     CheckpointEngineWorker 自身 2.58 GB + NCCL bucket 2.00 GB = 4.58 GB
>     31.37 − 27.55 = 1.23 GB 可用   ⇒ 差得远
> ```
> **推翻后**：trainer 每隔几步把权重推给 rollout 卡，NCCL checkpoint engine 会在
> **接收端**现开暂存区。`update_weights_bucket_megabytes` 默认 **2048 MB**，
> 而且源码注释写明 **send/recv 双缓冲 ⇒ 开销 2×bucket**。
> ⇒ 默认降到 **0.75**，bucket 做成显式参数 `--weight-sync-bucket-mb`。
>
> **教训（两条）**：
> 1. **"分卡之后没人和它抢"是想当然。** 抢的东西只是换了个名字
>    （actor → checkpoint engine），就没被算进预算。
>    ⇒ **凡是说"某某资源现在独占"，先列一遍"还有谁会在这张卡上分配内存"。**
> 2. **贴着墙的配置会把失败推迟到最贵的时刻。** 0.85 那次是"差 0.13 GB"，
>    而它炸在**第一步训练之后**——前面所有加载、建池、rollout 全白跑。
>    ⇒ 显存预算要留**结构性余量**，不是"刚好够"。

★ 连带修正 ostinato §25.1 的一个说法：「LoRA-only 同步只推 132 MB，很便宜」
—— **在 colocate 下对，分卡之后不对**：它有了①接收端 2×bucket 的显存
②经 PCIe/主机内存的传输时间（**没有 P2P**）。⇒ **D7 要把这笔重新量。**

### 7.4 墙④：NCCL 以为 P2P 可用 —— 见 §2.0.1

### 7.5 当前状态（截至 2026-08-13 15:20）

```
接线          ✅ 三个模式的入口、拓扑解析、硬失败校验、NCCL/bucket 参数都在
动态分池      ✅ one_step_off 下确认生效（日志有 [pool] 那行）
one_step_off  ✅ **跑通并调优完毕**，稳态 49.5–50.7 s/步（见 §7.6）
fully_async   ⬜ 没试过
```

一共翻过**九堵墙**（前四堵见 §7.1–7.4，后五堵）：

| # | 死在哪 | 性质 | 修法 |
|---|---|---|---|
| 5 | vLLM 起引擎 | 依赖 | numpy 2.5 → **2.2.6**（numba 卡 ≤2.2）。这就是交接文档「numpy 必须 2.2.6」一直没写的原因 |
| 6 | 第一次生成 token | 驱动/工具链 | vLLM 自带 FA2 的 sm_120 PTX 比驱动新 ⇒ `VLLM_ATTENTION_BACKEND=TRITON_ATTN`（FLASHINFER 也挂，FLEX_ATTENTION 可用） |
| 7 | rollout → 训练格式 | verl 上游不一致 | 依赖表写 `tensordict>=0.8.0`，代码里 `assert >=0.10` ⇒ 钉 **0.10.0**。colocate 不走 `to_tensordict`，所以以前从没暴露 |
| 8 | dump 训练数据 | verl 上游 bug | `OneStepOffRayTrainer` 漏调 `_init_dump_executor()`（`RayPPOTrainer` / `FullyAsync*` 三处都调了）。**触发条件是设了 `rollout_data_dir`** ⇒ 上游大概没开着 dump 跑过异步。补丁见 `syncopate/train/verl_patches.py`（子类 + 幂等，不改 site-packages） |
| 9 | — | 我自己算错 | `--rollout-gpu-util 0.85` 没给权重同步的 bucket 留位置，第一次同步就 OOM ⇒ 0.75（见 §7.3） |

★ **第 8 条不能用「关掉 dump」绕过**：那份 dump 是分布漂移的一半
（dump = 训练到的，`dispatched.jsonl` = 下发过的），关掉等于为了跑通把要测的东西关了。

### 7.6 ★★★ 调优：2×2 对照，从 1182 s/步到 49.5 s/步

跑通之后第一份耗时分解（1 训练卡 + 1 rollout 卡，稳态）就指明了方向：
**瓶颈已经不在 rollout**（gen 只占 4%），在训练侧的前向/反向。

两个旋钮，**各扫两档，拆开测**（这次没再犯"同时动两个变量"的错，虽然第一轮犯了）：

| **稳态 s/步**（update_actor） | 静态 micro_batch=1 | `use_dynamic_bsz` 16384 |
|---|---|---|
| **1 训练卡** | 116.0（84.5） | 234.8（187.8） |
| **3 训练卡 `fsdp_size=1`** | **49.5 / 50.7（30.0）** ✅ | 91.6（64.0） |

两个因素干净地**相乘**，且在两种卡数下比值一致：

```
DDP（fsdp_size=1）      update_actor ÷ 2.93 ~ 3.00     ← 完美线性
use_dynamic_bsz         update_actor × 2.19 ~ 2.22     ← **倒退**
```

#### ① `fsdp_size=1` 是这台机器上的必选项

`create_device_mesh`（`workers/engine/fsdp/utils.py:40`）：`fsdp_size` 取 1 ⇒
mesh `(world_size, 1)`、维度 `["ddp","fsdp"]` ⇒ fsdp 维长度 1 = **不切分**，
只在 ddp 维 all-reduce 梯度 = **DDP**。

**实测 3 卡 FULL_SHARD 每步 1182 s，1 卡不切分 198 s（同为 step 1）—— 多两张卡慢 6 倍。**
D2 那条「DDP 应该显著赢 FSDP」不只是被证实，是**多给卡变成净亏损**。
机制：FULL_SHARD 每层前向反向都要 all-gather 8 GB 权重，而卡间只有 6.4 GB/s；
LoRA 的 DDP 只同步 66M 梯度（~260 MB ⇒ 40 ms）。**三个数量级。**

⛔⛔ **2026-08-18 更正（见 `docs/infra_exp/E21-ddp-not-syncing.md`）**：
上面这段里的「DDP 每步同步 260 MB 梯度」**是算出来的（66M 参数 × 4 字节），从没量过**；
而 E21 已证实 **`fsdp_size=1` 下梯度根本没有跨 rank 同步** ⇒ **这段流量此前并不存在**。
⇒ 由它推出的一切（跨 socket 净代价、与分片的量级对比）**在修复之前都不成立**。
修复已于 2026-08-18 落地并复验，**但由它推出的数字要重测之后才能引用**。
★ 教训：**算出来的数字不是测出来的数字** —— 见 E21 §5.5-③。


#### ② ⛔ `use_dynamic_bsz` 在本机是倒退（原猜想被推翻）

**原猜想**：序列才 4.1k token，`micro_batch=1` 喂不饱 5090，打包成 16k 该更快。
**实测**：1 卡 84.5 → 184.9 s，3 卡 64.0（对照 28-30）—— **慢 2.2×，与卡数无关。**

**根因（推断，量级对得上）**：我们**没有真的 flash-attention，只有垫片**
（`scripts/install_flash_attn_shim.py` 提供的四个纯 PyTorch gather/scatter），
真正的注意力走 `sdpa`。真 flash-attn 会按 `cu_seqlens` 分段算，代价 Σ(单条²)；
sdpa 路径上打包 = 一条 16k 长序列 ⇒ **(16k)² ≈ 4×Σ(4k)²**。

> **教训**：交接文档 §17.1 那句「**垫片就够了，不用编译 flash-attn**」要限定范围 ——
> **对正确性够用，对打包不够用。** 装了真 flash-attn 之后这一条要重测，大概率会翻过来，
> 而且那时 dynamic_bsz 可能变成一个大的正收益。**这是目前最值得做的一项 infra 投资。**

#### 调优后的耗时构成与下一个目标

```
update_actor    30.0 s   61%
update_weights  13.3 s   27%   ← ★ 下一个目标
ref              6.0 s   12%
gen              4.1–6.2 s     ← 异步已经把它藏掉了（step1 是 89.6 s）
```

⚠️ **`update_weights` 13.3 s 是个未解之谜**：LoRA 只有 132 MB，按 6.4 GB/s 应是毫秒级。
时间显然不在传输上（收集？格式转换？进程同步？）。它占 27% 且**随训练卡数上升**
（1 卡 12.3 s → 3 卡 13.3–15.9 s），是异步方案的上限所在。

#### 已定的默认值（全部有实测支撑）

```
--fsdp-size 1              DDP，3.00× 完美扩展（verl 默认 -1 慢 6 倍）
--dynamic-bsz False        实测慢 2.2×
--trainer-gpus/--rollout-gpus  3 / 1     gen 只占 12%，rollout 不是瓶颈
--rollout-gpu-util 0.75    KV 池峰值只用 16.7%、零 preemption ⇒ **不是瓶颈，别动**
--attention-backend TRITON_ATTN
NCCL_CUMEM_ENABLE=0        多卡时自动设
```

★ **rollout 卡的显存分配是个反直觉点**：KV 池 110,496 token 听起来很大，
但实测**峰值只用 16.7%、零 preemption**，排队是 `max_num_seqs` 卡的、不是显存卡的。
⇒ 和单卡阶段 §17.4 那个"被自己的测量推翻的假设"是同一个结论，分卡后**更极端**。
**给 rollout 卡加显存换不来速度。**

⚠️ **`--mode` 的拓扑解析刻意做成硬失败而不是自动降级**：分卡模式静默退回单卡的表现是
「跑起来了、但测的根本不是异步」——**本项目最怕的那种失效**（交接文档 §15）。

---

## 8 · 🆕 框架与 MoE 选型决策（2026-08-13，Chaoyu 批准）

> 完整论证和显存账在 **`docs/infra_exp/E07-moe-ep.md`**，这里只记结论，别再重新讨论：

| 决策 | 结论 | 一句话理由 |
|---|---|---|
| 框架 | **verl 不换**（抛开沉没成本重选仍是它） | sm_120 上唯一「纯 PyTorch 训练路径 + 三档异步 + agentic rollout」三条全过且本机已实证 |
| Dense 线 | Qwen3-4B LoRA 收尾，one_step_off 3+1，trainer 用 DDP | 单卡放得下 ⇒ FSDP 分片是纯亏（D2 顺手验证） |
| MoE 线 | **GLM-4.7-Flash（30B-A3B，MIT）+ LoRA + GSPO** | 唯一为 agentic 调过的这个体量 MoE；GSPO 的序列级 ratio 对 MoE 路由抖动不敏感 |
| MoE 并行 | 分片(FSDP2×3) / 复制(QLoRA 4bit 纯 DP) / EP(toy→Megatron 探针) 三摆法对照 | 6.4 GB/s 上「量化复制」可能赢「分片聚合」——本机特有的命题 |
| slime/Miles | 不当底座，当教材 | Megatron 消费卡先例存在（4×4090，见 E07 §1.2），sm_120 待 P4 探针 |

---

## 附录 A · 开放问题

| # | 问题 | 何时回答 |
|---|---|---|
| ~~D-A1~~ | ~~5090 的 PCIe P2P 到底有没有被关掉~~ | ✅ **已答（2026-08-13）：全关**，见 §2.0。⚠️ **而且还有第二半**：NCCL 的自动探测**以为它可用**，必须显式 `NCCL_P2P_DISABLE=1`，见 §2.0.1 |
| D-A2 | 4 卡满载会不会降频（2.3kW） | D0，长跑采样；**它会污染所有对照实验** |
| D-A3 | 主板 PCIe 通道怎么分的（x16/x8/x4） | D0，`nvidia-smi topo -m` |
| D-A4 | 无 NVLink 下 TP 是不是必然负收益、从哪个规模转正 | D4 |
| D-A5 | 异步的实际收益能吃到 1.37–2.75× 上界的百分之多少 | D3 |
| D-A6 | 真异步的 σ²(k) 和离线合成的对不对得上 | D3（**对不上的话，单卡阶段那套合成方法就要重新审视**） |
| D-A7 | 多 rollout 副本会让 97% 的命中率掉到多少 | D5 |
| D-A8 | verl 的 dispatch 有没有做同组 rollout 的亲和性路由 | D5，读代码即可先答一半 |
| D-A9 | 4 卡上 fully_async 和 one_step_off_policy 的实际差别有多大（verl 两条路径的工程成熟度不同） | D3。🆕 **已有第一份证据**：成熟度**确实不同，而且各有各的坏**——one_step_off 的 hydra searchpath 是坏的、fully_async 的是对的；反过来 fully_async 不调 `create_rl_sampler` ⇒ 我们的动态分池在它下面不生效（§7.1/§7.2） |
| **D-A10** 🔴 | **megatron 还是 torchtitan**：哪条路在 sm_120 上装得起来、且 LoRA 能用 | D4 之前的**可行性探针**（§4.3.1）。**不探针就别排 TP/PP 的工期** |
| **D-A11** | `NCCL_P2P_DISABLE=1` 之后走 SHM／主机中转的**实际 all-reduce 带宽曲线**是什么形状（消息大小 1MB→1GB） | **D0，第一优先级**——它是后面所有"通信占比"的分母 |
| D-A12 | 权重同步 bucket（默认 2048 MB × 双缓冲）的最优值：调小省 rollout 卡显存 vs 调大省同步次数 | D3 冒烟稳定之后（§7.3） |
| D-A13 | 分卡后 rollout 卡 `gpu_util` 的真实上限（0.75 够不够；池子变大之后 D5 的基线要重定） | D3 冒烟跑通时顺手 |
