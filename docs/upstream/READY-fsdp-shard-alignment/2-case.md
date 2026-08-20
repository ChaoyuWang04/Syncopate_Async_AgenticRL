# Case · FSDP 分片不做 16 字节对齐 ⇒ NCCL all_gather 掉 12×（E18 / A17）

```
状态    READY —— 提交件成稿，等 Chaoyu 提交
目标    ① pytorch/pytorch 提 issue（修法**主推 FSDP2**，活代码）
        ② NVIDIA/nccl 在 #413 下**评论请求重开**（不提新 issue —— 2020 年维护者已确认并把
           修法指派给调用方："padding is a solution that would always work well"）
验证    微基准 FSDP1 12.2× / FSDP2 12.56×（+12 B 恢复）· A17 端到端 update_actor 3.6× / ref 6.9×
风险    🟡 FSDP1 那半可能被以「维护模式」驳回（包①正是这样被关的）
        ⇒ 已把修法重心放在 FSDP2；FSDP1 只作「顺带一提」
影响    ⚠️ 这条**不影响我们自己的训练**（生产路径已定 DDP 不分片）——纯社区贡献件，优先级最低
```

发现来源 [`../../infra_exp/E18-rank3-allgather-collapse.md`](../../infra_exp/E18-rank3-allgather-collapse.md) ·
提交件 [`3-submission.md`](3-submission.md)

---

> 状态：**草稿完成，待 Chaoyu 决定是否提交**　建于 2026-08-18
> 归属：这是一条**独立的线**（不属于 Track A/B 的兑现物，但由 E18 的调查产出）
> 完整实验记录：[`../infra_exp/E18-rank3-allgather-collapse.md`](../../infra_exp/E18-rank3-allgather-collapse.md)

> 🆕🆕 **2026-08-19 · 提交前核查完成，六条（定位要按它们重摆）**：
>
> 1. **PyTorch main 三处全部未修**（源码逐字核对）：FSDP1 `_get_unpadded_shard` 仍是裸
>    `.chunk(world_size)`；FSDP2 `_get_dim0_padded_size` 仍只补到 `dim0_factor` 整除；
>    FSDP1 的 `ALIGNMENT = 16`（`_flat_param.py`，注明来自 TorchInductor）仍只用于
>    **段内**对齐 —— ⇒ PR 框架：「你们已经在乎 16B，只是没对齐到分片边界」。
>    issue 侧四组搜索（`_get_dim0_padded_size`/shard alignment/…）**空白**。
> 2. **NCCL#413（2020）**：同一现象 6 年前就报过（+4 字节 ⇒ ~13×，与我们 12.2× 吻合），
>    负责人 sjeaugey 确认「expected」并把修法指派给调用方 ——
>    **"padding is a solution that would always work well"**；2025-07 被清理机器人关闭，
>    留了「仍相关请重开」的口。⇒ NCCL 侧动作 = **评论重开**，不是新 issue。
> 3. **NCCL 机制在 master 源码坐实**（`src/device/common_kernel.h:219-247`）：
>    `BigPackSize=16`；对齐检查是**警报式全员投票**（任一 src/dst 指针 %16≠0 ⇒
>    `__all_sync` 整体判负）⇒ 整段掉到 `BytePerPack=sizeof(T)`（bf16=2 字节/包），
>    **没有头尾剥离**。该文件自 2023 实质未动 ⇒ 2.27.5 → **2.31.2（当前版）机制零变化**。
> 4. ★ **DeepSpeed 先例**：`stage_1_and_2.py` 的 `nccl_start_alignment_factor`
>    （注释原话 *"align nccl all-gather send buffers to 4-byte boundary"*）——
>    把 flat 组补到 `factor × world_size` 的倍数并**断言每个分区起点对齐**。
>    与本文 §6 的修法**同构**（他们 4 字节，NCCL 向量化要 16）。
> 5. ★★ **veScale-FSDP（arXiv 2602.22437，字节，2026-02）独立撞到并记载**：
>    *"both FSDP1 and FSDP2 suffer from slow collectives due to unaligned communication
>    buffer"* · *"FSDP1 and FSDP2 overlook NCCL address alignment caveat, leading to
>    substantial degenerate communication"* —— 引用键就叫 `[nccl16byte]`；
>    他们的分片规划按 "collective preferred unit size" 对齐。
>    ⇒ **FSDP2 也中招从我们的 [推断] 升级为有独立来源**；且故事从「没人知道」
>    变成「**工业界各自私下绕开（DeepSpeed/veScale），上游 tracker 是空白**」—— 更硬。
> 6. FSDP1 维护模式的墙（pytorch#154888 先例）对 `_flat_param.py` 同样成立
>    ⇒ 修法主推 **FSDP2**（活代码），FSDP1 同报但预期可被 not_planned。

---

## 0 · 一句话

**PyTorch FSDP 在切分参数时只保证「每个 rank 的元素数相等」，不保证「每个 rank 的字节数是 16 的倍数」；
而 NCCL 的 Simple 协议 kernel 在分块字节数不是 16 的倍数时，会把整段搬运退化成标量路径 ——
实测 `all_gather` 从 13.4 GB/s 掉到 1.1 GB/s（12.2×），而只要每 rank 多补 4 个字节就全部恢复。**

---

## 1 · 为什么这个 bug 到现在才被撞见（成立范围要老实写）

★ **这是一个 edge case，而且是「工业界很少踩到」的那一类**：

| 条件 | 主流集群 | 我们 |
|---|---|---|
| rank 数 | 8 / 16 / 64（**2 的幂次**） | **3**（3 训练 + 1 生成，最自然的切法） |
| 卡间互联 | NVLink / NVSwitch | **无 P2P**（GeForce 从 4090 起驱动关掉），走 SHM 经主机内存 |
| 协议 | 大消息也常走 LL128 / NVLS | 走 **Simple**（就是有 16 字节向量化的那条） |

⇒ **触发条件是「非 2 的幂次 rank 数」**：常见张量尺寸是 2 的幂次，
**÷4 天然保持 16 字节对齐，÷3 几乎必然破坏它**。
⇒ 消费级多卡（4×5090 这类）做训练的人本来就少，
其中用 3 卡分片的更少 —— 所以它一直没被系统性报告过。

⚠️ **但「少见」不等于「不该修」**：

> 分块不整齐时**只有零头需要特殊处理**，而现在的行为是**整段（几十 MB）一起降级为标量搬运**。
> 差 4 个字节，付 12 倍的代价 —— 这个**惩罚的形状本身是不合理的**，
> 和「多一行多一列导致 tile 不整齐、性能差几个百分点」是完全不同量级的事。

---

## 2 · 环境指纹（可复现的前提）

```
GPU        4 × NVIDIA GeForce RTX 5090（sm_120, 32 GB）
           ⚠️ can_device_access_peer 4×4 全 0 —— PCIe P2P 全关（GeForce 的常态）
           ⇒ NCCL 走 SHM/direct/direct，经主机内存中转
CPU/拓扑   2 socket EPYC 9V74；GPU0/1 挂 node0、GPU2/3 挂 node1；PCIe Gen5 x16
驱动/CUDA  595.58.03 / CUDA 12.8
torch      2.9.0+cu128
NCCL       2.27.5（torch 自带）
关键环境   NCCL_CUMEM_ENABLE=0（Ray 给每 worker 只设一张卡时必需，否则 SHM 传输起不来）
框架       verl 0.8.0，FSDP1（`strategy: fsdp`）+ LoRA r32，Qwen3-4B，bf16
```

---

## 3 · 最小复现（不需要训练框架，3 张卡 + PyTorch 即可）

```python
# scripts/probe_alignment_cliff.py（本仓库内，已参数化）
# 固定每 rank 分块 ≈ 24 MB，只改末尾几个字节，其余一切不变
CLIFF_BASE=25165824 CLIFF_OFFS=0,4,8,12,16,32,64,128 python scripts/probe_alignment_cliff.py
```

核心循环就是标准的 `dist.all_gather_into_tensor`：

```python
s  = torch.ones(per_rank_elems, dtype=torch.float32, device=rank)
g  = torch.zeros(per_rank_elems * world_size, dtype=torch.float32, device=rank)
for _ in range(15):
    dist.all_gather_into_tensor(g, s)
```

**实测（3 卡，NCCL 默认协议）**：

| 每 rank 字节 | `% 16` | `% 128` | **all_gather** | reduce_scatter |
|---|---|---|---|---|
| 25,165,824 | **0** | 0 | **13.2 GB/s** | 32.9 GB/s |
| 25,165,828 | 4 | 4 | **1.1 GB/s** | 20.8 GB/s |
| 25,165,832 | 8 | 8 | 1.5 GB/s | 25.9 GB/s |
| 25,165,836 | 12 | 12 | 1.1 GB/s | 20.8 GB/s |
| 25,165,840 | **0** | 16 | **13.2 GB/s** | 32.9 GB/s |
| 25,165,888 | **0** | 64 | **13.3 GB/s** | 32.9 GB/s |
| 25,165,948 | 12 | 124 | 1.1 GB/s | 20.8 GB/s |

⇒ **悬崖精确落在 `% 16 == 0` 上；`% 128` 完全不相干**（16/32/48/…/112 全都快）。
⇒ `all_gather` 掉 **12×**；`reduce_scatter` 只掉 1.3–1.6×（它带规约计算，访存不是唯一瓶颈）。

---

## 4 · 这不是合成场景：真实 FSDP 训练里 99.9% 的字节都踩在上面

在 verl 0.8.0 + FSDP1 + Qwen3-4B(bf16) + 3 卡 ZeRO-3 的**真实训练**上，
用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` 抓 NCCL 自己打印的每一次分块字节数，
**按字节加权**统计：

| 算子 | 调用数 | 总字节 | **`%16 != 0` 的字节占比** | 协议 |
|---|---|---|---|---|
| **AllGather** | 71,677 | 335.5 GB | **★ 99.9%** | RING + SIMPLE |
| ReduceScatter | 16,128 | 2.82 GB | 100.0% | RING + SIMPLE |
| Broadcast | 542 | 8.31 GB | **0.0%** | RING + SIMPLE |

★ **Broadcast 那一行是天然对照组**：同一次跑、同一条链路、同一个协议，
**0% 错位** —— 因为它**不按 rank 数切分**。
⇒ 错位不是机器属性，是「**除以非 2 幂次的 rank 数**」的产物。

体量最大的那一档，来源一眼可见：

```
每 rank 分块 67,287,212 B × 2376 次      67,287,212 % 16 = 12   🔴
  × 3 = 201,861,636 B ← 一层 transformer 的 flat parameter
  = 100,930,818 个 bf16 元素；每 rank 33,643,606 个，而 33,643,606 % 8 = 6
  ⇒ **每 rank 只要再补 2 个元素（4 个字节）就回到 16 字节对齐**
```

**在这个真实尺寸上验证补齐**（同一个探针，`CLIFF_BASE=67287212`）：

| 每 rank 字节 | `%16` | **all_gather** | reduce_scatter |
|---|---|---|---|
| **67,287,212**（FSDP 实际产生的） | 12 | **1.1 GB/s** | 20.9 GB/s |
| **67,287,216（+4 字节）** | **0** | **★ 13.4 GB/s** | **32.7 GB/s** |

⇒ **12.2×，代价是每 rank 4 个字节。**

**端到端影响**（3 卡 colocate，ZeRO-3，只改 NCCL 协议作为旁证）：

```
DDP（不分片）                        update_actor  7.97 s   1.00×
ZeRO-3 默认（走 Simple，踩悬崖）      update_actor 47.94 s   6.02×
ZeRO-3 + NCCL_PROTO=LL128（绕开）     update_actor 14.40 s   1.81×   ← 3.33× 提速
```
（LL128 之所以能绕开，是因为它按自己的 128 字节格式打包，不走那条 16 字节向量化路径；
代价是 `all_reduce` −30%、`broadcast` −41%，所以它是应急手段不是解法。）

---

## 5 · 责任在哪一层

| 层 | 它做了什么 | 对不对 | 问题 |
|---|---|---|---|
| **① NCCL 的 Simple kernel** | 分块字节数非 16 倍数 ⇒ 走标量路径 | 正确性没错 | 🔴 **整段退化，而不是「对齐主体向量化 + 标量尾巴」**。24 MB 只多 4 个字节，前面 25,165,824 字节也跟着走标量 |
| **② NCCL 的成本模型** | 按带宽/拓扑/消息大小选算法与协议 | 逻辑没错 | 🟠 **输入里没有「对齐」这一维** ⇒ 它看不见自己 kernel 的 12× 悬崖，**结构上不可能选对** |
| **③ PyTorch FSDP** | 把分片补到「每 rank 元素数相等」 | 它要解决的是「分得均匀」，做到了 | 🟠 **只保证整除，不保证字节对齐** ⇒ 造出下游会掉悬崖的尺寸 |

★ **最该修的是 ①**（纯收益、零取舍、不需要跨层协商，修好之后 ③ 也不用改）。
★ **最容易先落地的是 ③**（改动极小、可立即验证、对所有后端都安全）。
⇒ **建议：两个都提，先提 PyTorch，NCCL 那条引用它作为下游实证。**

---

## 6 · 建议的修法（PyTorch 侧）

**FSDP1** —— ⛔ **2026-08-19 更正：修的层换了（原提案证伪，保留示错）。**
原提案是在 `_get_shard` / `_get_unpadded_shard`（**分片时**）pad —— **实测会崩**：
梯度簿记走另一条路按原始 numel 均分（81921÷3=27307 恰好整除），与被 pad 过的初始化分片
（27308）差 1 个元素，第一次反向就 `assigning a gradient of size '[27307]' to a tensor of
size '[27308]'`。**运行时在分片函数里 pad，改不全所有算尺寸的地方。**

✅ **正确的层是构造端** —— `_init_flat_param_and_metadata` 里**本来就有**
「补到 world_size 整除」的收尾 padding（自带注释 *"to avoid a copy for the post-backward
reduce-scatter"*），把除数换掉即可，总 numel 自身对齐后所有下游路径自然一致：

```python
# _flat_param.py:737（孪生站点 :853 同改）
align        = self.world_size * max(1, 16 // _get_dtype_size(dtype))   # bf16→24, fp32→12 (world=3)
numel_to_pad = align - (total_numel % align)
if numel_to_pad > 0 and numel_to_pad < align:
    ...                                        # 原有的 padding tensor 机制原样复用
```

⚠️ 该收尾 padding 块被 `aligned_numel > 0` 的门罩着 = **只在 `use_orig_params=True`
构造路径上存在**；`=False` 的路径连"补到 world 整除"都没有（分片时才对末 rank 补齐）
⇒ 那条路要修得把同样的收尾 padding 加进去。（A17-v2 就是被这道门挡住白跑了一轮 ——
verl 这条 engine 路径实测 `use_orig_params: False`。）

**FSDP2** —— `torch/distributed/fsdp/_fully_shard/_fsdp_common.py::_get_dim0_padded_size`
同样只补到 `dim0_factor`（= world_size）的倍数，问题同形。

**代价**：每个 flat parameter 最多多 `world_size × elems_per_16B - 1` 个元素
—— 本例中一层多 **12 个字节**（约 6e-8 的显存开销）。
**收益**：该层的 `all_gather` 最多快 **12×**。

---

## 7 · ⚠️ 还没做的一步（提交前要写清楚，或者补做）

**目前的证据链是三段**：
1. 机制：合成尺寸上，`%16` 是唯一的开关（§3）
2. 现场：真实 FSDP 训练里 99.9% 的字节踩在上面（§4）
3. 修复：**在真实那个尺寸上**补 4 字节，恢复 12.2×（§4）

~~缺的是第 4 段~~ ✅ **第 4 段已补（2026-08-19，A17-v3）**：构造端 diff 打进 torch 源码树，
colocate 3 卡 ZeRO-3 双臂（都 `use_orig_params=True` + 强制 Simple，唯一变量 = padding 除数）：

```
update_actor   100.0 → 27.8 s   3.60×      判据行 [A17-src] 541 条 vs 0 条
ref 前向        41.6 →  6.1 s   6.85×      （纯前向最接近微基准的 12×；update_actor 有算力 Amdahl）
old_log_prob    44.6 →  9.2 s   4.84×
gen（无 FSDP）   16.4 → 18.5 s   ~1×        对照组自洽
双臂 exit 0 · rewards 同量级 · 跑完 torch 已还原 stock
```

⚠️ 与 §4 的 47.94/14.40 **不同尺** —— 那对是旧预算 + `use_orig_params=False`；
这对是新预算 + `=True`。两对各自同尺内可比，跨对不可比。
⇒ 本仓库排成 **A17**。**不做也能提**（前三段已经互相独立且都可复现），
但补上之后 issue 会强很多 —— 从「这里有个悬崖」变成「**改这一行，端到端快 N 倍**」。

---

## 8 · 提交清单（真要提的时候照做）

- [ ] 决定仓库：`pytorch/pytorch`（FSDP padding）+ `NVIDIA/nccl`（kernel 整段退化）
- [ ] 把 `scripts/probe_alignment_cliff.py` 精简成**不依赖本仓库**的单文件复现脚本
- [ ] 复述 §2 环境指纹（尤其 **no-P2P + 3 rank** 这两个触发条件）
- [ ] 附 §3 的悬崖表 + §4 的 Broadcast 对照组（**这两张表是说服力的核心**）
- [ ] 若已完成 A17，补 §7 的端到端数字
- [ ] ⚠️ 提交前再核一遍 torch 主干是否已改（本文基于 2.9.0）
