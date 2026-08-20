# 提交件 · PyTorch issue + NCCL #413 评论（可直接粘贴）

> 状态：**草稿完成，等 Chaoyu 过目后提交**（2026-08-19）。
> 中文分析与证据链 → [`analysis.md`](analysis.md)。
> 两个动作：① `pytorch/pytorch` 提 issue（FSDP1+FSDP2 同报，修法主推 FSDP2）；
> ② `NVIDIA/nccl` **在 #413 下评论请求重开**（不提新 issue —— 维护者 2020 年已确认，
> 机器人关闭时留了"仍相关请重开"的口）。①②互相引用。
> 〔A17〕标记 = 等 A17 双臂跑完回填端到端数字。

---

## 1 · PyTorch issue

**Title:**

```
FSDP1/FSDP2 shard sizes ignore NCCL's 16-byte alignment — all_gather degrades
up to 12x for non-power-of-two world sizes (+4 bytes fixes it)
```

**Body:**

````markdown
## Summary

Both FSDP1 and FSDP2 pad shards only so that every rank gets an *equal element
count*. Neither guarantees that a rank's shard is a multiple of **16 bytes** —
the vectorization unit of NCCL's Simple-protocol kernels. When the per-rank
size is not 16B-aligned, NCCL demotes the **entire transfer** to
element-granularity copies:

- microbenchmark: `all_gather` **13.4 → 1.1 GB/s (12.2×)** from a 4-byte size change
- real FSDP1 training (Qwen3-4B, 3-GPU ZeRO-3): **99.9% of all all-gathered
  bytes** (335.5 GB over 71,677 calls) land on misaligned sizes
- real FSDP2 sharding of one Qwen3-4B decoder layer: per-rank all-gather buffer
  is 67,319,300 B (`% 16 = 4`); at that exact size, bf16 `all_gather` runs at
  **2.15 GB/s vs 27.03 GB/s** for the same buffer padded by **12 bytes** (12.56×)

The trigger is simply a **non-power-of-two world size** (3, 5, 6, 7 …): typical
tensor dims divide evenly by 2/4/8 and stay aligned, while ÷3 almost always
breaks 16B alignment. This is why the issue survives at scale-up clusters and
bites asymmetric setups (e.g. 3 trainer + 1 rollout GPU in RL frameworks).

This is a known NCCL caveat that frameworks are expected to handle:
[NVIDIA/nccl#413](https://github.com/NVIDIA/nccl/issues/413) (2020) — the NCCL
lead confirmed the behavior and prescribed the fix to callers: *"**padding is a
solution that would always work well**"*. Ecosystem projects already comply
privately: DeepSpeed pads its flat groups to `alignment_factor × world_size`
and asserts partition-start alignment (`nccl_start_alignment_factor`,
*"align nccl all-gather send buffers"*); the veScale-FSDP paper
(arXiv:2602.22437) states outright that *"both FSDP1 and FSDP2 suffer from slow
collectives due to unaligned communication buffer"* / *"FSDP1 and FSDP2
overlook NCCL address alignment caveat"* and re-plans sharding around it.
Everyone pays this tax or works around it downstream; the padding belongs here.

## Mechanism (NCCL side, for reference)

`nccl/src/device/common_kernel.h` (unchanged from 2.27 through current 2.31.2):
`BigPackSize = 16`; a warp-wide vote checks *every* src/dst pointer, and a
single misaligned pointer sends the **whole segment** down the
`BytePerPack=sizeof(T)` path (2-byte packs for bf16) — there is no head/tail
peeling. In ring all-gather the chunk boundaries are multiples of the per-rank
size, so per-rank size `% 16 != 0` ⇒ misaligned pointers ⇒ the cliff. The cost
model has no alignment input, so protocol selection cannot route around it.

## Where the sizes come from

**FSDP1** — `_flat_param.py::FlatParamHandle._get_unpadded_shard` splits the
flat parameter with a bare `.chunk(world_size)`: element-granularity, byte
alignment of the boundary is whatever it happens to be. Measured on a real
Qwen3-4B layer: per-rank chunk 67,287,212 B (`% 16 = 12`); padding each rank by
**4 bytes** restores 12.2×. Notably, `_flat_param.py` already contains
`ALIGNMENT = 16  # bytes` (`_get_aligned_numel`, used to align *intra*-flat-param
boundaries for TorchInductor when `use_orig_params=True`) — the constraint is
known in this very file; it just stops at the shard boundary.

**FSDP2** — `_fsdp_common.py::_get_dim0_padded_size` pads dim-0 only to a
multiple of `dim0_factor` (= shard world size). 2-D weights with 16B-aligned
row bytes stay aligned, but **1-D parameters poison the group buffer**: for one
Qwen3-4B decoder layer at world=3, the per-parameter shard bytes are

```
self_attn.q_norm.weight            [128]    86 B   % 16 = 6 self_attn.k_norm.weight            [128]    86 B   % 16 = 6 input_layernorm.weight             [2560]  1708 B  % 16 = 12 post_attention_layernorm.weight    [2560]  1708 B  % 16 = 12 (all 2-D projection weights: row-aligned, % 16 = 0) ──────────────────────────────────────────────────────────── group all-gather input per rank:  67,319,300 B   % 16 = 4
```

(predicted from the sharding math and confirmed by intercepting the actual
`all_gather_into_tensor` call issued by `fully_shard` + `unshard()`; base
pointers are 16B-aligned — the misalignment enters purely through the per-rank
size, i.e. the ring chunk boundaries.)

## Minimal reproduction (any 3 GPUs)

```python
# All that matters is the per-rank byte count mod 16. torchrun --nproc_per_node=3:
import os, time, torch, torch.distributed as dist dist.init_process_group("nccl") rank = dist.get_rank(); torch.cuda.set_device(rank); world = dist.get_world_size() for per_bytes in (67_287_212, 67_287_216):          # FSDP1's real chunk, then +4 B n = per_bytes // 4 s = torch.ones(n, dtype=torch.float32, device=rank) g = torch.zeros(n * world, dtype=torch.float32, device=rank) for _ in range(3): dist.all_gather_into_tensor(g, s) torch.cuda.synchronize(); t = time.perf_counter() for _ in range(15): dist.all_gather_into_tensor(g, s) torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 15 bw = per_bytes * world * (world - 1) / world / dt / 1e9 if rank == 0: print(f"per-rank {per_bytes:,} B (%16={per_bytes%16}): {bw:.1f} GB/s")
```

Measured (3× RTX 5090, PCIe, no P2P, NCCL 2.27.5, default protocol):
`% 16 = 12` → 1.1 GB/s; `% 16 = 0` → 13.4 GB/s. The cliff tracks `% 16 == 0`
exactly (`% 128` is irrelevant; 16/32/48/… are all fast).

End-to-end on the same machine (3-GPU ZeRO-3; two independent same-ruler pairs,
do not compare across them — configs differ):

```
LL128 sidestep        update_actor 47.9 s -> 14.4 s (NCCL_PROTO=LL128 dodges the 16B path, but costs
                       -30% all_reduce / -41% broadcast elsewhere)
shard padding alone   update_actor     100.0 -> 27.8 s   (3.6x) (this proposal)       ref fwd           41.6 ->  6.1 s   (6.9x) old_log_prob      44.6 ->  9.2 s   (4.8x) rollout generation (no FSDP collectives): unchanged
```

## Proposed fix

Pad shards so each rank's shard is a multiple of 16 bytes. The cost is at most
`world_size × 16` bytes per flat-param/group — order 1e-7 relative overhead —
and the padding is inert (FSDP already owns the "trailing padding belongs to no
parameter" semantics).

- **FSDP2** (`_get_dim0_padded_size`): pad dim-0 to a multiple of
  `dim0_factor × (16 // gcd(row_bytes, 16))` where
  `row_bytes = prod(size[1:]) × element_size` (= `element_size` for 1-D). For
  the layer above this pads the four 1-D params by 8–16 rows (+12 B per rank in
  total) and recovers 12.56×. We are happy to send a PR if this direction is
  acceptable (the padded size flows into DTensor metadata / copy-out views, so
  we would like a maintainer's read on the preferred plumbing first).
- **FSDP1** (`_flat_param.py::_init_flat_param_and_metadata`): the construction
  path already appends end padding to make the flat parameter divisible by
  `world_size` (its own comment: *"to avoid a copy for the post-backward
  reduce-scatter"*). Changing that divisor to `world_size × (16 // itemsize)`
  makes every rank's shard 16-byte aligned through the exact same mechanism.
  Validated in real training with only this diff (numbers above; the padding
  amounts to +4..+6 elements per flat parameter).
  Two details worth recording:
  - that end-padding block is gated on the `use_orig_params=True` construction
    path; the `use_orig_params=False` path performs no construction-time end
    padding at all and would need the same treatment;
  - padding at *shard* time instead (`_get_unpadded_shard`) is **not**
    sufficient — we tried it first, and gradient bookkeeping derives shard
    sizes independently, crashing on the first backward with
    `attempting to assign a gradient of size '[27307]' to a tensor of size
    '[27308]'`. The flat parameter's own numel has to carry the alignment.

  (We understand FSDP1 is in maintenance mode; reporting for completeness since
  it is the default backend of downstream frameworks such as verl.)

## Environment

3–4 × RTX 5090 (PCIe, no P2P — NCCL via SHM), torch 2.9.0+cu128, NCCL 2.27.5;
relevant torch code paths verified unchanged on current `main`; NCCL kernel
verified unchanged on master (through v2.31.2).
````

---

## 2 · NCCL #413 重开评论

**贴在 [NVIDIA/nccl#413](https://github.com/NVIDIA/nccl/issues/413) 下：**

````markdown
Still relevant on 2.27.5 (and the kernel path is unchanged on master through
v2.31.2 — `src/device/common_kernel.h`: `BigPackSize = 16`, warp-wide
alignment vote, whole-segment fallback to `BytePerPack=sizeof(T)`).

Fresh data (3× RTX 5090, PCIe/SHM, no P2P, fp32 ring all_gather, default
protocol): changing the per-rank size by 4 bytes moves `all_gather` between
13.4 GB/s (`% 16 == 0`) and 1.1 GB/s — **12.2×**. The cliff tracks
`% 16 == 0` exactly.

This still bites real workloads through PyTorch FSDP: both FSDP1 and FSDP2 pad
shards for equal element counts, not byte alignment, so any non-power-of-two
world size produces misaligned per-rank sizes (measured: 99.9% of all-gathered
bytes in a 3-GPU FSDP1 run; a 12.56× cliff at FSDP2's real group size, fixable
by +12 bytes). We are filing a PyTorch issue proposing shard padding
(<link>) — as prescribed here in 2020 ("padding is a solution that would
always work well"), and as DeepSpeed already does (`nccl_start_alignment_factor`).

Two asks, either of which would defuse the trap for everyone who has not
padded yet:

1. **Head/tail peeling in the fallback**: process the unaligned head/tail at
   element granularity and keep the 16B-aligned bulk on the fast path, instead
   of demoting the entire segment. A 24 MB transfer that is 4 bytes short of
   aligned currently pays the scalar path for all 24 MB.
2. Failing that, **document the 16B preference** prominently (the tuner/cost
   model does not see alignment, so protocol selection cannot route around it,
   and `NCCL_PROTO=LL128` as a workaround costs −30% all_reduce / −41%
   broadcast elsewhere).
````

---

## 3 · 提交时的注意事项

- [ ] PyTorch issue 先发，拿到链接后填进 NCCL 评论的 `<link>`
- [x] 〔A17〕已回填（2026-08-19，v3 双臂 use_orig_params=True，唯一变量=构造端 padding 除数）：
update_actor 100.0→27.8 s（3.6×）· ref 41.6→6.1 s（6.9×）· olp 44.6→9.2 s（4.8×）· gen 不变（对照组自洽）· 判据行 541 条 vs 0 条 · 双臂 exit 0、rewards 同量级 ⚠️ v1（分片时 pad）已证伪：梯度簿记另算尺寸，首次反向 27307 vs 27308 崩 —— 教训写进了正文
- [ ] FSDP2 修法 PR 明确写"先要 maintainer 对 plumbing 的意见再发"——padded size 牵动
DTensor 元数据与 copy-out 视图，不宜自作主张
- [ ] 与包①②不同：这条**不影响我们自己的生产路径**（我们已定 DDP 不分片），
正文别写受害叙事，写"community report + 修法已在真实训练验证"
