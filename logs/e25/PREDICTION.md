# E25 · 预测（2026-08-19，跑之前写死）

## 判据前提：保真度自检
baseline（gc=on, micro_batch=1, 16 条）必须落在生产实测 **17.56 s** 附近（±30%）。
⛔ 落不上 ⇒ 探针不保真，本次所有相对结论作废，不许只报相对值。

## P1 · 关掉 gradient_checkpointing
fwd+bwd **快 20–30%**。依据：GC 的代价是反向时重算一次前向；
无 GC 时 fwd+bwd ≈ 3×fwd，有 GC 时 ≈ 4×fwd ⇒ 去掉省 1/4。
显存峰值涨到 **25–30 GB**（生产 gc=on 时 15.55 GB）。
⇒ 若快得**明显少于 20%**，说明瓶颈不在重算，而在别处（启动/访存）——那本身更值钱。

## P2 · micro_batch 1 → 2
提升 **< 10%**。依据：一条序列已有 4850 token，不算小 matrix，喂不饱的空间有限。
⇒ 若提升 **> 25%**，说明 micro_batch=1 确实在挨饿，那 E01 那 51% 的 0.1–1ms 空档就有解释了。

## P3 · 显存天花板
gc=off + micro_batch≥2 **会 OOM**（32 GB）。
gc=on 时 micro_batch 能到 **4 或 8**。

## P4 · fwd_only（old_log_prob / ref 的代理）
不受 GC 影响（no_grad 下 GC 不生效），只受 micro_batch 影响，且提升幅度与 P2 同向。
