# E30 · A3 正餐 · TileLang 块缩放 GEMM（sm_120）

> 判据（01 §1-3 立）：**对拍等价 + 距 1026(FP8)/2055(FP4) 峰值的百分比**。
> 环境：隔离 venv `/workspace/venvs/tilelang`（tilelang 0.1.13 · torch 2.13+cu130），
> 上游示例与垫层在 `/workspace/a3_tilelang/`（守则⑧：不碰生产 venv）。
> 前情：E16 §7 峰值尺（PTX 实测 1026/2055）· E16 §8 库基线（cuBLAS-cu13 523 · Triton 主线 fp8 304/fp4 519）。

## 1 · 2026-08-27 · 第一天：工具链验证 + 基线树立（跑的是上游 kernel，不是我们的）

**侦察结论**：tilelang 0.1.13 的 sm120 块缩放路径**只接了 NVFP4**
（`mma_gemm_blockscaled`，m16n8k64.kind::mxf4nvf4，E2M1 操作数 + UE4M3 缩放，
配置表 `_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS` 仅一条）；**MXFP8 未接线**
⇒ FP8 GEMM 要么扩展该配置表（= 给 TileLang 的 PR 入口，与 CUTLASS #2867 同缺口），要么先做 FP4。

**上游官方示例实跑**（`examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`，--verify 全过）：

| 形状 | TFLOPS | 距 FP4 峰 2055 |
|---|---|---|
| 2048³ | 689 | 33.5% |
| 4096³ | 1173 | 57.1% |
| **8192³** | **1308** | **63.6%** |
| lm_head 4096×151936×2560 | 600 | 29.2%（瘦长形状，未调） |

官方基准（warp 特化版 + 真 bf16 量化输入）8192³ = 1261，与示例互证。

**当日阶梯**（同卡）：`bf16 258 → cuBLAS FP8 523 ≈ Triton主线 FP4 519 → TileLang NVFP4 1308 → 峰 2055`
⇒ 上游默认配置已比最好的库路径快 **2.5×**；剩余 36% 是手写空间。

## 2 · 配置扫描：默认即最优，墙是共享内存

11 组 block_m/n/k × stages 扫描（`/workspace/a3_tilelang/sweep1.log`）：

```
8192³   128/128/256/s2 = 1266 最优;  64/256 掉到 1111;
        所有更大 tile 或 s3 ⇒ "Failed to set dynamic shared memory 110592"
        —— 🔴 sm_120 消费卡共享内存 ~100KB/块（数据中心 227KB），是硬墙
lm_head block_n=256 全断言死：151936=128×1187（1187 奇数）不能被 256 整除
```

⇒ **结论①**：现 schedule 的参数空间已被 smem 预算钉死，1266–1308 就是它的顶。
   再往上 = 在 100KB 内重排流水线（TMA 直达 smem、更细的 stage 切分、
   缩放字压缩布局）——这是"我们的 kernel"要做的事，不是调参能到的。
⇒ **结论②**：lm_head/稀疏投影是"小 M × 巨 N"访存受限形态，与方形 GEMM 是两个问题；
   FP4 权重把 B 流量砍半在该形态**天然对症**，kernel 设计要按访存而非算力做。

## 3 · 下一步（按序）

1. 读懂上游 schedule（smem 布局/缩放字打包/发射循环）→ 自己的变体冲 >70% 峰值；
2. lm_head 形状专用 schedule（N 巨大 ⇒ B 常驻 L2 分条带 + 输出写合并）；
3. MXFP8 配置表扩展（mxf8f6f4 / ue8m0 / m16n8k32）→ 上游 PR 候选；
4. 出成果后回帖 triton#7550 / CUTLASS#2867（E16 §8 圈子地图）。
