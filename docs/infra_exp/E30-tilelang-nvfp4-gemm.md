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

11 组 block_m/n/k × stages 扫描（`logs/a3_nvfp4_sweep1.log`）：

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

## 4 · 2026-08-27 · MXFP8 正餐：自有 mxf8f6f4 kernel 首通（v0.2 = 543 TFLOPS，超 cuBLAS）

**这次是我们自己的 kernel**（`scripts/tl_mxfp8_gemm.py`）：TileLang 管分块/TMA/流水，
MMA 内层 = 自写设备函数发射 `mma.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X`
（T.import_source 逃生门，绕开 tilelang C++ 校验只认 NVFP4 的限制）。

**三件此前不存在的东西**：
① **缩放 lane 映射硬件反演**（`scripts/probe_mxf8_scale_mapping.cu`，全域唯一指数法）：
   A 行 r(<8)←lane(4r+tid·2)·byte(bid)，行 r+8←lane(4r+tid·2+1)；B 列 n←lane(4n+tid)。
   bid=字节窗 ⇒ 一个 uint32 装 4 个 k-block 缩放，内层 4 发 mma 各用 bid 0..3；
② **MXFP8 量化器**（OCP：块 32 · ue8m0 幂缩放 · e4m3, fp8_max=448）+ 逐块反量化 fp32 参考；
③ **kernel 原生数据布局**：行内 128B 组 16B 块 `chunk^=(row&7)` 预置换（宿主做），
   TMA 线性搬运，设备 ldmatrix 地址同款 XOR ⇒ bank 冲突消除。

**性能阶梯（8192³，全部 --verify 过，err ~2.9e-3 = bf16 输出舍入带）**：

| 版本 | TFLOPS | 距峰 1026 |
|---|---|---|
| v0.1 标量 smem 加载 | 296 | 28.9% |
| + ldmatrix（x4/x2） | 293 | ——（指令数非瓶颈） |
| **+ XOR swizzle（v0.2）** | **543** | **52.9%** |

⇒ **543 > cuBLAS-cu13 523（老指令路径顶棚）**：本机实测最快 FP8 GEMM，新指令路径首个可用实现。

**踩到的上游坑（两个，各值一个 issue）**：
- TileLang warp 特化把消费者放在 threadIdx 256..511——extern 设备函数必须用 `tx&255` 取
  逻辑线程号（首跑 nan 的根因）；
- 撤掉任一 smem 拷贝会让生产者分区缩到 128 线程而释放 barrier 计数仍 256 ⇒
  相位竞态、散点错块（4096+ 才显形，2048 以下侥幸过——**「小尺寸对拍过」不等于「没竞态」**）。
  已在 kernel 注释钉死"缩放必须走 smem"。

## 5 · 峰值冲刺三连：两负一定界（负结果照登）

| 弹药 | 结果 | 病因（实测） |
|---|---|---|
| 块 128×256（B 复用×2） | 🔴 94 TFLOPS | ptxas 被 launch_bounds(512) 钉在 128 寄存器/线程，acc 128+fragment≈190 ⇒ 溢出 1124B（-Xptxas -v 实锤）；`annotate_*_reg_alloc` 强开重分配在 256/256 分区**挂死**；threads=384 更糟（14.5）——正路被上游卡（DRAFT-tilelang-warpspec-barrier-count 关联） |
| 3 级流水（SFB 转全局挤 smem） | 🔴 527 < 543 | 流水不是瓶颈；全局缩放读还倒贴 |
| ldmatrix（§4 已记） | 🔴 293≈296 | 指令数不是瓶颈 |

⇒ **v0.2 = 543（52.9%）就是本骨架在"512 线程/128 寄存器"约束下的顶**；再往上唯一解锁
= 更宽 warp tile + 寄存器重分配，被 TileLang 上游机制挡住（已入 DRAFT 包，修复后可复活）。

## 6 · 瘦长形状（lm_head/稀疏投影）：轴序一改 +32%

病根（算的账，实测证实）：N=151936 时 B 权重 194MB；默认 N 维最快轴 ⇒ 每个 M 块把 B
从显存整个重拉一遍（32×194MB≈6.2GB）。**M 维改为 blockIdx.x（`--grid-m-fast`）**：
同一 B tile（320KB）被 32 个连续 M 块复用而常驻 L2，B 流量掉回 ~194MB。

| 形状 | 默认轴序 | M 最快轴 | 备注 |
|---|---|---|---|
| 4096×151936×2560 | 370 | **487（+32%）** | +sfb-global 无增益（486） |
| **256×151936×2560**（真实稀疏投影 M） | — | **440** | 访存受限形态的强读数 |

对拍全程 3.1–3.2e-3 ✅。⇒ FP4/FP8 在此形态的本质红利=权重流量减半/减四，
与稀疏投影融合（gather 进 kernel）挂 A4 线，等训练侧接线时一并做。

## 7 · 收官状态（2026-08-27）与复活条件

```
✅ 已交付   MXFP8 GEMM v0.2：8192³ 543 TFLOPS=峰值 52.9%（超 cuBLAS-cu13 523）·
            瘦长形状 487/440 · 对拍常驻 · 负结果三连入档 · 上游两 DRAFT 包
⏸ 停放     >53% 峰值：等 TileLang 寄存器重分配修复（DRAFT 包）或从源码构建自改
⏸ 停放     NVFP4 版自有 kernel（上游示例已 63.6%，我们复刻收益有限）；MXFP8 移植进
            tilelang 正规路径 = 上游 PR（DRAFT-tilelang-sm120-mxfp8-support）
→ 接力     稀疏投影融合（A4/E11 训练侧接线时做，调用点在我们手里）
```
