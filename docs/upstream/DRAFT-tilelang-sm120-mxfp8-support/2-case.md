# tilelang · sm120 块缩放 MMA 只接了 NVFP4：MXFP8（mxf8f6f4）支持缺位 + 可用原型

```
状态     DRAFT —— 待上游考据（Claude：确认上游是否已有在飞 PR + 按贡献指南把原型改造成配置表扩展）
目标仓库  tile-ai/tilelang（0.1.13）；关联 NVIDIA/cutlass#2867（同缺口）、triton#7550（同族已修）
性质     feature（带可用原型与硬件实测依据，非 bug）
原始发现  ../../infra_exp/E30-tilelang-nvfp4-gemm.md §1/§4
原型      scripts/tl_mxfp8_gemm.py（8192³ 543 TFLOPS=sm120 峰值 52.9%，超 cuBLAS-cu13 523；对拍 2.9e-3）
```

## 缺口

`_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS`（mma_sm120_macro_generator.py）仅一条 mxf4nvf4；
C++ 侧 codegen_cuda.cc `ICHECK(supported_mxf4nvf4_4x_ue4m3)` 硬拒其它 kind。
MXFP8（m16n8k32 · kind::mxf8f6f4 · scale_vec::1X · e4m3 · ue8m0）完全没接。

## 我们手里可直接给上游的三样

① **缩放因子 lane 映射（硬件反演，非文档抄录）**：scripts/probe_mxf8_scale_mapping.cu，
   A 行 r(<8)←lane(4r+tid·2)·byte(bid)，行 r+8←+1；B 列 n←lane(4n+tid)·byte(bid)；
   bid=字节窗 ⇒ uint32 一次装 4 个 k-block 缩放——这正是配置表扩展要填的 selector 逻辑；
② **工作原型**：import_source 逃生门版完整 kernel（对拍+性能双验），上游要做的是把它
   移植进正规路径（配置表 + 宏生成器 + codegen ICHECK + mma_block_scale.h 模板）；
③ 峰值/基线锚点：PTX 峰 1026 · cuBLAS-cu13 523 · triton 主线 304（E16 §7/§8），PR 描述可引用。

## 给上游同事的任务

① 查 tilelang 是否已有 sm120 mxf8 的 issue/PR 在飞（避免撞车；triton 侧同族修复者 ita9naiwa 也活跃在这类缺口）；
② 决定形态：feature request issue（附我们的映射+原型数据）还是直接 PR（工作量大：四层都要动，见 E30 §1 层次图）；
③ 提交名义与署名按本目录守则 §4。
