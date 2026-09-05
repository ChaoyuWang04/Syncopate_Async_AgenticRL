"""A3/E30 · MXFP8 块缩放 GEMM（sm_120，TileLang + 自有 mxf8f6f4 设备内核）。

TileLang 0.1.13 的 sm120 块缩放路径只接了 NVFP4（kind::mxf4nvf4），MXFP8 配置表为空。
本文件不等上游：TileLang 负责分块/共享内存/软件流水，MMA 内层用我们自己的设备函数
（T.import_source）直接发射 `mma...kind::mxf8f6f4.block_scale.scale_vec::1X`。

缩放因子 lane 映射（scripts/infra/probe_mxf8_scale_mapping.cu 在硬件上反演，2026-08-27）：
    A 行 r(<8) ← lane(4r+tid·2)·byte(bid)；行 r+8 ← lane(4r+tid·2+1)·byte(bid)
    B 列 n     ← lane(4n+tid)·byte(bid)
    ⇒ 取 tid=0；SFA/SFB 每行 4 个 k-block 的 ue8m0 字节连续存放 ⇒ 一个 uint32
      装 4 个 k-block 的缩放，内层 4 发 mma 各用 bid=0..3（asm 立即数，模板实例化）。

量化标准：MXFP8（OCP Microscaling）——32 元素/块共享一个 ue8m0（2 的幂）缩放；
元素 e4m3（fp8_max=448）。判据 = 对拍等价（vs fp32 逐块反量化参考）+ 距峰值 1026 的 %。

用法（隔离 venv，守则⑧）：
    /workspace/venvs/tilelang/bin/python -m syncopate.train.tilelang_mxfp8 --m 8192 --n 8192 --k 8192 --verify
"""

import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

# ────────────────────────── 设备侧：mxf8f6f4 warp MMA ──────────────────────────
# 布局约定（全部 k 连续）：
#   A_sh  [BM][BK]  uint8 e4m3（行主序）      B_sh  [BN][BK] uint8（即 B^T，n 行 k 连续）
#   SFA_sh[BM][BK/32] uint8 ue8m0             SFB_sh[BN][BK/32]
#   acc   每线程 float[MI*NI*4]（m16n8 标准 fragment，行 g/g+8 · 列 q*2/+1）
MXF8_DEVICE_SRC = r"""
#include <cstdint>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

// 一发 m16n8k32 mxf8f6f4 mma；KB = k-block 序号 = 缩放字节窗口 bid（立即数）
template <int KB>
__device__ __forceinline__ void tl_mxf8_mma_atom(
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1,
    const uint32_t sa, const uint32_t sb, float* d) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
      ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
      "{%10},{%11,0},{%12},{%11,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sa), "n"(KB), "r"(sb));
}

// 一个 warp 处理 warp_tile 64x32，BK=128（4 个 k-block），跨 8 warp 铺满 128x128
extern "C" __device__ void tl_mxf8_warp_gemm(
    const uint8_t* __restrict__ A_sh, const uint8_t* __restrict__ B_sh,
    const uint8_t* __restrict__ SFA_g, const uint8_t* __restrict__ SFB_g,
    float* __restrict__ acc, int lda, int ldb,
    int sfa_lds, int sfb_lds, int sfa_base_row, int sfb_base_row,
    int sfa_col, int sfb_col) {
  // ★ TileLang 会把带 TMA 流水的 kernel warp 特化成 512 线程（低 256 生产/高 256 消费），
  //   消费者的逻辑线程号 = threadIdx.x & 255（flat 256 线程时是恒等映射）
  const int tx = threadIdx.x & 255;
  const int lane = tx & 31, w = tx >> 5;
  const int warp_m = (w >> 2) * 64;      // 2 行 warp × 64
  const int warp_n = (w & 3) * 32;       // 4 列 warp × 32
  const int g = lane >> 2, q = lane & 3;


  const int l = lane;
  // ldmatrix 每 lane 行地址：A x4 → 行 (l&15)、字节窗 (l>>4)*16；B x2 → 行 (l&7)、字节窗 ((l>>3)&1)*16
  const int a_row_off = l & 15, a_byte_off = (l >> 4) << 4;
  const int b_row_off = l & 7,  b_byte_off = ((l >> 3) & 1) << 4;

  // 寄存器化 fragment：先把本 chunk 全部 A/B/缩放取齐，再连发 mma（减少访存-计算互锁）
  uint32_t af[4][4][4];   // [mi][kb][reg]
  uint32_t bf[4][4][2];   // [ni][kb][reg]
  uint32_t sfa_r[4], sfb_r[4];

  #pragma unroll
  for (int mi = 0; mi < 4; mi++) {
    const int sfa_row = warp_m + mi * 16 + g + ((q & 1) ? 8 : 0);
    sfa_r[mi] = *reinterpret_cast<const uint32_t*>(SFA_g + (int64_t)(sfa_base_row + sfa_row) * sfa_lds + sfa_col);
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
      const int ar = warp_m + mi * 16 + a_row_off;
      const int ac = (kb * 32 + a_byte_off) >> 4;                 // 16B chunk 序号
      const int acs = (ac & ~7) | ((ac & 7) ^ (ar & 7));          // 128B 组内 XOR swizzle
      const uint8_t* p = A_sh + (int64_t)ar * lda + (acs << 4);
      const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(af[mi][kb][0]), "=r"(af[mi][kb][1]), "=r"(af[mi][kb][2]), "=r"(af[mi][kb][3])
                   : "r"(saddr));
    }
  }
  #pragma unroll
  for (int ni = 0; ni < 4; ni++) {
    const int col = warp_n + ni * 8;
    sfb_r[ni] = *reinterpret_cast<const uint32_t*>(SFB_g + (int64_t)(sfb_base_row + col + g) * sfb_lds + sfb_col);
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
      const int br = col + b_row_off;
      const int bc = (kb * 32 + b_byte_off) >> 4;
      const int bcs = (bc & ~7) | ((bc & 7) ^ (br & 7));
      const uint8_t* p = B_sh + (int64_t)br * ldb + (bcs << 4);
      const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                   : "=r"(bf[ni][kb][0]), "=r"(bf[ni][kb][1])
                   : "r"(saddr));
    }
  }
  #pragma unroll
  for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      float* d = acc + (mi * 4 + ni) * 4;
      tl_mxf8_mma_atom<0>(af[mi][0][0], af[mi][0][1], af[mi][0][2], af[mi][0][3],
                          bf[ni][0][0], bf[ni][0][1], sfa_r[mi], sfb_r[ni], d);
      tl_mxf8_mma_atom<1>(af[mi][1][0], af[mi][1][1], af[mi][1][2], af[mi][1][3],
                          bf[ni][1][0], bf[ni][1][1], sfa_r[mi], sfb_r[ni], d);
      tl_mxf8_mma_atom<2>(af[mi][2][0], af[mi][2][1], af[mi][2][2], af[mi][2][3],
                          bf[ni][2][0], bf[ni][2][1], sfa_r[mi], sfb_r[ni], d);
      tl_mxf8_mma_atom<3>(af[mi][3][0], af[mi][3][1], af[mi][3][2], af[mi][3][3],
                          bf[ni][3][0], bf[ni][3][1], sfa_r[mi], sfb_r[ni], d);
    }
}


// 诊断变体：直读全局内存（无 smem/无流水）——用于隔离 TileLang smem 布局嫌疑
extern "C" __device__ void tl_mxf8_warp_gemm_g(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
    float* __restrict__ acc, int base_m, int base_n, int ko,
    int K, int BK) {
  const int sfk = K / 32;
  tl_mxf8_warp_gemm(A + (int64_t)base_m * K + ko * BK,
                    B + (int64_t)base_n * K + ko * BK,
                    SFA, SFB, acc, K, K,
                    sfk, sfk, base_m, base_n, ko * (BK / 32), ko * (BK / 32));
}

// 尾声：m16n8 fragment 按标准映射散回全局 C（bf16 输出）
extern "C" __device__ void tl_mxf8_epilogue(
    const float* __restrict__ acc, cutlass::bfloat16_t* __restrict__ C,
    int ldc, int base_m, int base_n) {
  const int tx = threadIdx.x & 255;   // 同上：warp 特化下取消费者逻辑线程号
  const int lane = tx & 31, w = tx >> 5;
  const int warp_m = (w >> 2) * 64, warp_n = (w & 3) * 32;
  const int g = lane >> 2, q = lane & 3;
  #pragma unroll
  for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      const float* d = acc + (mi * 4 + ni) * 4;
      const int r0 = base_m + warp_m + mi * 16 + g;
      const int c0 = base_n + warp_n + ni * 8 + q * 2;
      C[(int64_t)r0 * ldc + c0] = cutlass::bfloat16_t(d[0]);
      C[(int64_t)r0 * ldc + c0 + 1] = cutlass::bfloat16_t(d[1]);
      C[(int64_t)(r0 + 8) * ldc + c0] = cutlass::bfloat16_t(d[2]);
      C[(int64_t)(r0 + 8) * ldc + c0 + 1] = cutlass::bfloat16_t(d[3]);
    }
}
"""


# ────────────────────────── TileLang 主体 ──────────────────────────
@tilelang.jit
def build_kernel(M: int, N: int, K: int, block_M=128, block_N=128, block_K=128,
                 num_stages=2, threads=256, direct=False, sfb_global=False,
                 grid_m_fast=False):
    """grid_m_fast：M 维做 blockIdx.x（最快轴）——瘦长形状（N 巨大）下让同一 B tile
    被连续的 M 块复用而常驻 L2，B 的显存流量从 (M/BM)×N×K/2 掉回 N×K/2。"""
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    KB = block_K // 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "uint8"),
        B: T.Tensor((N, K), "uint8"),
        SFA: T.Tensor((M, K // 32), "uint8"),
        SFB: T.Tensor((N, K // 32), "uint8"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) if grid_m_fast else T.ceildiv(N, block_N),
            T.ceildiv(N, block_N) if grid_m_fast else T.ceildiv(M, block_M),
            threads=threads,
        ) as (b0, b1):
            bx = b1 if grid_m_fast else b0
            by = b0 if grid_m_fast else b1
            A_sh = T.alloc_shared((block_M, block_K), "uint8")
            B_sh = T.alloc_shared((block_N, block_K), "uint8")
            # ⚠️ 缩放必须走 smem 拷贝：撤掉它会让 TileLang 把生产者分区缩到 128 线程而
            #   释放 barrier 计数仍是 256 ⇒ 相位竞态、散点错块（2026-08-27 实测，报上游）
            SFA_sh = T.alloc_shared((block_M, KB), "uint8")
            SFB_sh = T.alloc_shared((block_N, KB), "uint8")
            acc = T.alloc_local((64,), "float32")

            T.import_source(MXF8_DEVICE_SRC)

            for i in T.serial(64):
                acc[i] = T.float32(0)

            if direct:
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.call_extern(
                        "tl_mxf8_warp_gemm_g",
                        T.access_ptr(A, "r"), T.access_ptr(B, "r"),
                        T.access_ptr(SFA, "r"), T.access_ptr(SFB, "r"),
                        T.access_ptr(acc, "rw"),
                        by * block_M, bx * block_N, ko, K, block_K,
                        dtype="int32",
                    )
            elif sfb_global:
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_sh)
                    T.copy(B[bx * block_N, ko * block_K], B_sh)
                    T.copy(SFA[by * block_M, ko * KB], SFA_sh)
                    T.call_extern(
                        "tl_mxf8_warp_gemm",
                        T.access_ptr(A_sh, "r"), T.access_ptr(B_sh, "r"),
                        T.access_ptr(SFA_sh, "r"), T.access_ptr(SFB, "r"),
                        T.access_ptr(acc, "rw"), block_K, block_K,
                        KB, K // 32, 0, bx * block_N, 0, ko * KB,
                        dtype="int32",
                    )
            else:
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_sh)
                    T.copy(B[bx * block_N, ko * block_K], B_sh)
                    T.copy(SFA[by * block_M, ko * KB], SFA_sh)
                    T.copy(SFB[bx * block_N, ko * KB], SFB_sh)
                    T.call_extern(
                        "tl_mxf8_warp_gemm",
                        T.access_ptr(A_sh, "r"), T.access_ptr(B_sh, "r"),
                        T.access_ptr(SFA_sh, "r"), T.access_ptr(SFB_sh, "r"),
                        T.access_ptr(acc, "rw"), block_K, block_K,
                        KB, KB, 0, 0, 0, 0,
                        dtype="int32",
                    )

            T.call_extern(
                "tl_mxf8_epilogue",
                T.access_ptr(acc, "r"), T.access_ptr(C, "w"),
                N, by * block_M, bx * block_N,
                dtype="int32",
            )

    return main


# v2：块 128×256×128 · warp tile 64×64（mi=4, ni=8）· 逐 kb 交错加载省寄存器
# SFA 走 smem（保生产者 256 分区），SFB 直读全局 L2（省 smem 挤进 2 级流水）
MXF8_DEVICE_SRC_V2 = r"""
#include <cstdint>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

template <int KB>
__device__ __forceinline__ void tl_mxf8_mma_atom_v2(
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1,
    const uint32_t sa, const uint32_t sb, float* d) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
      ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
      "{%10},{%11,0},{%12},{%11,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sa), "n"(KB), "r"(sb));
}

extern "C" __device__ void tl_mxf8_warp_gemm_v2(
    const uint8_t* __restrict__ A_sh, const uint8_t* __restrict__ B_sh,
    const uint8_t* __restrict__ SFA_sh, const uint8_t* __restrict__ SFB_g,
    float* __restrict__ acc, int sfb_base_row, int sfb_lds, int sfb_col) {
  const int tx = threadIdx.x & 255;
  const int lane = tx & 31, w = tx >> 5;
  const int warp_m = (w >> 2) * 64;          // 2×4 warp 网格，warp tile 64×64
  const int warp_n = (w & 3) * 64;
  const int g = lane >> 2, q = lane & 3;
  const int l = lane;
  const int a_row_off = l & 15, a_byte_off = (l >> 4) << 4;
  const int b_row_off = l & 7,  b_byte_off = ((l >> 3) & 1) << 4;

  uint32_t sfa_r[4], sfb_r[8];
  #pragma unroll
  for (int mi = 0; mi < 4; mi++) {
    const int sfa_row = warp_m + mi * 16 + g + ((q & 1) ? 8 : 0);
    sfa_r[mi] = *reinterpret_cast<const uint32_t*>(SFA_sh + sfa_row * 4);
  }
  #pragma unroll
  for (int ni = 0; ni < 8; ni++) {
    const int col = warp_n + ni * 8 + g;
    sfb_r[ni] = *reinterpret_cast<const uint32_t*>(
        SFB_g + (int64_t)(sfb_base_row + col) * sfb_lds + sfb_col);
  }

  #pragma unroll
  for (int kb = 0; kb < 4; kb++) {
    uint32_t af[4][4], bf[8][2];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
      const int ar = warp_m + mi * 16 + a_row_off;
      const int ac = (kb * 32 + a_byte_off) >> 4;
      const int acs = (ac & ~7) | ((ac & 7) ^ (ar & 7));
      const uint32_t sa_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(A_sh + (int64_t)ar * 128 + (acs << 4)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3])
                   : "r"(sa_addr));
    }
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
      const int br = warp_n + ni * 8 + b_row_off;
      const int bc = (kb * 32 + b_byte_off) >> 4;
      const int bcs = (bc & ~7) | ((bc & 7) ^ (br & 7));
      const uint32_t sb_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(B_sh + (int64_t)br * 128 + (bcs << 4)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                   : "=r"(bf[ni][0]), "=r"(bf[ni][1]) : "r"(sb_addr));
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
      #pragma unroll
      for (int ni = 0; ni < 8; ni++) {
        float* d = acc + (mi * 8 + ni) * 4;
        if (kb == 0) tl_mxf8_mma_atom_v2<0>(af[mi][0],af[mi][1],af[mi][2],af[mi][3],bf[ni][0],bf[ni][1],sfa_r[mi],sfb_r[ni],d);
        if (kb == 1) tl_mxf8_mma_atom_v2<1>(af[mi][0],af[mi][1],af[mi][2],af[mi][3],bf[ni][0],bf[ni][1],sfa_r[mi],sfb_r[ni],d);
        if (kb == 2) tl_mxf8_mma_atom_v2<2>(af[mi][0],af[mi][1],af[mi][2],af[mi][3],bf[ni][0],bf[ni][1],sfa_r[mi],sfb_r[ni],d);
        if (kb == 3) tl_mxf8_mma_atom_v2<3>(af[mi][0],af[mi][1],af[mi][2],af[mi][3],bf[ni][0],bf[ni][1],sfa_r[mi],sfb_r[ni],d);
      }
  }
}

extern "C" __device__ void tl_mxf8_epilogue_v2(
    const float* __restrict__ acc, cutlass::bfloat16_t* __restrict__ C,
    int ldc, int base_m, int base_n) {
  const int tx = threadIdx.x & 255;
  const int lane = tx & 31, w = tx >> 5;
  const int warp_m = (w >> 2) * 64, warp_n = (w & 3) * 64;
  const int g = lane >> 2, q = lane & 3;
  #pragma unroll
  for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
      const float* d = acc + (mi * 8 + ni) * 4;
      const int r0 = base_m + warp_m + mi * 16 + g;
      const int c0 = base_n + warp_n + ni * 8 + q * 2;
      C[(int64_t)r0 * ldc + c0] = cutlass::bfloat16_t(d[0]);
      C[(int64_t)r0 * ldc + c0 + 1] = cutlass::bfloat16_t(d[1]);
      C[(int64_t)(r0 + 8) * ldc + c0] = cutlass::bfloat16_t(d[2]);
      C[(int64_t)(r0 + 8) * ldc + c0 + 1] = cutlass::bfloat16_t(d[3]);
    }
}
"""


@tilelang.jit
def build_kernel_v2(M: int, N: int, K: int, num_stages=2, threads=384):
    """块 128×256×128；SFA smem / SFB 全局；smem 2×(48KB+0.5KB)=99328 < 上限。"""
    block_M, block_N, block_K = 128, 256, 128
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    KB = block_K // 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "uint8"),
        B: T.Tensor((N, K), "uint8"),
        SFA: T.Tensor((M, K // 32), "uint8"),
        SFB: T.Tensor((N, K // 32), "uint8"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), "uint8")
            B_sh = T.alloc_shared((block_N, block_K), "uint8")
            SFA_sh = T.alloc_shared((block_M, KB), "uint8")
            acc = T.alloc_local((128,), "float32")

            T.import_source(MXF8_DEVICE_SRC_V2)
            # ⚠️ 别用 annotate_producer_reg_dealloc/consumer_reg_alloc 强开重分配：
            #   在 256/256 分区上实测挂死（2026-08-27）。走 threads=384 让静态上限升到 170。

            for i in T.serial(128):
                acc[i] = T.float32(0)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_sh)
                T.copy(B[bx * block_N, ko * block_K], B_sh)
                T.copy(SFA[by * block_M, ko * KB], SFA_sh)
                T.call_extern(
                    "tl_mxf8_warp_gemm_v2",
                    T.access_ptr(A_sh, "r"), T.access_ptr(B_sh, "r"),
                    T.access_ptr(SFA_sh, "r"), T.access_ptr(SFB, "r"),
                    T.access_ptr(acc, "rw"),
                    bx * block_N, K // 32, ko * KB,
                    dtype="int32",
                )

            T.call_extern(
                "tl_mxf8_epilogue_v2",
                T.access_ptr(acc, "r"), T.access_ptr(C, "w"),
                N, by * block_M, bx * block_N,
                dtype="int32",
            )

    return main


# ────────────────────────── MXFP8 量化器与参考 ──────────────────────────
def quantize_mxfp8(x: torch.Tensor):
    """[R, K] bf16/fp32 → (u8 e4m3 [R,K], ue8m0 [R,K/32])。OCP MXFP8：块 32、缩放 2^e。"""
    R, K = x.shape
    assert K % 32 == 0
    xb = x.float().view(R, K // 32, 32)
    amax = xb.abs().amax(dim=-1)                                   # [R, K/32]
    e = torch.clamp(torch.ceil(torch.log2(amax / 448.0)), -127, 127)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    scale = torch.pow(2.0, e)                                      # [R, K/32]
    q = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    u8 = q.view(torch.uint8).view(R, K)
    sf = (e + 127).to(torch.uint8)                                 # ue8m0
    return u8.contiguous(), sf.contiguous()


def swizzle_rows(u8: torch.Tensor) -> torch.Tensor:
    """kernel 原生布局：每行 128B 组内的 16B 块按 chunk^=(row&7) 置换（消 ldmatrix bank 冲突）。"""
    R, K = u8.shape
    assert K % 128 == 0
    c = u8.view(R, K // 16, 16)
    chunk = torch.arange(K // 16, device=u8.device)
    row = torch.arange(R, device=u8.device)
    perm = (chunk.unsqueeze(0) & ~7) | ((chunk.unsqueeze(0) & 7) ^ (row.unsqueeze(1) & 7))
    out = torch.gather(c, 1, perm.unsqueeze(-1).expand(R, K // 16, 16))
    return out.reshape(R, K).contiguous()


def dequant_ref(u8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    R, K = u8.shape
    x = u8.view(torch.float8_e4m3fn).float().view(R, K // 32, 32)
    scale = torch.pow(2.0, sf.float() - 127.0).unsqueeze(-1)
    return (x * scale).view(R, K)


# ────────────────────────── 主流程 ──────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--num-stages", type=int, default=2)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--direct", action="store_true", help="诊断：直读全局内存，绕过 smem")
    p.add_argument("--v2", action="store_true", help="块 128x256（B 复用×2）")
    p.add_argument("--sfb-global", action="store_true", help="SFB 直读全局，省 smem 上 3 级流水")
    p.add_argument("--grid-m-fast", action="store_true", help="M 维最快轴：瘦长形状 B tile 常驻 L2")
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    torch.manual_seed(7)
    dev = "cuda"

    A = torch.randn(M, K, device=dev, dtype=torch.float32)
    Bt = torch.randn(N, K, device=dev, dtype=torch.float32)       # B^T：[N, K]
    A_u8, SFA = quantize_mxfp8(A)
    B_u8, SFB = quantize_mxfp8(Bt)
    A_k, B_k = swizzle_rows(A_u8), swizzle_rows(B_u8)             # kernel 原生布局
    C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    if args.v2:
        kernel = build_kernel_v2(M, N, K, num_stages=args.num_stages)
    else:
        kernel = build_kernel(M, N, K, num_stages=args.num_stages, direct=args.direct, sfb_global=args.sfb_global, grid_m_fast=args.grid_m_fast)
    kernel(A_k, B_k, SFA, SFB, C)

    if args.verify:
        ref = dequant_ref(A_u8, SFA) @ dequant_ref(B_u8, SFB).T   # 与内核吃同一份量化数
        err = (C.float() - ref).abs().max().item()
        ref_inf = ref.abs().max().item()
        rel = err / max(ref_inf, 1e-9)
        ok = rel < 2e-2                                            # bf16 输出舍入 + 累加序
        print(f"对拍 max|Δ|/‖ref‖∞ = {rel:.3e}  {'✅' if ok else '❌'}")
        if not ok:
            raise SystemExit(1)

    lat = do_bench(lambda: kernel(A_k, B_k, SFA, SFB, C), warmup=25, rep=100)
    tflops = 2 * M * N * K / (lat * 1e-3) / 1e12
    print(f"MXFP8 GEMM  {M}x{N}x{K}  {lat:.4f} ms  {tflops:.2f} TFLOPS"
          f"  = 峰值1026的 {tflops/1026*100:.1f}%")


if __name__ == "__main__":
    main()
