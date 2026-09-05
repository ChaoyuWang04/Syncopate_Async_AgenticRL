// E30 §10 · 裸 CUDA：消费级 sm_120 上 MXFP8 块缩放 GEMM 的真极限探针
//
// 动机：TileLang 的 warp 特化把 kernel 钉在 512 线程 ⇒ ptxas 静态上限 128 寄存器/线程，
//   64×64 及以上 warp tile（更高计算/搬料比）装不下（E30 §5 判死）。本文件绕开一切框架：
//   256 线程平铺（≤255 寄存器/线程）+ cp.async 自管多级流水 + 模板化 tile 形状直接扫表。
//   目标 = 找到"没有 tcgen05 张量内存、累加器只能住寄存器"的消费卡的真天花板，
//   给 tilelang 的上游 PR（DRAFT-tilelang-sm120-mxfp8-support）提供极限参考数据。
//
// 复用（皆已逐位验证）：mma 内层 + 缩放 lane 映射（probe_mxf8_scale_mapping.cu 反演）
//   + 行内 128B 组 XOR swizzle 数据布局（tl_mxfp8_gemm.py 同款，宿主预置换）。
//
// 编译：nvcc -arch=sm_120a -O3 -o /workspace/tmp/mxf8_limit scripts/infra/mxf8_gemm_limit.cu
// 用法：/workspace/tmp/mxf8_limit            # 全配置：先 512³ 对拍 CPU，再 8192³ 计时
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

// ────────────────── mma 内层（与 tilelang 版逐位同源） ──────────────────
template <int KB>
__device__ __forceinline__ void mma_atom(const uint32_t a0, const uint32_t a1,
                                         const uint32_t a2, const uint32_t a3,
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

__device__ __forceinline__ void cp16(uint32_t dst_smem, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_smem), "l"(src));
}

// ────────────────── 主 kernel：模板化 tile 形状 ──────────────────
// 布局：A[M,K] u8（行内已 XOR 预置换）· B[N,K] 同 · SFA[M,K/32] u8 · SFB[N,K/32]
// BK 固定 128（4 个 k-block；缩放 uint32 打包 bid 0..3）
template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
mxf8_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
          const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
          __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  constexpr int BK = 128, KB = 4;
  constexpr int NWARP = (BM / WM) * (BN / WN), THREADS = NWARP * 32;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int A_SZ = BM * BK, B_SZ = BN * BK, SA_SZ = BM * KB, SB_SZ = BN * KB;
  constexpr int STG = A_SZ + B_SZ + SA_SZ + SB_SZ;

  extern __shared__ uint8_t smem[];
  const int tid = threadIdx.x;
  const int bm = blockIdx.x * BM;                 // M 维最快轴（瘦长形状 L2 复用）
  const int bn = blockIdx.y * BN;
  const int sfk = K / 32;

  const uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);

  // ── 协作搬运一个 stage（cp.async；A/B 16B 粒度，缩放 4B 粒度normal load）──
  auto load_stage = [&](int kt, int slot) {
    const uint32_t s = smem_base + slot * STG;
    const int64_t ka = (int64_t)kt * BK;
    // A: BM 行 × 8 chunk(16B)
    for (int i = tid; i < BM * (BK / 16); i += THREADS) {
      const int r = i / (BK / 16), c = i % (BK / 16);
      cp16(s + r * BK + c * 16, A + (int64_t)(bm + r) * K + ka + c * 16);
    }
    for (int i = tid; i < BN * (BK / 16); i += THREADS) {
      const int r = i / (BK / 16), c = i % (BK / 16);
      cp16(s + A_SZ + r * BK + c * 16, B + (int64_t)(bn + r) * K + ka + c * 16);
    }
    asm volatile("cp.async.commit_group;");
    // 缩放（小，普通拷贝即可；置于 commit 之后由 __syncthreads 保序）
    for (int i = tid; i < BM; i += THREADS)
      *(uint32_t*)(smem + slot * STG + A_SZ + B_SZ + i * KB) =
          *(const uint32_t*)(SFA + (int64_t)(bm + i) * sfk + kt * KB);
    for (int i = tid; i < BN; i += THREADS)
      *(uint32_t*)(smem + slot * STG + A_SZ + B_SZ + SA_SZ + i * KB) =
          *(const uint32_t*)(SFB + (int64_t)(bn + i) * sfk + kt * KB);
  };

  // ── warp 坐标与 fragment 索引（与 tilelang 版同款映射）──
  const int w = tid >> 5, lane = tid & 31;
  const int warp_m = (w / (BN / WN)) * WM, warp_n = (w % (BN / WN)) * WN;
  const int g = lane >> 2, q = lane & 3;
  const int a_row_off = lane & 15, a_byte_off = (lane >> 4) << 4;
  const int b_row_off = lane & 7, b_byte_off = ((lane >> 3) & 1) << 4;

  float acc[MI * NI * 4];
#pragma unroll
  for (int i = 0; i < MI * NI * 4; i++) acc[i] = 0.f;

  const int ktiles = K / BK;
  // 序幕：填 STAGES-1 级
  for (int i = 0; i < STAGES - 1 && i < ktiles; i++) load_stage(i, i);

  for (int kt = 0; kt < ktiles; kt++) {
    const int slot = kt % STAGES;
    if (kt + STAGES - 1 < ktiles) {
      __syncthreads();                                    // 防覆写上一轮同槽
      load_stage(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
      asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    const uint8_t* As = smem + slot * STG;
    const uint8_t* Bs = As + A_SZ;
    const uint8_t* SAs = Bs + B_SZ;
    const uint8_t* SBs = SAs + SA_SZ;

    uint32_t sfa_r[MI], sfb_r[NI];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
      sfa_r[mi] = *(const uint32_t*)(SAs + (warp_m + mi * 16 + g + ((q & 1) ? 8 : 0)) * KB);
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
      sfb_r[ni] = *(const uint32_t*)(SBs + (warp_n + ni * 8 + g) * KB);

#pragma unroll
    for (int kb = 0; kb < 4; kb++) {
      uint32_t af[MI][4], bf[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int ar = warp_m + mi * 16 + a_row_off;
        const int ac = (kb * 32 + a_byte_off) >> 4;
        const int acs = (ac & ~7) | ((ac & 7) ^ (ar & 7));
        const uint32_t sa = smem_base + slot * STG + ar * BK + (acs << 4);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3])
                     : "r"(sa));
      }
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const int br = warp_n + ni * 8 + b_row_off;
        const int bc = (kb * 32 + b_byte_off) >> 4;
        const int bcs = (bc & ~7) | ((bc & 7) ^ (br & 7));
        const uint32_t sb = smem_base + slot * STG + A_SZ + br * BK + (bcs << 4);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(bf[ni][0]), "=r"(bf[ni][1]) : "r"(sb));
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          float* d = acc + (mi * NI + ni) * 4;
          if (kb == 0) mma_atom<0>(af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1], sfa_r[mi], sfb_r[ni], d);
          if (kb == 1) mma_atom<1>(af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1], sfa_r[mi], sfb_r[ni], d);
          if (kb == 2) mma_atom<2>(af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1], sfa_r[mi], sfb_r[ni], d);
          if (kb == 3) mma_atom<3>(af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1], sfa_r[mi], sfb_r[ni], d);
        }
    }
  }

  // 尾声
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const float* d = acc + (mi * NI + ni) * 4;
      const int r0 = bm + warp_m + mi * 16 + g;
      const int c0 = bn + warp_n + ni * 8 + q * 2;
      C[(int64_t)r0 * N + c0] = __float2bfloat16(d[0]);
      C[(int64_t)r0 * N + c0 + 1] = __float2bfloat16(d[1]);
      C[(int64_t)(r0 + 8) * N + c0] = __float2bfloat16(d[2]);
      C[(int64_t)(r0 + 8) * N + c0 + 1] = __float2bfloat16(d[3]);
    }
}

// ────────────────── 宿主侧 ──────────────────
struct Cfg { const char* name; int BM, BN, WM, WN, STAGES; };

template <int BM, int BN, int WM, int WN, int STAGES>
float run_cfg(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB,
              __nv_bfloat16* C, int M, int N, int K, int reps) {
  constexpr int BK = 128, KB = 4;
  const int smem = STAGES * (BM * BK + BN * BK + BM * KB + BN * KB);
  auto kfn = mxf8_gemm<BM, BN, WM, WN, STAGES>;
  if (cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
    printf("  smem %dB 超上限，跳过\n", smem); return -1.f;
  }
  dim3 grid(M / BM, N / BN);
  dim3 blk((BM / WM) * (BN / WN) * 32);
  kfn<<<grid, blk, smem>>>(A, B, SA, SB, C, M, N, K);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("  启动失败：%s\n", cudaGetErrorString(cudaGetLastError())); return -1.f;
  }
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < reps; i++) kfn<<<grid, blk, smem>>>(A, B, SA, SB, C, M, N, K);
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms; cudaEventElapsedTime(&ms, t0, t1);
  return ms / reps;
}

static void fill_fp8(uint8_t* p, float* f, int64_t n, unsigned seed) {
  srand(seed);
  for (int64_t i = 0; i < n; i++) {
    float v = (rand() % 2000 - 1000) / 250.f;
    __nv_fp8_e4m3 q(v);
    p[i] = *(uint8_t*)&q; f[i] = float(q);
  }
}

static void swizzle_host(uint8_t* dst, const uint8_t* src, int R, int K) {
  for (int r = 0; r < R; r++)
    for (int c = 0; c < K / 16; c++) {
      const int cs = (c & ~7) | ((c & 7) ^ (r & 7));
      memcpy(dst + (int64_t)r * K + cs * 16, src + (int64_t)r * K + c * 16, 16);
    }
}

int main(int argc, char** argv) {
  const Cfg cfgs[] = {
      {"C0 128x128 w64x32 s2 (=tilelang 上限对照)", 128, 128, 64, 32, 2},
      {"C1 128x128 w64x64 s2 (4 warp 大块)",        128, 128, 64, 64, 2},
      {"C2 128x256 w64x64 s2 (被 tilelang 锁死的)", 128, 256, 64, 64, 2},
      {"C3 256x128 w64x64 s2",                      256, 128, 64, 64, 2},
      {"C4 128x128 w64x64 s3 (三级流水)",           128, 128, 64, 64, 3},
      {"C5 256x256 w64x64 s2 (16warp 512thr 对照)", 256, 256, 64, 64, 2},
      {"C6 128x256 w64x64 s3",                      128, 256, 64, 64, 3},
  };
  // ── ① 512³ 对拍 CPU 参考（全配置）──
  {
    const int M = 512, N = 512, K = 512;
    uint8_t *hA = new uint8_t[(int64_t)M * K], *hB = new uint8_t[(int64_t)N * K];
    float *fA = new float[(int64_t)M * K], *fB = new float[(int64_t)N * K];
    fill_fp8(hA, fA, (int64_t)M * K, 7); fill_fp8(hB, fB, (int64_t)N * K, 13);
    uint8_t *hSA = new uint8_t[(int64_t)M * K / 32], *hSB = new uint8_t[(int64_t)N * K / 32];
    for (int64_t i = 0; i < (int64_t)M * K / 32; i++) hSA[i] = 125 + rand() % 5;
    for (int64_t i = 0; i < (int64_t)N * K / 32; i++) hSB[i] = 125 + rand() % 5;
    // CPU 参考（fp32，含块缩放）
    float* ref = new float[(int64_t)M * N];
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        float s = 0;
        for (int k = 0; k < K; k++)
          s += fA[(int64_t)m * K + k] * powf(2.f, (int)hSA[m * (K / 32) + k / 32] - 127) *
               fB[(int64_t)n * K + k] * powf(2.f, (int)hSB[n * (K / 32) + k / 32] - 127);
        ref[(int64_t)m * N + n] = s;
      }
    uint8_t *hAs = new uint8_t[(int64_t)M * K], *hBs = new uint8_t[(int64_t)N * K];
    swizzle_host(hAs, hA, M, K); swizzle_host(hBs, hB, N, K);
    uint8_t *dA, *dB, *dSA, *dSB; __nv_bfloat16* dC;
    CK(cudaMalloc(&dA, (int64_t)M * K)); CK(cudaMalloc(&dB, (int64_t)N * K));
    CK(cudaMalloc(&dSA, (int64_t)M * K / 32)); CK(cudaMalloc(&dSB, (int64_t)N * K / 32));
    CK(cudaMalloc(&dC, (int64_t)M * N * 2));
    CK(cudaMemcpy(dA, hAs, (int64_t)M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hBs, (int64_t)N * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSA, hSA, (int64_t)M * K / 32, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSB, hSB, (int64_t)N * K / 32, cudaMemcpyHostToDevice));
    __nv_bfloat16* hC = new __nv_bfloat16[(int64_t)M * N];
    printf("== ① 512³ 对拍（CPU fp32 参考）==\n");
    auto verify = [&](const char* name, float ms) {
      if (ms < 0) return;
      CK(cudaMemcpy(hC, dC, (int64_t)M * N * 2, cudaMemcpyDeviceToHost));
      float mx = 0, mr = 0;
      for (int64_t i = 0; i < (int64_t)M * N; i++) {
        mx = fmaxf(mx, fabsf(__bfloat162float(hC[i]) - ref[i]));
        mr = fmaxf(mr, fabsf(ref[i]));
      }
      printf("  %-42s max|Δ|/‖ref‖∞ = %.3e %s\n", name, mx / mr, mx / mr < 2e-2 ? "✅" : "❌");
    };
    verify(cfgs[0].name, run_cfg<128, 128, 64, 32, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[1].name, run_cfg<128, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[2].name, run_cfg<128, 256, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[3].name, run_cfg<256, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[4].name, run_cfg<128, 128, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[5].name, run_cfg<256, 256, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify(cfgs[6].name, run_cfg<128, 256, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] fA; delete[] fB; delete[] hSA; delete[] hSB;
    delete[] hAs; delete[] hBs; delete[] ref; delete[] hC;
  }
  // ── ② 8192³ 计时（随机字节即可，数值不读）──
  {
    const int M = 8192, N = 8192, K = 8192;
    uint8_t *dA, *dB, *dSA, *dSB; __nv_bfloat16* dC;
    CK(cudaMalloc(&dA, (int64_t)M * K)); CK(cudaMalloc(&dB, (int64_t)N * K));
    CK(cudaMalloc(&dSA, (int64_t)M * K / 32)); CK(cudaMalloc(&dSB, (int64_t)N * K / 32));
    CK(cudaMalloc(&dC, (int64_t)M * N * 2));
    CK(cudaMemset(dA, 0x30, (int64_t)M * K)); CK(cudaMemset(dB, 0x30, (int64_t)N * K));
    CK(cudaMemset(dSA, 127, (int64_t)M * K / 32)); CK(cudaMemset(dSB, 127, (int64_t)N * K / 32));
    const double fl = 2.0 * M * N * K;
    printf("== ② 8192³ 计时（峰值尺 1026 TFLOPS）==\n");
    auto bench = [&](const char* name, float ms) {
      if (ms < 0) return;
      printf("  %-42s %7.3f ms  %6.1f TFLOPS  = 峰值 %.1f%%\n",
             name, ms, fl / (ms * 1e-3) / 1e12, fl / (ms * 1e-3) / 1e12 / 1026 * 100);
    };
    bench(cfgs[0].name, run_cfg<128, 128, 64, 32, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[1].name, run_cfg<128, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[2].name, run_cfg<128, 256, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[3].name, run_cfg<256, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[4].name, run_cfg<128, 128, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[5].name, run_cfg<256, 256, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench(cfgs[6].name, run_cfg<128, 256, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 20));
  }
  return 0;
}
