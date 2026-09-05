// E30 §10b · 裸 CUDA 真极限第二型：TMA + warp 特化 + 大寄存器块（合体设计）
//
// §10a（cp.async 平铺 256 线程）的教训：寄存器余量有了（255/线程）但计算线程自己扛
// 搬运 ⇒ mma 发射口被抢，441 < tilelang 543。TileLang 赢在 TMA+warp 特化，输在
// 512 线程把寄存器钉死 128。本文件两样全要：
//   9 warp = 1 生产者（只发 TMA/等 barrier，几乎不用寄存器）+ 8 消费者（64×64 大块）
//   288 线程 ⇒ ptxas 静态上限 227 寄存器/线程 ⇒ 128 累加器 + fragment 无溢出
//   TMA（cuTensorMapEncodeTiled 四张描述符：A/B/SFA/SFB）+ mbarrier 满/空双向握手
//
// 编译：nvcc -arch=sm_120a -O3 -lcuda -o /workspace/tmp/mxf8_tma scripts/infra/mxf8_gemm_limit_tma.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
#define CKD(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(r, &s); printf("CU err %s @%d: %s\n", #x, __LINE__, s); exit(1);} } while (0)

// ─────────── mbarrier / TMA 裸 PTX 工具 ───────────
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr));
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
  asm volatile(
      "{.reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
      " @!p bra WAIT_%=; }" :: "r"(addr), "r"(parity));
}
__device__ __forceinline__ void tma_2d(const CUtensorMap* desc, uint32_t mbar,
                                       uint32_t dst, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(dst), "l"(reinterpret_cast<uint64_t>(desc)), "r"(x), "r"(y), "r"(mbar));
}

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

// ─────────── kernel：BMxBN 块 · 8 消费 warp（WMxWN）+ 1 生产 warp ───────────
template <int BM, int BN, int WM, int WN, int STAGES, bool SFBG>
__global__ void __launch_bounds__(((BM / WM) * (BN / WN) + 1) * 32)
mxf8_tma(const __grid_constant__ CUtensorMap dA, const __grid_constant__ CUtensorMap dB,
         const uint8_t* __restrict__ dSA_ptr, const uint8_t* __restrict__ dSB_ptr,
         __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  constexpr int BK = 128, KBW = 4;
  constexpr int NCONS = (BM / WM) * (BN / WN);            // 消费 warp 数
  constexpr int CTHREADS = NCONS * 32;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int A_SZ = BM * BK, B_SZ = BN * BK, SA_SZ = BM * KBW;
  constexpr int SB_SZ = SFBG ? 0 : BN * KBW;              // SFBG：B 缩放消费者直读全局（省 smem 上大块）
  constexpr int STG = A_SZ + B_SZ + SA_SZ + SB_SZ;
  constexpr uint32_t TX = A_SZ + B_SZ;   // 缩放走生产者普通拷贝（TMA box 需 16B 倍数，4B 缩放列不满足）

  extern __shared__ __align__(1024) uint8_t smem[];
  // 布局：STAGES 个数据槽 + 2*STAGES 个 mbarrier（full[s], empty[s]）
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + STAGES * STG);
  const uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_base = (uint32_t)__cvta_generic_to_shared(bars);
  auto full_bar = [&](int s) { return bar_base + s * 8; };
  auto empty_bar = [&](int s) { return bar_base + (STAGES + s) * 8; };

  const int tid = threadIdx.x;
  const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
  const int ktiles = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(full_bar(s), 1); mbar_init(empty_bar(s), NCONS); }
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();

  if (tid >= CTHREADS) {
    // ── 生产者 warp（32 线程协作拷缩放；lane0 发 TMA）──
    const int pl = tid - CTHREADS;
    const uint8_t* gSA = (const uint8_t*)dSA_ptr;
    const uint8_t* gSB = (const uint8_t*)dSB_ptr;
    for (int kt = 0; kt < ktiles; kt++) {
      const int s = kt % STAGES;
      const int use = kt / STAGES;
      if (use > 0) mbar_wait(empty_bar(s), (use - 1) & 1);
      uint8_t* sm = smem + s * STG + A_SZ + B_SZ;
      for (int i = pl; i < BM; i += 32)
        *(uint32_t*)(sm + i * KBW) = *(const uint32_t*)(gSA + (int64_t)(bm + i) * (K / 32) + kt * KBW);
      if (!SFBG)
        for (int i = pl; i < BN; i += 32)
          *(uint32_t*)(sm + SA_SZ + i * KBW) = *(const uint32_t*)(gSB + (int64_t)(bn + i) * (K / 32) + kt * KBW);
      __syncwarp();
      if (pl == 0) {                       // arrive 在缩放写入之后 ⇒ release 语义带上它们
        mbar_arrive_expect_tx(full_bar(s), TX);
        const uint32_t d = smem_base + s * STG;
        tma_2d(&dA, full_bar(s), d, kt * BK, bm);
        tma_2d(&dB, full_bar(s), d + A_SZ, kt * BK, bn);
      }
    }
    return;                                              // 生产者 warp 不做尾声
  }

  // ── 消费者：warp 坐标与 fragment 映射（与已验证版本同源）──
  const int w = tid >> 5, lane = tid & 31;
  const int warp_m = (w / (BN / WN)) * WM, warp_n = (w % (BN / WN)) * WN;
  const int g = lane >> 2, q = lane & 3;
  const int a_row_off = lane & 15, a_byte_off = (lane >> 4) << 4;
  const int b_row_off = lane & 7, b_byte_off = ((lane >> 3) & 1) << 4;

  float acc[MI * NI * 4];
#pragma unroll
  for (int i = 0; i < MI * NI * 4; i++) acc[i] = 0.f;

  for (int kt = 0; kt < ktiles; kt++) {
    const int s = kt % STAGES;
    mbar_wait(full_bar(s), (kt / STAGES) & 1);

    const uint8_t* As = smem + s * STG;
    const uint8_t* Bs = As + A_SZ;
    const uint8_t* SAs = Bs + B_SZ;
    const uint8_t* SBs = SAs + SA_SZ;

    uint32_t sfa_r[MI], sfb_r[NI];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
      sfa_r[mi] = *(const uint32_t*)(SAs + (warp_m + mi * 16 + g + ((q & 1) ? 8 : 0)) * KBW);
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
      sfb_r[ni] = SFBG
          ? *(const uint32_t*)(dSB_ptr + (int64_t)(bn + warp_n + ni * 8 + g) * (K / 32) + kt * KBW)
          : *(const uint32_t*)(SBs + (warp_n + ni * 8 + g) * KBW);

#pragma unroll
    for (int kb = 0; kb < 4; kb++) {
      uint32_t af[MI][4], bf[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int ar = warp_m + mi * 16 + a_row_off;
        const int ac = (kb * 32 + a_byte_off) >> 4;
        const int acs = (ac & ~7) | ((ac & 7) ^ (ar & 7));
        const uint32_t sa = smem_base + s * STG + ar * BK + (acs << 4);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3]) : "r"(sa));
      }
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const int br = warp_n + ni * 8 + b_row_off;
        const int bc = (kb * 32 + b_byte_off) >> 4;
        const int bcs = (bc & ~7) | ((bc & 7) ^ (br & 7));
        const uint32_t sb = smem_base + s * STG + A_SZ + br * BK + (bcs << 4);
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
    if (lane == 0) mbar_arrive(empty_bar(s));    // 每 warp 选举到达（计数=NCONS）
  }

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

// ─────────── 宿主 ───────────
static CUtensorMap make_map(const void* p, uint64_t rows, uint64_t cols,
                            uint32_t box_r, uint32_t box_c) {
  // 2D u8：dim0=cols（连续）、dim1=rows；globalStride[0]=cols 字节（须 16 的倍数）
  CUtensorMap m;
  uint64_t dims[2] = {cols, rows};
  uint64_t strides[1] = {cols};
  uint32_t box[2] = {box_c, box_r};
  uint32_t elemStr[2] = {1, 1};
  CKD(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)p,
                             dims, strides, box, elemStr,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

template <int BM, int BN, int WM, int WN, int STAGES, bool SFBG = false>
float run_tma(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB,
              __nv_bfloat16* C, int M, int N, int K, int reps) {
  constexpr int BK = 128, KBW = 4;
  const int smem = STAGES * (BM * BK + BN * BK + BM * KBW + (SFBG ? 0 : BN * KBW)) + 2 * STAGES * 8 + 16;
  auto kfn = mxf8_tma<BM, BN, WM, WN, STAGES, SFBG>;
  if (cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
    printf("  smem %dB 超上限，跳过\n", smem); return -1.f;
  }
  CUtensorMap mA = make_map(A, M, K, BM, BK), mB = make_map(B, N, K, BN, BK);
  dim3 grid(M / BM, N / BN), blk(((BM / WM) * (BN / WN) + 1) * 32);
  kfn<<<grid, blk, smem>>>(mA, mB, SA, SB, C, M, N, K);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("  启动失败：%s\n", cudaGetErrorString(cudaGetLastError())); return -1.f;
  }
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < reps; i++) kfn<<<grid, blk, smem>>>(mA, mB, SA, SB, C, M, N, K);
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

int main() {
  cuInit(0);
  // ① 512³ 对拍
  {
    const int M = 512, N = 512, K = 512;
    uint8_t *hA = new uint8_t[M * K], *hB = new uint8_t[N * K];
    float *fA = new float[M * K], *fB = new float[N * K];
    fill_fp8(hA, fA, M * K, 7); fill_fp8(hB, fB, N * K, 13);
    uint8_t *hSA = new uint8_t[M * K / 32], *hSB = new uint8_t[N * K / 32];
    for (int i = 0; i < M * K / 32; i++) hSA[i] = 125 + rand() % 5;
    for (int i = 0; i < N * K / 32; i++) hSB[i] = 125 + rand() % 5;
    float* ref = new float[M * N];
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        float s = 0;
        for (int k = 0; k < K; k++)
          s += fA[m * K + k] * powf(2.f, (int)hSA[m * (K / 32) + k / 32] - 127) *
               fB[n * K + k] * powf(2.f, (int)hSB[n * (K / 32) + k / 32] - 127);
        ref[m * N + n] = s;
      }
    uint8_t *hAs = new uint8_t[M * K], *hBs = new uint8_t[N * K];
    swizzle_host(hAs, hA, M, K); swizzle_host(hBs, hB, N, K);
    uint8_t *dA, *dB, *dSA, *dSB; __nv_bfloat16* dC;
    CK(cudaMalloc(&dA, M * K)); CK(cudaMalloc(&dB, N * K));
    CK(cudaMalloc(&dSA, M * K / 32)); CK(cudaMalloc(&dSB, N * K / 32));
    CK(cudaMalloc(&dC, M * N * 2));
    CK(cudaMemcpy(dA, hAs, M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hBs, N * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSA, hSA, M * K / 32, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSB, hSB, N * K / 32, cudaMemcpyHostToDevice));
    __nv_bfloat16* hC = new __nv_bfloat16[M * N];
    printf("== ① 512³ 对拍（TMA+warp 特化版）==\n");
    auto verify = [&](const char* name, float ms) {
      if (ms < 0) return;
      CK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));
      float mx = 0, mr = 0;
      for (int i = 0; i < M * N; i++) {
        mx = fmaxf(mx, fabsf(__bfloat162float(hC[i]) - ref[i]));
        mr = fmaxf(mr, fabsf(ref[i]));
      }
      printf("  %-38s max|Δ|/‖ref‖∞ = %.3e %s\n", name, mx / mr, mx / mr < 2e-2 ? "✅" : "❌");
    };
    verify("T0 128x128 w64x32 s2", run_tma<128, 128, 64, 32, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T1 128x128 w64x64 s2", run_tma<128, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T2 128x256 w64x64 s2 sfbG", run_tma<128, 256, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T3 128x128 w64x64 s3 sfbG", run_tma<128, 128, 64, 64, 3, true>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T4 256x128 w64x64 s2 sfbG", run_tma<256, 128, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T5 128x128 w64x64 s2 sfbG", run_tma<128, 128, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T6 128x128 w64x64 s3", run_tma<128, 128, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T7 64x128 w64x64 s2", run_tma<64, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    verify("T8 64x128 w64x64 s4", run_tma<64, 128, 64, 64, 4>(dA, dB, dSA, dSB, dC, M, N, K, 1));
    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] fA; delete[] fB; delete[] hSA; delete[] hSB;
    delete[] hAs; delete[] hBs; delete[] ref; delete[] hC;
  }
  // ② 8192³ 计时
  {
    const int M = 8192, N = 8192, K = 8192;
    uint8_t *dA, *dB, *dSA, *dSB; __nv_bfloat16* dC;
    CK(cudaMalloc(&dA, (int64_t)M * K)); CK(cudaMalloc(&dB, (int64_t)N * K));
    CK(cudaMalloc(&dSA, (int64_t)M * K / 32)); CK(cudaMalloc(&dSB, (int64_t)N * K / 32));
    CK(cudaMalloc(&dC, (int64_t)M * N * 2));
    CK(cudaMemset(dA, 0x30, (int64_t)M * K)); CK(cudaMemset(dB, 0x30, (int64_t)N * K));
    CK(cudaMemset(dSA, 127, (int64_t)M * K / 32)); CK(cudaMemset(dSB, 127, (int64_t)N * K / 32));
    const double fl = 2.0 * M * N * K;
    printf("== ② 8192³ 计时（峰值尺 1026 · tilelang 543 · cuBLAS 523）==\n");
    auto bench = [&](const char* name, float ms) {
      if (ms < 0) return;
      printf("  %-38s %7.3f ms  %6.1f TFLOPS  = 峰值 %.1f%%\n",
             name, ms, fl / (ms * 1e-3) / 1e12, fl / (ms * 1e-3) / 1e12 / 1026 * 100);
    };
    bench("T0 128x128 w64x32 s2", run_tma<128, 128, 64, 32, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T1 128x128 w64x64 s2", run_tma<128, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T2 128x256 w64x64 s2 sfbG", run_tma<128, 256, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T3 128x128 w64x64 s3 sfbG", run_tma<128, 128, 64, 64, 3, true>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T4 256x128 w64x64 s2 sfbG", run_tma<256, 128, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T5 128x128 w64x64 s2 sfbG", run_tma<128, 128, 64, 64, 2, true>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T6 128x128 w64x64 s3", run_tma<128, 128, 64, 64, 3>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T7 64x128 w64x64 s2", run_tma<64, 128, 64, 64, 2>(dA, dB, dSA, dSB, dC, M, N, K, 20));
    bench("T8 64x128 w64x64 s4", run_tma<64, 128, 64, 64, 4>(dA, dB, dSA, dSB, dC, M, N, K, 20));
  }
  return 0;
}
