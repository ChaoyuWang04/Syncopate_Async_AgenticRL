// A4 · MXFP8 lm_head GEMM 的 torch 扩展（生产 venv 用，零第三方依赖）
// kernel 主体 = scripts/infra/mxf8_gemm_limit_tma.cu 的 T1 冠军配置（627 TFLOPS，已对拍），
// 固定 BM=BN=128 · warp tile 64x64 · 2 级流水 · TMA+warp 特化（160 线程）。
// 布局契约（与 python 侧量化器一致）：A[M,K]/B[N,K] u8 已做行内 XOR swizzle；
// SFA[M,K/32]/SFB[N,K/32] u8 (ue8m0)；M/N/K 均为 128 倍数；C 为 bf16 [M,N]。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CKD(x) do { CUresult r = (x); TORCH_CHECK(r == CUDA_SUCCESS, "cuda driver err ", (int)r, " @", #x); } while (0)

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



static CUtensorMap make_map(const void* p, uint64_t rows, uint64_t cols,
                            uint32_t box_r, uint32_t box_c) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols};
  cuuint32_t box[2] = {box_c, box_r};
  cuuint32_t es[2] = {1, 1};
  CKD(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)p,
                             dims, strides, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

torch::Tensor mxf8_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor SFA, torch::Tensor SFB) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && SFA.is_cuda() && SFB.is_cuda(), "need cuda tensors");
  TORCH_CHECK(A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8, "A/B must be uint8");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && SFA.is_contiguous() && SFB.is_contiguous());
  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K && SFA.size(0) == M && SFB.size(0) == N);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "M/N/K must be x128, got ", M, "/", N, "/", K);
  auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

  constexpr int BK = 128, KBW = 4, STAGES = 2;
  constexpr int BM = 128, BN = 128;
  const int smem = STAGES * (BM * BK + BN * BK + BM * KBW + BN * KBW) + 2 * STAGES * 8 + 16;
  auto kfn = mxf8_tma<BM, BN, 64, 64, STAGES, false>;
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
    attr_set = true;
  }
  CUtensorMap mA = make_map(A.data_ptr(), M, K, BM, BK);
  CUtensorMap mB = make_map(B.data_ptr(), N, K, BN, BK);
  dim3 grid(M / BM, N / BN), blk((2 * 2 + 1) * 32);
  kfn<<<grid, blk, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, (const uint8_t*)SFA.data_ptr(), (const uint8_t*)SFB.data_ptr(),
      (__nv_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxf8_gemm", &mxf8_gemm, "MXFP8 block-scaled GEMM (sm_120)");
}
