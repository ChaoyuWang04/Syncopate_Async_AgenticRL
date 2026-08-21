// A3-② · sm_120a 块缩放 MMA 峰值吞吐（寄存器驻留 · 无访存 · 4 独立累加器防延迟链）
// 编译：nvcc -arch=sm_120a -O3 scripts/bench_fp4_ptx_peak.cu -o /tmp/bench_fp4_ptx_peak
// 目的：给「距硬件上限%」那把尺子立上限本身——bf16(HMMA)/FP8/MXFP8/MXFP4/NVFP4 五档。
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define ACC_DECL(i) float d##i##0 = 0.f + i, d##i##1 = 1.f, d##i##2 = 2.f, d##i##3 = 3.f
#define ACC_USE(i) (d##i##0 + d##i##1 + d##i##2 + d##i##3)

// 每个 bench kernel：ITER 次 × 4 条独立 mma；FLOP/mma = 2*M*N*K
template <int VARIANT>
__global__ void bench(float* out, int iters) {
    uint32_t a0 = threadIdx.x, a1 = threadIdx.x ^ 7, a2 = blockIdx.x, a3 = 0x3c003c00u;
    uint32_t b0 = threadIdx.x * 3 + 1, b1 = 0x3c003c00u;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;   // ue8m0 的 1.0 ×4
    ACC_DECL(0); ACC_DECL(1); ACC_DECL(2); ACC_DECL(3);
#define MMA_BF16(D0, D1, D2, D3) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#define MMA_FP8(D0, D1, D2, D3) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#define MMA_MXF8(D0, D1, D2, D3) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X" \
        ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}," \
        "{%10},{0,0},{%11},{0,0};" \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#define MMA_MXF4(D0, D1, D2, D3) \
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X" \
        ".f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}," \
        "{%10},{0,0},{%11},{0,0};" \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#define MMA_NVF4(D0, D1, D2, D3) \
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X" \
        ".f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}," \
        "{%10},{0,0},{%11},{0,0};" \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#define ROUND(M) M(d00, d01, d02, d03) M(d10, d11, d12, d13) M(d20, d21, d22, d23) M(d30, d31, d32, d33)
    for (int i = 0; i < iters; ++i) {
        if (VARIANT == 0) { ROUND(MMA_BF16) }
        if (VARIANT == 1) { ROUND(MMA_FP8) }
        if (VARIANT == 2) { ROUND(MMA_MXF8) }
        if (VARIANT == 3) { ROUND(MMA_MXF4) }
        if (VARIANT == 4) { ROUND(MMA_NVF4) }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = ACC_USE(0) + ACC_USE(1) + ACC_USE(2) + ACC_USE(3);
}

template <int V>
double run(const char* name, long long flop_per_mma, int iters, int blocks, int tpb, float* out) {
    bench<V><<<blocks, tpb>>>(out, 8);  // 预热
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    bench<V><<<blocks, tpb>>>(out, iters);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    long long warps = (long long)blocks * (tpb / 32);
    double tflops = (double)warps * iters * 4 /*独立mma*/ * flop_per_mma / (ms * 1e-3) / 1e12;
    cudaError_t e = cudaGetLastError();
    printf("%-22s %8.1f TFLOPS  (%.2f ms)%s\n", name, tflops, ms,
           e == cudaSuccess ? "" : cudaGetErrorString(e));
    return tflops;
}

int main() {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int blocks = sms * 8, tpb = 128, iters = 30000;
    float* out;
    cudaMalloc(&out, sizeof(float) * blocks * tpb);
    printf("SMs=%d blocks=%d tpb=%d iters=%d（寄存器驻留峰值，非真实 GEMM）\n", sms, blocks, tpb, iters);
    run<0>("bf16  m16n8k16", 2LL * 16 * 8 * 16, iters, blocks, tpb, out);
    run<1>("fp8   m16n8k32", 2LL * 16 * 8 * 32, iters, blocks, tpb, out);
    run<2>("mxf8  m16n8k32 bs", 2LL * 16 * 8 * 32, iters, blocks, tpb, out);
    run<3>("mxf4  m16n8k64 bs", 2LL * 16 * 8 * 64, iters, blocks, tpb, out);
    run<4>("nvf4  m16n8k64 bs", 2LL * 16 * 8 * 64, iters, blocks, tpb, out);
    return 0;
}
