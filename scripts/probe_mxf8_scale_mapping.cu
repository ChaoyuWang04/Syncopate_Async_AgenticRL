// A3/E30 · mxf8f6f4 scale_vec::1X 的缩放因子 lane 映射反演探针
// 问题：m16n8k32 块缩放 mma 里，A 的 16 行 / B 的 8 列各自的 ue8m0 缩放字节
//       从哪个 lane 的哪个字节取？（tid/bid 选择器语义 + lane 内布局）
// 方法：A、B 全 1（e4m3 精确），每 lane 的 SFA 寄存器装 4 个**独一无二**的指数
//       （lane L 字节 b 装 127 + (L*4+b) % 8 ⇒ 缩放 = 2^((L*4+b)%8)），SFB 全 127（×1）。
//       D[r][n] = 32 · 2^(sa_r-127) ⇒ log2(D/32) 直接读出行 r 用了哪个 (lane,byte)。
//       对 (tid_a, bid_a) 全组合各跑一次；B 侧对称再来一遍。
// 编译：nvcc -arch=sm_120a -O3 scripts/probe_mxf8_scale_mapping.cu -o /tmp/probe_map
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// tid/bid 是立即数 ⇒ 模板化，全组合实例化
template <int TID_A, int BID_A, int TID_B, int BID_B>
__global__ void mma_probe(const uint8_t* A, const uint8_t* B, float* D, int probe_a) {
    int t = threadIdx.x, g = t >> 2, q = t & 3;
    auto pack = [](const uint8_t* p, int stride, int r, int c) {
        return (uint32_t)p[r * stride + c] | ((uint32_t)p[r * stride + c + 1] << 8) |
               ((uint32_t)p[r * stride + c + 2] << 16) | ((uint32_t)p[r * stride + c + 3] << 24);
    };
    uint32_t a0 = pack(A, 32, g, q * 4), a1 = pack(A, 32, g + 8, q * 4);
    uint32_t a2 = pack(A, 32, g, q * 4 + 16), a3 = pack(A, 32, g + 8, q * 4 + 16);
    uint32_t b0 = (uint32_t)B[(q * 4 + 0) * 8 + g] | ((uint32_t)B[(q * 4 + 1) * 8 + g] << 8) |
                  ((uint32_t)B[(q * 4 + 2) * 8 + g] << 16) | ((uint32_t)B[(q * 4 + 3) * 8 + g] << 24);
    uint32_t b1 = (uint32_t)B[(q * 4 + 16) * 8 + g] | ((uint32_t)B[(q * 4 + 17) * 8 + g] << 8) |
                  ((uint32_t)B[(q * 4 + 18) * 8 + g] << 16) | ((uint32_t)B[(q * 4 + 19) * 8 + g] << 24);
    // 探测臂：SFA 每 (lane,byte) 全域唯一指数 idx=t*4+b ∈ 0..127 ⇒ 缩放 2^(idx-127)，
    //         D=32·2^(idx-127) 全部落在 fp32 正规数域，log2 反读无损；对照臂全 127（×1）
    uint32_t sa, sb;
    if (probe_a) {
        sa = 0; for (int b = 0; b < 4; b++) sa |= (uint32_t)(t * 4 + b) << (8 * b);
        sb = 0x7F7F7F7Fu;
    } else {
        sa = 0x7F7F7F7Fu;
        sb = 0; for (int b = 0; b < 4; b++) sb |= (uint32_t)(t * 4 + b) << (8 * b);
    }
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sa), "n"(BID_A), "n"(TID_A), "r"(sb), "n"(BID_B), "n"(TID_B));
    D[g * 8 + q * 2] = d0;
    D[g * 8 + q * 2 + 1] = d1;
    D[(g + 8) * 8 + q * 2] = d2;
    D[(g + 8) * 8 + q * 2 + 1] = d3;
}

template <int TID_A, int BID_A>
void run_case(const uint8_t* dA, const uint8_t* dB, float* dD, int probe_a) {
    mma_probe<TID_A, BID_A, TID_A, BID_A><<<1, 32>>>(dA, dB, dD, probe_a);
}

int main() {
    uint8_t hA[16 * 32], hB[32 * 8];
    __nv_fp8_e4m3 one(1.0f);
    for (auto& x : hA) x = *(uint8_t*)&one;
    for (auto& x : hB) x = *(uint8_t*)&one;
    uint8_t *dA, *dB; float* dD;
    cudaMalloc(&dA, sizeof hA); cudaMalloc(&dB, sizeof hB); cudaMalloc(&dD, 16 * 8 * 4);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);
    float hD[16 * 8];

    // 探 A 侧：D[r][0] = 32·2^e ⇒ e = (lane*4+byte)%8 ⇒ 反查 (lane*4+byte)≡e (mod 8)
    for (int probe_a = 1; probe_a >= 0; probe_a--) {
        printf("==== 探 %s 侧（另一侧缩放恒 ×1）====\n", probe_a ? "A(SFA, 16 行)" : "B(SFB, 8 列)");
        for (int tid = 0; tid < 2; tid++) {
            // bid 只试 0（1X 下按谱面 bid 选 4 字节窗口的起点；不同 bid 只是平移，先证 0）
            if (tid == 0) run_case<0, 0>(dA, dB, dD, probe_a);
            else          run_case<1, 0>(dA, dB, dD, probe_a);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf("  tid=%d ❌ %s\n", tid, cudaGetErrorString(e)); continue; }
            cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost);
            printf("  tid=%d:  ", tid);
            // idx = 127 + log2(D/32) ⇒ lane = idx/4, byte = idx%4
            if (probe_a) {
                for (int r = 0; r < 16; r++) {
                    int idx = 127 + (int)lroundf(log2f(hD[r * 8] / 32.f));
                    printf("行%d←L%d.b%d ", r, idx / 4, idx % 4);
                }
            } else {
                for (int n = 0; n < 8; n++) {
                    int idx = 127 + (int)lroundf(log2f(hD[n] / 32.f));
                    printf("列%d←L%d.b%d ", n, idx / 4, idx % 4);
                }
            }
            printf("\n");
        }
    }
    return 0;
}
