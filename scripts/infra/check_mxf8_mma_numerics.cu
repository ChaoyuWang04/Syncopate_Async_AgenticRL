// A3-② 后半① · kind::mxf8f6f4 块缩放 mma 的数值语义对拍（单 warp · m16n8k32 · 真数据）
// 验两件：① e4m3 元素的乘加与 fp32 参考逐位级一致（fp8 精确值 ⇒ 误差应在 1e-5 量级）
//         ② ue8m0 缩放语义：均匀缩放 s 下 D = 2^(sa-127) · 2^(sb-127) · A@B（两档 127/129）
// 编译：nvcc -arch=sm_120a -O3 scripts/infra/check_mxf8_mma_numerics.cu -o /tmp/check_mxf8
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void mma_mxf8(const uint8_t* A, const uint8_t* B, float* D, uint32_t scale_byte) {
    int t = threadIdx.x, g = t >> 2, q = t & 3;
    auto pack = [](const uint8_t* p, int stride, int r, int c) {
        return (uint32_t)p[r * stride + c] | ((uint32_t)p[r * stride + c + 1] << 8) |
               ((uint32_t)p[r * stride + c + 2] << 16) | ((uint32_t)p[r * stride + c + 3] << 24);
    };
    uint32_t a0 = pack(A, 32, g, q * 4), a1 = pack(A, 32, g + 8, q * 4);
    uint32_t a2 = pack(A, 32, g, q * 4 + 16), a3 = pack(A, 32, g + 8, q * 4 + 16);
    // B [k=32][n=8] 行主序；fragment 取 k 连续 4 个
    uint32_t b0 = (uint32_t)B[(q * 4 + 0) * 8 + g] | ((uint32_t)B[(q * 4 + 1) * 8 + g] << 8) |
                  ((uint32_t)B[(q * 4 + 2) * 8 + g] << 16) | ((uint32_t)B[(q * 4 + 3) * 8 + g] << 24);
    uint32_t b1 = (uint32_t)B[(q * 4 + 16) * 8 + g] | ((uint32_t)B[(q * 4 + 17) * 8 + g] << 8) |
                  ((uint32_t)B[(q * 4 + 18) * 8 + g] << 16) | ((uint32_t)B[(q * 4 + 19) * 8 + g] << 24);
    uint32_t s = scale_byte * 0x01010101u;      // 四字节同缩放 ⇒ 无论选择器映射到谁都=均匀
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{0,0},{%11},{0,0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(s), "r"(s));
    D[g * 8 + q * 2] = d0;
    D[g * 8 + q * 2 + 1] = d1;
    D[(g + 8) * 8 + q * 2] = d2;
    D[(g + 8) * 8 + q * 2 + 1] = d3;
}

int main() {
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8];
    srand(7);
    for (int i = 0; i < 16 * 32; i++) {
        float v = (rand() % 2000 - 1000) / 100.f;
        __nv_fp8_e4m3 q(v);
        hA[i] = *(uint8_t*)&q;
        fA[i] = float(q);
    }
    for (int i = 0; i < 32 * 8; i++) {
        float v = (rand() % 2000 - 1000) / 100.f;
        __nv_fp8_e4m3 q(v);
        hB[i] = *(uint8_t*)&q;
        fB[i] = float(q);
    }
    uint8_t *dA, *dB; float *dD;
    cudaMalloc(&dA, sizeof hA); cudaMalloc(&dB, sizeof hB); cudaMalloc(&dD, 16 * 8 * 4);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);

    int fails = 0;
    for (uint32_t sb : {127u, 129u}) {                    // 1.0 与 4.0（2^2）
        mma_mxf8<<<1, 32>>>(dA, dB, dD, sb);
        float hD[16 * 8];
        cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { printf("❌ launch: %s\n", cudaGetErrorString(e)); return 1; }
        float expect_scale = std::pow(2.f, float(sb) - 127.f);
        expect_scale *= expect_scale;                      // sa 与 sb 同值
        float maxrel = 0, maxref = 0;
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 8; n++) {
                float ref = 0;
                for (int k = 0; k < 32; k++) ref += fA[m * 32 + k] * fB[k * 8 + n];
                ref *= expect_scale;
                float got = hD[m * 8 + n];
                maxref = fmaxf(maxref, fabsf(ref));
                maxrel = fmaxf(maxrel, fabsf(got - ref));
            }
        maxrel /= fmaxf(maxref, 1e-9f);
        printf("scale_byte=%u（×%.0f）  max|Δ|/‖ref‖∞ = %.3e  %s\n", sb, expect_scale,
               maxrel, maxrel < 1e-4 ? "✅" : "❌");
        if (maxrel >= 1e-4) fails++;
    }
    printf(fails ? "VERDICT: ❌ 数值语义不符\n"
                 : "VERDICT: ✅ mxf8 块缩放 mma 数值语义对拍通过（元素乘加 + ue8m0 缩放两档）\n");
    return fails;
}
