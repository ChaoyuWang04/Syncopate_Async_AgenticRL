"""E31 第 0 步 · 训推 FP8 全盘一致的三个契约测试（T0.1 / T0.2 / T0.3）。

★ 为什么这三个测试非有不可（E31 §0 原理卡）
统一精度的全部收益建立在一个前提上：**量化项在 IS 比率中逐字节对消**。
对消要求 rollout 侧与 trainer 侧对同一份权重/激活产出**逐位相同**的量化字节——
差一个 ulp，序列级 IS 就会沿 1800 token 复利（E30 §9b：16/16 爆表）。
所以在碰任何训练代码之前，先把三件事钉成测试：
  T0.1  量化器位一致 —— 同值输入（不论 bf16/fp32 表示、不论内存布局、不论 stream）
        必须产出 torch.equal 的 uint8；五类张量含全部已知的数值边界。
  T0.2  GEMM 确定性 —— 同输入 ×100 + 换 stream 逐位相同；输出由量化字节唯一决定
        （对拍 dequant 参考）。没有确定性，"两侧一样"就无从谈起。
  T0.3  bf16 本底 —— 两引擎同轨迹 kl 的地板（kl_floor_bf16）是后面每一步验收的
        分母（"kl ≤ 1.5×floor"）；分母必须是新机实测、机器可读、来源可查。

GPU：T0.1/T0.2 需要 CUDA + sm_120 扩展（JIT，首次 ~50 s，有缓存）；显存占用 <1 GB，
属"只出是/否"的 🟡 类，可与训练共存。无 CUDA 时 skip（本机恒有卡，不会触发）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from syncopate.train.mxfp8_lmhead import quantize_mxfp8, swizzle_rows

ROOT = Path(__file__).resolve().parents[2]
KL_FLOOR_JSON = ROOT / "logs" / "e31" / "kl_floor_bf16.json"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")

R, K = 128, 128  # 满足量化块 32 与 swizzle/GEMM 的 128 约束


def _dequant(u8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """逐块反量化 fp32 参考（与 E30 §4 同一定义）。"""
    r, k = u8.shape
    q = u8.view(torch.float8_e4m3fn).float().view(r, k // 32, 32)
    scale = torch.pow(2.0, sf.float() - 127.0).unsqueeze(-1)
    return (q * scale).view(r, k)


def _five_classes(device: str) -> dict[str, torch.Tensor]:
    """E31 T0.1 钦定的五类张量。构造细节即边界本身，别偷懒改小。"""
    g = torch.Generator(device="cpu").manual_seed(31)
    rand = torch.randn(R, K, generator=g).to(torch.bfloat16)
    # 448 边界：fp32 才能表达 ±ulp（bf16 在 448 附近 ulp=2）；三行分别取 448 的
    # 下邻/正中/上邻当块 amax —— 上邻必须把缩放挡位顶上去，否则 e4m3 溢出
    boundary = torch.full((R, K), 1.0)
    edge = torch.tensor([torch.nextafter(torch.tensor(448.0), torch.tensor(0.0)),
                         torch.tensor(448.0),
                         torch.nextafter(torch.tensor(448.0), torch.tensor(1e9))])
    boundary[:3, 0] = edge
    pows = torch.pow(2.0, torch.randint(-6, 7, (R, K), generator=g).float()).to(torch.bfloat16)
    return {
        "random": rand.to(device),
        "amax_zero": torch.zeros(R, K, dtype=torch.bfloat16, device=device),
        "const": torch.full((R, K), 0.37, dtype=torch.bfloat16, device=device),
        "boundary_448": boundary.to(device),  # fp32：±ulp 是它存在的意义
        "pow2": pows.to(device),
    }


# ───────────────────────── T0.1 · 量化器位一致 ─────────────────────────

@needs_cuda
@pytest.mark.parametrize("name", ["random", "amax_zero", "const", "boundary_448", "pow2"])
def test_t01_quantizer_bit_identical(name: str) -> None:
    x = _five_classes("cuda")[name]

    u8_a, sf_a = quantize_mxfp8(x)
    u8_b, sf_b = quantize_mxfp8(x)
    assert torch.equal(u8_a, u8_b) and torch.equal(sf_a, sf_b), "同输入两次调用字节不同"

    # 表示不变性：训推两侧拿到的是同一批 bf16 数值，但上游可能给 fp32 视图或
    # 非连续布局（transpose 切片）。同值 ⇒ 必须同字节，这正是"两侧一致"的接缝。
    u8_c, sf_c = quantize_mxfp8(x.float())
    assert torch.equal(u8_a, u8_c) and torch.equal(sf_a, sf_c), "bf16 与 fp32 同值输入字节不同"

    x_nc = x.t().contiguous().t()
    assert not x_nc.is_contiguous()
    u8_d, sf_d = quantize_mxfp8(x_nc)
    assert torch.equal(u8_a, u8_d) and torch.equal(sf_a, sf_d), "非连续同值输入字节不同"

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        u8_e, sf_e = quantize_mxfp8(x.clone())
    stream.synchronize()
    assert torch.equal(u8_a, u8_e) and torch.equal(sf_a, sf_e), "换 stream 字节不同"


@needs_cuda
def test_t01_quantizer_semantics() -> None:
    """五类张量各自的数值语义 —— 位一致但语义错是"为错误的理由通过"。"""
    xs = _five_classes("cuda")

    u8, sf = quantize_mxfp8(xs["amax_zero"])
    assert torch.equal(u8, torch.zeros_like(u8)), "全零块的 payload 必须是 0"
    assert torch.equal(sf, torch.full_like(sf, 127)), "全零块的缩放指数必须钉 e=0（存 127）"

    u8, sf = quantize_mxfp8(xs["pow2"])
    assert torch.equal(_dequant(u8, sf), xs["pow2"].float()), "2 的整幂必须逐位无损重建"

    x = xs["boundary_448"]
    u8, sf = quantize_mxfp8(x)
    deq = _dequant(u8, sf)
    assert torch.isfinite(deq).all(), "448 边界溢出成 nan/inf —— 缩放挡位选错"
    fp8 = u8.view(torch.float8_e4m3fn).float()
    assert fp8.abs().max().item() <= 448.0, "e4m3 payload 超出最大正规数"
    # 正中与下邻在 scale=1 挡内必须精确；上邻换挡后误差 ≤ 半个 e4m3 ulp（2^-3 相对）
    assert deq[1, 0].item() == 448.0
    torch.testing.assert_close(deq[:3, 0], x[:3, 0], rtol=2 ** -3, atol=0.0)

    u8, _ = quantize_mxfp8(xs["const"])
    assert (u8 == u8[0, 0]).all(), "全同值张量的 payload 必须逐字节相同"

    # 随机张量的整体保真：dequant 相对误差在 e4m3 带内（防"字节稳定但全错"）
    x = xs["random"]
    u8, sf = quantize_mxfp8(x)
    rel = (_dequant(u8, sf) - x.float()).norm() / x.float().norm()
    assert rel.item() < 0.04, f"随机张量 dequant 相对误差 {rel.item():.4f} 超出 e4m3 带"


# ───────────────────────── T0.2 · GEMM 确定性 ─────────────────────────

@pytest.fixture(scope="session")
def gemm_ext():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    from syncopate.train.mxfp8_lmhead import _ext
    return _ext()


@needs_cuda
def test_t02_gemm_deterministic(gemm_ext) -> None:
    M = N = Kg = 256
    g = torch.Generator(device="cpu").manual_seed(202)
    A = torch.randn(M, Kg, generator=g).to(torch.bfloat16).cuda()
    B = torch.randn(N, Kg, generator=g).to(torch.bfloat16).cuda()
    qa, sfa = quantize_mxfp8(A)
    qb, sfb = quantize_mxfp8(B)
    qa_s, qb_s = swizzle_rows(qa), swizzle_rows(qb)

    c0 = gemm_ext.mxf8_gemm(qa_s, qb_s, sfa, sfb)
    for i in range(99):
        ci = gemm_ext.mxf8_gemm(qa_s, qb_s, sfa, sfb)
        assert torch.equal(c0, ci), f"第 {i + 2} 次调用与第 1 次逐位不同"

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cs = gemm_ext.mxf8_gemm(qa_s.clone(), qb_s.clone(), sfa.clone(), sfb.clone())
    stream.synchronize()
    assert torch.equal(c0, cs), "换 stream 后输出逐位不同"

    # 输出必须由量化字节唯一决定：对 dequant 参考只差 bf16 输出舍入
    c_ref = _dequant(qa, sfa) @ _dequant(qb, sfb).T
    rel = (c0.float() - c_ref).norm() / c_ref.norm()
    assert rel.item() < 5e-3, f"对 dequant 参考相对误差 {rel.item():.2e}（应仅剩 bf16 舍入）"


# ───────────────────────── T0.3 · bf16 本底标定 ─────────────────────────

def test_t03_kl_floor_bf16_calibrated() -> None:
    """kl_floor_bf16 必须存在、可溯源、且落在新机 bf16 实测带内。

    产生它：`python scripts/infra/e31_kl_floor.py <bf16 臂训练日志>`。
    这个数是第 1/2 步验收（kl ≤ 1.5×floor）的分母；分母丢了或漂了，
    后面的"通过"全部作废 —— 所以它的存在性与带宽本身是契约。
    """
    assert KL_FLOOR_JSON.exists(), (
        f"缺 {KL_FLOOR_JSON} —— 跑 scripts/infra/e31_kl_floor.py 标定（bf16 臂日志，"
        "如 logs/smoke_newbox_0827_kvauto.log）")
    d = json.loads(KL_FLOOR_JSON.read_text())
    for key in ("kl_values", "median", "max", "source_log", "date"):
        assert key in d, f"标定件缺字段 {key}（来源必须可查）"
    assert len(d["kl_values"]) >= 3, "同步点少于 3 个，标定不算数"
    # 新机 bf16 实测带 3.6–4.8e-4（00-START §6）；fp8 KV 臂是 ~5e-3，混进来会差 10×
    assert 1e-4 < d["median"] < 1e-3, f"median {d['median']:.2e} 不在 bf16 带内（拿错臂？）"
    assert d["max"] < 1.5e-3, f"max {d['max']:.2e} 超带 —— 不是干净的 bf16 本底"
