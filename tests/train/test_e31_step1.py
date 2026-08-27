"""E31 第 1/2 步 · 训推 lm_head 统一 MXFP8 的单元与验收测试。

★ 为什么这些测试非有不可
第 1 步的失败模式全是"哑的"：开关没接上（两侧还是 bf16，什么都不会报错）、
只接上一侧（恰好制造 §9b 判死的单侧毒状态）、语义与 verl 融合算子悄悄差一项
（temperature 忘除 / entropy dtype 错 / 反向公式差个符号——loss 照降，学的是错的）。
每一种都必须有一个会红的测试对着。

分层：
  U1  默认关 = 逐位走 verl 旧路（公共路径的兜底，比什么都重要）
  U2  开 = 语义与 verl 融合算子同带、分块不变性逐位、temperature 生效
  U3  反向管路 = 与独立复算的解析公式逐位同；entropy 梯度必须炸；wgrad 拒绝
  U4  vLLM 补丁 = 换得上、有效果、与 trainer 路径逐位同源、bias 拒绝、幂等
  A1  离线验收工件（scripts/e31_step1_offline.py 产）达标
  A2  48 步冒烟验收工件（scripts/e31_step1_smoke_check.py 产）达标
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from syncopate.train import unified_fp8

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_JSON = ROOT / "logs" / "e31" / "step1_offline.json"
SMOKE_JSON = ROOT / "logs" / "e31" / "step1_smoke.json"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")

V, K, T = 256, 128, 100  # 满足 kernel 128 约束的最小词表；T 刻意不整除 chunk


def _case(seed=7, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(T, K, generator=g).to(torch.bfloat16).cuda()
    W = torch.randn(V, K, generator=g).mul(0.05).to(torch.bfloat16).cuda()
    ids = torch.randint(0, V, (T,), generator=g).cuda()
    h.requires_grad_(requires_grad)
    return h, W, ids


# ───────────── U1 · 默认关 = 逐位走 verl 旧路 ─────────────

@needs_cuda
def test_u1_flag_off_is_bitwise_verl(monkeypatch) -> None:
    monkeypatch.delenv(unified_fp8.FLAG, raising=False)
    assert not unified_fp8.enabled()
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    h, W, ids = _case()
    lp_ref, ent_ref = FusedLinearForPPO()(hidden_states=h, vocab_weights=W,
                                          input_ids=ids, temperature=1.0)
    lp, ent = unified_fp8.linear_for_ppo(h, W, ids, temperature=1.0)
    assert torch.equal(lp, lp_ref) and torch.equal(ent, ent_ref), \
        "开关关着输出却和 verl 融合算子不同 —— 分派入口在动公共路径"


# ───────────── U2 · 开 = 语义同带 + 分块逐位不变 + temperature ─────────────

@needs_cuda
def test_u2_semantics_vs_verl(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    h, W, ids = _case()
    for temp in (1.0, 0.8):
        lp_ref, ent_ref = FusedLinearForPPO()(hidden_states=h, vocab_weights=W,
                                              input_ids=ids, temperature=temp)
        lp, ent = unified_fp8.linear_for_ppo(h, W, ids, temperature=temp)
        assert lp.dtype == lp_ref.dtype and ent.dtype == ent_ref.dtype
        # 语义带：差异只应来自 GEMM 的 MXFP8 化。随机小词表下量化扰动松，
        # 但中位必须在带内；temperature 忘除的话 lp 会整体偏 |logits|·(1/t−1) 量级，一抓一个准
        d = (lp - lp_ref).abs()
        assert d.median().item() < 0.05, f"temp={temp} 中位 |Δlp|={d.median().item():.3f} 出带"
        assert (ent.float() - ent_ref.float()).abs().median().item() < 0.05


@needs_cuda
def test_u2_chunk_invariance_bitwise(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    h, W, ids = _case()
    lp_a, ent_a = unified_fp8.linear_for_ppo(h, W, ids, chunk_size=32)
    lp_b, ent_b = unified_fp8.linear_for_ppo(h, W, ids, chunk_size=1024)
    # 分块只是内存策略，不许改数值：每个 token 的 lp 只依赖自己那行 logits
    assert torch.equal(lp_a, lp_b) and torch.equal(ent_a, ent_b), "分块大小改变了数值"


# ───────────── U3 · 反向管路 ─────────────

@needs_cuda
def test_u3_backward_matches_manual_formula(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    h, W, ids = _case(requires_grad=True)
    lp, _ent = unified_fp8.linear_for_ppo(h, W, ids, temperature=0.8, chunk_size=32)
    g = torch.Generator(device="cpu").manual_seed(11)
    dlp = torch.randn(T, generator=g).cuda()
    lp.backward(dlp)
    got = h.grad.clone()

    # 独立复算：dlogits = dlp·(onehot − softmax)/temp，再走同一 dgrad kernel
    with torch.no_grad():
        qw, qw_sf = unified_fp8._weight_cache(W, "fwd")
        qwt, qwt_sf = unified_fp8._weight_cache(W, "bwd")
        manual = torch.empty_like(h)
        for s in range(0, T, 32):
            e = min(s + 32, T)
            manual[s:e] = unified_fp8._bwd_chunk(dlp[s:e], h[s:e].detach(), qw, qw_sf,
                                                 qwt, qwt_sf, ids[s:e], 0.8)
    assert torch.equal(got, manual), "autograd 路径与解析公式复算不逐位同 —— 管路有私货"
    assert got.abs().sum() > 0, "梯度全零 —— 反向根本没走到"


@needs_cuda
def test_u3_entropy_grad_and_wgrad_refused(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    h, W, ids = _case(requires_grad=True)
    _lp, ent = unified_fp8.linear_for_ppo(h, W, ids)
    # entropy_coeff=0 已钉：谁要是把 entropy 真接进损失（非零梯度），必须炸而不是静默走错公式
    with pytest.raises(RuntimeError, match="entropy"):
        ent.float().sum().backward()
    W2 = W.clone().requires_grad_(True)
    with pytest.raises(AssertionError, match="冻结"):
        unified_fp8.linear_for_ppo(h, W2, ids)


@needs_cuda
def test_u3_zero_entropy_grad_is_legal(monkeypatch) -> None:
    """verl 在 entropy_coeff=0 时仍把 entropy 连在损失图里 ⇒ 反传全零 dentropy。
    这是合法形态，必须放行且梯度与纯 lp 路径逐位同（首次冒烟在这里炸过，钉死回归）。"""
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    h, W, ids = _case(requires_grad=True)
    lp, ent = unified_fp8.linear_for_ppo(h, W, ids)
    (lp.sum() + 0.0 * ent.float().sum()).backward()      # 0×entropy：verl 的真实图形态
    got = h.grad.clone()
    h2, _, _ = _case(requires_grad=True)
    lp2, _ = unified_fp8.linear_for_ppo(h2, W, ids)
    lp2.sum().backward()
    assert torch.equal(got, h2.grad), "零 dentropy 改变了 lp 梯度 —— 放行逻辑有私货"


# ───────────── U4 · vLLM 补丁 ─────────────

class _FakeLP:
    """LogitsProcessor 的最小替身：补丁只用到 _gather_logits / org_vocab_size。"""
    org_vocab_size = V

    def _gather_logits(self, logits):
        return logits


@needs_cuda
def test_u4_vllm_patch_engages_and_matches_trainer(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    assert unified_fp8.patch_logits_processor(_FakeLP) is True
    assert unified_fp8.patch_logits_processor(_FakeLP) is False, "补丁不幂等"
    h, W, ids = _case()
    lm_head = type("H", (), {"weight": W})()
    logits = _FakeLP()._get_logits(h, lm_head, None)
    assert logits.shape == (T, V) and logits.dtype == torch.bfloat16
    # 有效果：与 bf16 直乘必须可测地不同（防"注册了但没换实现"的第八形态）
    ref_bf16 = h @ W.T
    assert (logits.float() - ref_bf16.float()).abs().max().item() > 1e-3
    # 同源：与 trainer 路径的同一投影函数逐位同（两侧一致的本体）
    qw, qw_sf = unified_fp8._weight_cache(W, "fwd")
    assert torch.equal(logits, unified_fp8._mxf8_logits(h, qw, qw_sf))
    with pytest.raises(RuntimeError, match="bias"):
        _FakeLP()._get_logits(h, lm_head, torch.zeros(V).cuda())


def test_u4_entry_point_registered() -> None:
    from importlib.metadata import entry_points
    eps = {e.name: e.value for e in entry_points(group="vllm.general_plugins")}
    assert eps.get("syncopate_unified_fp8") == "syncopate.train.unified_fp8:register", \
        "vLLM 入口点没登记 —— 补丁到不了 spawn 的 Worker 进程（登记 ≠ 实现的反向：实现了没登记）"


# ───────────── A1/A2 · 验收工件 ─────────────

def test_a1_offline_acceptance() -> None:
    """离线验收（E31 第 1 步①，08-27 就地改写版——全部锚定同尺 bf16 对照臂）。

    ⛔ 原「序列 ΣΔ p95 < ln2」已判死：bf16 对照臂自己在长序列上 p95=2.54 ≫ ln2
    （引擎漂移本来就超 ln2）。阈值改锚对照臂；序列级最终裁决在冒烟（A2）。
    """
    assert OFFLINE_JSON.exists(), "缺离线验收工件 —— 跑 scripts/e31_step1_offline.py"
    d = json.loads(OFFLINE_JSON.read_text())
    base, uni, one = d["baseline_bf16"], d["unified_fp8"], d["one_sided"]
    assert uni["token_abs_median"] <= 2 * base["token_abs_median"], "token |Δlp| 中位 > 2×本底"
    assert uni["token_abs_mean"] <= 2 * base["token_abs_mean"], "token |Δlp| 均值 > 2×本底"
    # 对消的直接读数：统一后逐 token 签名偏置必须回到本底水平（温度偏置机理 E30 §11）
    assert abs(uni["token_bias"]) <= 2 * abs(base["token_bias"]), \
        f"签名偏置 {uni['token_bias']:+.2e} > 2×本底 {base['token_bias']:+.2e} —— 偏置没对消"
    assert uni["seq_abs_sum_p95"] <= 2 * base["seq_abs_sum_p95"], \
        f"序列 |ΣΔ| p95 {uni['seq_abs_sum_p95']:.2f} > 2×本底 {base['seq_abs_sum_p95']:.2f}"
    # 哨兵：单侧毒臂必须显形（判据要能对自己失败）——偏置 ≥5×本底且全序列同号
    assert abs(one["token_bias"]) >= 5 * abs(base["token_bias"]), "单侧毒臂偏置没显形 —— 测量失灵"
    assert one["seq_signed_pos"] in (0, d["n_seqs"]), "单侧毒臂符号不齐 —— 与 §9b 机理矛盾，查测量"
    # 补丁真的生效了：vLLM 两臂（bf16 vs fp8）必须可测地不同
    assert d["patch_effect"]["token_abs_mean"] > 1e-3, \
        "vLLM fp8 臂与 bf16 臂几乎相同 —— 开关没进 Worker 进程（第八形态）"
    assert d["n_seqs"] >= 8


def test_a2_smoke_acceptance() -> None:
    """48 步冒烟验收（E31 第 1 步② + 第 2 步序列 IS 活体）：产自 e31_step1_smoke_check.py。"""
    assert SMOKE_JSON.exists(), "缺冒烟验收工件 —— 跑完 48 步后过 scripts/e31_step1_smoke_check.py"
    d = json.loads(SMOKE_JSON.read_text())
    floor = json.loads((ROOT / "logs" / "e31" / "kl_floor_bf16.json").read_text())["median"]
    assert max(d["kl_values"]) <= 1.5 * floor, \
        f"kl max {max(d['kl_values']):.2e} > 1.5×floor {1.5*floor:.2e}"
    assert d["seq_truncation_frac_max"] <= 0.10, \
        f"序列 IS 截断比例 {d['seq_truncation_frac_max']:.3f} > 0.10"
    assert d["ess_min"] >= 0.85, f"ESS/N 最低 {d['ess_min']:.3f} < 0.85"
    for name, ok in d["eight_criteria"].items():
        assert ok, f"八判据之「{name}」没过"
    assert d["steps_completed"] >= 48
