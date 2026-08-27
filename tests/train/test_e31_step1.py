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


# ───────────── U5 · 第 3 步 trainer 内层 ─────────────

@needs_cuda
def test_u5_merged_equals_split_bitwise(monkeypatch) -> None:
    """位一致基石：vLLM 合并权重（qkv/gate_up 行拼接）与 trainer 分开权重，
    量化字节与 GEMM 输出都必须逐位同 —— 两侧一致的全部数学都压在这条上。"""
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    from syncopate.train.mxfp8_lmhead import quantize_mxfp8
    g = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn(100, 256, generator=g).to(torch.bfloat16).cuda()
    wq = torch.randn(256, 256, generator=g).mul(0.05).to(torch.bfloat16).cuda()
    wk = torch.randn(128, 256, generator=g).mul(0.05).to(torch.bfloat16).cuda()
    merged = torch.cat([wq, wk], 0)
    # 量化字节逐位同（行块量化不跨行）
    mq, msf = quantize_mxfp8(merged)
    q1, sf1 = quantize_mxfp8(wq)
    q2, sf2 = quantize_mxfp8(wk)
    assert torch.equal(mq, torch.cat([q1, q2], 0)) and torch.equal(msf, torch.cat([sf1, sf2], 0))
    # GEMM 输出逐位同（输出元素只依赖自己的行列）
    qwm, qwm_sf = unified_fp8._weight_cache(merged, "fwd")
    qwa, qwa_sf = unified_fp8._weight_cache(wq, "fwd")
    qwb, qwb_sf = unified_fp8._weight_cache(wk, "fwd")
    y_m = unified_fp8._mxf8_logits(x, qwm, qwm_sf)
    y_s = torch.cat([unified_fp8._mxf8_logits(x, qwa, qwa_sf),
                     unified_fp8._mxf8_logits(x, qwb, qwb_sf)], 1)
    assert torch.equal(y_m, y_s), "合并与分开 GEMM 输出不逐位同 —— 两侧一致的地基塌了"


@needs_cuda
def test_u5_inner_backward_matches_manual(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    from syncopate.train.mxfp8_lmhead import _quant_sw, _ext
    g = torch.Generator(device="cpu").manual_seed(41)
    x = torch.randn(100, 128, generator=g).to(torch.bfloat16).cuda().requires_grad_(True)
    W = torch.randn(256, 128, generator=g).mul(0.05).to(torch.bfloat16).cuda()
    y = unified_fp8._MXF8InnerLinearFn.apply(x, W)
    dY = torch.randn(100, 256, generator=g).to(torch.bfloat16).cuda()
    y.backward(dY)
    with torch.no_grad():
        qwt, qwt_sf = unified_fp8._weight_cache(W, "bwd")
        qdy, qdy_sf = _quant_sw(dY.contiguous())
        manual = _ext().mxf8_gemm(qdy, qwt, qdy_sf, qwt_sf)[:100].to(torch.bfloat16)
    assert torch.equal(x.grad, manual), "内层 dgrad 与解析复算不逐位同"
    assert x.grad.abs().sum() > 0
    W2 = W.clone().requires_grad_(True)
    with pytest.raises(AssertionError, match="冻结"):
        unified_fp8._MXF8InnerLinearFn.apply(x.detach(), W2)


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.mlp = torch.nn.Module()
        for host, names in ((self.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                            (self.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                lin = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
                lin.weight.requires_grad_(False)
                setattr(host, nm, lin)


class _TinyModel(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyLayer() for _ in range(n_layers)])


@needs_cuda
def test_u5_trainer_patch_selection_and_gate(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    monkeypatch.setenv(unified_fp8.LAYERS_FLAG, "2")
    monkeypatch.setitem(unified_fp8._TRAINER_INNER_DONE, "done", False)
    m = _TinyModel(3).cuda()
    assert unified_fp8.patch_trainer_inner(m) == 14, "前 2 层 ×7 应 swap 14 个"
    assert getattr(m.layers[1].self_attn.q_proj, "_syncopate_mxf8_inner", False)
    assert not getattr(m.layers[2].self_attn.q_proj, "_syncopate_mxf8_inner", False), "越界层被 swap"
    # 幂等：done 标志防重入，再调零动作（不许绕过标志重扫——严断言只在首扫成立）
    assert unified_fp8.patch_trainer_inner(m) == 0
    # 半接线拒绝：LAYERS>0 但 FLAG 没开 ⇒ 必须炸
    monkeypatch.delenv(unified_fp8.FLAG)
    with pytest.raises(RuntimeError, match="半接线"):
        unified_fp8.quant_layers()


@needs_cuda
def test_u5_flag_off_forward_untouched(monkeypatch) -> None:
    monkeypatch.delenv(unified_fp8.FLAG, raising=False)
    monkeypatch.delenv(unified_fp8.LAYERS_FLAG, raising=False)
    monkeypatch.setitem(unified_fp8._TRAINER_INNER_DONE, "done", False)
    m = _TinyModel(1).cuda()
    x = torch.randn(4, 128).to(torch.bfloat16).cuda()
    ref = m.layers[0].self_attn.q_proj(x)
    assert unified_fp8.patch_trainer_inner(m) == 0
    assert torch.equal(m.layers[0].self_attn.q_proj(x), ref), "开关关着 forward 却变了"


# ───────────── U6 · 第 3 步 vLLM 层选择 ─────────────

@needs_cuda
def test_u6_vllm_inner_selection(monkeypatch) -> None:
    monkeypatch.setenv(unified_fp8.FLAG, "1")
    monkeypatch.setenv(unified_fp8.LAYERS_FLAG, "2")

    class _FakeMethod:                      # UnquantizedLinearMethod 替身
        def apply(self, layer, x, bias=None):
            return x @ layer.weight.T       # bf16 原路径

    import re
    pat = re.compile(unified_fp8._VLLM_INNER_PAT)
    W = torch.randn(256, 128).mul(0.05).to(torch.bfloat16).cuda()
    x = torch.randn(10, 128).to(torch.bfloat16).cuda()

    def mk(prefix):
        return type("L", (), {"prefix": prefix, "weight": W})()

    # 选择逻辑与补丁行为分开验：先验正则本身
    assert pat.search("model.layers.1.self_attn.qkv_proj")
    assert pat.search("model.layers.1.mlp.gate_up_proj")
    assert not pat.search("model.layers.1.self_attn.q_norm")
    assert not pat.search("lm_head")
    # 补丁行为：命中层走 MXFP8（≠bf16 直乘），越界层走原路径（==bf16 直乘）
    import vllm.model_executor.layers.linear as vlin
    monkeypatch.setattr(vlin, "UnquantizedLinearMethod", _FakeMethod)
    assert unified_fp8.patch_vllm_inner() is True
    meth = _FakeMethod()
    y_hit = meth.apply(mk("model.layers.1.self_attn.qkv_proj"), x)
    y_miss = meth.apply(mk("model.layers.2.self_attn.qkv_proj"), x)
    ref = x @ W.T
    assert not torch.equal(y_hit, ref), "命中层没走 MXFP8（第八形态）"
    assert torch.equal(y_miss, ref), "越界层被误量化"
    qw, qw_sf = unified_fp8._weight_cache(W, "fwd")
    assert torch.equal(y_hit, unified_fp8._mxf8_logits(x, qw, qw_sf)), "与 trainer 投影不同源"
    with pytest.raises(RuntimeError, match="bias"):
        meth.apply(mk("model.layers.0.mlp.down_proj"), x, torch.zeros(256).cuda())


# ───────────── A3 · 第 3 步定界工件 + T5 · 第 5 步权重契约 ─────────────

def test_a3_step3_boundary_artifact_consistent() -> None:
    """第 3 步定界（负结果）工件的内部一致性：verdicts 必须能从 stats 复算出来。

    负结果与正结果同权：这个测试防的是工件被手改后叙事与数据脱节。
    复活条件（doc 同步）：token 级 IS 或同构引擎；届时重跑产新工件、判定表随之更新。
    """
    p = ROOT / "logs" / "e31" / "step3_offline.json"
    assert p.exists(), "缺第 3 步定界工件 —— 跑 scripts/e31_step3_offline.py"
    d = json.loads(p.read_text())
    base = d["baseline_bf16_eager"]
    prev = None
    for n in d["groups"]:
        u = d["unified"][str(n)]
        ok = (abs(u["token_bias"]) <= 2 * abs(base["token_bias"])
              and u["seq_abs_sum_p95"] <= 2 * base["seq_abs_sum_p95"]
              and (prev is None or u["token_abs_mean"] <= 1.5 * prev))
        assert d["verdicts"][str(n)] == ok, f"N={n} 的 verdict 与 stats 复算不一致"
        prev = u["token_abs_mean"]
    assert d["verdicts"]["0"], "G0'（仅 lm_head，eager 锚）都不过 —— 测量本身坏了"


@needs_cuda
def test_t5_weight_contract_disk_is_truth() -> None:
    """第 5 步 · 权重契约：两侧的 lm_head 权重都源自同一份磁盘字节（tie 到 embedding），
    两条独立加载路径逐位同 ⇒ 量化缓存必然逐位同。运行期由 [sync-payload] ‖W‖ 探针
    与 kl 地板判据兜底（字节漂了 kl 立刻起飞）。"""
    import json as _json
    from pathlib import Path as _P
    from safetensors import safe_open
    model_dir = _P("/workspace/hf_assets/bases/Qwen3-4B-sft-v13r2-e1")
    if not model_dir.exists():
        pytest.skip("底座不在本机")
    idx = _json.loads((model_dir / "model.safetensors.index.json").read_text())
    assert "lm_head.weight" not in idx["weight_map"], "tie 模型不应单独存 lm_head"
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    with safe_open(model_dir / shard, framework="pt") as f:
        w_disk = f.get_tensor("model.embed_tokens.weight")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    w_hf = m.get_output_embeddings().weight.detach()
    assert torch.equal(w_disk, w_hf.cpu()), "两条加载路径的 lm_head 字节不同 —— 契约破"
    del m
    from syncopate.train.mxfp8_lmhead import quantize_mxfp8
    q1, s1 = quantize_mxfp8(w_disk[:256].cuda().contiguous())
    q2, s2 = quantize_mxfp8(w_hf[:256].cuda().contiguous())
    assert torch.equal(q1, q2) and torch.equal(s1, s2)
