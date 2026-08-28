"""verl 补丁的验收测试。

★ 为什么这一份必须是**数值**测试，不能只看"跑起来了"

`verl_patches.P2` 补的是 fully_async 在 DDP 下的 CPU 参数快照，而它服务的是
`_compute_old_log_prob`：用**第一个策略版本**的参数重算 old_log_prob（MIS）。
存错或回填错**不会报错**，只会让 IS 权重和 advantage 悄悄偏掉 —— 正是
"机制在但没接上"最难发现的那一档。⇒ 用逐字节往返比对钉死它。
"""

from __future__ import annotations

import torch
from torch import nn

from syncopate.train.verl_patches import ddp_load_from_cpu, ddp_save_to_cpu


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_save_restore_roundtrip_is_bitwise_exact() -> None:
    """存 → 改坏 → 回填，必须逐字节回到原样。"""
    model = _model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    state, spec = ddp_save_to_cpu(model)
    assert spec is None, "DDP 下没有分片规则，spec 必须是 None（回填时靠它选分支）"

    with torch.no_grad():                      # 把权重改坏，模拟"训练又走了几步"
        for p in model.parameters():
            p.add_(1.0)
    assert not torch.equal(model[0].weight, before["0.weight"]), "前提：确实改坏了"

    ddp_load_from_cpu(model, state, spec)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name]), f"{name} 回填后与原值不一致"


def test_snapshot_is_detached_from_later_updates() -> None:
    """快照必须是**副本**，不能是引用 —— 否则"存下来的 v1"会跟着后续训练一起漂。"""
    model = _model()
    state, _ = ddp_save_to_cpu(model)
    saved = state["0.weight"][0].clone()

    with torch.no_grad():
        model[0].weight.add_(3.0)

    assert torch.equal(state["0.weight"][0], saved), "快照被后续更新改动了 ⇒ MIS 会用到错的 v1"


def test_missing_parameters_are_skipped_not_crashed() -> None:
    """存的时候没有的参数直接跳过（和上游 load 的语义一致），不能炸。"""
    model = _model()
    state, spec = ddp_save_to_cpu(model)
    del state["0.weight"]

    ddp_load_from_cpu(model, state, spec)      # 不抛异常即可


def test_patch_delegates_to_upstream_when_dtensor_present(monkeypatch) -> None:
    """★ 有 DTensor（真在分片）时必须原样交回上游，一个字都不改。

    这条守的是补丁的**作用范围**：它只该在 DDP 下接管。哪天有人把 fsdp_size 改回 -1，
    补丁不能把上游的分片逻辑顶掉。
    """
    from verl.utils import fsdp_utils

    from syncopate.train import verl_patches

    calls: list[str] = []
    monkeypatch.setattr(fsdp_utils, "fsdp2_sharded_save_to_cpu",
                        lambda model: (calls.append("upstream"), ({}, "spec"))[1])
    monkeypatch.setattr(fsdp_utils, "fsdp2_sharded_load_from_cpu",
                        lambda *a: calls.append("upstream_load"))
    monkeypatch.setattr(fsdp_utils, "_syncopate_ddp_cpu_copy", False, raising=False)
    monkeypatch.setattr(verl_patches, "_model_has_dtensor", lambda model: True)

    verl_patches._patch_fsdp_cpu_copy_for_ddp()
    fsdp_utils.fsdp2_sharded_save_to_cpu(_model())
    fsdp_utils.fsdp2_sharded_load_from_cpu(_model(), {}, "spec")

    assert calls == ["upstream", "upstream_load"], "有 DTensor 时补丁不该接管"


# ---- E13：只存可训练参数（proximal anchor 快照）----------------------------
#
# 背景：fully_async 的 decoupled 模式每步要把模型换成 v1 参数、算完再换回来。
# 原实现拷贝全部 named_parameters（Qwen3-4B+LoRA 实测 8.309 GB），而冻结基座
# 跨版本逐字节相同 —— 只有 3.18% 的 LoRA 参数会变。见 docs/infra_exp/E13-*.md。


def _lora_like_model() -> torch.nn.Module:
    """模拟 PEFT 的形状：大的冻结基座 + 小的可训练 adapter。"""
    m = torch.nn.Module()
    m.base = torch.nn.Linear(64, 64)          # 冻结
    m.lora_A = torch.nn.Linear(64, 4, bias=False)   # 可训练
    m.lora_B = torch.nn.Linear(4, 64, bias=False)   # 可训练
    for p in m.base.parameters():
        p.requires_grad = False
    return m


def test_snapshot_only_stores_trainable_params() -> None:
    model = _lora_like_model()
    state, spec = ddp_save_to_cpu(model)
    assert spec is None
    assert set(state) == {"lora_A.weight", "lora_B.weight"}, \
        f"冻结基座不该进快照，实际存了 {sorted(state)}"


def test_restore_recovers_trainable_and_leaves_frozen_untouched() -> None:
    model = _lora_like_model()
    state, spec = ddp_save_to_cpu(model)
    base_before = model.base.weight.detach().clone()

    with torch.no_grad():                      # 模拟又训了几步
        model.lora_A.weight.add_(1.0)
        model.lora_B.weight.add_(1.0)
    ddp_load_from_cpu(model, state, spec)

    assert torch.equal(model.lora_A.weight, state["lora_A.weight"][0]), "可训练参数没恢复"
    assert torch.equal(model.base.weight, base_before), "冻结基座被动过"


def test_full_finetune_model_still_saves_everything() -> None:
    """前提失效时要自动退回全量：全参微调下不能漏存。"""
    model = _lora_like_model()
    for p in model.base.parameters():
        p.requires_grad = True
    state, _ = ddp_save_to_cpu(model)
    assert "base.weight" in state and "base.bias" in state


def test_nvtx_wrapper_keeps_timing_semantics(monkeypatch):
    """NVTX 包装**不许**改变计时语义：耗时照记、异常照抛、range 成对。

    ★ 为什么要守：这层包装是为了让 nsys 能按阶段归属 kernel（E01 §8-1 / B12 的门槛）。
    如果它顺手吞了异常或漏了 `range_pop`，后果是**trace 里的层级全乱**，
    而且要等到分析 trace 那一刻才发现 —— 那时 GPU 时间已经花掉了。
    """
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace

    from syncopate.train import verl_patches as vp

    pushed: list[str] = []
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(nvtx=SimpleNamespace(
        range_push=lambda n: pushed.append(n),
        range_pop=lambda: pushed.append("<pop>"),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # 造一个最小的 verl.utils.profiler.performance
    calls: dict = {}

    @contextmanager
    def orig_marked(name, timing_raw, *a, **kw):
        timing_raw[name] = timing_raw.get(name, 0) + 1
        calls["entered"] = name
        yield

    perf = ModuleType("verl.utils.profiler.performance")
    perf.marked_timer = orig_marked
    profiler_pkg = ModuleType("verl.utils.profiler")
    profiler_pkg.performance = perf
    profiler_pkg.marked_timer = orig_marked          # 模拟「已经被别人绑走的名字」
    for name, mod in [("verl", ModuleType("verl")), ("verl.utils", ModuleType("verl.utils")),
                      ("verl.utils.profiler", profiler_pkg),
                      ("verl.utils.profiler.performance", perf)]:
        monkeypatch.setitem(sys.modules, name, mod)

    vp._patch_nvtx_timers()

    timing: dict = {}
    with perf.marked_timer("update_actor", timing):
        pass
    assert timing["update_actor"] == 1          # 原来的计时行为没变
    assert pushed == ["syncopate/update_actor", "<pop>"]
    # ★ 已经被别的模块绑走的那份引用也要换掉（否则补丁只对源模块生效）
    assert profiler_pkg.marked_timer is perf.marked_timer

    # 异常路径：必须抛出去，且 range 仍然闭合
    pushed.clear()
    try:
        with perf.marked_timer("ref", timing):
            raise ValueError("boom")
    except ValueError:
        pass
    assert pushed == ["syncopate/ref", "<pop>"]


def test_fsdp_align_patch_makes_every_shard_16b_aligned(monkeypatch):
    """A17 · 对齐补丁的两条判据：**每个分片 16 字节对齐** + **数据不丢**。

    ★ 为什么第二条同样重要：这个补丁改的是 flat parameter 的尺寸。
    只验"变快了"是不够的 —— 尺寸算错会静默地把参数切错位，而训练照样跑。
    """
    import torch
    from torch.distributed.fsdp import _flat_param as fp

    from syncopate.train import verl_patches as vp

    monkeypatch.setattr(fp.FlatParamHandle, "_syncopate_aligned", False, raising=False)
    vp._patch_fsdp_shard_alignment()

    for numel, dtype, world in [(100, torch.bfloat16, 3), (1000003, torch.bfloat16, 3),
                                (12345, torch.float32, 3), (33643606 * 3, torch.bfloat16, 3)]:
        t = torch.arange(numel, dtype=torch.float32).to(dtype)
        shards = []
        for r in range(world):
            chunk, pad = fp.FlatParamHandle._get_unpadded_shard(t, r, world)
            nbytes = (chunk.numel() + pad) * t.element_size()
            # ★ 判据①：补齐之后每 rank 的字节数是 16 的倍数
            assert nbytes % 16 == 0, f"numel={numel} dtype={dtype} rank={r} 字节={nbytes}"
            shards.append(chunk)
        # ★ 判据②：把分片首尾相接，前 numel 个元素必须和原张量逐个相等（尾部是零填充）
        joined = torch.cat([s.reshape(-1) for s in shards])
        assert joined.numel() >= numel
        assert torch.equal(joined[:numel].float(), t.float())


def test_convert_padding_right_nosync_bitwise_vs_library():
    """乒乓修理⑤：零同步版 convert_padding 必须与 PG 库函数逐位等价。

    覆盖三形态：右垫连续掩码（常态）· 带洞掩码（多轮工具段清零 ⇒ 需要真压缩，
    裁剪偷懒法在这会错）· 全空行边界。max_len 按库的口径 = mask 行和的最大值。
    """
    import torch
    from prefix_grouper import PrefixGrouper
    from syncopate.train.verl_patches import _convert_padding_right_nosync

    g = torch.Generator().manual_seed(28)
    for trial in range(6):
        b, S = 5, 40
        x = torch.randint(1, 1000, (b, S), generator=g)
        if trial % 3 == 0:      # 右垫连续
            lens = torch.randint(1, S + 1, (b,), generator=g)
            mask = torch.arange(S)[None, :] < lens[:, None]
        elif trial % 3 == 1:    # 带洞
            mask = torch.rand(b, S, generator=g) > 0.4
            mask[0] = False; mask[0, 3] = True     # 近空行
        else:                   # 洞 + 整行边界
            mask = torch.rand(b, S, generator=g) > 0.7
            mask[:, 0] = True                       # 保证每行非空（库要求 max>0）
        ref = PrefixGrouper.convert_padding(x, mask.long(), padding_mode="right")
        max_len = int(mask.sum(1).max())
        got = _convert_padding_right_nosync(x, mask.long(), max_len)
        assert got.shape == ref.shape, f"trial{trial} 形状 {got.shape} vs {ref.shape}"
        assert torch.equal(got, ref), f"trial{trial} 不逐位等价"
