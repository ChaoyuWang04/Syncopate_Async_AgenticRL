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
