"""E29 · ckpt 只存 LoRA（`_patch_lora_only_ckpt`）的验收测试。

守住的命题：**save 出去的模型分片只含 lora_ 键；load 回来后 lora 键取 ckpt 值、
其余键取当前模型（基座）值；全参训练（无 lora 键）自动回退全量，不靠开关兜底。**

补丁在真 FSDP 栈上的整跑行为由冒烟 A/B + 断点续跑验证（判据行 `[ckpt-lora]`）；
这里钉死纯逻辑与安装协议（守则②：降维成秒级复现）。
"""

from __future__ import annotations

import pytest
import torch

from syncopate.train.verl_patches import filter_lora_state, merge_lora_into


def _sd(lora: bool = True) -> dict:
    d = {
        "model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros(4),
        "model.embed_tokens.weight": torch.zeros(4),
    }
    if lora:
        d["model.layers.0.self_attn.q_proj.lora_A.default.weight"] = torch.ones(2)
        d["model.layers.0.self_attn.q_proj.lora_B.default.weight"] = torch.full((2,), 2.0)
    return d


def test_filter_keeps_only_lora_keys() -> None:
    out = filter_lora_state(_sd())
    assert out is not None and len(out) == 2
    assert all("lora_" in k for k in out)


def test_filter_returns_none_for_full_finetune() -> None:
    """全参训练必须回退全量——把没有 lora 的模型存成空字典是数据丢失，不是优化。"""
    assert filter_lora_state(_sd(lora=False)) is None


def test_merge_updates_lora_and_keeps_base() -> None:
    full = _sd()
    ckpt = {k: v + 10 for k, v in filter_lora_state(_sd()).items()}
    merged = merge_lora_into(full, ckpt)
    assert torch.equal(
        merged["model.layers.0.self_attn.q_proj.lora_A.default.weight"], torch.full((2,), 11.0)
    ), "lora 键必须来自 ckpt"
    assert torch.equal(
        merged["model.embed_tokens.weight"], torch.zeros(4)
    ), "非 lora 键必须保持当前模型（基座）的值"


def test_merge_rejects_unknown_keys() -> None:
    """ckpt 里有当前模型没有的键 = 模型结构/target_modules 变了，必须硬失败。"""
    with pytest.raises(RuntimeError, match="拒绝合成加载"):
        merge_lora_into(_sd(lora=False), {"model.layers.9.lora_A.weight": torch.ones(1)})


def test_patch_installs_wrappers_idempotently() -> None:
    """补丁装上后类方法被替换、幂等标记在位；重复安装不套娃。"""
    from verl.utils.checkpoint import fsdp_checkpoint_manager as M

    from syncopate.train.verl_patches import _patch_lora_only_ckpt

    _patch_lora_only_ckpt()
    save1, load1 = M.FSDPCheckpointManager.save_checkpoint, M.FSDPCheckpointManager.load_checkpoint
    assert getattr(M.FSDPCheckpointManager, "_syncopate_lora_only_ckpt", False)
    _patch_lora_only_ckpt()          # 第二次必须是 no-op
    assert M.FSDPCheckpointManager.save_checkpoint is save1
    assert M.FSDPCheckpointManager.load_checkpoint is load1
