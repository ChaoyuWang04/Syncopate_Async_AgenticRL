"""ckpt 相关的前提断言 —— **一份实现，所有读 ckpt 的路径共用**。

★ 为什么要单独一个模块（2026-08-18）

E21 是被 `syncopate.train.ckpt_to_adapter` 里一句顺手写的断言炸出来的：
「DDP 下各 rank 的 LoRA 应该相同」。而当时**另外两个读 ckpt 的脚本没有这句话** ——
`scripts/tools/rl_ckpt_drift.py` 和 `syncopate.train.prune_ckpts` 都在静默地只取一个 rank。

⇒ 这正是本项目记过的形状：**保护性逻辑写在了其中一条代码路径上**
   （见 .claude/memory/project-mechanism-not-wired.md 第四形态）。
   修法是**提成一个函数，所有路径都调**，而不是每处各写一遍。
"""

from __future__ import annotations

from pathlib import Path


def assert_ranks_identical(actor_dir: str | Path, sample: int = 24) -> int:
    """DDP 下每个 rank 是完整副本 ⇒ 只读一个 rank 就代表全部。**验证这个前提。**

    返回参与比较的张量数；单 rank / 已瘦身时返回 0（没得比，不算通过也不算失败）。

    ⚠️ 这条断言写在「两个东西应当相同」上：非黑即白、不需要阈值、
       不会因为基线漂移而失效 —— 这是它比任何「某个数应该在某范围里」的判据更值钱的原因。
    """
    import torch

    actor = Path(actor_dir)
    ranks = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if len(ranks) < 2:
        return 0
    a = torch.load(ranks[0], map_location="cpu", weights_only=False)
    b = torch.load(ranks[1], map_location="cpu", weights_only=False)
    keys = [k for k in a if "lora_" in k][:sample]
    bad = [k for k in keys if not torch.equal(a[k], b[k])]
    del a, b
    if bad:
        raise SystemExit(
            f"🔴 {ranks[0].name} 与 {ranks[1].name} 的 LoRA 权重不同"
            f"（抽查 {len(keys)} 个张量，{len(bad)} 个不一致）\n"
            "   ⇒ 梯度没有跨 rank 同步（见 docs/infra_exp/E21-ddp-not-syncing.md）。\n"
            "   只读一个 rank 会静默地只拿到 1/3 的训练结果。"
        )
    return len(keys)
