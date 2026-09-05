"""把 LoRA 训练留下的 RL ckpt 瘦身：只留 LoRA 权重，扔掉那份和基座一模一样的冻结权重。

★ 为什么会有这个问题

verl 0.8.0 的 FSDP checkpoint manager 存的是 `self.model.state_dict()` —— **全量**。
它认得 LoRA（会写 `lora_train_meta.json` 记下 r/alpha），但保存路径没有分叉：

    model_world_size_1_rank_0.pt   8.5 GB
      └─ 基座权重（399 个键）      8.22 GB   ← 和 models/Qwen3-4B 逐字节相同，从没被训过
      └─ LoRA 权重（504 个键）      252 MB   ← 只有这部分是训练产物

**97.1% 是浪费。** 12 个 ckpt × 8.2 GB = 98 GB 的同一份冻结权重复制品。
2026-08-13 磁盘到 82% 才发现。

★ 瘦身之后还能干什么、不能干什么

    ✅ 合并成完整模型（merge_adapter）—— 基座从 models/ 读，LoRA 从这里读
    ✅ staleness 研究（要相隔 k 步的两个 policy）—— 只需要 LoRA 差异
    ✅ 重新评测
    ❌ verl 的断点续跑 —— 它按原文件名和全量结构读

⇒ 所以**默认跳过最后一个 ckpt**（`--keep-last`），保留续跑能力。
   已经跑完、不打算续的 run 可以加 `--all` 一起瘦。

用法：
    python -m syncopate.train.prune_ckpts checkpoints/grpo/v8_sync_e1b --dry-run
    python -m syncopate.train.prune_ckpts checkpoints/grpo/v8_sync_e1b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from syncopate.train.ckpt_guards import assert_ranks_identical

ROOT = Path(__file__).resolve().parents[2]

LORA_ONLY = "model_lora_only.pt"




def prune_one(step_dir: Path, dry_run: bool) -> tuple[int, int]:
    """返回 (原大小, 瘦身后大小)，单位字节。0 表示跳过。"""
    import torch

    actor = step_dir / "actor"
    # ⚠️ 必须 sorted：`next(glob(...))` 取的是文件系统先给的那个，**留下的是哪个 rank 没有记录**。
    #    在 E21 之前这无所谓（"反正都一样"），之后它意味着我们不知道保留的是谁。
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        return 0, 0
    assert_ranks_identical(actor)          # ★ 三份不同就别压成一份 —— 删掉就回不来了
    full = shards[0]
    before = sum(x.stat().st_size for x in shards)   # ★ 全部分片，不是单个（旧版低报 3 倍）
    if dry_run:
        return before, before // 34        # 实测 97.1% 可回收，粗估
    state = torch.load(full, weights_only=False, map_location="cpu")
    lora = {k: v for k, v in state.items() if "lora_" in k}
    if not lora:
        # 不是 LoRA 训练的 ckpt（全参），一个字节都不能动
        return 0, 0
    torch.save(lora, actor / LORA_ONLY)
    after = (actor / LORA_ONLY).stat().st_size
    # ★★ 2026-08-18 修：旧版是 `full.unlink()` —— **只删了读的那一个分片**，
    #    另外两个 8.46 GB 原地留着，而报告打的是「8.46 GB → 255 MB」。
    #    实测残留：m7b_v13e1 的 step_20 剩 1 个、step_25 剩 2 个 = **25.4 GB 白占**。
    #    ⇒ 这是「报告的数和磁盘上的事实不是一回事」——瘦身脚本自己就该是这条的判据。
    for sh in shards:
        sh.unlink()
    return before, after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RL ckpt 瘦身（只留 LoRA 权重）")
    parser.add_argument("run_dir", help="如 checkpoints/grpo/v8_sync_e1b")
    parser.add_argument("--all", action="store_true",
                        help="连最后一个 ckpt 也瘦（会失去 verl 断点续跑能力）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run = ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    steps = sorted(run.glob("global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if not steps:
        print(f"{run} 下没有 global_step_* 目录")
        return 1
    targets = steps if args.all else steps[:-1]
    if not args.all and steps:
        print(f"保留最后一个 ckpt 供续跑: {steps[-1].name}（--all 可一并瘦身）")

    total_before = total_after = 0
    for step_dir in targets:
        before, after = prune_one(step_dir, args.dry_run)
        if before == 0:
            continue
        total_before += before
        total_after += after
        print(f"  {step_dir.name:<18} {before/2**30:6.2f} GB → {after/2**20:6.0f} MB")
    verb = "将回收" if args.dry_run else "已回收"
    print(f"\n{verb} {(total_before - total_after)/2**30:.1f} GB"
          f"（{total_before/2**30:.1f} → {total_after/2**30:.2f} GB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
