"""最小 LoRA SFT 训练器。

**为什么不用 verl 的 SFT trainer**：它会自己对 messages 做 chat template 渲染，
而整段渲染和我们 RL 侧的增量拼接**天生逐 token 不相等**（Qwen3 只给最后一个
assistant 轮加空 `<think>` 块）。详见 `pipeline/sft_replay.py` 的模块 docstring。

我们的数据已经是**预分词**的 `input_ids` + `loss_mask`，由同一个 rollout 循环
回放 gold 产出——所以训练器只需要：喂 token、按 mask 算 loss。这一百来行比接
verl SFT trainer 更简单，也彻底消除了两阶段分布不一致的风险。

单卡场景下 FSDP 也没有意义（world_size=1 会退化成 NO_SHARD），所以直接用
普通的 PyTorch 训练循环 + peft。

    python -m syncopate.train.sft --model models/Qwen3-0.6B --epochs 3
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]

# 重定向到文件时 Python 会缓冲 stdout，长训练看不到实时进度。
# 训练脚本的打印量很小，无脑 flush 没有代价。
print = functools.partial(builtins.print, flush=True)  # noqa: A001


class PretokenizedDataset(Dataset):
    """直接吃 parquet 里的 input_ids / loss_mask，不做任何再分词。"""

    def __init__(self, path: Path, max_length: int, group_key: str | None = None) -> None:
        import pandas as pd

        frame = pd.read_parquet(path)
        self.rows = [
            {
                "input_ids": list(row["input_ids"])[:max_length],
                "loss_mask": list(row["loss_mask"])[:max_length],
                "case_id": row["case_id"],
                "group": str(row[group_key]) if group_key and group_key in frame.columns else "all",
            }
            for _, row in frame.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """右侧 padding。

    ★ labels 的构造是这里唯一容易错的地方：
      - padding 位置 -> -100（不算 loss）
      - loss_mask=0 的位置 -> -100（prompt 和工具返回，不算 loss）
      - 只有 loss_mask=1 的 token 才是训练目标
    把工具返回也算进 loss，等于教模型复述环境给它的东西。
    """
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        ids, mask = item["input_ids"], item["loss_mask"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append([token if m == 1 else -100 for token, m in zip(ids, mask)] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def token_balanced_weights(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    """★ 按 **token 数** 而不是样本数做类别均衡。

    这是上一轮实测出来的问题：clarify / reject 各占 8.7% 的**样本**，
    但只占 1.4% 的**监督 token**——因为它们的 gold 是"零个工具调用 + 一句 JSON"，
    每条只有 32 个 token，而 tool_call 类有 200+。

    决定梯度的是 token 不是样本，所以按样本均衡是不够的。这里让每个类别的
    **期望 token 贡献**相等：权重 ∝ 1 / (该类总 token 数)。

    实测后果（v2，未均衡）：SFT 把 reject 从基座的 0.308 直接打到 0.000，
    behavior_mismatch 命中率 100%——模型学到的是"无脑调工具"。
    """
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row["group"], []).append(index)
    group_tokens = {
        g: sum(sum(rows[i]["loss_mask"]) for i in idx) for g, idx in by_group.items()
    }
    total = sum(group_tokens.values())
    weights = [0.0] * len(rows)
    for group, idx in by_group.items():
        # 该组每条样本的权重：让本组的期望 token 贡献 = 总量 / 组数
        share = total / (len(group_tokens) * max(1, group_tokens[group]))
        for i in idx:
            weights[i] = share * sum(rows[i]["loss_mask"])
    report = {
        "groups": {g: {"samples": len(idx), "tokens": group_tokens[g],
                       "token_share_before": round(group_tokens[g] / total, 4),
                       "token_share_after": round(1 / len(group_tokens), 4)}
                   for g, idx in by_group.items()},
    }
    return weights, report


def build_model(model_path: str, lora_rank: int, lora_alpha: int, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if lora_rank <= 0:
        return model, None

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        # ★ 必须挂全部线性层。只挂 q/v 是老习惯，容量差 2.8 倍
        #   （Qwen3-4B r=32：仅注意力 23.6M vs 全线性层 66.1M）。
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    return model, config


def trainable_summary(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"可训练 {trainable/1e6:.1f}M / 总计 {total/1e6:.1f}M ({trainable/total:.2%})"


@torch.no_grad()
def evaluate(model, dataset, device, pad_id: int, batch_size: int) -> dict[str, Any]:
    """按 token 加权的验证 loss，并**按组分别报**。

    只看总 loss 会掩盖问题：上一轮 clarify/reject 只占 1.4% 的 token，
    它们的 loss 就算完全没降，总 loss 也几乎看不出来。
    分组报出来才知道是不是某一类被牺牲了。
    """
    model.eval()
    by_group: dict[str, list[float]] = {}
    totals = [0.0, 0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: (collate(b, pad_id), [r["group"] for r in b]))
    for batch, groups in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        # 逐样本算 loss，才能按组聚合
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none").view(shift_labels.shape)
        valid = (shift_labels != -100)
        for i, group in enumerate(groups):
            n = int(valid[i].sum())
            if n == 0:
                continue
            loss_i = float(losses[i][valid[i]].sum())
            by_group.setdefault(group, []).append((loss_i, n))
            totals[0] += loss_i
            totals[1] += n
    model.train()
    out: dict[str, Any] = {
        "val_loss": totals[0] / max(1, totals[1]),
        "val_ppl": math.exp(min(20.0, totals[0] / max(1, totals[1]))),
        "val_tokens": totals[1],
        "by_group": {},
    }
    for group, items in sorted(by_group.items()):
        s_loss = sum(x for x, _ in items); s_n = sum(n for _, n in items)
        out["by_group"][group] = {"loss": s_loss / max(1, s_n), "tokens": s_n,
                                  "ppl": math.exp(min(20.0, s_loss / max(1, s_n)))}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syncopate 最小 LoRA SFT")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/sft/v3/train.parquet")
    parser.add_argument("--val-file", default="data/sft/v3/val.parquet")
    parser.add_argument("--out", default="checkpoints/sft/run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)      # LoRA 的 lr 比全参高一到两个量级
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1, help="每 N 个 epoch 评一次")
    parser.add_argument("--balance-by", default=None,
                        help="按该列做 token 加权采样，如 behavior。不给则不均衡")
    # ★ 默认开 wandb，和 `launch_rl.py` 对齐（那边 --logger 默认就是 console,wandb）。
    # 之前默认是 None，结果 v8 那轮 SFT 整轮没有任何上报 —— 曲线只剩一个人肉 tail 的日志文件。
    # 训练脚本的默认值必须是「跑完就有记录」，要关得显式说。
    parser.add_argument("--wandb-project", default="syncopate")
    parser.add_argument("--no-wandb", action="store_true", help="显式关掉上报（调试/跑测试用）")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoTokenizer

    model_path = str((ROOT / args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train_set = PretokenizedDataset(ROOT / args.train_file, args.max_length, args.balance_by)
    val_set = PretokenizedDataset(ROOT / args.val_file, args.max_length, args.balance_by)
    collate_fn = lambda batch: collate(batch, pad_id)  # noqa: E731

    sampler = None
    balance_report: dict[str, Any] = {}
    if args.balance_by:
        from torch.utils.data import WeightedRandomSampler

        weights, balance_report = token_balanced_weights(train_set.rows)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
        print(f"[均衡] 按 {args.balance_by} 的 token 加权采样")
        for group, stat in sorted(balance_report["groups"].items()):
            print(f"       {group:<10} 样本 {stat['samples']:>3}  token 占比 "
                  f"{stat['token_share_before']:>6.1%} -> {stat['token_share_after']:>6.1%}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate_fn, drop_last=False)

    model, _ = build_model(model_path, args.lora_rank, args.lora_alpha, dtype)
    model.to(device)
    # ★ LoRA + gradient checkpointing 必须配这两行，否则 backward 直接报
    #   "element 0 of tensors does not require grad"。
    # 原因：LoRA 把 embedding 冻住了，checkpoint 段的输入没有 requires_grad，
    # 重算时接不上计算图。enable_input_require_grads 给输入挂一个 hook 打开梯度；
    # use_reentrant=False 用新版实现，对部分冻结的模型更稳。
    if args.lora_rank > 0:
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    print(f"[模型] {args.model}  {trainable_summary(model)}")
    print(f"[数据] train={len(train_set)} val={len(val_set)}  "
          f"有效 batch = {args.batch_size}×{args.grad_accum} = {args.batch_size * args.grad_accum}")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    run = None
    if args.wandb_project and not args.no_wandb:
        import os

        import wandb

        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        # 名字带 sft- 前缀：SFT 和 RL 上报到同一个 project，不加前缀在列表里分不出来。
        run = wandb.init(project=args.wandb_project,
                         name=args.wandb_run or f"sft-{Path(args.out).name}",
                         job_type="sft",
                         config={**vars(args), "trainable": trainable_summary(model),
                                 "balance": balance_report})
        print(f"[wandb] {args.wandb_mode}  {run.url if args.wandb_mode == 'online' else run.dir}")

    def log(payload: dict[str, Any], step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    base = evaluate(model, val_set, device, pad_id, args.batch_size)
    print(f"[eval] epoch 0 (未训练)  val_loss={base['val_loss']:.4f}  "
          f"ppl={base['val_ppl']:.2f}  监督 token={base['val_tokens']}")
    for group, stat in base["by_group"].items():
        print(f"        {group:<10} loss={stat['loss']:.4f}  ppl={stat['ppl']:.2f}")
    log({"val/loss": base["val_loss"], "val/ppl": base["val_ppl"],
         **{f"val/loss_{g}": v["loss"] for g, v in base["by_group"].items()},
         **{f"val/ppl_{g}": v["ppl"] for g, v in base["by_group"].items()}}, step=0)

    history = [{"epoch": 0, **{k: v for k, v in base.items() if k != "by_group"},
                "by_group": base["by_group"]}]
    global_step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            seen += 1
            if micro_step % args.grad_accum == 0 or micro_step == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                log({"train/loss": float(loss) * args.grad_accum,
                     # grad_norm 是最早能看出训练崩没崩的信号：突然飙高 = 有坏样本或 lr 过大
                     "train/grad_norm": float(grad_norm),
                     "train/lr": scheduler.get_last_lr()[0],
                     "train/epoch": epoch}, step=global_step)
        train_loss = running / max(1, seen)
        line = (f"[epoch {epoch}] train_loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"step={global_step}/{total_steps}  用时={time.time()-started:.0f}s")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            stats = evaluate(model, val_set, device, pad_id, args.batch_size)
            line += f"  val_loss={stats['val_loss']:.4f}  ppl={stats['val_ppl']:.2f}"
            history.append({"epoch": epoch, "train_loss": train_loss, **stats})
            log({"val/loss": stats["val_loss"], "val/ppl": stats["val_ppl"],
                 **{f"val/loss_{g}": v["loss"] for g, v in stats["by_group"].items()},
                 **{f"val/ppl_{g}": v["ppl"] for g, v in stats["by_group"].items()}},
                step=global_step)
            print(line)
            # ★ 分组报出来，才看得到"某一类被牺牲了"
            for group, stat in stats["by_group"].items():
                print(f"          {group:<10} loss={stat['loss']:.4f}  ppl={stat['ppl']:.2f}")
        else:
            print(line)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"          显存峰值 {peak:.1f} GB")
            log({"perf/peak_memory_gb": peak}, step=global_step)

        # ★ 每个 epoch 都存一份 adapter（几十 MB，便宜）。
        #
        # 因为**该选哪个 ckpt 不看 val loss**：手册 §20——SFT 训得越狠，输出熵越低，
        # 接上 GRPO 就探索不动了。我们已经踩过一次：选了 val loss 最低的那个，
        # 结果零梯度格子 63%。要选的是「格式学会了但行为还没定型」的那一版，
        # 而那一版只有在**每个 epoch 都存下来**的前提下才选得到。
        epoch_dir = ROOT / args.out / f"epoch{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)          # LoRA 只存 adapter，几十 MB
    tokenizer.save_pretrained(out_dir)
    (out_dir / "training_history.json").write_text(
        json.dumps({"args": vars(args), "history": history}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[OK] adapter -> {out_dir}")
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
