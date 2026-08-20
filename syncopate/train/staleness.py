"""离线合成 staleness，量 σ²(k) 曲线 —— 单卡也能做的那部分研究主线。

★★★ 为什么可以不跑真异步

研究假设（docs/syncopate/23-research-question.md）的核心是

    ESS/N ≈ exp(−T · σ²(k))

其中 k 是 policy staleness、T 是序列长度。**异步只是产生 staleness 的一种方式**，
不是 staleness 的定义。要量 σ²(k)，只需要一对相隔 k 步的 policy：

    用 π_{t−k} 生成轨迹  →  用 π_t 重算同一串 token 的 logprob  →  ratio 分布 → ESS

这么做比真异步**更干净**：k 是精确控制的，不是由调度随机决定；
而且没有吞吐、抢占、partial rollout 这些混杂因素。

    真异步       k 随机、需要 ≥2 GPU、含混杂、能测吞吐和分布漂移
    离线合成     k 精确、单卡、干净、**测不了吞吐和漂移**

⇒ 单卡先把 H1/H2 的曲线做出来（假设主体），吞吐和漂移等上云再补（工程佐证）。

★★ 一个必须小心的地方：token 对齐

重算 logprob 必须**逐 token 对齐到同一串 token 上**。如果拿新 policy 重新生成一遍
再比，量到的是「两个 policy 生成的东西有多不一样」，那是另一个量，
和重要性采样权重没有关系。这里用 teacher forcing：喂同一串 token，取每个位置的 logprob。

    python -m syncopate.train.staleness \
        --old checkpoints/grpo/run1/global_step_10/actor/lora_adapter \
        --new checkpoints/grpo/run1/global_step_20/actor/lora_adapter --k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from syncopate.core.schemas import CaseBundle
from syncopate.domains.adcampaign import build_domain
from syncopate.pipeline.split import load_bucket
from syncopate.train.eval_local import HFEngine, load_model
from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
from syncopate.train.rollout_loop import RolloutConfig, run_rollout

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def logprobs_of(model, token_ids: list[int], start: int) -> list[float]:
    """teacher forcing：喂同一串 token，取 [start:] 每个位置**实际那个 token** 的 logprob。

    ⚠️ 不是重新生成再比。重新生成量到的是「两个 policy 写出来的东西差多少」，
    而 IS 权重要的是「同一串 token 在两个 policy 下的概率比」——完全不同的两个量。
    """
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[0, 1:]
    picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked[start - 1:].tolist()


def ess_from_log_ratios(log_ratios: list[list[float]], level: str) -> dict[str, float]:
    """由逐 token 的 log ratio 算 ESS/N。

    sequence-level：整条序列的 log ratio 求和后取指数 —— 无偏，但方差随 T 指数增长
    token-level   ：逐 token 权重 —— 有 O(T²Δ) 偏差，但方差与 T 无关

    这正是 23-research-question 里那个对偶：没有一方在所有 T 上占优。
    """
    if level == "sequence":
        weights = [math.exp(min(sum(seq), 20.0)) for seq in log_ratios]
    else:
        weights = [math.exp(min(x, 20.0)) for seq in log_ratios for x in seq]
    total = sum(weights)
    if total <= 0:
        return {"ess_over_n": 0.0, "n": len(weights)}
    normalized = [w / total for w in weights]
    ess = 1.0 / sum(w * w for w in normalized)
    return {
        "ess_over_n": ess / len(weights),
        "n": len(weights),
        "w_mean": statistics.mean(weights),
        "w_max": max(weights),
        "w_min": min(weights),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线合成 staleness，量 σ²(k)")
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--old", required=True, help="生成轨迹的那个 policy（π_{t−k}）")
    parser.add_argument("--new", default=None, help="重算 logprob 的那个 policy（π_t）；不给就是基座")
    parser.add_argument("--k", type=int, default=0, help="两个 ckpt 相隔多少个更新步，只用于记录")
    parser.add_argument("--batch", default="data/batches/v3")
    parser.add_argument("--split-dir", default="data/splits/v3")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="默认 = MAX_RESPONSE_LENGTH（契约推导，同 eval_local）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.max_new_tokens is None:
        from syncopate.train.rollout_budget import MAX_RESPONSE_LENGTH
        args.max_new_tokens = MAX_RESPONSE_LENGTH

    domain = build_domain()
    domain.registry.latency_scale = 0.0
    case_ids = load_bucket(ROOT / args.split_dir, "eval")[: args.limit]
    bundles = [CaseBundle.read(ROOT / args.batch, cid) for cid in case_ids]

    # ---- 1. 用 π_{t−k} 生成轨迹，记下 token 和它当时的 logprob ----
    old_model, tokenizer = load_model(str((ROOT / args.model).resolve()), args.old)
    engine = HFEngine(old_model, tokenizer, args.max_new_tokens, args.temperature)
    traces: list[dict[str, Any]] = []
    for bundle in bundles:
        output = asyncio.run(run_rollout(
            bundle, registry=domain.registry, tokenizer=tokenizer, generate=engine,
            config=RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                                 max_prompt_length=MAX_PROMPT_LENGTH, max_response_length=MAX_RESPONSE_LENGTH),
            rollout_id="stale"))
        if output.metrics["logprob_coverage"] < 0.99:
            # 占位 logprob 会让 ratio 变成噪声，这种轨迹不能进统计
            continue
        traces.append({
            "case_id": bundle.case_id,
            "tokens": output.prompt_ids + output.response_ids,
            "start": len(output.prompt_ids),
            "old_logprobs": list(output.response_logprobs),
        })
    del old_model
    torch.cuda.empty_cache()

    if not traces:
        print("没有一条轨迹带全 logprob —— 检查 rollout 是否返回了 log_probs")
        return 1

    # ---- 2. 用 π_t 对同一串 token 重算 logprob ----
    new_model, _ = load_model(str((ROOT / args.model).resolve()), args.new)
    log_ratios: list[list[float]] = []
    lengths: list[int] = []
    for trace in traces:
        new_lp = logprobs_of(new_model, trace["tokens"], trace["start"])
        old_lp = trace["old_logprobs"][: len(new_lp)]
        pairs = list(zip(new_lp, old_lp))
        if not pairs:
            continue
        log_ratios.append([n - o for n, o in pairs])
        lengths.append(len(pairs))

    flat = [x for seq in log_ratios for x in seq]
    sigma2 = statistics.pvariance(flat) if len(flat) > 1 else 0.0
    mean_T = statistics.mean(lengths)
    seq = ess_from_log_ratios(log_ratios, "sequence")
    tok = ess_from_log_ratios(log_ratios, "token")

    print(f"[staleness] k={args.k}   π_old={args.old}   π_new={args.new or '基座'}")
    print(f"  轨迹 {len(log_ratios)} 条，平均 T = {mean_T:.0f} token")
    print(f"  单 token log-ratio：均值 {statistics.mean(flat):+.5f}  方差 σ² = {sigma2:.3e}")
    print(f"  ★ sequence-level  ESS/N = {seq['ess_over_n']:.3f}"
          f"   （理论预测 exp(−T σ²) = {math.exp(-mean_T * sigma2):.3f}）")
    print(f"  ★ token-level     ESS/N = {tok['ess_over_n']:.3f}   （理论上与 T 无关）")
    print(f"  权重 max/min: {seq['w_max']:.3f} / {seq['w_min']:.3e}")
    if seq["ess_over_n"] < 0.3:
        print("  ⚠️ sequence-level 已跌破 0.3 运维红线 —— 该降级到 token-level")

    payload = {"k": args.k, "old": args.old, "new": args.new, "mean_T": mean_T,
               "sigma2_per_token": sigma2, "sequence": seq, "token": tok,
               "predicted_ess": math.exp(-mean_T * sigma2)}
    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  明细 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
