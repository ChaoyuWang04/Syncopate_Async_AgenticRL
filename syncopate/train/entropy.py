"""输出熵监控：SFT 有没有把「决策」也训成确定值。

★ 为什么这是投入产出比最高的监控项

GRPO 的全部机制是「同一题跑 N 遍、比较谁好谁坏」。模型的输出熵一旦压到 0，
N 遍跑出来一模一样 —— **没有比较对象，就没有 advantage，就没有梯度**。
这时候 RL 不是学得慢，是**完全空转**，而 loss 曲线上什么都看不出来。

我们此前一直用「饱和格子数」推断这件事。那只是熵坍缩的**影子**：
先要 8 次采样都拿同一个 reward，才会显示成饱和。熵才是本体，而且它**更早报警**。

★★ 但熵不是越低越坏，要看坍缩在哪个位置

    格式位（`{` `"behavior"` `}`）熵低是**好事** —— 格式本来就该确定
    决策位（clarify 还是 tool_call、调哪个工具、金额填多少）熵低是**坏事**
                                            —— 这里一确定，RL 就没得学了

所以这个模块分开报两件事：整体熵，和**决策位的熵**。
决策位的定位办法：模型自己生成的 token 里，落在 `"behavior": "…"` 值上、
以及 `<tool_call>` 里 `"name": "…"` 值上的那些位置——它们是「这一步走哪条路」的载体。

    python -m syncopate.train.entropy --model models/Qwen3-4B \
        --adapter checkpoints/sft/v3_ctrl/epoch1 --limit 16
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from syncopate.core.schemas import CaseBundle
from syncopate.domains.adcampaign import build_domain
from syncopate.pipeline.split import (
    DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, assert_same_data_version, load_bucket,
)
from syncopate.train.eval_local import load_model
from syncopate.train.rollout_loop import (
    CHAT_TEMPLATE_KWARGS, MAX_PROMPT_LENGTH, build_messages,
)

ROOT = Path(__file__).resolve().parents[2]
print = functools.partial(builtins.print, flush=True)  # noqa: A001

# 决策位的锚：生成文本里这些片段之后的那个值，就是「这一步走哪条路」
DECISION_ANCHORS = ('"behavior": "', '"behavior":"', '"name": "', '"name":"')


def token_entropy(logits: torch.Tensor) -> float:
    """单个位置的熵，单位 nat。用 log_softmax 而不是先 softmax 再 log —— 数值稳定。"""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum())


class RecordingEngine:
    """包一层 HF generate，边生成边记录每个位置的熵。

    ★ 必须跟着**完整 rollout** 测，不能只测第一轮。

    第一版只测第一个 assistant 轮，结果决策位只有 4 个 —— 因为第一轮几乎必然是
    「查一下现状」，本来就没什么可选的。GRPO 的组内方差主要来自**第 3、4 步**：
    读到 observation 之后往哪拐。只测第一轮等于在没有岔路的地方量岔路。
    """

    def __init__(self, model, tokenizer, max_new_tokens: int, temperature: float) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.device = next(model.parameters()).device
        self.entropy: list[float] = []
        self.decision_entropy: list[float] = []
        self.top1: list[float] = []

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.95 if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # ★ 要 logits 不要 scores：scores 是 temperature/top_p 处理**之后**的，
                # 拿它算熵，量到的是采样器的设置，不是模型的确定程度。
                return_dict_in_generate=True, output_logits=True)

        generated = out.sequences[0][len(prompt_ids):].tolist()
        for step, logits in enumerate(out.logits):
            if step >= len(generated):
                break
            h = token_entropy(logits[0])
            self.entropy.append(h)
            self.top1.append(float(torch.softmax(logits[0].float(), dim=-1).max()))
            prefix = self.tokenizer.decode(generated[:step])
            if any(prefix.endswith(a) for a in DECISION_ANCHORS):
                self.decision_entropy.append(h)
        return generated


def measure(model, tokenizer, bundles, domain, max_new_tokens: int,
            temperature: float) -> dict[str, Any]:
    import asyncio

    from syncopate.train.rollout_loop import MAX_PROMPT_LENGTH, RolloutConfig, run_rollout

    engine = RecordingEngine(model, tokenizer, max_new_tokens, temperature)
    domain.registry.latency_scale = 0.0
    per_case = []
    for bundle in bundles:
        before = len(engine.entropy)
        asyncio.run(run_rollout(
            bundle, registry=domain.registry, tokenizer=tokenizer, generate=engine,
            config=RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                                 max_prompt_length=MAX_PROMPT_LENGTH, max_response_length=2048),
            rollout_id="entropy"))
        chunk = engine.entropy[before:]
        per_case.append({"case_id": bundle.case_id, "tokens": len(chunk),
                         "mean_entropy": round(statistics.mean(chunk), 4) if chunk else 0.0})

    all_entropy, decision_entropy, top1_probs = engine.entropy, engine.decision_entropy, engine.top1
    near_zero = sum(h < 0.05 for h in all_entropy)
    return {
        "positions": len(all_entropy),
        "mean_entropy": round(statistics.mean(all_entropy), 4),
        "median_entropy": round(statistics.median(all_entropy), 4),
        # ★ 决策位单独报。整体熵被格式 token 稀释，看不出决策有没有被训死
        "decision_positions": len(decision_entropy),
        "decision_mean_entropy": round(statistics.mean(decision_entropy), 4) if decision_entropy else None,
        "near_deterministic_ratio": round(near_zero / max(1, len(all_entropy)), 4),
        "mean_top1_prob": round(statistics.mean(top1_probs), 4),
        "per_case": per_case,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出熵监控")
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--adapter", default=None, help="不给就是基座")
    # ★ 默认值来自**一份共用常量**（`pipeline/split.py`）——此前这里写死 v3，
    #   而 `data/batches/v3` 在本机根本不存在。⚠️ 这两个参数必须同时动，见下面的断言。
    parser.add_argument("--batch", default=DEFAULT_BATCH_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--limit", type=int, default=16, help="取冻结 EVAL 的前 N 条")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="只影响采样出的路径，不影响熵的计算（熵用原始 logits）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    # ★ 「两个东西应当相同」型判据：只改一个参数会静默量错另一个 case 集（见 split.py 那一节）
    data_version = assert_same_data_version(args.batch, args.split_dir)

    batch = ROOT / args.batch
    case_ids = load_bucket(ROOT / args.split_dir, "eval")[: args.limit]
    bundles = [CaseBundle.read(batch, cid) for cid in case_ids]
    domain = build_domain()
    model, tokenizer = load_model(str((ROOT / args.model).resolve()), args.adapter)

    label = args.model + (f" + {args.adapter}" if args.adapter else " (基座)")
    print(f"[熵] {label}   {len(bundles)} 条 case")
    stats = measure(model, tokenizer, bundles, domain,
                    args.max_new_tokens, args.temperature)

    print(f"  生成位置数          {stats['positions']}")
    print(f"  平均熵(nat)         {stats['mean_entropy']}")
    print(f"  中位数熵            {stats['median_entropy']}")
    print(f"  ★ 决策位平均熵       {stats['decision_mean_entropy']}   （{stats['decision_positions']} 个位置）")
    print(f"  近乎确定的位置占比    {stats['near_deterministic_ratio']:.1%}   （熵<0.05）")
    print(f"  平均 top-1 概率      {stats['mean_top1_prob']}")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        # ★ 把**数据版本**写进产物：否则下游（`select_sft_ckpt`）只能靠 label 里的
        #   model/adapter 路径判断，一份 v3 时代的审计会被静默当成本次的结果 ——
        #   写这个工具时就是这么撞上 v3 的 `M1_ctrl_epoch1.json` 的。
        path.write_text(json.dumps({"label": label, "data_version": data_version,
                                    "batch": args.batch, "split_dir": args.split_dir,
                                    **stats}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"  明细 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
