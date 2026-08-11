"""本地推理评测：让模型**自己生成**，跑完整 rollout 并打分。

★ 为什么必须有这个，SFT 的 val_loss 不够用

SFT 的 loss 是 **teacher forcing** 下算的——每一步都喂正确的前缀，模型只需要预测
下一个 token。而真实 rollout 是**自回归**的：第 3 步的输入是它自己第 2 步的输出，
错误会一路累积放大。

所以 `val_loss=0.001` 和 "能不能跑通一条轨迹" 是两个几乎无关的量。
0.6B 在冒烟测试里出过 42 次格式错误，而它的 teacher-forced loss 很低。
**唯一有意义的 SFT 效果度量，是真的让它自己走一遍。**

不走 verl / Ray，直接 transformers generate —— 几十条 case 足够看趋势，
而且能在几分钟内给出 SFT 前后的对照。

    python -m syncopate.train.eval_local --model models/Qwen3-0.6B --limit 20
    python -m syncopate.train.eval_local --model models/Qwen3-0.6B \
        --adapter checkpoints/sft/qwen06b_v1 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from syncopate.core.schemas import CaseBundle
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.train.rollout_loop import MAX_PROMPT_LENGTH, RolloutConfig, run_rollout

ROOT = Path(__file__).resolve().parents[2]


class HFEngine:
    """把 transformers 的 generate 包成核心循环要的接口。"""

    def __init__(self, model, tokenizer, max_new_tokens: int, temperature: float) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = next(model.parameters()).device

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.95 if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return out[0][len(prompt_ids):].tolist()


def load_model(model_path: str, adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # 合并进权重，推理更快
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def load_frozen_eval(batch_dir: Path, split_dir: Path, limit: int | None) -> list[CaseBundle]:
    """从冻结的 EVAL 桶取 case。

    ★ 这是唯一可信的评测来源。之前用「排序后每 N 条取一条」从 val 里选，
    而那个 val 和 train 同模板、且实测有 6 条内容完全相同的泄漏——
    在它上面得到的所有分数都不可采信。
    """
    ids = json.loads((split_dir / "eval_cases.json").read_text(encoding="utf-8"))["case_ids"]
    if limit:
        ids = ids[:limit]
    return [CaseBundle.read(batch_dir, cid) for cid in ids]


def load_cases(batch_dir: Path, per_class: int, split_every: int) -> list[CaseBundle]:
    """★ 按 signal_class **分层**取样，每类固定条数。

    早期版本是"排序后每 N 条取一条"，结果 20 条里全是 CLAR_ 和 GRAD_
    ——因为 case_id 字母序把它们排在了前面。于是 all_low / long_tail / high_risk
    这些**最该关心的难任务一条都没评到**，而那才是判断"能不能开始 RL"的依据。

    只从 val 切分里挑（和 data build 的 val_every 对齐），保证评的是没训过的。
    """
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(manifest["entries"], key=lambda e: e["case_id"])
    val_entries = [e for i, e in enumerate(entries) if i % split_every == 0]
    # ★ 按 case_id 前缀（= 模板）分层，不是按 signal_class。
    # clarify / reject 的 signal_class 也是 "graded"，按 signal_class 分层会让
    # 字母序靠前的 CLAR_ 吃满配额，真正的 GRAD_（240 条主力）一条都评不到。
    by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in val_entries:
        by_class.setdefault(entry["case_id"].split("_")[0], []).append(entry)
    picked: list[dict[str, Any]] = []
    for _, group in sorted(by_class.items()):
        picked.extend(group[:per_class])
    return [CaseBundle.read(batch_dir, e["case_id"]) for e in picked]


def _report_defer(rows: list[dict]) -> None:
    """★ M1 的验收指标：`defer` 的**双向**准确率。

    只测「该 defer 时 defer 了」会训出一个什么都不敢做的 agent——
    它在业务上和一个乱动手的 agent 一样没用，但单向指标上是满分。
    所以必须同时测「数据已经收敛时，有没有多余的 defer」。

    分母按**采样次数**算而不是 case 数：8 次里错 3 次，按众数统计会显示成全对。
    """
    def deferred(group: list[dict]) -> tuple[int, int]:
        return (sum(b == "defer" for r in group for b in r["behaviors"]),
                sum(len(r["behaviors"]) for r in group))

    hit, total = deferred([r for r in rows if r["expected_behavior"] == "defer"])
    if not total:
        return                      # 这批评测里没有 defer 类 case，指标无意义
    miss, miss_total = deferred([r for r in rows if r["expected_behavior"] != "defer"])
    print("\n★ defer 双向准确率 —— 单向达标没有意义")
    print(f"  该 defer 时 defer 了     {hit}/{total} ({hit / total:.0%})")
    print(f"  不该 defer 却 defer 了   {miss}/{miss_total} ({miss / miss_total:.1%})"
          "   ← 必须接近 0，否则就是训出了一个什么都不敢做的 agent")


# 恢复动作的痕迹。system.wait 是最干净的一个：正常流程里**完全用不到它**。
RECOVERY_MARKERS = ("system.wait",)


def _report_recovery(rows: list[dict]) -> None:
    """★ 恢复动作的**双向**准确率 —— 和 defer 双向指标同构。

    只测「该恢复时恢复了」会训出一个**过度恢复**的 agent：
    没出事也去等待、也去重复查证。它在单向指标上满分，
    但在业务上是把每次操作都拖慢几十秒。

    分组依据是 `env.failures` 非空 —— 由 case 声明，不是事后推断。
    分母按**采样次数**算：8 次里多等 3 次，按众数统计会显示成全对。

    ⚠️ 这个指标存在的直接原因：SFT 桶里 F 类占 45%（因为 F 贡献了 48% 的死格），
    我们需要**测**它有没有导致过度恢复，而不是靠调配额提前猜。
    """
    def rate(group: list[dict]) -> tuple[int, int]:
        hit = sum(any(t in RECOVERY_MARKERS for t in seq)
                  for r in group for seq in r.get("tool_seqs", []))
        total = sum(len(r.get("tool_seqs", [])) for r in group)
        return hit, total

    should = [r for r in rows if r.get("has_failure")]
    should_not = [r for r in rows if not r.get("has_failure")]
    hit, total = rate(should)
    if not total:
        return
    over, over_total = rate(should_not)
    print("\n★ 恢复动作双向准确率 —— 和 defer 同理，单向达标没有意义")
    print(f"  有故障时用了恢复动作   {hit}/{total} ({hit / total:.0%})")
    print(f"  无故障却用了恢复动作   {over}/{over_total} ({over / over_total:.1%})"
          "   ← 必须接近 0，否则就是训出了一个见谁都先等三十秒的 agent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地自回归推理评测")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录，不给就是基座")
    parser.add_argument("--batch", default="data/batches/v2")
    parser.add_argument("--split-dir", default="data/splits/v2",
                        help="用冻结 EVAL 桶（推荐）；设为空字符串则退回旧的 per-class 取样")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=4, help="每个 signal_class 取几条")
    parser.add_argument("--split-every", type=int, default=8, help="和 data build 的 val_every 对齐")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="测组内方差必须 >0；要看确定性行为才设 0")
    parser.add_argument("--samples-per-case", type=int, default=4,
                        help="★ 模拟 GRPO 的组大小。组内 reward 方差=0 就没有梯度")
    parser.add_argument("--latency-scale", type=float, default=0.0,
                        help="评测时把 480 秒审核压掉；测异步时才设 1.0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    domain = build_domain()
    domain.registry.latency_scale = args.latency_scale
    model, tokenizer = load_model(str((ROOT / args.model).resolve()), args.adapter)
    engine = HFEngine(model, tokenizer, args.max_new_tokens, args.temperature)
    if args.split_dir:
        bundles = load_frozen_eval(ROOT / args.batch, ROOT / args.split_dir, args.limit)
    else:
        bundles = load_cases(ROOT / args.batch, args.per_class, args.split_every)

    label = f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (基座)")
    print(f"[评测] {label}   {len(bundles)} 条 case，temperature={args.temperature}")

    rows = []
    started = time.time()
    for bundle in bundles:
        group = []
        for k in range(args.samples_per_case):
            output = asyncio.run(run_rollout(
                bundle, registry=domain.registry, tokenizer=tokenizer, generate=engine,
                config=RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                                     max_prompt_length=MAX_PROMPT_LENGTH, max_response_length=2048),
                rollout_id=f"eval{k}",
            ))
            result = score_trajectory(
                bundle, output.trajectory, output.sandbox,
                policy_scorer=domain.policy_scorer, decision_fn=domain.decision_fn, caps=domain.caps)
            group.append({
                "reward": result.reward,
                "parse_ok": output.trajectory.parse_ok,
                "parse_errors": output.metrics["parse_errors"],
                "tool_errors": output.metrics["tool_errors"],
                "truncated": output.metrics["truncated"],
                "num_steps": output.metrics["num_steps"],
                "caps": [h.name for h in result.cap_hits],
                "behavior": output.trajectory.behavior,
                # 恢复动作的双向指标要用：这一次采样调了哪些工具
                "tools": [a.name for a in output.trajectory.actions],
            })
        group_rewards = [g["reward"] for g in group]
        rows.append({
            "case_id": bundle.case_id,
            "signal_class": bundle.case.metadata.signal_class,
            "template": bundle.case_id.split("_")[0],
            "reward": statistics.mean(group_rewards),
            "reward_std": statistics.pstdev(group_rewards) if len(group_rewards) > 1 else 0.0,
            "reward_max": max(group_rewards),
            "group": group_rewards,
            "parse_ok": sum(g["parse_ok"] for g in group) / len(group),
            "parse_errors": sum(g["parse_errors"] for g in group),
            "tool_errors": sum(g["tool_errors"] for g in group),
            "truncated": sum(g["truncated"] for g in group) / len(group),
            "num_steps": statistics.mean(g["num_steps"] for g in group),
            "caps": [c for g in group for c in g["caps"]],
            "behavior": collections.Counter(g["behavior"] for g in group).most_common(1)[0][0],
            # ★ 逐次采样的行为要全留下：defer 的双向准确率是按采样次数算的，
            # 只留众数会把「8 次里错 3 次」压成一个看不见的 0
            "behaviors": [g["behavior"] for g in group],
            "expected_behavior": bundle.verifier.expected_behavior,
            "tool_seqs": [g["tools"] for g in group],
            # ★ 这条 case 有没有声明失败剧本 —— 恢复动作双向指标的分组依据
            "has_failure": bool(bundle.env.failures),
        })

    rewards = [r["reward"] for r in rows]
    print(f"\n{'指标':<26}{'值'}")
    print("-" * 46)
    print(f"{'平均 reward':<26}{statistics.mean(rewards):.3f}")
    print(f"{'reward > 0 的比例':<26}{sum(r > 0 for r in rewards)}/{len(rows)}")
    print(f"{'终答解析成功':<26}{sum(r['parse_ok'] for r in rows)}/{len(rows)}")
    print(f"{'格式错误总次数':<26}{sum(r['parse_errors'] for r in rows)}")
    print(f"{'工具报错总次数':<26}{sum(r['tool_errors'] for r in rows)}")
    print(f"{'撞步数上限':<26}{sum(r['truncated'] for r in rows)}/{len(rows)}")
    print(f"{'平均步数':<26}{statistics.mean(r['num_steps'] for r in rows):.1f}")
    print(f"{'耗时':<26}{time.time()-started:.0f}s")

    print("\n★ 按模板 —— 判断能不能开始 RL 看的是 组内std，不是 mean")
    print(f"  {'模板':<14}{'n':>3}{'mean':>8}{'best':>8}{'组内std':>9}{'有梯度':>8}{'截断率':>8}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    for name, group in sorted(by.items()):
        stds = [g["reward_std"] for g in group]
        has_grad = sum(s > 0.01 for s in stds)
        print(f"  {name:<14}{len(group):>3}{statistics.mean(g['reward'] for g in group):>8.3f}"
              f"{max(g['reward_max'] for g in group):>8.3f}{statistics.mean(stds):>9.3f}"
              f"{has_grad}/{len(group):>6}{statistics.mean(g['truncated'] for g in group):>8.0%}")
    all_stds = [r["reward_std"] for r in rows]
    live = [r for r in rows if r["reward_std"] > 0.01]
    sat = [r for r in rows if r["reward_std"] <= 0.01 and r["reward"] > 0.9]
    dead = [r for r in rows if r["reward_std"] <= 0.01 and r["reward"] < 0.15]
    stuck = [r for r in rows if r["reward_std"] <= 0.01 and 0.15 <= r["reward"] <= 0.9]
    print(f"\n★ 零梯度格子的构成 —— **决定 SFT 该往哪调**")
    print(f"  有梯度（σ>0.01）      {len(live):>3}/{len(rows)}   RL 能学的就是这些")
    print(f"  饱和（σ=0, r>0.9）    {len(sat):>3}/{len(rows)}   base 已经会了，**SFT 不该碰**")
    print(f"  全灭（σ=0, r<0.15）   {len(dead):>3}/{len(rows)}   ★ **SFT 冷启动的目标**")
    print(f"  卡死（σ=0, 中间分）   {len(stuck):>3}/{len(rows)}   系统性走偏，查 cap")
    if dead:
        print(f"  全灭清单: {[r['case_id'] for r in dead]}")
    if stuck:
        print(f"  卡死清单: {[(r['case_id'], round(r['reward'],2)) for r in stuck]}")

    _report_defer(rows)
    _report_recovery(rows)

    caps = collections.Counter(c for r in rows for c in r["caps"])
    print("\ncap 命中:", dict(caps) or "无")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"label": label, "rows": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"\n明细 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
