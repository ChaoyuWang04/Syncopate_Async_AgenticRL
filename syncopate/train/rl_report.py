"""把 RL 的每步 dump 聚合成曲线，并**补报到同一个 wandb run**。

★★★ 为什么需要这个模块：verl 不会把我们的指标上报

verl 的 `compute_data_metrics` 只认 `__num_turns__` 和 `tool_call_counts` 两个
non_tensor 字段。我们通过 `reward_extra_info` 回去的东西——19 条 cap、四个子分、
generate/tool/wall 三段耗时、logprob_coverage——**训练时一个都不上 wandb**，
只在 validation 时以 `val-aggregate/*` 出现。

而这些恰恰是两个目标的核心观测量：
  · cap 分解 → reward hacking 的唯一指纹（reward 涨了但 cap 没降）
  · 三段耗时 → 异步 vs 同步对照实验的因变量

改 verl 源码是下策（升级就丢）。`trainer.rollout_data_dir` 已经把每步**实际进入
训练**的样本连同这些字段 dump 成了 JSONL，这里把它读回来、聚合、用
`wandb.init(resume="allow")` 补写进**同一个 run 的同一个 step**。
曲线看起来和原生上报的没有区别，只是晚到几分钟。

    python -m syncopate.train.rl_report checkpoints/grpo/run1 --wandb-run <run_id>
    python -m syncopate.train.rl_report checkpoints/grpo/run1              # 只打印
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_steps(dump_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """按 step 读 dump。文件名就是 global_step。"""
    steps: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        steps[int(path.stem)] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return steps


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """一步的聚合。cap 取**命中率**不取次数——次数会随 batch 大小变，率才可比。"""
    n = len(rows)
    out: dict[str, float] = {"rollouts": n}
    for key in ("reward", "raw_reward", "num_steps", "tool_errors", "parse_errors",
                "truncated", "generate_seconds", "tool_seconds", "wall_seconds",
                "logprob_coverage", "num_active_caps"):
        if key in rows[0]:
            out[f"syncopate/{key}"] = statistics.mean(r[key] for r in rows)
    for key in rows[0]:
        if key.startswith(("cap/", "subscore/")):
            out[f"syncopate/{key}"] = sum(r[key] for r in rows) / n
    # ★ 阻塞代价：最慢的一条 vs 平均。sync 下整批等最慢的，这个比值就是浪费的上界
    walls = [r["wall_seconds"] for r in rows if "wall_seconds" in r]
    if walls:
        out["syncopate/wall_seconds_max"] = max(walls)
        out["syncopate/head_of_line_ratio"] = max(walls) / (statistics.mean(walls) or 1.0)
    return out


def template_mix(rows: list[dict[str, Any]]) -> dict[str, float]:
    """这一步**实际训练到**的任务类型构成。gts 里存的是 case_id。"""
    counts = collections.Counter(r["gts"].split("_")[0] for r in rows if isinstance(r.get("gts"), str))
    total = sum(counts.values()) or 1
    return {f"mix_trained/{k}": v / total for k, v in counts.items()}


def dispatched_mix(log_path: Path) -> dict[str, float] | None:
    """**下发并跑完**的任务类型构成（由 agent loop 逐条追加）。

    ★ 它和 mix_trained 的差就是「分布漂移」：
      sync 下有 barrier，两者恒等；async 下短任务先回、长任务被 partial_rollout
      切断，训练分布会系统性偏向短任务 —— 而长链正是 agentic 的核心能力。
    """
    if not log_path.exists():
        return None
    counts: collections.Counter = collections.Counter()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[json.loads(line)["case_id"].split("_")[0]] += 1
    total = sum(counts.values()) or 1
    return {f"mix_dispatched/{k}": v / total for k, v in counts.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RL dump 聚合 + wandb 补报")
    parser.add_argument("run_dir", help="trainer.default_local_dir，里面有 rollout_dumps/")
    parser.add_argument("--rollout-log", default=None,
                        help="agent loop 的下发日志（算分布漂移用），默认 <run_dir>/dispatched.jsonl")
    parser.add_argument("--wandb-run", default=None, help="给了就补报到这个 run id")
    parser.add_argument("--wandb-project", default="syncopate")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    steps = load_steps(run_dir / "rollout_dumps")
    if not steps:
        print(f"{run_dir}/rollout_dumps 里没有 dump —— 是不是没设 trainer.rollout_data_dir？")
        return 1

    log_path = Path(args.rollout_log) if args.rollout_log else run_dir / "dispatched.jsonl"
    disp = dispatched_mix(log_path)

    per_step = {}
    for step, rows in steps.items():
        metrics = {**aggregate(rows), **template_mix(rows)}
        if disp:
            # 漂移 = 训练分布 − 下发分布，逐类型报，并给一个总量（L1 距离的一半）
            drift = {}
            for key, trained in ((k, v) for k, v in metrics.items() if k.startswith("mix_trained/")):
                tpl = key.split("/", 1)[1]
                drift[f"mix_drift/{tpl}"] = trained - disp.get(f"mix_dispatched/{tpl}", 0.0)
            metrics.update(drift)
            metrics["mix_drift/total_variation"] = sum(abs(v) for v in drift.values()) / 2
        per_step[step] = metrics

    print(f"{'step':>5}{'n':>5}{'reward':>9}{'步数':>7}{'墙钟':>8}{'最慢/均':>9}{'cap数':>7}")
    for step, m in sorted(per_step.items()):
        print(f"{step:>5}{int(m['rollouts']):>5}{m.get('syncopate/reward', 0):>9.3f}"
              f"{m.get('syncopate/num_steps', 0):>7.1f}{m.get('syncopate/wall_seconds', 0):>8.1f}"
              f"{m.get('syncopate/head_of_line_ratio', 0):>9.2f}"
              f"{m.get('syncopate/num_active_caps', 0):>7.2f}")

    last = per_step[max(per_step)]
    caps = {k.split("/", 2)[-1]: v for k, v in last.items() if k.startswith("syncopate/cap/") and v}
    print(f"\n末步 cap 命中率: {({k: round(v, 2) for k, v in sorted(caps.items())}) or '无'}")
    if disp:
        print(f"末步分布漂移(total variation): {last.get('mix_drift/total_variation', 0):.3f}"
              "   （sync 下应恒为 0；显著大于 0 = 训练分布已经偏离下发分布）")
    else:
        print(f"（没找到 {log_path}，跳过分布漂移；async 跑之前要确认它在）")

    if args.wandb_run:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_run, resume="allow")
        for step, metrics in sorted(per_step.items()):
            run.log(metrics, step=step)
        run.finish()
        print(f"\n已补报 {len(per_step)} 步 -> wandb run {args.wandb_run}")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(per_step, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"明细 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
