"""Syncopate 统一入口。

    python -m syncopate cases list                    # 看有哪些 case
    python -m syncopate cases verify                  # 跑 gold，确认每条都拿得到高分
    python -m syncopate cases build --out data/batches/handwritten  # 落盘手写种子
    python -m syncopate tools list                    # 看工具菜单

设计原则：**一个入口 + 子命令**，不要一个任务一个脚本。
后续的 `data build` / `train sft` / `train rl` 都挂在这里，靠 --config 驱动。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from syncopate.authoring.seed_cases import SEED_BUILDERS, build_all, gold_plan
from syncopate.core.runner import run_plan
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL


def _score_gold(bundle, domain, latency_scale: float):
    """跑一遍 gold 并打分。latency_scale 让 480 秒的等待在验证时不至于卡死。"""
    original = domain.registry.latency_scale
    domain.registry.latency_scale = latency_scale
    try:
        trajectory, sandbox = asyncio.run(
            run_plan(bundle, domain.registry, gold_plan(bundle), final_answer=bundle.gold.final_answer)
        )
    finally:
        domain.registry.latency_scale = original
    result = score_trajectory(
        bundle, trajectory, sandbox,
        policy_scorer=domain.policy_scorer, decision_fn=domain.decision_fn, caps=domain.caps,
    )
    return result, trajectory


# --------------------------------------------------------------------------


def cmd_cases_list(args: argparse.Namespace) -> int:
    print(f"{'case_id':<20} {'signal_class':<14} {'bucket':<24} {'steps':>5}  {'工具菜单'}")
    print("-" * 92)
    for bundle in build_all():
        meta = bundle.case.metadata
        menu = "全部(10)" if bundle.case.tool_menu is None else f"子集({len(bundle.case.tool_menu)})"
        gold_steps = len(bundle.gold.actions) if bundle.gold else 0
        print(f"{bundle.case_id:<20} {meta.signal_class:<14} {meta.bucket:<24} {gold_steps:>5}  {menu}")
    return 0


def cmd_cases_verify(args: argparse.Namespace) -> int:
    """★ gold 必须真跑一遍。老师包里 gold 的分是预烤进文件的，从未被执行验证过。"""
    domain = build_domain()
    print(f"{'case_id':<20} {'reward':>7} {'outcome':>8} {'policy':>7} {'evid':>6} {'effi':>6}  {'状态'}")
    print("-" * 78)
    failures = 0
    for bundle in build_all():
        result, trajectory = _score_gold(bundle, domain, args.latency_scale)
        bad_tools = [(o.tool, o.error) for o in trajectory.observations if not o.ok]
        ok = (not bad_tools and not result.cap_hits
              and result.reward >= (bundle.gold.expected_reward_min if bundle.gold else 0.0))
        failures += (not ok)
        sub = result.subscores
        status = "OK" if ok else "FAIL"
        if bad_tools:
            status += f" 工具报错:{bad_tools}"
        elif result.cap_hits:
            status += f" cap:{[h.name for h in result.cap_hits]}"
        print(f"{bundle.case_id:<20} {result.reward:>7.3f} {sub['outcome']:>8.2f} "
              f"{sub['policy']:>7.2f} {sub['evidence']:>6.2f} {sub['efficiency']:>6.2f}  {status}")
    print("-" * 78)
    print(f"{len(SEED_BUILDERS) - failures}/{len(SEED_BUILDERS)} 条 gold 验证通过")
    return 1 if failures else 0


def cmd_cases_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    manifest = {"version": "manifest_v1", "domain": "adcampaign", "entries": []}
    for bundle in build_all():
        bundle.write(out)
        manifest["entries"].append({
            "case_id": bundle.case_id,
            "signal_class": bundle.case.metadata.signal_class,
            "bucket": bundle.case.metadata.bucket,
            "topology": bundle.case.metadata.topology,
            "difficulty": bundle.case.metadata.difficulty,
        })
    manifest["count"] = len(manifest["entries"])
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] 写出 {manifest['count']} 条四件套 -> {out}")
    for sub in ("cases", "env_snapshots", "verifier_specs", "gold_paths"):
        print(f"     {sub}/ : {len(list((out / sub).glob('*.json')))} 个文件")
    return 0


def cmd_cases_generate(args: argparse.Namespace) -> int:
    from syncopate.authoring.generate import run

    result = run(Path(args.spec), Path(args.out))
    print(f"[OK] 生成 {len(result.accepted)} 条 -> {args.out}")
    for label, counts in (("signal_class", result.counts),
                          ("bucket", result.counts_by("bucket")),
                          ("topology", result.counts_by("topology"))):
        print(f"  -- {label} --")
        for name, count in counts.items():
            print(f"     {name:<24} {count}")
    if result.rejected:
        print(f"     gold 验证刷掉 {len(result.rejected)} 条；前 5 条原因：")
        for r in result.rejected[:5]:
            print(f"       {r.case_id}: {r.reason}")
    return 0


def cmd_data_build(args: argparse.Namespace) -> int:
    from syncopate.pipeline.build_dataset import build

    result = build(
        batch_dir=Path(args.batch), out_dir=Path(args.out), pool=args.pool,
        val_every=args.val_every, artifact_root=Path(args.artifact_root),
        model_path=args.model, max_length=args.max_length,
        split_dir=Path(args.split_dir) if args.split_dir else None,
        full_batch=args.full_batch,
    )
    if args.full_batch:
        print("⚠️  --full-batch：整批构造，冻结 EVAL 也会进训练数据")
    print(f"[OK] {args.pool} 数据 -> {result.out_dir}  train={result.train_count} val={result.val_count}")
    print(f"     {result.train_path}")
    print(f"     {result.val_path}")
    return 0


def cmd_data_split(args: argparse.Namespace) -> int:
    from syncopate.pipeline import dead_grid as dg
    from syncopate.pipeline.split import load_bundles, split, stratum, write

    grids = quota = None
    if args.dead_from:
        # 死格模式：先把 base 审计翻译成格子，再交给 split 去池子里取样。
        bundles = load_bundles(Path(args.batch))
        rows = dg.load_audit(Path(args.dead_from))
        strata = {r["case_id"]: stratum(r["case_id"], bundles[r["case_id"]]) for r in rows}
        grids, kinds = dg.analyze(rows, strata)
        dead_count = len(grids)
        if not args.no_controls:
            grids = dg.add_controls(grids, rows, strata)
        quota = dict(dg.DEFAULT_QUOTA)
        print(f"[死格] 来源 {args.dead_from}（{len(rows)} 条 EVAL 实测）")
        print(f"       格子构成: " + "  ".join(f"{k}={v}" for k, v in kinds.most_common()))
        print(f"       死格 {dead_count} 个 + 对照格 {len(grids)-dead_count} 个   配额 {quota}")

    buckets, report = split(Path(args.batch), eval_per_stratum=args.eval_per_stratum,
                            sft_ratio=args.sft_ratio, dead_grids=grids, quota_by_kind=quota)
    write(buckets, report, Path(args.out))
    print(f"[OK] 三桶 -> {args.out}   模式={report['mode']}")
    print(f"     EVAL {report['counts']['eval']}（冻结） / SFT {report['counts']['sft']} / RL {report['counts']['rl']}")
    print(f"     分层格子数: {report['strata_count']}")
    for key, info in (report.get("dead_grid_selection") or {}).items():
        print(f"       {key:<44} {info['kind']:<10} 证据 {info['eval_evidence']}"
              f"  取 {info['taken']:>3}/{info['available']:<3}")
    print("     ★ 互斥性实测（SHA-256 内容级）:")
    for k, v in report["overlaps_by_content_sha256"].items():
        print(f"       {k:<12} {v}{'  ✅' if v == 0 else '  ❌ 泄漏！'}")
    print(f"       全局重复内容对   {report['duplicate_content_pairs']}"
          f"{'  ✅' if report['duplicate_content_pairs'] == 0 else '  ❌'}")
    return 0


def cmd_data_report(args: argparse.Namespace) -> int:
    from syncopate.pipeline.report import main as report_main

    argv = ["--batch", args.batch, "--val-every", str(args.val_every)]
    if args.out:
        argv += ["--out", args.out]
    return report_main(argv)


def cmd_tools_list(args: argparse.Namespace) -> int:
    domain = build_domain()
    print(f"{'工具名':<28} {'类型':<6} {'谓词':<20} {'延迟(秒)':>9}")
    print("-" * 68)
    for name in domain.default_tool_menu:
        spec = domain.registry.get(name)
        print(f"{spec.name:<28} {spec.kind:<6} {spec.fact_key or '-':<20} {spec.latency_seconds:>9.1f}")
    print("-" * 68)
    print(f"共 {len(domain.default_tool_menu)} 个 | cap 规则 {len(domain.caps.names())} 条: {domain.caps.names()}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncopate", description="Async agentic RL on verl")
    sub = parser.add_subparsers(dest="group", required=True)

    cases = sub.add_parser("cases", help="种子 case 的查看 / 验证 / 落盘").add_subparsers(dest="action", required=True)
    cases.add_parser("list", help="列出全部 case").set_defaults(func=cmd_cases_list)

    verify = cases.add_parser("verify", help="跑 gold 验证每条 case 都拿得到高分")
    verify.add_argument("--latency-scale", type=float, default=0.001,
                        help="把工具的真实延迟按比例缩放，默认 0.001（480s -> 0.48s）")
    verify.set_defaults(func=cmd_cases_verify)

    build = cases.add_parser("build", help="把四件套写到磁盘")
    build.add_argument("--out", default="data/batches/handwritten")
    build.set_defaults(func=cmd_cases_build)

    gen = cases.add_parser("generate", help="按配额批量生成 case（含 gold 实跑验证）")
    gen.add_argument("--spec", default="configs/buckets/v3.yaml")
    gen.add_argument("--out", default="data/batches/v3")
    gen.set_defaults(func=cmd_cases_generate)

    data = sub.add_parser("data", help="训练数据构造").add_subparsers(dest="action", required=True)
    dbuild = data.add_parser("build", help="四件套 -> parquet")
    dbuild.add_argument("--pool", choices=["rl", "sft"], required=True)
    dbuild.add_argument("--batch", default="data/batches/v3")
    dbuild.add_argument("--out", default=None)
    dbuild.add_argument("--val-every", type=int, default=5)
    dbuild.add_argument("--artifact-root", default="data/rollouts")
    dbuild.add_argument("--model", default=TEST_TOKENIZER, help="SFT 分词用")
    dbuild.add_argument("--max-length", type=int, default=8192)
    dbuild.add_argument("--split-dir", default=None,
                        help="三桶目录（如 data/splits/v2），只取 --pool 对应的桶。"
                             "★ 不给就必须显式 --full-batch")
    dbuild.add_argument("--full-batch", action="store_true",
                        help="不按桶、整批构造。会把冻结 EVAL 训进去，只在造 v1 之前的旧批次时用")
    dbuild.set_defaults(func=lambda a: cmd_data_build(_default_out(a)))

    dsplit = data.add_parser("split", help="切三桶（EVAL 冻结 / SFT / RL）并实测互斥性")
    dsplit.add_argument("--batch", default="data/batches/v2")
    dsplit.add_argument("--out", default="data/splits/v2")
    dsplit.add_argument("--eval-per-stratum", type=int, default=2)
    dsplit.add_argument("--sft-ratio", type=float, default=0.20)
    dsplit.add_argument("--dead-from", default=None,
                        help="base 评测审计 json（如 _audit/M0_base_4b.json）。"
                             "给了就走 dead_grid 模式，按实测死格选 SFT 桶")
    dsplit.add_argument("--no-controls", action="store_true",
                        help="不掺对照档。★ 实测过一次：只喂难例会把模型已经做对的行为"
                             "抹掉（defer 97%%→0%%），只在专门做这个对照实验时才用")
    dsplit.set_defaults(func=cmd_data_split)

    dreport = data.add_parser("report", help="训练数据分布体检")
    dreport.add_argument("--batch", default="data/batches/v2")
    dreport.add_argument("--val-every", type=int, default=8)
    dreport.add_argument("--out", default=None)
    dreport.set_defaults(func=cmd_data_report)

    tools = sub.add_parser("tools", help="工具表").add_subparsers(dest="action", required=True)
    tools.add_parser("list", help="列出工具菜单").set_defaults(func=cmd_tools_list)

    return parser


def _default_out(args: argparse.Namespace) -> argparse.Namespace:
    if args.out is None:
        args.out = f"data/{args.pool}/v3"
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
