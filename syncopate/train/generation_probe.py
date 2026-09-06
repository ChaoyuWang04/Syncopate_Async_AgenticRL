"""固定旧 smoke 的题目，只做生成诊断；不训练，不改正式采样参数。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

SEEDS = (1234, 1235, 1236, 1237)
REPEAT_WARNING_TOKENS = 128  # T1-1 跑前登记；只是读数，不提前打断生成。
ARMS = ("full", "model_defaults")
EXPECTED_CASES = 4


def parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(value.split(','))
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('只接受登记的独立实验臂，不接受空项或重复项')
    return arms


def validate_provenance(provenance: dict, source_run: Path) -> None:
    from syncopate.core.model_paths import STUDENT_MODEL
    expected_adapter = Path("checkpoints/sft") / source_run.name / "SELECTED"
    if (provenance.get("base") != STUDENT_MODEL or
            provenance.get("adapter") != expected_adapter.as_posix()):
        raise ValueError("SFT 模型与指定 smoke run 身份不符；不接受子串或近似路径")


def sampling_for(arm: str, model: Path) -> dict:
    from syncopate.train.rollout_budget import SAMPLING_TEMPERATURE, SAMPLING_TOP_P, SAMPLING_TOP_K
    if arm == "full":
        return dict(temperature=SAMPLING_TEMPERATURE, top_p=SAMPLING_TOP_P, top_k=SAMPLING_TOP_K)
    if arm == "model_defaults":
        settings = json.loads((model / "generation_config.json").read_text())
        return {key: settings[key] for key in ("temperature", "top_p", "top_k")}
    raise ValueError(f"未登记的诊断臂：{arm}")


def load_inputs(source_run: Path, batch: Path, tokenizer, registry):
    from syncopate.core.schemas import CaseBundle
    from syncopate.train.rollout_loop import build_messages, chat_template_ids, CHAT_TEMPLATE_KWARGS
    bundles, inputs = {}, {}
    dumps = sorted((source_run / "rollout_dumps").glob("*.jsonl"))
    if not dumps:
        raise FileNotFoundError("缺少指定 run 的 rollout_dumps")
    for dump in dumps:
        for line in dump.read_text().splitlines():
            row = json.loads(line)
            cid = str(row["gts"])
            if cid in bundles:
                continue
            bundle = CaseBundle.read(batch, cid)
            ids = chat_template_ids(tokenizer, build_messages(bundle, None, tokenizer=tokenizer),
                tools=registry.menu(None), add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
            if tokenizer.decode(ids, skip_special_tokens=True) != row["input"]:
                raise ValueError("重建 prompt 与旧 dump 不同，停止诊断")
            bundles[cid], inputs[cid] = bundle, ids
    return bundles, inputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, assistant_turn_budget
    from syncopate.train.rollout_loop import run_rollout, RolloutConfig
    from syncopate.train.verl_agent_loop import score, write_artifact
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    provenance = json.loads((args.model / "syncopate_provenance.json").read_text())
    validate_provenance(provenance, args.source_run)
    domain = build_domain()
    domain.registry.latency_scale = 0
    bundles, inputs = load_inputs(args.source_run, args.batch, tokenizer, domain.registry)
    if len(bundles) != EXPECTED_CASES:
        raise ValueError("题数与预注册的 4 道 B02 题不符")
    sampling = sampling_for(args.arm, args.model)
    plan = {"arm": args.arm, "model": str(args.model), "source_run": str(args.source_run),
            "provenance": provenance, "seeds": SEEDS, "sampling": sampling,
            "response_budget": MAX_RESPONSE_LENGTH,
            "inputs": [{"case_hash": hashlib.sha256(cid.encode()).hexdigest()[:10],
                        "tokens": len(ids), "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
                       for cid, ids in inputs.items()]}
    if args.dry_run:
        print(json.dumps({"dry_run_ok": True, **plan}, ensure_ascii=False))
        return 0
    if (args.out / "result.json").exists() or (args.out / "artifacts").exists():
        raise FileExistsError("不覆盖已有生成证据")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    from syncopate.train.eval_local import VLLMEngine
    engine = VLLMEngine(str(args.model), None, MAX_RESPONSE_LENGTH, sampling["temperature"], .85)
    plan["process_start_method"] = engine.process_start_method

    async def one(cid, bundle, seed):
        output = await run_rollout(bundle, registry=domain.registry, tokenizer=tokenizer, generate=engine,
            config=RolloutConfig(max_assistant_turns=assistant_turn_budget(bundle.case.max_steps),
                                max_prompt_length=MAX_PROMPT_LENGTH, max_response_length=MAX_RESPONSE_LENGTH),
            sampling_params={**sampling, "seed": seed}, rollout_id=f"seed_{seed}", run_id=args.out.name)
        if output.prompt_ids != inputs[cid]:
            raise ValueError("真实生成输入与预检 token 不相等")
        result = score(bundle, output, domain)
        write_artifact(args.out / "artifacts", bundle, output, result)
        metrics = output.metrics
        return {"case_hash": hashlib.sha256(cid.encode()).hexdigest()[:10], "seed": seed,
                "reward": result.reward, "parse_ok": output.trajectory.parse_ok,
                **{key: metrics[key] for key in ("num_steps", "tool_errors", "parse_errors", "truncated",
                    "truncation_reason", "unclosed_think_turns", "max_generation_tokens",
                    "max_repeat_span_tokens", "generation_stop_reason_missing", "generate_seconds")},
                "response_tokens": len(output.response_ids),
                "repeat_warning": metrics["max_repeat_span_tokens"] >= REPEAT_WARNING_TOKENS}

    async def run():
        rows = await asyncio.gather(*(one(cid, bundle, seed) for cid, bundle in bundles.items() for seed in SEEDS))
        result = {**plan, "rows": rows, "count": len(rows),
                  "repeat_warnings": sum(row["repeat_warning"] for row in rows),
                  "unclosed_trajectories": sum(row["unclosed_think_turns"] > 0 for row in rows),
                  "truncated": sum(row["truncated"] for row in rows),
                  "stop_reason_missing": sum(row["generation_stop_reason_missing"] for row in rows)}
        result["generation_shape_ok"] = not any(result[key] for key in (
            "repeat_warnings", "unclosed_trajectories", "truncated", "stop_reason_missing"))
        (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))

    from syncopate.train.vllm_process import run_with_engine
    asyncio.run(run_with_engine(engine, run))
    return 0  # 质量 WARN 只记账；身份或程序异常仍为非零退出。


if __name__ == "__main__":
    raise SystemExit(main())
