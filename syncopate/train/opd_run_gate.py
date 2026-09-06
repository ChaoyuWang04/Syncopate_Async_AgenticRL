"""OPD 训练段固定健康闸：真实更新、mask/KL/零泄漏与 final adapter 缺一不可。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from syncopate.train.opd_tokens import ByteLevelTokenMapper, GeneratedSample
from syncopate.train.opd_updates import evaluate_update_audits


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluate_token_audits(out_dir: Path, completion: dict, *, attempted: int,
                          real_attempts: list[tuple[int, int]], objectives: list[dict],
                          mapper: ByteLevelTokenMapper) -> dict:
    """Reconcile raw arrays and measured calls, never a producer's PASS label."""
    tokenizer_evidence = mapper.evidence()
    result = {"ok": False, "errors": [], "ranks": [], "route_forwards": {},
              "warning_counts": {}, "masked_tokens_by_attempt": {}, "tokenizer": tokenizer_evidence}
    routes, warnings, token_totals, loss_totals = Counter(), Counter(), Counter(), Counter()
    try:
        manifests = completion.get("token_audits")
        world = completion.get("world_size")
        _require(type(world) is int and world > 0, "missing world size")
        _require(isinstance(manifests, list) and len(manifests) == world, "missing rank token-audit manifests")
        _require([step for step, _ in real_attempts] == list(range(1, len(objectives) + 1)),
                 "real step/attempt mapping is incomplete")
        _require(len({attempt for _, attempt in real_attempts}) == len(real_attempts), "duplicate real attempt")
        probe_every = completion.get("probe_every")
        _require(type(probe_every) is int and probe_every > 0, "missing zero-control interval")
        zero_attempts = {attempt for step, attempt in real_attempts if step % probe_every == 0}
        _require(bool(zero_attempts), "no scheduled zero-mask control")
        runtime_identity = None
        for rank, manifest in enumerate(manifests):
            _require(manifest.get("rank") == rank, "rank manifest coverage mismatch")
            name = f"{completion['run_token']}.rank{rank}.jsonl"
            _require(Path(manifest["path"]).name == name, "token-audit run/rank path mismatch")
            path = out_dir / "token_audit" / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            _require(bool(rows) and rows[0].get("event") == "runtime", "missing generation API evidence")
            runtime = rows[0]["runtime"]
            _require(runtime.get("tokenizer") == tokenizer_evidence,
                     "runtime tokenizer evidence differs from the explicitly supplied frozen tokenizer")
            _require(runtime.get("position_policy") == "text_only_cumsum_attention_minus_one_pad_zero",
                     "unrecognized position policy")
            _require(all(isinstance(runtime.get("versions", {}).get(name), str)
                         and runtime["versions"][name] for name in ("transformers", "tokenizers", "torch")),
                     "missing generation package versions")
            for method in ("prepare_inputs_for_generation", "_prepare_position_ids_for_generation",
                           "_update_model_kwargs_for_generation"):
                source = runtime.get("generation_methods", {}).get(method, {})
                _require(isinstance(source.get("source"), str) and source["source"]
                         and source.get("sha256") == hashlib.sha256(source["source"].encode()).hexdigest(),
                         f"missing or corrupt generation method fingerprint: {method}")
            identity = (runtime["versions"], runtime["generation_methods"])
            if runtime_identity is None:
                runtime_identity = identity
            _require(identity == runtime_identity, "generation API differs across ranks")
            counts, rank_warnings, rank_routes = Counter(), Counter(), Counter()
            generations, normal_uses, zero_uses = {}, Counter(), Counter()
            coordinates = set()
            progress = {}
            for index, row in enumerate(rows):
                _require(row.get("run_token") == completion["run_token"] and row.get("rank") == rank,
                         "stale run or wrong rank in token audit")
                if index == 0:
                    continue
                attempt = row.get("attempt")
                _require(type(attempt) is int and 1 <= attempt <= attempted, "token audit has invalid attempt")
                family = row.get("family")
                _require(family in {"chat", "task", "task_neg"}, "unknown teacher route")
                gen_id = row.get("generation_id")
                if row.get("event") == "generation":
                    _require(gen_id == f"{rank}:{len(generations) + 1}" and gen_id not in generations,
                             "generation ID sequence has a duplicate or gap")
                    _require(type(row.get("final_turn")) is bool, "missing final-turn coverage")
                    _require(all(type(row.get(key)) is int and row[key] >= 0
                                 for key in ("sample_index", "turn_index")), "missing sample/turn index")
                    sample_key = (attempt, row["sample_index"])
                    coordinate = (*sample_key, row["turn_index"])
                    _require(coordinate not in coordinates, "duplicate generation sample/turn coordinate")
                    previous = progress.get(sample_key)
                    if previous is None:
                        _require(row["turn_index"] == 0, "sample history must start at turn zero")
                    else:
                        old_family, next_turn, was_final = previous
                        _require(family == old_family, "sample family changed across turns")
                        _require(not was_final, "sample generated another turn after final")
                        _require(row["turn_index"] == next_turn, "sample history has a missing or out-of-order turn")
                    coordinates.add(coordinate)
                    progress[sample_key] = (family, row["turn_index"] + 1, row["final_turn"])
                    sample = GeneratedSample.from_dict(row["sample"], mapper=mapper)
                    generations[gen_id] = (row, sample)
                    counts["generated_records"] += 1
                    counts["final_generation_records"] += int(row["final_turn"])
                    counts["warning_records"] += int(bool(sample.alignment.warnings))
                    rank_warnings.update(sample.alignment.warnings)
                    continue
                _require(row.get("event") == "kl" and gen_id in generations, "KL has no prior generation record")
                origin, sample = generations[gen_id]
                _require(origin["family"] == family and origin["attempt"] == attempt and origin["final_turn"],
                         "KL did not consume the corresponding final-turn sample/route")
                zero = row.get("zero_mask")
                _require(type(zero) is bool, "missing control identity")
                audit = row["audit"]
                active = [int(label == "text" and not zero) for label in sample.alignment.labels]
                _require(audit.get("response_ids") == list(sample.response_ids)
                         and audit.get("response_labels") == list(sample.alignment.labels)
                         and audit.get("response_mask") == active
                         and audit.get("masked_tokens") == sum(active)
                         and audit.get("reply_tokens") == len(sample.response_ids)
                         and audit.get("alignment_warnings") == list(sample.alignment.warnings),
                         "KL response identity/labels/mask differs from sampled tokens")
                execute = int(bool(sample.response_ids) and (zero or any(active)))
                _require(all(type(audit.get(call)) is int and audit[call] == execute
                             for call in ("student_forward", "aux_forward", "backward")),
                         "measured forward/backward calls do not cover the actual mask")
                for role in ("student", "aux"):
                    for key in ("input_ids", "attention_mask", "position_ids"):
                        observed = audit.get(f"{role}_{key}")
                        _require(observed == list(getattr(sample, key)) if execute else observed is None,
                                 f"{role} {key} differs from the generated input")
                loss = audit.get("loss_sum")
                _require(type(loss) in (int, float) and math.isfinite(loss), "missing/nonfinite measured KL sum")
                if not execute or zero:
                    _require(loss == 0, "zero-mask record has nonzero loss")
                prefix = "zero" if zero else "kl"
                counts[f"{prefix}_records"] += 1
                for call in ("student_forward", "aux_forward", "backward"):
                    counts[f"{prefix}_{call}"] += audit[call]
                rank_routes[family] += audit["aux_forward"]
                if zero:
                    _require(execute == 1 and audit.get("gradients", 0) > 0
                             and audit.get("zero_gradient_tensors") == audit["gradients"]
                             and audit.get("nonfinite_gradient_tensors") == 0
                             and audit.get("aux_gradient_tensors") == 0,
                             "zero control lacks measured finite zero gradients")
                    zero_uses[attempt] += 1
                else:
                    normal_uses[gen_id] += 1
                    token_totals[attempt] += sum(active)
                    loss_totals[attempt] += loss
            final_ids = {gen_id for gen_id, (row, _) in generations.items() if row["final_turn"]}
            _require(bool(generations) and bool(final_ids), "rank has no final generated samples")
            _require(normal_uses == Counter({gen_id: 1 for gen_id in final_ids}),
                     "KL must consume every final generation exactly once")
            _require(zero_uses == Counter({attempt: 1 for attempt in zero_attempts}),
                     "scheduled real zero-mask controls are not fully covered")
            expected = {"rank": rank, "path": manifest["path"], **dict(counts),
                        "warning_counts": dict(rank_warnings), "route_forwards": dict(rank_routes)}
            _require(manifest == expected, "completion token/forward/WARN counters do not match JSONL")
            result["ranks"].append(expected)
            routes.update(rank_routes)
            warnings.update(rank_warnings)
        nonzero_attempts = [attempt for attempt in range(1, attempted + 1) if token_totals[attempt] > 0]
        _require(nonzero_attempts == [attempt for _, attempt in real_attempts],
                 "raw masked-token counts do not cover every update/skip attempt")
        for (_, attempt), objective in zip(real_attempts, objectives):
            _require(objective["global_tokens"] == token_totals[attempt],
                     "objective denominator differs from raw masks")
            expected_loss = loss_totals[attempt] / token_totals[attempt]
            # The log intentionally prints .8g; allow only its rounding error.
            _require(objective["kl_per_token"] is not None
                     and math.isclose(objective["kl_per_token"], expected_loss, rel_tol=2e-7, abs_tol=1e-12),
                     "objective numerator differs from measured KL sums")
        result["ok"] = True
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        result["errors"].append(str(exc))
    result.update(route_forwards=dict(routes), warning_counts=dict(warnings),
                  masked_tokens_by_attempt=dict(token_totals))
    return result


def evaluate(log_path: Path, out_dir: Path, *, expected_real_steps: int,
             tokenizer_path: str | Path | None = None,
             mapper: ByteLevelTokenMapper | None = None) -> dict[str, Any]:
    if (tokenizer_path is None) == (mapper is None):
        raise ValueError("specify exactly one explicit tokenizer_path or frozen tokenizer mapper")
    if tokenizer_path is not None:
        from transformers import AutoTokenizer
        if not Path(tokenizer_path).is_dir():
            raise ValueError("the explicitly supplied tokenizer must be a local frozen directory")
        mapper = ByteLevelTokenMapper(AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True))
    if not isinstance(mapper, ByteLevelTokenMapper):
        raise ValueError("a real ByteLevel frozen tokenizer mapper is required")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    completion: dict[str, Any] = {}
    completion_error = ""
    try:
        completion = json.loads((out_dir / "completion.json").read_text(encoding="utf-8"))
    except Exception as exc:
        completion_error = repr(exc)
    summary = re.search(
        r"\[opd-summary\] attempted=(\d+) real=(\d+) skipped=(\d+) "
        r"target_real=(\S+) status=(\w+)", log)
    attempted = int(summary.group(1)) if summary else 0
    real = int(summary.group(2)) if summary else 0
    skipped = int(summary.group(3)) if summary else 0
    status = summary.group(5) if summary else "missing"
    masked = [int(x) for x in re.findall(r"\[opd-mask\].*?全局可蒸 token=(\d+)", log)]
    kl_chat = [float(x) for x in re.findall(r"kl_chat/tok=([-+0-9.eE]+)", log)]
    kl_task = [float(x) for x in re.findall(r"kl_task/tok=([-+0-9.eE]+)", log)]
    routes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in re.findall(
        r"\[opd-route\] chat=(\d+) task=(\d+) chat_masked=(\d+) task_masked=(\d+)",
        log,
    )]
    real_routes = [(int(total), int(chat_masked), int(task_masked))
                   for total, chat_masked, task_masked in re.findall(
        r"step \d+ .*?masked=(\d+) \[opd-route\] chat=\d+ task=\d+ "
        r"chat_masked=(\d+) task_masked=(\d+)", log
    )]
    adapter = out_dir / "final"
    prompt = re.search(
        r"\[opd-prompt\] full_menu=(\d+) answer_fields=0 "
        r"history=message_pairs hash=([0-9a-f]{16})", log,
    )
    run_token = re.search(r"\[opd-run\] token=([0-9a-f]{16})\b", log)
    fingerprints = re.findall(r"\[opd-sync\].*?完整可训练参数指纹一致 sha256=([0-9a-f]{64}) tensors=(\d+)", log)
    fingerprint = completion.get("parameter_fingerprint") or {}
    objective_rows = re.findall(r"\[opd-objective\] step (\d+) kl_per_token=(\S+) global_tokens=(\d+)", log)
    zero_controls = re.findall(r"\[opd-zero\].*?对照通过[^\n]*?student_forward=(\d+) "
                               r"aux_forward=(\d+) backward=(\d+) gradients=(\d+)", log)
    objectives = []
    for step, value, tokens in objective_rows:
        try:
            value = float(value)
        except ValueError:
            value = None
        if value is not None and not math.isfinite(value):
            value = None
        objectives.append({"step": int(step), "kl_per_token": value, "global_tokens": int(tokens)})
    real_attempts = [(int(step), int(attempt)) for step, attempt in
                     re.findall(r"\bstep (\d+) attempt=(\d+) ep\d+", log)]
    token_audit = evaluate_token_audits(out_dir, completion, attempted=attempted,
                                       real_attempts=real_attempts, objectives=objectives, mapper=mapper)
    update_audit = evaluate_update_audits(out_dir, completion, attempted=attempted, real_attempts=real_attempts)
    checks = {
        "log_exists": log_path.is_file() and bool(log.strip()),
        "vocab_identity_checked": "[opd-vocab]" in log,
        "sampling_seed_recorded": bool(re.search(
            r"\[opd-seed\] base=-?\d+ rank=\d+ effective=-?\d+", log)),
        "prompt_contract_checked": bool(prompt) and int(prompt.group(1)) > 0,
        "summary_pass": bool(summary) and status == "pass",
        "real_steps_complete": real >= expected_real_steps > 0,
        "nonzero_mask_observed": bool(masked) and any(x > 0 for x in masked),
        "chat_route_observed": any(chat > 0 for chat, _, _, _ in routes),
        "task_route_observed": any(task > 0 for _, task, _, _ in routes),
        "v15_chat_nl_observed": any(chat_masked > 0 for _, _, chat_masked, _ in routes),
        "route_mask_accounted": bool(real_routes)
        and all(total == chat_masked + task_masked
                for total, chat_masked, task_masked in real_routes),
        "kl_finite": bool(kl_chat or kl_task)
        and all(math.isfinite(x) for x in kl_chat + kl_task),
        "zero_mask_control_passed": bool(zero_controls) and all(
            (int(student), int(aux), int(backward)) == (1, 1, 1) and int(gradients) > 0
            for student, aux, backward, gradients in zero_controls),
        "raw_token_audit_complete": token_audit["ok"],
        "raw_update_audit_complete": update_audit["ok"],
        "teacher_forward_observed": token_audit["route_forwards"].get("chat", 0) > 0,
        "anchor_forward_observed": sum(token_audit["route_forwards"].get(family, 0)
                                       for family in ("task", "task_neg")) > 0,
        "attempts_accounted": attempted == real + skipped and attempted > 0
        and completion.get('attempted_steps') == attempted and completion.get('skipped_steps') == skipped,
        "rank_sync_passed": bool(fingerprints) and fingerprint.get("sha256") == fingerprints[-1][0]
        and fingerprint.get("tensors") == int(fingerprints[-1][1]) > 0,
        "global_token_objective": completion.get("objective") == "global_masked_token_mean_reverse_kl"
        and [row["step"] for row in objectives] == list(range(1, real + 1)) and real > 0
        and all(row["kl_per_token"] is not None and row["global_tokens"] > 0 for row in objectives),
        "final_adapter_exists": (adapter / "adapter_config.json").is_file()
        and (adapter / "adapter_model.safetensors").is_file(),
        "completion_marker_matches": not completion_error
        and completion.get("status") == "pass"
        and completion.get("real_steps") == real
        and real >= expected_real_steps
        and bool(run_token)
        and completion.get("run_token") == run_token.group(1)
        and bool(prompt)
        and completion.get("prompt_hash") == prompt.group(2),
        "no_traceback": "Traceback" not in log,
    }
    return {
        "ok": all(checks.values()),
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "expected_real_steps": expected_real_steps,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "checks": checks,
        "metrics": {
            "attempted_steps": attempted,
            "real_steps": real,
            "skipped_steps": skipped,
            "masked_tokens": masked,
            "kl_chat_per_token": kl_chat,
            "kl_task_per_token": kl_task,
            "routes": [{"chat": chat, "task": task, "chat_masked": chat_masked,
                        "task_masked": task_masked}
                       for chat, task, chat_masked, task_masked in routes],
            "real_route_masks": [{"total": total, "chat": chat, "task": task}
                                 for total, chat, task in real_routes],
            "summary_status": status,
            "prompt_hash": prompt.group(2) if prompt else None,
            "completion": completion,
            "completion_error": completion_error,
            "objectives": objectives,
            "parameter_fingerprint": fingerprint,
            "token_audit": token_audit,
            "update_audit": update_audit,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查 OPD 是否完成真实更新并产出有效 final adapter")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-real-steps", type=int, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True,
                    help="明确指定本轮冻结的合并底座 tokenizer；不从审计记录猜路径")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = evaluate(args.log, args.out_dir, expected_real_steps=args.expected_real_steps,
                      tokenizer_path=args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, passed in result["checks"].items():
        print(f"  {'✅' if passed else '🔴'} {name}")
    print(f"[opd-run-gate] {'PASS' if result['ok'] else 'FATAL'} -> {args.out}")
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
