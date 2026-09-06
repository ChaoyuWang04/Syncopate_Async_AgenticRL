"""检查真实 rollout 的张量接线；只返回匿名计数，不判断业务质量。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from syncopate.train.generation_audit import generation_is_incomplete


def validate_trace(trace: dict, metrics: dict) -> dict:
    if trace.get("schema_version") != 3:
        raise ValueError("RL 缺少 schema=3 的真实 token 证据")
    ids, mask, logprobs = (trace[key] for key in ("response_ids", "response_mask", "response_logprobs"))
    if not ids or not (len(ids) == len(mask) == len(logprobs)):
        raise ValueError("RL token/mask/logprob 长度不等或为空")
    if metrics.get("placeholder_logprobs") != 0:
        raise ValueError("真实模型 token 缺少采样 logprob，禁止进入更新")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in logprobs):
        raise ValueError("RL logprob 非有限")
    offset = 0
    template_tokens = model_tokens = 0
    for segment in trace["segments"]:
        count = segment["token_count"]
        expected = int(segment["type"] == "assistant")
        if (not isinstance(count, int) or count < 0 or segment["mask"] != expected or
                mask[offset:offset + count] != [expected] * count):
            raise ValueError("真实模型/环境/模板的 RL mask 接错")
        model_tokens += count * expected
        template_tokens += count if segment["type"] == "assistant_template" else 0
        offset += count
    if offset != len(ids) or model_tokens == 0:
        raise ValueError("segment 不覆盖完整 response 或没有模型 token")
    generated = 0
    for generation in trace["generations"]:
        if generation.get("finish_reason") is None and generation.get("stop_reason") is None:
            raise ValueError("缺少真实生成结束原因")
        start, kept = generation["response_offset"], generation["retained_model_tokens"]
        raw = generation["raw_token_ids"]
        if kept < 0 or kept > len(raw) or ids[start:start + kept] != raw[:kept] or mask[start:start + kept] != [1] * kept:
            raise ValueError("进入训练的模型 token 与原始采样不相等")
        generated += kept
    if generated != model_tokens:
        raise ValueError("生成轮与监督 token 数不等")
    return {"model_tokens": model_tokens, "template_tokens": template_tokens,
            "response_tokens": len(ids), "generations": len(trace["generations"])}


def summarize(root: Path, expected: int) -> dict:
    files = sorted(root.rglob("rollout.json"))
    counts = {"model_tokens": 0, "template_tokens": 0, "response_tokens": 0, "generations": 0}
    errors = []
    for index, path in enumerate(files):
        try:
            data = json.loads(path.read_text())
            result = validate_trace(data["token_trace"], data["metrics"])
            for key in counts:
                counts[key] += result[key]
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"artifact_index": index, "error": str(exc)})
    return {"health_ok": len(files) == expected and not errors, "artifacts": len(files),
            "expected_artifacts": expected, "counts": counts, "errors": errors}


def summarize_termination(run_dir: Path, *, steps_requested: int | None) -> dict:
    """从真实轨迹量结束原因；padding 宽度不是生成上限。

    目前只有固定 sync 入口能从启动配置证明 artifact 数量完整。异步包含
    预取/丢弃，必须另接消费账本，不能把 ``steps * batch * n`` 硬套进去。
    """
    expected = response_budget = None
    coverage_errors, trace_errors = [], []
    try:
        launch = json.loads((run_dir / "launch_config.json").read_text())
        overrides = {}
        for item in launch["overrides"]:
            key, separator, value = item.partition("=")
            if separator:
                overrides[key.lstrip("+")] = value.strip("'\"")
        response_budget = int(overrides["data.max_response_length"])
        if response_budget <= 0:
            raise ValueError("响应预算必须为正数")
        if overrides["trainer.v1.trainer_mode"] != "sync":
            raise ValueError("异步 artifact 覆盖口径尚未验收，不能套 sync 数量")
        steps, batch, n = (int(overrides[key]) for key in (
            "trainer.total_training_steps", "data.train_batch_size", "actor_rollout_ref.rollout.n"))
        if min(steps, batch, n) <= 0 or steps != steps_requested:
            raise ValueError("启动配置的步数或样本数与本轮声明不一致")
        expected = steps * batch * n
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        coverage_errors.append(str(exc))

    files = sorted((run_dir / "artifacts").rglob("rollout.json"))
    limits, finish_reasons = Counter(), Counter()
    token_limited = length_limited_generations = generations = max_response_tokens = 0
    for index, path in enumerate(files):
        try:
            data = json.loads(path.read_text())
            trace, metrics = data["token_trace"], data["metrics"]
            validate_trace(trace, metrics)
            reason = metrics["truncation_reason"]
            if reason not in (None, "tokens", "turns", "observation") or metrics["truncated"] is not (reason is not None):
                raise ValueError("轨迹截断标记与原因不一致")
            limited_generations = 0
            for generation in trace["generations"]:
                remaining = generation["remaining_response_budget"]
                offset = generation["response_offset"]
                dropped = generation["discarded_model_tokens"]
                if (type(remaining) is not int or remaining <= 0 or
                        type(offset) is not int or offset < 0 or type(dropped) is not int or dropped < 0):
                    raise ValueError("生成预算、位置或裁剪数无效")
                if response_budget is not None and (remaining + offset != response_budget or
                                                     len(trace["response_ids"]) > response_budget):
                    raise ValueError("轨迹预算与本轮启动配置不一致")
                if dropped != len(generation["raw_token_ids"]) - generation["retained_model_tokens"]:
                    raise ValueError("原始生成与裁剪 token 数不一致")
                limited_generations += int(generation_is_incomplete(
                    finish_reason=generation.get("finish_reason"),
                    stop_reason=generation.get("stop_reason"), discarded_tokens=dropped))
            # 以下只有整条证据通过后才累计，不把坏 artifact 的部分读数混进去。
            for generation in trace["generations"]:
                finish_reasons[str(generation.get("finish_reason"))] += 1
            if reason is not None:
                limits[reason] += 1
            token_limited += int(reason == "tokens" or limited_generations > 0)
            length_limited_generations += limited_generations
            generations += len(trace["generations"])
            max_response_tokens = max(max_response_tokens, len(trace["response_ids"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            trace_errors.append({"artifact_index": index, "error": str(exc)})

    if expected is not None and len(files) != expected:
        coverage_errors.append(f"artifact 数量 {len(files)} 不等于本轮预期 {expected}")
    complete = bool(files) and expected == len(files) and not coverage_errors and not trace_errors
    return {
        "measured": complete, "trace_valid": bool(files) and not trace_errors,
        "artifacts": len(files), "expected_artifacts": expected, "response_budget": response_budget,
        "coverage_errors": coverage_errors, "trace_errors": trace_errors,
        "trajectory_limits": dict(limits), "finish_reasons": dict(finish_reasons),
        "generations": generations, "max_response_tokens": max_response_tokens if files else None,
        "token_limited_generations": length_limited_generations,
        "token_limited_trajectories": token_limited,
        "token_limited_ratio": token_limited / len(files) if complete else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["health_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
