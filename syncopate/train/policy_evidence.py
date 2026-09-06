"""核对 B03 身份观察记录；不把 logprob 差的未知原因判成通过。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def validate_batch_trace(tensors: dict, row: int, trace: dict) -> int:
    """回读真实 trainer 张量，与原始生成记录对拍；概率先转成存储 dtype。"""
    import torch

    prompt, response, mask = (tensors[name][row] for name in ("prompts", "responses", "response_mask"))
    for name, actual, values in (("prompt", prompt, trace["prompt_ids"]),
                                  ("response", response, trace["response_ids"]),
                                  ("mask", mask, trace["response_mask"])):
        expected = torch.tensor(values, dtype=actual.dtype)
        if actual.ndim != 1 or not torch.equal(actual.cpu(), expected):
            raise ValueError(f"trainer {name} 与原始 rollout 不同")
    if not torch.equal(tensors["input_ids"][row], torch.cat([prompt, response])):
        raise ValueError("trainer input_ids 不是原始 prompt + response")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.sum()):
        raise ValueError("trainer mask 非 0/1 或没有模型 token")
    rollout = tensors["rollout_log_probs"][row]
    if not torch.equal(rollout.cpu(), torch.tensor(trace["response_logprobs"], dtype=rollout.dtype)):
        raise ValueError("trainer rollout_log_probs 与原始采样概率不同")
    for name in ("old_log_probs", "rollout_log_probs", "repeat_log_probs"):
        values = tensors[name][row]
        if values.shape != mask.shape or not bool(torch.isfinite(values[mask.bool()]).all()):
            raise ValueError(f"trainer {name} 的有效 token 形状或有限性错误")
    return int(mask.sum())


def validate_records(records: list[dict], *, steps: int, ranks: int) -> dict:
    errors = []
    groups = {kind: [r for r in records if r.get("kind") == kind]
              for kind in ("optimizer", "weight_source", "weight_receiver", "logprob")}

    def require(value, message):
        if not value: errors.append(message)

    require(steps > 0 and ranks > 0, "未登记更新数或 rank 数")
    updates = groups["optimizer"]
    require(Counter((r.get("rank"), r.get("attempt")) for r in updates)
            == Counter((rank, step) for rank in range(ranks) for step in range(1, steps + 1)),
            "optimizer 更新覆盖不完整或重复")
    for record in updates:
        label = f"rank={record.get('rank')}/attempt={record.get('attempt')}"
        before, after = record.get("parameters_before", {}), record.get("parameters_after", {})
        require(record.get("completed_optimizer_calls") == 1 and record.get("error") is None,
                f"{label} 没有一次成功的内层 optimizer 调用")
        require(bool(before) and before.keys() == after.keys(), f"{label} 可训练参数集合改变或为空")
        changed = sum(before[name].get("sha256") != after.get(name, {}).get("sha256") for name in before)
        require(changed > 0 and record.get("changed_parameters") == changed, f"{label} 没有可核实的权重变化")
        require(all(t.get("finite") is True for t in after.values()), f"{label} 权重非有限")
        gradients = record.get("gradients_before_clip", {})
        require(bool(gradients) and all(g.get("finite") is True for g in gradients.values()),
                f"{label} 梯度缺失或非有限")
        norm = record.get("reported_grad_norm")
        require(isinstance(norm, (float, int)) and math.isfinite(norm) and norm > 0,
                f"{label} 没有有限非零全局梯度")
        state = record.get("state_steps_after", {})
        require(bool(state) and all(float(key) == record.get("attempt") for key in state)
                and sum(state.values()) == len(before), f"{label} optimizer 保存状态没有完整前进")

    sources, receivers = groups["weight_source"], groups["weight_receiver"]
    wanted = Counter((version, rank) for version in range(steps + 1) for rank in range(ranks))
    require(Counter((r.get("global_steps"), r.get("rank")) for r in sources) == wanted,
            "同步源的策略版本/rank 覆盖不完整或重复")
    require(Counter((r.get("global_steps"), r.get("replica_rank")) for r in receivers) == wanted,
            "同步接收端的策略版本/replica 覆盖不完整或重复")
    previous = None
    for version in range(steps + 1):
        sent = [r for r in sources if r.get("global_steps") == version]
        received = [r for r in receivers if r.get("global_steps") == version]
        expected = sent[0].get("tensors", {}) if sent else {}
        if previous is not None:
            require(expected != previous, f"策略 {version} 的同步载荷没有随更新改变")
        previous = expected
        require(bool(expected) and all(t.get("finite") is True and t.get("placements") is None for t in expected.values()),
                f"策略 {version} 缺有限的完整同步张量")
        for record in sent:
            require(record.get("stream_complete") is True and record.get("tensors") == expected,
                    f"策略 {version} 的各 rank 完整 LoRA 载荷不同或未读完")
        for record in received:
            require(record.get("error") is None and record.get("tensors") == expected,
                    f"策略 {version} 接收字节与发送端不同")
            calls = record.get("add_lora_calls", [])
            require(len(calls) == 1 and calls[0].get("accepted") is True and calls[0].get("slot_present") is True
                    and calls[0].get("tensors") == expected, f"策略 {version} 没有实际接收正确 LoRA")

    probabilities = groups["logprob"]
    require(Counter(r.get("step") for r in probabilities) == Counter(range(1, steps + 1)),
            "logprob 观察没有覆盖全部更新")
    for record in probabilities:
        step = record.get("step")
        tags = record.get("tags", [])
        require(bool(tags) and len(tags) == len(record.get("keys", []))
                and all(t.get("min_global_steps") == step - 1 and t.get("max_global_steps") == step - 1 for t in tags),
                f"step {step} 不是本轮 sync 策略的轨迹")
        for key in ("trainer_rollout", "trainer_repeat"):
            values = record.get(key, {})
            require(values.get("tokens", 0) > 0
                    and all(isinstance(values.get(k), (int, float)) and math.isfinite(values[k])
                            for k in ("mean_signed", "mean_abs", "max_abs", "p99_abs")),
                    f"step {step} 的 {key} 不可测")
    return {"health_ok": not errors, "errors": errors, "counts": {k: len(v) for k, v in groups.items()},
            "logprob": [{k: r[k] for k in ("step", "trainer_rollout", "trainer_repeat")} for r in probabilities],
            "not_proven": ["跨 rank 梯度相对独立参考的正确性", "两引擎逐 token MoE 路由",
                           "checkpoint 恢复", "logprob 差已处于可接受噪声范围", "性能基线"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--run-id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.run_id:
        from syncopate.pipeline.cloud_execution import validate_run_id
        from syncopate.pipeline.split import DATA_VERSION
        validate_run_id(args.run_id)
        args.run_dir = Path.cwd() / "checkpoints/grpo" / f"{DATA_VERSION}_smoke_{args.run_id}"
    launch = json.loads((args.run_dir / "launch_config.json").read_text())
    arguments = launch["arguments"]
    if arguments["mode"] != "sync" or arguments["strategy"] != "fsdp2":
        raise ValueError("当前记录汇总只适用于 sync/FSDP2")
    records = [json.loads(path.read_text()) for path in sorted((args.run_dir / "policy_evidence").glob("*.json"))]
    result = validate_records(records, steps=arguments["steps"], ranks=arguments["gpus"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["health_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
