"""OPD update evidence: light record checks plus independent real CPU AdamW checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest


RUN = "0123456789abcdef"


def updates():
    assert importlib.util.find_spec("syncopate.train.opd_updates") is not None, (
        "OPD has no independent optimizer-update observer or evidence validator")
    return importlib.import_module("syncopate.train.opd_updates")


def fingerprint(values):
    """Small, explicit float32 bytes for parser tests; not numerical AdamW evidence."""
    raw = struct.pack(f"<{len(values)}f", *values)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "shape": [len(values)],
            "local_shape": [len(values)], "dtype": "torch.float32", "bytes": len(raw),
            "placements": None, "finite": True, "nonzero": sum(x != 0 for x in values)}


def light_records(*, rank=0, pairs=((1, 1), (2, 3))):
    before = {"weight": fingerprint([1., 2.]), "unused": fingerprint([3.])}
    states = {name: None for name in before}
    rows = [{"schema_version": 1, "event": "runtime", "rank": rank, "run_token": RUN,
             "optimizer_class": "torch.optim.adamw.AdamW", "initial_parameters": deepcopy(before),
             "initial_state_steps": dict(states)}]
    for step, attempt in pairs:
        after = {"weight": fingerprint([1. - step * .1, 2. - step * .1]),
                 "unused": deepcopy(before["unused"])}
        next_states = {"weight": step, "unused": None}
        rows.append({"schema_version": 1, "event": "update", "rank": rank, "run_token": RUN,
            "step": step, "attempt": attempt, "parameters_before": deepcopy(before),
            "parameters_after": deepcopy(after), "gradients": {"weight": fingerprint([1., 1.]), "unused": None},
            "state_steps_before": dict(states), "state_steps_after": dict(next_states),
            "completed_optimizer_calls": 1, "error": None, "changed_parameters": ["weight"],
            "nonzero_gradient_parameters": ["weight"], "signal_update": True})
        before, states = after, next_states
    return rows


def write_records(out, rows):
    rank = rows[0]["rank"]
    path = Path(out) / "update_audit" / f"{RUN}.rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    return {"path": str(path), "rank": rank, "update_records": len(rows) - 1,
            "signal_updates": sum(row.get("signal_update", False) for row in rows[1:]),
            "completed_optimizer_calls": sum(row["completed_optimizer_calls"] for row in rows[1:]),
            "error_records": sum(row["error"] is not None for row in rows[1:])}


def evaluate_rows(tmp_path, rows, *, pairs=((1, 1), (2, 3)), attempted=3):
    manifest = write_records(tmp_path, rows)
    completion = {"run_token": RUN, "world_size": 1, "real_steps": len(pairs),
                  "attempted_steps": attempted, "skipped_steps": attempted - len(pairs),
                  "update_audits": [manifest]}
    return updates().evaluate_update_audits(tmp_path, completion, attempted=attempted,
                                           real_attempts=list(pairs))


def test_two_update_records_with_a_skipped_attempt_are_accepted(tmp_path):
    result = evaluate_rows(tmp_path, light_records())
    assert result["ok"], result
    assert result["signal_updates"] == 2


@pytest.mark.parametrize("fault", [
    "same_weight", "zero_grad_decay", "nan_grad", "nan_after", "missing_parameter", "wrong_dtype",
    "wrong_shape", "wrong_bytes", "missing_state", "state_not_advanced", "state_plus_two",
    "no_grad_state_advanced", "no_grad_weight_changed", "missing_hook", "double_hook", "error",
    "counter_fake", "signal_fake", "broken_parameter_chain", "broken_state_chain", "wrong_step",
    "wrong_attempt", "duplicate_update", "stale_run", "wrong_rank", "bad_sha", "bad_nonzero",
    "hidden_missing_grad", "initial_mismatch",
])
def test_record_gate_recomputes_update_claims(tmp_path, fault):
    rows = light_records()
    row = rows[-1]
    if fault == "same_weight": row["parameters_after"] = deepcopy(row["parameters_before"])
    elif fault == "zero_grad_decay": row["gradients"]["weight"] = fingerprint([0., 0.])
    elif fault == "nan_grad": row["gradients"]["weight"]["finite"] = False
    elif fault == "nan_after": row["parameters_after"]["weight"]["finite"] = False
    elif fault == "missing_parameter": row["parameters_after"].pop("unused")
    elif fault == "wrong_dtype": row["parameters_after"]["weight"]["dtype"] = "torch.float64"
    elif fault == "wrong_shape": row["gradients"]["weight"]["shape"] = [1, 2]
    elif fault == "wrong_bytes": row["parameters_after"]["weight"]["bytes"] = 4
    elif fault == "missing_state": row["state_steps_after"].pop("weight")
    elif fault == "state_not_advanced": row["state_steps_after"]["weight"] = 1
    elif fault == "state_plus_two": row["state_steps_after"]["weight"] = 3
    elif fault == "no_grad_state_advanced": row["state_steps_after"]["unused"] = 1
    elif fault == "no_grad_weight_changed": row["parameters_after"]["unused"] = fingerprint([2.])
    elif fault == "missing_hook": row["completed_optimizer_calls"] = 0
    elif fault == "double_hook": row["completed_optimizer_calls"] = 2
    elif fault == "error": row["error"] = "RuntimeError: deliberate"
    elif fault == "counter_fake": row["changed_parameters"] = ["weight", "unused"]
    elif fault == "signal_fake": row["signal_update"] = False
    elif fault == "broken_parameter_chain": row["parameters_before"]["weight"] = fingerprint([2., 1.])
    elif fault == "broken_state_chain": row["state_steps_before"]["weight"] = 8
    elif fault == "wrong_step": row["step"] = 3
    elif fault == "wrong_attempt": row["attempt"] = 2
    elif fault == "duplicate_update": rows.append(deepcopy(row))
    elif fault == "stale_run": row["run_token"] = "a" * 16
    elif fault == "wrong_rank": row["rank"] = 1
    elif fault == "bad_sha": row["parameters_after"]["weight"]["sha256"] = "unknown"
    elif fault == "bad_nonzero": row["gradients"]["weight"]["nonzero"] = 3
    elif fault == "hidden_missing_grad": row["gradients"].pop("unused")
    elif fault == "initial_mismatch": rows[0]["initial_parameters"]["weight"] = fingerprint([4., 5.])
    result = evaluate_rows(tmp_path, rows)
    assert not result["ok"], result
    assert result["errors"]


@pytest.mark.parametrize("fault", ["missing_manifest", "missing_rank", "manifest_count", "manifest_signal",
    "manifest_calls", "wrong_path", "wrong_completed", "bad_mapping", "out_of_order_attempts"])
def test_rank_manifest_and_completion_cannot_substitute_for_records(tmp_path, fault):
    rows = light_records()
    manifest = write_records(tmp_path, rows)
    completion = {"run_token": RUN, "world_size": 1, "real_steps": 2, "attempted_steps": 3,
                  "skipped_steps": 1, "update_audits": [manifest]}
    pairs = [(1, 1), (2, 3)]
    if fault == "missing_manifest": completion.pop("update_audits")
    elif fault == "missing_rank": completion["world_size"] = 2
    elif fault == "manifest_count": manifest["update_records"] = 7
    elif fault == "manifest_signal": manifest["signal_updates"] = 7
    elif fault == "manifest_calls": manifest["completed_optimizer_calls"] = 7
    elif fault == "wrong_path": manifest["path"] = "some-other-run.rank0.jsonl"
    elif fault == "wrong_completed": completion["real_steps"] = 7
    elif fault == "bad_mapping": pairs = [(1, 1), (1, 3)]
    elif fault == "out_of_order_attempts": pairs = [(1, 3), (2, 1)]
    result = updates().evaluate_update_audits(tmp_path, completion, attempted=3, real_attempts=pairs)
    assert not result["ok"], result


def test_zero_signal_record_is_diagnostic_not_healthy(tmp_path):
    rows = light_records(pairs=((1, 1),))
    rows[1].update(gradients={"weight": fingerprint([0., 0.]), "unused": None},
                   nonzero_gradient_parameters=[], signal_update=False)
    result = evaluate_rows(tmp_path, rows, pairs=((1, 1),), attempted=1)
    assert not result["ok"] and result["signal_updates"] == 0
    assert result["ranks"][0]["update_records"] == 1


def test_parameter_change_and_target_gradient_must_overlap(tmp_path):
    rows = light_records(pairs=((1, 1),))
    rows[1]["gradients"]["weight"] = fingerprint([0., 0.])
    rows[1]["gradients"]["unused"] = fingerprint([1.])
    rows[1]["state_steps_after"]["unused"] = 1
    rows[1]["nonzero_gradient_parameters"] = ["unused"]
    rows[1]["signal_update"] = False
    assert not evaluate_rows(tmp_path, rows, pairs=((1, 1),), attempted=1)["ok"]


def test_all_rank_updates_need_individual_coverage(tmp_path):
    manifests = [write_records(tmp_path, light_records(rank=rank)) for rank in (0, 1)]
    completion = {"run_token": RUN, "world_size": 2, "real_steps": 2, "attempted_steps": 3,
                  "skipped_steps": 1, "update_audits": manifests}
    result = updates().evaluate_update_audits(tmp_path, completion, attempted=3, real_attempts=[(1, 1), (2, 3)])
    assert result["ok"] and result["signal_updates"] == 4


def real_setup(tmp_path, *, lr=.1, decay=.01):
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    observer = updates().OptimizerUpdateAudit(tmp_path, run_token=RUN, rank=0,
        named_parameters=model.named_parameters(), optimizer=opt)
    return torch, model, opt, observer


def observed_rows(observer):
    return [json.loads(line) for line in Path(observer.summary()["path"]).read_text().splitlines()]


def test_real_cpu_two_adamw_steps_preserve_math_and_record_state(tmp_path):
    torch, model, opt, observer = real_setup(tmp_path)
    reference = deepcopy(model)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=.1, weight_decay=.01)
    for step, attempt in ((1, 1), (2, 3)):
        for item, optimizer in ((model, opt), (reference, ref_opt)):
            optimizer.zero_grad(set_to_none=True)
            item(torch.tensor([[1., 2.]])).square().mean().backward()
        with observer.step(step=step, attempt=attempt):
            opt.step()
        ref_opt.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rows = observed_rows(observer)
    assert [row["state_steps_after"]["weight"] for row in rows[1:]] == [1, 2]
    assert all(row["completed_optimizer_calls"] == 1 for row in rows[1:])
    assert not opt._optimizer_step_post_hooks
    assert evaluate_rows(tmp_path, rows)["ok"]


def test_real_cpu_none_grad_does_not_advance_state_or_weight(tmp_path):
    torch, model, opt, observer = real_setup(tmp_path)
    model(torch.ones(1, 2)).sum().backward()
    model.bias.grad = None
    before = model.bias.detach().clone()
    with observer.step(step=1, attempt=1):
        opt.step()
    row = observed_rows(observer)[1]
    assert row["gradients"]["bias"] is None and row["state_steps_after"]["bias"] is None
    assert torch.equal(before, model.bias)
    assert evaluate_rows(tmp_path, observed_rows(observer), pairs=((1, 1),), attempted=1)["ok"]


@pytest.mark.parametrize("fault", ["noop", "noop_method", "exception", "nan_grad", "nan_parameter", "counterfake"])
def test_real_cpu_failed_updates_leave_evidence_and_raise(tmp_path, fault, monkeypatch):
    torch, model, opt, observer = real_setup(tmp_path)
    model(torch.ones(1, 2)).sum().backward()
    if fault == "nan_grad": model.weight.grad.fill_(float("nan"))
    if fault == "nan_parameter":
        with torch.no_grad(): model.weight.fill_(float("nan"))
    if fault == "noop_method": monkeypatch.setattr(opt, "step", lambda: None)
    if fault == "counterfake":
        original = opt.step
        def fake_counter():
            original()
            for state in opt.state.values(): state["step"].add_(1)
        monkeypatch.setattr(opt, "step", fake_counter)
    with pytest.raises((ValueError, RuntimeError)):
        with observer.step(step=1, attempt=1):
            if fault == "exception": raise RuntimeError("deliberate")
            if fault != "noop": opt.step()
    row = observed_rows(observer)[1]
    assert row["error"]
    assert not opt._optimizer_step_post_hooks
    assert not evaluate_rows(tmp_path, observed_rows(observer), pairs=((1, 1),), attempted=1)["ok"]


def test_real_zero_target_grad_with_adamw_decay_is_not_signal(tmp_path):
    torch, model, opt, observer = real_setup(tmp_path, decay=.1)
    model(torch.ones(1, 2)).mul(0).sum().backward()
    before = model.weight.detach().clone()
    with observer.step(step=1, attempt=1):
        opt.step()
    row = observed_rows(observer)[1]
    assert not torch.equal(before, model.weight), "negative control must actually apply weight decay"
    assert row["changed_parameters"] and row["nonzero_gradient_parameters"] == []
    assert row["signal_update"] is False
    assert not evaluate_rows(tmp_path, observed_rows(observer), pairs=((1, 1),), attempted=1)["ok"]


def test_real_none_grad_keeps_an_existing_optimizer_state_step(tmp_path):
    torch, model, opt, observer = real_setup(tmp_path)
    for step in (1, 2):
        opt.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        if step == 2: model.bias.grad = None
        with observer.step(step=step, attempt=step):
            opt.step()
    rows = observed_rows(observer)
    assert rows[2]["state_steps_before"]["bias"] == rows[2]["state_steps_after"]["bias"] == 1
    assert evaluate_rows(tmp_path, rows, pairs=((1, 1), (2, 2)), attempted=2)["ok"]


@pytest.mark.parametrize("fault", ["no_grad", "nonzero_grad"])
def test_real_zero_lr_cannot_claim_signal_even_when_state_advances(tmp_path, fault):
    torch, model, opt, observer = real_setup(tmp_path, lr=0)
    if fault == "nonzero_grad": model(torch.ones(1, 2)).sum().backward()
    with observer.step(step=1, attempt=1):
        opt.step()
    row = observed_rows(observer)[1]
    assert row["completed_optimizer_calls"] == 1
    assert row["signal_update"] is False
    assert not evaluate_rows(tmp_path, observed_rows(observer), pairs=((1, 1),), attempted=1)["ok"]


@pytest.mark.parametrize("fault", ["parameter_changed", "state_changed", "membership_changed"])
def test_real_out_of_band_change_is_rejected_before_optimizer_runs(tmp_path, fault):
    torch, model, opt, observer = real_setup(tmp_path)
    model(torch.ones(1, 2)).sum().backward()
    with observer.step(step=1, attempt=1): opt.step()
    model(torch.ones(1, 2)).sum().backward()
    if fault == "parameter_changed":
        with torch.no_grad(): model.weight.add_(1)
    elif fault == "state_changed": opt.state[model.weight]["step"].add_(1)
    else: opt.param_groups[0]["params"] = [model.weight]
    with pytest.raises(ValueError):
        with observer.step(step=2, attempt=2):
            pytest.fail("must reject out-of-band identity/state changes before the optimizer call")
    assert observed_rows(observer)[-1]["completed_optimizer_calls"] == 0


def test_real_nonzero_grad_without_parameter_change_is_not_signal(tmp_path):
    torch, model, opt, observer = real_setup(tmp_path, lr=0)
    model(torch.ones(1, 2)).sum().backward()
    with observer.step(step=1, attempt=1):
        opt.step()
    row = observed_rows(observer)[1]
    assert row["nonzero_gradient_parameters"] and row["changed_parameters"] == []
    assert row["signal_update"] is False


def test_real_equal_norm_different_bytes_are_distinguished():
    torch = pytest.importorskip("torch")
    from syncopate.train.policy_observer import tensor_fingerprint
    left, right = torch.tensor([1., 2.]), torch.tensor([2., 1.])
    assert left.norm() == right.norm()
    assert tensor_fingerprint(left)["sha256"] != tensor_fingerprint(right)["sha256"]


def test_real_observer_file_is_rank_exclusive(tmp_path):
    _, model, opt, _ = real_setup(tmp_path)
    with pytest.raises(FileExistsError):
        updates().OptimizerUpdateAudit(tmp_path, run_token=RUN, rank=0,
            named_parameters=model.named_parameters(), optimizer=opt)


def test_opd_production_update_uses_the_observer_and_completion_index():
    import ast
    source = Path("syncopate/train/opd.py").read_text()
    tree = ast.parse(source)
    observed = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                and any(isinstance(item.context_expr, ast.Call)
                        and ast.unparse(item.context_expr.func) == "update_audit.step" for item in node.items)]
    assert len(observed) == 1, "real opt.step is currently unobserved"
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == "opt.step"
               for node in ast.walk(observed[0]))
    assert '"update_audits": update_summaries' in source
