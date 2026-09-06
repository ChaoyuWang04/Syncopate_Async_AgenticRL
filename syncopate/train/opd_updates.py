"""Read-only OPD AdamW update observation and standard-library evidence validation.

The observer does not step, clip, zero, or otherwise change the optimizer. Full
trainable replicas only: this is a correctness probe, not a timing baseline.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import re

from syncopate.train.policy_observer import tensor_fingerprint


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _metadata(fingerprint):
    return {key: fingerprint[key] for key in ("shape", "local_shape", "dtype", "bytes", "placements")}


def _validate_fingerprint(value):
    _require(isinstance(value, dict), "missing tensor fingerprint")
    shape = value.get("shape")
    _require(isinstance(shape, list) and all(_integer(n, 1) for n in shape), "invalid tensor shape")
    _require(value.get("local_shape") == shape and value.get("placements") is None,
             "OPD update evidence requires full trainable replicas")
    element_bytes = {"torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4, "torch.float64": 8}
    dtype = value.get("dtype")
    _require(dtype in element_bytes, "unexpected/nonfloating trainable tensor dtype")
    _require(type(value.get("bytes")) is int and value["bytes"] == math.prod(shape) * element_bytes[dtype],
             "fingerprint byte count differs from shape/dtype")
    _require(isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
             "missing exact tensor content SHA256")
    _require(value.get("finite") is True, "nonfinite parameter or gradient")
    _require(_integer(value.get("nonzero")) and value["nonzero"] <= math.prod(shape),
             "invalid tensor nonzero count")


def _validate_parameters(values, *, expected=None):
    _require(isinstance(values, dict) and values and all(isinstance(name, str) and name for name in values),
             "missing trainable parameter set")
    if expected is not None:
        _require(values.keys() == expected.keys(), "trainable parameter set changed")
    for name, value in values.items():
        _validate_fingerprint(value)
        if expected is not None:
            _require(_metadata(value) == _metadata(expected[name]), f"parameter metadata changed: {name}")


def _validate_states(states, names):
    _require(isinstance(states, dict) and states.keys() == names, "missing per-parameter optimizer state")
    _require(all(step is None or _integer(step) for step in states.values()), "invalid optimizer state step")


def _claims(row):
    before, after, gradients = (row.get(name) or {} for name in
                               ("parameters_before", "parameters_after", "gradients"))
    changed = sorted(name for name in before.keys() & after.keys()
                     if before[name]["sha256"] != after[name]["sha256"])
    nonzero = sorted(name for name, value in gradients.items()
                     if value is not None and value.get("finite") is True and value.get("nonzero", 0) > 0)
    return {"changed_parameters": changed, "nonzero_gradient_parameters": nonzero,
            "signal_update": bool(set(changed) & set(nonzero))}


def _validate_transition(row, previous_parameters, previous_states):
    """Recompute from per-tensor evidence; producer counters never prove a step."""
    _require(row.get("error") is None, "optimizer observation contains an error")
    _require(type(row.get("completed_optimizer_calls")) is int and row["completed_optimizer_calls"] == 1,
             "inner optimizer must complete exactly once")
    before, after, gradients = (row.get(key) for key in ("parameters_before", "parameters_after", "gradients"))
    _validate_parameters(before, expected=previous_parameters)
    _validate_parameters(after, expected=before)
    _require(before == previous_parameters, "parameter content changed outside the observed step")
    _require(isinstance(gradients, dict) and gradients.keys() == before.keys(), "missing per-parameter gradient identity")
    states_before, states_after = row.get("state_steps_before"), row.get("state_steps_after")
    _validate_states(states_before, before.keys())
    _validate_states(states_after, before.keys())
    _require(states_before == previous_states, "optimizer state chain is discontinuous")
    for name, gradient in gradients.items():
        old, new = states_before[name], states_after[name]
        if gradient is None:
            _require(new == old, f"no-gradient optimizer state advanced: {name}")
            _require(before[name] == after[name], f"no-gradient parameter changed: {name}")
        else:
            _validate_fingerprint(gradient)
            _require(_metadata(gradient) == _metadata(before[name]), f"gradient metadata differs: {name}")
            _require(new == (old or 0) + 1, f"optimizer state did not advance exactly once: {name}")
    claims = _claims(row)
    _require(all(row.get(key) == value and type(row.get(key)) is type(value) for key, value in claims.items()),
             "update signal/content counters differ from per-tensor evidence")
    return claims


class OptimizerUpdateAudit:
    """Wrap the existing opt.step call; keep rank-exclusive evidence even on failure."""

    def __init__(self, out_dir, *, run_token, rank, named_parameters, optimizer):
        import torch

        _require(isinstance(run_token, str) and re.fullmatch(r"[0-9a-f]{16}", run_token), "invalid update run token")
        _require(_integer(rank), "invalid update rank")
        # Only the current plain AdamW/full-replica OPD path has been diagnosed.
        _require(type(optimizer) is torch.optim.AdamW, "OPD update observer requires the current plain AdamW")
        entries = [(name, param) for name, param in named_parameters if param.requires_grad]
        _require(entries and len({name for name, _ in entries}) == len(entries)
                 and len({id(param) for _, param in entries}) == len(entries), "duplicate/empty trainable identity")
        _require(not any(hasattr(param, "placements") for _, param in entries), "OPD observer cannot inspect sharded updates")
        self.parameters, self.optimizer = dict(entries), optimizer
        self._validate_membership()
        self.run_token, self.rank = run_token, rank
        self.path = Path(out_dir) / "update_audit" / f"{run_token}.rank{rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._counts = {"update_records": 0, "signal_updates": 0, "completed_optimizer_calls": 0, "error_records": 0}
        self._step = self._attempt = 0
        self._active = self._failed = False
        self._previous_parameters = self._parameters()
        self._previous_states = self._states()
        _validate_parameters(self._previous_parameters)
        _validate_states(self._previous_states, self.parameters.keys())
        header = self._base("runtime") | {"optimizer_class": "torch.optim.adamw.AdamW",
            "initial_parameters": self._previous_parameters, "initial_state_steps": self._previous_states}
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n")

    def _base(self, event):
        return {"schema_version": 1, "event": event, "run_token": self.run_token, "rank": self.rank}

    def _validate_membership(self):
        actual = [param for group in self.optimizer.param_groups for param in group["params"]]
        expected = {id(param) for param in self.parameters.values()}
        _require(len(actual) == len(expected) and {id(param) for param in actual} == expected
                 and all(param.requires_grad for param in self.parameters.values()),
                 "optimizer and observed trainable parameter identities differ")
        _require(all(id(param) in expected for param in self.optimizer.state), "optimizer has unobserved parameter state")

    def _parameters(self):
        return {name: tensor_fingerprint(param) for name, param in self.parameters.items()}

    def _states(self):
        result = {}
        for name, param in self.parameters.items():
            state = self.optimizer.state.get(param, {})
            if not state:
                result[name] = None
                continue
            _require("step" in state, f"optimizer state has no step: {name}")
            step = state["step"]
            value = step.item() if hasattr(step, "item") else step
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0 and value == int(value),
                     f"invalid/nonfinite optimizer state step: {name}")
            result[name] = int(value)
        return result

    def _append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._counts["update_records"] += 1
        self._counts["completed_optimizer_calls"] += row["completed_optimizer_calls"]
        self._counts["signal_updates"] += int(row["signal_update"] and row["error"] is None)
        self._counts["error_records"] += int(row["error"] is not None)

    @contextmanager
    def step(self, *, step, attempt):
        _require(not self._active and not self._failed, "cannot nest/reuse a failed update observation")
        _require(_integer(step, 1) and step == self._step + 1 and _integer(attempt, 1) and attempt > self._attempt,
                 "update step/attempt must be strictly sequential")
        self._active = True
        row = self._base("update") | {"step": step, "attempt": attempt, "completed_optimizer_calls": 0,
            "error": None, "parameters_before": None, "parameters_after": None, "gradients": None,
            "state_steps_before": None, "state_steps_after": None}
        handle = None
        try:
            self._validate_membership()
            row["parameters_before"] = self._parameters()
            row["state_steps_before"] = self._states()
            row["gradients"] = {name: tensor_fingerprint(param.grad) if param.grad is not None else None
                                for name, param in self.parameters.items()}
            _validate_parameters(row["parameters_before"], expected=self._previous_parameters)
            _require(row["parameters_before"] == self._previous_parameters,
                     "parameter content changed outside the observed step")
            _require(row["state_steps_before"] == self._previous_states,
                     "optimizer state chain is discontinuous")
            for gradient in row["gradients"].values():
                if gradient is not None:
                    _validate_fingerprint(gradient)

            def completed(*_):
                row["completed_optimizer_calls"] += 1

            handle = self.optimizer.register_step_post_hook(completed)
            yield
            handle.remove()
            handle = None
            self._validate_membership()
            row["parameters_after"], row["state_steps_after"] = self._parameters(), self._states()
            row.update(_claims(row))
            _validate_transition(row, self._previous_parameters, self._previous_states)
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self._failed = True
            raise
        finally:
            if handle is not None:
                handle.remove()
            if row["parameters_after"] is None or row["state_steps_after"] is None:
                try:
                    row["parameters_after"], row["state_steps_after"] = self._parameters(), self._states()
                except Exception as exc:
                    row["snapshot_error"] = f"{type(exc).__name__}: {exc}"
            row.update(_claims(row))
            if row["error"] is not None:
                row["signal_update"] = False
            self._append(row)
            self._active = False
        self._previous_parameters, self._previous_states = row["parameters_after"], row["state_steps_after"]
        self._step, self._attempt = step, attempt

    def summary(self):
        return {"path": str(self.path), "rank": self.rank, **self._counts}


def evaluate_update_audits(out_dir, completion, *, attempted, real_attempts):
    """Independently reconcile each JSONL with completion and the logged update map."""
    result = {"ok": False, "errors": [], "ranks": [], "signal_updates": 0}
    try:
        world, run_token = completion.get("world_size"), completion.get("run_token")
        _require(_integer(world, 1) and _integer(attempted, 1), "missing update rank/attempt coverage")
        _require(isinstance(run_token, str) and re.fullmatch(r"[0-9a-f]{16}", run_token), "missing update run token")
        count = len(real_attempts)
        _require(count > 0 and [step for step, _ in real_attempts] == list(range(1, count + 1))
                 and all(_integer(step, 1) and _integer(attempt, 1) and attempt <= attempted
                         for step, attempt in real_attempts)
                 and [attempt for _, attempt in real_attempts] == sorted({attempt for _, attempt in real_attempts}),
                 "incomplete/nonsequential real step/attempt mapping")
        _require(type(completion.get("real_steps")) is int and completion["real_steps"] == count
                 and type(completion.get("attempted_steps")) is int and completion["attempted_steps"] == attempted
                 and type(completion.get("skipped_steps")) is int and completion["skipped_steps"] == attempted - count,
                 "completion update/skip totals differ from actual step mapping")
        manifests = completion.get("update_audits")
        _require(isinstance(manifests, list) and len(manifests) == world, "missing rank update-audit manifests")
        all_signals = True
        rank_parameters = None
        for rank, manifest in enumerate(manifests):
            _require(type(manifest.get("rank")) is int and manifest["rank"] == rank, "update rank manifest mismatch")
            name = f"{run_token}.rank{rank}.jsonl"
            _require(Path(manifest["path"]).name == name, "update-audit run/rank path mismatch")
            path = Path(out_dir) / "update_audit" / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            _require(len(rows) == count + 1 and rows[0].get("event") == "runtime", "update record count/header mismatch")
            for row in rows:
                _require(type(row.get("schema_version")) is int and row["schema_version"] == 1
                         and row.get("run_token") == run_token and type(row.get("rank")) is int and row["rank"] == rank,
                         "stale run or wrong rank/schema in update audit")
            runtime = rows[0]
            _require(runtime.get("optimizer_class") == "torch.optim.adamw.AdamW", "unsupported optimizer evidence")
            previous_parameters, previous_states = runtime.get("initial_parameters"), runtime.get("initial_state_steps")
            _validate_parameters(previous_parameters)
            _validate_states(previous_states, previous_parameters.keys())
            metadata = {key: _metadata(value) for key, value in previous_parameters.items()}
            if rank_parameters is None:
                rank_parameters = metadata
            _require(metadata == rank_parameters, "trainable parameter metadata differs across ranks")
            totals = {"update_records": 0, "signal_updates": 0, "completed_optimizer_calls": 0, "error_records": 0}
            for row, (step, attempt) in zip(rows[1:], real_attempts):
                _require(row.get("event") == "update" and type(row.get("step")) is int and row["step"] == step
                         and type(row.get("attempt")) is int and row["attempt"] == attempt,
                         "update record differs from logged step/attempt")
                claims = _validate_transition(row, previous_parameters, previous_states)
                totals["update_records"] += 1
                totals["completed_optimizer_calls"] += row["completed_optimizer_calls"]
                totals["signal_updates"] += int(claims["signal_update"])
                previous_parameters, previous_states = row["parameters_after"], row["state_steps_after"]
            _require(all(type(manifest.get(key)) is int and manifest[key] == value for key, value in totals.items()),
                     "completion update counters differ from raw per-tensor evidence")
            result["ranks"].append({"rank": rank, **totals})
            result["signal_updates"] += totals["signal_updates"]
            all_signals &= totals["signal_updates"] == count
        _require(all_signals, "each registered update must change a parameter with a finite nonzero target gradient")
        result["ok"] = True
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
        result["errors"].append(str(exc))
    return result
