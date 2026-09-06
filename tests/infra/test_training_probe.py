"""B03 tiny full-attention Text MoE: independent loss, gradients and restart."""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def probe():
    try:
        return importlib.import_module("syncopate.infra.training_probe")
    except ModuleNotFoundError as exc:
        if exc.name == "syncopate.infra.training_probe":
            pytest.fail("B03 training probe has not been implemented")
        raise


def test_import_and_help_are_torch_and_cuda_free():
    program = """
import builtins, runpy, sys
old_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'verl', 'tensordict'}:
        raise AssertionError('heavy import: ' + name)
    return old_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['training_probe', '--help']
runpy.run_module('syncopate.infra.training_probe', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout and "--output" in result.stdout


def _non_tensor_calls():
    root = Path(__file__).resolve().parents[2]
    paths = (root / "syncopate/infra/training_probe.py", Path(__file__).resolve())
    return [pytest.param(path.name, call, id=f"{path.name}:{call.lineno}:{call.func.attr}")
            for path in paths for call in ast.walk(ast.parse(path.read_text()))
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "tu"]


@pytest.mark.parametrize("filename,call", _non_tensor_calls())
def test_non_tensor_calls_bind_target_verl_signatures_without_torch(filename, call):
    # Target verl 0.9 tensordict_utils.py SHA256:
    # 4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1.
    # Bind actual call expressions even when local torch tests must skip.
    parameter = inspect.Parameter
    positional = parameter.POSITIONAL_OR_KEYWORD
    signatures = {
        "get_non_tensor_data": inspect.Signature([
            parameter(name, positional) for name in ("data", "key", "default")]),
        "assign_non_tensor": inspect.Signature([
            parameter("tensor_dict", positional), parameter("kwargs", parameter.VAR_KEYWORD)]),
    }
    assert call.func.attr in signatures, f"unverified TensorDict API: {call.func.attr}"
    assert not any(isinstance(arg, ast.Starred) for arg in call.args)
    assert all(keyword.arg is not None for keyword in call.keywords)
    try:
        bound = signatures[call.func.attr].bind(
            *call.args, **{keyword.arg: keyword.value for keyword in call.keywords})
    except TypeError as exc:
        pytest.fail(f"{filename}:{call.lineno}: target verl signature mismatch: {exc}")
    if call.func.attr == "get_non_tensor_data":
        # Missing arm/denominator metadata must fail the existing assertions,
        # not silently inherit the expected value and manufacture a PASS.
        default = bound.arguments["default"]
        assert isinstance(default, ast.Constant) and default.value is None


def test_scope_and_thresholds_are_preregistered_not_hybrid(probe):
    plan = probe.preregistered_plan()
    assert plan["world_size"] == 2
    assert plan["arms"] == [False, True]
    assert plan["micro_batch_size_per_gpu"] == 1
    assert plan["gradient_relative_l2_max"] == 1e-4
    assert plan["gradient_atol"] == 1e-6
    assert plan["gradient_rtol"] == 1e-4
    assert plan["restart_equality"] == "exact"
    assert plan["hybrid_gdn_measured"] is False
    assert plan["lora_measured"] is False
    assert plan["performance_acceptance"] is False
    assert plan["worker_timeout_seconds"] < 600
    assert plan["distributed_timeout_seconds"] < 600
    assert plan["stop_grace_seconds"] == 105
    assert plan["experts_implementation"] == "eager"


def test_tiny_is_text_root_and_only_full_attention_but_real_moe(probe):
    kwargs = probe.tiny_config_kwargs()
    assert kwargs["model_type"] == "qwen3_5_moe_text"
    assert kwargs["architectures"] == ["Qwen3_5MoeForCausalLM"]
    assert kwargs["layer_types"] == ["full_attention", "full_attention"]
    assert kwargs["num_experts"] == 4 and kwargs["num_experts_per_tok"] == 2
    assert kwargs["rope_parameters"]["mrope_section"] == [2, 1, 1]
    assert kwargs["attention_dropout"] == 0


def test_two_distinct_rank_batches_have_unequal_token_denominators(probe):
    rows = probe.synthetic_batch(0)
    checked = probe.validate_batch(rows)
    assert checked["rows"] == 4
    assert len({tuple(r["input_ids"]) for r in rows}) == 4
    assert len({len(r["input_ids"]) for r in rows}) == 4
    counts = [sum(r["loss_mask"]) for r in rows]
    assert len(set(counts)) == 4
    assert counts[0] + counts[1] != counts[2] + counts[3]
    assert all(row["loss_mask"][-1] == 0 for row in rows)
    assert probe.synthetic_batch(1) != rows
    assert probe.batch_digest(rows) == probe.batch_digest(copy.deepcopy(rows))


@pytest.mark.parametrize("mutation", ["last_token", "wrong_mask_length", "bad_mask", "bad_id", "positions", "no_tokens"])
def test_mask_or_sequence_contract_violations_fail_closed(probe, mutation):
    rows = probe.synthetic_batch(0)
    if mutation == "last_token":
        rows[0]["loss_mask"][-1] = 1
    elif mutation == "wrong_mask_length":
        rows[0]["loss_mask"].pop()
    elif mutation == "bad_mask":
        rows[0]["loss_mask"][0] = 0.5
    elif mutation == "bad_id":
        rows[0]["input_ids"][0] = 128
    elif mutation == "positions":
        rows[0]["position_ids"][1] = 99
    else:
        for row in rows:
            row["loss_mask"] = [0] * len(row["input_ids"])
    with pytest.raises(ValueError):
        probe.validate_batch(rows)


def test_rank_average_recovers_global_token_mean_not_micro_mean(probe):
    # Four unequal micro denominators: a mean-of-means gives the wrong gradient.
    numerators, counts = [3.0, 20.0, 12.0, 35.0], [3, 5, 4, 7]
    scale = probe.loss_scale(sum(counts), 2)
    rank_losses = [(numerators[0] + numerators[1]) * scale,
                   (numerators[2] + numerators[3]) * scale]
    assert sum(rank_losses) / 2 == pytest.approx(sum(numerators) / sum(counts))
    assert sum(rank_losses) / 2 != pytest.approx(sum(n / c for n, c in zip(numerators, counts)) / 4)


@pytest.mark.parametrize("tokens,ranks", [(0, 2), (-1, 2), (math.nan, 2), (1, 0), (1, 1.5)])
def test_loss_scale_rejects_invalid_global_denominator(probe, tokens, ranks):
    with pytest.raises(ValueError):
        probe.loss_scale(tokens, ranks)


@pytest.mark.parametrize("updates,expected", [({}, True), ({"finite": False}, False),
    ({"same_shape": False}, False), ({"error_l2": 0.001}, False),
    ({"max_abs_error": 0.001}, False), ({"error_l2": math.nan}, False),
    ({"reference_l2": math.inf}, False), ({"reference_l2": 0.0, "error_l2": 0.0}, True),
    ({"reference_l2": 0.0, "error_l2": 1e-10}, False)])
def test_gradient_check_rejects_nonfinite_wrong_shape_and_wrong_math(probe, updates, expected):
    args = dict(error_l2=1e-6, reference_l2=1.0, max_abs_error=1e-7,
                max_abs_reference=1.0, finite=True, same_shape=True)
    args.update(updates)
    result = probe.gradient_verdict(**args)
    assert result["ok"] is expected
    json.dumps(result, allow_nan=False)


def test_gpu_gate_requires_exact_successful_cpu_artifact_and_initial_weights(probe, tmp_path):
    output = tmp_path / "gpu"
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.require_cpu_evidence(output)
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    evidence = {"mode": "cpu", "health_ok": True, "interfaces": {"nccl_compiled": True},
                "source_sha256": probe.source_sha256(), "plan": probe.preregistered_plan(),
                "tiny_model": {"path": str(cpu / "tiny_model"), "files": {}}}
    (cpu / "result.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        probe.require_cpu_evidence(output)


def test_installed_engine_config_is_fp32_static_full_training(probe):
    pytest.importorskip("torch")
    pytest.importorskip("verl")
    configs = probe.build_engine_configs("/not-created", False, construct_model=False)
    engine = configs["engine_config"]
    assert engine.strategy == "fsdp2" and engine.fsdp_size == 2
    assert engine.model_dtype == "fp32"
    assert engine.mixed_precision == {"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}
    assert not engine.use_dynamic_bsz and engine.micro_batch_size_per_gpu == 1
    assert not engine.use_torch_compile and not engine.forward_only
    assert not engine.offload_policy and not engine.param_offload and not engine.optimizer_offload
    assert configs["optimizer_config"].override_optimizer_config == {"eps": 1e-8, "foreach": False, "fused": False}
    assert configs["checkpoint_config"].save_contents == ["model", "optimizer", "extra"]


def test_nested_batch_preserves_masks_positions_and_non_tensor_control(probe):
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    from verl.utils import tensordict_utils as tu
    rows = probe.synthetic_batch(0)[:2]
    td = probe.make_batch(rows, use_remove_padding=True)
    assert td.batch_size == torch.Size([2])
    assert td["input_ids"].is_nested and td["loss_mask"].is_nested
    for field in ("input_ids", "position_ids", "loss_mask"):
        assert [v.tolist() for v in td[field].unbind()] == [r[field] for r in rows]
    for key, value in {"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 1,
                       "use_remove_padding": True, "use_fused_kernels": False}.items():
        assert tu.get_non_tensor_data(td, key, default=None) == value


def test_independent_reference_uses_global_token_loss_and_gradients(probe):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = probe.build_tiny_model()
    rows = probe.synthetic_batch(0)
    model.zero_grad(set_to_none=True)
    result = probe.independent_reference(model, rows)
    result["loss"].backward()
    total = sum((-lp * torch.tensor(row["loss_mask"])).sum()
                for lp, row in zip(result["log_probs"], rows))
    assert result["loss"].item() == pytest.approx((total / probe.validate_batch(rows)["tokens"]).item())
    grads = probe.capture_gradients(model)
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert sum(g.square().sum().item() for g in grads.values()) > 0
    assert not any("linear_attn" in name for name, _ in model.named_modules())
    assert model.config._experts_implementation == "eager"


def test_exact_tree_check_detects_optimizer_step_moment_scheduler_and_rng_mutation(probe):
    torch = pytest.importorskip("torch")
    snapshot = {"model": {"weight": torch.tensor([1., 2.])},
                "optimizer": {"weight": {"exp_avg": torch.tensor([.1, .2]), "step": torch.tensor(1.)}},
                "scheduler": {"last_epoch": 1}, "rng": {"cpu": torch.tensor([1, 2], dtype=torch.uint8)}}
    assert probe.compare_exact_trees(snapshot, copy.deepcopy(snapshot))["ok"]
    for section, key in [("model", "weight"), ("optimizer", "weight"), ("scheduler", "last_epoch"), ("rng", "cpu")]:
        other = copy.deepcopy(snapshot)
        if section == "optimizer":
            other[section][key]["step"] += 1
        else:
            other[section][key] += 1
        check = probe.compare_exact_trees(snapshot, other)
        assert not check["ok"] and check["mismatches"]


def test_real_torch_gradient_comparison_detects_rank_local_and_scaled_gradients(probe):
    torch = pytest.importorskip("torch")
    reference = {"a": torch.tensor([1., 2.]), "b": torch.tensor([0.])}
    assert probe.compare_gradients(reference, copy.deepcopy(reference))["ok"]
    assert not probe.compare_gradients(reference, {"a": torch.tensor([.5, 1.]), "b": torch.tensor([0.])})["ok"]
    assert not probe.compare_gradients(reference, {"a": torch.tensor([math.nan, 2.]), "b": torch.tensor([0.])})["ok"]
    assert not probe.compare_gradients(reference, {"a": torch.tensor([1., 2.])})["ok"]


def test_restart_tree_missing_keys_and_tensor_dtype_are_not_equal(probe):
    torch = pytest.importorskip("torch")
    assert not probe.compare_exact_trees({"a": torch.ones(2)}, {"b": torch.ones(2)})["ok"]
    assert not probe.compare_exact_trees(torch.ones(2), torch.ones(2, dtype=torch.float64))["ok"]


def test_exact_tree_detects_signed_zero_in_numpy_and_scalar_without_torch(probe, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(Tensor=type('UnusedTensor', (), {})))
    assert not probe.compare_exact_trees(0.0, -0.0)['ok']
    assert not probe.compare_exact_trees(np.array([0.0]), np.array([-0.0]))['ok']
    assert probe.compare_exact_trees(np.array([0.0]), np.array([0.0]))['ok']
    assert not probe.compare_exact_trees(np.zeros((1, 2)), np.zeros((2, 1)))['ok']


def test_exact_tree_detects_signed_zero_tensor(probe):
    torch = pytest.importorskip('torch')
    assert not probe.compare_exact_trees(torch.tensor(0.0), torch.tensor(-0.0))['ok']
    assert probe.compare_exact_trees(torch.tensor(0.0), torch.tensor(0.0))['ok']


def test_failed_arm_keeps_observed_root_before_removing_hooks(probe, monkeypatch, tmp_path):
    class Observation:
        closed = False
        def __init__(self, engine): pass
        def result(self):
            assert not self.closed
            return {'actual_root_calls': 1, 'first_actual_root': {'kwargs': ['input_ids']}, 'ok': False}
        def close(self): self.closed = True
    monkeypatch.setattr(probe, 'RootObservation', Observation)
    with pytest.raises(RuntimeError, match='numerical failure'):
        with probe.observed_root(None, tmp_path / 'root.json') as observation:
            raise RuntimeError('numerical failure')
    assert observation.closed
    assert json.loads((tmp_path / 'root.json').read_text())['actual_root_calls'] == 1


def test_initial_and_save_failures_write_evidence_before_collective_require(probe):
    import inspect
    source = inspect.getsource(probe._run_arm)
    assert source.index('"initial_weights.json"') < source.index('_collective_require(initial_check')
    assert source.index('"state_after_save.pt"') < source.index('_collective_require(save_stability')
    assert source.index('"save_stability.json"') < source.index('_collective_require(save_stability')
    assert source.index('"fresh_initial_weights.json"') < source.index('_collective_require(fresh_initial_check')


def test_launcher_is_bounded_two_rank_module_without_training_entry(probe, tmp_path):
    command = probe.launcher_command(tmp_path / "gpu")
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in command and "--max-restarts=0" in command
    assert "syncopate.infra.training_probe" in command and "--worker" in command
    assert not any("v16_pipeline" in token or "verl.trainer" in token for token in command)


def test_cpu_lazy_init_guard_refuses_cuda_and_restores_original(probe):
    torch = pytest.importorskip("torch")
    original = torch.cuda._lazy_init
    with probe.cpu_only_guard():
        with pytest.raises(RuntimeError, match="CPU preflight"):
            torch.cuda._lazy_init()
    assert torch.cuda._lazy_init is original


def test_cpu_refusal_and_existing_attempt_preserve_evidence(probe, tmp_path, monkeypatch):
    output = tmp_path / "cpu"
    def unavailable(_output):
        raise RuntimeError("missing installed interface")
    monkeypatch.setattr(probe, "run_cpu_probe", unavailable)
    assert probe.main(["--mode", "cpu", "--output", str(output)]) == 1
    original = (output / "result.json").read_bytes()
    result = json.loads(original)
    assert result["health_ok"] is False
    assert result["error"]["message"] == "missing installed interface"
    assert probe.main(["--mode", "cpu", "--output", str(output)]) == 2
    assert (output / "result.json").read_bytes() == original


def test_engine_step_exercises_real_root_before_capture_and_clip(probe):
    # Contract wiring check supplements, but never substitutes for target execution.
    import inspect
    source = inspect.getsource(probe._engine_step)
    markers = ["engine.train_mode(zero_grad_on_exit=False)", "engine.optimizer_zero_grad()",
               "engine.forward_backward_batch(", "capture_gradients(engine.module)",
               "engine.optimizer_step()", "engine.lr_scheduler_step()"]
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "engine.module.forward(" not in source
    assert "loss_scale(actual_tokens, actual_dp_size)" in source


def test_reference_reconstruction_precedes_loading_rng_and_checkpoint_save_is_collective(probe):
    import inspect
    source = inspect.getsource(probe._run_arm)
    assert source.index("resumed_reference, resumed_optimizer") < source.index("fresh.load_checkpoint(")
    assert "del_local_after_load=False" in source
    assert '_collective_require(not checkpoint_dir.exists()' in source


def test_optimizer_capture_records_named_finite_moments_step_and_groups(probe):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, foreach=False, fused=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = probe._capture_optimizer(model, optimizer)
    assert state["states"].keys() == dict(model.named_parameters()).keys()
    for item in state["states"].values():
        assert item["step"].item() == 1
        assert torch.isfinite(item["exp_avg"]).all()
        assert torch.isfinite(item["exp_avg_sq"]).all()
    assert state["param_groups"][0]["params"] == ["weight", "bias"]
    assert probe.compare_exact_trees(state, copy.deepcopy(state))["ok"]


def test_model_and_optimizer_oracles_have_explicit_identical_inputs(probe):
    import inspect
    source = inspect.getsource(probe._engine_step)
    assert 'reference_model.load_state_dict(before)' in source
    assert source.index('reference_model.load_state_dict(before)') < source.index('independent_reference(reference_model, rows)')
    assert 'same_state_optimizer_reference(before, optimizer_before, actual_gradients' in source
    assert 'independent_gradient_sidecheck' in source
    assert 'expected_before or engine_before' not in source
    assert probe.preregistered_plan()['optimizer_reference'] == 'CPU_AdamW_same_state_same_actual_gradient'


@pytest.mark.parametrize('step', [1, 2])
def test_same_state_optimizer_control_uses_nonzero_moments_without_aliasing(probe, step):
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0, foreach=False, fused=False)
    if step == 2:
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[1., 2., 3.]])).sum().backward()
    before = probe._capture_model(model)
    states = probe._capture_optimizer(model, optimizer)
    gradients = probe.capture_gradients(model)
    original = copy.deepcopy((before, states, gradients))
    control = probe.same_state_optimizer_reference(before, states, gradients, step=step)
    assert probe.compare_exact_trees(original, (before, states, gradients))['ok']
    optimizer.step()
    assert probe.compare_exact_trees(control['parameters'], probe._capture_model(model))['ok']
    assert probe.compare_exact_trees(control['optimizer'], probe._capture_optimizer(model, optimizer))['ok']
    assert control['input_check']['ok'] and control['inputs_unchanged']['ok']


@pytest.mark.parametrize('wrong', ['step', 'epsilon', 'grad_shape', 'nan', 'missing_state', 'extra_param'])
def test_same_state_optimizer_reference_refuses_changed_contract(probe, wrong):
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0, foreach=False, fused=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    states = probe._capture_optimizer(model, optimizer)
    parameters, gradients = probe._capture_model(model), probe.capture_gradients(model)
    if wrong == 'step': states['states']['weight']['step'].add_(1)
    elif wrong == 'epsilon': states['param_groups'][0]['eps'] = 1e-6
    elif wrong == 'grad_shape': gradients['weight'] = torch.zeros(1)
    elif wrong == 'nan': gradients['weight'][0, 0] = float('nan')
    elif wrong == 'missing_state': states['states'].pop('weight')
    else: states['param_groups'][0]['params'].append('not_in_model')
    with pytest.raises((ValueError, AssertionError)):
        probe.same_state_optimizer_reference(parameters, states, gradients, step=2)


def test_parameter_comparison_reports_actual_per_coordinate_bound(probe):
    torch = pytest.importorskip('torch')
    reference = {'p': torch.tensor([.001, 100.])}
    actual = {'p': torch.tensor([.00101, 100.])}
    result = probe._compare_parameters(reference, actual)
    assert not result['ok']
    assert result['per_parameter']['p']['max_error_ratio'] > 1
    assert 'max_abs_bound' not in result['per_parameter']['p']


@pytest.mark.parametrize(('value', 'expected'), [
    ('023ce55b-1c20-d3ee-0f44-1425a5f12b85', True),
    ('GPU-023ce55b-1c20-d3ee-0f44-1425a5f12b85', True),
    ('gpu-023ce55b-1c20-d3ee-0f44-1425a5f12b85', False),
    ('GPU-0', False),
    ('023ce55b-1c20-d3ee-0f44-1425a5f12b8', False),
    ('not-a-uuid', False),
    ('', False),
    (None, False),
])
def test_b200_uuid_accepts_torch_and_nvidia_smi_forms(probe, value, expected):
    assert probe.valid_gpu_uuid(value) is expected


@pytest.mark.parametrize('mutation', ['none', 'reset_moments', 'wrong_step', 'swap_names', 'wrong_group'])
def test_optimizer_moment_transition_detects_history_and_mapping_errors(probe, mutation):
    torch = pytest.importorskip('torch')
    model = torch.nn.ParameterDict({'a': torch.nn.Parameter(torch.tensor([.5])),
                                    'b': torch.nn.Parameter(torch.tensor([.2]))})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0, foreach=False, fused=False)
    model['a'].grad, model['b'].grad = torch.tensor([.2]), torch.tensor([.7])
    optimizer.step()
    model['a'].grad, model['b'].grad = torch.zeros(1), torch.zeros(1)
    before = probe._capture_optimizer(model, optimizer)
    gradients = probe.capture_gradients(model)
    control = probe.same_state_optimizer_reference(probe._capture_model(model), before, gradients, step=2)
    optimizer.step()
    actual = probe._capture_optimizer(model, optimizer)
    if mutation == 'reset_moments': actual['states']['a']['exp_avg'].zero_()
    elif mutation == 'wrong_step': actual['states']['a']['step'].add_(1)
    elif mutation == 'swap_names': actual['states']['a'], actual['states']['b'] = actual['states']['b'], actual['states']['a']
    elif mutation == 'wrong_group': actual['param_groups'][0]['eps'] = 1e-6
    assert probe.compare_optimizer_transition(before, gradients, control['optimizer'], actual)['ok'] is (mutation == 'none')
    assert abs(control['parameters']['a'].item() - .49832994) < 1e-7


def test_target_cpu_preflight_really_loads_saved_text_model_without_cuda(probe, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("verl")
    output = tmp_path / "cpu"
    output.mkdir()
    result = probe.run_cpu_probe(output)
    assert result["health_ok"] is True
    assert result["interfaces"]["nccl_compiled"] is True
    assert result["interfaces"]["cuda_initialized"] is False
    assert result["interfaces"]["hf_model_loader"]["source_sha256"]
    assert result["interfaces"]["hf_experts_config_setter"]["source_sha256"]
    assert result["cpu_reference"]["disk_reload_exact"]["ok"] is True
    assert result["cpu_reference"]["gradient_formula"]["ok"] is True
    assert not torch.cuda.is_initialized()
    assert (output / "cpu_reference.pt").is_file()
    assert (output / "tiny_model" / "config.json").is_file()


@pytest.mark.parametrize("before,after,expected", [
    ({"last_epoch": 0}, {"last_epoch": 1}, True),
    ({"last_epoch": 1}, {"last_epoch": 2}, True),
    ({"last_epoch": 1}, {"last_epoch": 1}, False),
    ({"last_epoch": 1}, {"last_epoch": 3}, False),
    ({}, {}, False),
])
def test_scheduler_must_actually_advance_one_step(probe, before, after, expected):
    assert probe.scheduler_advanced(before, after) is expected


def healthy_rank_results(probe):
    exact = {'ok': True, 'equality': 'exact', 'mismatches': []}
    metric = probe.gradient_verdict(error_l2=0., reference_l2=1., max_abs_error=0.,
                                   max_abs_reference=1., finite=True)
    parameter_metric = {key: value for key, value in metric.items() if key != 'max_abs_bound'}
    parameter_metric.update(max_error_ratio=0., bad_coordinates=0)
    rank_check = {'ok': True, 'dp_members': [0, 1], 'sha256_by_rank': ['a' * 64, 'a' * 64]}
    ranks = []
    for rank in (0, 1):
        arms = []
        for padding in (False, True):
            steps = []
            for step in (1, 2):
                callbacks = [{'row_id': row['row_id'], 'global_batch_num_tokens': 19, 'dp_size': 2,
                    'local_supervised_tokens': sum(row['loss_mask']), 'logprob_ok': True, 'loss_ok': True,
                    'loss': 1., 'reference_scaled_micro_loss': 1., 'logprob_max_abs_error': 0.,
                    'logprob_max_abs_reference': 1., 'logprob_max_error_ratio': 0.}
                    for row in probe.synthetic_batch(step - 1)[rank * 2:rank * 2 + 2]]
                steps.append({'ok': True, 'step': step, 'gradient_check': {
                    'ok': True, 'global': copy.deepcopy(metric), 'missing': [], 'extra': [],
                    'per_parameter': {'p': copy.deepcopy(metric)}}, 'gradient_nonzero': True,
                    'callback_checks_ok': True, 'callbacks': callbacks, 'reference_loss': 1.,
                    'global_batch_sha256': probe.batch_digest(probe.synthetic_batch(step - 1)),
                    'parameter_check': {'ok': True, 'atol': probe.PARAMETER_ATOL, 'rtol': probe.PARAMETER_RTOL,
                        'missing': [], 'extra': [], 'per_parameter': {'p': copy.deepcopy(parameter_metric)}},
                    'independent_gradient_sidecheck': {'ok': True, 'atol': probe.PARAMETER_ATOL, 'rtol': probe.PARAMETER_RTOL,
                        'missing': [], 'extra': [], 'per_parameter': {'p': copy.deepcopy(parameter_metric)}},
                    'optimizer_check': {'ok': True, 'same_names': True, 'metadata_exact': copy.deepcopy(exact),
                        'error_scale': probe.preregistered_plan()['moment_error_scale'],
                        'moments': {f'p/{key}': copy.deepcopy(parameter_metric) for key in ('exp_avg', 'exp_avg_sq')}},
                    **{key: copy.deepcopy(exact) for key in ('same_start_exact', 'previous_step_continuity',
                        'reference_preserves_actual', 'same_gradient_input', 'same_gradient_inputs_unchanged',
                        'independent_gradient_input', 'independent_gradient_inputs_unchanged')},
                    'rank_before': copy.deepcopy(rank_check), 'rank_after': copy.deepcopy(rank_check),
                    'gradient_norm_before_clip': 1., 'clip_threshold': probe.CLIP_GRAD, 'clipping_inactive': True,
                    'parameter_update_l2': .1, 'parameter_update_nonzero': True, 'optimizer_state_ok': True,
                    'optimizer_steps': {'p': step}, 'learning_rate': .001,
                    'scheduler': {'last_epoch': step}, 'scheduler_advanced_once': True})
            first_row = probe.synthetic_batch(0)[rank * 2]
            root = {'ok': True, 'actual_root_calls': 4, 'first_actual_root': {
                'class_mro': ['Qwen3_5MoeForCausalLM'], 'experts_implementation': 'eager',
                'attention_implementation': 'eager', 'kwargs': ['input_ids', 'position_ids'],
                'forward': {'callable': 'test.native.forward', 'source_sha256': '1' * 64},
                'input_ids': [first_row['input_ids']], 'position_ids': [first_row['position_ids']],
                'pass_packed_cu_seqlens': False,
                'boundary_arguments': {'seq_idx': None, 'cu_seq_lens_q': None, 'cu_seqlens': None}},
                'first_root_routers': [{'module': f'layer_{i}', 'finite': True,
                    'router_ids': [[0, 1] for _ in first_row['input_ids']],
                    'router_weights': [[.5, .5] for _ in first_row['input_ids']]} for i in (0, 1)],
                'gdn_modules': [], 'gdn_calls': 0, 'gdn_measured': False}
            restored_root = copy.deepcopy(root)
            restored_root['actual_root_calls'] = 2
            restored_root['first_actual_root']['input_ids'] = [probe.synthetic_batch(1)[rank * 2]['input_ids']]
            arms.append({'ok': True, 'use_remove_padding': padding, 'baseline_steps': steps,
                'restored_next_step': copy.deepcopy(steps[1]), 'root_observation': root,
                'restored_root_observation': restored_root, 'checkpoint_files_preserved': True,
                'checkpoint_manifest': {f'{kind}_world_size_2_rank_{r}.pt': '0' * 64
                    for kind in ('model', 'optim', 'extra_state') for r in (0, 1)},
                **{key: copy.deepcopy(exact) for key in ('initial_weights', 'fresh_initial_weights',
                    'save_state_unchanged', 'load_state_exact', 'rng_draws_exact', 'next_step_exact')}})
        ranks.append({'mode': 'gpu_worker', 'rank': rank, 'health_ok': True,
            'source_sha256': probe.source_sha256(), 'plan': probe.preregistered_plan(), 'arms': arms,
            'device': {'name': 'NVIDIA B200',
                       'uuid': f'GPU-00000000-0000-0000-0000-00000000000{rank}',
                       'capability': [10, 0]},
            'determinism': {'deterministic_algorithms': True, 'tf32_matmul': False,
                            'cublas_workspace_config': ':4096:8'}})
    return ranks


@pytest.mark.parametrize('mutation', ['none', 'source', 'mode', 'string_true', 'no_arms', 'arm_failed',
    'step_missing', 'wrong_tokens', 'bad_gradient', 'bad_parameter', 'wrong_optimizer', 'bad_restore',
    'missing_root', 'wrong_gpu', 'same_uuid', 'missing_checkpoint', 'logprob_error', 'lr',
    'wrong_parameter_keys', 'checkpoint_disagree', 'missing_router', 'missing_callable', 'wrong_root_tokens',
    'coordinate_ratio', 'bad_coordinate_count', 'moment_ratio', 'moment_missing', 'reference_mutation',
    'no_continuity', 'rank_disagree', 'wrong_dp_group', 'cross_report_rank_disagree', 'moment_nan', 'missing_sidecheck'])
def test_real_gpu_aggregator_rejects_contradictory_or_incomplete_rank_evidence(probe, tmp_path, monkeypatch, mutation):
    import syncopate.pipeline.process_guard as guard
    ranks = healthy_rank_results(probe)
    arm = ranks[0]['arms'][0]
    if mutation == 'source': ranks[0]['source_sha256'] = 'wrong'
    elif mutation == 'mode': ranks[0]['mode'] = 'cpu'
    elif mutation == 'string_true': ranks[0]['health_ok'] = 'false'
    elif mutation == 'no_arms': ranks[0]['arms'] = []
    elif mutation == 'arm_failed': arm['ok'] = False
    elif mutation == 'step_missing': arm['baseline_steps'].pop()
    elif mutation == 'wrong_tokens': arm['baseline_steps'][0]['callbacks'][0]['global_batch_num_tokens'] = 8
    elif mutation == 'bad_gradient': arm['baseline_steps'][0]['gradient_check']['per_parameter']['p']['max_abs_error'] = 1.
    elif mutation == 'bad_parameter': arm['baseline_steps'][0]['parameter_check']['per_parameter']['p']['finite'] = False
    elif mutation == 'wrong_optimizer': arm['baseline_steps'][0]['optimizer_steps']['p'] = 0
    elif mutation == 'bad_restore': arm['load_state_exact']['mismatches'] = ['state/model']
    elif mutation == 'missing_root': arm['root_observation']['actual_root_calls'] = 0
    elif mutation == 'wrong_gpu': ranks[0]['device']['capability'] = [9, 0]
    elif mutation == 'same_uuid': ranks[1]['device']['uuid'] = ranks[0]['device']['uuid']
    elif mutation == 'missing_checkpoint': arm['checkpoint_manifest'].pop('optim_world_size_2_rank_0.pt')
    elif mutation == 'logprob_error': arm['baseline_steps'][0]['callbacks'][0]['logprob_max_abs_error'] = 1e6
    elif mutation == 'lr': arm['baseline_steps'][0]['learning_rate'] = 1000
    elif mutation == 'wrong_parameter_keys':
        params = arm['baseline_steps'][0]['parameter_check']['per_parameter']
        params['other'] = params.pop('p')
    elif mutation == 'checkpoint_disagree': ranks[1]['arms'][0]['checkpoint_manifest']['model_world_size_2_rank_0.pt'] = 'f' * 64
    elif mutation == 'missing_router': arm['root_observation']['first_root_routers'][0].pop('router_ids')
    elif mutation == 'missing_callable': arm['root_observation']['first_actual_root'].pop('forward')
    elif mutation == 'wrong_root_tokens': arm['root_observation']['first_actual_root']['input_ids'][0][0] += 1
    elif mutation == 'coordinate_ratio': arm['baseline_steps'][0]['parameter_check']['per_parameter']['p']['max_error_ratio'] = 2.
    elif mutation == 'bad_coordinate_count': arm['baseline_steps'][0]['parameter_check']['per_parameter']['p']['bad_coordinates'] = 1
    elif mutation == 'moment_ratio': arm['baseline_steps'][0]['optimizer_check']['moments']['p/exp_avg']['max_error_ratio'] = 2.
    elif mutation == 'moment_missing': arm['baseline_steps'][0]['optimizer_check']['moments'].pop('p/exp_avg_sq')
    elif mutation == 'reference_mutation': arm['baseline_steps'][0]['reference_preserves_actual']['mismatches'] = ['state/optimizer']
    elif mutation == 'no_continuity': arm['baseline_steps'][1].pop('previous_step_continuity')
    elif mutation == 'rank_disagree': arm['baseline_steps'][0]['rank_before']['sha256_by_rank'][1] = 'b' * 64
    elif mutation == 'wrong_dp_group': arm['baseline_steps'][0]['rank_before']['dp_members'] = [0]
    elif mutation == 'cross_report_rank_disagree': ranks[1]['arms'][0]['baseline_steps'][0]['rank_before']['sha256_by_rank'] = ['b' * 64] * 2
    elif mutation == 'moment_nan': arm['baseline_steps'][0]['optimizer_check']['moments']['p/exp_avg']['error_l2'] = float('nan')
    elif mutation == 'missing_sidecheck': arm['baseline_steps'][0].pop('independent_gradient_sidecheck')
    monkeypatch.setattr(probe, 'require_cpu_evidence', lambda output: {
        'interfaces': {'native_text_root': {'callable': 'test.native.forward', 'source_sha256': '1' * 64}}})
    monkeypatch.setattr(guard, 'run_guarded', lambda *a, **kw: {
        'rc': 0, 'process_returncode': 0, 'timed_out': False})
    for rank in ranks:
        (tmp_path / f"rank_{rank['rank']}.json").write_text(json.dumps(rank))
    result = probe.run_gpu_probe(tmp_path)
    assert result['health_ok'] is (mutation == 'none')
