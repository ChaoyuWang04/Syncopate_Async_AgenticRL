"""B13 的字节口径、数值拒绝、CPU 真计算与证据防覆盖。"""
from __future__ import annotations

import importlib
import copy
import json
import math
import subprocess
import sys

import pytest


@pytest.fixture
def probe():
    try:
        return importlib.import_module("syncopate.infra.hardware_probe")
    except ModuleNotFoundError as exc:
        if exc.name in {"syncopate.infra", "syncopate.infra.hardware_probe"}:
            pytest.fail("B13 hardware probe has not been implemented")
        raise


@pytest.mark.parametrize("operation,inputs,outputs,algorithm", [
    ("all_reduce", 4, 4, 16),
    ("all_gather_into_tensor", 4, 8, 32),
    ("reduce_scatter_tensor", 8, 4, 32),
    ("send_recv_0_to_1", 4, 4, 16),
    ("send_recv_1_to_0", 4, 4, 16),
])
def test_size_means_per_rank_block_and_algorithm_uses_total_array(
    probe, operation, inputs, outputs, algorithm,
):
    layout = probe.collective_layout(operation, 16)
    assert layout["input_numel"] == inputs
    assert layout["output_numel"] == outputs
    assert layout["algorithm_bytes"] == algorithm
    assert layout["block_alignment_mod_16"] == 0


def test_predeclared_messages_straddle_alignment_without_rounding(probe):
    assert probe.MESSAGE_BYTES == (12, 16, 20, 1 << 20, 16 << 20, 64 << 20)
    for nbytes in probe.MESSAGE_BYTES:
        layout = probe.collective_layout("all_gather_into_tensor", nbytes)
        assert layout["input_numel"] * 4 == nbytes
        assert layout["block_alignment_mod_16"] == nbytes % 16


@pytest.mark.parametrize("operation,nbytes,world", [
    ("unknown", 16, 2), ("all_reduce", 0, 2),
    ("all_reduce", 15, 2), ("all_reduce", 16, 1),
])
def test_invalid_layout_is_rejected(probe, operation, nbytes, world):
    with pytest.raises(ValueError):
        probe.collective_layout(operation, nbytes, world_size=world)


@pytest.mark.parametrize("operation,algorithm,bus", [
    ("all_reduce", 1.0, 1.5),
    ("all_gather_into_tensor", 4.0, 3.0),
    ("reduce_scatter_tensor", 4.0, 3.0),
    ("send_recv_0_to_1", 1.0, 1.0),
])
def test_bandwidth_uses_decimal_gbps_and_nccl_total_array_factors(
    probe, operation, algorithm, bus,
):
    result = probe.bandwidth_gbps(operation, 1_000_000, 0.001, world_size=4)
    assert result["algorithm_GB_s"] == pytest.approx(algorithm)
    assert result["bus_GB_s"] == pytest.approx(bus)


@pytest.mark.parametrize("elapsed", [0, -1, math.inf, math.nan])
def test_bad_timing_cannot_produce_a_bandwidth(probe, elapsed):
    with pytest.raises(ValueError):
        probe.bandwidth_gbps("all_reduce", 16, elapsed)


def test_gemm_uses_two_multiply_add_flops(probe):
    assert probe.gemm_tflops(1024, 2048, 4096, 0.002) == pytest.approx(8.589934592)


@pytest.mark.parametrize("error,reference,finite,shape,expected", [
    (0.02, 1.0, True, True, True),
    (0.021, 1.0, True, True, False),
    (0.0, 1.0, False, True, False),
    (0.0, 1.0, True, False, False),
    (math.nan, 1.0, True, True, False),
    (0.0, math.inf, True, True, False),
    (0.0, 0.0, True, True, True),
    (1.0, 0.0, True, True, False),
])
def test_numeric_check_rejects_nonfinite_wrong_shape_and_incorrect_values(
    probe, error, reference, finite, shape, expected,
):
    result = probe.comparison_metrics(error, reference, finite=finite, same_shape=shape)
    assert result["ok"] is expected
    # Failed numerical evidence must also be valid strict JSON, not NaN/Infinity.
    json.dumps(result, allow_nan=False)


def test_window_aggregation_keeps_both_ranks_and_the_slowest_rank(probe):
    ranks = [
        {"rank": 0, "windows": [{"wall_seconds": [1.0, 2.0], "cuda_seconds": [0.9, 1.9]}]},
        {"rank": 1, "windows": [{"wall_seconds": [2.0, 4.0], "cuda_seconds": [1.9, 3.9]}]},
    ]
    result = probe.aggregate_timings(ranks)
    window = result["windows"][0]
    assert window["rank_elapsed_seconds"] == {"0": 3.0, "1": 6.0}
    assert window["max_rank_elapsed_seconds"] == 6.0
    assert window["seconds_per_call"] == 3.0
    assert window["max_rank_sample_seconds"] == [2.0, 4.0]
    assert window["sample_dispersion"]["median"] == 3.0
    assert window["sample_dispersion"]["sample_stdev"] == pytest.approx(math.sqrt(2))
    assert result["independent_host_samples"] == 1
    assert result["sample_unit"] == "warmed_windows_on_one_allocation"


def test_aggregation_rejects_missing_or_mismatched_ranks(probe):
    with pytest.raises(ValueError):
        probe.aggregate_timings([])
    with pytest.raises(ValueError):
        probe.aggregate_timings([
            {"rank": 0, "windows": [{"wall_seconds": [1.0], "cuda_seconds": [1.0]}]},
            {"rank": 1, "windows": []},
        ])


def test_existing_output_is_never_overwritten_even_after_failure(probe, tmp_path):
    output = tmp_path / "run" / "gpu"
    probe.prepare_output_dir(output)
    probe.write_json_once(output / "result.json", {"health_ok": False})
    original = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.prepare_output_dir(output)
    with pytest.raises(FileExistsError):
        probe.write_json_once(output / "result.json", {"health_ok": True})
    assert (output / "result.json").read_bytes() == original


def test_module_import_and_help_do_not_import_torch(probe):
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('module import touched torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from syncopate.infra.hardware_probe import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout


def test_launcher_is_exactly_two_local_ranks_without_elastic_retries(probe, tmp_path):
    argv = probe.launcher_command(tmp_path)
    assert argv[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in argv
    assert "--nnodes=1" in argv and "--nproc-per-node=2" in argv
    assert "--max-restarts=0" in argv
    assert "--worker" in argv and "--output" in argv
    assert 0 < probe.DISTRIBUTED_TIMEOUT_SECONDS < probe.WORKER_TIMEOUT_SECONDS < 600


def test_cpu_tensor_shapes_and_values_have_independent_known_answers(probe):
    torch = pytest.importorskip("torch")
    expected = {
        "all_reduce": [[16, 18, 20, 22], [16, 18, 20, 22]],
        "all_gather_into_tensor": [[-8, -7, -6, -5, 24, 25, 26, 27]] * 2,
        "reduce_scatter_tensor": [[16, 18, 20, 22], [48, 50, 52, 54]],
    }
    for operation, answers in expected.items():
        for rank in range(2):
            case = probe.collective_tensors(torch, operation, 16, rank, device="cpu")
            assert case["expected"].tolist() == answers[rank]
            assert case["input"].dtype == torch.float32


def test_cpu_formula_checks_use_real_torch_without_any_cuda_calls(probe, monkeypatch):
    torch = pytest.importorskip("torch")

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU formula check touched CUDA")

    for name in ("init", "is_available", "device_count", "set_device", "synchronize", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    result = probe.cpu_formula_checks(torch)
    assert result["health_ok"] is True
    assert result["gemm"]["forward"]["ok"] is True
    assert result["gemm"]["grad_a"]["ok"] is True
    assert result["gemm"]["grad_b"]["ok"] is True
    assert len(result["collectives"]) == 15
    assert all(case["ok"] for case in result["collectives"])


def test_exact_tensor_check_catches_nan_and_permutation(probe):
    torch = pytest.importorskip("torch")
    expected = torch.tensor([1.0, 2.0])
    assert probe.exact_tensor_check(expected, expected)["ok"] is True
    assert probe.exact_tensor_check(expected.flip(0), expected)["ok"] is False
    assert probe.exact_tensor_check(torch.tensor([math.nan, 2.0]), expected)["ok"] is False


def test_cpu_cli_records_installed_interfaces_and_real_math(probe, tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "cpu"
    rc = probe.main(["--mode", "cpu", "--output", str(output)])
    result = json.loads((output / "result.json").read_text())
    assert result["health_ok"] is result["interfaces"]["nccl_compiled"]
    assert rc == (0 if result["health_ok"] else 1)
    assert result["formula_checks"]["health_ok"] is True
    assert result["interfaces"]["torchrun_args"]["nproc_per_node"] == "2"
    assert "all_gather_into_tensor" in result["interfaces"]["signatures"]
    assert result["capabilities"]["cuda_queried"] is False


def test_incomplete_gpu_records_cannot_report_health(probe):
    result = probe.aggregate_gpu_results([], {"rc": 0, "timed_out": False})
    assert result["health_ok"] is False
    assert result["errors"]
    result = probe.aggregate_gpu_results([
        {"rank": 0, "health_ok": True}, {"rank": 1, "health_ok": True},
    ], {"rc": 0, "timed_out": False})
    assert result["health_ok"] is False
    assert result["errors"]


def test_chrome_trace_lists_device_kernels_without_guessing_tensor_core_utilization(probe):
    names = probe.kernel_names_from_trace({"traceEvents": [
        {"cat": "cpu_op", "name": "aten::mm"},
        {"cat": "kernel", "name": "synthetic_gemm_kernel"},
        {"cat": "kernel", "name": "ncclDevKernel_AllReduce"},
        {"cat": "kernel", "name": "synthetic_gemm_kernel"},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel"},
    ]})
    assert names == ["ncclDevKernel_AllReduce", "synthetic_gemm_kernel"]
    assert probe.kernel_names_from_trace({"traceEvents": []}) == []


def _complete_rank_records(probe):
    def timed(rank, postcheck):
        return {"rank": rank, "windows": [
            {"window": index, "wall_seconds": [0.001] * 10, "cuda_seconds": [0.0009] * 10} for index in range(3)
        ], "postchecks": [dict(postcheck) for _ in range(3)]}

    def exact(op, size):
        length = probe.collective_layout(op, size)['output_numel']
        return {'ok': True, 'finite': True, 'same_shape': True, 'mismatched_elements': 0,
                'actual_shape': [length], 'expected_shape': [length]}

    return [{
        "rank": rank, "health_ok": True, "process_group_destroyed": True,
        "gemm_correctness": {"ok": True, **{name: probe.comparison_metrics(0.001, 1.0, finite=True)
                                          for name in ("forward", "grad_a", "grad_b")}},
        "capabilities": {"rank": rank, "name": "NVIDIA B200", "compute_capability": [10, 0],
                         "uuid": f"GPU-{rank}", "uuid_measured": True},
        "profile": {"measured": False, "error": "CUPTI unavailable", "cuda_kernel_names": []},
        "collectives": [{**timed(rank, exact(op, size)), **probe.collective_layout(op, size),
                          "correctness": exact(op, size)}
                        for op in probe.OPERATIONS for size in probe.MESSAGE_BYTES],
        "gemms": [{**timed(rank, {'ok': True, 'finite': True, 'shape': [size, size]}),
                   "shape_mnk": [size, size, size]} for size in probe.GEMM_SIZES],
    } for rank in range(2)]


def test_complete_numeric_records_pass_with_profiler_explicitly_unmeasured(probe):
    result = probe.aggregate_gpu_results(_complete_rank_records(probe), {"rc": 0, "timed_out": False})
    assert result["health_ok"] is True, result["errors"]
    assert len(result["collectives"]) == 30
    assert len(result["gemms"]) == 3
    assert len(result["gemms"][0]["per_rank"]) == 2
    assert all(profile["measured"] is False for profile in result["profiles"])


@pytest.mark.parametrize("bad", ["backward_missing", "postcheck", "sample_nan", "timeout"])
def test_forged_health_flag_cannot_hide_missing_or_failed_numerical_evidence(probe, bad):
    ranks = copy.deepcopy(_complete_rank_records(probe))
    launcher = {"rc": 0, "timed_out": False}
    if bad == "backward_missing":
        del ranks[1]["gemm_correctness"]["grad_b"]
    elif bad == "postcheck":
        ranks[1]["collectives"][0]["postchecks"][1] = {"ok": False}
    elif bad == "sample_nan":
        ranks[1]["collectives"][0]["windows"][1]["wall_seconds"][3] = math.nan
    else:
        launcher["timed_out"] = True
    assert probe.aggregate_gpu_results(ranks, launcher)["health_ok"] is False


def test_all_reduce_reset_does_not_reduce_the_previous_sum(probe):
    torch = pytest.importorskip("torch")
    left = probe.collective_tensors(torch, "all_reduce", 16, 0, device="cpu")
    right = probe.collective_tensors(torch, "all_reduce", 16, 1, device="cpu")
    for _ in range(100):
        probe._reset_collective(left)
        left["input"].add_(right["input"])
        assert probe._check_collective(left)["ok"] is True


@pytest.mark.parametrize('bad', ['wrong_gpu', 'missing_gpu', 'wrong_arch', 'wrong_rank',
    'nonfinite', 'wrong_value', 'wrong_shape', 'missing_count', 'postcheck_nonfinite',
    'gemm_postcheck_nonfinite', 'wrong_window'])
def test_summary_rechecks_identity_and_communication_content_not_success_flags(probe, bad):
    ranks = _complete_rank_records(probe)
    rank = ranks[1]
    check = rank['collectives'][0]['correctness']
    if bad == 'wrong_gpu': rank['capabilities']['name'] = 'NVIDIA A100'
    elif bad == 'missing_gpu': rank['capabilities'] = {}
    elif bad == 'wrong_arch': rank['capabilities']['compute_capability'] = [8, 0]
    elif bad == 'wrong_rank': rank['capabilities']['rank'] = 0
    elif bad == 'nonfinite': check['finite'] = False
    elif bad == 'wrong_value': check['mismatched_elements'] = 1
    elif bad == 'wrong_shape': check['actual_shape'] = [100]
    elif bad == 'missing_count': del check['mismatched_elements']
    elif bad == 'postcheck_nonfinite': rank['collectives'][0]['postchecks'][0]['finite'] = False
    elif bad == 'gemm_postcheck_nonfinite': rank['gemms'][0]['postchecks'][0]['finite'] = False
    else: rank['collectives'][0]['windows'][0]['window'] = 2
    assert probe.aggregate_gpu_results(ranks, {'rc': 0, 'timed_out': False})['health_ok'] is False


def test_launcher_allows_installed_torchrun_to_reap_separate_worker_sessions(probe, tmp_path, monkeypatch):
    def guarded(command, **kwargs):
        # Installed close(): a shared 30s TERM window, then up to 30s per worker.
        assert kwargs['stop_grace'] == 105
        return {'rc': -1, 'timed_out': True}
    monkeypatch.setattr('syncopate.pipeline.process_guard.run_guarded', guarded)
    result = probe.run_gpu_probe(tmp_path)
    assert result['health_ok'] is False
