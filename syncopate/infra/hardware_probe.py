"""B13 synthetic CPU/NCCL/BF16 hardware probe; torch is loaded only on execution."""
from __future__ import annotations

import argparse
from datetime import timedelta
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


WORLD_SIZE = 2
MESSAGE_BYTES = (12, 16, 20, 1 << 20, 16 << 20, 64 << 20)
OPERATIONS = ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
              "send_recv_0_to_1", "send_recv_1_to_0")
GEMM_SIZES = (1024, 2048, 4096)
WINDOWS, WARMUPS, REPETITIONS = 3, 2, 10
GEMM_RELATIVE_TOLERANCE = 0.02
DISTRIBUTED_TIMEOUT_SECONDS = 90
WORKER_TIMEOUT_SECONDS = 480
LAUNCHER_STOP_GRACE_SECONDS = 105  # torchrun: 30s TERM + up to 30s KILL wait per worker.


def _positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def collective_layout(operation: str, block_bytes: int, *, world_size: int = WORLD_SIZE) -> dict:
    """A block is one rank's gather input / scatter output, not their total array."""
    if operation not in OPERATIONS:
        raise ValueError(f"unregistered collective: {operation}")
    if type(world_size) is not int or world_size < 2:
        raise ValueError("world_size must be an integer >= 2")
    if type(block_bytes) is not int or block_bytes <= 0 or block_bytes % 4:
        raise ValueError("float32 block_bytes must be a positive multiple of 4")
    count = block_bytes // 4
    inputs = count * world_size if operation == "reduce_scatter_tensor" else count
    outputs = count * world_size if operation == "all_gather_into_tensor" else count
    return {"operation": operation, "world_size": world_size, "dtype": "float32",
            "block_bytes": block_bytes, "input_numel": inputs, "output_numel": outputs,
            "algorithm_bytes": max(inputs, outputs) * 4,
            "block_alignment_mod_16": block_bytes % 16}


def bandwidth_gbps(operation: str, block_bytes: int, seconds: float, *, world_size: int = WORLD_SIZE) -> dict:
    """NCCL-tests convention: total-array S/t; decimal GB/s (1e9 bytes/s)."""
    _positive(seconds, "elapsed seconds")
    layout = collective_layout(operation, block_bytes, world_size=world_size)
    if operation == "all_reduce":
        factor = 2 * (world_size - 1) / world_size
    elif operation in {"all_gather_into_tensor", "reduce_scatter_tensor"}:
        factor = (world_size - 1) / world_size
    else:
        factor = 1.0  # Separate one-way send/recv measurements; never double bytes.
    algorithm = layout["algorithm_bytes"] / seconds / 1e9
    return {"algorithm_GB_s": algorithm, "bus_GB_s": algorithm * factor,
            "bus_factor": factor, "algorithm_bytes": layout["algorithm_bytes"]}


def gemm_tflops(m: int, n: int, k: int, seconds: float) -> float:
    for value, label in ((m, "M"), (n, "N"), (k, "K"), (seconds, "elapsed seconds")):
        _positive(value, label)
    return 2 * m * n * k / seconds / 1e12


def comparison_metrics(error_l2: float, reference_l2: float, *, finite: bool,
                       same_shape: bool = True, tolerance: float = GEMM_RELATIVE_TOLERANCE) -> dict:
    _positive(tolerance, "relative tolerance")
    valid = all(math.isfinite(x) and x >= 0 for x in (error_l2, reference_l2))
    relative = None
    if valid and reference_l2 > 0:
        relative = error_l2 / reference_l2
    elif valid and error_l2 == 0:
        relative = 0.0
    if relative is not None and not math.isfinite(relative):
        relative = None
    return {"ok": bool(finite and same_shape and valid and relative is not None and relative <= tolerance),
            "finite": bool(finite and valid), "same_shape": bool(same_shape),
            "error_l2": error_l2 if math.isfinite(error_l2) else None,
            "reference_l2": reference_l2 if math.isfinite(reference_l2) else None,
            "relative_l2": relative, "relative_tolerance": tolerance}


def summarize_samples(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty timing samples")
    for value in values:
        _positive(value, "timing sample")
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}


def aggregate_timings(ranks: list[dict]) -> dict:
    """Keep raw samples and use max rank window elapsed / reps for primary latency."""
    if len(ranks) != WORLD_SIZE or {r["rank"] for r in ranks} != set(range(WORLD_SIZE)):
        raise ValueError("exactly ranks 0 and 1 are required")
    ranks = sorted(ranks, key=lambda item: item["rank"])
    count = len(ranks[0]["windows"])
    if not count or any(len(r["windows"]) != count for r in ranks):
        raise ValueError("missing or different timing windows")
    windows = []
    for index in range(count):
        raw = {str(r["rank"]): r["windows"][index] for r in ranks}
        reps = len(raw["0"]["wall_seconds"])
        if not reps or any(len(w[key]) != reps for w in raw.values()
                           for key in ("wall_seconds", "cuda_seconds")):
            raise ValueError("missing or different timing samples")
        for window in raw.values():
            summarize_samples(window["wall_seconds"])
            summarize_samples(window["cuda_seconds"])
        elapsed = {rank: sum(w["wall_seconds"]) for rank, w in raw.items()}
        maximums = [max(w["wall_seconds"][i] for w in raw.values()) for i in range(reps)]
        windows.append({"window": index, "raw_ranks": raw, "rank_elapsed_seconds": elapsed,
                        "max_rank_elapsed_seconds": max(elapsed.values()),
                        "seconds_per_call": max(elapsed.values()) / reps,
                        "max_rank_sample_seconds": maximums,
                        "sample_dispersion": summarize_samples(maximums)})
    return {"windows": windows,
            "seconds_per_call_summary": summarize_samples([w["seconds_per_call"] for w in windows]),
            "independent_host_samples": 1, "sample_unit": "warmed_windows_on_one_allocation"}


def prepare_output_dir(output: Path) -> Path:
    """An existing attempt, even failed or incomplete, is evidence and is preserved."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json_once(path: Path, record: dict) -> None:
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def launcher_command(output: Path) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0", "--module",
            "syncopate.infra.hardware_probe", "--mode", "gpu", "--worker", "--output", str(output)]


def registered_plan() -> dict:
    return {"world_size": WORLD_SIZE, "gpu": "B200", "message_block_bytes": MESSAGE_BYTES,
            "operations": OPERATIONS, "communication_dtype": "float32", "gemm_dtype": "bfloat16",
            "gemm_sizes": GEMM_SIZES, "gemm_relative_l2_limit": GEMM_RELATIVE_TOLERANCE,
            "windows": WINDOWS, "warmups_per_window": WARMUPS, "repetitions_per_window": REPETITIONS,
            "distributed_timeout_seconds": DISTRIBUTED_TIMEOUT_SECONDS,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "launcher_stop_grace_seconds": LAUNCHER_STOP_GRACE_SECONDS,
            "sample_unit": "warmed_windows_on_one_allocation", "independent_host_samples": 1,
            "primary_timer": "host wall seconds from dispatch through device synchronize",
            "excluded_from_timer": ["input reset", "rank barrier", "pre-dispatch device synchronize"],
            "secondary_timer": "CUDA events surrounding dispatch, current stream waits for NCCL",
            "tensor_core_instruction_utilization": "unmeasured", "acceleration_claim": False}


def installed_interfaces(torch) -> dict:
    """Inspect the installed target APIs and parse the exact torchrun CLI, without CUDA calls."""
    import torch.distributed as dist
    from torch.distributed.run import get_args_parser

    signatures = {}
    bindings = {
        "init_process_group": {"backend": "nccl", "rank": 0, "world_size": WORLD_SIZE,
                               "timeout": timedelta(seconds=DISTRIBUTED_TIMEOUT_SECONDS)},
        "all_reduce": {"tensor": None},
        "all_gather_into_tensor": {"output_tensor": None, "input_tensor": None},
        "reduce_scatter_tensor": {"output": None, "input": None},
        "send": {"tensor": None, "dst": 1}, "recv": {"tensor": None, "src": 0},
        "barrier": {"device_ids": [0]}, "destroy_process_group": {},
    }
    for name, kwargs in bindings.items():
        signature = inspect.signature(getattr(dist, name))
        signature.bind(**kwargs)
        signatures[name] = str(signature)
    signatures["Tensor.backward"] = str(inspect.signature(torch.Tensor.backward))
    signatures["aten.mm"] = str(torch.ops.aten.mm.default._schema)
    args = get_args_parser().parse_args(launcher_command(Path("CPU_SIGNATURE_CHECK"))[3:])
    if not (args.module and args.standalone and args.nnodes == "1" and args.nproc_per_node == "2"
            and args.max_restarts == 0 and args.training_script == "syncopate.infra.hardware_probe"):
        raise ValueError("installed torchrun parsed a different launch configuration")
    return {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "distributed_available": dist.is_available(), "nccl_compiled": dist.is_nccl_available(),
            "signatures": signatures,
            "torchrun_args": {key: getattr(args, key) for key in
                              ("module", "standalone", "nnodes", "nproc_per_node", "max_restarts")}}


def exact_tensor_check(actual, expected) -> dict:
    import torch

    same_shape = tuple(actual.shape) == tuple(expected.shape)
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    mismatch = int((actual != expected).sum().item()) if same_shape else None
    return {"ok": finite and same_shape and mismatch == 0, "finite": finite,
            "same_shape": same_shape, "mismatched_elements": mismatch,
            "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}


def _relative_tensor_check(actual, expected, tolerance: float) -> dict:
    import torch

    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    same_shape = tuple(actual.shape) == tuple(expected.shape)
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    error = float((actual - expected).norm().item()) if same_shape else math.nan
    return comparison_metrics(error, float(expected.norm().item()), finite=finite,
                              same_shape=same_shape, tolerance=tolerance)


def gemm_correctness(torch, *, device: str, dtype) -> dict:
    """Compare BF16 autograd to independent CPU FP32 matrix derivatives on exact inputs."""
    m, n, k = 37, 29, 43
    a_ref = ((torch.arange(m * k, dtype=torch.float32, device="cpu").reshape(m, k) % 17) - 8) / 16
    b_ref = ((torch.arange(k * n, dtype=torch.float32, device="cpu").reshape(k, n) % 13) - 6) / 16
    g_ref = ((torch.arange(m * n, dtype=torch.float32, device="cpu").reshape(m, n) % 7) - 3) / 16
    a = a_ref.to(device=device, dtype=dtype).requires_grad_()
    b = b_ref.to(device=device, dtype=dtype).requires_grad_()
    output = a @ b
    output.backward(g_ref.to(device=device, dtype=dtype))
    tolerance = GEMM_RELATIVE_TOLERANCE if dtype == torch.bfloat16 else 1e-6
    result = {"shape_mnk": [m, n, k], "dtype": str(dtype),
              "reference": "CPU float32 from exactly representable inputs; analytical dA and dB",
              "forward": _relative_tensor_check(output, a_ref @ b_ref, tolerance),
              "grad_a": _relative_tensor_check(a.grad, g_ref @ b_ref.T, tolerance),
              "grad_b": _relative_tensor_check(b.grad, a_ref.T @ g_ref, tolerance)}
    result["ok"] = all(result[key]["ok"] for key in ("forward", "grad_a", "grad_b"))
    return result


def collective_tensors(torch, operation: str, block_bytes: int, rank: int, *, device: str) -> dict:
    """Use exactly representable values varying by rank, chunk and position."""
    layout = collective_layout(operation, block_bytes)
    if rank not in range(WORLD_SIZE):
        raise ValueError("rank must be 0 or 1")
    base = (torch.arange(block_bytes // 4, dtype=torch.int32, device=device) % 17).float() - 8
    source = base + 32 * rank
    if operation == "all_reduce":
        expected = WORLD_SIZE * base + 32
    elif operation == "all_gather_into_tensor":
        expected = torch.cat([base + 32 * peer for peer in range(WORLD_SIZE)])
    elif operation == "reduce_scatter_tensor":
        source = torch.cat([source + 16 * chunk for chunk in range(WORLD_SIZE)])
        expected = WORLD_SIZE * (base + 16 * rank) + 32
    else:
        sender = 0 if operation == "send_recv_0_to_1" else 1
        expected = base + 32 * sender
    return {"layout": layout, "input": source.clone(), "source": source,
            "output": torch.empty(layout["output_numel"], dtype=torch.float32, device=device),
            "expected": expected, "rank": rank}


def cpu_formula_checks(torch) -> dict:
    result = {"gemm": gemm_correctness(torch, device="cpu", dtype=torch.float32), "collectives": []}
    for block_bytes in MESSAGE_BYTES[:3]:
        for operation in OPERATIONS:
            cases = [collective_tensors(torch, operation, block_bytes, rank, device="cpu")
                     for rank in range(WORLD_SIZE)]
            if operation == "all_reduce":
                reduced = torch.stack([case["input"] for case in cases]).sum(dim=0)
                actual = [reduced, reduced]
            elif operation == "all_gather_into_tensor":
                gathered = torch.cat([case["input"] for case in cases])
                actual = [gathered, gathered]
            elif operation == "reduce_scatter_tensor":
                reduced = torch.stack([case["input"] for case in cases]).sum(dim=0)
                actual = list(reduced.chunk(WORLD_SIZE))
            else:
                sender = 0 if operation == "send_recv_0_to_1" else 1
                actual = [cases[sender]["input"].clone() for _ in range(WORLD_SIZE)]
            checks = [exact_tensor_check(value, case["expected"]) for value, case in zip(actual, cases)]
            result["collectives"].append({"operation": operation, "block_bytes": block_bytes,
                                          "ranks": checks, "ok": all(c["ok"] for c in checks)})
    result["health_ok"] = result["gemm"]["ok"] and all(c["ok"] for c in result["collectives"])
    return result


def run_cpu_probe() -> dict:
    import torch

    formulas = cpu_formula_checks(torch)
    interfaces = installed_interfaces(torch)
    return {"mode": "cpu", "health_ok": bool(formulas["health_ok"] and interfaces["distributed_available"]
                                              and interfaces["nccl_compiled"]),
            "formula_checks": formulas, "interfaces": interfaces,
            "capabilities": {"cuda_queried": False, "scope": "CPU formulas and installed API signatures"},
            "large_shape_layouts": [collective_layout(op, size) for op in OPERATIONS for size in MESSAGE_BYTES],
            "nccl_executed": False}


def kernel_names_from_trace(trace: dict) -> list[str]:
    return sorted({event["name"] for event in trace.get("traceEvents", [])
                   if "kernel" in str(event.get("cat", "")).split(",") and event.get("name")})


def _require_ok(check: dict, label: str) -> None:
    if check.get("ok") is not True:
        raise ValueError(f"{label} failed: {json.dumps(check, allow_nan=False)}")


def _gpu_capabilities(torch, rank: int) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError("B13 requires exactly two visible CUDA GPUs")
    properties = torch.cuda.get_device_properties(rank)
    if "B200" not in properties.name or (properties.major, properties.minor) != (10, 0):
        raise ValueError(f"GPU {rank} is not a B200 sm_100: {properties}")
    peers = {}
    for other in range(WORLD_SIZE):
        if other != rank:
            try:
                peers[str(other)] = {"measured": True,
                                     "accessible": bool(torch.cuda.can_device_access_peer(rank, other))}
            except Exception as exc:
                peers[str(other)] = {"measured": False, "error": str(exc)}
    driver = {"measured": False}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15)
        rows = [row.split(",") for row in query.stdout.splitlines() if row.strip()]
        if query.returncode == 0 and rows and all(len(row) == 2 for row in rows):
            driver = {"measured": True, "uuid_driver_rows": [
                {"uuid": row[0].strip(), "driver_version": row[1].strip()} for row in rows]}
        else:
            driver["error"] = query.stderr.strip() or "nvidia-smi returned no valid UUID/driver rows"
    except (OSError, subprocess.TimeoutExpired) as exc:
        driver["error"] = str(exc)
    uuid = getattr(properties, "uuid", None)
    return {"rank": rank, "name": properties.name, "compute_capability": [properties.major, properties.minor],
            "uuid": str(uuid) if uuid is not None else None, "uuid_measured": uuid is not None,
            "total_memory_bytes": properties.total_memory, "multiprocessor_count": properties.multi_processor_count,
            "peer_access": peers, "driver": driver, "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "nccl_version": torch.cuda.nccl.version()}


def _reset_collective(case: dict) -> None:
    # In-place all_reduce must not consume the previous sum. Reset is outside all timers.
    case["input"].copy_(case["source"])
    case["output"].fill_(math.nan)


def _call_collective(dist, case: dict) -> None:
    operation = case["layout"]["operation"]
    if operation == "all_reduce":
        dist.all_reduce(case["input"])
    elif operation == "all_gather_into_tensor":
        dist.all_gather_into_tensor(case["output"], case["input"])
    elif operation == "reduce_scatter_tensor":
        dist.reduce_scatter_tensor(case["output"], case["input"])
    else:
        sender = 0 if operation == "send_recv_0_to_1" else 1
        if case["rank"] == sender:
            dist.send(case["input"], dst=1 - sender)
        else:
            dist.recv(case["output"], src=sender)


def _check_collective(case: dict) -> dict:
    operation = case["layout"]["operation"]
    sender = 0 if operation == "send_recv_0_to_1" else 1
    sending = operation.startswith("send_recv") and case["rank"] == sender
    actual = case["input"] if operation == "all_reduce" or sending else case["output"]
    result = exact_tensor_check(actual, case["expected"])
    result["role"] = "sender_input" if sending else "computed_or_received_output"
    return result


def _barrier(torch, dist, rank: int) -> None:
    dist.barrier(device_ids=[rank])
    torch.cuda.synchronize(rank)


def _time_windows(torch, dist, rank: int, call, *, reset=None, check=None) -> tuple[list[dict], list[dict]]:
    windows, postchecks = [], []
    for index in range(WINDOWS):
        for _ in range(WARMUPS):
            if reset is not None:
                reset()
            _barrier(torch, dist, rank)
            call()
            torch.cuda.synchronize(rank)
        wall_seconds, cuda_seconds = [], []
        # Events are reusable; reset/collective barrier/sync finish before each start event.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(REPETITIONS):
            if reset is not None:
                reset()
            _barrier(torch, dist, rank)
            start.record()
            t0 = time.perf_counter()
            call()
            end.record()
            torch.cuda.synchronize(rank)
            wall_seconds.append(time.perf_counter() - t0)
            cuda_seconds.append(start.elapsed_time(end) / 1000)
        summarize_samples(wall_seconds)
        summarize_samples(cuda_seconds)
        windows.append({"window": index, "wall_seconds": wall_seconds, "cuda_seconds": cuda_seconds})
        if check is not None:
            value = check()
            postchecks.append(value)
            _require_ok(value, f"post-window {index}")
    return windows, postchecks


def _gemm_timing(torch, dist, rank: int, size: int) -> dict:
    device = f"cuda:{rank}"
    generator = torch.Generator(device=device).manual_seed(1300 + rank + size)
    a = torch.randn((size, size), device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((size, size), device=device, dtype=torch.bfloat16, generator=generator)
    output = torch.empty_like(a)

    def call():
        torch.mm(a, b, out=output)

    def check():
        finite = bool(torch.isfinite(output).all().item())
        return {"ok": finite, "finite": finite, "shape": list(output.shape)}

    call()
    _require_ok(check(), f"GEMM {size} finite precheck")
    windows, postchecks = _time_windows(torch, dist, rank, call, check=check)
    return {"rank": rank, "shape_mnk": [size, size, size], "dtype": "bfloat16",
            "operation": "gemm_forward", "flops_per_call": 2 * size ** 3,
            "windows": windows, "postchecks": postchecks}


def _profile_interval(torch, dist, rank: int, output: Path) -> dict:
    """Profiler errors are unmeasured; actual math/transport failures still escape."""
    result = {"measured": False, "trace_file": None, "cuda_kernel_names": [],
              "scope": "separate GEMM 1024 and all_reduce 1 MiB interval",
              "tensor_core_instruction_utilization": "unmeasured"}
    profiler = None
    try:
        from torch.profiler import ProfilerActivity, profile
        candidate = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
        candidate.__enter__()
        profiler = candidate
    except Exception as exc:
        result["error"] = f"profiler setup unavailable: {exc}"
    # Both ranks always execute these operations, even when only one profiler is available.
    try:
        device = f"cuda:{rank}"
        matrix = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
        case = collective_tensors(torch, "all_reduce", 1 << 20, rank, device=device)
        _reset_collective(case)
        _barrier(torch, dist, rank)
        product = matrix @ matrix
        _call_collective(dist, case)
        torch.cuda.synchronize(rank)
        _require_ok(_check_collective(case), "profile interval all_reduce")
        _require_ok(exact_tensor_check(product, torch.full_like(product, 1024)), "profile interval GEMM")
    finally:
        if profiler is not None:
            try:
                profiler.__exit__(None, None, None)
            except Exception as exc:
                result["error"] = f"profiler finish unavailable: {exc}"
                profiler = None
    if profiler is not None:
        try:
            trace_path = output / "profile.trace.json"
            if trace_path.exists():
                raise FileExistsError(f"preserved existing trace: {trace_path}")
            profiler.export_chrome_trace(str(trace_path))
            result["trace_file"] = str(trace_path)
            result["cuda_kernel_names"] = kernel_names_from_trace(json.loads(trace_path.read_text()))
            result["measured"] = bool(result["cuda_kernel_names"])
            if not result["measured"]:
                result["error"] = "trace contains no CUDA kernel events; GPU profiling unmeasured"
        except Exception as exc:
            result["error"] = f"profiler export unavailable: {exc}"
    return result


def run_worker(root: Path) -> int:
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE or int(os.environ["LOCAL_RANK"]) != rank:
        raise ValueError("worker must be one of two local torchrun ranks")
    if rank not in range(WORLD_SIZE) or not (root / "plan.json").is_file():
        raise ValueError("worker rank or parent output is invalid")
    output = prepare_output_dir(root / f"rank_{rank}")
    result = {"rank": rank, "pid": os.getpid(), "health_ok": False, "collectives": [], "gemms": []}
    dist = None
    try:
        import torch
        import torch.distributed as dist

        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        result["capabilities"] = _gpu_capabilities(torch, rank)
        write_json_once(output / "capabilities.json", result["capabilities"])
        result["gemm_correctness"] = gemm_correctness(torch, device=f"cuda:{rank}", dtype=torch.bfloat16)
        write_json_once(output / "gemm_correctness.json", result["gemm_correctness"])
        _require_ok(result["gemm_correctness"], f"GPU {rank} BF16 forward/backward")
        dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE,
                                timeout=timedelta(seconds=DISTRIBUTED_TIMEOUT_SECONDS))
        # No rank can start measurements until both GPUs have passed GEMM and all communication cases.
        _barrier(torch, dist, rank)
        for operation in OPERATIONS:
            for size in MESSAGE_BYTES:
                case = collective_tensors(torch, operation, size, rank, device=f"cuda:{rank}")
                _reset_collective(case)
                _barrier(torch, dist, rank)
                _call_collective(dist, case)
                torch.cuda.synchronize(rank)
                record = {"rank": rank, **case["layout"], "correctness": _check_collective(case)}
                result["collectives"].append(record)
                write_json_once(output / f"check_{operation}_{size}.json", record)
                _require_ok(record["correctness"], f"{operation} {size} bytes rank {rank}")
                del case
        _barrier(torch, dist, rank)
        for record in result["collectives"]:
            operation, size = record["operation"], record["block_bytes"]
            case = collective_tensors(torch, operation, size, rank, device=f"cuda:{rank}")
            windows, postchecks = _time_windows(torch, dist, rank, lambda: _call_collective(dist, case),
                                                reset=lambda: _reset_collective(case),
                                                check=lambda: _check_collective(case))
            record.update({"windows": windows, "postchecks": postchecks})
            write_json_once(output / f"timing_{operation}_{size}.json", record)
            del case
        for size in GEMM_SIZES:
            record = _gemm_timing(torch, dist, rank, size)
            result["gemms"].append(record)
            write_json_once(output / f"gemm_{size}.json", record)
        result["profile"] = _profile_interval(torch, dist, rank, output)
        write_json_once(output / "profile.json", result["profile"])
        _barrier(torch, dist, rank)
        result["health_ok"] = True
    except Exception as exc:
        result.update({"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
                result["process_group_destroyed"] = True
            except Exception as exc:
                result["health_ok"] = False
                result["teardown_error"] = str(exc)
        else:
            result["process_group_destroyed"] = False
        write_json_once(output / "result.json", result)
    return 0 if result["health_ok"] else 1


def aggregate_gpu_results(ranks: list[dict], launcher: dict) -> dict:
    result = {"mode": "gpu", "health_ok": False, "ranks": ranks, "launcher": launcher,
              "collectives": [], "gemms": [], "errors": [], "acceleration_claim": False}
    try:
        if launcher.get("rc") != 0 or launcher.get("timed_out") is not False:
            raise ValueError("torchrun failed or exceeded its timeout")
        if len(ranks) != WORLD_SIZE or {r.get("rank") for r in ranks} != set(range(WORLD_SIZE)):
            raise ValueError("missing or duplicate rank result")
        for rank in ranks:
            if rank.get("health_ok") is not True or rank.get("process_group_destroyed") is not True:
                raise ValueError(f"rank {rank['rank']} did not complete successfully")
            if rank.get("gemm_correctness", {}).get("ok") is not True:
                raise ValueError(f"rank {rank['rank']} has no successful GEMM forward/backward check")
            capabilities = rank.get("capabilities", {})
            if (capabilities.get("rank") != rank["rank"]
                    or "B200" not in str(capabilities.get("name", ""))
                    or capabilities.get("compute_capability") != [10, 0]):
                raise ValueError(f"rank {rank['rank']} lacks B200 sm_100 identity")
            for name in ("forward", "grad_a", "grad_b"):
                metric = rank["gemm_correctness"].get(name, {})
                recomputed = comparison_metrics(metric.get("error_l2", math.nan),
                                                metric.get("reference_l2", math.nan),
                                                finite=metric.get("finite") is True,
                                                same_shape=metric.get("same_shape") is True)
                if metric.get("ok") is not True or recomputed["ok"] is not True:
                    raise ValueError(f"rank {rank['rank']} lacks valid {name} evidence")
            if len(rank.get("collectives", [])) != len(OPERATIONS) * len(MESSAGE_BYTES):
                raise ValueError(f"rank {rank['rank']} has missing collective cases")
            if len(rank.get("gemms", [])) != len(GEMM_SIZES):
                raise ValueError(f"rank {rank['rank']} has missing GEMM cases")
        for operation in OPERATIONS:
            for size in MESSAGE_BYTES:
                records = []
                for rank in ranks:
                    matches = [c for c in rank["collectives"] if c["operation"] == operation and c["block_bytes"] == size]
                    if len(matches) != 1 or matches[0].get("correctness", {}).get("ok") is not True:
                        raise ValueError(f"missing or failed {operation} {size} bytes on rank {rank['rank']}")
                    expected_shape = [collective_layout(operation, size)["output_numel"]]
                    _require_exact_evidence(matches[0]["correctness"], expected_shape)
                    records.append(matches[0])
                measurement = _aggregate_complete_measurement(records, exact_shape=expected_shape)
                for window in measurement["windows"]:
                    window["bandwidth"] = bandwidth_gbps(operation, size, window["seconds_per_call"])
                measurement.update(collective_layout(operation, size))
                measurement["bandwidth_at_median_latency"] = bandwidth_gbps(
                    operation, size, measurement["seconds_per_call_summary"]["median"])
                result["collectives"].append(measurement)
        for size in GEMM_SIZES:
            records = []
            for rank in ranks:
                matches = [g for g in rank["gemms"] if g["shape_mnk"] == [size, size, size]]
                if len(matches) != 1:
                    raise ValueError(f"missing or duplicate GEMM {size} on rank {rank['rank']}")
                records.append(matches[0])
            measurement = _aggregate_complete_measurement(records, gemm_shape=[size, size])
            per_rank = []
            for record in records:
                latencies = [sum(w["wall_seconds"]) / REPETITIONS for w in record["windows"]]
                per_rank.append({"rank": record["rank"], "seconds_per_call": latencies,
                                 "TFLOP_s": [gemm_tflops(size, size, size, value) for value in latencies],
                                 "seconds_per_call_summary": summarize_samples(latencies)})
            measurement.update({"shape_mnk": [size, size, size], "flops_per_call": 2 * size ** 3,
                                "dtype": "bfloat16", "per_rank": per_rank})
            result["gemms"].append(measurement)
        result["capabilities"] = [r["capabilities"] for r in sorted(ranks, key=lambda row: row["rank"])]
        result["profiles"] = [{"rank": rank["rank"], **rank["profile"]} for rank in ranks]
        result["health_ok"] = True
    except (KeyError, TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
    return result


def _require_exact_evidence(check: dict, shape: list[int]) -> None:
    if (check.get("ok") is not True or check.get("finite") is not True
            or check.get("same_shape") is not True
            or type(check.get("mismatched_elements")) is not int or check["mismatched_elements"] != 0
            or check.get("actual_shape") != shape or check.get("expected_shape") != shape):
        raise ValueError("missing or failed exact communication evidence")


def _aggregate_complete_measurement(records: list[dict], *, exact_shape=None, gemm_shape=None) -> dict:
    for record in records:
        if len(record.get("windows", [])) != WINDOWS or any(
            len(window.get("wall_seconds", [])) != REPETITIONS or window.get("window") != index
            for index, window in enumerate(record["windows"])
        ):
            raise ValueError("missing preregistered window or timing samples")
        if len(record.get("postchecks", [])) != WINDOWS or any(c.get("ok") is not True for c in record["postchecks"]):
            raise ValueError("missing or failed post-window numerical check")
        for check in record["postchecks"]:
            if exact_shape is not None:
                _require_exact_evidence(check, exact_shape)
            elif check.get("finite") is not True or check.get("shape") != gemm_shape:
                raise ValueError("missing or failed GEMM post-window numerical evidence")
    return aggregate_timings(records)


def run_gpu_probe(output: Path) -> dict:
    from syncopate.pipeline.process_guard import run_guarded

    launcher = run_guarded(launcher_command(output), cwd=Path(__file__).resolve().parents[2],
                           timeout=WORKER_TIMEOUT_SECONDS, stop_grace=LAUNCHER_STOP_GRACE_SECONDS)
    write_json_once(output / "launcher.json", launcher)
    ranks, read_errors = [], []
    for rank in range(WORLD_SIZE):
        path = output / f"rank_{rank}" / "result.json"
        try:
            ranks.append(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            read_errors.append(f"cannot read rank {rank} result: {exc}")
    result = aggregate_gpu_results(ranks, launcher)
    result["errors"].extend(read_errors)
    result["health_ok"] = result["health_ok"] and not read_errors
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.mode != "gpu":
            parser.error("--worker is only valid for gpu mode")
        return run_worker(args.output)
    try:
        output = prepare_output_dir(args.output.absolute())
    except FileExistsError:
        print(f"output already exists; preserved: {args.output}", file=sys.stderr)
        return 2
    write_json_once(output / "plan.json", registered_plan())
    started = time.monotonic()
    try:
        result = run_cpu_probe() if args.mode == "cpu" else run_gpu_probe(output)
    except Exception as exc:
        result = {"mode": args.mode, "health_ok": False, "error": str(exc),
                  "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
    result.update({"schema_version": 1, "elapsed_seconds": time.monotonic() - started,
                   "plan": registered_plan(), "output": str(output)})
    write_json_once(output / "result.json", result)
    print(json.dumps({"health_ok": result["health_ok"], "result": str(output / "result.json")}))
    return 0 if result["health_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
