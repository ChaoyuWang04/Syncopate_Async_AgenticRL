"""B03 bounded full-parameter Text MoE/FSDP2 correctness, not a speed claim.

Heavy dependencies are deliberately imported only by execution helpers. The CPU
arm creates the tiny native model; only the GPU worker initializes the engine.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import sys
import traceback

from syncopate.infra.hardware_probe import prepare_output_dir, write_json_once


WORLD_SIZE = 2
WORKER_TIMEOUT_SECONDS = 480
DISTRIBUTED_TIMEOUT_SECONDS = 90
STOP_GRACE_SECONDS = 105
GRADIENT_RELATIVE_L2_MAX = 1e-4
GRADIENT_ATOL, GRADIENT_RTOL = 1e-6, 1e-4
PARAMETER_ATOL, PARAMETER_RTOL = 2e-6, 1e-4
LOGPROB_ATOL, LOGPROB_RTOL = 1e-5, 1e-5
CLIP_GRAD = 1e6
LEARNING_RATE = 1e-3


def preregistered_plan() -> dict:
    return {"world_size": WORLD_SIZE, "arms": [False, True], "updates": 2,
            "micro_batch_size_per_gpu": 1, "rows_per_rank": 2,
            "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
            "gradient_atol": GRADIENT_ATOL, "gradient_rtol": GRADIENT_RTOL,
            "parameter_atol": PARAMETER_ATOL, "parameter_rtol": PARAMETER_RTOL,
            "logprob_atol": LOGPROB_ATOL, "logprob_rtol": LOGPROB_RTOL,
            "restart_equality": "exact", "hybrid_gdn_measured": False,
            "lora_measured": False, "performance_acceptance": False,
            "multi_sequence_packed_micro_measured": False, "model_35b_measured": False,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "distributed_timeout_seconds": DISTRIBUTED_TIMEOUT_SECONDS,
            "stop_grace_seconds": STOP_GRACE_SECONDS,
            "reference": "native_unsharded_CPU_global_token_mean",
            "optimizer_reference": "CPU_AdamW_same_state_same_actual_gradient",
            "moment_error_scale": "8*eps32*max(abs(old),abs(g_or_g_squared),abs(expected))",
            "reference_trajectory": "rebased_to_actual_pre_step_state",
            "experts_implementation": "eager",
            "primary": "full_pre_clip_parameter_gradients",
            "scope": "two_full_attention_layers_with_native_MoE"}


def tiny_config_kwargs() -> dict:
    return {"model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"],
            "vocab_size": 128, "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "layer_types": ["full_attention", "full_attention"],
            "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
            "max_position_embeddings": 128, "attention_dropout": 0.0,
            "output_router_logits": False, "router_aux_loss_coef": 0.0,
            "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 1,
            "eos_token_id": 2, "use_cache": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]}}


def synthetic_batch(step: int) -> list[dict]:
    if step not in (0, 1):
        raise ValueError("only the two preregistered updates exist")
    rows = []
    for index, (length, count) in enumerate(zip((7, 11, 9, 13), (3, 5, 4, 7))):
        ids = [3 + ((index * 23 + position * 7 + step * 17) % 125) for position in range(length)]
        mask = [int(length - 1 - count <= position < length - 1) for position in range(length)]
        rows.append({"row_id": index, "input_ids": ids, "position_ids": list(range(length)),
                     "loss_mask": mask})
    validate_batch(rows)
    return rows


def validate_batch(rows: list[dict]) -> dict:
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty distinct rows are required")
    counts = []
    for row in rows:
        ids, positions, mask = (row[key] for key in ("input_ids", "position_ids", "loss_mask"))
        if len(ids) < 2 or len(mask) != len(ids) or positions != list(range(len(ids))):
            raise ValueError("input, position and mask sequence shape contract failed")
        if any(type(token) is not int or not 0 <= token < 128 for token in ids):
            raise ValueError("token id outside tiny vocabulary")
        if any(type(value) is not int or value not in (0, 1) for value in mask) or mask[-1] != 0:
            raise ValueError("binary loss mask with last token zero is required")
        if sum(mask) <= 0:
            raise ValueError("each row must supervise at least one next token")
        counts.append(sum(mask))
    return {"rows": len(rows), "tokens": sum(counts), "tokens_by_row": counts}


def batch_digest(rows: list[dict]) -> str:
    validate_batch(rows)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def loss_scale(batch_num_tokens: float, dp_size: int) -> float:
    if (not math.isfinite(batch_num_tokens) or batch_num_tokens <= 0
            or type(dp_size) is not int or dp_size <= 0):
        raise ValueError("finite positive global token denominator and integer DP size required")
    return dp_size / batch_num_tokens


def scheduler_advanced(before: dict, after: dict) -> bool:
    start, end = before.get("last_epoch"), after.get("last_epoch")
    return type(start) is int and type(end) is int and end == start + 1


def gradient_verdict(*, error_l2: float, reference_l2: float, max_abs_error: float,
                     max_abs_reference: float, finite: bool, same_shape: bool = True) -> dict:
    values = (error_l2, reference_l2, max_abs_error, max_abs_reference)
    valid = finite and same_shape and all(math.isfinite(x) and x >= 0 for x in values)
    relative = (error_l2 / reference_l2 if reference_l2 > 0 else
                0.0 if error_l2 == 0 else None) if valid else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    absolute_bound = GRADIENT_ATOL + GRADIENT_RTOL * max_abs_reference if valid else None
    return {"ok": bool(valid and relative is not None and relative <= GRADIENT_RELATIVE_L2_MAX
                       and max_abs_error <= absolute_bound),
            "finite": bool(finite and all(math.isfinite(x) for x in values)),
            "same_shape": same_shape, "relative_l2": relative,
            "error_l2": error_l2 if math.isfinite(error_l2) else None,
            "reference_l2": reference_l2 if math.isfinite(reference_l2) else None,
            "max_abs_error": max_abs_error if math.isfinite(max_abs_error) else None,
            "max_abs_reference": max_abs_reference if math.isfinite(max_abs_reference) else None,
            "max_abs_bound": absolute_bound}


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def valid_gpu_uuid(value) -> bool:
    """Accept the same UUID with torch's raw form or nvidia-smi's GPU- prefix."""
    return isinstance(value, str) and re.fullmatch(
        r'(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', value) is not None


def _file_hashes(directory: Path) -> dict:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def require_cpu_evidence(output: Path) -> dict:
    cpu = output.parent / "cpu"
    result = json.loads((cpu / "result.json").read_text())
    if (result.get("mode") != "cpu" or result.get("health_ok") is not True
            or result.get("interfaces", {}).get("nccl_compiled") is not True
            or result.get("source_sha256") != source_sha256()
            or result.get("plan") != preregistered_plan()):
        raise ValueError("successful matching-source CPU/NCCL preflight is required")
    model = result.get("tiny_model", {})
    expected_dir = cpu / "tiny_model"
    # Only this arm's sibling artifact is consumed; no external/global model path.
    if (Path(model.get("path", "")).resolve() != expected_dir.resolve()
            or not model.get("files") or _file_hashes(expected_dir) != model["files"]
            or "config.json" not in model["files"]
            or not any(name.endswith(".safetensors") for name in model["files"])):
        raise ValueError("tiny initial weights are missing, changed, or outside this run")
    return result


def build_tiny_model():
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    # CPU generator only: torch.manual_seed would also queue device seed calls.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(137)
        kwargs = tiny_config_kwargs()
        kwargs.pop("model_type")
        config = Qwen3_5MoeTextConfig(**kwargs)
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        model = Qwen3_5MoeForCausalLM(config).float().cpu()
    if model.__class__.__name__ != "Qwen3_5MoeForCausalLM":
        raise AssertionError("wrong native Text root")
    if any("GatedDeltaNet" in type(module).__name__ for module in model.modules()):
        raise AssertionError("GDN is outside this bounded full-attention probe")
    if model.config._experts_implementation != "eager":
        raise AssertionError("native tiny expert implementation is not eager")
    return model


def build_engine_configs(tiny_dir: str, use_remove_padding: bool, *, construct_model: bool = True) -> dict:
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig

    configs = {
        "engine_config": FSDPEngineConfig(
            strategy="fsdp2", fsdp_size=2, dtype="fp32", model_dtype="fp32",
            mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            use_dynamic_bsz=False, micro_batch_size_per_gpu=1, infer_micro_batch_size_per_gpu=1,
            use_torch_compile=False, forward_only=False, offload_policy=False,
            param_offload=False, optimizer_offload=False, grad_offload=False,
            use_remove_padding=use_remove_padding, use_fused_kernels=False),
        "optimizer_config": FSDPOptimizerConfig(
            lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0, clip_grad=CLIP_GRAD,
            total_training_steps=2, lr_warmup_steps=0, lr_warmup_steps_ratio=0.0,
            lr_scheduler_type="constant", optimizer="AdamW", optimizer_impl="torch.optim",
            override_optimizer_config={"eps": 1e-8, "foreach": False, "fused": False}),
        "checkpoint_config": CheckpointConfig(save_contents=["model", "optimizer", "extra"],
                                               load_contents=["model", "optimizer", "extra"],
                                               save_lora_only=False),
    }
    if construct_model:
        configs["model_config"] = HFModelConfig(
            path=tiny_dir, load_tokenizer=False, enable_gradient_checkpointing=False,
            use_remove_padding=use_remove_padding, use_fused_kernels=False,
            override_config={"attn_implementation": "eager", "_experts_implementation": "eager"})
    return configs


def make_batch(rows: list[dict], *, use_remove_padding: bool):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    validate_batch(rows)
    values = {key: torch.nested.as_nested_tensor(
        [torch.tensor(row[key], dtype=torch.float32 if key == "loss_mask" else torch.long)
         for row in rows], layout=torch.jagged)
              for key in ("input_ids", "position_ids", "loss_mask")}
    values["row_id"] = torch.tensor([row["row_id"] for row in rows], dtype=torch.long)
    values["temperature"] = torch.ones(len(rows), dtype=torch.float32)
    td = TensorDict(values, batch_size=[len(rows)])
    tu.assign_non_tensor(td, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                         use_remove_padding=use_remove_padding, use_fused_kernels=False,
                         return_model_output=False)
    return td


def independent_reference(model, rows: list[dict]) -> dict:
    """Native non-FSDP root and torch log_softmax, never verl loss/logprob helpers."""
    import torch

    tokens = validate_batch(rows)["tokens"]
    device = next(model.parameters()).device
    log_probs, numerators = [], []
    for row in rows:
        ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
        positions = torch.tensor([row["position_ids"]], device=device, dtype=torch.long)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       position_ids=positions, use_cache=False)
        labels = torch.roll(ids, shifts=-1, dims=-1)
        lp = torch.log_softmax(output.logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
        mask = torch.tensor(row["loss_mask"], device=device, dtype=torch.float32)
        numerators.append(-(lp * mask).sum())
        log_probs.append(lp)
    return {"loss": torch.stack(numerators).sum() / tokens, "log_probs": log_probs,
            "tokens": tokens}


def _full_cpu(tensor):
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu().clone()


def capture_gradients(model) -> dict:
    missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    if missing:
        raise AssertionError(f"missing parameter gradients: {missing}")
    return {name: _full_cpu(param.grad) for name, param in model.named_parameters() if param.requires_grad}


def _tensor_metrics(reference, actual) -> dict:
    import torch

    shape = reference.shape == actual.shape
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    ref = reference.double()
    diff = actual.double() - ref if shape else torch.tensor([float("nan")])
    return gradient_verdict(error_l2=diff.norm().item(), reference_l2=ref.norm().item(),
                            max_abs_error=diff.abs().max().item() if diff.numel() else 0.0,
                            max_abs_reference=ref.abs().max().item() if ref.numel() else 0.0,
                            finite=finite, same_shape=shape)


def compare_gradients(reference: dict, actual: dict) -> dict:
    common = sorted(reference.keys() & actual.keys())
    per_parameter = {name: _tensor_metrics(reference[name], actual[name]) for name in common}
    same_keys = reference.keys() == actual.keys()
    valid = same_keys and bool(common) and all(p["finite"] and p["same_shape"] for p in per_parameter.values())
    error_l2 = math.sqrt(sum(p["error_l2"] ** 2 for p in per_parameter.values())) if valid else math.inf
    reference_l2 = math.sqrt(sum(p["reference_l2"] ** 2 for p in per_parameter.values())) if valid else math.inf
    global_check = gradient_verdict(
        error_l2=error_l2, reference_l2=reference_l2,
        max_abs_error=max((p["max_abs_error"] for p in per_parameter.values()), default=math.inf) if valid else math.inf,
        max_abs_reference=max((p["max_abs_reference"] for p in per_parameter.values()), default=math.inf) if valid else math.inf,
        finite=valid, same_shape=same_keys)
    # The registered relative bound is global; every tensor separately must pass
    # its absolute+relative max-absolute bound, including exact-zero references.
    per_tensor_ok = valid and all(p["max_abs_error"] <= p["max_abs_bound"] for p in per_parameter.values())
    return {"ok": bool(global_check["ok"] and per_tensor_ok), "global": global_check,
            "missing": sorted(reference.keys() - actual.keys()),
            "extra": sorted(actual.keys() - reference.keys()), "per_parameter": per_parameter}


def compare_exact_trees(reference, actual) -> dict:
    import numpy as np
    import torch

    mismatches = []

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            equal = (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                     and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
                     and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                     right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
        elif isinstance(left, np.ndarray):
            equal = (isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape
                     and np.isfinite(left).all() and np.isfinite(right).all()
                     and left.tobytes(order='C') == right.tobytes(order='C'))
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                mismatches.append(path + "/keys")
                return
            for key in left:
                visit(left[key], right[key], f"{path}/{key}")
            return
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                mismatches.append(path + "/sequence")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}/{index}")
            return
        elif type(left) is float:
            equal = (type(right) is float and math.isfinite(left) and math.isfinite(right)
                     and struct.pack('!d', left) == struct.pack('!d', right))
        else:
            equal = type(left) is type(right) and left == right
        if not equal:
            mismatches.append(path)

    visit(reference, actual, "state")
    return {"ok": not mismatches, "equality": "exact", "mismatches": mismatches}


@contextmanager
def cpu_only_guard():
    """Fail, rather than silently initialize CUDA while checking CPU interfaces."""
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight started after CUDA initialization")
    original = torch.cuda._lazy_init

    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU preflight attempted CUDA initialization")

    torch.cuda._lazy_init = forbidden
    try:
        yield
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    finally:
        torch.cuda._lazy_init = original


def _callable_record(fn) -> dict:
    fn = inspect.unwrap(fn)
    source = inspect.getsourcefile(fn)
    return {"callable": f"{fn.__module__}.{fn.__qualname__}", "signature": str(inspect.signature(fn)),
            "source": source,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}


def _save_tensor_once(path: Path, value) -> None:
    import torch

    with path.open("xb") as stream:
        torch.save(value, stream)


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def launcher_command(output: Path) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0", "--module", "syncopate.infra.training_probe",
            "--mode", "gpu", "--output", str(output), "--worker"]


def run_cpu_probe(output: Path) -> dict:
    import torch

    result = {"mode": "cpu", "health_ok": False, "plan": preregistered_plan(),
              "source_sha256": source_sha256()}
    with cpu_only_guard():
        from importlib.metadata import version
        from torch.distributed.run import get_args_parser
        from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead
        from verl.workers.engine.utils import prepare_micro_batches
        from verl.utils import tensordict_utils as tu

        torch.set_num_threads(1)
        model = build_tiny_model()
        model_dir = output / "tiny_model"
        model_dir.mkdir()
        model.save_pretrained(model_dir, safe_serialization=True)
        initial_state = _capture_model(model)
        model, _ = _native_reference_from_weights(model_dir)
        disk_reload_check = compare_exact_trees(initial_state, _capture_model(model))
        configurations = []
        batches = []
        for remove_padding in (False, True):
            cfg = build_engine_configs(str(model_dir), remove_padding)
            if cfg["model_config"].hf_config.model_type != "qwen3_5_moe_text":
                raise AssertionError("installed HF configuration selected a non-Text root")
            if cfg["model_config"].hf_config._experts_implementation != "eager":
                raise AssertionError("HFModelConfig did not consume the eager expert override")
            td = make_batch(synthetic_batch(0)[:2], use_remove_padding=remove_padding)
            micro_batches, indices = prepare_micro_batches(td)
            if indices is not None or len(micro_batches) != 2 or any(len(m) != 1 for m in micro_batches):
                raise AssertionError("installed microbatch splitter does not preserve one sequence per micro")
            if any(tu.get_non_tensor_data(m, "use_remove_padding", default=None) is not remove_padding for m in micro_batches):
                raise AssertionError("TensorDict lost its non-tensor arm control")
            configurations.append({"use_remove_padding": remove_padding,
                                   "model_class": cfg["model_config"].hf_config.architectures,
                                   "fsdp_size": cfg["engine_config"].fsdp_size,
                                   "mixed_precision": cfg["engine_config"].mixed_precision,
                                   "experts_implementation": cfg["model_config"].hf_config._experts_implementation,
                                   "optimizer_override": cfg["optimizer_config"].override_optimizer_config})
            batches.append({"use_remove_padding": remove_padding, "micro_rows": [len(m) for m in micro_batches],
                            "local_tokens": td["loss_mask"].values().sum().item()})
        methods = {name: _callable_record(getattr(FSDPEngineWithLMHead, name)) for name in (
            "initialize", "forward_backward_batch", "optimizer_zero_grad", "optimizer_step",
            "lr_scheduler_step", "save_checkpoint", "load_checkpoint", "train_mode")}
        required = {"forward_backward_batch": {"data", "loss_function", "forward_only"},
                    "save_checkpoint": {"local_path", "global_step"},
                    "load_checkpoint": {"local_path", "del_local_after_load"}}
        signature_ok = all(names <= set(inspect.signature(getattr(FSDPEngineWithLMHead, method)).parameters)
                           for method, names in required.items())
        registry_ok = (EngineRegistry._engines.get("language_model", {}).get("fsdp2", {}).get("cuda")
                       is FSDPEngineWithLMHead)
        cli = get_args_parser().parse_args(launcher_command(output / "gpu")[3:])
        interfaces = {"versions": {name: version(name) for name in ("torch", "transformers", "verl", "tensordict")},
                      "nccl_compiled": torch.distributed.is_nccl_available(),
                      "engine_registry_exact": registry_ok, "engine_signatures_ok": signature_ok,
                      "engine_methods": methods, "native_text_root": _callable_record(type(model).forward),
                      "hf_model_loader": _callable_record(type(model).from_pretrained),
                      "hf_experts_model_setter": _callable_record(type(model).set_experts_implementation),
                      "hf_experts_config_setter": _callable_record(type(model.config)._experts_implementation.fset),
                      "torchrun_parser_ok": str(cli.nproc_per_node) == "2" and cli.max_restarts == 0 and cli.module,
                      "cuda_initialized": torch.cuda.is_initialized()}
        rows = synthetic_batch(0)
        model.train()
        reference = independent_reference(model, rows)
        reference["loss"].backward()
        gradients = capture_gradients(model)
        local_gradients = []
        for rank in range(WORLD_SIZE):
            local = build_tiny_model().train()
            local_ref = independent_reference(local, rows[rank * 2:rank * 2 + 2])
            scaled = local_ref["loss"] * local_ref["tokens"] * loss_scale(reference["tokens"], WORLD_SIZE)
            scaled.backward()
            local_gradients.append(capture_gradients(local))
        averaged = {key: sum(grads[key] for grads in local_gradients) / WORLD_SIZE for key in gradients}
        formula_check = compare_gradients(gradients, averaged)
        finite_loss = math.isfinite(reference["loss"].item())
        nonzero_gradient = sum(t.double().square().sum().item() for t in gradients.values()) > 0
        _save_tensor_once(output / "cpu_reference.pt", {"gradients": gradients, "averaged_local_gradients": averaged,
                                                       "log_probs": [lp.detach() for lp in reference["log_probs"]]})
        result.update(interfaces=interfaces, configurations=configurations, tensor_batches=batches,
                      tiny_model={"path": str(model_dir), "files": _file_hashes(model_dir)},
                      cpu_reference={"loss": reference["loss"].item(), "finite": finite_loss,
                                     "nonzero_gradient": nonzero_gradient, "gradient_formula": formula_check,
                                     "disk_reload_exact": disk_reload_check},
                      batches=[{"sha256": batch_digest(synthetic_batch(step)), **validate_batch(synthetic_batch(step))}
                               for step in (0, 1)])
        result["health_ok"] = bool(signature_ok and registry_ok and interfaces["nccl_compiled"]
                                   and interfaces["torchrun_parser_ok"] and not interfaces["cuda_initialized"]
                                   and finite_loss and nonzero_gradient and formula_check["ok"] and disk_reload_check["ok"])
    return _json_finite(result)


def _capture_model(model) -> dict:
    return {name: _full_cpu(value) for name, value in model.state_dict().items()}


def _capture_optimizer(model, optimizer) -> dict:
    import torch

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    states = {}
    for parameter, state in optimizer.state.items():
        states[names[id(parameter)]] = {
            key: _full_cpu(value) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in state.items()}
    groups = [{key: [names[id(p)] for p in value] if key == "params" else copy.deepcopy(value)
               for key, value in group.items()} for group in optimizer.param_groups]
    return {"states": states, "param_groups": groups}


def _capture_state(engine) -> dict:
    return {"model": _capture_model(engine.module),
            "optimizer": _capture_optimizer(engine.module, engine.optimizer),
            "scheduler": copy.deepcopy(engine.lr_scheduler.state_dict()),
            "rng": copy.deepcopy(engine.checkpoint_manager.get_rng_state())}


def same_state_optimizer_reference(parameters: dict, named_optimizer: dict, gradients: dict, *, step: int) -> dict:
    """One independent CPU AdamW transition from immutable, named actual inputs."""
    import torch
    from types import SimpleNamespace

    if type(step) is not int or step not in (1, 2) or not parameters or parameters.keys() != gradients.keys():
        raise ValueError("registered step and exact parameter/gradient names are required")
    frozen_inputs = copy.deepcopy((parameters, named_optimizer, gradients))
    groups, states = named_optimizer["param_groups"], named_optimizer["states"]
    names = [name for group in groups for name in group["params"]]
    if len(names) != len(set(names)) or set(names) != set(parameters):
        raise ValueError("optimizer groups must cover every named parameter exactly once")
    required = {"lr": LEARNING_RATE, "betas": (0.9, 0.999), "eps": 1e-8,
                "weight_decay": 0.0, "foreach": False, "fused": False, "amsgrad": False,
                "maximize": False, "capturable": False, "differentiable": False}
    for group in groups:
        if any(group.get(key) != value for key, value in required.items()):
            raise ValueError("actual optimizer options differ from the registered AdamW")
        if group.get("decoupled_weight_decay", True) is not True or group.get("initial_lr", LEARNING_RATE) != LEARNING_RATE:
            raise ValueError("actual optimizer decay or initial learning rate changed")
    if set(states) != (set(parameters) if step == 2 else set()):
        raise ValueError("optimizer states do not cover this actual pre-step")
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if (parameter.dtype != torch.float32 or gradient.dtype != parameter.dtype
                or parameter.shape != gradient.shape
                or not torch.isfinite(parameter).all() or not torch.isfinite(gradient).all()):
            raise ValueError(f"invalid parameter/gradient {name}")
        if step == 2:
            state = states[name]
            if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                    or state["step"].numel() != 1 or state["step"].item() != step - 1
                    or any(state[key].shape != parameter.shape or state[key].dtype != parameter.dtype
                           or not torch.isfinite(state[key]).all() for key in ("exp_avg", "exp_avg_sq"))
                    or not torch.isfinite(state["step"]).all() or (state["exp_avg_sq"] < 0).any()):
                raise ValueError(f"invalid optimizer moment/step {name}")
    if step == 2 and not any(torch.count_nonzero(s["exp_avg_sq"]).item() for s in states.values()):
        raise ValueError("step two requires nonzero moments from a real first update")
    cloned = {name: torch.nn.Parameter(_full_cpu(value)) for name, value in parameters.items()}
    mapped_groups = [{key: [cloned[name] for name in value] if key == "params" else copy.deepcopy(value)
                      for key, value in group.items()} for group in groups]
    optimizer = torch.optim.AdamW(mapped_groups, **required)
    for name, state in states.items():
        optimizer.state[cloned[name]] = copy.deepcopy(state)
    for name, parameter in cloned.items():
        parameter.grad = _full_cpu(gradients[name])
    named_model = SimpleNamespace(named_parameters=lambda: cloned.items())
    input_check = compare_exact_trees(
        frozen_inputs, ({name: _full_cpu(p) for name, p in cloned.items()},
                        _capture_optimizer(named_model, optimizer), capture_gradients(named_model)))
    if not input_check["ok"]:
        raise AssertionError("CPU AdamW did not start from the exact actual inputs")
    optimizer.step()
    inputs_unchanged = compare_exact_trees(frozen_inputs, (parameters, named_optimizer, gradients))
    if not inputs_unchanged["ok"]:
        raise AssertionError("CPU AdamW mutated an input snapshot")
    return {"parameters": {name: _full_cpu(p) for name, p in cloned.items()},
            "optimizer": _capture_optimizer(named_model, optimizer),
            "input_check": input_check, "inputs_unchanged": inputs_unchanged}


def _bounded_tensor_comparison(reference, actual, bound) -> dict:
    import torch

    metrics = _tensor_metrics(reference, actual)
    metrics.pop("max_abs_bound")  # This was a gradient bound, not this comparison's bound.
    valid = metrics["finite"] and metrics["same_shape"] and reference.dtype == actual.dtype
    if valid:
        error = (actual.double() - reference.double()).abs()
        ratio = torch.where(bound > 0, error / bound,
                            torch.where(error == 0, 0.0, float("inf")))
        maximum = ratio.max().item() if ratio.numel() else 0.0
        bad = int((error > bound).sum().item())
    else:
        maximum, bad = None, None
    metrics.update(ok=bool(valid and bad == 0), max_error_ratio=maximum, bad_coordinates=bad)
    return metrics


def compare_optimizer_transition(before: dict, gradients: dict, expected: dict, actual: dict) -> dict:
    """Moment rounding bounds are separate from parameter and gradient bounds."""
    import torch

    same_names = bool(expected["states"] and expected["states"].keys() == actual["states"].keys() == gradients.keys())
    metadata = compare_exact_trees(
        {"groups": expected["param_groups"], "steps": {n: s["step"] for n, s in expected["states"].items()}},
        {"groups": actual["param_groups"], "steps": {n: s.get("step") for n, s in actual["states"].items()}})
    moments = {}
    for name in sorted(expected["states"].keys() & actual["states"].keys() & gradients.keys()):
        if set(expected["states"][name]) != set(actual["states"][name]):
            same_names = False
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in actual["states"][name]:
                same_names = False
                continue
            ref, observed = expected["states"][name][key], actual["states"][name][key]
            old = before["states"].get(name, {}).get(key, torch.zeros_like(ref))
            gradient = gradients[name].double()
            term = gradient if key == "exp_avg" else gradient.square()
            scale = torch.maximum(torch.maximum(old.double().abs(), term.abs()), ref.double().abs())
            bound = 8 * torch.finfo(torch.float32).eps * scale
            moments[f"{name}/{key}"] = _bounded_tensor_comparison(ref, observed, bound)
    return {"ok": bool(same_names and metadata["ok"] and moments and all(v["ok"] for v in moments.values())),
            "same_names": same_names, "metadata_exact": metadata, "moments": moments,
            "error_scale": preregistered_plan()["moment_error_scale"]}


def _training_state(state: dict) -> dict:
    return {key: state[key] for key in ("model", "optimizer", "scheduler")}


def _cross_rank_check(value, engine) -> dict:
    import torch.distributed as dist
    from syncopate.train.policy_observer import tensor_fingerprint
    import torch

    def serializable(item):
        if isinstance(item, torch.Tensor):
            return tensor_fingerprint(item)
        if isinstance(item, dict):
            return {key: serializable(v) for key, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [serializable(v) for v in item]
        return item

    members = dist.get_process_group_ranks(engine.get_data_parallel_group())
    digest = hashlib.sha256(json.dumps(serializable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()
    hashes = [None] * WORLD_SIZE
    dist.all_gather_object(hashes, digest)
    return {"ok": members == [0, 1] and len(set(hashes)) == 1, "dp_members": members, "sha256_by_rank": hashes}


def _compare_parameters(reference: dict, actual: dict) -> dict:
    import torch

    per_parameter = {}
    for name in sorted(reference.keys() & actual.keys()):
        ref, observed = reference[name], actual[name]
        per_parameter[name] = _bounded_tensor_comparison(ref, observed, PARAMETER_ATOL + PARAMETER_RTOL * ref.double().abs())
    return {"ok": bool(reference and reference.keys() == actual.keys()
                       and all(value["ok"] for value in per_parameter.values())),
            "atol": PARAMETER_ATOL, "rtol": PARAMETER_RTOL, "per_parameter": per_parameter,
            "missing": sorted(reference.keys() - actual.keys()), "extra": sorted(actual.keys() - reference.keys())}


class RootObservation:
    """Read-only hooks; never replace model/GDN/router forward implementations."""

    def __init__(self, engine):
        self.engine = engine
        self.first_root = None
        self.root_calls = 0
        self.routers = []
        self.handles = [engine.module.register_forward_pre_hook(self._root, with_kwargs=True)]
        self.gdn_modules = [name for name, module in engine.module.named_modules()
                            if "GatedDeltaNet" in type(module).__name__]
        for name, module in engine.module.named_modules():
            if "TopKRouter" in type(module).__name__:
                self.handles.append(module.register_forward_hook(self._router_hook(name)))

    def _root(self, module, args, kwargs):
        self.root_calls += 1
        if self.first_root is None:
            self.first_root = {"class": f"{type(module).__module__}.{type(module).__name__}",
                               "class_mro": [cls.__name__ for cls in type(module).__mro__],
                               "forward": _callable_record(module.forward),
                               "pass_packed_cu_seqlens": self.engine.pass_packed_cu_seqlens,
                               "experts_implementation": module.config._experts_implementation,
                               "attention_implementation": module.config._attn_implementation,
                               "input_ids": kwargs["input_ids"].detach().cpu().tolist(),
                               "position_ids": kwargs["position_ids"].detach().cpu().tolist(),
                               "kwargs": sorted(kwargs),
                               "boundary_arguments": {key: (kwargs[key].detach().cpu().tolist()
                                                           if hasattr(kwargs.get(key), "detach") else kwargs.get(key))
                                                      for key in ("seq_idx", "cu_seq_lens_q", "cu_seqlens")}}

    def _router_hook(self, name):
        def record(module, args, output):
            # Capture only the first real root interval, before gradient work.
            if self.root_calls == 1:
                import torch

                logits, weights, indices = output
                self.routers.append({"module": name, "class": type(module).__name__,
                                     "finite": bool(torch.isfinite(logits).all() and torch.isfinite(weights).all()),
                                     "router_ids": indices.detach().cpu().tolist(),
                                     "router_weights": weights.detach().cpu().tolist()})
        return record

    def result(self) -> dict:
        return {"first_actual_root": self.first_root, "actual_root_calls": self.root_calls,
                "first_root_routers": self.routers, "gdn_modules": self.gdn_modules,
                "gdn_calls": 0, "gdn_measured": False,
                "ok": bool(self.first_root and self.root_calls > 0 and len(self.routers) == 2
                           and all(record["finite"] for record in self.routers) and not self.gdn_modules)}

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def observed_root(engine, path: Path):
    """Keep the actual first call even when a subsequent numerical gate fails."""
    observation = RootObservation(engine)
    try:
        yield observation
    finally:
        try:
            write_json_once(path, _json_finite(observation.result()))
        finally:
            observation.close()


def _collective_require(ok: bool, reason: str) -> None:
    import torch
    import torch.distributed as dist

    flag = torch.tensor(int(bool(ok)), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() != WORLD_SIZE:
        raise AssertionError(reason)


def _new_engine(tiny_dir: Path, remove_padding: bool):
    from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead

    engine = EngineRegistry.new(model_type="language_model", backend="fsdp2",
                                **build_engine_configs(str(tiny_dir), remove_padding))
    if type(engine) is not FSDPEngineWithLMHead:
        raise AssertionError("factory did not return the registered official FSDPEngineWithLMHead")
    engine.initialize()
    if engine._autocast_dtype.__str__() != "torch.float32" or engine.get_data_parallel_size() != 2:
        raise AssertionError("engine did not consume FP32 / DP=2 configuration")
    if (engine.module.config._experts_implementation != "eager"
            or engine.module.config._attn_implementation != "eager"):
        raise AssertionError("engine did not consume eager attention and experts")
    if (engine.module.config.model_type != "qwen3_5_moe_text" or engine._is_lora
            or "Qwen3_5MoeForCausalLM" not in {cls.__name__ for cls in type(engine.module).__mro__}):
        raise AssertionError("engine root is not full-parameter Qwen3.5 Text MoE")
    return engine


def _native_reference_from_weights(tiny_dir: Path):
    import torch
    from transformers import Qwen3_5MoeForCausalLM

    model = Qwen3_5MoeForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32,
                                                attn_implementation="eager", experts_implementation="eager").cpu().train()
    if model.config._experts_implementation != "eager":
        raise AssertionError("native CPU reference did not consume eager experts")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0.0, foreach=False, fused=False)
    return model, optimizer


def _engine_step(engine, reference_model, reference_optimizer, *, step: int, rank: int,
                 remove_padding: bool, directory: Path, label: str, expected_before: dict | None = None) -> dict:
    import torch
    from verl.utils import tensordict_utils as tu

    rows = synthetic_batch(step)
    total_tokens = validate_batch(rows)["tokens"]
    local_rows = rows[rank * 2:rank * 2 + 2]
    engine_before = _capture_state(engine)
    before, optimizer_before = engine_before["model"], engine_before["optimizer"]
    _collective_require(step == 0 or expected_before is not None, "second step requires the actual first-step checkpoint state")
    continuity = compare_exact_trees(_training_state(engine_before if expected_before is None else expected_before),
                                    _training_state(engine_before))
    reference_model.load_state_dict(before)
    same_start = compare_exact_trees(before, _capture_model(reference_model))
    _collective_require(continuity["ok"] and same_start["ok"], "step start changed or CPU reference did not load exact weights")
    reference_optimizer.zero_grad(set_to_none=True)
    reference = independent_reference(reference_model, rows)
    reference["loss"].backward()
    reference_gradients = capture_gradients(reference_model)
    reference_lp = {row["row_id"]: lp.detach().cpu() for row, lp in zip(rows, reference["log_probs"])}
    callback_records = []
    actual_lp = {}

    def token_loss(*, model_output, data, dp_group):
        if len(data) != 1:
            raise AssertionError("each actual microbatch must contain exactly one sequence")
        actual_tokens = tu.get_non_tensor_data(data, "batch_num_tokens", default=None)
        actual_dp_size = tu.get_non_tensor_data(data, "dp_size", default=None)
        row_id = int(data["row_id"].item())
        if actual_tokens != total_tokens or actual_dp_size != WORLD_SIZE:
            raise AssertionError("root engine did not supply the registered global token denominator")
        if row_id not in {row["row_id"] for row in local_rows}:
            raise AssertionError("engine callback saw another rank's input identity")
        log_probs = model_output["log_probs"].values()
        mask = data["loss_mask"].values()
        loss = -(log_probs * mask).sum() * loss_scale(actual_tokens, actual_dp_size)
        observed = log_probs.detach().cpu()
        expected = reference_lp[row_id]
        logprob_ok = bool(observed.shape == expected.shape and torch.isfinite(observed).all()
                          and torch.isfinite(expected).all()
                          and torch.allclose(observed, expected, atol=LOGPROB_ATOL, rtol=LOGPROB_RTOL))
        expected_loss = -(expected * torch.tensor(rows[row_id]["loss_mask"])).sum() * loss_scale(total_tokens, WORLD_SIZE)
        loss_ok = bool(torch.isfinite(loss).item() and torch.isclose(
            loss.detach().cpu(), expected_loss, atol=LOGPROB_ATOL, rtol=LOGPROB_RTOL))
        actual_lp[row_id] = observed.clone()
        callback_records.append({"row_id": row_id, "global_batch_num_tokens": actual_tokens,
                                 "dp_size": actual_dp_size, "local_supervised_tokens": int(mask.sum().item()),
                                 "logprob_ok": logprob_ok, "loss_ok": loss_ok,
                                 "loss": loss.detach().item(), "reference_scaled_micro_loss": expected_loss.item(),
                                 "logprob_max_abs_error": (observed - expected).abs().max().item()
                                 if observed.shape == expected.shape else None,
                                 "logprob_max_abs_reference": expected.abs().max().item(),
                                 "logprob_max_error_ratio": ((observed - expected).abs() /
                                     (LOGPROB_ATOL + LOGPROB_RTOL * expected.abs())).max().item()
                                 if observed.shape == expected.shape else None})
        return loss, {"b03_scaled_micro_loss": loss.detach().item()}

    scheduler_before = engine_before["scheduler"]
    with engine.train_mode(zero_grad_on_exit=False):
        engine.optimizer_zero_grad()
        engine.forward_backward_batch(make_batch(local_rows, use_remove_padding=remove_padding), token_loss, False)
        # Every rank gathers the full gradient before the official clip/step.
        actual_gradients = capture_gradients(engine.module)
        gradient_check = compare_gradients(reference_gradients, actual_gradients)
        gradients_nonzero = sum(g.double().square().sum().item() for g in actual_gradients.values()) > 0
        callbacks_ok = (len(callback_records) == 2
                        and {record["row_id"] for record in callback_records} == {row["row_id"] for row in local_rows}
                        and all(record["logprob_ok"] and record["loss_ok"] for record in callback_records))
        pre_step = {"step": step + 1, "gradient_check": gradient_check,
                    "gradient_nonzero": gradients_nonzero, "callbacks": callback_records,
                    "callback_checks_ok": callbacks_ok, "reference_loss": reference["loss"].item(),
                    "global_batch_sha256": batch_digest(rows), "same_start_exact": same_start,
                    "previous_step_continuity": continuity}
        _save_tensor_once(directory / f"{label}_pre_clip.pt", {
            "actual_gradients": actual_gradients, "reference_gradients": reference_gradients,
            "actual_log_probs": actual_lp, "reference_log_probs": reference_lp, "parameters_before": before,
            "optimizer_before": optimizer_before, "scheduler_before": scheduler_before})
        write_json_once(directory / f"{label}_pre_clip.json", _json_finite(pre_step))
        _collective_require(gradient_check["ok"] and gradients_nonzero and callbacks_ok,
                            "pre-clip independent gradient/logprob/token-mean check failed; no update accepted")
        same_gradient_reference = same_state_optimizer_reference(before, optimizer_before, actual_gradients, step=step + 1)
        independent_gradient_reference = same_state_optimizer_reference(before, optimizer_before, reference_gradients, step=step + 1)
        reference_preserves_actual = compare_exact_trees(engine_before, _capture_state(engine))
        rank_before = _cross_rank_check({"state": _training_state(engine_before), "gradients": actual_gradients}, engine)
        _save_tensor_once(directory / f"{label}_optimizer_controls.pt", {
            "same_gradient": same_gradient_reference, "independent_gradient": independent_gradient_reference})
        write_json_once(directory / f"{label}_optimizer_inputs.json", {
            "reference_preserves_actual": reference_preserves_actual, "rank_before": rank_before,
            "same_gradient_input": same_gradient_reference["input_check"],
            "independent_gradient_input": independent_gradient_reference["input_check"]})
        _collective_require(reference_preserves_actual["ok"] and rank_before["ok"],
                            "reference calculation modified actual state or full rank states/gradients disagree")
        grad_norm = engine.optimizer_step()
        learning_rate = engine.lr_scheduler_step()
    actual_after = _capture_model(engine.module)
    expected_after = same_gradient_reference["parameters"]
    parameter_check = _compare_parameters(expected_after, actual_after)
    independent_gradient_sidecheck = _compare_parameters(independent_gradient_reference["parameters"], actual_after)
    update_l2 = math.sqrt(sum((actual_after[name].double() - before[name].double()).square().sum().item()
                              for name in before))
    optimizer_state = _capture_optimizer(engine.module, engine.optimizer)
    optimizer_check = compare_optimizer_transition(optimizer_before, actual_gradients,
                                                   same_gradient_reference["optimizer"], optimizer_state)
    optimizer_steps = {name: float(state["step"].item()) for name, state in optimizer_state["states"].items()}
    optimizer_ok = bool(optimizer_steps and all(value == step + 1 for value in optimizer_steps.values())
                        and all({"exp_avg", "exp_avg_sq", "step"} <= state.keys()
                                and all(torch.isfinite(state[key]).all().item()
                                        for key in ("exp_avg", "exp_avg_sq", "step"))
                                for state in optimizer_state["states"].values()))
    scheduler_after = copy.deepcopy(engine.lr_scheduler.state_dict())
    rank_after = _cross_rank_check({"model": actual_after, "optimizer": optimizer_state, "scheduler": scheduler_after}, engine)
    result = {**pre_step, "parameter_check": parameter_check, "gradient_norm_before_clip": grad_norm,
              "independent_gradient_sidecheck": independent_gradient_sidecheck,
              "optimizer_check": optimizer_check, "reference_preserves_actual": reference_preserves_actual,
              "same_gradient_input": same_gradient_reference["input_check"],
              "same_gradient_inputs_unchanged": same_gradient_reference["inputs_unchanged"],
              "independent_gradient_input": independent_gradient_reference["input_check"],
              "independent_gradient_inputs_unchanged": independent_gradient_reference["inputs_unchanged"],
              "rank_before": rank_before, "rank_after": rank_after,
              "clip_threshold": CLIP_GRAD, "clipping_inactive": math.isfinite(grad_norm) and 0 < grad_norm < CLIP_GRAD,
              "parameter_update_l2": update_l2, "parameter_update_nonzero": math.isfinite(update_l2) and update_l2 > 0,
              "optimizer_steps": optimizer_steps, "optimizer_state_ok": optimizer_ok,
              "learning_rate": learning_rate, "scheduler": scheduler_after,
              "scheduler_advanced_once": scheduler_advanced(scheduler_before, scheduler_after)}
    result["ok"] = bool(parameter_check["ok"] and optimizer_check["ok"] and rank_after["ok"] and result["clipping_inactive"]
                        and result["parameter_update_nonzero"] and optimizer_ok and math.isfinite(learning_rate)
                        and result["scheduler_advanced_once"])
    _save_tensor_once(directory / f"{label}_post_step.pt", {
        "actual_parameters": actual_after, "reference_parameters": expected_after,
        "actual_optimizer": optimizer_state, "reference_optimizer": same_gradient_reference["optimizer"],
        "independent_gradient_reference": independent_gradient_reference})
    write_json_once(directory / f"{label}_post_step.json", _json_finite(result))
    _collective_require(result["ok"], "post-step parameter/optimizer/clip check failed")
    return result


def _rng_draws() -> dict:
    import numpy as np
    import torch

    return {"python": [random.random() for _ in range(4)], "numpy": np.random.random(4),
            "cpu": torch.rand(4, device="cpu"), "cuda": torch.rand(4, device="cuda").cpu()}


def _run_arm(tiny_dir: Path, output: Path, rank: int, remove_padding: bool) -> dict:
    import gc
    import torch
    import torch.distributed as dist

    arm = "remove_padding_on" if remove_padding else "remove_padding_off"
    directory = output / f"rank_{rank}" / arm
    directory.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints" / arm
    engine = _new_engine(tiny_dir, remove_padding)
    reference, reference_optimizer = _native_reference_from_weights(tiny_dir)
    initial = _capture_model(engine.module)
    reference_initial = _capture_model(reference)
    initial_check = compare_exact_trees(reference_initial, initial)
    _save_tensor_once(directory / "initial_weights.pt", {"engine": initial, "reference": reference_initial})
    write_json_once(directory / "initial_weights.json", initial_check)
    _collective_require(initial_check["ok"], "engine and independent native reference did not load identical initial weights")
    with observed_root(engine, directory / "baseline_root.json") as observation:
        first = _engine_step(engine, reference, reference_optimizer, step=0, rank=rank,
                             remove_padding=remove_padding, directory=directory, label="baseline_step1")
        saved_state = _capture_state(engine)
        _save_tensor_once(directory / "state_before_save.pt", saved_state)
        _collective_require(not checkpoint_dir.exists(), f"refusing to overwrite checkpoint {checkpoint_dir}")
        # The official manager writes rank-specific shards and rank-0 HF config.
        engine.save_checkpoint(str(checkpoint_dir), global_step=1, max_ckpt_to_keep=None)
        dist.barrier()
        after_save = _capture_state(engine)
        save_stability = compare_exact_trees(saved_state, after_save)
        _save_tensor_once(directory / "state_after_save.pt", after_save)
        manifest = _file_hashes(checkpoint_dir)
        expected_shards = {f"{kind}_world_size_2_rank_{r}.pt"
                           for kind in ("model", "optim", "extra_state") for r in (0, 1)}
        checkpoint_files_ok = expected_shards <= manifest.keys()
        write_json_once(directory / "save_stability.json", {"state_unchanged": save_stability,
            "checkpoint_manifest": manifest, "checkpoint_files_ok": checkpoint_files_ok})
        _collective_require(save_stability["ok"] and checkpoint_files_ok,
                            "normal checkpoint changed state or omitted model/optimizer/extra shards")
        baseline_draws = _rng_draws()
        second = _engine_step(engine, reference, reference_optimizer, step=1, rank=rank,
                              remove_padding=remove_padding, directory=directory, label="baseline_step2", expected_before=saved_state)
        baseline_final = _capture_state(engine)
        root_evidence = observation.result()
        _save_tensor_once(directory / "baseline_final.pt", {"state": baseline_final, "rng_draws": baseline_draws})
    del engine, observation, reference, reference_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    fresh = _new_engine(tiny_dir, remove_padding)
    fresh_initial = _capture_model(fresh.module)
    fresh_initial_check = compare_exact_trees(initial, fresh_initial)
    _save_tensor_once(directory / "fresh_initial_weights.pt", fresh_initial)
    write_json_once(directory / "fresh_initial_weights.json", fresh_initial_check)
    _collective_require(fresh_initial_check["ok"], "fresh engine is not initialized from the original tiny weights")
    resumed_reference, resumed_optimizer = _native_reference_from_weights(tiny_dir)
    # Reconstruct the independent oracle before restoring RNG: model creation
    # must not consume the checkpoint's resumed random stream.
    # Preserve checkpoint artifacts: normal restart, never failure injection.
    fresh.load_checkpoint(str(checkpoint_dir), del_local_after_load=False)
    loaded = _capture_state(fresh)
    load_check = compare_exact_trees(saved_state, loaded)
    resumed_draws = _rng_draws()
    rng_check = compare_exact_trees(baseline_draws, resumed_draws)
    _save_tensor_once(directory / "restored_state.pt", {"state": loaded, "rng_draws": resumed_draws})
    write_json_once(directory / "restart_pre_step.json", {"load_state": load_check, "rng_draws": rng_check})
    _collective_require(load_check["ok"] and rng_check["ok"], "normal checkpoint restore is not exactly equal")
    with observed_root(fresh, directory / "restored_root.json") as resumed_observation:
        resumed = _engine_step(fresh, resumed_reference, resumed_optimizer, step=1, rank=rank,
                               remove_padding=remove_padding, directory=directory, label="restored_step2", expected_before=saved_state)
        restored_final = _capture_state(fresh)
        next_step_check = compare_exact_trees(baseline_final, restored_final)
        preserved_files = _file_hashes(checkpoint_dir) == manifest
        resumed_root = resumed_observation.result()
        _save_tensor_once(directory / "restored_final.pt", restored_final)
    result = {"use_remove_padding": remove_padding, "initial_weights": initial_check,
              "fresh_initial_weights": fresh_initial_check, "baseline_steps": [first, second],
              "restored_next_step": resumed, "save_state_unchanged": save_stability,
              "load_state_exact": load_check, "rng_draws_exact": rng_check, "next_step_exact": next_step_check,
              "checkpoint_path": str(checkpoint_dir), "checkpoint_manifest": manifest,
              "checkpoint_files_preserved": preserved_files,
              "root_observation": root_evidence, "restored_root_observation": resumed_root}
    result["ok"] = bool(all(item["ok"] for item in (first, second, resumed, load_check, rng_check, next_step_check,
                                                    save_stability, root_evidence, resumed_root)) and preserved_files)
    write_json_once(directory / "result.json", _json_finite(result))
    _collective_require(result["ok"], "checkpoint next-step or actual root observation check failed")
    del fresh, resumed_observation, resumed_reference, resumed_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_gpu_worker(output: Path) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    result = {"mode": "gpu_worker", "rank": rank, "health_ok": False, "arms": [],
              "source_sha256": source_sha256(), "plan": preregistered_plan()}
    try:
        cpu = require_cpu_evidence(output)
        if (int(os.environ["WORLD_SIZE"]) != 2 or int(os.environ["LOCAL_WORLD_SIZE"]) != 2
                or int(os.environ["LOCAL_RANK"]) != rank or rank not in (0, 1)
                or torch.cuda.device_count() != 2):
            raise ValueError("exactly two local ranks on exactly two visible B200 GPUs are required")
        torch.cuda.set_device(rank)
        if any("B200" not in torch.cuda.get_device_name(device) for device in (0, 1)):
            raise ValueError("this probe is registered only for B200")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        dist.init_process_group("nccl", timeout=timedelta(seconds=DISTRIBUTED_TIMEOUT_SECONDS),
                                device_id=torch.device("cuda", rank))
        props = torch.cuda.get_device_properties(rank)
        result["device"] = {"name": props.name, "uuid": str(props.uuid),
                            "capability": [props.major, props.minor], "total_memory_bytes": props.total_memory}
        result["determinism"] = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                 "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                 "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        for remove_padding in (False, True):
            result["arms"].append(_run_arm(Path(cpu["tiny_model"]["path"]), output, rank, remove_padding))
        result["health_ok"] = len(result["arms"]) == 2 and all(arm["ok"] for arm in result["arms"])
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            result["health_ok"] = False
            result["teardown_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json_once(output / f"rank_{rank}.json", _json_finite(result))
    return 0 if result["health_ok"] else 1


def validate_rank_results(ranks: list[dict], native_root: dict) -> dict:
    """Recheck identities, coverage and numerical records, not a worker's label."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def exact(check):
        return check['ok'] is True and check['equality'] == 'exact' and check['mismatches'] == []

    def parameter_metrics(metrics, atol, rtol):
        require(bool(metrics), 'missing per-parameter measurements')
        for value in metrics.values():
            require(value['finite'] is True and value['same_shape'] is True
                    and all(finite(value[key]) and value[key] >= 0 for key in (
                        'error_l2', 'reference_l2', 'max_abs_error', 'max_abs_reference'))
                    and value['max_abs_error'] <= atol + rtol * value['max_abs_reference'],
                    'parameter measurement failed')

    def coordinate_metrics(metrics, *, require_pass=True):
        require(bool(metrics), 'missing coordinate measurements')
        for value in metrics.values():
            require(type(value['ok']) is bool and value['finite'] is True and value['same_shape'] is True
                    and all(finite(value[key]) and value[key] >= 0 for key in (
                        'error_l2', 'reference_l2', 'max_abs_error', 'max_abs_reference', 'max_error_ratio'))
                    and type(value['bad_coordinates']) is int and value['bad_coordinates'] >= 0
                    and value['ok'] == (value['max_error_ratio'] <= 1 and value['bad_coordinates'] == 0)
                    and (not require_pass or value['ok']),
                    'coordinate error bound failed')

    cross_rank_reports = {}

    def step_check(step, expected_step, rank, slot):
        require(step['ok'] is True and type(step['step']) is int and step['step'] == expected_step,
                'missing or wrong optimizer step')
        rows = synthetic_batch(expected_step - 1)
        require(step['global_batch_sha256'] == batch_digest(rows), 'wrong batch identity')
        require(all(step[key] is True for key in ('gradient_nonzero', 'callback_checks_ok', 'clipping_inactive',
                'parameter_update_nonzero', 'optimizer_state_ok', 'scheduler_advanced_once')), 'step checks failed')
        gradient = step['gradient_check']
        require(gradient['ok'] is True and gradient['missing'] == gradient['extra'] == [], 'gradient keys failed')
        parameter_metrics(gradient['per_parameter'], GRADIENT_ATOL, GRADIENT_RTOL)
        metrics = gradient['per_parameter'].values()
        error = math.sqrt(sum(value['error_l2'] ** 2 for value in metrics))
        reference = math.sqrt(sum(value['reference_l2'] ** 2 for value in metrics))
        require(reference > 0 and error / reference <= GRADIENT_RELATIVE_L2_MAX
                and gradient['global']['error_l2'] == error
                and gradient['global']['reference_l2'] == reference, 'global gradient measurement failed')
        parameters = step['parameter_check']
        require(parameters['ok'] is True and parameters['missing'] == parameters['extra'] == []
                and set(parameters['per_parameter']) == set(gradient['per_parameter'])
                and parameters['atol'] == PARAMETER_ATOL and parameters['rtol'] == PARAMETER_RTOL,
                'parameter comparison failed')
        parameter_metrics(parameters['per_parameter'], PARAMETER_ATOL, PARAMETER_RTOL)
        coordinate_metrics(parameters['per_parameter'])
        side = step['independent_gradient_sidecheck']
        require(type(side['ok']) is bool and side['missing'] == side['extra'] == []
                and set(side['per_parameter']) == set(gradient['per_parameter'])
                and side['atol'] == PARAMETER_ATOL and side['rtol'] == PARAMETER_RTOL,
                'independent-gradient side evidence missing')
        coordinate_metrics(side['per_parameter'], require_pass=False)
        require(side['ok'] == all(value['ok'] for value in side['per_parameter'].values()), 'contradictory side evidence')
        require(all(exact(step[key]) for key in ('same_start_exact', 'previous_step_continuity',
                'reference_preserves_actual', 'same_gradient_input', 'same_gradient_inputs_unchanged',
                'independent_gradient_input', 'independent_gradient_inputs_unchanged')), 'optimizer/reference inputs changed')
        optimizer = step['optimizer_check']
        require(optimizer['ok'] is True and optimizer['same_names'] is True and exact(optimizer['metadata_exact'])
                and optimizer['error_scale'] == preregistered_plan()['moment_error_scale']
                and set(optimizer['moments']) == {f'{name}/{key}' for name in gradient['per_parameter']
                                                  for key in ('exp_avg', 'exp_avg_sq')}, 'optimizer transition incomplete')
        coordinate_metrics(optimizer['moments'])
        for key in ('rank_before', 'rank_after'):
            check = step[key]
            require(check['ok'] is True and check['dp_members'] == [0, 1]
                    and len(check['sha256_by_rank']) == WORLD_SIZE and len(set(check['sha256_by_rank'])) == 1
                    and all(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) for sha in check['sha256_by_rank']),
                    'full cross-rank identity failed')
            recorded = cross_rank_reports.setdefault((*slot, key), check['sha256_by_rank'])
            require(recorded == check['sha256_by_rank'], 'rank reports disagree about full-state identity')
        require(finite(step['gradient_norm_before_clip']) and 0 < step['gradient_norm_before_clip'] < CLIP_GRAD
                and step['clip_threshold'] == CLIP_GRAD
                and finite(step['parameter_update_l2']) and step['parameter_update_l2'] > 0
                and finite(step['learning_rate']) and step['learning_rate'] == LEARNING_RATE
                and finite(step['reference_loss']), 'invalid clip/update/loss/lr values')
        require(step['optimizer_steps'] and set(step['optimizer_steps']) == set(gradient['per_parameter'])
                and all(finite(value) and value == expected_step for value in step['optimizer_steps'].values())
                and step['scheduler']['last_epoch'] == expected_step, 'optimizer/scheduler did not advance')
        callbacks = step['callbacks']
        local_rows = {row['row_id']: row for row in rows[rank * 2:rank * 2 + 2]}
        require(len(callbacks) == 2 and {row['row_id'] for row in callbacks} == set(local_rows),
                'missing actual microbatch callback')
        for callback in callbacks:
            require(callback['logprob_ok'] is True and callback['loss_ok'] is True
                    and callback['global_batch_num_tokens'] == validate_batch(rows)['tokens']
                    and callback['dp_size'] == WORLD_SIZE
                    and callback['local_supervised_tokens'] == sum(local_rows[callback['row_id']]['loss_mask'])
                    and finite(callback['logprob_max_abs_error']) and callback['logprob_max_abs_error'] >= 0
                    and finite(callback['logprob_max_abs_reference']) and callback['logprob_max_abs_reference'] >= 0
                    and callback['logprob_max_abs_error'] <= LOGPROB_ATOL + LOGPROB_RTOL * callback['logprob_max_abs_reference']
                    and finite(callback['logprob_max_error_ratio']) and 0 <= callback['logprob_max_error_ratio'] <= 1
                    and finite(callback['loss']) and finite(callback['reference_scaled_micro_loss'])
                    and abs(callback['loss'] - callback['reference_scaled_micro_loss']) <=
                        LOGPROB_ATOL + LOGPROB_RTOL * abs(callback['reference_scaled_micro_loss']),
                    'microbatch numerical or token-denominator check failed')

    def root_check(root, calls, rank, step):
        first = root['first_actual_root']
        expected_row = synthetic_batch(step)[rank * 2]
        require(root['ok'] is True and root['actual_root_calls'] == calls
                and 'Qwen3_5MoeForCausalLM' in first['class_mro']
                and first['experts_implementation'] == first['attention_implementation'] == 'eager'
                and {'input_ids', 'position_ids'} <= set(first['kwargs'])
                and all(first['forward'][key] == native_root[key] for key in ('callable', 'source_sha256'))
                and first['input_ids'] == [expected_row['input_ids']]
                and first['position_ids'] == [expected_row['position_ids']]
                and type(first['pass_packed_cu_seqlens']) is bool
                and set(first['boundary_arguments']) == {'seq_idx', 'cu_seq_lens_q', 'cu_seqlens'}
                and len(root['first_root_routers']) == 2
                and all(item['finite'] is True for item in root['first_root_routers'])
                and root['gdn_modules'] == [] and root['gdn_calls'] == 0 and root['gdn_measured'] is False,
                'actual root/experts observation missing')
        require(len({item['module'] for item in root['first_root_routers']}) == 2, 'duplicate router observation')
        for router in root['first_root_routers']:
            require(len(router['router_ids']) == len(router['router_weights']) == len(expected_row['input_ids']),
                    'router token coverage missing')
            for ids, weights in zip(router['router_ids'], router['router_weights']):
                require(len(ids) == len(weights) == 2 and len(set(ids)) == 2
                        and all(type(value) is int and 0 <= value < 4 for value in ids)
                        and all(finite(value) and 0 <= value <= 1 for value in weights)
                        and abs(sum(weights) - 1) <= LOGPROB_ATOL, 'invalid actual expert IDs/weights')

    try:
        require(isinstance(native_root['callable'], str) and native_root['callable']
                and re.fullmatch('[0-9a-f]{64}', native_root['source_sha256']), 'missing CPU native root identity')
        require(len(ranks) == WORLD_SIZE and {row['rank'] for row in ranks} == {0, 1}, 'missing or repeated rank')
        uuids = set()
        checkpoint_manifests = {}
        for row in ranks:
            rank = row['rank']
            require(type(rank) is int and row['health_ok'] is True and row['mode'] == 'gpu_worker'
                    and row['source_sha256'] == source_sha256() and row['plan'] == preregistered_plan()
                    and 'error' not in row and 'teardown_error' not in row, 'wrong rank identity or failed worker')
            device = row['device']
            require('B200' in device['name'] and device['capability'] == [10, 0]
                    and valid_gpu_uuid(device['uuid']), 'wrong device identity')
            uuids.add(device['uuid'])
            determinism = row['determinism']
            require(determinism['deterministic_algorithms'] is True and determinism['tf32_matmul'] is False
                    and determinism['cublas_workspace_config'] == ':4096:8', 'wrong deterministic settings')
            arms = row['arms']
            require(len(arms) == 2 and [arm['use_remove_padding'] for arm in arms] == [False, True]
                    and all(type(arm['use_remove_padding']) is bool for arm in arms), 'missing arm identity')
            for arm in arms:
                require(arm['ok'] is True and all(exact(arm[key]) for key in ('initial_weights',
                    'fresh_initial_weights', 'save_state_unchanged', 'load_state_exact', 'rng_draws_exact',
                    'next_step_exact')), 'initial/save/restart equality failed')
                require(len(arm['baseline_steps']) == 2, 'missing baseline step')
                for index, step in enumerate(arm['baseline_steps'], 1):
                    step_check(step, index, rank, (arm['use_remove_padding'], 'baseline', index))
                step_check(arm['restored_next_step'], 2, rank, (arm['use_remove_padding'], 'restored', 2))
                root_check(arm['root_observation'], 4, rank, 0)
                root_check(arm['restored_root_observation'], 2, rank, 1)
                manifest = arm['checkpoint_manifest']
                expected = {f'{kind}_world_size_2_rank_{r}.pt'
                            for kind in ('model', 'optim', 'extra_state') for r in (0, 1)}
                require(arm['checkpoint_files_preserved'] is True and expected <= manifest.keys()
                        and all(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha)
                                for sha in manifest.values()), 'missing or changed checkpoint files')
                previous = checkpoint_manifests.setdefault(arm['use_remove_padding'], manifest)
                require(previous == manifest, 'rank checkpoint file hashes disagree')
        require(len(uuids) == WORLD_SIZE, 'ranks did not use two distinct devices')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {'ok': False, 'error': str(exc)}
    return {'ok': True, 'ranks': WORLD_SIZE, 'arms_per_rank': 2, 'baseline_updates_per_arm': 2}


def run_gpu_probe(output: Path) -> dict:
    from syncopate.pipeline.process_guard import run_guarded

    cpu = require_cpu_evidence(output)
    environment = dict(os.environ)
    environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                       TORCH_NCCL_BLOCKING_WAIT="1")
    launcher = run_guarded(launcher_command(output), cwd=Path(__file__).resolve().parents[2], env=environment,
                           timeout=WORKER_TIMEOUT_SECONDS, stop_grace=STOP_GRACE_SECONDS)
    ranks = []
    for rank in (0, 1):
        path = output / f"rank_{rank}.json"
        if path.exists():
            ranks.append(json.loads(path.read_text()))
    validated = validate_rank_results(ranks, cpu.get('interfaces', {}).get('native_text_root', {}))
    return {"mode": "gpu", "source_sha256": source_sha256(), "plan": preregistered_plan(),
            "launcher": launcher, "ranks": ranks, "rank_validation": validated,
            "health_ok": bool(launcher["rc"] == 0 and launcher.get('process_returncode') == 0
                              and launcher["timed_out"] is False and validated['ok'] is True)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.mode != "gpu":
            parser.error("internal workers are GPU-only")
        return run_gpu_worker(args.output)
    try:
        prepare_output_dir(args.output)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        result = run_cpu_probe(args.output) if args.mode == "cpu" else run_gpu_probe(args.output)
    except Exception as exc:
        result = {"mode": args.mode, "health_ok": False, "source_sha256": source_sha256(),
                  "plan": preregistered_plan(),
                  "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    write_json_once(args.output / "result.json", _json_finite(result))
    print(json.dumps({"mode": args.mode, "health_ok": result["health_ok"], "result": str(args.output / "result.json")}))
    return 0 if result["health_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
