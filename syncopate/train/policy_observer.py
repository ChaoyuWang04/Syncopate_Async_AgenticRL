"""B03 的显式身份诊断；不改变训练返回值，不把诊断耗时当性能基线。

模块加载只用标准库。Torch/verl 只能在框架正常导入、分卡后按需读取。
FSDP 分片只与自身更新前后比较；不同分片不会被误判为应当相等的 DP 副本。
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from uuid import uuid4

_sync_version: ContextVar[int | None] = ContextVar("syncopate_sync_version", default=None)


def _directory() -> Path:
    value = os.environ.get("SYNCOPATE_POLICY_AUDIT_DIR")
    if not value:
        raise RuntimeError("身份观察器缺独立审计目录")
    directory = Path(value)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write(kind: str, details: dict) -> dict:
    import torch.distributed as dist

    record = {"schema_version": 1, "kind": kind, "host": socket.gethostname(),
              "pid": os.getpid(), "rank": dist.get_rank() if dist.is_initialized() else None,
              **details}
    path = _directory() / f"{kind}-{os.getpid()}-{uuid4().hex}.json"
    with path.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False,
                  default=lambda value: sorted(value) if isinstance(value, set) else _unsupported_json(value))
        stream.write("\n")
    print(f"[policy-observer] {kind} -> {path}", flush=True)
    return record


def _unsupported_json(value):
    raise TypeError(f"身份记录不支持 {type(value).__name__}")


def tensor_fingerprint(tensor) -> dict:
    """记录准确字节、形状和 dtype；DTensor 只读本地分片，不隐式做 collective。"""
    import torch

    shard = hasattr(tensor, "placements")
    local = tensor.to_local() if shard else tensor
    value = local.detach().to("cpu").contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "shape": list(tensor.shape),
            "local_shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw),
            "placements": [str(p) for p in tensor.placements] if shard else None,
            "finite": bool(torch.isfinite(value).all()), "nonzero": int(torch.count_nonzero(value))}


def parameter_fingerprints(module, *, gradients: bool = False) -> dict:
    return {name: tensor_fingerprint(param.grad if gradients else param)
            for name, param in module.named_parameters()
            if param.requires_grad and (not gradients or param.grad is not None)}


def optimizer_steps(optimizer) -> dict:
    from collections import Counter

    counts = Counter()
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            value = value.item() if hasattr(value, "item") else value
            counts[str(value)] += 1
    return dict(counts)


def observe_optimizer(engine_class) -> None:
    if engine_class.__dict__.get("_syncopate_policy_optimizer", False):
        return
    original = engine_class.optimizer_step

    @wraps(original)
    def optimizer_step(self, *args, **kwargs):
        before = parameter_fingerprints(self.module)
        gradients = parameter_fingerprints(self.module, gradients=True)
        states_before = optimizer_steps(self.optimizer)
        completed_calls = 0

        def completed(*_):
            nonlocal completed_calls
            completed_calls += 1

        # 外层 engine 方法可能合法跳步；只有内层 optimizer 成功返回才计数。
        handle = self.optimizer.register_step_post_hook(completed)
        error = None
        result = None
        try:
            result = original(self, *args, **kwargs)
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            handle.remove()
            after = parameter_fingerprints(self.module)
            attempts = getattr(self, "_syncopate_policy_attempts", 0) + 1
            self._syncopate_policy_attempts = attempts
            _write("optimizer", {"attempt": attempts, "completed_optimizer_calls": completed_calls,
                "error": error, "reported_grad_norm": result if result is None or math.isfinite(result) else str(result),
                "parameters_before": before, "parameters_after": after, "gradients_before_clip": gradients,
                "changed_parameters": sum(before[name]["sha256"] != after[name]["sha256"] for name in before),
                "state_steps_before": states_before, "state_steps_after": optimizer_steps(self.optimizer)})

    engine_class.optimizer_step = optimizer_step
    engine_class._syncopate_policy_optimizer = True


def observe_weight_source(engine_class) -> None:
    if engine_class.__dict__.get("_syncopate_policy_payload", False):
        return
    original = engine_class.get_per_tensor_param

    @wraps(original)
    def get_per_tensor_param(self, *args, **kwargs):
        weights, peft_config = original(self, *args, **kwargs)
        # 当前只验 adapter 同步；不把整个 35B 基座拷到 CPU 做无关哈希。
        if peft_config is None or not kwargs.get("base_sync_done", False):
            return weights, peft_config
        version = _sync_version.get()

        def observed():
            manifest = {}
            complete = False
            try:
                for name, tensor in weights:
                    if name in manifest:
                        raise ValueError(f"同步载荷重名: {name}")
                    manifest[name] = tensor_fingerprint(tensor)
                    yield name, tensor
                complete = True
            finally:
                _write("weight_source", {"global_steps": version, "stream_complete": complete,
                    "peft_config": peft_config, "tensors": manifest})

        return observed(), peft_config

    engine_class.get_per_tensor_param = get_per_tensor_param
    engine_class._syncopate_policy_payload = True


def observe_sync_context(worker_class) -> None:
    if worker_class.__dict__.get("_syncopate_policy_sync", False):
        return
    original = worker_class.update_weights

    @wraps(original)
    async def update_weights(self, global_steps=None, *args, **kwargs):
        token = _sync_version.set(global_steps)
        try:
            return await original(self, global_steps, *args, **kwargs)
        finally:
            _sync_version.reset(token)

    worker_class.update_weights = update_weights
    worker_class._syncopate_policy_sync = True


def observe_server_adapter(adapter_class) -> None:
    """把版本只传给本项目观察扩展，不改原始权重或官方传输协议。"""
    if adapter_class.__dict__.get("_syncopate_policy_adapter", False):
        return
    original = adapter_class.update_weights

    @wraps(original)
    async def update_weights(self, weights, global_steps=None, **kwargs):
        return await original(self, weights, global_steps=global_steps,
            syncopate_global_steps=global_steps, syncopate_replica_rank=self.replica_rank, **kwargs)

    adapter_class.update_weights = update_weights
    adapter_class._syncopate_policy_adapter = True


def observe_extension_selection(server_class) -> None:
    if server_class.__dict__.get("_syncopate_policy_extension", False):
        return
    original = server_class._get_worker_extension_cls

    @wraps(original)
    def extension(self):
        value = original(self)
        if value != "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension":
            raise ValueError("未知 vLLM worker 扩展；不能静默替换成身份观察器")
        return "syncopate.train.policy_vllm_extension.PolicyWorkerExtension"

    server_class._get_worker_extension_cls = extension
    server_class._syncopate_policy_extension = True


def logprob_comparison(left_rows, right_rows, masks) -> dict:
    """只比较模型采样 token，拒绝形状、mask 和有限性错误；不事后发明通过线。"""
    import torch

    differences = []
    if not (len(left_rows) == len(right_rows) == len(masks)):
        raise ValueError("logprob 样本数不一致")
    for left, right, mask in zip(left_rows, right_rows, masks, strict=True):
        if left.ndim != 1 or left.shape != right.shape or left.shape != mask.shape:
            raise ValueError("logprob/token mask 形状不一致")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("mask 不是 0/1")
        active = mask.bool()
        x, y = left[active].double(), right[active].double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("有效 token 的 logprob 非有限")
        differences.append(x - y)
    delta = torch.cat(differences)
    if not delta.numel():
        raise ValueError("没有可对拍的采样 token")
    absolute = delta.abs()
    return {"tokens": delta.numel(), "mean_signed": delta.mean().item(),
            "mean_abs": absolute.mean().item(), "max_abs": absolute.max().item(),
            "p99_abs": torch.quantile(absolute, .99).item(), "exact_equal": bool((delta == 0).all())}


def _rows(data, key) -> list:
    return [row.detach().to("cpu", copy=True) for row in data[key].unbind()]


def observe_logprobs(trainer_class, queue) -> None:
    if trainer_class.__dict__.get("_syncopate_policy_logprob", False):
        return
    original = trainer_class._compute_old_log_prob

    @wraps(original)
    def compute(self, batch, metrics):
        import torch

        result = original(self, batch, metrics)
        fields = ["input_ids", "prompts", "responses", "response_mask", "old_log_probs", "rollout_log_probs", "entropy"]
        first = queue.kv_batch_get(keys=result.keys, partition_id=result.partition_id, select_fields=fields).clone()
        saved = {key: _rows(first, key) for key in fields}
        for ids, prompt, response in zip(saved["input_ids"], saved["prompts"], saved["responses"], strict=True):
            if not torch.equal(ids, torch.cat([prompt, response])):
                raise ValueError("trainer 的 input_ids 不是原 prompt + response")
        if any(response.shape != mask.shape for response, mask in zip(saved["responses"], saved["response_mask"], strict=True)):
            raise ValueError("response token 与 mask 形状不一致")
        # 重算同一批、同一策略得到数值噪声；训练仍使用第一次的概率和 entropy。
        try:
            original(self, result, {})
            repeated = queue.kv_batch_get(keys=result.keys, partition_id=result.partition_id,
                                           select_fields=["old_log_probs"])
            saved["repeat_log_probs"] = _rows(repeated, "old_log_probs")
        finally:
            queue.kv_batch_put(keys=result.keys, partition_id=result.partition_id,
                               fields=first.select("old_log_probs", "entropy"))
        comparison = logprob_comparison(saved["old_log_probs"], saved["rollout_log_probs"], saved["response_mask"])
        noise = logprob_comparison(saved["old_log_probs"], saved["repeat_log_probs"], saved["response_mask"])
        artifact = _directory() / f"logprobs-step{self.global_steps}-{uuid4().hex}.pt"
        with artifact.open("xb") as stream:
            torch.save({"schema_version": 1, "step": self.global_steps, "keys": list(result.keys),
                        "tags": result.tags, "tensors": saved}, stream)
        _write("logprob", {"step": self.global_steps, "keys": list(result.keys), "tags": result.tags,
                            "artifact": str(artifact), "trainer_rollout": comparison, "trainer_repeat": noise})
        return result

    trainer_class._compute_old_log_prob = compute
    trainer_class._syncopate_policy_logprob = True


def install_engine() -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    observe_optimizer(FSDPEngine)
    observe_weight_source(FSDPEngine)


def install_worker() -> None:
    from verl.workers.engine_workers import ActorRolloutRefWorker
    observe_sync_context(ActorRolloutRefWorker)


def install_trainer() -> None:
    import transfer_queue
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    observe_logprobs(PPOTrainer, transfer_queue)


def install_server_adapter() -> None:
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
    observe_server_adapter(ServerAdapter)


def install_server() -> None:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    observe_extension_selection(vLLMHttpServer)
