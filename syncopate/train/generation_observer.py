"""保留 vLLM 原始结束原因，不改变 token、采样或上游的调度状态。

verl 0.9 会把 stop 和 length 都翻译成 completed。completed 只能说明请求结束，
不能说明答案完整。这个观察器在翻译前抄下读数，放进 TokenOutput.extra_fields。
本模块只有标准库；Ray 分卡前不会导入 CUDA、torch 或 vLLM。
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

_capture: ContextVar[dict | None] = ContextVar("syncopate_generation_capture", default=None)
_client_capture: ContextVar[list | None] = ContextVar("syncopate_client_capture", default=None)
FIELD = "syncopate_generation"
VERSION_FIELDS = ("global_steps", "min_global_steps", "max_global_steps")


def policy_version(extra: dict) -> dict:
    """只保留引擎实际报告的版本；0 是有效版本，缺失不能猜成 0。"""
    return {key: extra[key] for key in VERSION_FIELDS if key in extra}


def sampling_snapshot(params) -> dict:
    fields = ("temperature", "top_p", "top_k", "max_tokens", "min_tokens", "seed",
              "stop", "stop_token_ids", "ignore_eos", "repetition_penalty")
    return {name: getattr(params, name) for name in fields if hasattr(params, name)}


def observe_server(server_class) -> None:
    if getattr(server_class, "_syncopate_generation_observed", False):
        return
    original = server_class.generate

    @wraps(original)
    async def generate(self, *args, **kwargs):
        engine = self.engine
        if not getattr(engine, "_syncopate_generation_observed", False):
            engine_generate = engine.generate

            @wraps(engine_generate)
            async def observed_stream(*args, **kwargs):
                async for result in engine_generate(*args, **kwargs):
                    capture = _capture.get()
                    if capture is not None and result.outputs:
                        output = result.outputs[0]
                        capture.update(finish_reason=getattr(output, "finish_reason", None),
                                       stop_reason=getattr(output, "stop_reason", None))
                        if "sampling_params" in kwargs:
                            capture["sampling_params"] = sampling_snapshot(kwargs["sampling_params"])
                    yield result

            engine.generate = observed_stream
            engine._syncopate_generation_observed = True
        capture = {}
        token = _capture.set(capture)
        try:
            result = await original(self, *args, **kwargs)
            capture["policy_version"] = policy_version(result.extra_fields)
            capture["replica_rank"] = getattr(self, "replica_rank", None)
            result.extra_fields = {**result.extra_fields, FIELD: capture}
            return result
        finally:
            _capture.reset(token)

    server_class.generate = generate
    server_class._syncopate_generation_observed = True


def install_verl_observer() -> None:
    # 只由延迟导入钩子调用：上游已经正常导入后才取类。
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    observe_server(vLLMHttpServer)
    print("[generation-observer] 已装；是否真正命中以 artifact 的原始结束原因为准", flush=True)


def observe_resume_client(resume_class, base_class) -> None:
    """续写层会重建 TokenOutput，只补回自己的观察字段，不碰它的合并/重试逻辑。"""
    if resume_class.__dict__.get("_syncopate_resume_observed", False):
        return
    base_generate, resume_generate = base_class.generate, resume_class.generate

    @wraps(base_generate)
    async def capture_part(self, *args, **kwargs):
        result = await base_generate(self, *args, **kwargs)
        capture = _client_capture.get()
        if capture is not None:
            capture.append({**result.extra_fields.get(FIELD, {}),
                            "generated_tokens": len(result.token_ids)})
        return result

    @wraps(resume_generate)
    async def preserve_parts(self, *args, **kwargs):
        parts = []
        token = _client_capture.set(parts)
        try:
            result = await resume_generate(self, *args, **kwargs)
            result.extra_fields = {**result.extra_fields, FIELD: {
                **(parts[-1] if parts else {}), "chunks": parts,
                "policy_version": policy_version(result.extra_fields),
                "manager_stop_reason": result.stop_reason}}
            return result
        finally:
            _client_capture.reset(token)

    base_class.generate = capture_part
    resume_class.generate = preserve_parts
    resume_class._syncopate_resume_observed = True


def install_verl_client_observer() -> None:
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
    observe_resume_client(FullyAsyncLLMServerClient, LLMServerClient)
    print("[generation-observer] 续写转发观察器已装；实际命中看 artifact", flush=True)
