import asyncio
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

from syncopate.train.generation_observer import FIELD, observe_resume_client, observe_server


@pytest.mark.parametrize("load_upstream", [False, True])
def test_worker_hook_defers_heavy_imports_and_installs_in_real_modules(load_upstream):
    if load_upstream:
        pytest.importorskip("verl")
    code = """
import sys
from syncopate.train.verl_patches import setup_worker
setup_worker()
assert not any(name in sys.modules for name in ('torch', 'verl', 'vllm'))
"""
    if load_upstream:
        code += """
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
assert FullyAsyncLLMServerClient.__dict__.get('_syncopate_resume_observed')
assert vLLMHttpServer.__dict__.get('_syncopate_generation_observed')
"""
    result = subprocess.run([sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=180,
        env={**os.environ, "SYNCOPATE_GENERATION_OBSERVER": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


def test_observer_preserves_outputs_and_separates_concurrent_requests():
    class Engine:
        async def generate(self, request):
            await asyncio.sleep(0)
            output = SimpleNamespace(token_ids=[request], finish_reason="stop" if request == 1 else "length",
                                     stop_reason=248046 if request == 1 else None)
            yield SimpleNamespace(outputs=[output])

    class Server:
        def __init__(self):
            self.engine = Engine()

        async def generate(self, request):
            async for output in self.engine.generate(request):
                final = output.outputs[0]
            return SimpleNamespace(token_ids=final.token_ids, log_probs=[-.5], stop_reason="completed",
                                   extra_fields={"global_steps": 7})

    async def run():
        original = Server()
        before = await asyncio.gather(*(original.generate(i) for i in [1, 2]))
        observe_server(Server)
        observe_server(Server)  # 安装两遍仍只有一层包装。
        after = await asyncio.gather(*(Server().generate(i) for i in [1, 2]))
        shared = Server()
        concurrent = await asyncio.gather(*(shared.generate(i) for i in [1, 2]))
        for old, new, overlap in zip(before, after, concurrent, strict=True):
            assert (old.token_ids, old.log_probs, old.stop_reason) == (new.token_ids, new.log_probs, new.stop_reason)
            assert new.extra_fields["global_steps"] == old.extra_fields["global_steps"]
            assert new.extra_fields[FIELD] == overlap.extra_fields[FIELD]
        identity = {"policy_version": {"global_steps": 7}, "replica_rank": None}
        assert concurrent[0].extra_fields[FIELD] == {"finish_reason": "stop", "stop_reason": 248046, **identity}
        assert concurrent[1].extra_fields[FIELD] == {"finish_reason": "length", "stop_reason": None, **identity}

    asyncio.run(run())


def test_resuming_client_keeps_observation_without_changing_merged_outputs():
    class Base:
        async def generate(self, request, part):
            await asyncio.sleep(0)
            return SimpleNamespace(token_ids=[part], log_probs=[-.1 * part], extra_fields={
                FIELD: {"finish_reason": "abort" if part == 1 else request}})

    class Resume(Base):
        async def generate(self, request):
            ids, log_probs = [], []
            for part in (1, 2):
                output = await super().generate(request, part)
                ids.extend(output.token_ids)
                log_probs.extend(output.log_probs)
            return SimpleNamespace(token_ids=ids, log_probs=log_probs, stop_reason="completed",
                                   extra_fields={"min_global_steps": 1, "max_global_steps": 2})

    async def run():
        old = await Resume().generate("stop")
        observe_resume_client(Resume, Base)
        observe_resume_client(Resume, Base)
        for result, reason in zip(await asyncio.gather(Resume().generate("stop"), Resume().generate("length")),
                                  ("stop", "length"), strict=True):
            assert (result.token_ids, result.log_probs, result.stop_reason) == (old.token_ids, old.log_probs, old.stop_reason)
            observation = result.extra_fields.pop(FIELD)
            assert result.extra_fields == old.extra_fields
            assert observation["finish_reason"] == reason
            assert observation["policy_version"] == {"min_global_steps": 1, "max_global_steps": 2}
            assert [part["finish_reason"] for part in observation["chunks"]] == ["abort", reason]

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_observer_on_installed_verl_resume_method_without_gpu(monkeypatch, reason):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient as Client, LLMServerClient as Base
    from verl.workers.rollout.replica import TokenOutput
    counts = {}

    async def fake_generate(self, request_id, **kwargs):
        part = counts.get(request_id, 0) + 1
        counts[request_id] = part
        await asyncio.sleep(0)
        return TokenOutput(token_ids=[part], log_probs=[-.1 * part],
            stop_reason="aborted" if part == 1 else "completed",
            extra_fields={"global_steps": part, FIELD: {
                "finish_reason": "abort" if part == 1 else reason, "stop_reason": None}})

    monkeypatch.setattr(Base, "generate", fake_generate)
    monkeypatch.setattr(Client, "generate", Client.generate)
    monkeypatch.setattr(Client, "_syncopate_resume_observed", False, raising=False)
    shell = object.__new__(Client)
    shell.config = OmegaConf.create({})

    async def run():
        params = {"max_tokens": 2 if reason == "length" else 10}
        old = await Client.generate(shell, "before", prompt_ids=[3], sampling_params=params)
        assert FIELD not in old.extra_fields  # 负对照：实际续写层确实丢弃观察字段。
        observe_resume_client(Client, Base)
        new = await Client.generate(shell, "after", prompt_ids=[3], sampling_params=params)
        observation = new.extra_fields.pop(FIELD)
        assert new.model_dump() == old.model_dump()  # token、logprob、版本、停止状态逐项相等。
        assert observation["finish_reason"] == reason
        assert observation["manager_stop_reason"] == ("length" if reason == "length" else "completed")
        assert len(observation["chunks"]) == 2

    asyncio.run(run())


def test_observer_on_installed_verl_method_without_gpu(monkeypatch):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer as Server

    class Engine:
        async def generate(self, **kwargs):
            reason = "length" if kwargs["request_id"] == "length" else "stop"
            await asyncio.sleep(0)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2], finish_reason=reason,
                stop_reason=248046 if reason == "stop" else None)], prompt_logprobs=None, metrics=None)

    shell = SimpleNamespace(engine=Engine(), _disaggregation_role=None, replica_rank=0, global_steps=7,
        lora_as_adapter=False, model_config=SimpleNamespace(processor=None),
        config=OmegaConf.create(dict(max_model_len=128, response_length=32, prompt_length=96,
            full_determinism=False, enable_rollout_routing_replay=False, mtp=None)))
    # 恢复真实类，避免把本测试的观察器留给同进程的别的测试。
    monkeypatch.setattr(Server, "generate", Server.generate)
    monkeypatch.setattr(Server, "_syncopate_generation_observed", False, raising=False)

    async def run():
        before = await Server.generate(shell, prompt_ids=[3, 4], sampling_params={}, request_id="length")
        observe_server(Server)
        after = await asyncio.gather(*(Server.generate(shell, prompt_ids=[3, 4],
                                      sampling_params={}, request_id=reason) for reason in ["stop", "length"]))
        for result in after:
            assert result.token_ids == before.token_ids
            assert result.log_probs == before.log_probs
            assert result.stop_reason == before.stop_reason == "completed"
        assert after[0].extra_fields[FIELD]["finish_reason"] == "stop"
        assert after[1].extra_fields[FIELD]["finish_reason"] == "length"
        assert all(result.extra_fields[FIELD]["policy_version"] == {"global_steps": 7} for result in after)
        assert all(result.extra_fields[FIELD]["replica_rank"] == 0 for result in after)

    asyncio.run(run())


def test_policy_version_does_not_invent_missing_or_drop_initial_version():
    from syncopate.train.generation_observer import policy_version
    assert policy_version({}) == {}
    assert policy_version({"global_steps": 0, "unrelated": 9}) == {"global_steps": 0}
    assert policy_version({"global_steps": None}) == {"global_steps": None}
