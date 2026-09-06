import os
from pathlib import Path

import pytest

from syncopate.train.vllm_process import prepare_vllm_process, redirect_stdout_logging


@pytest.mark.parametrize('fail', [False, True])
def test_engine_closed_once_before_loop_exit_on_success_or_error(fail):
    import asyncio
    from types import SimpleNamespace
    from syncopate.train.vllm_process import VLLMEngineLifecycle, run_with_engine
    calls = []
    def shutdown(*, timeout):
        assert asyncio.get_running_loop().is_running()
        calls.append(timeout)
    owner = VLLMEngineLifecycle()
    owner.engine = SimpleNamespace(shutdown=shutdown)
    async def operation():
        if fail:
            raise ValueError('generation failed')
        return 42
    if fail:
        with pytest.raises(ValueError, match='generation failed'):
            asyncio.run(run_with_engine(owner, operation))
    else:
        assert asyncio.run(run_with_engine(owner, operation)) == 42
    owner.close()
    assert calls == [30] and owner.engine is None


def test_shutdown_error_is_not_silently_accepted():
    from types import SimpleNamespace
    from syncopate.train.vllm_process import VLLMEngineLifecycle
    def shutdown(**kwargs):
        raise RuntimeError('shutdown failed')
    owner = VLLMEngineLifecycle()
    owner.engine = SimpleNamespace(shutdown=shutdown)
    with pytest.raises(RuntimeError, match='shutdown failed'):
        owner.close()


def test_current_vllm_shutdown_cleans_native_paths_without_gpu(monkeypatch):
    vllm = pytest.importorskip('vllm')
    from types import SimpleNamespace
    from syncopate.train.vllm_process import VLLMEngineLifecycle
    method = vllm.AsyncLLMEngine.shutdown
    calls = []
    monkeypatch.setitem(method.__globals__, 'shutdown_prometheus', lambda: calls.append('metrics'))
    native = SimpleNamespace(renderer=SimpleNamespace(shutdown=lambda: calls.append('renderer')),
                             engine_core=SimpleNamespace(shutdown=lambda *, timeout: calls.append(timeout)),
                             output_handler=None)
    native.shutdown = method.__get__(native)
    owner = VLLMEngineLifecycle()
    owner.engine = native
    owner.close()
    owner.close()
    assert calls == ['metrics', 'renderer', 30]


def test_logging_redirection_does_not_take_over_other_handlers():
    import io
    import logging
    import sys
    class CaptureHandler(logging.StreamHandler):
        pass
    logger = logging.Logger("isolated-vllm-test")
    regular = logging.StreamHandler(sys.stdout)
    capture = CaptureHandler(sys.stdout)
    stream = io.StringIO()
    foreign = logging.StreamHandler(stream)
    logger.handlers = [regular, capture, foreign]
    redirect_stdout_logging(logger)
    assert regular.stream is sys.stderr
    assert capture.stream is sys.stdout and foreign.stream is stream


def test_library_entry_selects_spawn_before_constructing_engine(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert prepare_vllm_process() == "spawn"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_explicit_fork_is_rejected_before_gpu_use(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with pytest.raises(ValueError, match="拒绝 fork"):
        prepare_vllm_process()


def test_real_vllm_constructor_setup_uses_spawn_without_allocating_model(monkeypatch):
    pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    from syncopate.core.model_paths import STUDENT_MODEL
    from syncopate.train.eval_local import VLLMEngine
    from vllm.utils.system_utils import get_mp_context
    from transformers import GenerationConfig
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    seen = []

    def factory(args):
        assert get_mp_context().get_start_method() == "spawn"
        seen.append(args)
        return object()

    monkeypatch.setattr(vllm.AsyncLLMEngine, "from_engine_args", staticmethod(factory))
    if not (Path(STUDENT_MODEL) / "generation_config.json").is_file():
        pytest.fail("云端启动前置缺少正式模型配置，不能以 skip 通过")
    engine = VLLMEngine(STUDENT_MODEL, None, 12288, 1.0, .85)
    assert len(seen) == 1 and seen[0].model == STUDENT_MODEL
    assert engine.process_start_method == "spawn"
    eos = GenerationConfig.from_pretrained(STUDENT_MODEL, local_files_only=True).eos_token_id
    assert engine.sampling_defaults["stop_token_ids"] == eos
    assert engine.sampling_defaults["max_tokens"] == 12288


def test_missing_eos_does_not_silently_disable_stop_conditions(monkeypatch):
    pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from syncopate.train.eval_local import VLLMEngine
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setattr(GenerationConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(eos_token_id=None))
    monkeypatch.setattr(vllm.AsyncLLMEngine, "from_engine_args",
                        lambda *a, **k: pytest.fail("无 EOS 时不许构造模型"))
    with pytest.raises(ValueError, match="EOS"):
        VLLMEngine("unused", None, 12288, 1.0, .85)
