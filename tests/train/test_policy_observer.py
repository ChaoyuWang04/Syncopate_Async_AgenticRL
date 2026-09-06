"""观察器必须识别真更新/跳步/异常，并保持原张量、返回值和训练概率不变。"""
import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from syncopate.train import policy_observer as po


@pytest.fixture
def records(monkeypatch, tmp_path):
    result = []
    monkeypatch.setenv("SYNCOPATE_POLICY_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(po, "_write", lambda kind, value: result.append({"kind": kind, **value}))
    return result


def test_equal_norm_different_bytes_and_scalar_fingerprints():
    a, b = torch.tensor([1., 2.]), torch.tensor([2., 1.])
    assert a.norm() == b.norm()
    assert po.tensor_fingerprint(a)["sha256"] != po.tensor_fingerprint(b)["sha256"]
    assert po.tensor_fingerprint(torch.tensor(2., dtype=torch.bfloat16))["bytes"] == 2


@pytest.mark.parametrize("action", ["step", "skip", "exception"])
def test_installed_engine_optimizer_method_observed_without_gpu(monkeypatch, records, action):
    pytest.importorskip("verl")
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    # 使用安装版的真实方法；小型普通 CPU 模块走其非 FSDP 分支。
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    shell = SimpleNamespace(module=model, optimizer=optimizer,
                            optimizer_config=SimpleNamespace(clip_grad=1.), scaler=None, _qat_enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    if action == "skip":
        model.weight.grad.fill_(float("nan"))
    before = model.weight.detach().clone()
    monkeypatch.setattr(FSDPEngine, "optimizer_step", FSDPEngine.optimizer_step)
    monkeypatch.setattr(FSDPEngine, "_syncopate_policy_optimizer", False, raising=False)
    # clip 抛错时也必须原样向上传播并卸载内层 hook。
    if action == "exception":
        def broken_clip(*a, **kw):
            raise RuntimeError("deliberate clip failure")
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", broken_clip)
    po.observe_optimizer(FSDPEngine)
    po.observe_optimizer(FSDPEngine)
    if action == "exception":
        with pytest.raises(RuntimeError, match="deliberate"):
            FSDPEngine.optimizer_step(shell)
    else:
        value = FSDPEngine.optimizer_step(shell)
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value)) == (action == "step")
    record = records[-1]
    assert record["completed_optimizer_calls"] == int(action == "step")
    assert bool(record["changed_parameters"]) == (action == "step")
    assert torch.equal(before, model.weight) == (action != "step")
    assert not optimizer._optimizer_step_post_hooks
    assert record["state_steps_after"] == ({"1.0": 1} if action == "step" else {})


def test_weight_stream_preserves_storage_context_and_complete_consumption(records):
    tensors = {"a.lora_A.weight": torch.ones(2, 3), "a.lora_B.weight": torch.zeros(3, 2)}

    class Engine:
        def get_per_tensor_param(self, **kw):
            return iter(tensors.items()), {"r": 2}

    class Worker:
        async def update_weights(self, global_steps=None):
            weights, config = Engine().get_per_tensor_param(base_sync_done=True)
            for name, tensor in weights:
                assert tensor is tensors[name]
            return config

    Worker.update_weights.dispatch_marker = "preserve"
    po.observe_weight_source(Engine)
    po.observe_sync_context(Worker)
    assert Worker.update_weights.dispatch_marker == "preserve"
    assert asyncio.run(Worker().update_weights(0)) == {"r": 2}
    assert records[-1]["global_steps"] == 0
    assert records[-1]["stream_complete"]
    assert set(records[-1]["tensors"]) == set(tensors)
    weights, _ = Engine().get_per_tensor_param(base_sync_done=True)
    next(weights)
    weights.close()
    assert not records[-1]["stream_complete"]
    assert records[-1]["global_steps"] is None


@pytest.mark.parametrize("bad", ["shape", "mask", "nan", "empty"])
def test_logprob_invalid_inputs_are_not_numeric_noise(bad):
    left, right, mask = [torch.tensor([-.1, -.2])], [torch.tensor([-.1, -.2])], [torch.ones(2)]
    if bad == "shape": right[0] = right[0][:1]
    if bad == "mask": mask[0][0] = .5
    if bad == "nan": left[0][0] = float("nan")
    if bad == "empty": mask[0].zero_()
    with pytest.raises(ValueError):
        po.logprob_comparison(left, right, mask)


def test_logprob_detects_wrong_policy_but_ignores_masked_tool_tokens():
    left, right = [torch.tensor([-.2, float("nan"), -.5])], [torch.tensor([-.2, 9., -.5])]
    mask = [torch.tensor([1, 0, 1])]
    assert po.logprob_comparison(left, right, mask)["exact_equal"]
    right[0][-1] += .5
    result = po.logprob_comparison(left, right, mask)
    assert result["tokens"] == 2 and result["max_abs"] == pytest.approx(.5)
    assert not result["exact_equal"]


@pytest.mark.parametrize("repeat_fails", [False, True])
def test_repeated_logprob_restores_first_training_target_and_metrics(records, repeat_fails):
    from tensordict import TensorDict

    data = TensorDict({"input_ids": torch.tensor([[1, 2, 3]]), "prompts": torch.tensor([[1]]),
        "responses": torch.tensor([[2, 3]]), "response_mask": torch.tensor([[1, 0]]),
        "old_log_probs": torch.tensor([[-.5, 0.]]), "rollout_log_probs": torch.tensor([[-.4, 0.]]),
        "entropy": torch.tensor([[1., 0.]])}, batch_size=[1])
    batch = SimpleNamespace(keys=["a_0_0"], partition_id="train", tags=[{"min_global_steps": 0, "max_global_steps": 0}])

    class Queue:
        @staticmethod
        def kv_batch_get(*, select_fields, **kw): return data.select(*select_fields)
        @staticmethod
        def kv_batch_put(*, fields, **kw): data.update(fields)

    class Trainer:
        global_steps = 1
        calls = 0

        def _compute_old_log_prob(self, batch, metrics):
            self.calls += 1
            metrics["actor/entropy"] = self.calls
            if self.calls == 2:
                data["old_log_probs"].add_(1.)
                data["entropy"].add_(1.)
                if repeat_fails:
                    raise RuntimeError("repeat failed")
            return batch

    before = data.clone()
    metrics = {}
    po.observe_logprobs(Trainer, Queue)
    if repeat_fails:
        with pytest.raises(RuntimeError, match="repeat failed"):
            Trainer()._compute_old_log_prob(batch, metrics)
    else:
        assert Trainer()._compute_old_log_prob(batch, metrics) is batch
        assert records[-1]["trainer_repeat"]["max_abs"] == pytest.approx(1.)
        saved = torch.load(records[-1]["artifact"], weights_only=True)
        assert saved["tensors"]["old_log_probs"][0][0] == -.5
    assert metrics == {"actor/entropy": 1}
    for key in before.keys():
        assert torch.equal(before[key], data[key]), key


def test_deferred_hook_is_installed_in_real_worker_and_trainer(tmp_path):
    pytest.importorskip("verl")
    code = """
import sys
from syncopate.train.verl_patches import setup_worker
setup_worker()
assert 'torch' not in sys.modules
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
assert FSDPEngine.__dict__.get('_syncopate_policy_optimizer')
assert FSDPEngine.__dict__.get('_syncopate_policy_payload')
assert ActorRolloutRefWorker.__dict__.get('_syncopate_policy_sync')
assert PPOTrainer.__dict__.get('_syncopate_policy_logprob')
"""
    env = {**os.environ, "SYNCOPATE_POLICY_AUDIT_DIR": str(tmp_path), "SYNCOPATE_FSDP_DDP_FIX": "0",
           "SYNCOPATE_FIX_POSTPROC_CONCAT": "0", "SYNCOPATE_LORA_ADAPTER_SYNC": "0", "SYNCOPATE_CKPT_LORA_ONLY": "0"}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr


def test_actual_json_records_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNCOPATE_POLICY_AUDIT_DIR", str(tmp_path))
    for _ in range(2): po._write("control", {"version": 0})
    records = list(tmp_path.glob("control-*.json"))
    assert len(records) == 2
    assert all(json.loads(path.read_text())["version"] == 0 for path in records)


@pytest.mark.parametrize("accepted", [True, False])
def test_real_receiver_passes_same_tensor_bytes_and_records_acceptance(records, accepted):
    pytest.importorskip("vllm")
    from syncopate.train.policy_vllm_extension import PolicyWorkerExtension
    shell = object.__new__(PolicyWorkerExtension)
    original_tensors = {"x.lora_A.weight": torch.ones(2, 3), "x.lora_B.weight": torch.zeros(3, 2)}
    slots = set()

    def add(request):
        for name, value in request.lora_tensors.items():
            assert torch.equal(value, original_tensors[name])
            assert value.data_ptr() != original_tensors[name].data_ptr()  # 官方克隆不能被观察器绕过。
        if accepted: slots.add(request.lora_int_id)
        return accepted

    shell.add_lora, shell.list_loras = add, lambda: slots
    shell._syncopate_received_version, shell._syncopate_replica_rank = 0, 1
    result = shell._update_weights(list(original_tensors.items()), {"r": 2}, True)
    assert result is None and shell.add_lora is add
    record = records[-1]
    call = record["add_lora_calls"][0]
    assert record["global_steps"] == 0 and record["replica_rank"] == 1
    assert call["accepted"] is accepted and call["slot_present"] is accepted
    assert call["tensors"] == record["tensors"]


def test_installed_server_adapter_delivers_observer_version_without_changing_weights(monkeypatch):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf
    import verl.workers.rollout.vllm_rollout.vllm_rollout as module
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    weights = [("a.lora_A.weight", torch.ones(1))]
    requests = []

    class Sender:
        def __init__(self, **kw): pass
        async def async_send_weights(self, sent):
            assert list(sent) == weights

    async def execute(method, **kw):
        requests.append((method, kw))

    shell = SimpleNamespace(_execute_method=execute, use_shm=True, zmq_handle="unused",
        config=OmegaConf.create({"checkpoint_engine": {"update_weights_bucket_megabytes": 1}}),
        _has_server=False, replica_rank=1, rollout_rank=0)
    monkeypatch.setattr(module, "BucketedWeightSender", Sender)
    monkeypatch.setattr(module.ServerAdapter, "update_weights", module.ServerAdapter.update_weights)
    monkeypatch.setattr(module.ServerAdapter, "_syncopate_policy_adapter", False, raising=False)
    po.observe_server_adapter(module.ServerAdapter)
    asyncio.run(module.ServerAdapter.update_weights(shell, weights, global_steps=0, peft_config={"r": 2}, base_sync_done=True))
    assert requests[0][0] == "update_weights_from_ipc"
    assert requests[0][1]["kwargs"] == {"peft_config": {"r": 2}, "base_sync_done": True, "use_shm": True,
                                        "syncopate_global_steps": 0, "syncopate_replica_rank": 1}
    monkeypatch.setattr(vLLMHttpServer, "_get_worker_extension_cls", vLLMHttpServer._get_worker_extension_cls)
    monkeypatch.setattr(vLLMHttpServer, "_syncopate_policy_extension", False, raising=False)
    po.observe_extension_selection(vLLMHttpServer)
    assert vLLMHttpServer._get_worker_extension_cls(None) == "syncopate.train.policy_vllm_extension.PolicyWorkerExtension"


def test_installed_v1_logprob_method_with_real_jagged_tensors(monkeypatch, records):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    import verl.trainer.ppo.v1.trainer_base as module

    def nested(rows):
        return torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)

    data = TensorDict({"input_ids": nested([[1, 2, 3, 4], [5, 6, 7]]), "prompts": nested([[1], [5, 6]]),
        "responses": nested([[2, 3, 4], [7]]), "response_mask": nested([[1, 0, 1], [1]]),
        "rollout_log_probs": nested([[-.4, 0., -.2], [-.6]])}, batch_size=[2])
    batch = SimpleNamespace(keys=["a_0_0", "b_0_0"], partition_id="train", extra_info={},
                            tags=[{"min_global_steps": 0, "max_global_steps": 0}] * 2)
    # KVBatchMeta 的 len；保留真实方法依赖的属性，不启动队列服务或 Ray。
    class Meta(SimpleNamespace):
        def __len__(self): return len(self.keys)
    batch = Meta(**vars(batch))

    class Queue:
        @staticmethod
        def kv_batch_get(*, select_fields, **kw): return data.select(*select_fields)
        @staticmethod
        def kv_batch_put(*, fields, **kw):
            data.update(fields)
            return batch

    class Worker:
        calls = 0
        def compute_log_prob(self, meta):
            self.calls += 1
            data["log_probs"] = nested([[-.4, -.3, -.2, -9.], [-.7, -.6, -9.]])
            data["entropy"] = nested([[1., 1., 1., 1.], [1., 1., 1.]])
            return meta

    class Driver:
        _compute_old_log_prob = module.PPOTrainer._compute_old_log_prob

    monkeypatch.setattr(module, "tq", Queue)
    shell = SimpleNamespace(config=OmegaConf.create({"algorithm": {"rollout_correction": {"bypass_mode": False}},
        "actor_rollout_ref": {"rollout": {"calculate_log_probs": True, "temperature": 1.},
                              "actor": {"loss_agg_mode": "token-mean", "loss_scale_factor": None}}}),
        actor_rollout_wg=Worker(), global_steps=1)
    po.observe_logprobs(Driver, Queue)
    assert Driver._compute_old_log_prob(shell, batch, {}) is batch
    assert shell.actor_rollout_wg.calls == 2
    assert records[-1]["trainer_repeat"]["exact_equal"]
    assert records[-1]["trainer_rollout"]["exact_equal"]
    assert records[-1]["trainer_rollout"]["tokens"] == 3
