"""仅在显式身份诊断中由 vLLM worker 加载；继承官方扩展，不另写权重加载器。"""
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

from syncopate.train import policy_observer as po


class PolicyWorkerExtension(vLLMColocateWorkerExtension):
    def update_weights_from_ipc(self, *args, syncopate_global_steps=None,
                                syncopate_replica_rank=None, **kwargs):
        self._syncopate_received_version = syncopate_global_steps
        self._syncopate_replica_rank = syncopate_replica_rank
        return super().update_weights_from_ipc(*args, **kwargs)

    def _update_weights(self, weights, peft_config, base_sync_done):
        if not peft_config or not base_sync_done:
            return super()._update_weights(weights, peft_config, base_sync_done)
        incoming = {name: po.tensor_fingerprint(tensor) for name, tensor in weights}
        if len(incoming) != len(weights):
            raise ValueError("接收端的 LoRA 载荷有重名")
        original_add = self.add_lora
        had_own_add = "add_lora" in self.__dict__
        own_add = self.__dict__.get("add_lora")
        calls = []

        def add_lora(request):
            actual = {name: po.tensor_fingerprint(tensor) for name, tensor in request.lora_tensors.items()}
            accepted = original_add(request)
            calls.append({"adapter_id": request.lora_int_id, "accepted": bool(accepted),
                          "slot_present": request.lora_int_id in self.list_loras(), "tensors": actual})
            return accepted

        self.add_lora = add_lora
        error = None
        try:
            return super()._update_weights(weights, peft_config, base_sync_done)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            if had_own_add:
                self.add_lora = own_add
            else:
                del self.add_lora
            po._write("weight_receiver", {"global_steps": getattr(self, "_syncopate_received_version", None),
                "replica_rank": getattr(self, "_syncopate_replica_rank", None), "tensors": incoming,
                "add_lora_calls": calls, "error": error})
