"""给 verl 打的补丁，集中放这里 —— 不 fork、不改 site-packages。

★ 为什么单独一个模块，而不是写在 main_ppo_pool 里

补丁要在 **Ray worker 进程**里也生效。Ray 用 cloudpickle 传 actor：
**能 import 到的东西按引用传，传不到的才按值传**。
`python -m syncopate.train.main_ppo_pool` 跑的时候那个模块是 `__main__`，
在 worker 里 import 不回来 —— 靠它承载补丁就得赌 cloudpickle 的序列化策略。
放在这个**能被正常 import 的模块**里，worker 那边照常 import 一次，补丁跟着过去。
（`launch_rl` 已经把仓库根塞进子进程的 PYTHONPATH。）

---

## P1 · `OneStepOffRayTrainer` 漏调 `_init_dump_executor()`（verl 0.8.0 上游 bug）

    AttributeError: 'OneStepOffRayTrainer' object has no attribute '_dump_executor'
      ← ray_trainer.py:376 fit_step → _fit_dump_data → _log_rollout_data → _dump_generations

`_init_dump_executor()` 建的是那个 ThreadPoolExecutor。三处该调的地方里：

    RayPPOTrainer.__init__            ray_trainer.py:372            ✅ 调了
    FullyAsyncTrainer.__init__        fully_async_trainer.py:111    ✅ 调了
    FullyAsyncRollouter.__init__      fully_async_rollouter.py:442  ✅ 调了
    OneStepOffRayTrainer.__init__                                   ❌ **漏了**

⚠️ **触发条件是 `trainer.rollout_data_dir` 非空** —— 也就是说，只要你不 dump 训练数据
就撞不上，所以上游大概没测过。而我们**必须 dump**：
1. verl 的 `compute_data_metrics` 只认两个字段，我们的 cap/subscore 全靠这份 dump；
2. **它是分布漂移的一半**（dump = 训练到的，`dispatched.jsonl` = 下发过的），
   而"异步会不会系统性丢掉长任务"正是这条研究线最关键的问题。
⇒ **不能用"关掉 dump"绕过去**，那等于为了跑通把要测的东西关了。

---

## P2 · `fsdp2_sharded_save/load_to_cpu` 在 DDP（不切分）下被自己的断言挡住

    AssertionError: No DTensor-type parameters found in the model.
                    FSDP2 sharding may not be enabled.
      ← fsdp_utils.py:1082  ← engine_workers.py:127 save_model_to_cpu
      ← fully_async_trainer.py:477 _compute_old_log_prob

**触发条件是「`bypass_mode=False` × `fsdp_size=1`」两个条件叠加**，两个我们都必须要：

- `bypass_mode=False`（decoupled）：只有它产出 ESS，而停止条件 P6 靠 ESS —— 没有刹车不能长跑。
  这个模式下 `_compute_old_log_prob` 要用**第一个版本**的参数重算 old_log_prob（MIS），
  于是需要把当前参数存到 CPU、换上 v1、算完再换回来。
- `fsdp_size=1`（DDP）：本机没有 P2P，FSDP 实测慢 6 倍，**不是优化是必选项**。

⇒ 上游默认 rollout 和 training 分卡时 trainer 侧一定在分片，没考虑"复制而不切分"。

★ **这不是我们绕过一个安全检查** —— 那两个函数的**逐参数逻辑本来就写了普通张量的分支**
（save 里 `if not isinstance(param, DTensor): ... continue`，load 里 `else: param.data.copy_()`），
只是首尾各有一句"至少得有一个 DTensor"的断言。DDP 下所有参数都是普通张量，
断言必然触发。⇒ 补丁只在**一个 DTensor 都没有**时接管，有 DTensor 一律原样交回上游。

⚠️ 这条路径喂的是 old_log_prob，**存错/回填错会静默污染 IS 权重和 advantage**
（不报错、只是训歪）。所以配了一个直接的数值往返测试：存 → 改坏 → 回填 → 逐字节比对
（`tests/train/test_verl_patches.py`）。这类补丁不能只靠"跑起来了"验收。
"""

from __future__ import annotations


def _patch_one_step_off_dump_executor() -> None:
    """给 OneStepOffRayTrainer 补上漏掉的 _init_dump_executor()。"""
    from verl.experimental.one_step_off_policy import ray_trainer as ost

    base = ost.OneStepOffRayTrainer
    if getattr(base, "_syncopate_dump_fix", False):
        return

    class OneStepOffRayTrainerWithDumpFix(base):  # type: ignore[misc, valid-type]
        _syncopate_dump_fix = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 幂等：哪天上游修了，这里就什么都不做
            if not hasattr(self, "_dump_executor"):
                self._init_dump_executor()
                print("[verl-patch] 补上 OneStepOffRayTrainer._init_dump_executor()", flush=True)

    OneStepOffRayTrainerWithDumpFix.__name__ = base.__name__
    OneStepOffRayTrainerWithDumpFix.__qualname__ = base.__qualname__
    ost.OneStepOffRayTrainer = OneStepOffRayTrainerWithDumpFix


def _model_has_dtensor(model) -> bool:
    from torch.distributed.tensor import DTensor

    return any(isinstance(p, DTensor) for p in model.parameters())


def ddp_save_to_cpu(model):
    """DDP（不切分）下的整份 CPU 快照。

    每个 rank 持有完整副本，所以直接逐参数拷贝就是对的（跨 rank 冗余但不影响正确性）。
    返回值刻意和上游同形：`(state, global_spec)`，`global_spec=None` 表示"没有分片规则"。

    ⚠️ 必须 `copy=True`。`param.detach().cpu()`（上游那行的写法）在参数**已经在 CPU 上**时
    是空操作，返回的是**共享存储的视图** —— 快照会跟着后续训练一起漂，MIS 就拿到了错的 v1，
    而且不报错。GPU 上碰巧不中招（跨设备必然拷贝），但这不该靠运气；
    `param_offload` 一开参数就在 CPU 上了。测试里钉死了这条。
    """
    state = {
        name: (param.detach().to("cpu", copy=True), None)
        for name, param in model.named_parameters()
    }
    return state, None


def ddp_load_from_cpu(model, cpu_sharded_state, target_spec) -> None:
    """DDP 下的回填。`target_spec is None` 才走这里，有 spec 一律交回上游。"""
    import torch.distributed as dist

    for name, param in model.named_parameters():
        if name not in cpu_sharded_state:      # 上游同款语义：存的时候没有就跳过
            continue
        cpu_tensor, _ = cpu_sharded_state[name]
        param.data.copy_(cpu_tensor.to(param.device))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()                          # 和上游 load 结尾的同步保持一致


def _patch_fsdp_cpu_copy_for_ddp() -> None:
    """让 save/restore_model_to_cpu 在 fsdp_size=1（DDP）下也能用。见模块 docstring P2。"""
    from verl.utils import fsdp_utils

    if getattr(fsdp_utils, "_syncopate_ddp_cpu_copy", False):
        return

    upstream_save = fsdp_utils.fsdp2_sharded_save_to_cpu
    upstream_load = fsdp_utils.fsdp2_sharded_load_from_cpu

    def save(model):
        if _model_has_dtensor(model):
            return upstream_save(model)         # 有分片 ⇒ 一个字都不改，原样交给上游
        return ddp_save_to_cpu(model)

    def load(model, cpu_sharded_state, target_spec):
        if target_spec is not None:
            return upstream_load(model, cpu_sharded_state, target_spec)
        return ddp_load_from_cpu(model, cpu_sharded_state, target_spec)

    fsdp_utils.fsdp2_sharded_save_to_cpu = save
    fsdp_utils.fsdp2_sharded_load_from_cpu = load
    fsdp_utils._syncopate_ddp_cpu_copy = True
    # 判据行：这个补丁靠替换模块属性生效，没有这行就是没接上。
    print("[verl-patch] fsdp2 CPU 快照支持 DDP（无 DTensor）路径", flush=True)


def setup_worker() -> None:
    """★ Ray **worker 进程**的启动钩子（`runtime_env.worker_process_setup_hook`）。

    ⚠️⚠️ P2 和 P1 的作用域不一样，这是 2026-08-14 用一次失败的冒烟换来的：

        P1 补的 `OneStepOffRayTrainer` 活在 **driver**（TaskRunner）进程里 ⇒ 在
           `apply()` 里打就够了。
        P2 补的 `save_model_to_cpu` 活在 **`WorkerDict` 这个 Ray actor** 里 ——
           那个进程**从不 import 我们的包**（会 import 的是 AgentLoopWorker，
           它加载 verl_agent_loop）。⇒ 在 driver 里打补丁，判据行照常打印，
           断言照常在 worker 里触发。

    **判据行打出来了 ≠ 补丁在需要它的那个进程里生效。** 判据要和作用域一起看。
    """
    _patch_fsdp_cpu_copy_for_ddp()


def apply(mode: str) -> None:
    """按模式打对应的补丁。必须在 verl 的 main 被 import 之前调。

    ⚠️ 这里打的只覆盖 driver 进程。worker 进程靠 `setup_worker`（见上），
    由 `launch_rl` 通过 `ray_kwargs.ray_init.runtime_env.worker_process_setup_hook` 挂上。
    """
    if mode == "one_step_off":
        _patch_one_step_off_dump_executor()
    if mode == "fully_async":
        _patch_fsdp_cpu_copy_for_ddp()
