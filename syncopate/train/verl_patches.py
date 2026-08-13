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


def apply(mode: str) -> None:
    """按模式打对应的补丁。必须在 verl 的 main 被 import 之前调。"""
    if mode == "one_step_off":
        _patch_one_step_off_dump_executor()
