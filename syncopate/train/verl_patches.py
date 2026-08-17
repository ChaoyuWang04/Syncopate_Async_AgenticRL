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

import os


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

    ★★ **只存可训练参数**（2026-08-14，E13）。原实现遍历全部 `named_parameters()`，
    实测 Qwen3-4B + LoRA r32 下是 **902 个张量 / 8.309 GB，而可训练的只有 504 个 /
    0.264 GB（3.18%）**。冻结基座 `requires_grad=False`、优化器从不更新它 ⇒
    **v1 与当前版本的基座逐字节相同，存了也是白存**。
    实测收益：`save` 3.579 → 0.037 s（96×），按 fully_async 的调用序列
    （3/4 的步要 1 save + 2 restore）**平均每步 4.34 → 0.083 s，省 74.1 s 的 5.7%**。
    ⇒ 详见 `docs/infra_exp/E13-proximal-anchor-snapshot.md`。

    ⚠️ 这个优化**依赖「基座确实冻结」这个前提**。若哪天改成全参微调（`requires_grad`
    全为 True），本函数自动退回全量拷贝——语义仍然正确，只是不再省。
    测试里钉死了「只存可训练参数 + 全参模型照常全存」两条。
    """
    state = {
        name: (param.detach().to("cpu", copy=True), None)
        for name, param in model.named_parameters()
        if param.requires_grad
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


def _patch_sync_step_timing() -> None:
    """★ 实验用（可选，`SYNCOPATE_SYNC_TIMING=1` 才装）：把权重同步的 8 步分别计时。

    动机（E12）：稳态 `param_sync` 55.8 s，两点反解出 **98.6% 不是传输**；
    而 buffer 分配（2.6 ms）与 `empty_cache`（12.2 ms）微基准也已排除。
    剩下的嫌疑只能靠真跑分步计时来分：abort / release_kv / build_group / resume_kv / resume_gen。

    ⚠️ `CheckpointEngineManager` 活在 **FullyAsyncTrainer 这个 Ray actor** 里，
    所以必须挂 `setup_worker`（worker_process_setup_hook），driver 侧打无效 —— P2 的同一条教训。
    """
    import time

    from verl.checkpoint_engine.base import CheckpointEngineManager as M

    if getattr(M, "_syncopate_sync_timing", False):
        return

    def wrap_async(name):
        orig = getattr(M, name)

        async def inner(self, *a, **kw):
            t0 = time.perf_counter()
            r = await orig(self, *a, **kw)
            print(f"[sync-timing] {name}: {time.perf_counter() - t0:.3f} s", flush=True)
            return r

        return inner

    def wrap_sync(name):
        orig = getattr(M, name)

        def inner(self, *a, **kw):
            t0 = time.perf_counter()
            r = orig(self, *a, **kw)
            print(f"[sync-timing] {name}: {time.perf_counter() - t0:.3f} s", flush=True)
            return r

        return inner

    for n in ("abort_replicas", "release_kv_cache_replicas",
              "resume_kv_cache_replicas", "resume_generation_replicas"):
        # 这四个挂了 @auto_await，取到的是包装后的可调用；直接换掉属性即可
        setattr(M, n, wrap_async(n))
    M.build_process_group = wrap_sync("build_process_group")

    # ★ 第二层：`build_process_group` 实测 46 s（占一次同步的绝大部分），
    #   它内部还有两段，必须再拆一层才知道是哪段：
    #     prepare()            每个 worker 分配 2×bucket buffer
    #                          ——空卡上微基准只要 2.6 ms，但 rollout 卡被 vLLM 占了 75%，
    #                            在快满的卡上分配 4 GB 可能完全是另一回事
    #     init_process_group() 建组（rebuild_group=False 时应跳过）+ **collective.barrier()**
    #                          ——barrier 会等所有 worker 到齐，
    #                            很可能在这里替 abort/release_kv 那两个"立刻返回"的异步调用**补等**
    #   这两个跑在 **worker 进程**里，同样靠 setup_worker 覆盖到。
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine as E

    # ★ 第四层：A/B 实测 `layered_summon` True/False **都是 ~60 s**（2026-08-14），
    #   而两者走的是完全不同的取参数路径（逐层 summon vs 整体 summon_full_params）
    #   ⇒ **成本不在"取参数"，在两条路径共有的部分** ⇒ 只剩 `send_weights`。
    #   `send_weights` 是 async，单独包一层就能把 60 s 切成「取参数」和「发」两半。
    if not getattr(E, "_syncopate_send_timing", False):
        _orig_send = E.send_weights

        async def _timed_send(self, *a, **kw):
            t0 = time.perf_counter()
            r = await _orig_send(self, *a, **kw)
            print(f"[sync-timing]     engine.send_weights: {time.perf_counter() - t0:.3f} s", flush=True)
            return r

        E.send_weights = _timed_send
        E._syncopate_send_timing = True

    if not getattr(E, "_syncopate_sync_timing", False):
        for n in ("prepare", "init_process_group", "finalize"):
            orig_e = getattr(E, n)

            def mk(name, f):
                def inner(self, *a, **kw):
                    t0 = time.perf_counter()
                    r = f(self, *a, **kw)
                    dt = time.perf_counter() - t0
                    if dt > 0.05:      # 只报值得看的，别刷屏
                        print(f"[sync-timing]   engine.{name}(rank={getattr(self,'rank',None)}): "
                              f"{dt:.3f} s", flush=True)
                    return r
                return inner

            setattr(E, n, mk(n, orig_e))
        E._syncopate_sync_timing = True

    # ★ 第三层：实测（2026-08-14）第二次同步起 `build_process_group` 只剩 0.023 s
    #   （首次 45.967 s 是一次性建 NCCL 组），而稳态 param_sync 仍是 ~56 s
    #   ⇒ **钱在第 5 步「传输 + 两侧 update_weights」里**，而两点反解说传输本身只有 0.8 s
    #   ⇒ 必须把 trainer 侧（取参数 + send）和 rollout 侧（recv + 装进 vLLM）分开量。
    # ⛔ **2026-08-14 踩过的坑，别再犯**：第一版这里用一个裸的 `async def inner` 换掉方法，
    #    结果整跑崩在
    #        AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'
    #    根因：`update_weights` 上有 verl 的 `@register(...)` 装饰器，它把
    #    `{"dispatch_mode", "execute_mode", "blocking"}` 挂在函数的
    #    `MAGIC_ATTR = "attrs_3141562937"` 属性上（decorator.py:440-441），
    #    **`RayWorkerGroup` 靠扫描这个属性自动生成组级方法**。
    #    我的包装把它丢了 ⇒ 组级 `update_weights` 直接不存在。
    #    ⇒ **包装带装饰器的方法时，必须把装饰器挂的属性一起带过去。**
    import functools

    MAGIC_ATTR = "attrs_3141562937"

    def wrap_worker(cls, name, tag):
        if getattr(cls, f"_syncopate_t_{name}", False):
            return
        orig_w = getattr(cls, name)

        @functools.wraps(orig_w)
        async def inner(self, *a, **kw):
            t0 = time.perf_counter()
            r = await orig_w(self, *a, **kw)
            print(f"[sync-timing]   {tag}: {time.perf_counter() - t0:.3f} s", flush=True)
            return r

        # ★ 把 verl 的 dispatch 元数据原样带过去，否则 RayWorkerGroup 生成不出组级方法
        if hasattr(orig_w, MAGIC_ATTR):
            setattr(inner, MAGIC_ATTR, getattr(orig_w, MAGIC_ATTR))
        else:
            print(f"[verl-patch] ⚠️ {cls.__name__}.{name} 没有 {MAGIC_ATTR}，跳过探针以免破坏 dispatch",
                  flush=True)
            return

        setattr(cls, name, inner)
        setattr(cls, f"_syncopate_t_{name}", True)

    try:
        from verl.checkpoint_engine.base import CheckpointEngineWorker as CEW

        wrap_worker(CEW, "update_weights", "rollout侧 recv+装进vLLM")
    except Exception as exc:      # 探针失败不许拖垮训练
        print(f"[verl-patch] ⚠️ rollout 侧探针未装上: {exc}", flush=True)

    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker as ARW

        wrap_worker(ARW, "update_weights", "trainer侧 取参数+send")
    except Exception as exc:
        print(f"[verl-patch] ⚠️ trainer 侧探针未装上: {exc}", flush=True)

    M._syncopate_sync_timing = True
    print("[verl-patch] 权重同步分步计时已启用（SYNCOPATE_SYNC_TIMING=1）", flush=True)


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
    # ★★★ 2026-08-17：这里**绝不能直接 import verl**（见 _defer_until_imported 的说明）。
    _defer_until_imported("verl.utils.fsdp_utils", _patch_fsdp_cpu_copy_for_ddp)

    # ★★★ 动态分池（2026-08-17 补上，此前只打在 driver ⇒ fully_async 下静默失效）
    #
    # `FullyAsyncRollouter` 是个 **Ray actor**，活在另一个进程里，而它
    # （`fully_async_rollouter.py:464`）**确实调** `create_rl_sampler`，
    # 且那个 `import` 写在函数体内 ⇒ 调用时才解析 ⇒ 对 monkeypatch 最友好。
    #
    # ⚠️ 曾经有一行日志断言「fully_async 不调 create_rl_sampler ⇒ 本轮不生效」——
    #    **那句话是错的**，它把"其实能行、只是没接上"整个盖住了，而且长得像个合格判据。
    #    真因一直是**作用域**：补丁只打在 driver。
    #    ⇒ 教训：判据行只许写观测，不许写断言（05-handoff §6 变种②）。
    if os.environ.get("SYNCOPATE_POOL", "1") == "1":
        _defer_until_imported("verl.trainer.main_ppo", _patch_pool_sampler)
    if os.environ.get("SYNCOPATE_NVTX") == "1":
        # 源模块 + 三个已知的消费方都要挂：谁先被 import 都能兜住（见 _patch_nvtx_timers 注释①）
        for _m in ("verl.utils.profiler.performance",
                   "verl.utils.debug",
                   "verl.trainer.ppo.ray_trainer",
                   "verl.experimental.fully_async_policy.fully_async_trainer",
                   "verl.experimental.fully_async_policy.fully_async_rollouter",
                   "verl.experimental.one_step_off_policy.ray_trainer"):
            _defer_until_imported(_m, _patch_nvtx_timers_or_rebind)
    if os.environ.get("SYNCOPATE_DEVICE_PROBE") == "1":
        _defer_until_imported("verl.single_controller.base.worker", _patch_device_probe)
    if os.environ.get("SYNCOPATE_GRAD_PROBE") == "1":
        _defer_until_imported("verl.workers.engine.fsdp.transformer_impl", _patch_grad_probe)


# ★★★ NVTX 阶段标注（E01 / A5 的门槛，2026-08-17 加）
#
# nsys 采到 trace 之后才发现：**切不开阶段**。verl 里那个上下文管理器叫
# `marked_timer`，docstring 写着 "adds platform markers for profiling" ——
# 而**函数体只是 `yield from _timer(...)`，一个 marker 都没有**
# （`verl/utils/profiler/performance.py:172`）。
# ⇒ 又一个「名字在、机制没接上」，这次在上游。
#
# 打上之后，nsys 的 trace 里就有 `gen / old_log_prob / ref / update_actor / param_sync`
# 这些 range，kernel 才能按阶段归属 —— **这是 B12/E17 的门槛**：
# 没有阶段归属，「ref 那一遍前向能省多少」就只能猜。
#
# ⚠️ 两个作用域细节（都踩过同型的坑）：
#   ① 消费方写的是 `from verl.utils.debug import marked_timer` ⇒ **import 时就绑死了名字**，
#      只改源模块**对已经导过的模块无效** ⇒ 必须把所有 sys.modules 里指向原函数的名字一起换。
#   ② 阶段 range 打在 **trainer driver** 进程，kernel 跑在 **WorkerDict** 进程 ——
#      两者靠 nsys 的**同一条时间轴**对齐（按时间区间归属），不是靠同进程。
_NVTX_ORIGINALS: set = set()


def _rebind_everywhere(mapping: dict) -> int:
    """把 sys.modules 里所有指向 `mapping` 里旧函数的属性换成新函数。返回换了几处。"""
    import sys

    n = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for attr in ("marked_timer", "simple_timer"):
            cur = getattr(mod, attr, None)
            if cur is not None and cur in mapping:
                try:
                    setattr(mod, attr, mapping[cur])
                    n += 1
                except Exception:      # noqa: BLE001 - 只读模块跳过即可
                    pass
    return n


def _patch_nvtx_timers() -> None:
    """给 verl 的每个计时段套一层 NVTX range（`SYNCOPATE_NVTX=1` 才开）。"""
    from contextlib import contextmanager

    from verl.utils.profiler import performance as perf

    if getattr(perf, "_syncopate_nvtx", False):
        return

    def wrap(orig, tag):
        @contextmanager
        def inner(name, timing_raw, *a, **kw):
            import torch

            torch.cuda.nvtx.range_push(f"syncopate/{name}")
            try:
                with orig(name, timing_raw, *a, **kw):
                    yield
            finally:
                torch.cuda.nvtx.range_pop()

        inner.__name__ = tag
        return inner

    mapping = {}
    for attr in ("marked_timer", "simple_timer"):
        orig = getattr(perf, attr, None)
        if orig is None:
            continue
        new = wrap(orig, attr)
        mapping[orig] = new
        setattr(perf, attr, new)
    _NVTX_ORIGINALS.update(mapping)
    n = _rebind_everywhere(mapping)
    perf._syncopate_nvtx = True
    # 判据行：没有这行就是没生效（本项目纪律）。带上换了几处，方便判断是不是只改到源模块。
    print(f"[verl-patch] NVTX 阶段标注 ✓ 已替换 {n} 处引用（marked_timer/simple_timer）", flush=True)


def _patch_nvtx_timers_or_rebind() -> None:
    """第一次调用 = 打补丁；之后每次 = 把新导入的模块里那份旧引用再换掉。"""
    _patch_nvtx_timers()
    if _NVTX_ORIGINALS:
        from contextlib import suppress
        with suppress(Exception):
            import sys

            from verl.utils.profiler import performance as perf
            _rebind_everywhere({o: getattr(perf, o.__name__, None) for o in _NVTX_ORIGINALS
                                if getattr(perf, o.__name__, None) is not None})


def _patch_pool_sampler() -> None:
    """在 worker 进程里装动态分池的 sampler 补丁。判据行由 DynamicPoolSampler 自己打。"""
    from syncopate.train.main_ppo_pool import install_sampler_patch

    install_sampler_patch()
    print("[verl-patch] worker 进程：动态分池 sampler 已装", flush=True)


def _patch_grad_probe() -> None:
    """临时探针：`optimizer_step` 之前，逐参数报告谁的梯度是 nan/inf。

    背景：2026-08-17 发现 `actor/grad_norm=nan`，而 verl 在非有限时**直接
    `optimizer.zero_grad()` 跳过更新** ⇒ 模型一步都没更新过（WARN 行每步都打）。
    `fsdp2_clip_grad_norm_` 已经正确过滤了 `p.grad is None` ⇒ 是梯度本身坏了。
    """
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    orig = FSDPEngine.optimizer_step

    def probed(self):
        import torch
        tot = bad = nog = 0
        names_bad, first = [], None
        for n, p in self.module.named_parameters():
            if not p.requires_grad:
                continue
            tot += 1
            if p.grad is None:
                nog += 1
                continue
            if not torch.isfinite(p.grad).all():
                bad += 1
                if len(names_bad) < 6:
                    names_bad.append(n)
            elif first is None:
                first = (n, float(p.grad.norm()))
        print(f"[grad-probe] 可训练={tot} 无梯度={nog} 非有限={bad} "
              f"首个正常参数={first} 坏的前几个={names_bad}", flush=True)
        return orig(self)

    FSDPEngine.optimizer_step = probed
    print("[verl-patch] grad-probe 已挂上", flush=True)
    if os.environ.get("SYNCOPATE_SYNC_TIMING") == "1":
        _patch_sync_step_timing()


def _defer_until_imported(module_name: str, patch: "callable") -> None:
    """在 `module_name` **被别人 import 完的那一刻**执行 `patch`，我们自己不 import 它。

    ★★★ 为什么必须这样 —— 2026-08-17 花了一整轮定位出来的：

    `worker_process_setup_hook` 在 Ray worker 进程**刚起来时**跑，而 Ray 是在
    **之后**才给这个 actor 设 `CUDA_VISIBLE_DEVICES` 的。如果钩子在那之前
    `import verl.utils.fsdp_utils`，这条 import 链会把 **CUDA 的设备枚举固化下来**；
    等 Ray 再设 CVD，**可见数量变了、设备→物理卡的映射却不变** ⇒ 每个 worker 的
    `cuda:0` 都指向物理 GPU0。

    表现：分卡模式下 3 个 trainer rank **全挤在 GPU0**，第一次权重同步 OOM；
    而 colocate 完全正常（它不挂这个钩子）—— 对照实验见交接文档。
    最小复现：

        python -c "import os,torch; os.environ['CUDA_VISIBLE_DEVICES']='2';
                   print(hex(torch.cuda.get_device_properties(0).pci_bus_id))"   → 0xa1 ✅
        同上但先 `from syncopate.train.verl_patches import setup_worker; setup_worker()`  → 0x21 🔴

    ⚠️ 这个钩子当初正是为了修「补丁打在 driver、断言在 worker」的**作用域**问题才加的
    （见 setup_worker 的 docstring）—— 修一个坑挖出一个更深的，而且这个是**静默**的：
    训练照常起来，只是三张卡变一张。⇒ 新纪律：**进程启动钩子里只许做纯 Python 的事，
    任何可能碰 CUDA 的 import 都要延迟到设备确定之后。**
    """
    import importlib.abc
    import importlib.util
    import sys

    if module_name in sys.modules:      # 已经被导过：直接打，此时设备已定
        patch()
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module_name:
                return None
            sys.meta_path.remove(self)          # 先摘掉，避免 find_spec 递归
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(mod):
                inner(mod)
                patch()                          # ★ 模块装载完才打补丁

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _patch_device_probe() -> None:
    """临时探针：在 **worker 进程内、运行时** 打出每个 rank 落到哪张物理卡。

    ⚠️ 为什么不能读 `/proc/<pid>/environ`：那是 **exec 时**的环境快照，
    Ray 是在 actor 跑起来之后改 `os.environ` 的，改动不会反映进去 ——
    照着它判「CVD 没设」会得到错误结论（2026-08-17 我就这么错过一次）。
    """
    from verl.single_controller.base.worker import Worker

    orig = Worker._setup_env_cuda_visible_devices

    def probed(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            import torch

            from verl.utils.ray_utils import ray_noset_visible_devices
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<未设置>")
            dev = torch.cuda.current_device()
            bus = torch.cuda.get_device_properties(dev).pci_bus_id
            print(f"[device-probe] pid={os.getpid()} rank={os.environ.get('RANK')} "
                  f"local_rank={os.environ.get('LOCAL_RANK')} CVD={cvd} "
                  f"current_device={dev} 物理总线=0x{bus:02x} "
                  f"noset={ray_noset_visible_devices()}", flush=True)
        except Exception as e:  # 探针绝不能弄挂训练
            print(f"[device-probe] 失败: {e}", flush=True)
        return r

    Worker._setup_env_cuda_visible_devices = probed
    print("[verl-patch] device-probe 已挂上", flush=True)


def apply(mode: str) -> None:
    """按模式打对应的补丁。必须在 verl 的 main 被 import 之前调。

    ⚠️ 这里打的只覆盖 driver 进程。worker 进程靠 `setup_worker`（见上），
    由 `launch_rl` 通过 `ray_kwargs.ray_init.runtime_env.worker_process_setup_hook` 挂上。
    """
    if mode == "one_step_off":
        _patch_one_step_off_dump_executor()
    if mode == "fully_async":
        _patch_fsdp_cpu_copy_for_ddp()
    # ★ NVTX 阶段标注要**两侧都打**：range 打在 driver（阶段边界在这），
    #   kernel 跑在 worker（setup_worker 那边挂）。两边靠 nsys 的同一条时间轴对齐。
    if os.environ.get("SYNCOPATE_NVTX") == "1":
        for _m in ("verl.utils.profiler.performance",
                   "verl.utils.debug",
                   "verl.trainer.ppo.ray_trainer",
                   "verl.experimental.fully_async_policy.fully_async_trainer",
                   "verl.experimental.fully_async_policy.fully_async_rollouter",
                   "verl.experimental.one_step_off_policy.ray_trainer"):
            _defer_until_imported(_m, _patch_nvtx_timers_or_rebind)
