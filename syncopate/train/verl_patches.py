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
    #    ⇒ 教训：判据行只许写观测，不许写断言（00-START §6 变种②）。
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
    # ★ E21：默认**开启**（这是正确性修复，不是可选优化）。设 =0 可关掉做对照。
    if os.environ.get("SYNCOPATE_FSDP_DDP_FIX", "1") == "1":
        _defer_until_imported("torch.distributed.fsdp", _patch_fsdp_degenerate_mesh)
    if os.environ.get("SYNCOPATE_DDP_PROBE") == "1":
        _defer_until_imported("verl.workers.engine.fsdp.transformer_impl", _patch_ddp_sync_probe)
    if os.environ.get("SYNCOPATE_GRAD_PROBE") == "1":
        _defer_until_imported("verl.workers.engine.fsdp.transformer_impl", _patch_grad_probe)
    # ★ E14 批2-①：torch profiler 抓 update_actor 同步调用点（探针，默认关）
    if os.environ.get("SYNCOPATE_TORCH_PROF"):
        _defer_until_imported("verl.workers.engine_workers", _patch_torch_prof)
    # ⛔⛔ 2026-08-18 修：这两行原来**嵌在 `_patch_grad_probe()` 的函数体末尾** ——
    #    也就是只有同时开 `SYNCOPATE_GRAD_PROBE=1` 才会装上分步计时。
    #    后果：B15 那一跑设了 `SYNCOPATE_SYNC_TIMING=1`，**一行判据都没打**，
    #    而日志里也没有任何东西说它没生效 ⇒ 白跑一次（8 分钟 GPU）。
    #    ★ 又一次「机制在但没接上」，这次是**缩进层级**造成的：
    #      代码在、环境变量在、判据行也在，只是那段代码活在另一个开关底下。
    #    ⇒ 判据行**必须能独立触发**，不能挂在别的开关的成功路径上。
    if os.environ.get("SYNCOPATE_SYNC_TIMING") == "1":
        _defer_until_imported("verl.checkpoint_engine.base", _patch_sync_step_timing)
    # 0-B：独立开关，**不挂在别的开关的成功路径上**（上面那段注释记的就是这个教训）
    # ★ E22 修法①：先做成开关（验证过再谈默认值）。它同时改 trainer 与 rollout 两侧，
    #   两个模块都要等到被 import 之后再打。
    # ★ E22 修法①：**默认开启**（2026-08-18）—— 这是正确性修复，不是可选优化。
    #   不开 = disaggregated 下每次只推冻结基座、rollout 的策略永远停在起点，而且**不报错**。
    #   设 =0 可关掉做对照（比如复现坏基线）。
    if os.environ.get("SYNCOPATE_LORA_ADAPTER_SYNC", "1") == "1":
        for _m in ("verl.checkpoint_engine.base", "verl.workers.engine_workers"):
            _defer_until_imported(_m, _patch_lora_adapter_sync)
    # ★ E29（2026-08-20）：LoRA 下 ckpt 只存可训练部分（save 55 s → 秒级）。默认开；
    #   =0 回退全量（对照/全参训练均安全：无 lora 键时补丁自动回退，不靠开关兜底）。
    if os.environ.get("SYNCOPATE_CKPT_LORA_ONLY", "1") == "1":
        _defer_until_imported("verl.utils.checkpoint.fsdp_checkpoint_manager",
                              _patch_lora_only_ckpt)
    if os.environ.get("SYNCOPATE_OPT_STEP_PROBE") == "1":
        _defer_until_imported("verl.workers.engine.fsdp.transformer_impl", _patch_opt_step_counter)
    if os.environ.get("SYNCOPATE_SYNC_PAYLOAD") == "1":
        _defer_until_imported("verl.checkpoint_engine.nccl_checkpoint_engine",
                              _patch_sync_payload_probe)
        _defer_until_imported("verl.workers.rollout.vllm_rollout.vllm_async_server",
                              _patch_vllm_lora_probe)
    if os.environ.get("SYNCOPATE_PREFIX_GROUPER") == "1":
        _patch_prefix_grouper()
        # ★ E14 §4.7-②：PG 的 repeat_interleave 免同步修理（默认开，=0 做 A/B 对照）
        if os.environ.get("SYNCOPATE_FIX_PG_RI", "1") == "1":
            _defer_until_imported("prefix_grouper", _fix_pg_repeat_interleave)
    if os.environ.get("SYNCOPATE_FSDP_ALIGN") == "1":
        _defer_until_imported("torch.distributed.fsdp._flat_param", _patch_fsdp_shard_alignment)


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


# ★★★ A17（2026-08-18）：把 FSDP1 的分片补齐到 **16 字节**，验证「改这一行就好了」。
#
# 背景见 `docs/upstream/pytorch-fsdp-16b-alignment.md`：
#   FSDP1 的 `FlatParamHandle._get_unpadded_shard` 只把 flat parameter 切成
#   **每 rank 元素数相等**的块，**不管块的字节数是不是 16 的倍数**；
#   而 NCCL 的 Simple kernel 在非 16 倍数时**整段**退化成标量搬运 ⇒ all_gather 掉 12×。
#   实测 Qwen3-4B 的一层：每 rank 67,287,212 B（%16=12），**补 4 个字节就回到 13.4 GB/s**。
#
# 本补丁做的事：在切分**之前**把 flat parameter 补到 `world_size × (16/itemsize)` 的倍数，
# 于是每个 rank 的块天然是 16 字节的倍数。
#
# ⚠️⚠️ 这是**实验性补丁**，只在 `SYNCOPATE_FSDP_ALIGN=1` 时装：
#   ① 多出来的是尾部零填充，FSDP 本来就有「尾部 padding 不属于任何参数」的语义
#      （`numel_to_pad` 就是干这个的）⇒ 语义上安全；
#   ② 但它**改变了 flat parameter 的尺寸**，而尺寸在别处也被算过
#      ⇒ **必须用「grad_norm / loss 与未打补丁的跑对得上」当判据**，不能只看变快了。
#
# ⛔⛔ 2026-08-19 实测：**上面②那句担心兑现了 —— 这一层是错的层，别再用这个补丁。**
#   A17 双臂第一跑：aligned 臂在**第一次反向**就炸 ——
#     `attempting to assign a gradient of size '[27307]' to a tensor of size '[27308]'`
#   根因：本补丁只改了 `_get_unpadded_shard`（初始化分片走它 ⇒ 27308），
#   而梯度分片的尺寸由另一条路按**原始** numel 均分（81921÷3=27307，恰好整除不触发 pad）
#   ⇒ 两处各算各的，差 1 个元素。**运行时在分片函数里 pad，改不全所有算尺寸的地方。**
#   ✅ 正确的层：**flat param 构造端**（`_flat_param.py:737/:853` 那两处孪生的
#   「补到 world 整除」，把除数换成 `world_size × (16 // itemsize)`）——
#   总 numel 自身对齐后，所有下游路径自然一致。已以源码 diff 形式验证（A17-v2），
#   这也正是拟提给上游的那一行。见 `docs/upstream/fsdp-shard-alignment/`。
def _patch_fsdp_shard_alignment() -> None:
    import torch.nn.functional as F
    from torch.distributed.fsdp import _flat_param as fp

    H = fp.FlatParamHandle
    if getattr(H, "_syncopate_aligned", False):
        return
    orig = H._get_unpadded_shard.__func__ if hasattr(H._get_unpadded_shard, "__func__") \
        else H._get_unpadded_shard

    def aligned(tensor, rank, world_size):
        # 每 rank 要多少个元素才凑满 16 字节（bf16/fp16 → 8，fp32 → 4）
        per16 = max(1, 16 // tensor.element_size())
        align = world_size * per16
        n = tensor.numel()
        if n % align:
            tensor = F.pad(tensor.reshape(-1), [0, align - (n % align)])
        return orig(tensor, rank, world_size)

    H._get_unpadded_shard = staticmethod(aligned)
    H._syncopate_aligned = True
    print("[verl-patch] FSDP 分片按 16 字节对齐已启用（SYNCOPATE_FSDP_ALIGN=1）"
          " —— 判据是 grad_norm/loss 要和未打补丁的一致", flush=True)


# ★★★ DDP 同步探针（2026-08-18）：三个 trainer rank 到底有没有在同步梯度。
#
# 起因：写 ckpt→adapter 转换器时，"三个 rank 的 LoRA 应该相同"这条断言当场炸了。
# 静态证据（读存下来的 ckpt）：
#   基座权重跨 rank **完全相同**（相对差 0.0）⇒ 是复制不是分片，排除"读错分片"
#   lora_A 范数几乎一样但相对差 **1.415**（≈√2）⇒ 各自随机初始化、从没广播
#   lora_B（**零初始化**）范数各不相同、相对差 1.3 ⇒ **更新历史不同**
#   Adam 的 exp_avg_sq 相对差 **99%**            ⇒ **梯度历史不同**
# ⇒ 静态证据指向「三个 rank 各训各的」，但那是**保存下来的状态**，
#   有可能只是保存路径各存各的、训练本身是同步的。**必须在运行时直接看。**
#
# 本探针在**每次 optimizer step 之前**打印本 rank 的：
#   ① 某个 lora_A / lora_B 的权重范数   ② 它们的**梯度**范数
# 三个 rank 打出不同的梯度范数 ⇒ **梯度没有 all-reduce**，实锤。
def _patch_ddp_sync_probe() -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine, "_syncopate_ddp_probe", False):
        return
    orig = FSDPEngine.optimizer_step
    state = {"n": 0}

    def probed(self):
        import torch
        import torch.distributed as dist

        state["n"] += 1
        if state["n"] <= 4:                      # 只打前 4 次，不刷屏
            rank = dist.get_rank() if dist.is_initialized() else -1
            picks = []
            for name, prm in self.module.named_parameters():
                if not prm.requires_grad:
                    continue
                if ("lora_A" in name or "lora_B" in name) and "layers.0." in name:
                    g = prm.grad
                    picks.append((name.split("layers.")[-1][:28],
                                  prm.detach().float().norm().item(),
                                  float("nan") if g is None else g.detach().float().norm().item()))
                if len(picks) >= 2:
                    break
            for nm, wn, gn in picks:
                print(f"[ddp-probe] step={state['n']} rank={rank} {nm}  "
                      f"权重范数={wn:.6f}  **梯度范数={gn:.6e}**", flush=True)
        return orig(self)

    FSDPEngine.optimizer_step = probed
    FSDPEngine._syncopate_ddp_probe = True
    print("[verl-patch] DDP 同步探针已挂上（SYNCOPATE_DDP_PROBE=1）—— "
          "判据：三个 rank 的**梯度范数**若不同 ⇒ 梯度没有 all-reduce", flush=True)


# ★★★ E21 修复（2026-08-18）：`fsdp_size=1` 下梯度不同步。
#
# 根因链（完整证据见 docs/infra_exp/E21-ddp-not-syncing.md 与 docs/upstream/）：
#   verl:    fsdp_size=1, world_size=3 ⇒ mesh (3,1) ["ddp","fsdp"] ⇒ 二维 ⇒ HYBRID_SHARD
#   PyTorch: 见分片维只有 1 个 rank ⇒ 降级成 NO_SHARD（只打一行 UserWarning）
#            ⇒ 而梯度归约走的是**那个大小为 1 的组** ⇒ 空操作
#   ⇒ 三个 rank 各训各的 LoRA，训练照常跑完、所有指标正常。**静默失效。**
#
# 修法（脱离 verl 的最小复现已验证，见 scripts/repro_fsdp_hybrid_nosync.py）：
#   退化网格下改用 `NO_SHARD` + **默认进程组**（不传 device_mesh）
#   ⇒ 实测与纯 DDP 打出**逐位相同**的梯度。
#
# ⚠️ 为什么拦 FSDP 的构造而不是改 verl 的 `create_device_mesh`：
#   `self.device_mesh` 在 verl 里还被别处用（fsdp2 路径、state_dict 加载）
#   ⇒ 把它整个置空风险大。这里**只在"HYBRID_SHARD + 分片维为 1"这一种情况下**改写两个入参，
#   其余一律原样放行 —— 改动面最小。
def _patch_fsdp_degenerate_mesh() -> None:
    import torch.distributed as dist
    import torch.distributed.fsdp as tfsdp
    from torch.distributed.fsdp import ShardingStrategy

    cls = tfsdp.FullyShardedDataParallel
    if getattr(cls, "_syncopate_degenerate_fix", False):
        return
    orig_init = cls.__init__
    hybrid = {ShardingStrategy.HYBRID_SHARD, getattr(ShardingStrategy, "_HYBRID_SHARD_ZERO2", None)}

    def patched(self, module=None, *args, **kwargs):
        mesh = kwargs.get("device_mesh")
        strat = kwargs.get("sharding_strategy")
        if mesh is not None and strat in hybrid:
            shard_dim = None
            try:
                shard_dim = mesh.size(mesh.ndim - 1)       # 最后一维是 "fsdp"（分片维）
            except Exception:                               # noqa: BLE001
                pass
            if shard_dim == 1:
                # ★ 0-A 引入的常驻断言（2026-08-18）：把 device_mesh 置空 ⇒ FSDP 改用**默认进程组**
                #   来归约梯度。于是「FSDP 除以谁」这件事的来源，从 mesh 变成了默认进程组
                #   —— 它必须覆盖**同一批 rank**，否则归约的分母就和 verl 乘的 dp_size 对不上。
                #   （E21 的形状就是"两层各自合理、缝里掉东西"，所以这里把前提写成断言而不是注释。）
                world = dist.get_world_size() if dist.is_initialized() else None
                if world is not None and world != mesh.size():
                    raise RuntimeError(
                        f"★ E21 修复的前提不成立：默认进程组有 {world} 个 rank，"
                        f"而被替换掉的 device_mesh 覆盖 {mesh.size()} 个 ⇒ 归约的分母会和 verl 的 "
                        f"dp_size 对不上（梯度会系统性偏大/偏小）。停下来查，别让它静默跑过去。"
                    )
                kwargs["sharding_strategy"] = ShardingStrategy.NO_SHARD
                kwargs["device_mesh"] = None
                print(f"[verl-patch] ★ E21 修复生效：检测到退化网格（{strat}，分片维=1）"
                      f" ⇒ 改用 NO_SHARD + 默认进程组（world={world}），梯度才会真正 all-reduce",
                      flush=True)
        return orig_init(self, module, *args, **kwargs)

    cls.__init__ = patched
    cls._syncopate_degenerate_fix = True
    print("[verl-patch] E21 退化网格修复已装（SYNCOPATE_FSDP_DDP_FIX=1）", flush=True)


# ★★ 0-B 探针（`SYNCOPATE_SYNC_PAYLOAD=1`）：权重同步**推的到底是什么**。
#
# 起因（E21 之后的同族排查）：读发送侧代码发现 disaggregated（fully_async）那条路
#   `engine_workers.py:698`  per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
# **不传任何参数** ⇒ `base_sync_done=False` ⇒ `collect_lora_params` 里那段会
# **显式跳过所有含 `lora_` 的张量**（`fsdp_utils.py:705`）。
# 而 colocate（naive）那条路**调了两次**：先基座、再 adapter。
#   ⇒ [推断] fully_async 可能每次只推基座、从不推 LoRA。
#   ⇒ 离线已验证该分支的行为（`scripts/probe_weight_sync_payload.py`：0 个 lora_ 张量），
#      **但"分支这样"不等于"真实跑就这样"** —— 本探针就是把它变成实测。
#
# 判据行（"某集合应当完整"型，不设阈值）：每次同步打一行
#   张量个数 / 总字节 / 其中含 lora_ 的个数
#   ⇒ **含 lora_ 的必须 > 0**（或整体字节数 ≈ 基座 ⇒ 说明是 merge 后的全量）。
#   ⇒ 若恒为 0 且字节 ≈ 基座大小 ⇒ **rollout 拿到的策略永远是起点**。
def _patch_sync_payload_probe() -> None:
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine as E

    if getattr(E, "_syncopate_payload_probe", False):
        return
    orig_send = E.send_weights

    def _tee(weights, stats):
        """流式穿过：只累计计数，**不把张量攒下来**（攒下来会改内存行为，探针就成了变量）。"""
        for name, tensor in weights:
            stats["n"] += 1
            try:
                stats["bytes"] += tensor.numel() * tensor.element_size()
            except Exception:                       # noqa: BLE001  探针不许拖垮训练
                pass
            if "lora_" in str(name).lower():
                stats["lora"] += 1
            elif stats["first"] is None:
                stats["first"] = str(name)
            # ★ 追加判据（"两个东西应当相同"型）：盯住一个**被 LoRA 适配过**的层。
            #   若推出去的值逐次不变、且等于磁盘上起点模型的那一份
            #   ⇒ 推的是**冻结基座**，LoRA 的增量根本没上路。
            stats["names"].append(str(name))
            if str(name) == stats["watch"]:
                stats["watch_norm"] = tensor.detach().float().norm().item()
            yield name, tensor

    async def probed(self, weights, *args, **kwargs):
        stats = {"n": 0, "bytes": 0, "lora": 0, "first": None, "names": [],
                 "watch": os.environ.get("SYNCOPATE_SYNC_WATCH",
                                         "model.layers.0.self_attn.q_proj.weight"),
                 "watch_norm": None}
        try:
            return await orig_send(self, _tee(weights, stats), *args, **kwargs)
        finally:
            verdict = "✅ 含 LoRA" if stats["lora"] else "🔴 **一个 lora_ 都没有**"
            print(f"[sync-payload] 本次同步推出去：{stats['n']} 个张量 / "
                  f"{stats['bytes'] / 2**20:,.1f} MiB / 其中 lora_ {stats['lora']} 个 ⇒ {verdict}"
                  f"（首个非 lora 张量名：{stats['first']}）", flush=True)
            # ⛔ 判据没绑上时**绝不能打结论** —— 空判据被读成通过，是本项目栽过的坑。
            if stats["watch_norm"] is None:
                hit = [n for n in stats["names"] if "q_proj" in n][:3]
                print(f"[sync-payload] 🔴 判据无效：盯住的层 `{stats['watch']}` "
                      f"在 {stats['n']} 个张量里**一个都没匹配上** ⇒ 这一行不能当通过读。"
                      f" 实际名字里含 q_proj 的样例：{hit}", flush=True)
            else:
                # ⛔ 第二次犯同一个错：上一版这里**直接打了结论**（"与磁盘起点相同"），
                #    而根本没做比较 —— merge=True 那跑打出 75.3974（明明不同）却照样说"相同"。
                #    ⇒ 判据必须**真的比**，比不了就说比不了。
                ref = os.environ.get("SYNCOPATE_SYNC_REF")
                if ref:
                    d = abs(stats["watch_norm"] - float(ref)) / max(abs(float(ref)), 1e-12)
                    ok = d > 1e-6      # 与起点**不同**才说明增量上路了
                    mark = ("✅ 与起点不同 ⇒ 增量已随权重推出去"
                            if ok else "🔴 **与起点逐位相同** ⇒ 推的是冻结基座，增量没上路")
                    print(f"[sync-payload] 盯住的层 {stats['watch']} ‖W‖={stats['watch_norm']:.6f}"
                          f"　起点参考 {float(ref):.6f}　相对差 {d:.3e} ⇒ {mark}", flush=True)
                else:
                    print(f"[sync-payload] 盯住的层 {stats['watch']} ‖W‖={stats['watch_norm']:.6f}"
                          f"（未给 SYNCOPATE_SYNC_REF ⇒ **只报数、不下判定**）", flush=True)

    E.send_weights = probed
    E._syncopate_payload_probe = True
    print("[verl-patch] 0-B 权重同步载荷探针已装（SYNCOPATE_SYNC_PAYLOAD=1）—— "
          "判据：每次同步打一行「张量数 / 字节 / lora_ 个数」", flush=True)

# ★★★ E22 修法① · 让 disaggregated（fully_async / one_step_off）真正把 **adapter** 推给 rollout
#
# 背景（完整证据见 docs/infra_exp/E22-lora-never-synced.md）：
#   engine_workers.py:698 在 disaggregated 分支上**只调一次** `get_per_tensor_param()` 且不传参
#   ⇒ `base_sync_done=False` ⇒ `collect_lora_params` **显式跳过所有 lora_ 张量**
#   ⇒ 每次同步推 8.4 GB **冻结基座**，adapter 一个字节都没推 ⇒ rollout 的策略永远是 π₀。
#   （colocate 那条路调两次：先基座、再 adapter，是对的。）
#
# 为什么能自己补：**两端的能力都在，断的只是中间那段传参**（E22 §6.3 查实）
#   trainer 侧  get_per_tensor_param(base_sync_done=True) ⇒ 直接吐 LoRA 张量 + peft_config   ✅
#   🔴 断点     CheckpointEngineWorker.update_weights 签名里没有 peft_config（base.py:323）
#   rollout 侧  TensorLoRARequest(lora_tensors=…) + add_lora ⇒ 能直接从张量装 LoRA          ✅
#               （vllm_rollout/utils.py:262，colocate 每次同步都在用它）
#   而且 `lora_as_adapter` 在 merge=False 下**本来就是 True**（vllm_async_server.py:186）
#   —— 生成时会查 `list_loras()`，只是那里一直是空的。
#
# ⚠️ 为什么不改 `lora.merge=True` 了事：R0-b 实测那条 bf16 合并会毁掉 adapter **一半**的作用
#    （logprob 偏移中位 1.717e-02 = adapter 自身作用的 50%，是引擎地板的 50×）。
#
# 两侧各自维护「基座推过了没有」的计数 —— 它们由 `CheckpointEngineManager.update_weights`
# **同一步里成对调用**（base.py:497-500），所以天然同步；判据行会把两侧的状态都打出来。
def _patch_lora_adapter_sync() -> None:
    import torch.distributed as dist   # noqa: F401  （保持与本模块其它补丁一致的导入风格）

    # ⚠️ 两侧活在**不同的进程**里（trainer 是 WorkerDict，rollout 是 CheckpointEngineWorker），
    #    模块的 import 时机也不同 ⇒ 各自 try/except、各自幂等，并且**两个模块任一被 import 都会来补一次**。
    #    （不这样的话会重演"补丁只在一半的模式/进程里接上"那个形状 —— 本项目已记过五次。）
    done_any = False

    # ---- ① trainer 侧：首次推基座，之后推 adapter ----
    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker as ARW
    except Exception:                                     # noqa: BLE001  本进程没有 trainer 侧
        ARW = None
    if ARW is not None and not getattr(ARW, "_syncopate_adapter_sync", False):
        orig_update = ARW.update_weights

        async def trainer_update_weights(self, global_steps: int = None, mode: str = "auto"):
            effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend
            if effective_mode == "naive":
                return await orig_update(self, global_steps=global_steps, mode=mode)

            # ⛔ 互斥守卫：`lora.merge=True` 会让 get_per_tensor_param 走"合并后全量"分支
            #    （返回 399 个基座张量、peft_config=None），而 rollout 侧仍会按 adapter 装载
            #    ⇒ 把整份基座当 adapter 喂给 add_lora。**宁可启动就报错，也不要静默跑歪。**
            if getattr(self.actor.engine, "model_config", None) is not None and \
                    self.actor.engine.model_config.lora.get("merge", False):
                raise ValueError(
                    "★ `model.lora.merge=True` 与 E22 修法①（推 adapter）互斥：\n"
                    "  修法①每次推的是 adapter 张量，而 merge=True 会让引擎吐出合并后的全量权重。\n"
                    "  ⇒ 二选一：去掉 `--lora-merge`（**推荐**，R0-b 实测 bf16 合并毁掉 adapter 一半作用），\n"
                    "         或设 SYNCOPATE_LORA_ADAPTER_SYNC=0 退回合并模式。")
            base_done = getattr(self, "_syncopate_base_sync_done", False)
            per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
                base_sync_done=base_done)
            await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            self._syncopate_base_sync_done = True
            # ★ V1 数值判据：把 trainer **真实**的 peft_config 打出来，供与 rollout 侧
            #   **重建**的那份逐字段比。scaling = lora_alpha / r ——
            #   这个数错了的话，一切表象都正常（张量数对、大小对、list_loras 非空），
            #   **而策略被整体缩放**。这是本补丁唯一"重建"的环节，也是唯一可能错的地方。
            # ⚠️ peft_config 的类型不保证（LoraConfig / dict / DictConfig 都可能）——
            #    第一版探针只用 getattr 读，全打成 None ⇒ **判据自己失效了**。
            #    ⇒ 改成三路都试，并且**把类型打出来**；读不到就明说读不到，不许打成"无"。
            def _fld(o, k):
                if isinstance(o, dict):
                    return o.get(k)
                v = getattr(o, k, None)
                return v
            pc = peft_config
            desc = "None（引擎没给）"
            if pc is not None:
                r_, a_, tm = _fld(pc, "r"), _fld(pc, "lora_alpha"), _fld(pc, "target_modules")
                scale = (a_ / r_) if (r_ and a_) else None
                desc = (f"[{type(pc).__name__}] r={r_} alpha={a_} **scaling={scale}** "
                        f"target={sorted(tm) if tm else tm}")
            # ★ 同时打**两侧共同的源头** model_config —— rollout 侧就是拿它重建的，
            #   两边同源才谈得上"重建是对的"。
            mc = getattr(self.actor.engine, "model_config", None)
            if mc is not None:
                desc += (f" ｜ model_config: rank={getattr(mc, 'lora_rank', None)} "
                         f"alpha={getattr(mc, 'lora_alpha', None)}")
            print(f"[adapter-sync] trainer 侧：{'adapter' if base_done else '基座（首次）'}"
                  f" 已发出 · peft_config(真实)= {desc}", flush=True)
            return

        trainer_update_weights.__dict__.update(orig_update.__dict__)   # ★ 保住 @register 的元数据
        ARW.update_weights = trainer_update_weights
        ARW._syncopate_adapter_sync = True
        done_any = True
        print("[verl-patch] ★ E22 修法① · trainer 侧已装", flush=True)

    # ---- ② rollout 侧：把 peft_config / base_sync_done 传下去 ----
    try:
        from verl.checkpoint_engine.base import CheckpointEngineWorker as CEW
    except Exception:                                     # noqa: BLE001  本进程没有 rollout 侧
        CEW = None
    if CEW is not None and not getattr(CEW, "_syncopate_adapter_sync", False):
        orig_ce_update = CEW.update_weights

        def _peft_config_dict(model_config):
            """在 rollout 侧就地重建 peft_config。

            ★ 刻意**不跨进程传** PEFT 对象：`model_config` 两侧同源（同一份 Hydra 配置），
              而 `PEFTHelper.from_dict` 只需要 r / lora_alpha / target_modules 三个必填字段
              （vllm/lora/peft_helper.py:28-30，多余的键会被过滤掉）。
            ⇒ 少一条序列化路径就少一个静默失败的地方。
            """
            from verl.utils.py_functional import convert_to_regular_types   # ⚠️ 不在 utils.model 里

            rank = getattr(model_config, "lora_rank", 0) or 0
            if rank <= 0:
                return None
            cfg = {
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": getattr(model_config, "lora_alpha", 2 * rank),
                "target_modules": convert_to_regular_types(model_config.target_modules),
                "bias": "none",
            }
            exclude = convert_to_regular_types(getattr(model_config, "exclude_modules", None))
            if exclude:
                cfg["exclude_modules"] = exclude
            return cfg

        async def ce_update_weights(self, global_steps: int = None):
            weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
            base_done = getattr(self, "_syncopate_base_sync_done", False)
            kwargs = {}
            if base_done:
                pc = _peft_config_dict(self.model_config)
                if pc is not None:
                    kwargs = {"peft_config": pc, "base_sync_done": True}
            await self.server_adapter.update_weights(weights, global_steps=global_steps, **kwargs)
            self._syncopate_base_sync_done = True
            pc = kwargs.get("peft_config")
            desc = f"未带（首次送基座）｜ model_config: rank={getattr(self.model_config,'lora_rank',None)} " \
                   f"alpha={getattr(self.model_config,'lora_alpha',None)}"
            if pc:
                scale = pc["lora_alpha"] / pc["r"] if pc.get("r") else None
                tm = pc.get("target_modules")
                # ⚠️ 别对字符串做 sorted() —— "all-linear" 会被拆成字符列表，
                #    看起来像"重建错了"，其实是**探针的显示 bug**（2026-08-18 自己撞的）。
                tm_s = sorted(tm) if isinstance(tm, (list, tuple, set)) else repr(tm)
                desc = (f"r={pc['r']} alpha={pc['lora_alpha']} **scaling={scale}** target={tm_s}")
            print(f"[adapter-sync] rollout 侧：按 "
                  f"{'**adapter**' if kwargs else '基座（首次）'} 装载"
                  f"（base_done={base_done}）· peft_config(重建)= {desc}", flush=True)

        ce_update_weights.__dict__.update(orig_ce_update.__dict__)   # ★ 同上，@register 必须保住
        CEW.update_weights = ce_update_weights
        CEW._syncopate_adapter_sync = True
        done_any = True
        print("[verl-patch] ★ E22 修法① · rollout 侧已装", flush=True)

    if done_any:
        print("[verl-patch] ★ E22 修法①（SYNCOPATE_LORA_ADAPTER_SYNC=1）—— 判据："
              "第 2 次同步起载荷应从 ~8.4 GB 掉到 ~132 MB、含 lora_ 的张量数 > 0", flush=True)

# ★ 0-B 的最后一环：**vLLM 引擎里到底有没有那个 adapter**。
#   前面的判据只能证明"推出去了 / 传到 rollout 侧了"，而 vLLM 的 worker 是它自己 spawn 的
#   子进程，我们的钩子够不着；但 `vLLMHttpServer` 是 Ray actor，够得着。
#   ⇒ 挂在每次权重同步收尾都会调的 `set_global_steps` 上，打一行 `list_loras()`。
#   判据：**adapter 模式下这个集合必须非空**（vllm_async_server.py:527 就是拿它决定要不要用 LoRA 的）。
def _patch_vllm_lora_probe() -> None:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer as S

    if getattr(S, "_syncopate_lora_probe", False):
        return
    orig = S.set_global_steps

    async def probed(self, global_steps=None, *a, **kw):
        r = await orig(self, global_steps, *a, **kw) if orig.__code__.co_argcount > 1 \
            else await orig(self, *a, **kw)
        try:
            loras = await self.engine.list_loras()
            ok = "✅ 引擎里有 adapter" if loras else "🔴 **引擎里一个 adapter 都没有** ⇒ 生成用的是裸基座"
            print(f"[lora-probe] step={global_steps} engine.list_loras()={loras} ⇒ {ok}", flush=True)
        except Exception as exc:                      # noqa: BLE001  探针不许拖垮训练
            print(f"[lora-probe] ⚠️ 读不到 list_loras：{exc}", flush=True)
        return r

    S.set_global_steps = probed
    S._syncopate_lora_probe = True
    print("[verl-patch] vLLM adapter 探针已装 —— 判据：每次同步后 list_loras() 必须非空", flush=True)


# ★ 优化器步数计数器（`SYNCOPATE_OPT_STEP_PROBE=1`）
#
# 起因（E20 §7.8）：**产物里没有任何东西能告诉你真实的优化器更新次数。**
#   training/global_step  = fit step
#   rollout_dumps 文件数  = dump 次数（只在 mini_batch==train_batch 时**碰巧**等于更新次数）
#   metric 记录次数        = param_version
#   ⇒ 三个都不是它。而 E20 原因②（"一个 epoch 只更新 110 次"）**整条结论都建立在这个数上**。
# ⇒ 直接数 `optimizer_step` 被调了几次 —— 这是唯一不会因配置变化而变成另一件事的口径。
def _patch_opt_step_counter() -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as E

    if getattr(E, "_syncopate_opt_counter", False):
        return
    orig = E.optimizer_step

    def counted(self, *a, **kw):
        n = getattr(self, "_syncopate_opt_steps", 0) + 1
        self._syncopate_opt_steps = n
        r = orig(self, *a, **kw)
        # ⛔ 第一版只在 `n<=5 or n%20==0` 时打 ⇒ **最终值可能永远打不出来**
        #    （A1 实测：24 个 fit step，最后打的是 20 —— 真实值到底是 20 还是 24 分不出来）
        #    ★ 又一次「判据没打出我要的那个数」。⇒ **每次都打**，Ray 会折叠重复行，不会刷屏。
        print(f"[opt-step] optimizer_step #{n}", flush=True)
        return r

    E.optimizer_step = counted
    E._syncopate_opt_counter = True
    print("[verl-patch] 优化器步数计数器已装（SYNCOPATE_OPT_STEP_PROBE=1）—— "
          "判据：**真实**更新次数，不是 fit step / dump 数 / param_version", flush=True)


def _patch_pool_sampler() -> None:
    """在 worker 进程里装动态分池的 sampler 补丁。判据行由 DynamicPoolSampler 自己打。"""
    from syncopate.train.main_ppo_pool import install_sampler_patch

    install_sampler_patch()
    print("[verl-patch] worker 进程：动态分池 sampler 已装", flush=True)


def filter_lora_state(sd: dict) -> "dict | None":
    """E29：从全量 state_dict 里挑出 LoRA 键。**没有 lora_ 键返回 None（全参训练，
    调用方必须回退全量）**——兜底必须是对的那个，不能把全参模型静默存成空字典。"""
    lora = {k: v for k, v in sd.items() if "lora_" in k}
    return lora or None


def merge_lora_into(full: dict, lora: dict) -> dict:
    """E29：把 lora-only ckpt 合回「基座已就位」的当前 state_dict。
    键空间必须一致——ckpt 里有当前模型没有的键说明模型结构变了，硬失败不猜。"""
    missing = [k for k in lora if k not in full]
    if missing:
        raise RuntimeError(f"[ckpt-lora] lora-only ckpt 有 {len(missing)} 个键不在当前模型里"
                           f"（如 {missing[:3]}）—— 模型结构或 target_modules 变了，拒绝合成加载")
    full.update(lora)
    return full


def assert_loaded_matches(after: dict, expect: dict) -> int:
    """E29 数值判据（E22 §7.1「送到了≠送对了」同款）：load 之后**逐位**比回 ckpt。
    比不上就炸——加载了错的权重继续训比崩溃贵得多。返回校验的键数。"""
    import torch

    bad = [k for k in expect
           if k not in after or not torch.equal(after[k].detach().cpu(), expect[k].cpu())]
    if bad:
        raise RuntimeError(f"[ckpt-lora] 🔴 数值校验失败：{len(bad)}/{len(expect)} 个 lora 键"
                           f" load 后与 ckpt 不一致（如 {bad[:3]}）—— 拒绝继续")
    return len(expect)


def _patch_lora_only_ckpt() -> None:
    """★ E29（2026-08-20）：LoRA 训练下 ckpt 只存可训练部分。

    动机（cand_v13r2_e1 实测）：save_checkpoint 占步 **19.5%**、单次 ~55 s——每 rank
    全量 state_dict 8.5 GB × 3 rank，**97% 是与基座逐字节相同的冻结权重**；
    磁盘直写 3.9 GB/s ⇒ 时间不在盘，在**字节量驱动的序列化 + D2H 拷贝**。砍字节=砍时间。
    verl 0.8.0 无此功能（checkpoint_contents 只能整类开关）⇒ 上游第 5 包的素材。

    修法（save/load 两端都动，只动 model 分片；optimizer 状态本来就只含可训练参数）：
      save  拦在 `self.model.state_dict` 上按 lora_ 键过滤；无 lora 键回退全量
      load  检测到 lora-only ckpt ⇒ 取当前（基座已由模型初始化就位）state_dict、
            更新 lora 键、整载 —— 不赌 FSDP 分片 load_state_dict 的 strict=False 语义
    ⚠️ 装在 **trainer WorkerDict 进程**（守则②）；`SYNCOPATE_CKPT_LORA_ONLY=0` 关掉回全量。
    """
    from verl.utils.checkpoint import fsdp_checkpoint_manager as M

    T = M.FSDPCheckpointManager
    if getattr(T, "_syncopate_lora_only_ckpt", False):
        return
    orig_save, orig_load = T.save_checkpoint, T.load_checkpoint

    def save_filtered(self, *a, **kw):
        orig_sd_fn = self.model.state_dict

        def filtered(*sa, **skw):
            sd = orig_sd_fn(*sa, **skw)
            lora = filter_lora_state(sd)
            if lora is None:
                print("[ckpt-lora] ⚠️ 没有 lora_ 键（全参训练？）⇒ 回退全量保存", flush=True)
                return sd
            mb = sum(v.numel() * v.element_size() for v in lora.values()) / 2**20
            print(f"[ckpt-lora] 模型分片按 lora_ 过滤：{len(lora)}/{len(sd)} 键，"
                  f"{mb:.0f} MB（全量 ≈8.5 GB）", flush=True)
            return lora

        self.model.state_dict = filtered        # 实例属性遮蔽，只影响本次 save
        try:
            return orig_save(self, *a, **kw)
        finally:
            del self.model.state_dict

    def load_merged(self, *a, **kw):
        orig_load_fn = self.model.load_state_dict

        def merged(sd, *la, **lkw):
            if isinstance(sd, dict) and sd and all("lora_" in k for k in sd):
                full = merge_lora_into(self.model.state_dict(), sd)
                print(f"[ckpt-lora] 检测到 lora-only ckpt ⇒ 合成加载（更新 {len(sd)} 键）",
                      flush=True)
                ret = orig_load_fn(full, *la, **lkw)
                # ★ 数值判据：load 完把内存里的 lora 键逐位比回 ckpt（仍在同一 FSDP ctx 内）。
                #   只在 resume 时发生，代价一次 state_dict()；不一致 = 硬失败。
                after = {k: v for k, v in self.model.state_dict().items() if k in sd}
                n = assert_loaded_matches(after, sd)
                print(f"[ckpt-lora] ✅ 数值校验：{n}/{len(sd)} 个 lora 键 load 后与 ckpt 逐位相同",
                      flush=True)
                return ret
            return orig_load_fn(sd, *la, **lkw)   # 旧全量 ckpt 原样加载，向后兼容

        self.model.load_state_dict = merged
        try:
            return orig_load(self, *a, **kw)
        finally:
            del self.model.load_state_dict

    T.save_checkpoint = save_filtered
    T.load_checkpoint = load_merged
    T._syncopate_lora_only_ckpt = True
    print("[verl-patch] E29 ckpt 只存 LoRA 已装（SYNCOPATE_CKPT_LORA_ONLY，默认开）——"
          "判据：save 时打 [ckpt-lora] 过滤行；resume 时打合成加载行", flush=True)


def _fix_pg_repeat_interleave() -> None:
    """★ E14 §4.7-②（SYNCOPATE_FIX_PG_RI，默认开）：PG 库 `batch_repeat_cat` 的
    `repeat_interleave(tensor)` 不带 `output_size` ⇒ 每次调用都要同步问 GPU 输出多大
    （实测 ×584/update_actor）。call site 恒为 cat_dim=1（沿序列拼接）⇒ 重复后的
    prefix 行数必须等于 `suffix.shape[0]`——**尺寸在 CPU 侧免费可得**。
    cat_dim=0 时该推导不成立，回退原路径（宁慢不错）。
    ⚠️ `from .utils import` 的第二个绑定名一起换（NVTX 补丁的同款教训）。"""
    import prefix_grouper
    from prefix_grouper import utils as pg_utils

    if getattr(pg_utils, "_syncopate_ri_fix", False):
        return
    import torch
    orig = pg_utils.batch_repeat_cat

    def fixed(prefix, suffix, cat_dim, num_samples):
        if cat_dim == 0:
            return orig(prefix, suffix, cat_dim, num_samples)
        return torch.cat(
            [prefix.repeat_interleave(num_samples.to(prefix.device), dim=0,
                                      output_size=suffix.shape[0]),
             suffix], dim=cat_dim)

    pg_utils.batch_repeat_cat = fixed
    prefix_grouper.batch_repeat_cat = fixed
    pg_utils._syncopate_ri_fix = True
    print("[verl-patch] E14-② PG repeat_interleave 已补 output_size（免同步取尺寸，×584→0）",
          flush=True)


def _patch_torch_prof() -> None:
    """★ E14 批2-①（探针，默认关）：torch profiler 抓 update_actor 的同步乒乓调用点。

    `SYNCOPATE_TORCH_PROF=<N>` ⇒ rank 0 的第 N 次 update_actor 整体包进
    torch.profiler（with_stack）：导出 chrome trace + 直接打印**同步类算子账本**——
    `aten::_local_scalar_dense`（= `.item()` 的真身）/ 显式 synchronize / D2H copy 的
    次数、耗时与 Python 调用栈。nsys 答不了"谁调的"（E14 §7：对 Ray trainer 还丢事件），
    这里在进程内记账，天然无截断。⚠️ 装在 WorkerDict 进程；@register 元数据按
    E12 §262 教训原样搬运，丢了组级方法直接消失。
    """
    import functools

    from verl.workers.engine_workers import ActorRolloutRefWorker as W

    if getattr(W, "_syncopate_torch_prof", False):
        return
    target = int(os.environ.get("SYNCOPATE_TORCH_PROF", "0") or 0)
    MAGIC_ATTR = "attrs_3141562937"
    orig = W.update_actor
    state = {"n": 0}

    @functools.wraps(orig)
    def profiled(self, *a, **kw):
        state["n"] += 1
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if state["n"] != target or rank != 0:
            return orig(self, *a, **kw)
        from torch.profiler import ProfilerActivity, profile
        print(f"[torch-prof] 抓取第 {target} 次 update_actor（rank 0，with_stack）…", flush=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_stack=True) as prof:
            out = orig(self, *a, **kw)
        from pathlib import Path
        outdir = Path(__file__).resolve().parents[2] / "logs" / "torchprof"
        outdir.mkdir(parents=True, exist_ok=True)
        trace = outdir / f"update_actor_call{target}.json"
        prof.export_chrome_trace(str(trace))
        # ① 全局 top 表（和 SFT H0 观测仪同款口径）
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
        # ② 同步类算子专表：次数 × 耗时 × 栈（乒乓元凶名单）
        ka = prof.key_averages(group_by_stack_n=6)
        rows = [r for r in ka if any(s in r.key for s in
                ("_local_scalar_dense", "Synchronize", "synchronize", "copy_", "to_copy", "item"))]
        rows.sort(key=lambda r: -r.self_cpu_time_total)
        print("[torch-prof] ── 同步类算子账本（top10）──", flush=True)
        for r in rows[:10]:
            stack = " <- ".join(s.split("/")[-1] for s in (r.stack or [])[:3])
            print(f"[torch-prof]   {r.key:<38} ×{r.count:>5}  cpu {r.self_cpu_time_total/1e3:8.1f} ms"
                  f"  | {stack}", flush=True)
        print(f"[torch-prof] ✅ trace → {trace}", flush=True)
        return out

    if hasattr(orig, MAGIC_ATTR):
        setattr(profiled, MAGIC_ATTR, getattr(orig, MAGIC_ATTR))
    else:
        print("[verl-patch] ⚠️ update_actor 没有 dispatch 元数据，torch-prof 探针放弃安装", flush=True)
        return
    W.update_actor = profiled
    W._syncopate_torch_prof = True
    print(f"[verl-patch] torch-prof 探针已装（SYNCOPATE_TORCH_PROF={target}）——"
          "判据：目标步打「抓取…」行 + 同步类算子账本", flush=True)


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


# ─────────────────────────────────────────────────────────────────────────────
# PrefixGrouper：一组 8 条样本共享的题面**只算一次**（E25 §5 / docs/upstream/verl-prefix-grouper-not-wired）
#
# ⚠️⚠️ 实验性，`SYNCOPATE_PREFIX_GROUPER=1` 才装，**默认关**。
#
# verl 0.8.0 里 `actor.use_prefix_grouper=True` 是个**空开关**（四处断线，前三处已复现）：
#   ① `apply_monkey_patch()` 全包唯一调用点（transformer_impl.py:292）**不传这个参数**
#      ⇒ ALL_ATTENTION_FUNCTIONS 没被包装 ⇒ prefix_grouper= 这个 kwarg 会被模型忽略
#   ② `forward_micro_batch_with_prefix_grouper()` **零调用者** ⇒ 打包前向是死代码
#   ③ `build_pg_from_micro_batch()` 把 `response_mask`（**梯度**掩码）当 `suffix_mask`
#      （**存在**掩码）⇒ 多轮下工具 observation 的 token 会被从模型输入里删掉
#      —— verl 自带的 tool_agent_loop.py:400 也是这个语义，不是我们特有的
#   ④ 它还要 `micro_batch["uid"]`，而 uid 在 `non_tensor_batch` 里，**到不了 engine 的 TensorDict**
#
# ⇒ ★ 所以「打开开关量快了多少」这个判据会**为错误的理由通过**（测出 0 收益后放弃）。
#   本补丁的第一条判据因此是「**这条路径真的被走到了**」，第二条才是等价性，第三条才是快慢。
#
# 验收顺序（缺一条都不许往下走）：
#   判据A  [prefix-grouper] 打包前向已生效  —— 路径真的被走到
#   判据B  log_probs 与关闭时**逐位相同**   —— 论文保证训练等价，不相同就是我们接错了
#   判据C  吞吐 / 任务尺子
def _patch_prefix_grouper() -> None:
    """PrefixGrouper 接线 · 整体对齐上游 #7202 的形状 + 我们独有的两处修复。

    ★ 为什么是这个形状（2026-08-19）
      #7202（已被维护者关闭）已经做对了两块：**在隐状态上切分** + **只对 suffix 投影**。
      我们独立收敛到同一个设计 ⇒ 那是这条路上的唯一自然解 ⇒ **照抄，不自创**。
      我们独有的是它缺的两块：③ 掩码语义、⑤ 因果对齐。

    ⚠️ 上游踩过的两个坑（照抄以免重踩）：
      · reshape 必须在 `FusedLinearForPPO` **外面**做 —— 在它 forward 里 flatten 会跑在
        no_grad 下、**丢掉隐状态的梯度**（又一个不报错只训歪）
      · 不要用 `output_hidden_states=True` —— 它会物化全部 36 层（≈1.7 GB 白付），
        直接取基座模型的最后一层输出
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from verl.models.transformers import monkey_patch as _mp

    # ⚠️⚠️⚠️ **绝对不能在这里 import `verl.workers.engine.fsdp.transformer_impl`**（2026-08-19 实测）
    #
    #   实测：`import verl.workers.engine.fsdp.transformer_impl` 会让
    #        `torch.cuda.is_initialized()` 从 False 变 **True**
    #        （`prefix_grouper` / `monkey_patch` / `modeling_utils` 都不会）
    #
    #   而 `setup_worker()` 是靠 Ray 的 `worker_process_setup_hook` 跑的 ——
    #   **它在进程启动时执行，早于 Ray 把这个进程分配给某张卡**。
    #   ⇒ CUDA 上下文提前绑到 device 0，Ray 后面再设 CUDA_VISIBLE_DEVICES 已经无效
    #   ⇒ 三个 trainer rank 全绑在同一张卡上：
    #        `NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 21000`
    #   ⇒ 训练**一次都起不来**，而报错完全不像"你 import 早了"。
    #
    #   ★ 修法利用 verl 自己的 from-import 顺序：
    #     `transformer_impl` 在**模块导入时**做 `from ...monkey_patch import apply_monkey_patch`
    #     ⇒ 只要我们在它被导入**之前**改掉 `_mp.apply_monkey_patch`（轻量、不碰 CUDA），
    #        它自然会绑到我们这一版；
    #     ⇒ 而 `forward_step` / `pg_forward` 的接线推迟到**那个函数被调用时**再做 ——
    #        那时模型正在构建，卡早就分好了。
    #   ★ 判据：`grep -c 'Duplicate GPU' <log>` 必须是 0，且**要等到终态才读**
    #     （"还没走到"和"不会发生"在日志里长得一模一样 —— 今天为此误判了三次）。

    # ⚠️⚠️ **`prefix_grouper_utils` 必须惰性导入**（2026-08-19 实测）。
    #   它顶层 `from prefix_grouper import PrefixGrouper`，而这个 import 会在
    #   **Ray 给 worker 分配 GPU 之前**初始化 CUDA 上下文 ⇒ 所有 worker 都绑到 device 0
    #   ⇒ NCCL 起不来：`Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 21000`
    #   ★ 判据：关掉 SYNCOPATE_PREFIX_GROUPER 后该告警 0 次、训练正常 ⇒ 是本补丁引入的。
    #   ★ 这条报错**完全不像**"你自己在 setup 里 import 早了"，像硬件/调度问题 ——
    #     又一个「误导型失败」。
    class _LazyPGU:
        _m = None
        def _mod(self):
            if _LazyPGU._m is None:
                from verl.trainer.ppo import prefix_grouper_utils as m
                _LazyPGU._m = m
            return _LazyPGU._m
        def __getattr__(self, k):
            return getattr(self._mod(), k)
        def __setattr__(self, k, v):
            setattr(self._mod(), k, v)

    _pgu = _LazyPGU()

    # ── ⑤ 因果对齐（本项目独有的发现，也是整条线唯一真正算错的地方）──────────
    #
    # PrefixGrouper 把自己的 **2D padding mask** 当第 4 个位置参数传给 attn_func，而
    # suffix 子调用的形状是 q 长 R、k 长 P+R ⇒ 它需要的是**右下对齐**的因果掩码
    # （query i 在绝对位置 P+i，应看到 key 0..P+i）。HF 的后端拿到那个 mask 之后：
    #     eager / sdpa        构造**左上对齐**因果掩码 ⇒ 结构性错误（fp32 实测差 21 万倍）
    #     flash_attention_2   2D mask 长度 ≠ query 长度 ⇒ 走 _upad_input 变长路径 ⇒ 错位
    # ⇒ 正解：**不传那个 mask**（打包布局里本来就没有 padding），让后端走纯 causal。
    #   实测 `flash_attn_func(causal=True)` **就是右下对齐**（与显式右下参考逐位相同，
    #   与左上差 3.63）⇒ FA2 是对的后端。
    #
    # ★ 只替换 wrapper **工厂**，让上游的 apply_prefix_grouper_patch 装我们这一版 ——
    #   **不要在外面再套一层**（双层包装实测会在 batch_repeat_cat 里非法访存）。
    def _wrapper_factory(original_fn):
        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            pg = kwargs.pop("prefix_grouper", None)
            if pg is None:
                return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

            def attn_func(q, k, v, _pg_mask, *ia, **ik):
                # ★ 同时丢掉 position_ids：HF 的 FA2 用它探测「打包序列」并切到变长路径，
                #   而这里传进来的是**整条 grouped 序列**的 position_ids（长 P+G·R），
                #   与子调用的形状（B=G, Lq=R）对不上 ⇒ 越界访存。
                #   ⚠️ 旋转位置编码在进入注意力函数**之前**就已经加过了，这里丢掉是安全的。
                ik.pop("position_ids", None)
                return original_fn(module, q, k, v, None, *ia, **ik)[0]

            return pg.forward(attn_func, query, key, value, *args, **kwargs), None
        return wrapped

    _mp._create_prefix_grouper_wrapper = _wrapper_factory


    # ── ① 让 apply_monkey_patch 收到 use_prefix_grouper ──────────────────────
    _orig_apply = _mp.apply_monkey_patch

    _wired = {"done": False}
    # ★ 存下 CausalLM 的 **HF 原版 forward**（verl 的 fused-kernels patch 会把它换掉，
    #   而 fused 版签名是**固定参数、不透传 kwargs** ⇒ `prefix_grouper` 到不了 attention，
    #   还会对整条 grouped 序列做一次多余的全长投影）。_pg_forward 里临时换回原版调用。
    _orig_cls_forward: dict = {}

    def _apply_with_pg(*args, **kwargs):
        kwargs["use_prefix_grouper"] = True
        _m = args[0] if args else kwargs.get("model")
        if _m is not None and type(_m) not in _orig_cls_forward:
            _orig_cls_forward[type(_m)] = type(_m).forward
        r = _orig_apply(*args, **kwargs)
        # ★ 到这里模型正在构建 ⇒ Ray 早已分好卡 ⇒ 现在才碰重模块是安全的
        if not _wired["done"]:
            _wired["done"] = True
            _wire_engine()
        return r

    _mp.apply_monkey_patch = _apply_with_pg

    # ── ③④ 打包用「存在掩码」；组边界由 prompt 逐位相同推出（uid 到不了 engine）──
    _grp_reported = {"done": False}

    def _build_pg(micro_batch, pad_token_id, padding_mode="right"):
        from prefix_grouper import PrefixGrouper

        prompts = micro_batch["prompts"]
        responses = micro_batch["responses"]
        prompt_len = prompts.size(1)
        # ★③ 存在掩码，**不是** response_mask（那是梯度掩码，工具 observation 为 0）
        #    上游 #7202 这一行没改 ⇒ 多轮下工具返回会被静默删掉
        existence = micro_batch["attention_mask"][:, prompt_len:]

        group_sizes, cur = [], 1
        for i in range(1, prompts.size(0)):
            if torch.equal(prompts[i], prompts[i - 1]):
                cur += 1
            else:
                group_sizes.append(cur); cur = 1
        group_sizes.append(cur)
        # ⚠️ 组大小**不必相等**（PrefixGrouper 接受一个列表）—— 当初写死"必须相等"是
        #   过度保守，会把一个**合法但低效**的情形判成错误。正确性由「同组连续」保证，
        #   而那一条已由上面的重排保证。这里只做一次性报告，让低效自己显形。
        if not _grp_reported["done"]:
            _grp_reported["done"] = True
            _n1 = sum(1 for g in group_sizes if g == 1)
            print(f"[prefix-grouper] 组构成 {group_sizes}（{len(group_sizes)} 组 / "
                  f"{sum(group_sizes)} 条，其中孤立样本 {_n1} 条）"
                  f"{'  ⚠️ 碎片化严重 ⇒ 打包收益会被吃掉' if _n1 > len(group_sizes)//2 else ''}",
                  flush=True)

        idx = torch.tensor([sum(group_sizes[:i]) for i in range(len(group_sizes))],
                           device=prompts.device)
        prefix_ids = prompts.index_select(0, idx)
        # ★ 前缀掩码也用 attention_mask（与 response 一致）——
        #   不要用 `ne(pad_token_id)`：pad id 猜错时会把 padding 一起打包进去
        #   （实测 grouped 里前缀贡献了 5120 = 完整补齐宽度，而真实 prompt 只有 ~4111）
        prefix_mask = micro_batch["attention_mask"][:, :prompt_len].index_select(0, idx).bool()
        pg = PrefixGrouper.from_ungrouped_masks(
            prefix_mask=prefix_mask, suffix_mask=existence,
            group_sizes=group_sizes, padding_mode=padding_mode, device=prompts.device)
        concat_ids = pg.concat_input(prefix_ids, prefix_mask, responses, existence)
        pos_ids = _pgu.build_position_ids_for_prefix_grouper(pg)
        # 第 4/5 个返回值当 completion_ids / completion_mask 传进 pg_forward，
        # 必须与打包用的是**同一个**掩码，否则 convert_padding 会把位置对错
        return pg, concat_ids, pg.padding_mask, pos_ids, responses, existence

    _pending = {"build": _build_pg}

    # ── ② response-only 投影（照抄 #7202：隐状态上切分，只对 suffix 投影）───────
    #
    # 🔴 不做这一步会 OOM：整条 grouped 序列过 vocab 头 = 9,428 × 151,936
    #    ⇒ fp32 + 梯度 10.67 GB。而我们本来靠 use_fused_kernels 从不物化。
    def _base_model(module):
        m = module
        for _ in range(6):
            inner = getattr(m, "model", None)
            if inner is None or inner is m:
                break
            m = inner
        return m

    def _lm_head_weight(module):
        m = module
        for _ in range(6):
            fn = getattr(m, "get_output_embeddings", None)
            if callable(fn):
                emb = fn()
                if emb is not None and hasattr(emb, "weight"):
                    return emb.weight
            m = getattr(m, "module", None) or getattr(m, "base_model", None) or getattr(m, "model", None)
            if m is None:
                break
        raise RuntimeError("[prefix-grouper] 找不到 lm_head —— 拒绝退回物化 logits 的路")

    _logged = {"done": False}

    def _pg_forward(model, prefix_grouper, concat_input_ids, attention_mask, position_ids,
                    completion_ids, completion_mask, *, temperature=1.0, padding_mode="right",
                    include_prefix_last=1, calculate_entropy=False, entropy_fn=None):
        # ★★★ 2026-08-19 修（E26 §6.3 的真正根因，脱 Ray 复现 scripts/repro_pg_dtype.py）：
        #   前向必须走**根 FSDP 模块**、且损失必须流经**根 forward 的输出**。
        #
        #   此前这里直接调 `_base_model(model)(...)` 绕过根 ⇒ FSDP1 的 pre-backward hook
        #   （注册在根输出张量上）永不触发 ⇒ `_post_backward_final_callback` 没人排队 ⇒
        #   归约(fp32 all-reduce)+转回 bf16 全部挂在 `_post_backward_stream` 上**没有人等**
        #   ⇒ optimizer 读到竞态快照：有时没归约（E21 形状的静默错误！三 rank 各学各的），
        #   有时抓到半途的 fp32（Adam `expected BFloat16 for 'end'` —— 幸运的金丝雀）。
        #   「以前能跑」纯属意外：根没被 lazy_init 时子单元各自自封根；而 adapter 同步
        #   在启动时做的一次 state_dict() 会把真根 init 掉，此后每一步都在竞态里。
        #
        #   做法：
        #   ① 临时把 CausalLM 的 class forward 换回 HF 原版（verl fused 版不透传 kwargs，
        #      prefix_grouper 会被吞掉；HF 原版还支持 logits_to_keep=1 ⇒ lm_head 只投影
        #      1 个位置，不物化全量 logits）
        #   ② hidden 用 forward hook 从基座捕获（仍不用 output_hidden_states）
        #   ③ 给 log_probs 加一个 0×根输出 的锚，保证根的 pre-backward 在 backward 一开始
        #      就触发（判据：repro 探针「final排队/执行」每次 backward 各 +1，
        #      三 rank 梯度和与健康路径**逐位相同**）
        # E31 第 3 步：内层 MXFP8 的 swap 必须发生在根前向**之前**（首调一次，幂等；
        # SYNCOPATE_UNIFIED_FP8_LAYERS=0 时零动作）。swap 数断言 N×7，半接线直接拒跑。
        from syncopate.train.unified_fp8 import patch_trainer_inner as _ufp8_inner
        _ufp8_inner(model)
        _causal = None
        for _m in model.modules():
            if type(_m) in _orig_cls_forward:
                _causal = _m
                break
        _captured = []
        _h = _base_model(model).register_forward_hook(
            lambda _mod, _i, _o: _captured.append(_o[0] if isinstance(_o, tuple) else _o.last_hidden_state))
        _cls, _swapped = (type(_causal), type(_causal).forward) if _causal is not None else (None, None)
        try:
            if _cls is not None:
                _cls.forward = _orig_cls_forward[_cls]
            out = model(input_ids=concat_input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=False,
                        prefix_grouper=prefix_grouper, logits_to_keep=1, return_dict=True)
        finally:
            if _cls is not None:
                _cls.forward = _swapped
            _h.remove()
        if not _captured:
            raise RuntimeError("[prefix-grouper] 根前向跑了但基座 hook 没捕获到 hidden —— 模块层级变了？")
        hidden = _captured[0]
        # 锚：0×根输出。把损失接到根 forward 的输出张量上，唯一作用是触发根的 pre-backward。
        _anchor = out.logits.float().sum() * 0.0
        _, _, suffix_h, suffix_mask_raw = prefix_grouper.split_output(
            hidden, include_prefix_last=include_prefix_last)
        completion_right = prefix_grouper.convert_padding(
            completion_ids, completion_mask, padding_mode=padding_mode)

        # ★ reshape 在 FusedLinearForPPO **外面**做（上游 #7202 的坑：里面 flatten 会跑在
        #   no_grad 下、丢掉隐状态梯度 —— 不报错，只训歪）
        h = suffix_h[:, :-1].contiguous()
        B, T, D = h.shape
        # E31：投影唯一入口在 unified_fp8.linear_for_ppo —— SYNCOPATE_UNIFIED_FP8=1 时
        # 走 MXFP8（与 vLLM 侧同量化器同 kernel，量化项在 IS 比率对消）；
        # 关（默认）= 原样走 FusedLinearForPPO，逐位不变（tests/train/test_e31_step1.py 钉）。
        from syncopate.train.unified_fp8 import linear_for_ppo as _linear_for_ppo
        log_probs, entropy = _linear_for_ppo(
            hidden_states=h.view(B * T, D),
            vocab_weights=_lm_head_weight(model),
            input_ids=completion_right.reshape(B * T),
            temperature=temperature,
        )
        log_probs = log_probs.view(B, T) + _anchor
        # ⚠️ **始终把熵带出去**：verl 的 `_compute_old_log_prob` 会无条件地对它调
        #   `no_padding_2_padding(entropy, ...)`，传 None 进去就是
        #   `AttributeError: 'NoneType' object has no attribute 'is_nested'`
        #   —— 报在 verl 的 padding.py 里，离根因很远。
        #   而 `FusedLinearForPPO` 本来就同时返回熵，我们之前是算了却没给出去。
        entropy = entropy.view(B, T) if entropy is not None else None
        if not _logged["done"]:
            _logged["done"] = True
            # ★ 判据A：没有这一行就是没走到打包路径
            print(f"[prefix-grouper] 打包前向已生效 · response-only 投影 · "
                  f"grouped {tuple(concat_input_ids.shape)} → 投影 {B*T} 个位置", flush=True)
        return log_probs, entropy, suffix_mask_raw[:, 1:]

    _pending["fwd"] = _pg_forward

    # ── forward_step 接到打包前向（**延迟执行**，见文件上方的 CUDA 初始化说明）────
    def _wire_engine():
        from verl.workers.engine.fsdp import transformer_impl as _ti
        _orig_forward_step = _ti.FSDPEngineWithLMHead.forward_step

        def _forward_step(self, micro_batch, loss_function, forward_only):
            from verl.utils.device import get_device_id, get_device_name

            micro_batch = micro_batch.to(get_device_id())
            need = ("prompts", "responses", "response_mask", "attention_mask")
            # ★ PrefixGrouper 只能在**同一个 micro-batch 内**共享题面 ⇒ 一整组必须在一起。
            #   micro_batch=1 时它无组可分：判据行会打出「投影 31 个位置」这种明显不对的数，
            #   而且下游形状会崩。⇒ 用 --micro-batch-size = rollout_n。
            if not all(k in micro_batch.keys() for k in need):
                return _orig_forward_step(self, micro_batch, loss_function, forward_only)

            if _pending:
                # ★ 到这里 Ray 已经分好卡了，现在才 import prefix_grouper（见上面的惰性导入注释）
                _pgu.build_pg_from_micro_batch = _pending.pop("build")
                _pgu.pg_forward = _pending.pop("fwd")
            # ★ 自己把同组样本排到一起（不依赖 verl 的组感知划分——那条路径 0.8.0 里是坏的）。
            #   按 prompt 内容做稳定分组：相同 prompt 的行连续排列，算完再按逆排列还原。
            _p = micro_batch["prompts"]
            _key = {}
            _order = []
            for _i in range(_p.size(0)):
                _h = hash(_p[_i].detach().cpu().numpy().tobytes())
                _key.setdefault(_h, []).append(_i)
            for _v in _key.values():
                _order.extend(_v)
            _perm = torch.tensor(_order, device=_p.device)
            _inv = torch.empty_like(_perm); _inv[_perm] = torch.arange(len(_order), device=_p.device)
            mb = {k: micro_batch[k].index_select(0, _perm) for k in need}
            mb["pad_token_id"] = getattr(self, "pad_token_id", 0)
            entropy, log_probs = _pgu.forward_micro_batch_with_prefix_grouper(
                micro_batch=mb, model=self.module,
                temperature=1.0,
                calculate_entropy=True,   # ★ 见 _pg_forward 里的说明：verl 无条件要熵
                device_name=get_device_name(),
                param_dtype=getattr(self, "_autocast_dtype", torch.bfloat16),
            )
            # ✅ 2026-08-19 已解：那个 Adam `expected BFloat16 for 'end' got float` 的根因
            #    **不是 dtype 转换**，是 `_pg_forward` 绕过了根 FSDP forward ⇒ 归约与
            #    cast-back 挂在没人等的 post-backward stream 上（竞态，见 _pg_forward 顶部
            #    注释与 scripts/repro_pg_dtype.py）。dtype 报错只是竞态的一种表现；
            #    更危险的表现是**梯度静默不跨 rank 归约**（E21 形状）且不报错。
            # ⛔ 历史弯路留档：曾把 log_probs 转 bf16（方向反了，FusedLinearForPPO 契约就是
            #    fp32）、曾 patch AdamW 扫 dtype（verl 用的类经 lr_scheduler 包装，且问题
            #    本就不在 optimizer）。都对现象「一字不变」⇒ 印证解药④。
            # ★ 还原成 micro_batch 原来的行顺序
            log_probs = log_probs.index_select(0, _inv)
            if entropy is not None:
                entropy = entropy.index_select(0, _inv)

            # ── ⑦ 转成 verl 要的**锯齿(nested)格式** ────────────────────────
            #
            # verl 的 `postprocess_batch_func` 只支持一种格式（其余 raise NotImplementedError）：
            #     tensors = [t for nt in model_output[key] for t in nt.unbind()]
            #     torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            # ⇒ 每条序列一个**长度 = 该序列真实 token 数**的向量（prompt+response）。
            # 而下游 `no_padding_2_padding` 取的是 `values[off - R - 1 : off - 1]`（左移一位）
            # —— **正是我们算的那一段**。所以只要摆到对的偏移上即可。
            #
            # ⛔ 走过的弯路：先给稠密 [bsz, max_resp_len] + 打旁路 —— 两次都失败
            #    （第一次装错进程；第二次 verl 已经把它拼成锯齿张量了）。
            #    ⇒ **按下游要的格式产出，比在下游加旁路更稳**：少一处补丁，也少一个会漏的判据。
            _am = micro_batch["attention_mask"]
            _P = micro_batch["prompts"].shape[1]
            _plen = _am[:, :_P].sum(1)
            _rlen = _am[:, _P:].sum(1)
            # ★ E14 §4.7-③ 乒乓修理（SYNCOPATE_FIX_JAGGED，默认开）：原写法逐行 .item()
            #   = 每 micro-batch ~32 次强制同步（全 update_actor ×96）。长度一次性 .tolist()
            #   —— 2 次同步替代 ~32 次，数值语义逐位不变（同样的整数，换个搬法）。
            if os.environ.get("SYNCOPATE_FIX_JAGGED", "1") == "1":
                _plen_c, _rlen_c = _plen.tolist(), _rlen.tolist()
            else:
                _plen_c, _rlen_c = None, None      # A/B 对照：走旧逐行 .item() 路径

            def _to_jagged(x):
                if x is None:
                    return None
                _outs = []
                for _i in range(x.shape[0]):
                    if _plen_c is not None:
                        _L = int(_plen_c[_i] + _rlen_c[_i]); _R = int(_rlen_c[_i])
                    else:
                        _L = int(_plen[_i].item() + _rlen[_i].item()); _R = int(_rlen[_i].item())
                    _v = x.new_zeros(_L)
                    if _R > 0:
                        _v[_L - _R - 1:_L - 1] = x[_i, :_R]
                    _outs.append(_v)
                return torch.nested.as_nested_tensor(_outs, layout=torch.jagged)

            model_output = {"log_probs": _to_jagged(log_probs)}
            if entropy is not None:
                model_output["entropy"] = _to_jagged(entropy)
            if loss_function is not None:
                loss, metrics = loss_function(model_output=model_output, data=micro_batch,
                                              dp_group=self.get_data_parallel_group())
            else:
                assert forward_only
                loss, metrics = torch.tensor(1.0, device=get_device_name()), {}
            # ⚠️ verl 原版是 `return loss, output`（**loss 在前**）。写反了不会报形状错，
            #    而是让 verl 的 postprocess 拿 Tensor 去做 `"model_output" in <Tensor>`
            #    ⇒ RuntimeError: Tensor.__contains__ ... —— 报在离根因很远的地方。
            return loss, {"model_output": model_output,
                          "loss": loss.detach().item(), "metrics": metrics}

        _ti.FSDPEngineWithLMHead.forward_step = _forward_step
    print("[verl-patch] PrefixGrouper 已接线（SYNCOPATE_PREFIX_GROUPER=1，对齐 #7202 + 掩码/因果两处修复）"
          " —— ★ 没看到 [prefix-grouper] 打包前向已生效 就是没走到", flush=True)


