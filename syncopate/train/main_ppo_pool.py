"""verl 入口的薄壳：把训练集采样器换成动态分池。

★ 为什么要一层壳，不能直接配

verl 的 `create_rl_sampler` 是**写死**的（`main_ppo.py:348`），没有配置挂点：
shuffle=True 给 RandomSampler，否则 SequentialSampler。两个都是均匀采样。

而均匀采样在我们这里是明确错的：GRPO 的梯度**完全来自组内 reward 的方差**，
一条 8 次全对、方差为 0 的题贡献**精确地等于零**，却照样吃掉 8 次 rollout ——
而 rollout 是整条链上最贵的一步。v11 实测饱和格子占三分之一。

⇒ 这层壳只做一件事：`monkeypatch create_rl_sampler`，其余原样交给 verl。
不 fork、不改 verl 的文件，升级 verl 时这层壳大概率还能用。

    python -m syncopate.train.main_ppo_pool <和 main_ppo 完全相同的 hydra override>

★ 反馈是怎么闭环的

    AgentLoop 每跑完一条 rollout → 追加一行到 dispatched.jsonl（case_id/reward/num_steps）
    采样器每出一个 batch 之前   → 增量读那个文件，更新 per-case 的 ema_std
    下一个 batch                → 按新权重抽

两边通过**文件**通信而不是共享内存：rollout 跑在 Ray 的 worker 进程里，
和 trainer 不同进程；文件是唯一不需要额外接线的通道，而它本来就存在
（`record_dispatch` 是为分布漂移诊断写的，这里顺带复用）。

★★ 这层壳同时负责选 verl 的哪个 main（2026-08-13 上 4 卡后新增）

    SYNCOPATE_RL_MODE = colocate | one_step_off | fully_async

三个 main 是**三套不同的 trainer**，不是一个开关：

    colocate      verl.trainer.main_ppo                              rollout 和 train 同卡
    one_step_off  verl.experimental.one_step_off_policy.main_ppo      分卡，落后一步
    fully_async   verl.experimental.fully_async_policy.fully_async_main  分卡，两个独立池

⚠️⚠️ **补丁必须打两处，否则在 one_step_off 下静默失效**：
`one_step_off_policy/main_ppo.py` 用的是
`from verl.trainer.main_ppo import create_rl_sampler` —— 名字在**导入时**就绑进了
它自己的模块命名空间，改 `verl.trainer.main_ppo.create_rl_sampler` 对它一点用都没有。
**这正是本文件原注释里预告过的那种失效**（"如果哪天 verl 改成从别处 import"），
只不过它不是"哪天"，是换个 trainer 就已经发生。⇒ 照旧看 `[pool] 动态分池启用` 那行日志。

⛔ **一条已被推翻的结论（2026-08-14 更正，原文保留见下）**

原注释写着：「`fully_async` 根本不调 `create_rl_sampler`（它自己排采样计划）
⇒ 动态分池在那个模式下不生效」。**读码核实：它调了。**

    verl/experimental/fully_async_policy/fully_async_rollouter.py
      392  @ray.remote(num_cpus=10, max_concurrency=100)      ← 它是个 Ray actor
      400  class FullyAsyncRollouter(SeparateRayPPOTrainer): def __init__
      447      from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
      464      train_sampler = create_rl_sampler(config.data, train_dataset)   ★

而且那个 `import` 写在**函数体内** ⇒ 调用时才解析
`verl.trainer.main_ppo.create_rl_sampler` ⇒ **对 monkeypatch 最友好的写法**。

真正的原因不是「没有挂点」，是**作用域**：`FullyAsyncRollouter` 跑在**另一个 Ray
worker 进程**里，而 `_install()` 只打在 driver。`worker_process_setup_hook`
（`verl_patches.setup_worker`）**已经在用了**，但当前只装 `_patch_fsdp_cpu_copy_for_ddp`。
⇒ **修法：把 sampler patch 也装进 `setup_worker()`。**（M7 跑完再改，见交接文档。）

★ 教训：这是「机制建好了但没接上」的**新变种**——**不是忘了接，是断定接不上、而断定错了。**
同一天 `verl_patches.py` 里刚写下同一条：「判据行打出来了 ≠ 补丁在需要它的那个进程里生效」。
⇒ **凡是说「verl 不支持 X」，先查三层：① 配置项在不在 ② 代码路径调不调 ③ 它在哪个进程里跑。**
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syncopate.train import verl_patches  # noqa: E402
from syncopate.train.pool import Pool  # noqa: E402

# 采样器把池子状态存这里，断点续跑时接着用。由 launch_rl 通过环境变量指定。
POOL_STATE_ENV = "SYNCOPATE_POOL_STATE"
DISPATCH_LOG_ENV = "SYNCOPATE_DISPATCH_LOG"


class DynamicPoolSampler:
    """按「还有没有梯度」加权的无放回采样器。

    torch 的 Sampler 协议只要求 `__iter__` 产出**索引**、`__len__` 给出长度。
    verl 的 StatefulDataLoader 每个 epoch 调一次 `iter(sampler)`，
    然后按 batch_size 把索引切成 batch —— 所以我们在生成器里**每凑够一个 batch
    就重新吃一次反馈**，粒度就是每步一次，而不是每 epoch 一次。

    ⚠️ `__len__` 必须等于数据集大小：verl 拿它算 `total_training_steps`
    和学习率调度的总步数。改了它，warmup 和 cosine 衰减会全部错位。
    """

    def __init__(self, dataset, batch_size: int, seed: int = 0) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = seed
        self.epoch = 0

        # 索引 ↔ case_id。case_id 在 parquet 的 extra_info 里，和 AgentLoop 读的是同一处。
        self.case_ids: list[str] = []
        for i in range(len(dataset)):
            row = dataset[i]
            extra = row.get("extra_info") if isinstance(row, dict) else None
            cid = (extra or {}).get("case_id") if isinstance(extra, dict) else None
            self.case_ids.append(str(cid) if cid else f"__row_{i}")
        self.index_of = {cid: i for i, cid in enumerate(self.case_ids)}

        state_path = os.environ.get(POOL_STATE_ENV)
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists():
            self.pool = Pool.load(self.state_path)
            # 数据集可能变了（换了 vN）：补上新 case，丢掉不在数据集里的
            for cid in self.case_ids:
                self.pool.states.setdefault(cid, Pool([cid]).states[cid])
        else:
            self.pool = Pool(self.case_ids)

        log = os.environ.get(DISPATCH_LOG_ENV)
        self.dispatch_log = Path(log) if log else None
        self._step = 0
        print(f"[pool] 动态分池启用：{len(self.case_ids)} 条 case，"
              f"batch={self.batch_size}，反馈来源={self.dispatch_log}", flush=True)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __iter__(self):
        self.epoch += 1
        emitted = 0
        while emitted < len(self.case_ids):
            if self.dispatch_log:
                self.pool.ingest(self.dispatch_log, self._step)
            batch = self.pool.sample(self.batch_size, step=self._step,
                                     seed=self.seed * 100003 + self._step)
            if not batch:
                break
            if self._step % 10 == 0:
                snap = self.pool.snapshot(self._step)
                print(f"[pool] step={self._step} {snap}", flush=True)
                if self.state_path:
                    self.pool.save(self.state_path)
            for cid in batch:
                yield self.index_of[cid]
                emitted += 1
            self._step += 1
        if self.state_path:
            self.pool.save(self.state_path)

    # StatefulDataLoader 会尝试存取采样器状态（用于断点续跑）
    def state_dict(self) -> dict:
        return {"step": self._step, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self._step = int(state.get("step", 0))
        self.epoch = int(state.get("epoch", 0))


MODE_ENV = "SYNCOPATE_RL_MODE"

# 模式 → (verl 的 main 模块, 该模块里需要打补丁的命名空间)
_MAINS = {
    "colocate": "verl.trainer.main_ppo",
    "one_step_off": "verl.experimental.one_step_off_policy.main_ppo",
    "fully_async": "verl.experimental.fully_async_policy.fully_async_main",
}


def install_sampler_patch() -> None:
    """把 verl 的 `create_rl_sampler` 换成动态分池。

    ★ 抽成公开函数，是因为它要在**两个进程**里各装一次：
      driver（TaskRunner）—— `_install()` 调
      Ray worker         —— `verl_patches.setup_worker()` 调（fully_async 的
                            `FullyAsyncRollouter` 是个 Ray actor，活在另一个进程里）

    ⚠️ **幂等**：装第二次会把补丁版当成 original 套娃，权重更新就乱了。
    """
    from verl.trainer import main_ppo

    if getattr(main_ppo.create_rl_sampler, "_syncopate_pool", False):
        return                                  # 已经装过

    original = main_ppo.create_rl_sampler

    def patched(data_config, dataset):
        if os.environ.get("SYNCOPATE_POOL", "1") != "1":
            return original(data_config, dataset)
        return DynamicPoolSampler(
            dataset,
            batch_size=int(data_config.train_batch_size),
            seed=int(data_config.get("seed") or 0),
        )

    patched._syncopate_pool = True              # 幂等标记
    main_ppo.create_rl_sampler = patched
    # ⚠️ colocate 下 TaskRunner 是同模块内直接调用（`train_sampler = create_rl_sampler(...)`），
    # 所以改模块属性就够了。
    #
    # one_step_off 用的是 `from verl.trainer.main_ppo import create_rl_sampler`（导入时绑名），
    # 按说改上面这行对它无效 —— 但我们是用 runpy **重新导入**它的（见 main()），
    # 那次导入发生在这个补丁**之后**，所以它绑到的就是补丁版。
    # ⇒ 判据还是那一条：日志里有没有 `[pool] 动态分池启用`。**别只看代码，看那行日志。**


def main() -> None:
    mode = os.environ.get(MODE_ENV, "colocate")
    if mode not in _MAINS:
        raise SystemExit(f"未知的 {MODE_ENV}={mode}，可选：{'/'.join(_MAINS)}")

    install_sampler_patch()
    # verl 自己的 bug 补在这里（见该模块的 docstring）。必须在 verl 的 main 被 import
    # 之前调 —— 补丁换的是模块属性，import 之后再换就来不及了。
    verl_patches.apply(mode)
    print(f"[rl] 模式={mode}  verl 入口={_MAINS[mode]}", flush=True)

    if mode == "colocate":
        from verl.trainer.main_ppo import main as verl_main

        verl_main()
        return

    if mode == "fully_async":
        # ⛔ 原因已更正（2026-08-14，见模块 docstring）：
        # ✅ 2026-08-17 已修：sampler patch 装进了 `verl_patches.setup_worker()`，
        #    在 rollouter 那个 Ray worker 进程里生效。
        #
        # ⚠️⚠️ **这行原来写的是「⇒ 动态分池本轮不生效」，那是个断言，而且修好之后它就错了** ——
        #    真正生效的判据是 rollouter 进程打出来的
        #    `[pool] 动态分池启用：N 条 case`（带 `(FullyAsyncRollouter pid=…)` 前缀）。
        #    ⇒ 判据行**只许写观测，不许写断言**：一行断言"为什么不行"的日志，
        #      能把"其实已经行了"整个盖住，而且它长得像个合格判据（05-handoff §6 变种②）。
        print("[pool] fully_async：sampler 在 rollouter 的 Ray worker 进程里装 —— "
              "**判据是那个进程打出的 `[pool] 动态分池启用`，不是这一行**",
              flush=True)

    # ⚠️⚠️ 两个实验性入口**必须当 `__main__` 跑，不能 import 了再调**（2026-08-13 实测）
    #
    #     Primary config module 'verl.experimental.one_step_off_policy.config' not found.
    #     Check that it's correct and contains an __init__.py file
    #
    # 根因：`@hydra.main(config_path="config")` 解析 config_path 的方式取决于
    # **task function 的 `__module__`** —— 是 `__main__` 就按**文件路径**找，
    # 否则按**模块**找（要求那个 config 目录是个包）。而实测：
    #
    #     verl/trainer/config/__init__.py                        有   ← 所以 colocate 直接 import 能跑
    #     verl/experimental/one_step_off_policy/config/__init__.py  没有
    #     verl/experimental/fully_async_policy/config/__init__.py   没有
    #
    # 两个实验性配置目录**不是包**，verl 只考虑了 `python -m` 的用法。
    # runpy 以 `__main__` 跑等价于 `python -m`，且不用碰 site-packages（那是旁路）。
    runpy.run_module(_MAINS[mode], run_name="__main__")


if __name__ == "__main__":
    main()
