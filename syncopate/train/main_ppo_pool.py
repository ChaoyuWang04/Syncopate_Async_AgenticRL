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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _install() -> None:
    """把 verl 的 create_rl_sampler 换掉。必须在 TaskRunner 跑起来之前做。"""
    from verl.trainer import main_ppo

    original = main_ppo.create_rl_sampler

    def patched(data_config, dataset):
        if os.environ.get("SYNCOPATE_POOL", "1") != "1":
            return original(data_config, dataset)
        return DynamicPoolSampler(
            dataset,
            batch_size=int(data_config.train_batch_size),
            seed=int(data_config.get("seed") or 0),
        )

    main_ppo.create_rl_sampler = patched
    # ⚠️ TaskRunner 里是 `from ... import create_rl_sampler` 还是 `main_ppo.create_rl_sampler`？
    # 实测是同模块内直接调用（`train_sampler = create_rl_sampler(...)`），
    # 所以改模块属性就够了；如果哪天 verl 改成从别处 import，这里会**静默失效** ——
    # 所以 DynamicPoolSampler 启动时会打一行 `[pool] 动态分池启用`，
    # 日志里没这行就说明补丁没生效。**别只看代码，看那行日志。**


def main() -> None:
    _install()
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
