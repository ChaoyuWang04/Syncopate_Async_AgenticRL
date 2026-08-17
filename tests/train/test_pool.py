"""动态分池。

守住的核心命题：**一条 8 次采样全对、方差为 0 的题，它消耗的 rollout 对梯度的
贡献精确地等于零。** 采样必须跟着"还有没有梯度"走，而不是定死。

★★ 这里最重要的两条是 `test_saturated_case_is_downweighted_but_never_dropped`
和 `test_idle_case_gets_rescued`：**降权不等于剔除**。策略在漂移，这一步饱和的题
几十步之后可能重新有梯度；永久剔除等于把回归检测也一起关掉 ——
而"教了 A 忘了 B"正是这个项目反复栽跟头的地方（defer 97%→0%、四模板归零）。
"""

from __future__ import annotations

import json

from syncopate.train.pool import (
    OPTIMISTIC_WEIGHT, SHORT_TASK_STEPS, STALE_AFTER_STEPS, WEIGHT_FLOOR,
    CaseState, Pool, weight_of,
)


def _saturated(cid: str = "S", step: int = 0) -> CaseState:
    s = CaseState(cid)
    for _ in range(5):
        s.observe([1.0] * 8, [6.0] * 8, step)      # 8 次全对，方差 0
    return s


def _graded(cid: str = "G", step: int = 0) -> CaseState:
    s = CaseState(cid)
    for _ in range(5):
        s.observe([0.2, 0.9, 0.4, 0.8, 0.3, 1.0, 0.5, 0.6], [6.0] * 8, step)
    return s


# --------------------------------------------------------------------------
# 权重策略
# --------------------------------------------------------------------------


def test_never_sampled_case_wins():
    """冷启动必须保证覆盖：先抽到的几条不能把权重全占了。"""
    assert weight_of(CaseState("new"), step=0) == OPTIMISTIC_WEIGHT
    assert OPTIMISTIC_WEIGHT > weight_of(_graded(), step=0)


def test_graded_case_outweighs_saturated():
    """有方差的题必须比没方差的题更容易被抽中 —— 这是整个机制的目的。"""
    assert weight_of(_graded(), step=1) > weight_of(_saturated(), step=1)


def test_saturated_case_is_downweighted_but_never_dropped():
    """★★ 降权不等于剔除。

    永久剔除 = 把回归检测一起关掉。策略漂移后这条题可能重新有梯度，
    而我们再也不会知道 —— 这正是 defer 97%→0% 那类事故的机制。
    """
    w = weight_of(_saturated(), step=1)
    assert w <= 0.5
    assert w >= WEIGHT_FLOOR, "饱和的题被彻底剔除了，回归检测也一起没了"


def test_short_and_always_right_is_extra_downweighted():
    """「简单短任务」额外降一档 —— 但只在已经没方差时。短本身不是罪。"""
    short = _saturated()
    short.ema_steps = SHORT_TASK_STEPS - 1
    long = _saturated()
    long.ema_steps = SHORT_TASK_STEPS + 6
    assert weight_of(short, step=1) < weight_of(long, step=1)


def test_short_but_still_graded_is_not_punished():
    """短且**仍有方差**的题不该被降 —— 它还在提供梯度。"""
    short_graded = _graded()
    short_graded.ema_steps = 2.0
    long_graded = _graded()
    long_graded.ema_steps = 10.0
    assert weight_of(short_graded, step=1) == weight_of(long_graded, step=1)


def test_idle_case_gets_rescued():
    """★ 太久没体检就把权重顶回来，不让它永远沉底。"""
    stale = _saturated(step=0)
    fresh = _saturated(step=100)
    assert weight_of(stale, step=100 + STALE_AFTER_STEPS) > weight_of(fresh, step=100)


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------


def test_sample_is_without_replacement():
    """⚠️ 同一 batch 里出现两条相同的 case，GRPO 会把它们当成两个独立的组，
    而它们的 advantage 是相关的 —— 组内比较的前提就被破坏了。"""
    pool = Pool([f"C_{i}" for i in range(10)])
    picked = pool.sample(k=6, step=1, seed=1)
    assert len(picked) == len(set(picked)) == 6


def test_sample_is_reproducible():
    """⚠️ 断点续跑必须能重放出同一个 batch。
    「RL 里任何跨 rollout 的随机性都是污染」在这里同样适用。"""
    pool = Pool([f"C_{i}" for i in range(20)])
    assert pool.sample(4, step=3, seed=3) == pool.sample(4, step=3, seed=3)


def test_sample_handles_k_larger_than_pool():
    pool = Pool(["a", "b"])
    assert sorted(pool.sample(k=10, step=1, seed=1)) == ["a", "b"]


def test_saturated_cases_are_sampled_less_often_in_aggregate():
    """端到端：一半题饱和、一半有梯度，长期抽样里有梯度的那半必须占多数。"""
    pool = Pool([f"S_{i}" for i in range(10)] + [f"G_{i}" for i in range(10)])
    for i in range(10):
        pool.states[f"S_{i}"] = _saturated(f"S_{i}", step=1)
        pool.states[f"G_{i}"] = _graded(f"G_{i}", step=1)
    graded_hits = sum(
        1 for step in range(60) for cid in pool.sample(4, step=2, seed=step) if cid.startswith("G")
    )
    total = 60 * 4
    assert graded_hits / total > 0.65, f"有梯度的题只占 {graded_hits/total:.0%}，动态分池没起作用"


# --------------------------------------------------------------------------
# 反馈与断点
# --------------------------------------------------------------------------


def test_ingest_aggregates_a_group_and_is_incremental(tmp_path):
    """★ 增量读：这个文件会长到几十万行，每步全量解析会变成训练循环里的隐形开销。"""
    log = tmp_path / "dispatched.jsonl"
    rows = [{"case_id": "A", "reward": r, "num_steps": 5} for r in (0.2, 0.8, 0.5, 0.9)]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    pool = Pool(["A", "B"])
    assert pool.ingest(log, step=1) == 1
    assert pool.states["A"].seen == 1
    assert pool.states["A"].ema_std > 0.2
    # 第二次读同一个文件不该重复消费
    assert pool.ingest(log, step=2) == 0
    assert pool.states["A"].seen == 1


def test_ingest_survives_a_half_written_line(tmp_path):
    """并发追加可能写出半行 —— 不能因此炸掉整个训练循环。"""
    log = tmp_path / "dispatched.jsonl"
    log.write_text('{"case_id": "A", "reward": 0.5, "num_steps": 3}\n{"case_id": "A", "rew',
                   encoding="utf-8")
    pool = Pool(["A"])
    pool.ingest(log, step=1)
    assert pool.states["A"].seen == 1


def test_snapshot_reports_the_evidence_that_pooling_works():
    pool = Pool([f"S_{i}" for i in range(8)] + [f"G_{i}" for i in range(2)])
    for i in range(8):
        pool.states[f"S_{i}"] = _saturated(f"S_{i}", step=1)
    for i in range(2):
        pool.states[f"G_{i}"] = _graded(f"G_{i}", step=1)
    snap = pool.snapshot(step=2)
    assert snap["pool/size"] == 10
    assert snap["pool/saturated_share"] == 0.8
    # 有效池 < 实际池：说明权重确实不均匀（均匀时两者相等）
    assert snap["pool/effective_size"] < 10


def test_save_load_roundtrip(tmp_path):
    pool = Pool(["A", "B"])
    pool.states["A"] = _graded("A", step=4)
    pool.save(tmp_path / "pool.json")
    back = Pool.load(tmp_path / "pool.json")
    assert back.states["A"].seen == pool.states["A"].seen
    assert abs(back.states["A"].ema_std - pool.states["A"].ema_std) < 1e-9


def test_ingest_ignores_dispatch_and_abort_events(tmp_path):
    """★★ B4 之后必须守住的一条：**只吃 complete，不吃 dispatch/abort**。

    仪器改成三类事件之后，`dispatch` / `abort` 行同样带 `case_id` 但**没有 reward**。
    如果照收，`float(row.get("reward") or 0.0)` 会把「还不知道」当成「确实是 0 分」，
    EMA 方差被稀释、采样权重被污染，**而且全程不报错**。

    ⇒ 这就是 TRACK-B §0.6 从 AReaL 源码抄回来的那条硬约束的本地版：
       **「reward 还不知道」和「reward 确实是 0」是两种完全不同的状态。**
    """
    log = tmp_path / "dispatched.jsonl"
    log.write_text(
        '{"event": "dispatch", "case_id": "A", "rollout_id": "r1"}\n'
        '{"event": "complete", "case_id": "A", "rollout_id": "r1", "reward": 0.8, "num_steps": 6}\n'
        '{"event": "dispatch", "case_id": "A", "rollout_id": "r2"}\n'
        '{"event": "abort", "case_id": "A", "rollout_id": "r2", "reason": "cancelled"}\n',
        encoding="utf-8",
    )
    pool = Pool(["A"])
    pool.ingest(log, step=1)
    # 只有那条 0.8 被吃进去：mean=0.8、方差=0（组里只有一个样本）
    assert pool.states["A"].seen == 1
    assert abs(pool.states["A"].ema_reward - 0.8) < 1e-9
    # ⚠️ 判据的关键：若 abort/dispatch 被当成 0 分收下，均值会掉到 0.2 左右
    assert pool.states["A"].ema_reward > 0.5


def test_ingest_still_reads_the_old_format(tmp_path):
    """旧的 dispatched.jsonl 没有 `event` 键 —— 那时每行都是完成行，必须照收。"""
    log = tmp_path / "dispatched.jsonl"
    log.write_text('{"case_id": "A", "reward": 0.5, "num_steps": 3}\n', encoding="utf-8")
    pool = Pool(["A"])
    pool.ingest(log, step=1)
    assert pool.states["A"].seen == 1
