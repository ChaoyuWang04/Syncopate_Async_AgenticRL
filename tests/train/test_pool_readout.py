"""分池读数：**方向**必须对，而且"完成"和"停机"必须分得开。

★★★ 起因（2026-08-19，写这个读数时当场犯的）

第一版我把完成判据写成「零梯度率**不再下降**」，
而实测轨迹是 `15% → 17% → 37% → 39% → 55% → 52%` —— 明明在**上升**，
判据却打印「仍在下降」。

⇒ 想清楚才对：**零梯度率上升 = 越来越多的题被学会**（或饱和）。
  所以"学到头了"的信号是它**不再创新高**。
⇒ ★ 这正是记了很多次的形状：判据写满了、真的在跑、也真的报了个数，
  **只是它量的方向和你以为的相反**。
  ⚠️ 而它是被**读数本身**照出来的 —— 不是靠 code review。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = ROOT / "scripts" / "pool_readout.py"
    spec = importlib.util.spec_from_file_location("_pool_readout", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pool_readout"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_rising_rate_means_still_learning():
    """★★ 方向：零梯度率**还在创新高** ⇒ 还有东西在被学会，**不该停**。

    实测那条轨迹（60 步）就是这个形状 —— 它证明 60 步远没跑到头。
    """
    m = _load()
    assert m.plateaued([0.15, 0.17, 0.37, 0.39, 0.55, 0.52]) is False


def test_a_flat_top_means_done():
    m = _load()
    assert m.plateaued([0.20, 0.45, 0.70, 0.70, 0.69, 0.70]) is True


def test_a_single_dip_does_not_count_as_done():
    """★ 单个窗口的抖动不算到顶（实测就有 55%→52% 这种）。

    ⚠️ 用"连续 N 个窗口单调不增"会把一个还在上升的趋势误判成学完了
      ⇒ **看最大值，不看逐点。**
    """
    m = _load()
    assert m.plateaued([0.10, 0.20, 0.30, 0.28, 0.45]) is False


def test_too_short_a_trajectory_is_not_done():
    """样本不够就报"还没到" —— **不猜**（守则④）。"""
    m = _load()
    assert m.plateaued([0.5, 0.5]) is False


def test_completion_is_not_a_kill_signal():
    """★★ 完成信号**不许**进 `rl_guard` 的停机路径。

        停机 = 训练**坏了**，继续跑是在浪费或在把错的训进权重
        完成 = 训练**做完了**，继续跑只是收益递减

    ⇒ 混在一起的话，"跑完了"和"崩了"会长得一样（同 agent_loop 那四种停法）。
    """
    guard = (ROOT / "scripts" / "rl_guard.sh").read_text(encoding="utf-8")
    assert "pool_readout" not in guard, (
        "把完成判据接进了停机守卫 ⇒ 会把'学完了'报成'挂了'")


def test_cases_seen_once_are_not_classified():
    """★ 只抽到 1 次的题，`ema_std` 是**一次观测**不是估计。

    拿它分类会把"还没量准"说成"已经学会" ⇒ **宁可报"没量够"。**
    """
    m = _load()
    g = m.classify({
        "A": {"seen": 1, "ema_std": 0.0, "ema_reward": 1.0},     # 看着像饱和
        "B": {"seen": 3, "ema_std": 0.0, "ema_reward": 1.0},
    })
    assert g["没量够"] == ["A"] and g["饱和"] == ["B"]


def test_the_three_classes_are_distinguished_by_reward_not_just_variance():
    """★★ 三类的**出口完全不同**，所以不能只按方差分成"有梯度/没梯度"两类。

        饱和  分高  已经会了      ⇒ 降权
        卡死  中间  在里面打转    ⇒ 查缺工具还是缺信息，curriculum 只适用这一类
        死格  分低  从没探索到    ⇒ **RL 结构上救不了**，该由 SFT 覆盖去解
    """
    m = _load()
    g = m.classify({
        "sat":  {"seen": 3, "ema_std": 0.0, "ema_reward": 0.98},
        "stuck": {"seen": 3, "ema_std": 0.0, "ema_reward": 0.50},
        "dead": {"seen": 3, "ema_std": 0.0, "ema_reward": 0.02},
        "live": {"seen": 3, "ema_std": 0.30, "ema_reward": 0.50},
    })
    assert g["饱和"] == ["sat"] and g["卡死"] == ["stuck"]
    assert g["死格"] == ["dead"] and g["有梯度"] == ["live"]


def test_the_flat_threshold_matches_the_other_tools():
    """★ `FLAT_STD` 必须和 `compare` / `select_sft_ckpt` 用的是同一个数。

    三处不一致的话，同一条题会在不同报告里被分进不同的类 ——
    而那种"两份实现慢慢漂开"正是本项目付过多次钱的东西。
    """
    m = _load()
    sel = (ROOT / "scripts" / "select_sft_ckpt.py").read_text(encoding="utf-8")
    assert f"> {m.FLAT_STD}" in sel or f"<= {m.FLAT_STD}" in sel
    assert m.SATURATED_REWARD == 0.9 and m.DEAD_REWARD == 0.15
