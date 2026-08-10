"""数据成熟度：把「这个数字现在可不可信」从模型的猜测变成可计算的事实。

★ 为什么这是本业务的第一性约束

归因是延迟的。今天改的预算，7 天后才知道 ROAS 对不对；而 D1 的数字**今天就有，
而且极易被当成结论**。一个只见过「指标 = 一个确定的数」的沙盒，训出来的 agent
上线第一天就会拿 D1 ROAS 去砍预算——这不是模型笨，是我们没把世界建对。

所以沙盒必须有两件东西：

    as_of        今天是哪天（由 env.reference_now 给，不由模型声明）
    成熟曲线     每个指标要跑多少天才收敛（ROAS 7 天 / CPI 3 天 / CTR 1 天）

★★ 未收敛时给区间，不给点估计

这个设计本身就在教模型「现在还说不准」。点估计无论多不准，看起来都像个结论；
而一个横跨安全线两侧的区间，**从形式上就下不了决策**。

    安全线 0.40，D7 已收敛 → ROAS 0.45           → 可以扩量
    安全线 0.40，D2 未收敛 → ROAS ∈ [0.25, 0.65] → 跨线，defer

区间中心用真值，不是为了藏答案——判据是**区间跨不跨线**，不是中心在哪。
藏中心反而会让 verifier 无法用规则判对错。

★★★ 一个刻意的偏差：`as_of` 不做成工具参数

设计文档 §10 的签名是 `metrics.get_freshness(metric, campaign_id, as_of)`。
我们去掉了 `as_of`——**模型不该有权声明「今天是哪天」**。
它一旦能传 as_of，「数据还没到」这件事就变成了它可以绕过去的参数，
premature_decision_cap 会被它自己填的日期架空。时间是世界的属性，不是请求的属性。
"""

from __future__ import annotations

from typing import Any

# 各指标的收敛天数。ROAS 最慢（要等回收），CTR 当天就稳。
CONVERGE_DAYS: dict[str, int] = {
    "roas_d7": 7,
    "cpi": 3,
    "ctr": 1,
    "retention_d1": 1,
    "installs": 3,
}

# 观测值相对终值的形变。shape(1.0) 必须 == 1.0，否则收敛时和世界里的真值对不上。
#   roas  收入是累积的 → 早期系统性偏低（D1 大约只有终值的三成）
#   cpi   学习期贵 → 早期系统性偏高
_SHAPE = {
    "roas_d7": lambda x: x ** 0.6,
    "cpi": lambda x: 1.0 + 0.45 * (1.0 - x),
    "ctr": lambda x: 1.0 + 0.15 * (1.0 - x),
    "retention_d1": lambda x: 1.0,
}

# 区间半宽随进度收窄：刚开投时 ±50%，收敛时归零。
_HALF_WIDTH_AT_ZERO = 0.50

# 低于这个进度就算 immature（设计文档 §8.2：只有 D1 → immature；D3 有、D7 未到 → partial）
PARTIAL_FRACTION = 0.4

# 归因所需的最小样本量（附录 A3 的默认值）。样本不够时，
# **时间到了也不算成熟**——这是两种不同的「不可信」。
MIN_SAMPLE_INSTALLS = 300

MATURE, PARTIAL, IMMATURE = "mature", "partial", "immature"


def progress(metric: str, days_elapsed: float) -> float:
    """投放进度 x ∈ [0,1]：已跑天数 / 该指标的收敛天数。"""
    converge = CONVERGE_DAYS.get(metric, 7)
    return max(0.0, min(1.0, days_elapsed / converge)) if converge > 0 else 1.0


def observed_value(metric: str, final_value: float, days_elapsed: float) -> float:
    """世界里存的是**终值**；这个函数给出「今天看上去是多少」。"""
    shape = _SHAPE.get(metric)
    return round(final_value * shape(progress(metric, days_elapsed)), 4) if shape else final_value


def sample_size_at(installs_7d: float, days_elapsed: float) -> int:
    """安装量按天累积。样本量不够是 insufficient_sample 的判据，和时间是两回事。"""
    return int(installs_7d * min(1.0, max(0.0, days_elapsed) / 7.0))


def metric_maturity(
    metric: str,
    *,
    days_elapsed: float,
    final_value: float,
    installs_7d: float,
) -> dict[str, Any]:
    """一个指标此刻的成熟度全貌。工具、verifier、cap 三边共用它，口径只有一份。"""
    converge = CONVERGE_DAYS.get(metric, 7)
    x = progress(metric, days_elapsed)
    current = observed_value(metric, final_value, days_elapsed)
    sample = sample_size_at(installs_7d, days_elapsed)

    converged_by_time = days_elapsed >= converge
    enough_sample = sample >= MIN_SAMPLE_INSTALLS

    if converged_by_time and enough_sample:
        level, reason = MATURE, "已过收敛期且样本量充足"
    elif not enough_sample and converged_by_time:
        level = IMMATURE
        reason = f"时间已到但样本量不足（{sample} < {MIN_SAMPLE_INSTALLS}）"
    elif x >= PARTIAL_FRACTION:
        level, reason = PARTIAL, f"已跑 {days_elapsed:.0f} 天，尚未到 {converge} 天收敛期"
    else:
        level = IMMATURE
        reason = f"仅跑了 {days_elapsed:.0f} 天，远未到 {converge} 天收敛期"

    half = _HALF_WIDTH_AT_ZERO * (1.0 - x)
    return {
        "metric": metric,
        "maturity": level,
        "is_converged": converged_by_time and enough_sample,
        "days_elapsed": int(days_elapsed),
        "converge_at_day": converge,
        "converge_eta_days": max(0, converge - int(days_elapsed)),
        "sample_size": sample,
        "min_sample_size": MIN_SAMPLE_INSTALLS,
        "current_value": current,
        # ★ 未收敛时这是区间；收敛时上下界重合，退化成点估计
        "expected_final_range": [round(final_value * (1 - half), 4),
                                 round(final_value * (1 + half), 4)],
        "reason": reason,
    }


def straddles(range_: list[float], threshold: float) -> bool:
    """区间跨不跨某条线。跨线 = 这个决策现在做不了 = 该 defer。"""
    return range_[0] <= threshold <= range_[1]


def campaign_maturity(row: dict[str, Any], metric: str = "roas_d7") -> dict[str, Any]:
    """从 campaign 行直接算成熟度。`started_days_ago` 是世界给的，不是模型给的。"""
    return metric_maturity(
        metric,
        days_elapsed=float(row.get("started_days_ago", 30)),
        final_value=float(row.get(metric, 0.0)),
        installs_7d=float(row.get("installs_7d", 0.0)),
    )
