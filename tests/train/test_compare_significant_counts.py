"""逐题三计数：门槛必须跟着每题自己的采样噪声走，不能是固定值。

★ 为什么要守这条（2026-08-17 用一次误读换来的）

M7-b 的配对差值 +0.020 恰好等于 MDE ⇒ 结论「没测出差异」，
而它被读成了「模型基本没变」—— 于是下一步去调 lr / 加步数。
逐题拆开完全不是那回事：**均值是相抵之后的残差**。

★ 而"逐题"本身也有一个坑：`compare` 原来的赢/平/输门槛是 `1e-9`
（任何非零都算），实测 343 题里 272 题"变了" —— 那个数被采样噪声主导。
本文件守的就是修好之后的行为。
"""

from __future__ import annotations

from syncopate.train.compare import significant_counts


def _rows(specs):
    """specs: {case_id: (reward, reward_std)}"""
    return {cid: {"case_id": cid, "reward": r, "reward_std": sd} for cid, (r, sd) in specs.items()}


def test_high_variance_case_small_diff_counts_as_unchanged():
    """高方差题上的小差值 = 采样抖动，**不能**算显著变化。

    固定门槛（比如 ±0.05）会把它误判成"变好" —— 这正是要避免的。
    """
    a = _rows({"INJ_0001": (0.50, 0.32)})          # 8 次采样 std 0.32 ⇒ SE≈0.113 ⇒ 门槛≈0.226
    b = _rows({"INJ_0001": (0.56, 0.32)})          # 差值 +0.06 > 0.05，但远小于门槛
    out = significant_counts(a, b, ["INJ_0001"])
    assert out == {**out, "better": 0, "worse": 0, "flat": 1}, out


def test_zero_variance_case_small_diff_counts_as_significant():
    """零方差题上的同样大小的差值，**是**真变化。

    ⇒ 同一个差值 +0.06，在两道题上应当得到相反的判定 ——
      这就是"门槛必须跟着每题走"的全部理由。
    """
    a = _rows({"CHAT_0001": (0.90, 0.0)})
    b = _rows({"CHAT_0001": (0.96, 0.0)})
    out = significant_counts(a, b, ["CHAT_0001"])
    assert (out["better"], out["worse"], out["flat"]) == (1, 0, 0), out


def test_worse_side_is_reported_by_template():
    """「显著变差」非零就不能说没变化 ⇒ 必须能一眼看出变差集中在哪一类题。"""
    ids = [f"POLU_{i:04d}" for i in range(3)] + ["CHAT_0000"]
    a = _rows({c: (0.95, 0.0) for c in ids})
    b = _rows({**{c: (0.80, 0.0) for c in ids[:3]}, "CHAT_0000": (0.95, 0.0)})
    out = significant_counts(a, b, ids)
    assert out["worse"] == 3 and out["flat"] == 1
    assert out["worse_by_template"]["POLU"] == 3
    assert out["worst_cases"][0][0].startswith("POLU")


def test_counts_partition_the_case_set():
    """三个数必须恰好把所有 case 分完 —— 漏掉一类就会得出"没变化"的错觉。"""
    ids = [f"T_{i:04d}" for i in range(20)]
    a = _rows({c: (0.5, 0.1) for c in ids})
    b = _rows({c: (0.5 + 0.03 * (i % 5 - 2), 0.1) for i, c in enumerate(ids)})
    out = significant_counts(a, b, ids)
    assert out["better"] + out["worse"] + out["flat"] == len(ids)


def test_missing_reward_std_degrades_to_exact_comparison():
    """老审计没有 reward_std ⇒ 门槛退化成 0，任何非零都算变化。

    ⚠️ 这是**刻意**的：宁可退回旧口径，也不要静默地按 0 噪声当"没噪声"来放宽门槛。
    """
    a = {"X_0000": {"case_id": "X_0000", "reward": 0.5}}
    b = {"X_0000": {"case_id": "X_0000", "reward": 0.5001}}
    out = significant_counts(a, b, ["X_0000"])
    assert (out["better"], out["worse"], out["flat"]) == (1, 0, 0), out
