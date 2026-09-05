"""行为塌陷的两道闸：**跑中**（defer_watch）与**跑后**（compare 的 verdict）。

★ 起因（2026-08-19）

`[实测]` lr 1e-4 那跑，「该 defer 时 defer 了」从 **97% 掉到 0%**，
**而任务总分仍然 +0.063（t=+3.9，显著为正）**。

算术闭合：9 条该 defer 的题 × 0.971 = −8.74；其余 334 条 × +0.091 = +30.24
⇒ 净 +0.0627 ≈ 报出来的 +0.063。
**总分那个"漂亮的正收益"，逐位等于「牺牲 9 条换 334 条」这笔交换。**

⇒ 而我们手上每一把打包型尺子都被实测证明会盖住它：
    总分     +0.063 好看        defer 已经 0%
    三计数   打平 +0.000        defer 差 14 个点（41好/269没动/33差，像噪声）
    训练分   说 lr1e-4 更好     任务分说它显著更差（配对 −0.039，t=−3.1）

⚠️⚠️ 而且它**不可逆**：塌了之后那 9 条组内 std=0 ⇒ GRPO 的 advantage 恒为 0
   ⇒ 再也不产生梯度，RL 自己爬不出来。别的指标坏了继续跑还有救，这条没有。
   ⇒ 所以跑中那道闸是**停机**，不是报警。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_watch():
    path = ROOT / "scripts" / "tools" / "defer_watch.py"
    spec = importlib.util.spec_from_file_location("_defer_watch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_defer_watch"] = mod
    spec.loader.exec_module(mod)
    return mod


# ────────────────────────── 跑中：defer_watch ──────────────────────────

def test_trailing_streak_not_longest(  # noqa: D103
):
    """★★ 守卫必须看**当前**连零，不是历史最长。

    `[实测]` `r1_tokenis` 历史最长连零 **18 步**，但当前连零 0 —— 它恢复了。
    用"历史最长"会把它误停，而**假警报会训练人忽略这条判据**（守则③）。
    """
    w = _load_watch()
    counts = [0] * 18 + [3, 1, 2]          # 早期长连零，后来恢复
    assert w.longest_zero_streak(counts) == 18
    assert w.trailing_zero_streak(counts) == 0


def test_trailing_streak_counts_from_the_end():
    w = _load_watch()
    assert w.trailing_zero_streak([5, 1, 0, 0, 0]) == 3
    assert w.trailing_zero_streak([]) == 0
    assert w.trailing_zero_streak([0, 0]) == 2


def _make_run(tmp_path: Path, per_step: list[int]) -> Path:
    """造一个假的 rollout_dumps：每步指定多少条 defer，其余填 tool_call。"""
    d = tmp_path / "run" / "rollout_dumps"
    d.mkdir(parents=True)
    for i, k in enumerate(per_step, start=1):
        lines = []
        for j in range(8):
            beh = "defer" if j < k else "tool_call"
            lines.append(json.dumps({"output": f'... {{"behavior": "{beh}"}}'}))
        (d / f"{i}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path / "run"


def test_healthy_run_passes(tmp_path):
    w = _load_watch()
    run = _make_run(tmp_path, [1, 0, 0, 2, 0, 1])
    assert w.main([str(run)]) == 0


def test_collapsed_run_trips(tmp_path):
    w = _load_watch()
    run = _make_run(tmp_path, [2] + [0] * 30)      # 起头有，之后一路零
    assert w.main([str(run)]) == 1


def test_threshold_is_not_crossed_just_below(tmp_path):
    """门槛必须是「≥ 才停」，别差一步就误停。"""
    w = _load_watch()
    run = _make_run(tmp_path, [1] + [0] * (w.MAX_ZERO_STREAK - 1))
    assert w.main([str(run)]) == 0


def test_missing_dumps_is_not_a_collapse(tmp_path):
    """跑刚起来还没 dump ⇒ 报"没有"，**不要当成塌陷**（守则④）。"""
    w = _load_watch()
    (tmp_path / "run").mkdir()
    assert w.main([str(tmp_path / "run")]) == 0


def test_last_behavior_wins(tmp_path):
    """一条轨迹有多步，**终答那步**才是它的行为。"""
    w = _load_watch()
    d = tmp_path / "run" / "rollout_dumps"
    d.mkdir(parents=True)
    (d / "1.jsonl").write_text(json.dumps(
        {"output": '{"behavior": "defer"} 然后 {"behavior": "tool_call"}'}), encoding="utf-8")
    assert w.defer_counts(tmp_path / "run") == [0]


def test_threshold_is_far_above_the_healthy_baseline():
    """★ 门槛必须**远高于**直觉：RL 桶里只有 3.4% 的题该 defer，每步只抽 6 题
    ⇒ 多数步天然就是 0。实测健康跑最长连零 9 / 16 / 18 步。"""
    w = _load_watch()
    assert w.MAX_ZERO_STREAK > 18, "门槛不高于实测的健康上界，就会天天误报"


# ────────────────────────── 跑后：compare 的 verdict ──────────────────────────

def _rows(defer_rate: float, n_defer: int = 9, n_other: int = 100):
    """造审计行：`behaviors` 是 8 次采样的行为列表。"""
    out = {}
    for i in range(n_defer):
        k = round(defer_rate * 8)
        out[f"IMM_{i:04d}"] = {
            "case_id": f"IMM_{i:04d}", "reward": 1.0, "reward_std": 0.1, "caps": [],
            "expected_behavior": "defer",
            "behaviors": ["defer"] * k + ["answer"] * (8 - k)}
    for i in range(n_other):
        out[f"ATTR_{i:04d}"] = {
            "case_id": f"ATTR_{i:04d}", "reward": 0.7, "reward_std": 0.1, "caps": [],
            "expected_behavior": "tool_call", "behaviors": ["tool_call"] * 8}
    return out


def test_verdict_reds_on_a_real_collapse(capsys):
    from syncopate.train.compare import behavior_verdict
    a, b = _rows(1.0), _rows(0.0)
    behavior_verdict("基线", a, "候选", b, sorted(a))
    out = capsys.readouterr().out
    assert "🔴" in out and "不可逆" in out


def test_verdict_greens_when_defer_is_preserved(capsys):
    """`r1_seqis` 的形状：该 defer 保住 ⇒ 必须放行，否则又是一条假警报。"""
    from syncopate.train.compare import behavior_verdict
    a, b = _rows(1.0), _rows(1.0)
    behavior_verdict("基线", a, "候选", b, sorted(a))
    out = capsys.readouterr().out
    assert "✅ 该 defer 率没有退化" in out
    assert "🔴 该 defer" not in out


def test_verdict_catches_the_14_point_drop(capsys):
    """★ 钉住 `r1_tokenis` 那个形状：总分**完全打平**，而 defer 掉 14 个点。
    门槛必须抓得住它 —— 那正是三计数没抓住的那一次。"""
    from syncopate.train.compare import behavior_verdict
    a, b = _rows(1.0), _rows(0.83)
    behavior_verdict("基线", a, "候选", b, sorted(a))
    assert "🔴" in capsys.readouterr().out


def test_missing_behaviors_field_reports_undetermined(capsys):
    """旧产物没有 `behaviors` ⇒ 打「**无法判定**」，不许读成通过（守则⑦）。"""
    from syncopate.train.compare import behavior_verdict
    a = _rows(1.0)
    b = {k: {kk: vv for kk, vv in v.items() if kk != "behaviors"} for k, v in a.items()}
    behavior_verdict("基线", a, "候选", b, sorted(a))
    out = capsys.readouterr().out
    assert "无法判定" in out and "不是通过" in out
