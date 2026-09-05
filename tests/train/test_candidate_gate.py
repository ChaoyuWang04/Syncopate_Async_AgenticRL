"""v16 候选资格检查：smoke 不冒充 candidate，唯一常量不漂移。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _v16_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SYNCOPATE_CONTRACT"] = "v15"
    env["SYNCOPATE_THINK"] = "1"
    return env


def _gate():
    from syncopate.train import candidate_gate
    return candidate_gate


GROUPS_PER_STEP = 12


def _make_run(tmp: Path, *, purpose, steps, flat_per_window) -> Path:
    """造一个假跑。

    ⚠️ `flat_per_window` 直接给**每个 10 步窗口里有几组是零方差**，
      不给占比 —— 第一版给占比，`round(占比×组数)` 取整后到顶被截平，
      **反而把一条一路创新高的轨迹判成了"到顶"**。造测试数据也会量错对象。
    """
    run = tmp / "run"
    (run / "rollout_dumps").mkdir(parents=True)
    if purpose is not None:
        (run / "run_purpose.json").write_text(
            json.dumps({"purpose": purpose, "steps_requested": steps}), encoding="utf-8")
    for i in range(steps):
        flat = flat_per_window[min(i // 10, len(flat_per_window) - 1)]
        lines = []
        for g in range(GROUPS_PER_STEP):
            rs = [0.5] * 8 if g < flat else [0.1 * k for k in range(8)]
            for r in rs:
                lines.append(json.dumps({"input": f"s{i}g{g}", "reward": r}))
        (run / "rollout_dumps" / f"{i+1}.jsonl").write_text("\n".join(lines),
                                                            encoding="utf-8")
    return run


# ── 不挡 infra ──────────────────────────────────────────────────────────

def test_launch_allows_short_smoke_runs():
    """默认 smoke，短跑不受 candidate 下限约束。"""
    r = subprocess.run([sys.executable, "-m", "syncopate.train.launch_rl_v1",
                        "--dry-run", "--profile", "smoke", "--steps", "60"],
                       cwd=ROOT, env=_v16_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


def test_launch_refuses_a_short_candidate_run():
    """声明 candidate 却只跑 60 步 ⇒ **起跑就硬失败**（这个便宜，先挡）。"""
    r = subprocess.run([sys.executable, "-m", "syncopate.train.launch_rl_v1",
                        "--dry-run", "--profile", "candidate", "--steps", "60"],
                       cwd=ROOT, env=_v16_env(), capture_output=True, text=True)
    assert r.returncode != 0
    assert "至少" in (r.stdout + r.stderr)


# ── 晋级闸的三条判据 ────────────────────────────────────────────────────

def test_a_smoke_run_never_qualifies_even_if_long_and_plateaued():
    """★★ 用途是**当初声明**的，**不许事后追认**。

    允许追认的话，「这跑本来是实验，跑得还不错，就当候选吧」会变成常态 ——
    而那正是"用一个没按候选标准跑的东西上线"的入口。
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(Path(d), purpose="smoke", steps=420,
                        flat_per_window=[1, 4, 8] + [8] * 40)
        ok, reasons = _gate().evaluate(run)
        assert ok is False
        assert any("不是 candidate" in r for r in reasons)


def test_a_candidate_that_stopped_early_is_refused():
    """★★★ 这是整道闸的核心：**跑到步数就停，但还在创新高 ⇒ 不够格。**"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # 一路创新高，**最后三个窗口仍在破纪录**
        rising = [min(11, 1 + i // 4) for i in range(45)]
        run = _make_run(Path(d), purpose="candidate", steps=420,
                        flat_per_window=rising)
        ok, reasons = _gate().evaluate(run)
        assert ok is False
        assert any("还在创新高" in r for r in reasons)


def test_a_proper_candidate_passes():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(Path(d), purpose="candidate", steps=420,
                        flat_per_window=[1, 4, 6, 8] + [8] * 45)   # 到顶后走平
        ok, reasons = _gate().evaluate(run)
        assert ok is True, reasons


def test_all_reasons_are_reported_not_just_the_first():
    """★ 理由**永远给全**，不在第一条不过时短路。

    短路的话，人修完第一条再跑一次才发现还有第二条 ——
    **一次只告诉一个坏消息，是让人反复上机的最快方式。**
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(Path(d), purpose=None, steps=12, flat_per_window=[1, 6])
        ok, reasons = _gate().evaluate(run)
        assert ok is False
        assert len([r for r in reasons if r.startswith("🔴")]) >= 3


# ── 两个数只能有一份 ────────────────────────────────────────────────────

def test_the_minimum_comes_from_v16_launcher_not_a_second_copy():
    """最少步数从 v16 启动器取，闸里不许再写一个数。

    两份会慢慢漂开，而它们都在判"够不够格"，漂开的后果是同一条跑两处结论不同。
    """
    g = _gate()
    from syncopate.train.launch_rl_v1 import MIN_CANDIDATE_STEPS
    assert g.min_candidate_steps() == MIN_CANDIDATE_STEPS
    src = (ROOT / "syncopate" / "train" / "candidate_gate.py").read_text(encoding="utf-8")
    assert "launch_rl.py" not in src


def test_the_gate_reuses_the_readout_implementation():
    """完成判据也只有一份（`pool_readout.plateaued`），闸里不另抄。"""
    src = (ROOT / "syncopate" / "train" / "candidate_gate.py").read_text(encoding="utf-8")
    assert "pool_readout" in src
    assert "def plateaued" not in src, "闸里又实现了一遍完成判据"
