"""管线检查器的**结局归类**：「查出违反」和「没查成」必须分开。

★ 起因（2026-08-18 晚，当场踩的）

`check_pipeline_invariants.py` 原本把「检查抛异常」和「检查查出违反」写进同一个
`failed` 列表，汇总只打一句「🔴 N 条不通过」。
用错解释器跑了一次（`/venv/main` 而不是项目 `.venv`），4 条检查因
`ModuleNotFoundError: torch / safetensors / syncopate` 变成"不通过"，
和 3 条**真红**混报成「🔴 7 条不通过」——其中 4 条是假的。

⇒ 危险方向不是"假通过"，是**假失败**。而**假警报会训练人忽略这条判据，
  比没有判据更糟**（守则③）。这个项目现在正靠这一族检查兜底。
⇒ 但「无法判定」也**不许算过**（守则⑦：空着的门槛应读作"无法判定"）
  ⇒ 判据是：**非零退出码 + 和"违反"分开显示**。

★ 同一天 `E22 §6.5.3` 记的那条说的就是这件事：
  **判据可以假通过，也可以假失败；打印本身要能区分"读不到"和"读到了是空"。**

⚠️ 这里钉的是**归类逻辑**，不是某一条具体检查 —— 所以用临时注册的假检查来测，
   不依赖仓库里的真产物（真产物的红/绿会随重跑变，那样测试就成了不可信的尺子）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    """按**文件路径**导入 `scripts/` 下的脚本。

    ⚠️ 不用 `importorskip("scripts.xxx")` —— `scripts/` 不是包，那样会**静默 skip**，
    而「跳过不是通过」是记过的一条。验收口径是 **0 skipped**。
    """
    path = ROOT / "scripts" / "check_pipeline_invariants.py"
    spec = importlib.util.spec_from_file_location("_cpi_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cpi_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cpi():
    """拿到模块，并在测试期间把 `CHECKS` 换成我们自己注册的假检查。"""
    mod = _load()
    saved = list(mod.CHECKS)
    mod.CHECKS.clear()
    try:
        yield mod
    finally:
        mod.CHECKS[:] = saved


def _register(mod, group, name, fn):
    mod.CHECKS.append((group, name, fn))


def test_violation_and_undetermined_are_counted_separately(cpi, capsys):
    """一条真违反 + 一条抛异常 ⇒ **两个计数分开**，不再合成一句"N 条不通过"。"""
    _register(cpi, "g", "真的查出违反", lambda log: False)
    _register(cpi, "g", "探针没跑成", lambda log: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'torch'")))

    violated, undetermined = cpi.run_checks(["g"])

    assert violated == ["真的查出违反"]
    assert [n for n, _ in undetermined] == ["探针没跑成"]


def test_undetermined_is_not_a_pass(cpi):
    """★ 最要紧的一条：**只有无法判定**时，退出码必须非零。

    守则⑦：空着的门槛应读作"无法判定"，不能读作通过。
    """
    _register(cpi, "g", "探针没跑成", lambda log: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cpi.main(["--only", "g"]) == 2


def test_violation_outranks_undetermined_in_exit_code(cpi):
    """同时有违反和无法判定 ⇒ 退出码报**违反**（1），因为那条才有行动可做。"""
    _register(cpi, "g", "真的查出违反", lambda log: False)
    _register(cpi, "g", "探针没跑成", lambda log: (_ for _ in ()).throw(ValueError("x")))

    assert cpi.main(["--only", "g"]) == 1


def test_all_green_still_returns_zero(cpi):
    """回归：全过仍然是 0 —— 上面的改动不许把"绿"也变成非零。"""
    _register(cpi, "g", "过", lambda log: True)

    assert cpi.main(["--only", "g"]) == 0


def test_undetermined_wording_does_not_say_failed(cpi, capsys):
    """措辞判据：无法判定那一段**不许**出现"不通过"。

    ⚠️ 这条看着像在测文案，但它钉的是这次事故的**直接成因**：
    人是照着屏幕上那句「🔴 7 条不通过」去行动的。措辞就是判据的界面。
    """
    _register(cpi, "g", "探针没跑成", lambda log: (_ for _ in ()).throw(OSError("nope")))

    cpi.main(["--only", "g"])
    out = capsys.readouterr().out
    assert "无法判定" in out
    assert "不通过" not in out
    assert "不等于通过" in out


def test_exception_type_is_surfaced_not_swallowed(cpi, capsys):
    """异常类型要打出来 —— 不然"没跑成"和"没跑成的原因"又混成一件事。"""
    _register(cpi, "g", "探针没跑成", lambda log: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'safetensors'")))

    cpi.main(["--only", "g"])
    out = capsys.readouterr().out
    assert "ModuleNotFoundError" in out
    assert "safetensors" in out          # 单行里要能看到到底缺哪个


def test_interpreter_note_only_fires_off_venv(cpi, monkeypatch, tmp_path):
    """解释器提示：在项目 `.venv` 里跑不该出现；不在则应出现。

    ⚠️ 它只是提示不是判据 —— 别的解释器跑这个脚本是合法的。
    ⚠️ 用 tmp_path 造假 `.venv` 而不是依赖本机真的有一个：
       「本机恰好没有 ⇒ 这条静默 skip」正是「跳过不是通过」那条要防的。
    """
    fake_venv = tmp_path / ".venv"
    fake_venv.mkdir()
    monkeypatch.setattr(cpi, "ROOT", tmp_path)

    monkeypatch.setattr(cpi.sys, "prefix", str(fake_venv))
    assert cpi._interpreter_note() is None

    monkeypatch.setattr(cpi.sys, "prefix", "/venv/main")
    note = cpi._interpreter_note()
    assert note is not None and "/venv/main" in note


def test_no_note_when_project_has_no_venv(cpi, monkeypatch, tmp_path):
    """项目里没有 `.venv` 时不许提示 —— 干净机器上不该冒出一条无从执行的建议。"""
    monkeypatch.setattr(cpi, "ROOT", tmp_path)
    monkeypatch.setattr(cpi.sys, "prefix", "/usr")
    assert cpi._interpreter_note() is None
