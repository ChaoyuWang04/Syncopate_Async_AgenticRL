"""固定管线：**参数只有一份，脚本不许各抄一份**。

★ 起因（2026-08-19）

2026-08-18 的实况：15 个启动脚本**全都**走 `launch_rl` ——
「入口只有两个」（`08 §4`）这条纪律其实是守住的。
但**每个脚本各自抄了一份契约参数**，于是抄着抄着漂成了两套：

    11 个   --max-prompt-length 5120 --max-response-length 2048   ✅ 当时是对的
     4 个   --max-prompt-length 3584 --max-response-length 1536   🔴 停在旧值

⇒ ★ 关键在于**那 11 个当时是对的**。旧判据只比"传的值对不对"，所以它们一路绿灯 ——
  而它们的问题不是值错，是**存在一份副本**：副本在下一次改契约时必然漏掉几个。
⇒ 所以判据要写在「**只该有一份**」上，而不是「这一份的值对不对」上（守则①）。

⚠️ 逃生口是刻意留的：确实要做契约参数的 A/B 时写 `# CONTRACT-OVERRIDE: <理由>`。
   没有逃生口，合法的对照实验会天天报红，而**假警报会训练人忽略这条判据**（守则③）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    """按**文件路径**导入（`scripts/` 不是包，`importorskip` 会静默 skip）。"""
    path = ROOT / "scripts" / "check_pipeline_invariants.py"
    spec = importlib.util.spec_from_file_location("_cpi_contract", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cpi_contract"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cpi():
    return _load()


@pytest.fixture()
def fake_scripts(cpi, monkeypatch, tmp_path):
    """把检查器的 ROOT 指到临时目录，用**我们自己造的脚本**测判据。

    ⚠️ 不拿仓库真脚本测：那样测试会随别人改脚本而红/绿，
       变成一把不可信的尺子 —— 而"判据不可信"正是这一族问题的主题。
    """
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(cpi, "ROOT", tmp_path)
    return tmp_path / "scripts"


def _run(cpi, name):
    """跑指定的检查，返回 (通过与否, 打出来的行)。"""
    lines: list[str] = []
    for _g, n, fn in cpi.CHECKS:
        if n == name:
            return fn(lines.append), lines
    raise AssertionError(f"没找到检查：{name}")


CONTRACT_CHECK = "★ 跑训练的脚本一律不许显式传契约参数（长度预算 + 采样参数）"
ENTRY_CHECK = "★ 起训练只能走固定入口（sft / launch_rl），不许另起一套"


def test_clean_scripts_pass(cpi, fake_scripts):
    (fake_scripts / "ok.sh").write_text(
        "python -m syncopate.train.launch_rl --lora-rank 32 --train-batch-size 6\n")
    ok, lines = _run(cpi, CONTRACT_CHECK)
    assert ok, lines


def test_explicit_contract_param_is_a_violation_even_when_the_value_is_correct(
        cpi, fake_scripts):
    """★★ 这条是整族的核心：**值对也不行**。

    那 11 个脚本传的正是当时正确的 5120/2048，旧判据因此放过了它们 ——
    而它们才是下一次漂移的来源。
    """
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
    (fake_scripts / "correct_but_copied.sh").write_text(
        f"python -m syncopate.train.launch_rl "
        f"--max-prompt-length {MAX_PROMPT_LENGTH} "
        f"--max-response-length {MAX_RESPONSE_LENGTH}\n")
    ok, lines = _run(cpi, CONTRACT_CHECK)
    assert not ok, "传的值虽对，但它是一份副本 ⇒ 必须报违反"
    assert any("显式传了契约参数" in x for x in lines)


def test_stale_value_is_a_violation(cpi, fake_scripts):
    (fake_scripts / "stale.sh").write_text(
        "python -m syncopate.train.launch_rl "
        "--max-prompt-length 3584 --max-response-length 1536\n")
    ok, _ = _run(cpi, CONTRACT_CHECK)
    assert not ok


def test_sampling_params_are_covered_too(cpi, fake_scripts):
    """契约不只是长度 —— 采样参数漂移过一次（评测 0.95/20 vs 训练 1.0/-1）。"""
    (fake_scripts / "sampling.sh").write_text(
        "python -m syncopate.train.eval_local --top-p 0.95 --top-k 20\n")
    ok, _ = _run(cpi, CONTRACT_CHECK)
    assert not ok


def test_comment_lines_do_not_trigger(cpi, fake_scripts):
    """注释里提到参数名不算违反 —— 否则解释这条纪律的注释自己会把判据点红。"""
    (fake_scripts / "commented.sh").write_text(
        "# 不要在这里传 --max-prompt-length，见 rollout_budget.py\n"
        "python -m syncopate.train.launch_rl --lora-rank 32\n")
    ok, lines = _run(cpi, CONTRACT_CHECK)
    assert ok, lines


def test_override_marker_waives_but_is_reported(cpi, fake_scripts):
    """逃生口：声明了就不算违反，但必须**留痕**（打出来给人看见）。"""
    (fake_scripts / "ab.sh").write_text(
        "# CONTRACT-OVERRIDE: E9 要扫长度预算对截断率的影响\n"
        "python -m syncopate.train.launch_rl --max-response-length 1024\n")
    ok, lines = _run(cpi, CONTRACT_CHECK)
    assert ok
    assert any("已显式声明覆盖" in x for x in lines), "豁免必须留痕，不能静默"


def test_override_marker_must_be_adjacent(cpi, fake_scripts):
    """标记只在同行或上一行生效 —— 否则文件顶上写一句就把整个文件豁免了。"""
    (fake_scripts / "far.sh").write_text(
        "# CONTRACT-OVERRIDE: 顶上写一句\n"
        "echo 中间隔了一行\n"
        "python -m syncopate.train.launch_rl --max-response-length 1024\n")
    ok, _ = _run(cpi, CONTRACT_CHECK)
    assert not ok, "隔行的标记不该生效，否则豁免会越滚越大"


def test_bypassing_the_entrypoint_is_a_violation(cpi, fake_scripts):
    """直接调训练框架 = 绕过 launch_rl 上挂着的契约默认值 / 起手断言 / 守卫。"""
    (fake_scripts / "bypass.sh").write_text(
        "python -m verl.trainer.main_ppo trainer.nnodes=1\n")
    ok, lines = _run(cpi, ENTRY_CHECK)
    assert not ok, lines


def test_canonical_entrypoints_pass(cpi, fake_scripts):
    (fake_scripts / "good.sh").write_text(
        "python -m syncopate.train.sft --model models/Qwen3-4B\n"
        "python -m syncopate.train.launch_rl --lora-rank 32\n")
    ok, lines = _run(cpi, ENTRY_CHECK)
    assert ok, lines


def test_the_repo_itself_is_clean(cpi):
    """★ 回归：仓库现在必须是干净的（本次清理的验收口径）。

    ⚠️ 这条**故意**跑真仓库 —— 上面那些跑的是假脚本，测的是判据本身；
       这一条测的是「清理做完了没有」。两者都要有。
    """
    for name in (CONTRACT_CHECK, ENTRY_CHECK):
        ok, lines = _run(cpi, name)
        assert ok, f"{name} 红了：{lines}"
