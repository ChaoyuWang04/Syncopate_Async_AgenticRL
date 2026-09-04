"""三查脚本自身的判据（`26 §W0`）：修订表零缺口，且旧表必须报出缺口（负向认证）。

⚠️ 读的是进了版本管理的落盘文件（judged_v15r3c_r1..4 · blind_scores_v145 · _audit/v15_r5）。
   文件不在 ⇒ 脚本按守则④报"无读数"，本测试会因可达性空格而红——那是对的。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "v16_gate_triage.py"


# 09-05：三查不再有默认输入（此前默认读这些 v15-R3 文件，把旧读数当本次的）⇒ 测试显式把版本管理里的历史夹具传进去
FIXTURE = ["--judged", "logs/u_route/judged_v15r3c_r*_context_v3.jsonl",
           "--blind", "logs/u_route/blind_scores_v145.json", "--blind-key", "logs/u_route/blind_key.json",
           "--r5-audit", "_audit/v15_r5/r3_sel_f25.json"]


def _run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *FIXTURE, *args, "--out", "/tmp/gate_triage_test.json"],
                       cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout


def test_revised_table_has_no_gaps_and_no_blank_cells():
    code, out = _run()
    assert code == 0, out
    assert "缺口 0 条" in out
    # 三查表每条都要有 n / 读数 / 可达性依据：任何一格空着脚本就会报缺口（上面已断言 0）
    assert "门槛 38 条" in out or "门槛 3" in out


def test_legacy_table_is_caught():
    """判据必须能对自己失败：08-30 那张表至少报出 26 §2.3 的五条病。"""
    code, out = _run("--legacy")
    assert code == 2, out
    for must in ("n/遍=8<20", "挂错阶段", "没有测量装置", "SE_diff 1.2pp > 阈值/2 0.5pp"):
        assert must in out, must
