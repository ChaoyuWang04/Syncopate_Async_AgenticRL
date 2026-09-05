"""考卷 v4 的结构判据（`26 §W1` 门槛①②③）：多样性断言 · L1–L4 与 v3 逐字相等 · 判卷器负向认证。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data/u_route/context_v4_exam.jsonl"
V3 = ROOT / "data/u_route/context_v3_exam.jsonl"
NEW = ["REJ", "DEF", "CLA", "HARD", "DEF-F", "REJ-F", "CLA-F", "L2-x", "WIN"]


def _rows(p):
    return [json.loads(x) for x in open(p)]


def test_l1_to_l4_inherited_verbatim():
    v3 = {r["id"]: r for r in _rows(V3) if r["level"] in ("L1", "L2", "L3", "L4")}
    v4 = {r["id"]: r for r in _rows(V4) if r["id"] in v3}
    assert v4 == v3, "L1–L4 与 v3 不逐字相等 ⇒ 跨版本读数不可比"
    assert len(v3) == 125


def test_every_new_tier_is_resolvable_and_diverse():
    by = {}
    for r in _rows(V4):
        by.setdefault(r["level"], []).append(r)
    for lv in NEW:
        xs = by[lv]
        assert len(xs) >= 20, f"{lv}: n={len(xs)} <20 ⇒ 1 题 >5pp"
        assert len({r["turns"][-1][:6] for r in xs}) >= 2, f"{lv}: 只有一种问法"
        if lv != "CLA":
            assert len({r["judge"].get("campaign") for r in xs} - {None}) >= 2, f"{lv}: 单一对象"
    paired = [r for r in _rows(V4) if "pair" in r]
    assert all(n == 2 for n in Counter(r["pair"] for r in paired).values()), "对照不成对"
    hard = sum(len(by[l]) for l in ("REJ", "DEF", "CLA", "L4", "DEF-F", "REJ-F", "CLA-F"))
    assert hard >= 150, f"硬预期行为题 {hard} <150"
    assert min(len(r["prior"]) for r in by["WIN"] if not r["judge"]["in_window"]) >= 7
    assert len({len(r.get("prior") or []) for r in by["L2-x"]}) >= 2, "L2-x 轮距没散开"
    endings = {(p.get("result") or {}).get("signal") or "answer" for r in _rows(V4) for p in r.get("prior") or []}
    assert {"answer", "defer", "reject", "clarify"} <= endings


def test_generator_assertions_can_fail():
    """撤掉一条多样性（只留一个对象）⇒ 生成器的结构断言必须红。"""
    import importlib
    from syncopate.evaluation import build_exams as g
    rows = [dict(r) for r in _rows(V4)]
    for r in rows:
        if r["level"] == "DEF":
            r["judge"] = {**r["judge"], "campaign": "CMP_2"}
    with pytest.raises(AssertionError):
        g._assert_structure(rows)
    importlib.reload(g)  # 生成器在导入时已重建文件；reload 保证产物与磁盘一致


def test_judge_v4_negative_certification():
    env = {**os.environ, "SYNCOPATE_CONTRACT": "v15"}
    p = subprocess.run([sys.executable, "-m", "syncopate.evaluation.exam_certify"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "负向认证通过" in p.stdout
