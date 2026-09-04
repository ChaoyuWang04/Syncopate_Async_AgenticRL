"""三桶隔离硬机制（2026-09-04）：出口闸必须拦住越桶底题；派生行编号能反推底题；正规 data build 出口也过闸。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from syncopate.pipeline.split import (assert_split_isolation, base_case_id, load_split_bundles,
                                      split_isolation_report, DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR)

SPLIT = Path(DEFAULT_SPLIT_DIR)
pytestmark = pytest.mark.skipif(not (SPLIT / "sft_cases.json").exists(), reason="需要切分目录")


def _ids(name):
    return json.load(open(SPLIT / f"{name}_cases.json"))["case_ids"]


def test_base_case_id_strips_derived_suffixes():
    assert base_case_id("FRESH_0039_DEFF") == "FRESH_0039"
    assert base_case_id("LONG_0048_MT5") == "LONG_0048"
    assert base_case_id("BUD_0030_COT15") == "BUD_0030"
    assert base_case_id("CHAT_0000_WIN") == "CHAT_0000"
    assert base_case_id("FAIL_0116_L2X") == "FAIL_0116"
    assert base_case_id("L1F_0007") == "L1F_0007"          # 无底题：原样


def test_gate_blocks_eval_and_rl_bases_in_sft_product():
    ev, rl, sft = _ids("eval")[0], _ids("rl")[0], _ids("sft")[0]
    rows = [{"case_id": sft}, {"case_id": f"{sft}_MT5"}, {"case_id": f"{ev}_DEFF"}, {"case_id": rl}]
    rep = split_isolation_report(rows, SPLIT, "sft")
    assert not rep["ok"] and len(rep["offenders"]) == 2
    assert {b for _, _, b in rep["offenders"]} == {"eval", "rl"}
    with pytest.raises(AssertionError, match="三桶隔离"):
        assert_split_isolation(rows, SPLIT, "sft")


def test_gate_uses_source_case_ids_column_when_present():
    ev, sft = _ids("eval")[0], _ids("sft")[0]
    # 编号看着无害（L1F_）但登记的底题在 eval ⇒ 必须红
    rows = [{"case_id": "L1F_0001", "source_case_ids": [ev]}, {"case_id": "L1F_0002", "source_case_ids": [sft]}]
    rep = split_isolation_report(rows, SPLIT, "sft")
    assert rep["offenders"] == [("L1F_0001", ev, "eval")]


def test_sft_only_rows_pass():
    rows = [{"case_id": c} for c in _ids("sft")[:20]] + [{"case_id": "L1F_0000"}]
    rep = assert_split_isolation(rows, SPLIT, "sft")
    assert rep["ok"] and rep["counts"]["eval"] == 0 and rep["counts"]["rl"] == 0


@pytest.mark.skipif(not Path(DEFAULT_BATCH_DIR, "manifest.json").exists(), reason="需要题库")
def test_load_split_bundles_never_returns_other_buckets():
    b = load_split_bundles(Path(DEFAULT_BATCH_DIR), SPLIT, "sft")
    assert set(b) == set(_ids("sft"))
    assert not (set(b) & set(_ids("eval"))) and not (set(b) & set(_ids("rl")))
