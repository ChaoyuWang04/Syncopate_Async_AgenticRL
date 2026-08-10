"""守住一个真实发生过的泄漏：三桶切好了，但数据构造那步没读它。

M0 的 `split_report.json` 写着「三桶零重叠 ✅」，而 `data/sft/v2` 是全部 580 条——
**52 条冻结 EVAL 一直在训练数据里**。切分文件和训练数据是两件事。
"""

from __future__ import annotations

import json

import pytest

from syncopate.pipeline.build_dataset import build
from syncopate.pipeline.split import split, write
from tests.pipeline.test_dead_grid import make_batch


@pytest.fixture
def batch_and_split(tmp_path):
    batch = make_batch(tmp_path / "batch", {
        "BUD|tool_call|denied|id_given": 12,
        "CLAR|clarify|-|id_given": 12,
    })
    buckets, report = split(batch)
    split_dir = tmp_path / "splits"
    write(buckets, report, split_dir)
    return batch, split_dir, buckets


def test_rl_dataset_excludes_frozen_eval(batch_and_split, tmp_path):
    batch, split_dir, buckets = batch_and_split
    build(batch_dir=batch, out_dir=tmp_path / "out", pool="rl",
          artifact_root=tmp_path / "rollouts", split_dir=split_dir)

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(buckets.rl)
    assert manifest["excluded_by_split"] == len(buckets.eval) + len(buckets.sft)

    import pyarrow.parquet as pq
    built = set()
    for name in ("train.parquet", "val.parquet"):
        table = pq.read_table(tmp_path / "out" / name).to_pylist()
        built |= {r["extra_info"]["case_id"] for r in table}
    assert built & set(buckets.eval) == set()
    assert built == set(buckets.rl)


def test_forgetting_the_split_is_an_error_not_a_silent_leak(batch_and_split, tmp_path):
    """★ 忘了声明必须当场报错。安静地退回全量，就是当初泄漏的成因。"""
    batch, _, _ = batch_and_split
    with pytest.raises(ValueError, match="二选一"):
        build(batch_dir=batch, out_dir=tmp_path / "out", pool="rl",
              artifact_root=tmp_path / "rollouts")


def test_full_batch_must_be_explicit(batch_and_split, tmp_path):
    batch, split_dir, _ = batch_and_split
    with pytest.raises(ValueError, match="二选一"):   # 两个都给也是错的
        build(batch_dir=batch, out_dir=tmp_path / "out", pool="rl",
              artifact_root=tmp_path / "rollouts", split_dir=split_dir, full_batch=True)

    build(batch_dir=batch, out_dir=tmp_path / "out", pool="rl",
          artifact_root=tmp_path / "rollouts", full_batch=True)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 24 and manifest["split_dir"] is None


def test_mismatched_batch_and_split_fails_loudly(batch_and_split, tmp_path):
    """split 和 batch 不是同一版本时，一条都对不上——要报错，不要产出空数据集。"""
    _, split_dir, _ = batch_and_split
    other = make_batch(tmp_path / "other", {"XXX|tool_call|-|id_given": 4})
    with pytest.raises(ValueError, match="一条都对不上"):
        build(batch_dir=other, out_dir=tmp_path / "out", pool="rl",
              artifact_root=tmp_path / "rollouts", split_dir=split_dir)
