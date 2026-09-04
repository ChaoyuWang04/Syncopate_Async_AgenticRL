"""SFT 选点工具：**匹配错了比找不到更糟**。

★ 起因（2026-08-18，写这个工具时当场撞的）

第一版用文件名的模糊包含去找审计（`_epoch1.` in name），
给 `checkpoints/sft/v13/epoch1` 匹配到了 **v3 时代**的 `M1_ctrl_epoch1.json`，
并打出「决策位熵 0.483 / 有梯度 44」—— 一组**看起来完全合理、其实是另一个模型**的数字。

⇒ 这正是本项目反复栽的那个形状（`19 §2.1` 第三层，最贵的一层）：
  **判据看起来在量，量的却是另一件事，而且它产出了一个具体的数字。**
⇒ 修法：只认 label 里出现该候选的相对路径；认不出就返回 None。
  **宁可报"没有"，也不要猜。**
"""

from __future__ import annotations

import json

from syncopate.pipeline.split import DATA_VERSION
from pathlib import Path

import importlib.util
import sys

import pytest


def _load(name: str, path: Path):
    """按**文件路径**导入 scripts/ 下的脚本。

    ⚠️ 不用 `importorskip("scripts.xxx")` —— `scripts/` 不是包，那样会**静默 skip**，
    而「跳过不是通过」是这个项目记过的一条（45 条 runtime 测试曾经就这么静默过去）。
    验收口径是 **0 skipped**，所以这里宁可 import 失败也不要 skip。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


select = _load("select_sft_ckpt",
               Path(__file__).resolve().parents[2] / "scripts" / "select_sft_ckpt.py")


def _write(tmp: Path, name: str, label: str, extra: dict) -> None:
    (tmp / "_audit").mkdir(parents=True, exist_ok=True)
    (tmp / "_audit" / name).write_text(
        # 09-05：没写 data_version 的审计一律不认（此前会被接受 ⇒ 旧版本审计可被选点）；夹具默认带当前版本
        json.dumps({"label": label, "data_version": DATA_VERSION, **extra}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(select, "ROOT", tmp_path)
    return tmp_path


def test_does_not_match_a_different_run_with_the_same_epoch_name(fake_root):
    """★ 核心：同名 epoch 但**不同批次**的审计，绝不能被认成这个候选的。"""
    _write(fake_root, "M1_ctrl_epoch1.json",
           "models/Qwen3-4B + checkpoints/sft/v3_ctrl/epoch1",
           {"decision_mean_entropy": 0.483})
    cand = fake_root / "checkpoints/sft/v13/epoch1"
    cand.mkdir(parents=True)
    assert select._metric("entropy", cand) is None, "撞到了别的批次的审计"


def test_matches_the_right_one_by_label_path(fake_root):
    _write(fake_root, "v13_entropy_e1.json",
           "models/Qwen3-4B + checkpoints/sft/v13/epoch1",
           {"decision_mean_entropy": 0.158})
    cand = fake_root / "checkpoints/sft/v13/epoch1"
    cand.mkdir(parents=True)
    got = select._metric("entropy", cand)
    assert got is not None and got["decision_mean_entropy"] == pytest.approx(0.158)


def test_entropy_and_eval_audits_are_not_confused(fake_root):
    """熵审计和评测审计 label 可能只差一个 `[vllm]` —— 必须靠字段区分，不是靠名字。"""
    _write(fake_root, "v13_entropy_e1.json",
           "models/Qwen3-4B + checkpoints/sft/v13/epoch1", {"decision_mean_entropy": 0.158})
    _write(fake_root, "v13_sft_e1.json",
           "models/Qwen3-4B + checkpoints/sft/v13/epoch1 [vllm]",
           {"rows": [{"reward": 0.9, "reward_std": 0.2}]})
    cand = fake_root / "checkpoints/sft/v13/epoch1"
    cand.mkdir(parents=True)
    assert "decision_mean_entropy" in select._metric("entropy", cand)
    assert "rows" in select._metric("eval", cand)


def test_prune_refuses_without_keep(fake_root, capsys):
    """⚠️ `--prune` 不配 `--keep` = 不知道选中谁就删 ⇒ 必须硬失败。"""
    out = fake_root / "checkpoints/sft/v14"
    (out / "sel_f0.25").mkdir(parents=True)
    (out / "epoch1").mkdir(parents=True)
    rc = select.main([str(out), "--prune"])
    assert rc == 1
    assert (out / "sel_f0.25").exists(), "没指定 keep 就不该删任何东西"


def test_prune_keeps_epochs_and_the_chosen_one(fake_root):
    """只删未选中的 `sel_f*`；`epoch*` 一律保留（它们不是临时产物）。"""
    out = fake_root / "checkpoints/sft/v14"
    for n in ("sel_f0.25", "sel_f0.5", "epoch1"):
        (out / n).mkdir(parents=True)
        (out / n / "adapter_model.safetensors").write_bytes(b"x" * 1024)
    assert select.main([str(out), "--keep", "sel_f0.25", "--prune"]) == 0
    assert (out / "sel_f0.25").exists() and (out / "epoch1").exists()
    assert not (out / "sel_f0.5").exists()
