"""固定管线（2026-09-04 Chaoyu：每一段都必须能靠默认值健康跑完，不依赖谁临场敲参数）。
判据：① 各入口的默认值全部从 DATA_VERSION / model_paths 派生，不含旧版本字面量；② runbook 的每个 stage 在 --dry-run 下能列出命令，
引用的脚本/模块都存在；③ RL 启动器 candidate 档默认值 = 注册值。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = {"SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1"}


def _defaults(path: str) -> dict[str, str]:
    s = (ROOT / path).read_text(encoding="utf-8")
    return dict(re.findall(r'add_argument\("(--[a-z-]+)"[^)]*?default=([^,)]+)', s))


@pytest.mark.parametrize("path", ["syncopate/cli.py", "syncopate/train/sft.py", "syncopate/train/eval_local.py",
                                  "syncopate/train/launch_rl_v1.py", "syncopate/train/opd.py", "syncopate/train/entropy.py"])
def test_no_stale_version_literals_in_defaults(path):
    d = _defaults(path)
    bad = {k: v for k, v in d.items() if re.search(r"v(2|3|8|13|14|15)\b", v) and "TEST_TOKENIZER" not in v}
    assert not bad, f"{path} 默认值里还有旧版本字面量：{bad}"


def test_sft_and_eval_default_to_student_and_data_version():
    assert _defaults("syncopate/train/sft.py")["--model"] == "STUDENT_MODEL"
    assert "_SFT_DIR" in _defaults("syncopate/train/sft.py")["--train-file"]
    assert _defaults("syncopate/train/eval_local.py")["--model"] == "STUDENT_MODEL"


def test_runbook_dry_run_lists_every_stage():
    rb = ROOT / "scripts" / "v16_pipeline.sh"
    assert rb.exists(), "固定管线 runbook 不存在"
    r = subprocess.run(["bash", str(rb), "--dry-run", "all"], cwd=ROOT, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **ENV})
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    out = r.stdout
    for stage in ("cases", "menus", "split", "gates", "rl-data", "sft-data", "sft-train", "sft-eval", "sft-select", "merge",
                  "exam", "rl-train", "rl-adapter", "rl-eval", "opd-train", "opd-eval"):
        assert f"[stage {stage}]" in out, f"runbook 缺 stage {stage}"
    # 引用到的脚本/模块必须存在
    for m in re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)", out):
        assert (ROOT / m).exists(), f"runbook 引用了不存在的脚本 {m}"
    for m in re.findall(r"-m (syncopate(?:\.[a-z_0-9]+)+)", out):
        assert (ROOT / (m.replace(".", "/") + ".py")).exists() or (ROOT / m.replace(".", "/") / "__main__.py").exists(), f"模块 {m} 不存在"


def test_rl_candidate_profile_registered_values():
    from syncopate.train import launch_rl_v1 as L
    s = (ROOT / "syncopate/train/launch_rl_v1.py").read_text(encoding="utf-8")
    assert 'steps=400' in s and 'save_freq=25' in s and 'rollout_n=8' in s, "candidate 档注册值（400/25/8）被改动"
    assert '"candidate"' in s and '"smoke"' in s
