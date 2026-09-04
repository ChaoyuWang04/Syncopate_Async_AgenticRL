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


OLD_NAME = re.compile(r"scripts/[A-Za-z0-9_./-]*(v8|v13|v14|v145|v15|u_build|u_exam|u_p[0-9])[A-Za-z0-9_./-]*\.(py|sh)")


def test_runbook_references_no_old_version_scripts():
    """09-05 Chaoyu：v16 路径上不许再出现旧版本名的脚本（此前 sft-data/exam 段调的是 u_build_v14_5 / v15_r5_exam_chain 等）。"""
    rb = ROOT / "scripts" / "v16_pipeline.sh"
    r = subprocess.run(["bash", str(rb), "--dry-run", "all"], cwd=ROOT, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **ENV})
    bad = sorted(set(m.group(0) for m in OLD_NAME.finditer(r.stdout)))
    assert not bad, f"runbook 还在调旧版本名的脚本：{bad}"
    # v16 建库两个脚本的代码行里不许出现旧物料/旧脚本名（探针 stale 判据的本机版）
    for f in ("scripts/v16_build_sft.py", "scripts/v16_multiturn.py", "scripts/v16_exam_chain.sh"):
        code = [l for l in (ROOT / f).read_text(encoding="utf-8").splitlines() if not l.strip().startswith("#")]
        hits = [l for l in code if re.search(r"v145_|v15_(cot_rows|l2l1_rows|ballast_replies|fam_rows|materials)|cand_v13r2|u_build_v14|pre_v16|v8_sft_epoch1", l)]
        assert not hits, f"{f} 代码行里还有旧物料/旧脚本名：{hits[:3]}"


def test_supply_check_imports_floors_and_runs():
    """09-04 实案：check_supply_vs_floors 抠源码正则，assert 收成 gate() 后静默崩，而 runbook 的 supply 段在 --dry-run 下只打印不执行 ⇒ 没人发现。
    ① 下限必须 import 同一份常量（不许 re.search 源码）② 数据在时真跑一遍且退出码 0。"""
    src = (ROOT / "scripts/check_supply_vs_floors.py").read_text(encoding="utf-8")
    assert "re.search" not in src and "L2_FLOOR" in src and "COT_FLOOR" in src
    from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR
    if not (ROOT / DEFAULT_BATCH_DIR / "manifest.json").exists() or not (ROOT / DEFAULT_SPLIT_DIR / "sft_cases.json").exists():
        pytest.skip(f"题库/切分不在本机（{DEFAULT_BATCH_DIR}）—— 跳过不是通过")
    r = subprocess.run([sys.executable, "scripts/check_supply_vs_floors.py"], cwd=ROOT, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **ENV}, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "🔴" not in r.stdout


def test_exam_judge_and_triage_take_this_runs_files():
    """09-05：判卷接多遍文件（此前单值，四遍传进来 argparse 报错）；三查必须显式给 --judged（此前默认读 v15-R3 旧判卷文件）。"""
    j = (ROOT / "scripts/v16_exam_judge.py").read_text(encoding="utf-8")
    assert 'add_argument("--context", nargs="+"' in j
    t = (ROOT / "scripts/v16_gate_triage.py").read_text(encoding="utf-8")
    assert 'add_argument("--judged", required=True' in t and "judged_v15r3c" not in t
    r = subprocess.run([sys.executable, "scripts/v16_gate_triage.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--judged" in r.stderr
    chain = (ROOT / "scripts/v16_exam_chain.sh").read_text(encoding="utf-8")
    assert chain.index("mkdir -p logs/v15_r5") < chain.index("logs/v15_r5/exam_vllm.log"), "日志目录必须在第一次写之前建好"
    assert "--judged \"logs/u_route/judged_${ARM}_r*_${EXAM}.jsonl\"" in chain
