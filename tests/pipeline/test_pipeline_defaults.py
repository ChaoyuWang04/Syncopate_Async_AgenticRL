"""固定管线（2026-09-04 Chaoyu：每一段都必须能靠默认值健康跑完，不依赖谁临场敲参数）。
判据：① 各入口的默认值全部从 DATA_VERSION / model_paths 派生，不含旧版本字面量；② runbook 的每个 stage 在 --dry-run 下能列出命令，
引用的脚本/模块都存在；③ RL 启动器 candidate 档默认值 = 注册值。"""
from __future__ import annotations

import ast
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
    assert "🔴" not in out, "dry-run 自己不生成上游产物，不能把预期缺失伪装成红灯"
    for stage in ("cases", "menus", "split", "gates", "supply", "rl-data", "teacher", "sft-data", "teacher-stop", "sft-train", "sft-eval", "sft-select", "merge",
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


def test_default_profile_is_smoke_and_candidate_is_explicit():
    rb = ROOT / "scripts/v16_pipeline.sh"
    base = ["bash", str(rb), "--dry-run"]
    smoke = subprocess.run([*base, "sft-train"], cwd=ROOT, capture_output=True, text=True,
                           env={**dict(__import__("os").environ), **ENV})
    assert smoke.returncode == 0, smoke.stderr
    assert "profile=smoke, gate=observe" in smoke.stdout
    assert "--out checkpoints/sft/v16_smoke" in smoke.stdout
    assert "--gpus 1 --effective-batch 8" in smoke.stdout

    candidate = subprocess.run([*base, "--profile", "candidate", "sft-train"], cwd=ROOT,
                               capture_output=True, text=True,
                               env={**dict(__import__("os").environ), **ENV})
    assert candidate.returncode == 0, candidate.stderr
    assert "profile=candidate, gate=strict" in candidate.stdout
    assert "--out checkpoints/sft/v16 " in candidate.stdout


def test_smoke_chain_has_no_bare_model_or_cross_profile_fallback():
    rb = ROOT / "scripts/v16_pipeline.sh"
    env = {**dict(__import__("os").environ), **ENV}
    outputs = {}
    for stage in ("sft-eval", "sft-select", "merge", "rl-train", "rl-eval", "opd-train", "opd-eval"):
        run = subprocess.run(["bash", str(rb), "--dry-run", stage], cwd=ROOT,
                             capture_output=True, text=True, env=env)
        assert run.returncode == 0, run.stderr
        outputs[stage] = run.stdout
    assert "checkpoints/sft/v16_smoke" in outputs["sft-eval"]
    assert "checkpoints/sft/v16_smoke" in outputs["sft-select"]
    assert "models/Qwen3.6-35B-A3B-sft-v16_smoke" in outputs["merge"]
    assert "--model models/Qwen3.6-35B-A3B-sft-v16_smoke" in outputs["rl-train"]
    assert "--adapter models/adapters/rl_v16_smoke/lora_adapter" in outputs["rl-eval"]
    assert "--base models/Qwen3.6-35B-A3B-sft-v16_smoke" in outputs["opd-train"]
    assert "--adapter models/adapters/rl_v16_smoke/lora_adapter" in outputs["opd-train"]
    assert "--max-steps 1 --max-attempts 8 --batch 2" in outputs["opd-train"]
    assert "completion.json" in outputs["opd-eval"]
    source = rb.read_text(encoding="utf-8")
    assert "opd_base()" not in source and "opd_adapter()" not in source
    assert "-m syncopate.train.sft_run_gate" in source
    assert "-m syncopate.train.rl_run_gate" in source
    assert "-m syncopate.train.opd_run_gate" in source
    assert "--audit-dir _audit/v16/runs/smoke_manual" in outputs["sft-select"]
    assert "--limit 8 --samples-per-case 2" in outputs["rl-eval"]
    assert "--limit 8 --samples-per-case 2" in outputs["opd-eval"]


def test_opd_uses_current_v15_prompt_and_mask_components():
    trainer = (ROOT / "syncopate/train/opd.py").read_text(encoding="utf-8")
    renderer = (ROOT / "syncopate/train/opd_render.py").read_text(encoding="utf-8")
    assert "REGISTRY.menu(None)" in trainer
    assert "load_system_prompt()" in renderer
    assert 'text_value_keys={"reply"}' not in renderer
    assert "return 0 if completed else 3" in trainer


def test_generated_v15_parsing_is_told_about_qwen_prompt_side_think_open():
    rollout = (ROOT / "syncopate/train/rollout_loop.py").read_text(encoding="utf-8")
    decider = (ROOT / "syncopate/runtime/decider.py").read_text(encoding="utf-8")
    assert "parse_step_v15(text, implicit_think_open=ENABLE_THINKING)" in rollout
    assert 'implicit_think_open=bool(_kw.get("enable_thinking"))' in decider


def test_train_all_uses_existing_data_and_prints_every_training_command():
    rb = ROOT / "scripts/v16_pipeline.sh"
    run = subprocess.run(
        ["bash", str(rb), "--dry-run", "--run-id", "shape", "train-all"],
        cwd=ROOT, capture_output=True, text=True,
        env={**dict(__import__("os").environ), **ENV},
    )
    assert run.returncode == 0, run.stderr
    assert "[stage cases]" not in run.stdout and "[stage sft-data]" not in run.stdout
    for stage in ("sft-train", "sft-eval", "sft-select", "merge", "exam", "rl-train",
                  "rl-adapter", "rl-eval", "opd-train", "opd-eval"):
        assert f"[stage {stage}]" in run.stdout
    assert "-m syncopate.train.entropy" in run.stdout
    assert "-m syncopate.train.eval_local" in run.stdout


OLD_NAME = re.compile(r"scripts/[A-Za-z0-9_./-]*(v8|v13|v14|v145|v15|u_build|u_exam|u_p[0-9])[A-Za-z0-9_./-]*\.(py|sh)")


def test_runbook_references_no_old_version_scripts():
    """09-05 Chaoyu：v16 路径上不许再出现旧版本名的脚本（此前 sft-data/exam 段调的是 u_build_v14_5 / v15_r5_exam_chain 等）。"""
    rb = ROOT / "scripts" / "v16_pipeline.sh"
    r = subprocess.run(["bash", str(rb), "--dry-run", "all"], cwd=ROOT, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **ENV})
    bad = sorted(set(m.group(0) for m in OLD_NAME.finditer(r.stdout)))
    assert not bad, f"runbook 还在调旧版本名的脚本：{bad}"
    # v16 建库两个脚本的代码行里不许出现旧物料/旧脚本名（探针 stale 判据的本机版）
    for f in ("syncopate/pipeline/build_sft.py", "syncopate/pipeline/multiturn.py", "scripts/v16/exam_chain.sh"):
        code = [l for l in (ROOT / f).read_text(encoding="utf-8").splitlines() if not l.strip().startswith("#")]
        hits = [l for l in code if re.search(r"v145_|v15_(cot_rows|l2l1_rows|ballast_replies|fam_rows|materials)|cand_v13r2|u_build_v14|pre_v16|v8_sft_epoch1", l)]
        assert not hits, f"{f} 代码行里还有旧物料/旧脚本名：{hits[:3]}"


def test_scripts_layout_keeps_reusable_components_in_package():
    """scripts 只留编排/探针；可复用 Python 组件归 syncopate。"""
    scripts = ROOT / "scripts"
    top_level_code = sorted(
        p.name for p in scripts.iterdir()
        if p.is_file() and p.suffix in {".py", ".sh", ".cu"}
    )
    assert top_level_code == ["v16_pipeline.sh"]

    active_dirs = {p.name for p in scripts.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert active_dirs == {"archive", "infra", "serving", "tools", "v16"}

    for path in (ROOT / "syncopate").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:from|import)\s+scripts(?:\.|\s|$)", source), \
            f"{path.relative_to(ROOT)} 反向 import 了 scripts"
        assert "sys.path.insert(0, \"scripts\")" not in source


def test_runbook_invokes_python_components_as_modules():
    source = (ROOT / "scripts/v16_pipeline.sh").read_text(encoding="utf-8")
    assert not re.search(r"\$PY\s+scripts/[^\s]+\.py", source)
    assert "-m syncopate.pipeline.build_sft" in source
    assert "-m syncopate.train.select_sft_ckpt" in source
    assert "-m syncopate.evaluation.exam_run" in (
        ROOT / "scripts/v16/exam_chain.sh"
    ).read_text(encoding="utf-8")


def test_supply_check_imports_floors_and_runs():
    """09-04 实案：check_supply_vs_floors 抠源码正则，assert 收成 gate() 后静默崩，而 runbook 的 supply 段在 --dry-run 下只打印不执行 ⇒ 没人发现。
    ① 下限必须 import 同一份常量（不许 re.search 源码）② 数据在时真跑一遍且退出码 0。"""
    src = (ROOT / "syncopate/pipeline/supply_gate.py").read_text(encoding="utf-8")
    assert "re.search" not in src and "L2_FLOOR" in src and "COT_FLOOR" in src
    from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR
    if not (ROOT / DEFAULT_BATCH_DIR / "manifest.json").exists() or not (ROOT / DEFAULT_SPLIT_DIR / "sft_cases.json").exists():
        pytest.skip(f"题库/切分不在本机（{DEFAULT_BATCH_DIR}）—— 跳过不是通过")
    r = subprocess.run([sys.executable, "-m", "syncopate.pipeline.supply_gate"], cwd=ROOT, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **ENV}, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "🔴" not in r.stdout


def test_exam_judge_and_gate_take_this_runs_files():
    """判卷和门禁只读本轮文件；旧 R5/R6/R7 三查表不能再决定 v16 pipeline 状态。"""
    j = (ROOT / "syncopate/evaluation/exam_judge.py").read_text(encoding="utf-8")
    assert 'add_argument("--context", nargs="+"' in j
    t = (ROOT / "syncopate/evaluation/gate_triage.py").read_text(encoding="utf-8")
    assert 'add_argument("--judged", required=True' in t and "judged_v15r3c" not in t
    r = subprocess.run([sys.executable, "-m", "syncopate.evaluation.gate_triage"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--judged" in r.stderr
    chain = (ROOT / "scripts/v16/exam_chain.sh").read_text(encoding="utf-8")
    assert 'mkdir -p "$AUD" logs/u_route' in chain
    assert "__SYNCOPATE_EXAM_CONSTANTS__" in chain and "sed -n" in chain
    assert '[[ "${MAX_MODEL_LEN:-}" =~ ^[0-9]+$ ]]' in chain
    assert "-m syncopate.evaluation.exam_run_gate" in chain
    assert "-m syncopate.evaluation.gate_triage" not in chain
    assert '--raw "$AUD/run_${ARM}_r*_${EXAM}.jsonl"' in chain
    assert '--judged "$AUD/judged_${ARM}_r*_${EXAM}.jsonl"' in chain
    assert 'cp "$run_file" "$AUD/$(basename "$run_file")"' in chain
    assert 'curl -sf http://127.0.0.1:8100/v1/models > "$AUD/models.json"' in chain
    assert "EXAM_PROFILE" in chain
    assert "EXAM_GATE_MODE" in chain and "exit 10" in chain and "exit 20" in chain
    assert "logs/v15_r5" not in chain
    assert "pkill -f" not in chain and "nvidia-smi --query-compute-apps" not in chain


def test_offline_sft_downloads_cache_identity_before_row_caches():
    """行缓存和切分身份 sidecar 必须一起搬；漏 sidecar 会让离线重建先删缓存再索要教师。"""
    source = (ROOT / "scripts/v16_pipeline.sh").read_text(encoding="utf-8")
    line = next(x for x in source.splitlines() if "for f in v16_cache_split" in x)
    assert line.index("v16_cache_split") < line.index("v16_l2l1_rows")
    assert line.index("v16_cache_split") < line.index("v16_fam_rows")
    assert line.index("v16_cache_split") < line.index("v16_cot_rows")
    assert "modal volume get" in line and "|| return 1" in line


def test_modal_pipeline_only_delegates_to_the_fixed_runbook():
    source = (ROOT / "modal_app/stack_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "p_pipeline")
    body = ast.get_source_segment(source, fn) or ""
    assert "bash scripts/v16_pipeline.sh" in body
    assert "launch_rl_v1" not in body and "syncopate.train.sft" not in body and "syncopate.train.opd" not in body
    assert 'profile: str = "smoke"' in body
    assert 'stage: str = "train-all"' in body
    assert 'resume: bool = False' in body and '" --resume" if resume' in body
    assert 'manifest.get("all_passed") is True' in body


def test_modal_uploads_only_the_current_source_overlay():
    source = (ROOT / "modal_app/stack_probe.py").read_text(encoding="utf-8")
    assert 'OVERLAY_DIRS = ("syncopate", "scripts", "configs", "tests", "docs", "modal_app")' in source
    assert 'OVERLAY_FILES = ("pyproject.toml", "alembic.ini")' in source
    assert '.add_local_dir(LOCAL_ROOT,' not in source, "不能把整个仓库（数据/模型/.git）上传到镜像"
    assert "local_overlay_sha256" in source and "remote_git_head" in source


def test_modal_children_are_stopped_by_their_recorded_process_groups():
    """并发实验不能按命令名扫杀；每个 Popen 都建立自己的进程组。"""
    source = (ROOT / "modal_app/stack_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"]
    assert popens
    for call in popens:
        kwargs = {x.arg: x.value for x in call.keywords if x.arg}
        assert isinstance(kwargs.get("start_new_session"), ast.Constant) and kwargs["start_new_session"].value is True
    executable_strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any("pkill -f" in s for s in executable_strings if not s.strip().startswith("精确停止"))
    assert "os.killpg(proc.pid, signal.SIGTERM)" in source
