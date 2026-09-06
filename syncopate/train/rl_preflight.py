"""RL 的 CPU 前置：当前依赖回归、冻结输入、Hydra 实际配置；不生成或训练。"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from syncopate.pipeline.rl_input import bind_input, file_sha256


TESTS = (
    "tests/train/test_generation_observer.py", "tests/train/test_generation_probe.py",
    "tests/train/test_rollout_loop.py", "tests/train/test_v16_launch_profiles.py",
    "tests/train/test_rl_evidence.py", "tests/train/test_rl_run_gate.py",
    "tests/pipeline/test_rl_input.py", "tests/pipeline/test_pipeline_defaults.py",
    "tests/pipeline/test_cloud_execution.py", "tests/pipeline/test_process_guard.py",
    "tests/train/test_source_snapshot.py", "tests/train/test_vllm_process.py",
    "tests/train/test_gpu_telemetry.py",
    "tests/train/test_policy_observer.py", "tests/train/test_policy_observer_wiring.py",
    "tests/train/test_policy_evidence.py", "tests/train/test_policy_roundtrip.py",
)


def run_check(command: list[str], log: Path) -> dict:
    with log.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    return {"returncode": result.returncode, "log": str(log), "tail": log.read_text()[-2200:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"health_ok": False, "checks": {}}
    try:
        from syncopate.core.model_paths import build_tokenizer_path
        os.environ.setdefault("SYNCOPATE_TEST_TOKENIZER", build_tokenizer_path())
        result["tests"] = run_check([sys.executable, "-m", "pytest", *TESTS, "-q", "-rs"],
                                    args.out.parent / "tests.log")
        result["checks"]["tests"] = result["tests"]["returncode"] == 0
        if not result["checks"]["tests"]:
            raise ValueError("CPU 测试失败，不分配 GPU")
        record = bind_input(root, args.input_run, args.run_id, hash_weights=True)
        result["input"] = record
        result["runbook_inputs"] = run_check([
            "bash", "scripts/v16_pipeline.sh", "--check-inputs", "--profile", "smoke",
            "--gate-mode", "observe", "--run-id", args.run_id, "--rl-input-run", args.input_run,
            "rl-train"], args.out.parent / "runbook_inputs.log")
        result["checks"]["runbook_inputs"] = result["runbook_inputs"]["returncode"] == 0
        if not result["checks"]["runbook_inputs"]:
            raise ValueError("固定 runbook 输入检查失败，不分配 GPU")
        from syncopate.pipeline.split import DEFAULT_RL_DIR, DEFAULT_SPLIT_DIR, DATA_VERSION
        paths = [root / DEFAULT_RL_DIR / f"{name}.parquet" for name in ("train", "val")]
        paths += [root / DEFAULT_SPLIT_DIR / "rl_cases.json"]
        result["data_sha256"] = {str(p.relative_to(root)): file_sha256(p) for p in paths}
        result["isolation"] = run_check([sys.executable, "-m", "syncopate.pipeline.split_isolation",
                                         *map(str, paths[:2]), "--pool", "rl"], args.out.parent / "isolation.log")
        result["checks"]["isolation"] = result["isolation"]["returncode"] == 0
        result["config"] = run_check([sys.executable, "-m", "syncopate.train.launch_rl_v1",
                                       "--cfg-only", "--profile", "smoke", "--logger", "console",
                                       "--model", record["model"], "--save-path",
                                       f"checkpoints/grpo/{DATA_VERSION}_smoke_{args.run_id}"],
                                      args.out.parent / "resolved_config.log")
        result["checks"]["config"] = result["config"]["returncode"] == 0
        # 读取本次真实安装的源码，不用旧 FSDP 经验猜当前 optimizer 钩子。
        engine_path = Path(importlib.metadata.distribution("verl").locate_file(
            "verl/workers/engine/fsdp/transformer_impl.py"))
        source = engine_path.read_text()
        tree = ast.parse(source)
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngine")
        method = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "optimizer_step")
        result["optimizer_source"] = ast.get_source_segment(source, method)
        result["optimizer_source_sha256"] = file_sha256(engine_path)
        result["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "verl", "vllm", "transformers", "peft")}
        result["health_ok"] = all(result["checks"].values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input"}, ensure_ascii=False))
    return 0 if result["health_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
