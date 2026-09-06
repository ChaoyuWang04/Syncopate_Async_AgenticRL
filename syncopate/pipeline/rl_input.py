"""给独立 RL 对照显式复用一个已完成的合并 SFT，不伪造本轮 SFT 记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

from syncopate.pipeline.cloud_execution import validate_run_id
from syncopate.pipeline.model_input import model_files


def validate_request(stage: str, profile: str, run_id: str, input_run: str) -> None:
    for value in (run_id, input_run):
        validate_run_id(value)
    if run_id == input_run or profile != "smoke" or stage not in {
        "rl-train", "rl-adapter", "rl-eval", "opd-train", "opd-eval",
    }:
        raise ValueError("外部 SFT 输入只能用于独立 smoke 对照，不能冒充全链或 candidate")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input(root: Path, input_run: str, run_id: str) -> dict:
    from syncopate.core.model_paths import STUDENT_MODEL
    from syncopate.pipeline.split import DATA_VERSION

    validate_run_id(input_run)
    validate_run_id(run_id)
    if input_run == run_id:
        raise ValueError("独立对照的 input-run 必须与输出 run-id 不同")
    manifest = root / "_audit" / DATA_VERSION / "runs" / input_run / "manifest.json"
    data = json.loads(manifest.read_text())
    if (data.get("run_id") != input_run or data.get("profile") != "smoke" or
            data.get("stages", {}).get("merge", {}).get("status") != "pass"):
        raise ValueError("上游不是指定的已通过 merge 的 smoke run")
    model = root / "models" / f"{Path(STUDENT_MODEL).name}-sft-{DATA_VERSION}_smoke_{input_run}"
    if not model.is_dir() or (model / "lora_adapter").exists():
        raise ValueError("上游缺少完整合并模型，不能使用底座或 adapter fallback")
    files = model_files(model, indexed_only=True)
    return {
        "schema_version": 1, "kind": "external_sft_for_rl_diagnostic",
        "run_id": run_id, "input_run": input_run,
        "input_manifest_sha256": file_sha256(manifest),
        "model": str(model.relative_to(root)),
        "files": [{"name": str(p.relative_to(model)), "bytes": p.stat().st_size}
                  for p in files],
    }


def bind_input(root: Path, input_run: str, run_id: str, *, hash_weights: bool = False) -> dict:
    from syncopate.pipeline.split import DATA_VERSION

    record = resolve_input(root, input_run, run_id)
    target = root / "_audit" / DATA_VERSION / "runs" / run_id / "rl_input.json"
    if target.exists():
        existing = json.loads(target.read_text())
        identity = {key: existing.get(key) for key in record}
        if identity != record:
            raise ValueError("本轮上游身份改变，必须新建 run-id")
        hashes = existing.get('content_sha256')
        if (not isinstance(hashes, dict) or set(hashes) != {item['name'] for item in record['files']}
                or any(not isinstance(sha, str) or re.fullmatch(r'[0-9a-f]{64}', sha) is None for sha in hashes.values())):
            raise ValueError("上游权重尚未经过完整 CPU 内容校验")
        if hash_weights:
            hashes = {entry["name"]: file_sha256(root / record["model"] / entry["name"])
                      for entry in record["files"]}
            if hashes != existing["content_sha256"]:
                raise ValueError("上游权重内容改变，必须新建 run-id")
        return existing
    if not hash_weights:
        raise ValueError("先在 CPU 绑定上游权重内容，不能占用 GPU 做准备")
    record["content_sha256"] = {
        entry["name"]: file_sha256(root / record["model"] / entry["name"])
        for entry in record["files"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    # CLI 的 stdout 是机器接口：Shell 会把整段捕获成路径。保留导入提示，但移到 stderr。
    with redirect_stdout(sys.stderr):
        record = bind_input(Path.cwd(), args.input_run, args.run_id, hash_weights=args.hash_weights)
    print(record["model"] if args.model_only else json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
