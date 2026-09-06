"""云端单写者、源码身份和分臂缓存。标准库实现，方便先在 Mac 测试。"""
from __future__ import annotations

import json
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def validate_run_id(value: str) -> str:
    """实验目录名必须从字母或数字开始；拒绝 .、.. 和路径/命令字符。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value):
        raise ValueError("run-id 须以字母或数字开头，只含字母、数字、点、下划线、短横线，最长 96 字符")
    return value


@contextmanager
def writer_claims(registry, keys: list[str], owner: str):
    """registry.put(skip_if_exists=True) 必须是服务端原子操作，不能用 Volume 文件锁。"""
    acquired = []
    try:
        for key in sorted(set(keys)):
            if not registry.put(key, owner, skip_if_exists=True):
                raise RuntimeError(f"已有写者占用 {key}，本次不运行；若是中断遗留，先核对容器已退出")
            acquired.append(key)
        yield
    finally:
        for key in reversed(acquired):
            if registry.get(key) == owner:
                registry.pop(key)


def bind_source(directory: Path, source: dict) -> None:
    path = directory / "source.json"
    if path.exists():
        if json.loads(path.read_text()) != source:
            raise RuntimeError("同一 run_id 的源码改变了；必须使用新的 run_id")
        return
    if (directory / "manifest.json").exists():
        raise RuntimeError("旧账本没有固定源码身份，不能接着写；使用新的 run_id")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n")


def isolated_cache(volume: Path, invocation: str) -> dict[str, str]:
    """旧缓存只读种子；每个容器复制一份，绝不并发写同一编译缓存。"""
    root = volume / "_cache_runs" / invocation
    root.mkdir(parents=True, exist_ok=False)
    env = {}
    for name, variable in (("flashinfer_cache", "FLASHINFER_WORKSPACE_BASE"),
                           ("vllm_cache", "VLLM_CACHE_ROOT")):
        target = root / name
        seed = volume / name
        if seed.is_dir():
            shutil.copytree(seed, target)
        else:
            target.mkdir()
        env[variable] = str(target)
    for name, variable in (("triton", "TRITON_CACHE_DIR"),
                           ("torch_extensions", "TORCH_EXTENSIONS_DIR"),
                           ("cuda", "CUDA_CACHE_PATH")):
        target = root / name
        target.mkdir()
        env[variable] = str(target)
    return env


def prepared_cache(directory: Path, source: dict) -> dict[str, str]:
    """GPU 只读取 CPU 已完成的准备结果，不在 GPU 上复制缓存。"""
    data = json.loads((directory / "cache_ready.json").read_text())
    if data["source"] != source:
        raise RuntimeError("缓存准备与运行源码不相等")
    env = data["env"]
    cache_writer_keys(env)
    if not env or any(not Path(value).is_dir() for value in env.values()):
        raise RuntimeError("已准备的缓存目录缺失")
    return env


def cache_writer_keys(env: dict[str, str]) -> list[str]:
    names = {"FLASHINFER_WORKSPACE_BASE": "flashinfer_cache", "VLLM_CACHE_ROOT": "vllm_cache",
             "TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch_extensions",
             "CUDA_CACHE_PATH": "cuda"}
    if set(env) != set(names):
        raise RuntimeError("缓存变量清单不完整")
    paths = {key: Path(value).resolve() for key, value in env.items()}
    roots = {value.parent for value in paths.values()}
    if (len(roots) != 1 or next(iter(roots)).parent.name != "_cache_runs" or
            any(paths[key].name != name for key, name in names.items())):
        raise RuntimeError("缓存必须全部位于同一个独立 _cache_runs 目录")
    return ["cache:" + str(next(iter(roots)))]


def audit_cache_path(volume: Path, relative_path: str) -> Path:
    """挂载点可以是软链接；边界两侧都解析，但拒绝借软链接越出审计目录。"""
    requested = Path(relative_path)
    if not relative_path or requested.is_absolute() or ".." in requested.parts:
        raise ValueError("缓存来源必须是 Volume 内不含 .. 的相对路径")
    allowed = (volume / "_audit").resolve(strict=True)
    candidate = (volume / requested).resolve(strict=True)
    if candidate == allowed or not candidate.is_relative_to(allowed) or not candidate.is_dir():
        raise ValueError("只能显式复用 Volume 审计目录中的完成缓存")
    return candidate


def reuse_prepared_cache(previous: Path, directory: Path, source: dict) -> dict:
    """显式借用已完成缓存；保留旧来源，调用方另占用缓存目录的写者键。"""
    bind_source(directory, source)
    marker = directory / "cache_ready.json"
    if marker.exists():
        prepared_cache(directory, source)
        return json.loads(marker.read_text())
    old = json.loads((previous / "cache_ready.json").read_text())
    env = prepared_cache(previous, old["source"])
    result = {"source": source, "env": env, "prepare_seconds": 0,
              "reused_from": {"directory": str(previous), "source": old["source"]}}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(marker)
    return result


def prepare_cache(volume: Path, directory: Path, source: dict) -> dict:
    """只由 CPU 前置调用；全部复制结束后才写完成标记。中断不算完成。"""
    bind_source(directory, source)
    marker = directory / "cache_ready.json"
    if marker.exists():
        prepared_cache(directory, source)
        return json.loads(marker.read_text())
    started = time.monotonic()
    env = isolated_cache(volume, uuid4().hex)
    result = {"source": source, "env": env, "prepare_seconds": round(time.monotonic() - started, 3)}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(marker)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检查云端运行目录名")
    parser.add_argument("--validate-run-id", required=True, type=validate_run_id)
    parser.parse_args()
