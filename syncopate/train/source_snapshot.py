"""给一次 Modal 调用冻结源码，避免上传时本地编辑改变实验输入。"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def verify_orchestrator(current: Path, frozen: Path) -> str:
    """Modal 自动上传的入口文件，也必须等于镜像里的冻结副本。"""
    data = current.read_bytes()
    if data != frozen.read_bytes():
        raise RuntimeError("调度入口与冻结源码不相等；可能在镜像打包期间编辑了入口，本次不运行")
    return hashlib.sha256(data).hexdigest()


def source_digest(root: Path, directories: tuple[str, ...], files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    paths = [p for name in directories for p in (root / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}]
    paths += [root / name for name in files]
    for path in sorted(paths):
        relative, data = path.relative_to(root).as_posix(), path.read_bytes()
        digest.update(relative.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest()


def snapshot_source(root: Path, directories: tuple[str, ...], files: tuple[str, ...]):
    before = source_digest(root, directories, files)
    owner = tempfile.TemporaryDirectory(prefix="syncopate-source-")
    snapshot = Path(owner.name)
    try:
        for name in directories:
            shutil.copytree(root / name, snapshot / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        for name in files:
            shutil.copy2(root / name, snapshot / name)
        after = source_digest(snapshot, directories, files)
        if before != after or before != source_digest(root, directories, files):
            raise RuntimeError("复制源码时工作树发生变化；本次未启动远程计算，请重新提交调用")
        return owner, snapshot, after
    except BaseException:
        owner.cleanup()
        raise


def archive_source(root: Path, directories: tuple[str, ...], files: tuple[str, ...], *,
                   expected: str, output: Path) -> dict:
    """保全实际使用的快照，回读归档校验字节；已有归档不覆盖。"""
    paths = [p for name in directories for p in (root / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}]
    paths += [root / name for name in files]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("源码归档只能包含普通文件")
    if source_digest(root, directories, files) != expected:
        raise ValueError("待归档源码不等于已登记的运行快照")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with tarfile.open(output, "x:gz") as archive:
            for path in sorted(paths):
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    expected_names = {path.relative_to(root).as_posix() for path in paths}
    digest = hashlib.sha256()
    with tarfile.open(output, "r:gz") as archive:
        members = sorted(archive.getmembers(), key=lambda entry: Path(entry.name))
        if len(members) != len(paths) or {entry.name for entry in members} != expected_names:
            raise ValueError("归档的文件集合不等于运行快照")
        for member in members:
            if not member.isfile():
                raise ValueError("源码归档混入了非普通文件")
            data = archive.extractfile(member).read()
            digest.update(member.name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    if digest.hexdigest() != expected:
        raise ValueError("归档内容不等于运行快照；保留原文件，禁止覆盖")
    result = {"overlay_sha256": expected, "members": len(members), "bytes": output.stat().st_size,
              "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    record = output.with_suffix(output.suffix + ".json")
    if record.exists():
        if json.loads(record.read_text()) != result:
            raise ValueError("已有归档清单不同，禁止覆盖")
    else:
        with record.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return result
