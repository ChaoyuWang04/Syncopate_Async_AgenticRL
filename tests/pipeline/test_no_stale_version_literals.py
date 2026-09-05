"""裁定⑩：数据版本只在 `split.DATA_VERSION` 一处；活跃代码里不许再写 data/(batches|splits|sft|rl)/v1[0-5] 字面量。

负向认证：往任一活跃文件塞一行 `x = "data/batches/v13"` 本测试必红。
LEGACY 白名单 = 只为重放历史审计而保留的脚本（换版本不需要它们跟着动）；新文件不许进白名单。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAT = re.compile(r"data/(batches|splits|sft|rl)/v1[0-5]\b")
LEGACY = {
    "syncopate/train/quantize_nvfp4.py", "scripts/serving/runtime_loadtest.py",
    "scripts/infra/eval_parallel.sh", "syncopate/authoring/calibrate_retrieval.py",
    "tests/pipeline/test_data_version_contract.py",   # 字面量对是 assert_same_data_version 的输入样例
    "syncopate/pipeline/split.py",                     # docstring 举例
    "tests/pipeline/test_no_stale_version_literals.py",
}
# infra 的 E 报告复现脚本（run_*.sh / 各阶段 chain）：改了就复现不了历史读数，整类按 legacy 处理
LEGACY_GLOBS = ("scripts/infra/run_*.sh",)
COMMENT_PREFIXES = ("#", '"""', "'''")


def _code_lines(path: Path):
    """跳过注释与三引号 docstring 内的行（用法示例都写在 docstring 里）。"""
    in_doc = False
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        s = line.strip()
        quotes = s.count('"""') + s.count("'''")
        if in_doc:
            if quotes % 2 == 1:
                in_doc = False
            continue
        if quotes % 2 == 1:
            in_doc = True
            continue
        if s.startswith(COMMENT_PREFIXES):
            continue
        yield i, line


def test_no_stale_data_version_literals_in_active_code():
    offenders = []
    for sub in ("syncopate", "scripts", "tests"):
        for p in (ROOT / sub).rglob("*"):
            if p.suffix not in (".py", ".sh") or not p.is_file():
                continue
            rel = str(p.relative_to(ROOT))
            if rel.startswith("scripts/archive/"):
                continue
            if rel in LEGACY or any(p.relative_to(ROOT).match(g) for g in LEGACY_GLOBS):
                continue
            for i, line in _code_lines(p):
                # 用法示例行（docstring 里的 `python scripts/… --batch data/batches/v13`）不算代码
                if PAT.search(line) and "usage" not in line.lower() and "python " not in line and "SYNCOPATE_CONTRACT=" not in line:
                    offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, "旧数据版本字面量（应改用 split.DEFAULT_*）：\n" + "\n".join(offenders)
