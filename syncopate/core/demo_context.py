"""线上「当前投放任务」context 的**唯一**构造函数（decider 与训练建库共用）。

只列标识不列指标：指标必须靠工具查（"没有 observation 证明的事实不许写进结论"是训过的纪律，
塞进 context 等于替它把调查做了）。09-02 从 runtime/decider._demo_context 抽到 core：
训练侧 chat 行要用同一份（守则⑮ #4），而 runtime 包的导入链会触发治理表断言，数据构建不该背这个依赖。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

PLATFORM_STATE = "data/demo/platform_state.json"


def demo_context(path: str = PLATFORM_STATE) -> dict[str, Any]:
    f = pathlib.Path(path)
    if not f.is_file():
        return {"account_id": "ACC_DEMO"}
    state = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for cid, c in state.get("campaigns", {}).items():
        if cid.startswith("_"):
            continue
        rows.append(f"{cid}({c.get('name', '')}·产品 {c.get('product_id', '')}"
                    f"·地域 {c.get('region', '')})")
    return {"account_id": state.get("account_id", "ACC_DEMO"),
            "在投 campaign": "；".join(rows)}
