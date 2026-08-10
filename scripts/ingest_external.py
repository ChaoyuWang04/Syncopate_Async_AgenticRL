"""把外部资料（Excel / 视觉标签 / 时令日历）转成运行时能查的结构化表。

    data/external/safety_lines/*.xlsx   运营每周更新
    data/external/creative_tags.json    离线 VLM 打标产物
    data/external/seasonal_calendar.json
                    ↓  （离线跑一次）
    data/external/ingested.json         运行时唯一入口

★ 为什么要有这一层，而不是让工具直接读 Excel

1. **rollout 不该碰 Excel**。openpyxl 解析一次几十毫秒，一次训练要跑几万条 rollout。
2. **每周更新只影响这一步**。运营换个 xlsx、重跑 ingest，下游一行不用改。
3. **可复现**。ingested.json 进版本管理，就能确切知道某次实验用的是哪一周的安全线。

同理，视觉标签是**离线**产物：在线 rollout 绝不跑 VLM，只查标签。

    python scripts/ingest_external.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
OUTPUT = EXTERNAL / "ingested.json"


def latest_safety_line_file() -> Path:
    """只取最新一周。按你的要求不做趋势分析，所以不需要时间轴。"""
    files = sorted((EXTERNAL / "safety_lines").glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("data/external/safety_lines/ 下没有 xlsx，"
                                "先跑 scripts/make_test_external_data.py")
    return files[-1]


def load_safety_lines(path: Path) -> dict[str, dict[str, Any]]:
    """产品 × 地域 -> 安全线。key 形如 "PUZ_QUEST|US"。"""
    import pandas as pd

    frame = pd.read_excel(path)
    table: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = f"{row['product_id']}|{row['region']}"
        table[key] = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return table


def load_creative_tags() -> dict[str, dict[str, Any]]:
    payload = json.loads((EXTERNAL / "creative_tags.json").read_text(encoding="utf-8"))
    return {item["creative_id"]: item for item in payload["items"]}


def load_calendar() -> list[dict[str, Any]]:
    return json.loads((EXTERNAL / "seasonal_calendar.json").read_text(encoding="utf-8"))["events"]


def main() -> int:
    source = latest_safety_line_file()
    safety = load_safety_lines(source)
    tags = load_creative_tags()
    events = load_calendar()

    OUTPUT.write_text(json.dumps({
        "version": "external_v1",
        "safety_line_source": source.name,
        "safety_lines": safety,
        "creative_catalog": tags,
        "seasonal_events": events,
    }, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    print(f"[OK] -> {OUTPUT.relative_to(ROOT)}")
    print(f"     安全线   {len(safety)} 条（来源 {source.name}）")
    print(f"     素材目录 {len(tags)} 条")
    print(f"     时令事件 {len(events)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
