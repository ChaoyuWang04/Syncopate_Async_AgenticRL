"""E31 T0.3 · 从 bf16 臂训练日志提取两引擎同轨迹 kl，固化为 kl_floor_bf16。

rollout_corr/kl 每个同步点比一次「vLLM 报的 logprob vs trainer 重算的 logprob」
（同一批 token）—— 这就是"两引擎同轨迹 kl"，不需要专门再跑一次。
输出 logs/e31/kl_floor_bf16.json（进 git），是 E31 第 1/2 步验收 kl ≤ 1.5×floor 的分母。

用法：.venv/bin/python scripts/infra/e31_kl_floor.py logs/smoke_newbox_0827_kvauto.log
守卫：日志若含 fp8 KV 痕迹（kl ~5e-3 带）直接拒绝 —— 分母拿错臂比没有分母更毒。
"""

from __future__ import annotations

import datetime
import json
import re
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "logs" / "e31" / "kl_floor_bf16.json"
PAT = re.compile(r"rollout_corr/kl[:=]([0-9.eE+-]+)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    log = Path(sys.argv[1])
    values = [float(m.group(1)) for m in PAT.finditer(log.read_text())]
    if len(values) < 3:
        print(f"❌ {log} 里只有 {len(values)} 个 rollout_corr/kl 同步点（<3），不够标定")
        return 1
    med = statistics.median(values)
    if not (1e-4 < med < 1e-3):
        print(f"❌ median {med:.2e} 不在 bf16 带（3.6–4.8e-4 附近）—— "
              "这像 fp8 KV 臂（~5e-3）或别的坏臂，拒绝写入")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "kl_values": values,
        "median": med,
        "max": max(values),
        "source_log": str(log),
        "date": datetime.date.today().isoformat(),
        "note": "bf16 两引擎同轨迹 kl 本底（E31 T0.3）；第 1/2 步验收分母 = 1.5×median",
    }, indent=2) + "\n")
    print(f"✅ kl_floor_bf16: median {med:.3e} · max {max(values):.3e} · "
          f"n={len(values)} → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
