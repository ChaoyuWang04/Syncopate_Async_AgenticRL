#!/usr/bin/env python
"""B-7 · 灰测读数：**线上没有 gold，尺子只能是别的东西**。

    python scripts/canary_readout.py --org <org_id>     # 一个租户
    python scripts/canary_readout.py                    # 全部

★★★ 为什么不能用离线那套任务尺子

离线评测靠的是 **gold path + verifier**：每道题有标准答案，能算 reward、能配对比较。
**线上一条 gold 都没有。** 用户不会告诉你"正确答案是什么"。

⇒ 线上真正能拿到的信号只有四个，而且它们的**信息量差得很远**：

```
① 人工修正率   approval_cases.modified_params 非空的比例
               ★★ 这是唯一「人主动给出正确答案」的信号 —— 也是飞轮回路 2 的燃料（§37）
               人改得越多 ⇒ agent 越不该放权
② 审批率       多少动作停下来问人 —— 高不一定坏（保守），但**趋势**要看
③ 硬拒率       release_gate / 前置条件 / 权限 —— 被闸门挡下的比例
④ D7 回收结果  outcome_result —— **唯一的真值**，但要等 7 天
```

⚠️⚠️ **④ 决定了灰度的节奏**：归因延迟是本项目的第一性约束（设计 §0.3），
   而它在运营层的直接后果是 —— **灰度每一档至少要待满 7 天**，
   否则你是在用"还没收敛的数据"决定要不要放权。
   ⇒ 想快只能靠 ①②③，而它们**都不是真值**，只是早期信号。

★ 判据形状：这一族**没有绝对阈值**（我们不知道"人工修正率 12% 是好是坏"）。
  能立住的是**趋势**和**对照**：这一档比上一档更高还是更低。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from syncopate.runtime.db import Database
from syncopate.runtime.release import current_state


async def collect(db: Database, org_id: str | None) -> dict[str, Any]:
    where = "WHERE org_id = $1" if org_id else ""
    args = [org_id] if org_id else []
    async with db.tx() as conn:
        runs = await conn.fetchrow(
            f"SELECT count(*) AS n,"
            f"       count(*) FILTER (WHERE status='succeeded') AS ok,"
            f"       count(*) FILTER (WHERE status='waiting_for_user') AS waiting,"
            f"       count(*) FILTER (WHERE status='failed') AS failed,"
            f"       count(*) FILTER (WHERE status='cancelled') AS cancelled"
            f"  FROM agent_runs {where}", *args)
        cases = await conn.fetchrow(
            f"SELECT count(*) AS n,"
            f"       count(*) FILTER (WHERE modified_params IS NOT NULL) AS modified,"
            f"       count(*) FILTER (WHERE status='approved') AS approved,"
            f"       count(*) FILTER (WHERE status='rejected') AS rejected,"
            f"       count(*) FILTER (WHERE outcome_result IS NOT NULL) AS with_outcome"
            f"  FROM approval_cases {where}", *args)
        by_trigger = await conn.fetch(
            f"SELECT trigger_reason, count(*) AS n FROM approval_cases {where}"
            f" GROUP BY trigger_reason ORDER BY n DESC", *args)
    return {"runs": dict(runs), "cases": dict(cases),
            "by_trigger": [dict(r) for r in by_trigger]}


def render(data: dict[str, Any]) -> None:
    r, c = data["runs"], data["cases"]
    rel = current_state()
    print(f"── 灰测闸门 ──")
    print(f"  halted={rel.halted}  max_tier={rel.max_tier}"
          + (f"  原因={rel.halt_reason}" if rel.halt_reason else ""))

    print(f"\n── run（共 {r['n']}）──")
    if r["n"]:
        for k, label in (("ok", "成功"), ("waiting", "等人裁决"),
                         ("failed", "失败"), ("cancelled", "取消")):
            print(f"  {label:<8}{r[k]:>6}  {r[k]/r['n']:>6.1%}")

    print(f"\n── 审批单（共 {c['n']}）──")
    if c["n"]:
        # ★★ 人工修正率：**唯一「人主动给出正确答案」的信号**
        print(f"  ★ 人工修正率  {c['modified']:>5} / {c['n']}  "
              f"= {c['modified']/c['n']:.1%}   ← 人改得越多，agent 越不该放权")
        print(f"    批准 {c['approved']}  驳回 {c['rejected']}")
        # ④ 真值：要等 7 天
        print(f"  ⏳ 有 D7 回收结果的  {c['with_outcome']} / {c['n']}"
              f"   ← **唯一的真值**，归因延迟决定它 7 天后才有")
        if c["n"] and c["with_outcome"] == 0:
            print("     ⚠️ 一条都还没有 ⇒ **现在做的任何放权决定都没有真值支撑**")

    if data["by_trigger"]:
        print(f"\n── 停下来的原因 ──")
        for row in data["by_trigger"]:
            print(f"  {str(row['trigger_reason']):<26}{row['n']:>5}")

    print("\n⚠️ 这一族**没有绝对阈值** —— 我们不知道「人工修正率 12%」是好是坏。")
    print("   能立住的是**趋势**和**对照**：这一档比上一档更高还是更低。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="灰测读数")
    ap.add_argument("--org", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    async def run() -> dict[str, Any]:
        db = Database()
        await db.connect(max_size=3)
        try:
            return await collect(db, args.org)
        finally:
            await db.close()

    data = asyncio.run(run())
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    else:
        render(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
