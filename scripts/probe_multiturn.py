#!/usr/bin/env python
"""多轮壳层探针：拼上下文之后，模型还守不守格式 · 认不认指代。

    python scripts/probe_multiturn.py

★ 只量两件事（Chaoyu 2026-08-20 减配：单任务不回退那条不测，
  因为壳层多轮走的是**同一条上下文拼接路径**，不是新的训练分布问题）：

    ① 格式保持率   第二/三轮还能不能正常出终答（不 parse_error、不空转）
    ② 指代小测     "我是王超宇 → 我是谁" · "查了 CMP_1 → 那它上周呢"

⚠️ 判据是人读的（指代对不对没有自动尺子）——脚本把三轮的问答原样打出来，
  结论由人判。这比"编一个看起来自动的判据"诚实（守则③同族）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

import httpx

BASE = "http://127.0.0.1:8000"
HDRS = {"Authorization": "Bearer dev-token-demo", "Content-Type": "application/json"}

SCRIPTS: dict[str, list[tuple[str, str]]] = {
    # 纯闲聊指代：底座能力，不该需要训练
    "指代-姓名": [
        ("I01", "我是王超宇，先打个招呼"),
        ("I01", "我是谁？"),
    ],
    # 任务指代：第二轮不重复给 campaign_id，看它认不认上文
    "指代-任务对象": [
        ("I01", "查一下 CMP_1 昨天的花费"),
        ("I01", "那它的转化呢？"),
        ("I01", "我刚才一共问了你几个问题？"),
    ],
}


async def wait_terminal(c: httpx.AsyncClient, cid: str, run_id: str,
                        deadline: float = 120.0) -> tuple[str, dict]:
    """⚠️ 必须从**会话历史**读，不能读 `GET /runs/{id}` ——
    后者的 response_model 是白名单，**不含 result**（第一版探针就栽在这，
    5/5 跑通却打印出一串 null：判据读错了对象，不是模型没答）。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        msgs = (await c.get(f"/conversations/{cid}/messages")).json()
        row = next((m for m in msgs if m["run_id"] == run_id), None)
        if row and row["status"] in ("succeeded", "failed", "cancelled",
                                     "waiting_for_user"):
            return row["status"], row
        await asyncio.sleep(1.0)
    return "deadline", {}


async def main() -> int:
    ok_format = 0
    total = 0
    async with httpx.AsyncClient(base_url=BASE, headers=HDRS, timeout=60) as c:
        for name, turns in SCRIPTS.items():
            cid = (await c.post("/conversations",
                                json={"title": f"probe-{name}"})).json()["conversation_id"]
            print(f"\n{'=' * 70}\n【{name}】 会话 {cid}\n{'=' * 70}")
            for i, (intent, msg) in enumerate(turns, 1):
                total += 1
                run = (await c.post(
                    f"/conversations/{cid}/messages",
                    json={"user_message": msg, "intent": intent, "automation_tier": "C"},
                    headers={**HDRS, "Idempotency-Key": f"probe-{uuid.uuid4().hex}"}
                )).json()
                status, row = await wait_terminal(c, cid, run["run_id"])
                print(f"\n  第{i}轮 · 用户：{msg}")
                if status == "succeeded":
                    ok_format += 1
                    res = row.get("result") or {}
                    if isinstance(res, str):
                        res = json.loads(res)
                    print(f"  第{i}轮 · agent（{res.get('behavior')}）："
                          f"{json.dumps(res.get('answer'), ensure_ascii=False)}")
                else:
                    print(f"  第{i}轮 · 🔴 {status}：{row.get('error')}")

    print(f"\n{'=' * 70}")
    print(f"① 格式保持：{ok_format}/{total} 轮正常收尾")
    print("② 指代对不对 —— 看上面的回答，人判（第 2/3 轮有没有认出上文）")
    return 0 if ok_format == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
