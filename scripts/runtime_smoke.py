#!/usr/bin/env python
"""Runtime 冒烟驱动：固定 query 集 → 真 HTTP 请求 → 跟 SSE 到终态 → 报延迟。

用法（先起库、API、worker，见 docs/syncopate/09 §0）：

    python scripts/runtime_smoke.py                 # 跑固定 query 集，逐条判卷
    python scripts/runtime_smoke.py --burst 20      # 附加：并发提交 20 条看排队

★ 它是**驱动器不是判据的家**：判定"对不对"的口径在 11-runtime-acceptance.md，
  这里只把「提交 → 事件流 → 终态」这条路真的走一遍并量墙钟。
★ 每条 query 带 expect（终态事件名）：跑出别的终态就算失败 ——
  「跑完了」和「跑成什么样」必须分开报（09 §4.5.12 四种停法的同一条纪律）。
⚠️ 审批路径不是失败：expect=run.waiting_for_user 的条目会去 /approvals 批准，
  然后继续跟到下一个终态 —— 「暂停必须能恢复」整条链在这里被真的走到。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

TERMINAL = {"run.completed", "run.failed", "run.cancelled", "run.waiting_for_user"}

# 跟单条 run 的事件流最多等这么久 —— 冒烟里"永远等不到终态"本身就是一种失败，
# 必须响亮地报出来（第一版没有死线，正是被"cancelled 不发终态事件"的 bug 挂死的）。
FOLLOW_DEADLINE = 30.0

# --------------------------------------------------------------------------
# 固定 query 集。名字对齐设计文档的意图 ID（I01 读 / I09 扩量…），
# 但⚠️ 当前编排是写死的三步计划（worker._execute），user_message 还不影响行为 ——
# 这组 query 是**接口契约**：B-4 接上真模型后，同一组输入应当走出不同的路径。
# --------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    user_message: str
    intent: str | None = None
    automation_tier: str | None = None
    # 第一段要等到的终态事件
    expect: str = "run.waiting_for_user"
    # 到 waiting_for_user 后是否演「人批准 → 恢复 → 跑完」的后半段（False = 人拒绝）
    approve: bool = False
    # 裁决后第二段要等到的终态（拒绝路径应是 run.cancelled）
    expect_after_approval: str = "run.completed"


# ⚠️ tier 的选法对齐灰测闸门的**默认**上限 C（release.py：忘了配置就该太严）：
# 冒烟不改 SYNCOPATE_RELEASE_MAX_TIER，所以 B/A 档在这里被拦是**正确行为**，
# 我们把它也写成一条 case —— fail-closed 本身要有人测。
CASES: list[Case] = [
    # 写路径 + 审批闸：C 档写动作必停审批；批准后必须跑完
    Case(name="budget_raise_approved",
         user_message="把 CMP_1 的日预算提高到 1200 元",
         intent="I09", automation_tier="C",
         expect="run.waiting_for_user", approve=True),
    # 同上但人来拒绝 —— 拒绝也是一种"恢复"，run 必须落到 cancelled 而不是卡死
    Case(name="budget_raise_rejected",
         user_message="把 CMP_1 的日预算提高到 1200 元（这条会被拒）",
         intent="I09", automation_tier="C",
         expect="run.waiting_for_user", approve=False,
         expect_after_approval="run.cancelled"),
    # 没声明档位：release/gateway 都按 C 处理（09 §4.6.4 的缺口在这里显形）
    Case(name="undeclared_tier",
         user_message="查一下 CMP_1 昨天的花费和转化",
         intent="I01", automation_tier=None,
         expect="run.waiting_for_user", approve=True),
    # B 档自主度超过灰测默认上限 C ⇒ 该被闸门拦下（fail-closed 的正测）
    Case(name="tier_b_release_gated",
         user_message="把 CMP_1 的日预算提高到 1200 元（B 档，应被灰测闸门拦）",
         intent="I09", automation_tier="B",
         expect="run.cancelled"),
    # D 档：永不自动 —— 连审批单都不该开，直接取消
    Case(name="tier_d_refused",
         user_message="把整个账户的预算翻倍",
         intent="I11", automation_tier="D",
         expect="run.cancelled"),
]


@dataclass
class Result:
    name: str
    run_id: str = ""
    ok: bool = False
    detail: str = ""
    submit_ms: float = 0.0
    to_terminal_ms: float = 0.0
    events: list[str] = field(default_factory=list)


async def _follow(client: httpx.AsyncClient, run_id: str, *, after: int = 0,
                  timeout: float = FOLLOW_DEADLINE) -> tuple[str, int, list[str]]:
    """跟 SSE 到终态。返回 (终态事件, 最后 seq, 全部事件名)。

    ⚠️ 死线是**总墙钟**，不是单次读超时 —— SSE 每 5s 有 keepalive，
    读超时永远不会触发，挂死只能靠总死线抓。
    """
    kinds: list[str] = []
    last = after

    async def _stream() -> str:
        nonlocal last
        headers = {"Last-Event-ID": str(after)} if after else {}
        async with client.stream("GET", f"/runs/{run_id}/events",
                                 headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            event = ""
            async for line in resp.aiter_lines():
                if line.startswith("id: "):
                    last = int(line[4:])
                elif line.startswith("event: "):
                    event = line[7:]
                elif line == "" and event:
                    kinds.append(event)
                    if event in TERMINAL:
                        return event
                    event = ""
        return "stream_closed"

    try:
        terminal = await asyncio.wait_for(_stream(), timeout=timeout)
    except asyncio.TimeoutError:
        terminal = f"deadline_{timeout:.0f}s"
    return terminal, last, kinds


async def _approve(client: httpx.AsyncClient, run_id: str, *, decision: str) -> str:
    """找到这条 run 的审批单并裁决。返回 case_ref。"""
    cases = (await client.get("/approvals")).json()
    mine = [c for c in cases if c["run_id"] == run_id]
    if not mine:
        raise RuntimeError(f"{run_id} 停在 waiting_for_user 但 /approvals 里没有它的单")
    ref = mine[0]["case_ref"]
    r = await client.post(f"/approvals/{ref}",
                          json={"decision": decision, "reviewer_id": "smoke"})
    r.raise_for_status()
    return ref


async def run_case(client: httpx.AsyncClient, case: Case) -> Result:
    res = Result(name=case.name)
    t0 = time.monotonic()
    body: dict[str, Any] = {"user_message": case.user_message}
    if case.intent:
        body["intent"] = case.intent
    if case.automation_tier:
        body["automation_tier"] = case.automation_tier
    r = await client.post("/runs", json=body,
                          headers={"Idempotency-Key": f"smoke-{case.name}-{uuid.uuid4().hex[:8]}"})
    res.submit_ms = (time.monotonic() - t0) * 1000
    if r.status_code != 201:
        res.detail = f"POST /runs → {r.status_code}: {r.text[:200]}"
        return res
    res.run_id = r.json()["run_id"]

    terminal, seq, kinds = await _follow(client, res.run_id)
    res.events += kinds
    if terminal != case.expect:
        res.to_terminal_ms = (time.monotonic() - t0) * 1000
        res.detail = f"终态 {terminal}，期望 {case.expect}"
        return res

    if terminal == "run.waiting_for_user":
        decision = "approved" if case.approve else "rejected"
        await _approve(client, res.run_id, decision=decision)
        expect2 = case.expect_after_approval
        terminal2, _, kinds2 = await _follow(client, res.run_id, after=seq)
        res.events += kinds2
        res.to_terminal_ms = (time.monotonic() - t0) * 1000
        if terminal2 != expect2:
            res.detail = f"裁决({decision})后终态 {terminal2}，期望 {expect2}"
            return res
    else:
        res.to_terminal_ms = (time.monotonic() - t0) * 1000

    res.ok = True
    return res


async def run_burst(client: httpx.AsyncClient, n: int) -> None:
    """并发提交 n 条同型 run，量提交延迟与全部到达终态的墙钟。"""
    t0 = time.monotonic()

    async def one(i: int) -> float:
        t = time.monotonic()
        r = await client.post("/runs", json={
            "user_message": f"burst 压测第 {i} 条：查 CMP_1 指标",
            "intent": "I01", "automation_tier": "A"})
        r.raise_for_status()
        rid = r.json()["run_id"]
        await _follow(client, rid, timeout=120.0)
        return (time.monotonic() - t) * 1000

    lat = await asyncio.gather(*(one(i) for i in range(n)))
    lat = sorted(lat)
    wall = time.monotonic() - t0
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
    print(f"\nburst {n} 条（终态含 waiting_for_user，未裁决）：总墙钟 {wall:.2f}s")
    print(f"  单条到首个终态 ms：P50 {p(0.50):.0f} · P95 {p(0.95):.0f} · max {lat[-1]:.0f}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="dev-token-acme")
    ap.add_argument("--burst", type=int, default=0, help="附加并发提交 N 条")
    args = ap.parse_args()

    async with httpx.AsyncClient(
            base_url=args.base_url,
            headers={"Authorization": f"Bearer {args.token}"}, timeout=30.0) as client:
        health = (await client.get("/healthz")).json()
        if health.get("status") != "ok":
            print(f"🔴 /healthz = {health} —— 服务或库没起对，先修这个")
            return 1
        print(f"✅ /healthz ok（{args.base_url}）\n")

        results = [await run_case(client, c) for c in CASES]

        wide = max(len(r.name) for r in results)
        for r in results:
            mark = "✅" if r.ok else "🔴"
            print(f"{mark} {r.name:<{wide}}  submit {r.submit_ms:6.1f} ms · "
                  f"到终态 {r.to_terminal_ms:8.1f} ms · {r.run_id}")
            print(f"   事件链: {' → '.join(r.events) if r.events else '(无)'}")
            if r.detail:
                print(f"   ⚠️ {r.detail}")

        if args.burst:
            await run_burst(client, args.burst)

        failed = [r for r in results if not r.ok]
        print(f"\n{'🔴' if failed else '✅'} {len(results) - len(failed)}/{len(results)} 条通过")
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
