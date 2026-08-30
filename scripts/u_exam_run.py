#!/usr/bin/env python
"""U 路 P0 · 考场运行器：把考题打进**真实会话路径**（conversation API→worker→decider），
逐轮收集 行为/回复/工具调用/审批提案，落机读原始结果。

    .venv/bin/python scripts/u_exam_run.py --exam context --arm cand
    .venv/bin/python scripts/u_exam_run.py --exam talk --arm base --concurrency 4

⚠️ 打的是哪个模型由**当前 org_demo worker 的 decider 指向**决定（换臂=换 worker env），
   本脚本把 arm 名只当标签记录——标签与实际模型不一致是人祸，跑前自查 worker 日志。
产物：logs/u_route/run_<arm>_<exam>.jsonl（每题一行：逐轮 terminal/behavior/reply/tools/proposal）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import asyncpg
import httpx

BASE = "http://127.0.0.1:8000"
H = {"Authorization": "Bearer dev-token-demo"}
DSN = os.environ.get("SYNCOPATE_PG_DSN",
                     "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate")
TERMINAL = {"succeeded", "failed", "cancelled", "waiting_for_user"}


async def run_turn(c: httpx.AsyncClient, pg: asyncpg.Pool, cid: str, msg: str) -> dict:
    r = await c.post(f"/conversations/{cid}/messages", json={"user_message": msg},
                     headers={"Idempotency-Key": f"u-{uuid.uuid4().hex[:10]}"})
    r.raise_for_status()
    rid = r.json()["run_id"]
    for _ in range(120):
        await asyncio.sleep(1.5)
        st = await pg.fetchval(
            "SELECT status FROM agent_runs WHERE org_id='org_demo' AND run_id=$1", rid)
        if st in TERMINAL:
            break
    term = await pg.fetchrow(
        """SELECT kind, payload FROM run_events WHERE org_id='org_demo' AND run_id=$1
           AND kind LIKE 'run.%' AND kind!='run.started' ORDER BY seq DESC LIMIT 1""", rid)
    tools = await pg.fetch(
        "SELECT tool, arguments FROM tool_calls WHERE org_id='org_demo' AND run_id=$1 "
        "ORDER BY id", rid)
    # ★ v15：三条信令各有**不同的终止语义**（R4①）⇒ clarify 走挂起（waiting_for_user）、
    #   reject 走终止，**状态不再都是 succeeded**，而 finish_run 的终态事件表里没有
    #   waiting_for_user 这一项 ⇒ 只看终态事件会把这些题读成空值（实测 17 题）。
    #   ⇒ 权威记录是 agent_loop 发的 `session.*` 事件，优先读它。
    sig_ev = await pg.fetchrow(
        """SELECT kind, payload FROM run_events WHERE org_id='org_demo' AND run_id=$1
           AND kind LIKE 'session.%' ORDER BY seq DESC LIMIT 1""", rid)
    prop = await pg.fetchrow(
        "SELECT proposed_params, action_type AS tool FROM approval_cases "
        "WHERE org_id='org_demo' AND run_id=$1 ORDER BY created_at DESC LIMIT 1", rid)
    raw = term["payload"] if term else None
    if isinstance(raw, str):                 # 裸 asyncpg 无 jsonb codec ⇒ 字符串
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    payload = raw or {}
    ans = payload.get("answer") or {}
    tool_names = [t["tool"] for t in tools]
    # ★ 契约感知：v15 的终态 payload 是 {signal, arguments, text}，**没有 behavior/answer**。
    #   ⛔ 2026-08-30：不适配的话考场会把每一题都读成 behavior=None / reply=""，
    #     判分器全线判错 —— 而且不报错（「机制在但没接上」的考场版）。
    if sig_ev is not None:
        sp = sig_ev["payload"]
        if isinstance(sp, str):
            try:
                sp = json.loads(sp)
            except json.JSONDecodeError:
                sp = {}
        sp = sp or {}
        beh = sig_ev["kind"].split(".", 1)[1]          # session.defer → defer
        args = sp.get("arguments") or {}
        rep = sp.get("text") or ""
        clar = (args.get("question") or "") if isinstance(args, dict) else ""
        summ = ""
        extra = {"signal_arguments": args}
    elif "signal" in payload or "text" in payload:
        sig = payload.get("signal")
        args = payload.get("arguments") or {}
        beh = sig or ("tool_call" if any(not t.startswith("session.") for t in tool_names)
                      else "answer")
        rep = payload.get("text") or ""
        clar = args.get("question") or (args.get("missing_fields") or [""])[0] \
            if isinstance(args, dict) else ""
        summ = ""                                    # v15 已废除 summary
        extra = {"signal_arguments": args}
    else:
        beh = payload.get("behavior")
        rep = (ans.get("reply") or "") if isinstance(ans, dict) else ""
        summ = (ans.get("summary") or "") if isinstance(ans, dict) else ""
        clar = (ans.get("clarification") or ans.get("missing_field") or "") \
            if isinstance(ans, dict) else ""
        extra = {}
    return {
        "run_id": rid, "status": st, "terminal": term["kind"] if term else None,
        "behavior": beh,
        "reply": rep,
        "summary": summ,
        "clarification": clar,
        **extra,
        "tools": [{"tool": t["tool"], "arguments": t["arguments"]} for t in tools],
        "proposal": ({"tool": prop["tool"], "params": prop["proposed_params"]}
                     if prop else None),
    }


async def one_item(c, pg, sem, item, out, arm):
    async with sem:
        cv = await c.post("/conversations", json={"title": f"uexam-{arm}-{item['id']}"})
        cid = cv.json()["conversation_id"]
        turns = []
        for msg in item["turns"]:
            turns.append(await run_turn(c, pg, cid, msg))
        out.append({"id": item["id"], **{k: item[k] for k in ("cat", "level") if k in item},
                    "conversation_id": cid, "turns": turns})
        print(f"  {item['id']} done ({len(out)})", flush=True)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", choices=["talk", "context", "context_v2", "context_v3"],
                    required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    items = [json.loads(x) for x in open(f"data/u_route/{args.exam}_exam.jsonl")]
    if args.limit:
        items = items[: args.limit]
    pg = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    out: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(base_url=BASE, headers=H, timeout=300) as c:
        hz = (await c.get("/healthz")).json()
        assert hz.get("status") == "ok", hz
        await asyncio.gather(*[one_item(c, pg, sem, it, out, args.arm) for it in items])
    await pg.close()
    out.sort(key=lambda r: r["id"])
    p = Path(f"logs/u_route/run_{args.arm}_{args.exam}.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ {p}  {len(out)} 题  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
