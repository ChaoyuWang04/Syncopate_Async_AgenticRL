"""K10 门槛（27 §12）：① 反馈闭环（同 run 多条、后条推翻前条）· ② 归因强制（负样本无 reason_code 必拒；external 类零进入
负样本池）· ③ 抽取覆盖（四路各有产出；"跑对了但业务结果坏"只有业务结果通道能抽到）· ④ 通道纪律（原始 trace 哈希不变；无
expected 不能入库；manifest 与实际条数一致；密钥残留出局）· ⑤ 版本切片 · 词表复用（cap 名 / behavior / 六族前缀）。"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from syncopate.runtime import flywheel, metrics
from syncopate.runtime.api import create_app
from syncopate.runtime.db import Database, claim_run, create_run, finish_run
from tests.runtime.test_api import ACME, Client, _pg_available

pytestmark = pytest.mark.skipif(not _pg_available(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")
TRACE = {"Authorization": "Bearer dev-token-acme-trace"}


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=4)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _ids():
    return f"org_{uuid.uuid4().hex[:8]}", f"run_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


# --------------------------------------------------------------------------
# 门槛①：反馈闭环 + 词表
# --------------------------------------------------------------------------


def test_feedback_multiple_rows_and_later_overrides_earlier(client) -> None:
    run_id = client.post("/runs", json={"user_message": "x"}, headers=ACME).json()["run_id"]
    r1 = client.post(f"/runs/{run_id}/feedback", json={"rating": -1, "label": "false_claim_cap"}, headers=ACME)
    assert r1.status_code == 201, r1.text
    r2 = client.post(f"/runs/{run_id}/feedback", json={"rating": 1, "label": "good", "comment": "再看是对的"}, headers=ACME)
    assert r2.status_code == 201
    bad = client.post(f"/runs/{run_id}/feedback", json={"rating": -1, "label": "not_a_label"}, headers=ACME)
    assert bad.status_code == 422
    rows = client.get(f"/runs/{run_id}/feedback", headers=ACME).json()
    assert [r["rating"] for r in rows] == [-1, 1]

    async def cands():
        return await flywheel.extract_candidates(client.app.state.db, org_id="org_acme", window_hours=1, limit=200)
    c = client.loop.run_until_complete(cands())
    assert run_id not in {x["run_id"] for x in c["negative_feedback"]}, "后条推翻前条：不该再在负反馈池里"


def test_label_vocab_is_reused_not_invented() -> None:
    v = flywheel.label_vocab()
    caps = {x for x in v if x.endswith("_cap")}
    assert len(caps) >= 10, "cap 名没从 verifier_engine 取到"
    assert "behavior:reject" in v and "behavior:clarify" in v and "F4_commitment" in v
    assert flywheel.label_ok("F6_meta:window_boundary") and not flywheel.label_ok("wrong_answer")


# --------------------------------------------------------------------------
# 门槛②：归因强制 · 门槛④：通道纪律
# --------------------------------------------------------------------------


def test_annotation_requires_role_and_vocab(client) -> None:
    run_id = client.post("/runs", json={"user_message": "x"}, headers=ACME).json()["run_id"]
    assert client.post(f"/runs/{run_id}/annotations", json={"label": "good", "reason_code": "unknown"}, headers=ACME).status_code == 403
    r = client.post(f"/runs/{run_id}/annotations",
                    json={"label": "false_claim_cap", "reason_code": "external_system_error",
                          "expected": {"text": "应该说查不到"}, "notes": "环境空"}, headers=TRACE)
    assert r.status_code == 201, r.text
    assert client.post(f"/runs/{run_id}/annotations", json={"label": "false_claim_cap", "reason_code": "nope"},
                       headers=TRACE).status_code == 422


def test_admission_and_rejection_lists() -> None:
    base = {"turns": ["问"], "note": {"text": "expected"}, "meta": {"contract": "syncopate_v1", "label": "false_claim_cap",
                                                                "reason_code": "model_reasoning", "source_run_id": "r"}}
    ok, why = flywheel.admission_verdict(base)
    assert ok, why
    no_exp = {**base, "note": None}
    assert flywheel.admission_verdict(no_exp) == (False, "no_expected")
    ext = {**base, "meta": {**base["meta"], "reason_code": "external_system_error"}}
    assert flywheel.admission_verdict(ext)[1].startswith("external_reason")
    noreason = {**base, "meta": {**base["meta"], "reason_code": None}}
    assert flywheel.admission_verdict(noreason) == (False, "negative_without_reason_code")
    leak = {**base, "meta": {**base["meta"], "x": "Authorization: Bearer dev-token-acme"}}
    assert flywheel.admission_verdict(leak) == (False, "secret_residue")
    good = {**base, "meta": {**base["meta"], "label": "good", "reason_code": None}}
    assert flywheel.admission_verdict(good)[0]


def test_trace_to_case_keeps_raw_trace_intact_and_export_manifest_matches(tmp_path) -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="扩量到 12 万")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute("INSERT INTO tool_calls (run_id, org_id, step, tool, status, ok, external_idempotency_key, side_effect, registry_version) "
                               "VALUES ($1,$2,1,'campaign.update_budget','succeeded',TRUE,'k-secret-1',TRUE,'reg_x')", run, org)
        await finish_run(db, org_id=org, run_id=run, status="succeeded", result={"text": "已扩量"})
        from syncopate.runtime.db import trace
        before = flywheel.trace_digest(await trace(db, org_id=org, run_id=run))
        case_unsigned = await flywheel.build_case_from_run(db, org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute("INSERT INTO run_annotations (org_id, run_id, label, reason_code, expected_json, annotator) "
                               "VALUES ($1,$2,'good',NULL,$3,'chaoyu')", org, run, {"text": "扩量正确"})
        case_signed = await flywheel.build_case_from_run(db, org_id=org, run_id=run)
        after = flywheel.trace_digest(await trace(db, org_id=org, run_id=run))
        manifest = await flywheel.export_batch(db, [case_unsigned, case_signed], dataset_version="v15-fb-test",
                                              created_by="test", root=tmp_path)
        async with db.tx() as conn:
            row = await conn.fetchrow("SELECT n_cases, n_rejected, manifest FROM training_exports WHERE batch_id=$1", manifest["batch_id"])
        return before, after, case_signed, manifest, dict(row)

    before, after, case, manifest, row = with_db(go)
    assert before == after, "加工过程改动了原始 trace"
    assert case["level"] == "FB" and case["turns"] == ["扩量到 12 万"] and case["meta"]["contract"] == "syncopate_v1"
    assert "k-secret-1" not in json.dumps(case, ensure_ascii=False), "幂等键这类密钥类字段没删"
    assert manifest["n_cases"] == 1 and manifest["n_rejected"] == 1 and manifest["rejected"][0]["why"] == "no_expected"
    lines = (Path(manifest["destination"]) / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest["n_cases"] == row["n_cases"]
    assert json.loads(lines[0])["id"].startswith("FB_")


# --------------------------------------------------------------------------
# 门槛③：抽取覆盖 —— "跑对了但业务结果坏"只有业务结果通道抽得到
# --------------------------------------------------------------------------


def test_business_outcome_route_catches_succeeded_run_with_bad_outcome() -> None:
    async def go(db):
        org, run = _ids()
        await create_run(db, org_id=org, run_id=run, user_message="x")
        await claim_run(db, worker_id="w", org_id=org, run_id=run)
        async with db.tx() as conn:
            await conn.execute("INSERT INTO tool_calls (run_id, org_id, step, tool, status, ok, side_effect, external_idempotency_key) "
                               "VALUES ($1,$2,1,'campaign.update_budget','response_lost',NULL,TRUE,'k')", run, org)
        await finish_run(db, org_id=org, run_id=run, status="succeeded", result={"text": "done"})
        # 另一条：普通失败 + 一条高价值成功
        org2 = org
        await create_run(db, org_id=org2, run_id="run_f", user_message="x")
        await claim_run(db, worker_id="w", org_id=org2, run_id="run_f")
        await finish_run(db, org_id=org2, run_id="run_f", status="failed", error="boom")
        await create_run(db, org_id=org2, run_id="run_hv", user_message="x")
        await claim_run(db, worker_id="w", org_id=org2, run_id="run_hv")
        async with db.tx() as conn:
            await conn.execute("INSERT INTO tool_calls (run_id, org_id, step, tool, status, ok, side_effect, external_idempotency_key) "
                               "VALUES ($1,$2,1,'campaign.update_budget','succeeded',TRUE,TRUE,'k2')", "run_hv", org2)
        await finish_run(db, org_id=org2, run_id="run_hv", status="succeeded", result={"text": "ok"})
        return run, await flywheel.extract_candidates(db, org_id=org, window_hours=1)

    run, c = with_db(go)
    routes = {r: {x["run_id"] for x in v} for r, v in c.items()}
    assert run in routes["business_outcome"]
    assert run not in routes["failed"] and run not in routes["negative_feedback"], "只有业务结果通道能抽到它"
    assert "run_f" in routes["failed"] and "run_hv" in routes["high_value_success"]


# --------------------------------------------------------------------------
# 门槛⑤：版本切片
# --------------------------------------------------------------------------


def test_metrics_slice_by_version(client) -> None:
    client.post("/runs", json={"user_message": "x"}, headers=ACME)
    rows = client.get("/metrics/by_version?key=prompt_version", headers=ACME).json()
    assert rows and any(r["version"] == "syncopate_prompt_v1" for r in rows)
    assert all({"version", "runs", "failed", "failed_ratio"} <= set(r) for r in rows)
    assert client.get("/metrics/by_version?key=bogus", headers=ACME).status_code == 422
