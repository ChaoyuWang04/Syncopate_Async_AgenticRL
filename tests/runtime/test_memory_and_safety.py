"""记忆库与安全线（B-2 第二批）：**三条硬边界**，每条错了都不报错、只会悄悄错。

    ① `episodic` lane **agent 不可写** —— 沙盒里是「工具直接报错，等价于真实 API 的 403」
    ② 写类工具**只提案，不入库** —— 「不会立即入库，需经审核」
    ③ 安全线**不替模型判断过没过期** —— 只如实返回 valid_to

★★ ③ 最容易被"顺手做好"：加一个 `expired: true` 看起来是帮忙，实际是
   **把这道判断从模型手里拿走**，而且与训练侧不一致。`axes.py` 的原话：

     「工具不替模型判断过没过期，只如实返回 valid_to。
       真实世界里没人会在返回里塞一个 expired: true。
       模型必须自己拿它和今天比 —— 所以 reference_now 必须进 prompt。」
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest

from syncopate.runtime import tool_impls as impl
from syncopate.runtime.db import Database


def _pg() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg(), reason="需要 PostgreSQL：bash scripts/serving/pg_bootstrap.sh")


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


def _org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


# ── ① episodic 是硬边界 ─────────────────────────────────────────────────

def test_episodic_lane_is_refused_not_silently_dropped():
    """★★ 写系统专属 lane ⇒ **硬拒**，不是静默丢弃。

    静默丢弃的话，模型会以为写成功了 —— 而这条记忆永远不存在。
    """
    async def body(db):
        with pytest.raises(impl.MemoryWriteRefused, match="system_only_lane"):
            await impl.memory_write_proposal(
                db, _org(), "run_1", lane="episodic", content="x",
                confidence=0.9, evidence_refs=["a", "b"])
    with_db(body)


def test_refusal_is_not_a_retriable_failure():
    """★ 硬边界用**自己的异常类型**，不复用 `PlatformError`。

    后者的语义是"外部世界拒绝了"，可以带 retriable；
    这个是"我们自己的规则不允许"，**重试永远没用**。
    ⇒ 类型混了，`ToolRuntime` 就可能去重试一件永远不会成功的事。
    """
    from syncopate.runtime.platform import PlatformError
    assert not issubclass(impl.MemoryWriteRefused, PlatformError)


def test_confidence_and_evidence_thresholds_are_enforced():
    async def body(db):
        org = _org()
        with pytest.raises(impl.MemoryWriteRefused, match="low_confidence"):
            await impl.memory_write_proposal(db, org, "r", lane="semantic", content="x",
                                             confidence=0.5, evidence_refs=["a", "b"])
        with pytest.raises(impl.MemoryWriteRefused, match="insufficient_evidence"):
            await impl.memory_write_proposal(db, org, "r", lane="semantic", content="x",
                                             confidence=0.9, evidence_refs=["a"])
    with_db(body)


# ── ② 提案不入库 ───────────────────────────────────────────────────────

def test_a_proposal_writes_zero_rows_to_the_memory_store():
    """★★★ 「不会立即入库，需经审核」—— 那就**一行都不许写进 memory_records**。

    ⚠️ 这条如果破了，"需经审核"就只是一句话：
      提案接口写了记录，审核环节就成了装饰。
    """
    async def body(db):
        org = _org()
        out = await impl.memory_write_proposal(
            db, org, "run_1", lane="semantic", content="真人出镜素材在 SEA 更好",
            confidence=0.85, evidence_refs=["CMP_1", "CMP_2"])
        assert out["applied"] is False and out["status"] == "pending"
        async with db.tx() as conn:
            n = await conn.fetchval(
                "SELECT count(*) FROM memory_records WHERE org_id=$1", org)
            p = await conn.fetchval(
                "SELECT count(*) FROM memory_proposals WHERE org_id=$1", org)
        assert n == 0, "★ 提案直接入库了 —— 审核环节被绕过"
        assert p == 1
    with_db(body)


def test_invalidate_does_not_delete_the_original():
    """「只是**提议**作废，不会立即生效，也**不删除**原记录 ——
    你需要知道『我们曾经这么以为』。」"""
    async def body(db):
        org = _org()
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO memory_records (org_id, record_id, lane, content, confidence) "
                "VALUES ($1,'M1','semantic','旧结论',0.9)", org)
        await impl.memory_invalidate(db, org, "r", record_id="M1", reason="素材已下线")
        async with db.tx() as conn:
            row = await conn.fetchrow(
                "SELECT invalidated_at FROM memory_records WHERE org_id=$1 AND record_id='M1'",
                org)
        assert row is not None, "★ 原记录被删了"
        assert row["invalidated_at"] is None, "★ 提议就立即生效了"
    with_db(body)


# ── read 与 search 的可见性刻意不同 ─────────────────────────────────────

def test_search_drops_expired_but_read_does_not():
    """★★ 两个工具对同一条记录的可见性**刻意不同**：

        memory.search  自动剔除已过 TTL 的（沙盒描述原话）
        memory.read    **不校验**这条记忆现在还成不成立

    ⇒ "顺手统一"会让「我想看看那条过期的记忆当初写了什么」变成做不到。
    """
    async def body(db):
        org = _org()
        past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO memory_records (org_id, record_id, lane, content,"
                " confidence, expires_at) VALUES ($1,'OLD','semantic','过期的',0.9,$2)",
                org, past)
        found = await impl.memory_search(db, org, lane="semantic")
        assert found["records"] == [], "★ search 没有剔除过期记录"
        one = await impl.memory_read(db, org, record_id="OLD")
        assert one["found"] is True, "★ read 也把过期的藏了 —— 那就查不到历史了"
    with_db(body)


def test_unknown_record_reports_not_found_rather_than_empty_content():
    async def body(db):
        out = await impl.memory_read(db, _org(), record_id="NOPE")
        assert out["found"] is False
    with_db(body)


# ── ③ 安全线不替模型判过期 ──────────────────────────────────────────────

def test_safety_line_returns_valid_to_and_no_expired_verdict():
    """★★★ 只如实返回 `valid_to`，**不许有 `expired` / `is_stale` 这类字段**。"""
    async def body(db):
        org = _org()
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO safety_lines (org_id, product_id, region, cpi_d7_max,"
                " roas_d7_min, valid_from, valid_to) "
                "VALUES ($1,'P1','US',2.5,0.4,'2020-01-01','2020-01-07')", org)
        out = await impl.benchmark_get_safety_line(db, org, product_id="P1", region="US")
        assert out["found"] is True
        assert "valid_to" in out and "valid_from" in out
        verdicts = {"expired", "is_stale", "usable", "should_use", "stale"}
        assert not (verdicts & set(out)), (
            f"★ 工具替模型判了过期：{verdicts & set(out)}\n"
            f"⇒ 这道判断必须留给模型（axes.py），而且训练侧也没有这个字段")
    with_db(body)


def test_missing_safety_line_is_not_an_empty_line():
    """查不到 ⇒ 明确报"没有"，**不返回一条空的线**。

    返回空线会被读成"没有限制" —— 而真相是"我们不知道有没有限制"。
    同 `policy.search` 那条三态：**两种空不能长得一样。**
    """
    async def body(db):
        out = await impl.benchmark_get_safety_line(db, _org(), product_id="X", region="Y")
        assert out["found"] is False and "safety_line_not_found" in out["error"]
    with_db(body)
