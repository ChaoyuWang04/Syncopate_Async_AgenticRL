"""数据源类工具（B-2 第五批）：**每个都有一条"不许多做"**。

★ 沙盒描述里每个工具都写了它**不做什么**。那些"不做"不是省事，
  是**把某个判断留给模型** —— 多做一步，就把对应的能力从训练目标里抹掉了。

★★ 而 `mmp.get_attribution` 是这批里唯一**不能建成随机噪声**的一条，
   理由见 `07 §2.2 A4`（下面那条测试里抄了原话）。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from syncopate.runtime import tool_impls as impl
from syncopate.runtime.db import Database
from syncopate.runtime.platform import FakeAdPlatform


def _pg() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2); await db.close(); return True
        except Exception:
            return False
    return asyncio.run(probe())


def _run(coro):
    return asyncio.run(coro)


def with_db(body):
    async def main():
        db = Database()
        await db.connect(max_size=5)
        try:
            return await body(db)
        finally:
            await db.close()
    return asyncio.run(main())


# ── ★★★ mmp：确定的成因和方向，不是噪声 ────────────────────────────────

def test_mmp_shortfall_has_a_fixed_direction_not_random_noise():
    """★★★ `07 §2.2` 的原话：

        「原计划是给两个源加一个**随机偏差**。**那是假的** ——
          真实的打架有**确定的成因和方向**：它来自归因窗口配置，
          而且偏差方向可预测（AF 少算、Meta 多算）。
          **模型该学的是识别成因并据此判断该信谁，不是识别噪声。**」

    ⇒ 判据：窗口更短时，MMP 必须**恒定地少算**，而且**每次跑结果一样**。
    """
    p = FakeAdPlatform()
    # ⚠️ 字段名以**沙盒**为准（B-5b 对照台纠正过：不是 installs 是 installs_7d）
    runs = [_run(impl.mmp_get_attribution(p, campaign_id="C1")) for _ in range(3)]
    assert len({r["installs_7d"] for r in runs}) == 1, "★ 结果在抖 —— 那就是噪声不是机制"
    r = runs[0]
    # ★ 被短窗口漏掉的那批**落到自然量里** —— 这正是 A4 的机制本身
    assert r["organic_installs_7d"] > 0, "★ 短窗口下 MMP 必须少算"


def test_aligned_windows_produce_no_gap():
    """★ 窗口一致时差异应当消失 —— 这证明差异**来自窗口**，而不是凭空加的。"""
    p = FakeAdPlatform()
    aligned = _run(impl.mmp_get_attribution(
        p, campaign_id="C1",
        mmp_click_window_days=impl.PLATFORM_CLICK_WINDOW_DAYS))
    assert aligned["organic_installs_7d"] == 0, "★ 窗口一致就不该有漏掉的那批"
    assert (aligned["attribution_window"]["click_days"]
            == aligned["platform_attribution_window"]["click_days"])


def test_both_windows_are_reported():
    """沙盒描述：「做判断前先看两边的窗口是不是一致」——
    不给窗口，那句话就**没法执行**。"""
    p = FakeAdPlatform()
    out = _run(impl.mmp_get_attribution(p, campaign_id="C1"))
    # ★ 两个窗口是**两个平级字段**（同沙盒），不是嵌在一个 dict 里
    assert "attribution_window" in out and "platform_attribution_window" in out
    assert out["attribution_window"] != out["platform_attribution_window"]


def test_mmp_does_not_say_which_source_to_trust():
    """★ **不做"该信谁"的判断** —— 那正是模型要学的那一步。"""
    p = FakeAdPlatform()
    out = _run(impl.mmp_get_attribution(p, campaign_id="C1"))
    assert not ({"trust", "use_source", "recommended_source", "authoritative"} & set(out))


# ── 几条"不许多做" ──────────────────────────────────────────────────────

def test_geo_breakdown_reports_asset_count_and_no_scale_verdict():
    """★ 只给各地域现状，**不告诉能不能扩**；
    但 `asset_count` **必须给** —— 否则模型分不出
    「这个地域不行」和「这个地域样本太少」。"""
    async def body(db):
        pid = f"P_{uuid.uuid4().hex[:6]}"
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO geo_performance (product_id, region, roas_d7, cpi_d7,"
                " asset_count) VALUES ($1,'US',0.55,2.1,40),($1,'JP',0.20,3.9,2)", pid)
        out = await impl.analysis_geo_breakdown(db, product_id=pid)
        assert {r["region"] for r in out["regions"]} == {"US", "JP"}
        assert all("asset_count" in r for r in out["regions"])
        assert not ({"can_scale", "recommended_regions"} & set(out))
    with_db(body)


def test_feature_lift_requires_a_region_and_reports_significance():
    """★★ **必须逐地域算**：同一 feature 在不同地域可能**符号相反**。

    ⇒ `region` 是必填，而且这里**不做**跨地域聚合的兜底。
    ⚠️ 还必须带置信区间与样本量 —— 只给点估计的话，
      「lift=+3% 样本 12 条」和「lift=+3% 样本 8000 条」长得一模一样。
    """
    async def body(db):
        f = f"feat_{uuid.uuid4().hex[:6]}"
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO feature_lifts (feature, region, lift, ci_low, ci_high,"
                " n_treatment, n_control) VALUES ($1,'US',0.12,0.04,0.20,800,760),"
                "($1,'JP',-0.09,-0.18,-0.01,300,290)", f)
        us = await impl.analysis_feature_lift(db, feature=f, region="US")
        jp = await impl.analysis_feature_lift(db, feature=f, region="JP")
        assert us["lift"] > 0 > jp["lift"], "★ 符号相反正是必须逐地域算的理由"
        assert us["significant"] is True
        assert {"ci_low", "ci_high", "n_treatment", "n_control"} <= set(us)
    with_db(body)


def test_lift_spanning_zero_is_not_significant():
    async def body(db):
        f = f"feat_{uuid.uuid4().hex[:6]}"
        async with db.tx() as conn:
            await conn.execute(
                "INSERT INTO feature_lifts (feature, region, lift, ci_low, ci_high,"
                " n_treatment, n_control) VALUES ($1,'US',0.03,-0.06,0.12,12,11)", f)
        out = await impl.analysis_feature_lift(db, feature=f, region="US")
        assert out["significant"] is False
    with_db(body)


def test_detect_anomalies_gives_types_not_a_playbook():
    """只**定性**给异常类型，**不给**方案（那在 `playbook.get_optimization`）。

    ⚠️ 三件事分给三个工具是刻意的：异常是什么 · 怎么办 · 数据够不够下结论。
      合并任意两个，模型就不用自己串这条链了 —— 而串链正是要训的东西。
    """
    p = FakeAdPlatform()
    out = _run(impl.campaign_detect_anomalies(p, campaign_id="C1"))
    assert "anomalies" in out
    assert not ({"steps", "playbook", "recommendation"} & set(out))


def test_unknown_anomaly_type_is_not_matched_to_a_near_one():
    """★ 报"没有"，**不猜一个相近的打法** —— 猜错的方案会被照着执行。"""
    async def body(db):
        out = await impl.playbook_get_optimization(db, anomaly_type="cpi_spik")
        assert out["found"] is False
    with_db(body)


def test_missing_risk_record_does_not_mean_cleared():
    """★★★ 查不到风控记录 ⇒ **不能默认放行**。

    「没有风控记录」和「查过了、没问题」是两件事。
    把前者当后者，就是在**未知状态下放行** —— 同 `policy.search` 那条三态。
    """
    async def body(db):
        out = await impl.risk_check_account(db, f"org_{uuid.uuid4().hex[:6]}",
                                            account_id="ACC_1")
        assert out["found"] is False
        assert out.get("allow_increase") is not True, "★ 查不到却给了放行"
    with_db(body)


pytestmark = pytest.mark.skipif(not _pg(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")
