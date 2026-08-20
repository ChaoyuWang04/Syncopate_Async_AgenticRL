#!/usr/bin/env python
"""给 demo 租户播种参考数据 —— **让 agent 有东西可查**。

    python scripts/seed_demo_data.py                 # 播到 org_demo
    python scripts/seed_demo_data.py --org org_x     # 换租户
    python scripts/seed_demo_data.py --check         # 只报告现状，不写

★★★ 为什么需要这个脚本（2026-08-20 抓到的一个"机制在但没接上"新变种）

工具实现 30/30、测试 224 全绿、压测 24/25 —— 但**真人租户手里一条参考数据都没有**：

    safety_lines        org_demo 0 条（全库 70 条全是测试临时 org 的残留）
    budget_rules        表本身 0 行
    industry_baselines  表本身 0 行
    memory_records      org_demo 0 条
    seasonal_events / playbooks / account_risk   0 行

⇒ 模型查什么都查不到，只好回「policy_not_found」「no_data」「安全线缺货」——
  **那些结论是如实的，不是模型不会**。伪装得极好的原因：
  **每个测试都自己造数据**，所以没有任何判据在问"真实租户手里有什么"。

⇒ 判据（本脚本 `--check`）：真人租户的每张参考表都不许为空。

★ 播的数据刻意**有多样性**：不同 campaign 的成熟度/表现/账户状态不同，
  不同 product×region 的安全线不同 ⇒ 不同的问题会问出不同的答案。
  全都一样的话，看起来"能查到了"，但任何问题都得到同一个回答，等于没测。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")
from syncopate.runtime.db import Database  # noqa: E402

PLATFORM_STATE = Path("data/demo/platform_state.json")

# ── 内部安全线：**决定能不能扩量的那把尺子**（不是行业基准）──────────────────
# 逐 product×region，刻意给不同的阈值 —— 同一个动作在不同地域结论应该不同。
SAFETY_LINES = [
    # (product_id, region, cpi_d7_max, roas_d7_min, retention_d1_min, daily_budget_max)
    ("GAME_PUZZLE", "华东",   2.60, 0.50, 0.32, 120_000),
    ("GAME_PUZZLE", "华南",   2.80, 0.45, 0.30,  80_000),
    ("GAME_PUZZLE", "华北",   2.70, 0.48, 0.31,  90_000),
    ("GAME_PUZZLE", "北美",   4.50, 0.70, 0.35, 200_000),
    ("GAME_PUZZLE", "东南亚", 1.60, 0.40, 0.28,  60_000),
    ("GAME_SLG",    "华东",   5.50, 0.55, 0.38, 150_000),
    ("GAME_SLG",    "华南",   5.80, 0.52, 0.36, 120_000),
    ("GAME_SLG",    "华北",   5.20, 0.58, 0.39, 130_000),
    ("GAME_SLG",    "北美",   9.00, 0.80, 0.42, 300_000),
    ("GAME_SLG",    "东南亚", 3.20, 0.60, 0.34, 100_000),
]

# ── 行业基准：⚠️ 只是参考，**不是**决策依据（safety_lines 才是）────────────
INDUSTRY_BASELINES = [
    # (platform, game_genre, metric, p25, p50, p75, sample_size)
    ("meta",   "puzzle",   "cpi",        1.60, 2.30, 3.40, 1820),
    ("meta",   "puzzle",   "roas_d7",    0.28, 0.52, 0.88, 1820),
    ("meta",   "puzzle",   "ctr",       0.012, 0.019, 0.031, 1820),
    ("meta",   "puzzle",   "retention_d1", 0.24, 0.31, 0.40, 1560),
    ("meta",   "strategy", "cpi",        3.20, 5.10, 8.60,  940),
    ("meta",   "strategy", "roas_d7",    0.31, 0.57, 0.95,  940),
    ("meta",   "strategy", "ctr",       0.008, 0.014, 0.024, 940),
    ("google", "puzzle",   "cpi",        1.40, 2.10, 3.10, 1120),
    ("google", "puzzle",   "roas_d7",    0.30, 0.55, 0.92, 1120),
    ("google", "strategy", "cpi",        2.90, 4.60, 7.80,  610),
    ("google", "strategy", "roas_d7",    0.35, 0.63, 1.05,  610),
]

# ── 账户风控：三种状态都要有，否则"受限"这条路永远测不到 ───────────────────
ACCOUNT_RISK = [
    ("ACC_DEMO", [], "normal", True),
    ("ACC_RISK", ["payment_retry", "creative_policy_warning"], "restricted", False),
    ("ACC_FROZEN", ["chargeback_dispute"], "frozen", False),
]

# ── 打法库：key 必须是 detect_anomalies 真会返回的那些（cpi_spike / roas_drop）──
PLAYBOOKS = [
    ("cpi_spike",
     ["拉取近 7 天分素材 CPI，定位是否单条素材拖高",
      "检查 frequency：>3.5 说明受众饱和，优先扩受众而不是加预算",
      "对比同 product×region 的安全线 cpi_d7_max，确认是否已越线",
      "若为素材问题，替换素材后观察 3 天再评估"],
     ["★ 不要在 CPI 异常期加预算 —— 会把坏素材的量放大",
      "D7 未收敛时不要据此下结论"]),
    ("roas_drop",
     ["先用 metrics.get_freshness 确认数据是否已收敛（D7）",
      "查 mmp.get_attribution 看归因窗口内是否有缺口",
      "分地域看 analysis.geo_breakdown，判断是全局下滑还是单地域",
      "对比 benchmark.get_safety_line 的 roas_d7_min 判断是否触线"],
     ["ROAS 波动在 D1–D3 极大，未收敛前的下滑多为假象",
      "★ 触线才是行动依据，行业基准只是参考"]),
    ("frequency_fatigue",
     ["扩受众包或放宽定向，而不是直接加预算",
      "轮换素材，观察 CTR 是否回升"],
     ["加预算会让 frequency 继续升高，通常适得其反"]),
    ("budget_underspend",
     ["检查出价是否过低、定向是否过窄",
      "确认账户风控状态是否 restricted"],
     ["先排除账户级限制，再动出价"]),
]

# ── 时令日历：只给背景，不判断该不该投 ────────────────────────────────────
def seasonal_rows() -> list[tuple]:
    today = date.today()
    return [
        ("华东",   "春节",       today + timedelta(days=12), 1.80, ["红色", "团圆", "礼盒"]),
        ("华东",   "618 大促",   today + timedelta(days=40), 1.45, ["折扣", "限时"]),
        ("华南",   "春节",       today + timedelta(days=12), 1.75, ["红色", "团圆"]),
        ("华北",   "开学季",     today + timedelta(days=25), 1.20, ["校园", "青春"]),
        ("北美",   "Black Friday", today + timedelta(days=55), 2.10, ["discount", "bundle"]),
        ("东南亚", "斋月",       today + timedelta(days=18), 1.60, ["家庭", "夜间"]),
        ("东南亚", "双 11",      today + timedelta(days=48), 1.35, ["折扣"]),
    ]

# ── 记忆库：四个 lane 都要有；含**已过期**和**已作废**各一条 ─────────────────
# ★ 过期的那条是判据：`memory.search` 必须剔除它，而 `memory.read` 必须还能读到
#   （两个工具对同一条记录的可见性刻意不同，schema 注释里写着）。
def memory_rows() -> list[tuple]:
    now = date.today()
    return [
        ("mem_ep_001", "episodic", {"campaign_id": "CMP_1"},
         "2026-07 对 CMP_1 做过一次 +20% 扩量，两周后 ROAS 从 0.58 回落到 0.49，随后回滚。",
         0.85, ["run_20260712", "audit_88421"], now + timedelta(days=120), None, None),
        ("mem_ep_002", "episodic", {"campaign_id": "CMP_3"},
         "CMP_3 在 2026-06 换过一次素材包，CPI 从 6.4 降到 5.1，但一个月后回升。",
         0.78, ["run_20260603", "run_20260701"], now + timedelta(days=90), None, None),
        ("mem_sem_001", "semantic", {"product_id": "GAME_PUZZLE", "region": "华东"},
         "华东地区的消除类素材，'团圆/家庭'主题在节前两周 CTR 平均高 18%。",
         0.72, ["insight_2214", "insight_2288"], now + timedelta(days=200), None, None),
        ("mem_sem_002", "semantic", {"product_id": "GAME_SLG"},
         "SLG 品类的首日留存与 D7 ROAS 相关性弱，不要用 D1 留存预判 ROAS。",
         0.81, ["insight_1902", "insight_2011"], now + timedelta(days=200), None, None),
        ("mem_biz_001", "business", {"account_id": "ACC_DEMO"},
         "本季度买量目标：ROAS_D7 ≥ 0.55，单月预算上限 300 万。",
         0.95, ["okr_2026Q3", "finance_memo_08"], now + timedelta(days=60), None, None),
        ("mem_risk_001", "risk", {"account_id": "ACC_RISK"},
         "ACC_RISK 因素材合规警告被限流，加预算前必须先解除风控。",
         0.90, ["risk_case_771", "platform_notice_0812"], now + timedelta(days=30), None, None),
        # ★ 已过期：memory.search 应剔除、memory.read 仍可读
        ("mem_sem_old", "semantic", {"product_id": "GAME_PUZZLE"},
         "（已过期）2025 年的结论：北美消除类 CPI 低于 3.0 即可扩量。",
         0.60, ["insight_0912", "insight_0955"], now - timedelta(days=5), None, None),
        # ★ 已作废：作废是**标记**不是删除（"你需要知道我们曾经这么以为"）
        ("mem_ep_void", "episodic", {"campaign_id": "CMP_5"},
         "（已作废）CMP_5 暂停原因记为预算耗尽。",
         0.55, ["run_20260520", "audit_66120"], now + timedelta(days=100),
         now - timedelta(days=3), "复盘发现真实原因是素材审核未通过，非预算耗尽"),
    ]


async def seed(db: Database, org: str, *, check_only: bool) -> int:
    state = json.loads(PLATFORM_STATE.read_text(encoding="utf-8"))
    campaigns = {k: v for k, v in state["campaigns"].items() if not k.startswith("_")}
    today = date.today()

    if check_only:
        print(f"== {org} 的参考数据现状 ==")
        bad = 0
        async with db.tx() as conn:
            checks = [
                ("safety_lines", "SELECT count(*) FROM safety_lines WHERE org_id=$1", True),
                ("memory_records", "SELECT count(*) FROM memory_records WHERE org_id=$1", True),
                ("budget_rules", "SELECT count(*) FROM budget_rules WHERE org_id=$1", True),
                ("account_risk", "SELECT count(*) FROM account_risk WHERE org_id=$1", True),
                ("industry_baselines", "SELECT count(*) FROM industry_baselines", False),
                ("seasonal_events", "SELECT count(*) FROM seasonal_events", False),
                ("playbooks", "SELECT count(*) FROM playbooks", False),
                ("policy_clauses", "SELECT count(*) FROM policy_clauses", False),
            ]
            for name, sql, scoped in checks:
                n = await (conn.fetchval(sql, org) if scoped else conn.fetchval(sql))
                mark = "✅" if n else "🔴"
                if not n:
                    bad += 1
                print(f"  {mark} {name:<20} {n}")
        print(f"\n{'🔴 有空表 ⇒ agent 查不到东西，任何能力评价都不成立' if bad else '✅ 都有数据'}")
        return 1 if bad else 0

    async with db.tx() as conn:
        # 安全线：给一段**当前有效**的日期窗口（工具按 valid_from/valid_to 过滤）
        for pid, region, cpi, roas, ret, cap in SAFETY_LINES:
            await conn.execute(
                """
                INSERT INTO safety_lines (org_id, product_id, region, cpi_d7_max,
                    roas_d7_min, retention_d1_min, daily_budget_max, valid_from, valid_to)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (org_id, product_id, region, valid_from) DO UPDATE SET
                    cpi_d7_max=EXCLUDED.cpi_d7_max, roas_d7_min=EXCLUDED.roas_d7_min,
                    retention_d1_min=EXCLUDED.retention_d1_min,
                    daily_budget_max=EXCLUDED.daily_budget_max, valid_to=EXCLUDED.valid_to
                """, org, pid, region, cpi, roas, ret, cap,
                today - timedelta(days=30), today + timedelta(days=180))

        await conn.execute(
            """
            INSERT INTO budget_rules (org_id, max_increase_pct, approval_threshold,
                                      risk_check_required, monthly_cap)
            VALUES ($1, 20, 100000, true, 3000000)
            ON CONFLICT (org_id) DO UPDATE SET
                max_increase_pct=EXCLUDED.max_increase_pct,
                approval_threshold=EXCLUDED.approval_threshold,
                monthly_cap=EXCLUDED.monthly_cap
            """, org)

        for acc, flags, state_, allow in ACCOUNT_RISK:
            await conn.execute(
                """
                INSERT INTO account_risk (org_id, account_id, flags, state, allow_increase)
                VALUES ($1,$2,$3,$4,$5)
                ON CONFLICT (org_id, account_id) DO UPDATE SET
                    flags=EXCLUDED.flags, state=EXCLUDED.state,
                    allow_increase=EXCLUDED.allow_increase
                """, org, acc, json.dumps(flags), state_, allow)

        for platform, genre, metric, p25, p50, p75, n in INDUSTRY_BASELINES:
            await conn.execute(
                """
                INSERT INTO industry_baselines (platform, game_genre, metric,
                                                p25, p50, p75, sample_size)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (platform, game_genre, metric) DO UPDATE SET
                    p25=EXCLUDED.p25, p50=EXCLUDED.p50, p75=EXCLUDED.p75,
                    sample_size=EXCLUDED.sample_size
                """, platform, genre, metric, p25, p50, p75, n)

        for region, event, evdate, lift, tags in seasonal_rows():
            await conn.execute(
                """
                INSERT INTO seasonal_events (region, event, event_date, lift_factor,
                                             creative_tags)
                VALUES ($1,$2,$3,$4,$5)
                ON CONFLICT (region, event, event_date) DO UPDATE SET
                    lift_factor=EXCLUDED.lift_factor, creative_tags=EXCLUDED.creative_tags
                """, region, event, evdate, lift, json.dumps(tags, ensure_ascii=False))

        for atype, steps, cautions in PLAYBOOKS:
            await conn.execute(
                """
                INSERT INTO playbooks (anomaly_type, steps, cautions) VALUES ($1,$2,$3)
                ON CONFLICT (anomaly_type) DO UPDATE SET
                    steps=EXCLUDED.steps, cautions=EXCLUDED.cautions
                """, atype, json.dumps(steps, ensure_ascii=False),
                json.dumps(cautions, ensure_ascii=False))

        for (rid, lane, subject, content, conf, refs, expires,
             invalidated, reason) in memory_rows():
            await conn.execute(
                """
                INSERT INTO memory_records (org_id, record_id, lane, subject, content,
                    confidence, evidence_refs, expires_at, invalidated_at,
                    invalidate_reason)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (org_id, record_id) DO UPDATE SET
                    content=EXCLUDED.content, confidence=EXCLUDED.confidence,
                    expires_at=EXCLUDED.expires_at,
                    invalidated_at=EXCLUDED.invalidated_at,
                    invalidate_reason=EXCLUDED.invalidate_reason
                """, org, rid, lane, json.dumps(subject, ensure_ascii=False), content,
                conf, json.dumps(refs), expires, invalidated, reason)

        # 地域表现 + feature lift：按 demo 的 product×region 补齐
        # ⚠️ asset_count 必须真实反映样本量 —— 模型要靠它分辨"这个地域不行"和"样本太少"
        geo = [("GAME_PUZZLE", "华东", 0.62, 2.10, 48), ("GAME_PUZZLE", "华南", 0.35, 2.60, 9),
               ("GAME_PUZZLE", "华北", 0.51, 2.45, 27), ("GAME_PUZZLE", "北美", 0.78, 4.10, 62),
               ("GAME_PUZZLE", "东南亚", 0.41, 1.55, 15),
               ("GAME_SLG", "华东", 0.44, 6.10, 33), ("GAME_SLG", "华南", 0.58, 5.30, 21),
               ("GAME_SLG", "华北", 0.31, 5.80, 6), ("GAME_SLG", "北美", 0.86, 8.40, 55),
               ("GAME_SLG", "东南亚", 1.24, 1.50, 41)]
        for pid, region, roas, cpi, n in geo:
            await conn.execute(
                """
                INSERT INTO geo_performance (product_id, region, roas_d7, cpi_d7, asset_count)
                VALUES ($1,$2,$3,$4,$5)
                ON CONFLICT (product_id, region) DO UPDATE SET
                    roas_d7=EXCLUDED.roas_d7, cpi_d7=EXCLUDED.cpi_d7,
                    asset_count=EXCLUDED.asset_count
                """, pid, region, roas, cpi, n)

        # ★ 同一个 feature 在不同地域**符号相反** —— 这正是 feature_lifts 逐地域存的理由
        feats = [("真人出镜", "华东", 0.18, 0.09, 0.27, 5200, 5100),
                 ("真人出镜", "东南亚", -0.12, -0.21, -0.03, 3100, 3050),
                 ("快节奏剪辑", "华东", 0.09, 0.02, 0.16, 4800, 4700),
                 ("快节奏剪辑", "北美", 0.24, 0.15, 0.33, 6100, 6000),
                 ("竖版素材", "华南", 0.31, 0.22, 0.40, 2900, 2850),
                 ("字幕强化", "东南亚", 0.27, 0.18, 0.36, 3400, 3380)]
        for feat, region, lift, lo, hi, nt, nc in feats:
            await conn.execute(
                """
                INSERT INTO feature_lifts (feature, region, product_id, lift, ci_low,
                                           ci_high, n_treatment, n_control)
                VALUES ($1,$2,NULL,$3,$4,$5,$6,$7)
                ON CONFLICT (feature, region, product_id) DO UPDATE SET
                    lift=EXCLUDED.lift, ci_low=EXCLUDED.ci_low, ci_high=EXCLUDED.ci_high
                """, feat, region, lift, lo, hi, nt, nc)

    print(f"✅ 已播种到 {org}：")
    print(f"   安全线 {len(SAFETY_LINES)}（5 地域 × 2 品类）· 行业基准 {len(INDUSTRY_BASELINES)}"
          f" · 打法 {len(PLAYBOOKS)} · 时令 {len(seasonal_rows())}")
    print(f"   记忆 {len(memory_rows())}（四个 lane + 1 条已过期 + 1 条已作废）"
          f" · 账户风控 {len(ACCOUNT_RISK)}（normal/restricted/frozen）")
    print(f"   预算政策 1（涨幅上限 20% · 审批阈值 100000 · 月上限 300 万）")
    print(f"   假平台 campaign {len(campaigns)} 个（成熟度/表现/状态各不同，见 "
          f"{PLATFORM_STATE}）")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--org", default="org_demo")
    ap.add_argument("--check", action="store_true", help="只报告现状，不写")
    args = ap.parse_args()

    db = Database()
    await db.connect(max_size=4)
    try:
        return await seed(db, args.org, check_only=args.check)
    finally:
        await db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
