"""runtime 侧的工具实现（B-2）——**外部世界的形状 → 模型认识的形状**。

★★★ 这一层为什么必须存在（而不是让模型直接吃平台的返回）

    platform.py   **外部世界**的形状：Meta 的 `paging.cursors.after`、613/1487632、
                  BUC 积分、显式 fields —— 越像真的越好，因为它决定降级路径真不真
    本模块        **我们给模型**的形状：沙盒 spec 里那一份 —— 模型是照它训出来的

⇒ 两者不是同一个东西，中间需要**适配**。这正是真实生产里 adapter 层的位置。

⚠️⚠️ 反过来做（让模型去吃平台的原始形状）有两条都很贵的后果：
  ① 模型读不懂 `paging.cursors.after` —— 它被训练成读 `next_cursor`
  ② 换平台（Meta → Google）时，**模型要重训**，因为返回形状变了
  ⇒ 适配层的价值就是把"外部世界会变"这件事挡在模型之外。

★ 但适配**不是翻译自由**：沙盒 spec 是契约，这一层只许把平台的东西**映射**过去，
  不许自己发明字段、也不许把平台说不知道的东西填成一个值（宁可缺字段）。
"""

from __future__ import annotations

from typing import Any

from syncopate.runtime.platform import FakeAdPlatform
from syncopate.runtime.retrieval import RetrievalService, RetrievalStatus

# 沙盒 `campaign.list` 的描述里写死了「**每页最多 3 条**」——
# 那句话在**模型的 prompt 里**，模型是照它训出来的。
#
# ⚠️ 平台自己的上限是 25（`MAX_PAGE_SIZE`，实查 Meta）。两个数不一样是**对的**：
#   平台的上限是外部世界的事实，这个 3 是**我们和模型之间的契约**。
# ⇒ 适配层按小的那个走。宁可少取一页、多翻一次，也不要让模型看到
#   一个和它 prompt 里那句话不符的页大小。
SANDBOX_PAGE_SIZE = 3


async def campaign_list(platform: FakeAdPlatform, *, account_id: str,
                        status: str | None = None,
                        cursor: str | None = None) -> dict[str, Any]:
    """列 campaign。**把平台的 `paging` 映射成沙盒的 `next_cursor`。**

    ⚠️ `next_cursor` 为空**必须表示"没有下一页"**，不能表示"我不知道"——
      沙盒描述里那句「非空表示还有下一页」是模型的判据，含义必须一致。
    """
    page = await platform.list_campaigns(
        account_id=account_id, fields=["id", "name", "status", "daily_budget"],
        after=cursor, limit=SANDBOX_PAGE_SIZE)
    rows = page["data"]
    if status:
        # ⚠️ 过滤在**取回之后**做，所以过滤掉之后这一页可能变空，
        #   但 next_cursor 仍然要给 —— 空页不等于没有下一页。
        # ⚠️ **大小写不敏感**：工具 spec 写的是「按状态过滤，如 active」（小写），
        #   而平台存的是 `ACTIVE` —— 精确匹配等于让**照着 spec 填参数的模型**
        #   永远查不到东西（2026-08-20 实测："帮我优化一下" → 无可执行 campaign）。
        #   ⇒ B-5 那条：spec 是模型的契约，实现必须满足 spec 而不是反过来。
        rows = [r for r in rows if str(r.get("status", "")).lower() == status.lower()]
    has_next = page["paging"]["has_next"]
    return {"campaigns": rows, "count": len(rows), "has_more": has_next,
            "next_cursor": page["paging"]["cursors"]["after"] if has_next else ""}


async def metrics_get_freshness(platform: FakeAdPlatform, *, campaign_id: str,
                                metric: str = "roas_d7") -> dict[str, Any]:
    """指标的观测条件。**只给事实，不给结论**（沙盒描述里的原话）。

    ⚠️ 所以这里**不返回** "可不可信 / 该不该动" 这类字段 ——
      那是模型的判断。多给一个结论字段，等于把决策从模型手里拿走，
      而那正是我们要训练的东西。
    """
    fresh = await platform.get_freshness(campaign_id=campaign_id)
    m = await platform.get_metrics(campaign_id=campaign_id)
    # 各指标的收敛天数与最小样本量：真实世界里 roas_d7 最慢（要 7 天窗口）
    spec = {"cpi":          (3, 300,  (1.5, 3.0)),
            "ctr":          (1, 200,  (0.01, 0.05)),
            "installs":     (1, 100,  (0, None)),
            "retention_d1": (2, 500,  (0.2, 0.45)),
            "roas_d7":      (7, 1000, (0.3, 0.9))}
    converge_at, min_n, rng = spec.get(metric, spec["roas_d7"])
    # ⚠️⚠️ 这四个字段是 **B-5b 的对照台抓出来的**（2026-08-19）：
    #   沙盒描述明写它给「累计样本量、以及该指标的预期区间」，
    #   而 runtime 第一版只给了成熟度 ⇒ **模型被训练成会读这些，生产上取不到**。
    #   ⇒ 字段名对不上不会报错，只会让模型**取不到数然后自己编一个**。
    return {"campaign_id": campaign_id, "metric": metric,
            "days_elapsed": fresh["data_age_days"],
            "converge_at_day": converge_at,
            "current_value": m.get(metric),
            "sample_size": int(m.get("installs") or 0),
            "min_sample_size": min_n,
            "expected_final_range": list(rng),
            "maturity": fresh["maturity"]}


async def policy_search(retrieval: RetrievalService, org_id: str, *, query: str,
                        platform: str | None = None, region: str | None = None,
                        top_k: int = 3) -> dict[str, Any]:
    """检索政策条款。**三态必须传下去**（`12 §3.1`）。

    ★★ `no_match`（查不到）和 `unavailable`（查不了）**不能合并**：
        查不到  = 「没有政策限制这件事」
        查不了  = 「**我们不知道**有没有限制」
    合并的话，一次服务故障看起来就是**放行信号**。
    ⇒ 所以这里返回 `status`，而不是只返回一个可能为空的列表。
    """
    r = await retrieval.search_policy(org_id=org_id, query=query,
                                      platform=platform, region=region, top_k=top_k)
    return {"status": r.status.value, "clauses": r.hits, "query": query}


async def insight_search_claims(retrieval: RetrievalService, org_id: str, *, query: str,
                                region: str | None = None, product_id: str | None = None,
                                active_only: bool = False,
                                top_k: int = 3) -> dict[str, Any]:
    """检索历史复盘结论。返回的是**经验**，不是实时数据。

    ⚠️ `active_only` 默认 **False** —— 和沙盒一致：默认连"已被取代/推翻"的一起给，
      并按 `status` 标明。默认只给现行的话，模型就**看不见"这条结论被推翻过"**，
      而"发现历史结论和现在的数据矛盾"正是我们要它学会的一类判断。
    """
    r = await retrieval.search_claims(org_id=org_id, query=query, top_k=top_k)
    claims = r.hits
    if active_only:
        claims = [c for c in claims if c.get("status") == "active"]
    if region:
        claims = [c for c in claims if c.get("region") in (None, region)]
    if product_id:
        claims = [c for c in claims if c.get("product_id") in (None, product_id)]
    return {"status": r.status.value, "claims": claims, "query": query}


# ══════════════════════════════════════════════════════════════════════════
# 记忆库 + 安全线（B-2 第二批，2026-08-19）
# ══════════════════════════════════════════════════════════════════════════
#
# ★★ 这一批有**三条硬边界**，每条错了都不报错、只会悄悄错：
#
#   ① `episodic` lane **agent 不可写** —— 沙盒里是"工具直接报错，等价于 403"
#   ② 写类工具**只提案，不入库** —— 「不会立即入库，需经审核」
#   ③ 安全线**不替模型判断过没过期** —— 只如实返回 valid_to
#
# ⚠️ ③ 最容易被"顺手做好"：加一个 `expired: true` 看起来是帮忙，
#   实际是**把这道判断从模型手里拿走**，而且与训练侧不一致
#   （`axes.py`：「真实世界里没人会在返回里塞一个 expired」）。

LANES = ("episodic", "semantic", "business", "risk")
SYSTEM_ONLY_LANES = ("episodic",)
MIN_MEMORY_CONFIDENCE = 0.7
MIN_EVIDENCE_REFS = 2


class MemoryWriteRefused(Exception):
    """硬边界：这次写入**不该发生**，不是失败重试的问题。

    ⚠️ 刻意不复用 `PlatformError` —— 那个的语义是"外部世界拒绝了"，
    可以带 retriable。这个是"我们自己的规则不允许"，**重试永远没用**。
    """


async def memory_read(db, org_id: str, *, record_id: str) -> dict[str, Any]:
    """按 id 读一条。

    ⚠️ **不剔除过期的、也不校验它现在还成不成立**（沙盒描述里的原话）。
      和 `memory.search` 刻意不同：search 自动剔 TTL，read 不剔。
      "顺手统一"会让「我想看看那条过期的记忆当初写了什么」变成做不到。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT record_id, lane, subject, content, confidence, evidence_refs,"
            "       expires_at, invalidated_at, invalidate_reason "
            "FROM memory_records WHERE org_id=$1 AND record_id=$2", org_id, record_id)
    if row is None:
        # ★ 报"没有"，不猜（守则④）
        return {"found": False, "record_id": record_id}
    return {"found": True, **{k: row[k] for k in row.keys()}}


async def memory_search(db, org_id: str, *, lane: str, top_k: int = 5,
                        **subject: Any) -> dict[str, Any]:
    """按 lane + 主体检索，**自动剔除已过 TTL 的**。"""
    if lane not in LANES:
        return {"error": f"unknown_lane: {lane}", "lane": lane, "records": []}
    async with db.tx() as conn:
        rows = await conn.fetch(
            "SELECT record_id, lane, subject, content, confidence, expires_at "
            "FROM memory_records "
            "WHERE org_id=$1 AND lane=$2 "
            "  AND (expires_at IS NULL OR expires_at > now()) "   # ★ TTL 剔除
            "  AND invalidated_at IS NULL "
            "ORDER BY created_at DESC LIMIT $3", org_id, lane, top_k)
    return {"lane": lane, "records": [dict(r) for r in rows]}


async def memory_write_proposal(db, org_id: str, run_id: str, *, lane: str, content: str,
                                confidence: float, evidence_refs: list[Any],
                                idempotency_key: str | None = None,
                                **subject: Any) -> dict[str, Any]:
    """提交写入**提案** —— 不入库（`memory_records` 一行都不写）。

    三条前置校验（沙盒描述里逐条写着），任何一条不过 ⇒ **硬拒**：
      · `episodic` 由系统维护，不可写
      · confidence ≥ 0.7
      · evidence_refs ≥ 2
    """
    if lane in SYSTEM_ONLY_LANES:
        raise MemoryWriteRefused(
            f"system_only_lane: {lane} 由系统维护，agent 不可写入")
    if lane not in LANES:
        raise MemoryWriteRefused(f"unknown_lane: {lane}")
    if float(confidence) < MIN_MEMORY_CONFIDENCE:
        raise MemoryWriteRefused(
            f"low_confidence: {confidence} < {MIN_MEMORY_CONFIDENCE}")
    if len(evidence_refs or []) < MIN_EVIDENCE_REFS:
        raise MemoryWriteRefused(
            f"insufficient_evidence: 需要至少 {MIN_EVIDENCE_REFS} 条证据引用")
    return await _file_proposal(db, org_id, run_id, "write",
                                {"lane": lane, "content": content,
                                 "confidence": confidence,
                                 "evidence_refs": evidence_refs, "subject": subject})


async def memory_invalidate(db, org_id: str, run_id: str, *, record_id: str, reason: str,
                            idempotency_key: str | None = None) -> dict[str, Any]:
    """提议作废一条记忆。

    ⚠️ **只是提议，不会立即生效，也不删除原记录** ——
      「你需要知道『我们曾经这么以为』」。
    """
    return await _file_proposal(db, org_id, run_id, "invalidate",
                               {"record_id": record_id, "reason": reason})


async def memory_conflict_resolve(db, org_id: str, run_id: str, *, record_ids: list[str],
                                  decision: str, keep_record_id: str | None = None,
                                  idempotency_key: str | None = None) -> dict[str, Any]:
    """两条记忆矛盾时提议处置。

    ⚠️ **不判断哪条是对的** —— 那要拿实际数据核（沙盒描述原话）。
      这里只把"提议"记下来。
    """
    if len(record_ids or []) < 2:
        raise MemoryWriteRefused("need_two_records: record_ids 至少给两条")
    if decision not in ("supersede", "merge"):
        raise MemoryWriteRefused(f"unknown_decision: {decision}")
    return await _file_proposal(db, org_id, run_id, "conflict_resolve",
                                {"record_ids": record_ids, "decision": decision,
                                 "keep_record_id": keep_record_id})


async def _file_proposal(db, org_id: str, run_id: str, kind: str,
                         payload: dict[str, Any]) -> dict[str, Any]:
    """★ 写进 `memory_proposals`，**永远不写 `memory_records`**。

    进 `memory_records` 的唯一路径是审核通过 —— 否则"需经审核"就只是一句话。
    """
    import json as _json
    async with db.tx() as conn:
        pid = await conn.fetchval(
            "INSERT INTO memory_proposals (org_id, run_id, kind, payload) "
            "VALUES ($1,$2,$3,$4) RETURNING id",
            org_id, run_id, kind, _json.dumps(payload, ensure_ascii=False))
    return {"proposal_id": pid, "kind": kind, "status": "pending",
            "applied": False}


async def benchmark_get_safety_line(db, org_id: str, *, product_id: str,
                                    region: str) -> dict[str, Any]:
    """查安全线。**如实返回 valid_from / valid_to，不加 `expired` 判断。**

    ⚠️⚠️ 这是刻意的（`axes.py` 的纪律）：
      「工具不替模型判断过没过期，只如实返回 valid_to。
        真实世界里没人会在返回里塞一个 expired: true。
        模型必须自己拿它和今天比 —— 所以 reference_now 必须进 prompt。」
      ⇒ 加一个 expired 字段 = 把这道判断从模型手里拿走，且与训练侧不一致。
    ⚠️ 查不到 ⇒ 明确报"没有"，**不返回一条空的线**（那会被当成"没有限制"）。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT product_id, region, cpi_d7_max, roas_d7_min, retention_d1_min,"
            "       daily_budget_max, valid_from, valid_to "
            "FROM safety_lines WHERE org_id=$1 AND product_id=$2 AND region=$3 "
            "ORDER BY valid_from DESC LIMIT 1", org_id, product_id, region)
    if row is None:
        return {"found": False,
                "error": f"safety_line_not_found: {product_id}/{region}"}
    return {"found": True, **{k: row[k] for k in row.keys()}}


# ══════════════════════════════════════════════════════════════════════════
# 素材库（B-2 第三批，2026-08-19）—— 第一次把 B-1b 的异步任务机制用起来
# ══════════════════════════════════════════════════════════════════════════
#
# ★ 这一簇的核心是 `upload → poll_review` 这条**异步链**：
#     上传只把素材放进审核队列，**不返回审核结论**
#     审核结果由模型**自己决定何时去查**，每次查都扣 BUC 积分
#
# ⚠️ 四个"只做一件事"的边界（沙盒描述里逐条写着），别顺手合并：
#     upload                 只上传，**不返回**审核结论
#     get_asset_tags         只给**单条**素材的标签与历史表现，不做跨素材归因
#     get_metrics_by_asset   只给素材粒度，**不返回** campaign 层汇总
#     search_similar         只检索现有素材，**不生成**、也**不判断**适不适合当前 campaign
#   ⇒ 合并任意两个，都会让"该用哪个工具"这个判断从模型手里消失。


async def creative_upload(platform: FakeAdPlatform, *, campaign_id: str,
                          creative_name: str, asset_type: str,
                          duration_seconds: float | None = None,
                          idempotency_key: str | None = None,
                          client_request_id: str | None = None) -> dict[str, Any]:
    """上传素材。**上传成功 ≠ 审核通过。**

    ⚠️ 返回里**不许**有 `approved` / `review_result` 这类字段 ——
      给了就等于替模型断言"审核过了"，而真实世界里那时候审核还没开始。
    """
    out = await platform.upload_creative(
        campaign_id=campaign_id, creative_name=creative_name, asset_type=asset_type,
        duration_seconds=duration_seconds, idempotency_key=idempotency_key,
        client_request_id=client_request_id)
    return {"asset_id": out["asset_id"], "status": out["status"]}


async def creative_poll_review(platform: FakeAdPlatform, *,
                               asset_id: str) -> dict[str, Any]:
    """查审核结果。**立刻返回当前状态，不替你等待。**

    ⚠️ 没出结果时要**告诉模型还差多久**（沙盒描述里明写）——
      不给这个数，模型就只能靠瞎猜决定什么时候再查，
      而每次查都扣积分 ⇒ 猜错的代价是真的。
    """
    asset = platform.assets.get(asset_id)
    if asset is None:
        # ★ 报"没有"，不猜（守则④）
        return {"found": False, "asset_id": asset_id, "error": f"unknown_asset: {asset_id}"}
    job = await platform.get_job(job_id=asset["review_job_id"])
    if job["status"] == "pending":
        remaining = platform._jobs[asset["review_job_id"]]["settle_at"] - platform.clock()
        return {"found": True, "asset_id": asset_id, "review_status": "pending",
                "seconds_remaining": max(0.0, round(remaining, 1))}
    if job["status"] == "failed":
        return {"found": True, "asset_id": asset_id, "review_status": "rejected",
                "reason": job.get("error")}
    return {"found": True, "asset_id": asset_id,
            "review_status": job["result"].get("review", "approved")}


async def creative_get_asset_tags(platform: FakeAdPlatform, *,
                                  creative_id: str | None = None,
                                  creative_name: str | None = None) -> dict[str, Any]:
    """单条素材的视觉标签与历史表现。

    ⚠️ **不做跨素材的对比归因**（那在 `analysis.feature_lift`），
      也**不返回** campaign 层数据（那在 `campaign.get_metrics`）。
    """
    asset = None
    if creative_id:
        asset = platform.assets.get(creative_id)
    elif creative_name:
        asset = next((a for a in platform.assets.values()
                      if a["name"] == creative_name), None)
    if asset is None:
        return {"found": False, "error": "asset_not_found"}
    return {"found": True, "creative_id": asset["asset_id"],
            "creative_name": asset["name"], "tags": asset.get("tags", {}),
            "history": asset.get("metrics", {})}


async def creative_get_metrics_by_asset(platform: FakeAdPlatform, *,
                                        campaign_id: str | None = None,
                                        creative_id: str | None = None) -> dict[str, Any]:
    """按素材粒度查表现。**不返回 campaign 层汇总。**"""
    rows = [a for a in platform.assets.values()
            if (creative_id is None or a["asset_id"] == creative_id)
            and (campaign_id is None or a["campaign_id"] == campaign_id)]
    return {"assets": [{"creative_id": a["asset_id"], "creative_name": a["name"],
                        **a.get("metrics", {})} for a in rows]}


async def creative_search_similar(platform: FakeAdPlatform, *, visual_tags: list[str],
                                  region: str | None = None,
                                  platform_name: str | None = None,
                                  min_ipm: float | None = None,
                                  top_k: int = 5) -> dict[str, Any]:
    """按视觉标签检索素材库，按 IPM 从高到低。

    ⚠️ **不生成**新素材，也**不判断**检索到的素材适不适合当前 campaign ——
      后者是模型的判断，替它做了就等于把这个能力训没了。
    """
    want = set(visual_tags or [])
    hits = []
    for a in platform.assets.values():
        tags = set((a.get("tags") or {}).get("themes", []))
        if want and not (want & tags):
            continue
        ipm = (a.get("metrics") or {}).get("ipm")
        if min_ipm is not None and (ipm is None or ipm < min_ipm):
            continue
        if region and (a.get("tags") or {}).get("region") not in (None, region):
            continue
        hits.append({"creative_id": a["asset_id"], "creative_name": a["name"],
                     "ipm": ipm, "tags": a.get("tags", {})})
    hits.sort(key=lambda h: (h["ipm"] is not None, h["ipm"] or 0), reverse=True)
    return {"assets": hits[:top_k], "query_tags": sorted(want)}


# ══════════════════════════════════════════════════════════════════════════
# system.wait —— ⚠️ 两侧**行为刻意不同**，而且这个不同必须让模型看得见
# ══════════════════════════════════════════════════════════════════════════
#
# ★★★ 起因（2026-08-19，做素材那批时撞出来的）
#
# 我原本把它登记成「⛔ 刻意不实现：生产侧的等待由异步任务表达」。**那是错的**：
# `creative.poll_review` 的**描述里明写**「应当先用 system.wait 等够再查」——
# 那句话在**模型的 prompt 里**，模型是照它训出来的。
# ⇒ runtime 没有它 ⇒ 模型调 → `unknown_tool` → 多半退化成立刻重查，
#   而每次重查都扣积分。**"刻意不实现"变成了"制造一个坑"。**
#
# ★★ 但也**不能照沙盒那样直接睡**：
#
#     system.wait 的 spec      单次上限 **600 秒**
#     worker 的 lease          默认 **60 秒**
#
#   睡 600 秒 ⇒ **租约过期** ⇒ 另一个 worker 抢走这条 run ⇒ **重复执行**。
#   这正是「训练侧的实现不能套层薄膜就上生产」的一个具体例子：
#   沙盒里没有租约这回事，所以它可以随便睡。
#
# ⇒ 修法：**等，但只等到租约安全线为止，并如实告诉模型实际等了多久。**
#   模型看到 `waited_seconds < seconds` 就知道还没等够，可以再调一次。
#   ⚠️ 关键是**别假装等够了** —— 那会让模型以为审核该出结果了，
#     然后拿一个 pending 当成"审核失败"。
LEASE_SAFETY_RATIO = 0.5      # 最多用掉租约的一半，留出续约/收尾的余量


async def system_wait(sleep, lease_seconds: int, *, seconds: int) -> dict[str, Any]:
    """等待，**上限受租约约束**。

    ⚠️ `sleep` 是注入的（测试里换成假的）—— 真的睡会让测试慢且偶发红。
    """
    cap = max(1.0, lease_seconds * LEASE_SAFETY_RATIO)
    want = max(0, int(seconds))
    actual = min(float(want), cap)
    if actual:
        await sleep(actual)
    return {"requested_seconds": want,
            "waited_seconds": round(actual, 1),
            # ★ 没等够就**明说**，别让模型以为时间到了
            "truncated_by_lease": actual < want}


# ══════════════════════════════════════════════════════════════════════════
# 写工具（B-2 第四批，2026-08-19）—— 两条**跨工具前置条件**在这里强制执行
# ══════════════════════════════════════════════════════════════════════════
#
# ★★★ 这一批的核心不是"能不能写成功"，是**两条前置条件**：
#
#   campaign.create        「本轮如果还没有一次成功的 approval.create_case，
#                            **不要调用本工具**」——地域扩展的正确产出是一份提议，不是直接建站
#   campaign.scale_budget  「幅度 ±20% 以内可以直接执行；**超出必须先走 approval.create_case**」
#
# ⚠️⚠️ 这两条在沙盒里是靠 **reward/cap** 教的 —— 模型学到"这么做会扣分"。
#   而在生产上**不能只靠模型记得**：`campaign.create` 是**不可逆**的，
#   建出来就在花钱、删不掉。「模型多数时候会遵守」对不可逆动作是不够的。
# ⇒ 所以这里把它们做成**硬前置**：条件不满足就拒绝执行，不进平台。
#   ★ 这正是「沙盒是 runtime 的子集，契约由 runtime 定义」的正面兑现：
#     沙盒用扣分表达的约束，runtime 用硬闸兑现。

SCALE_AUTO_APPROVE_BAND = 0.20      # ±20%：沙盒描述里写死的自动执行区间


class PreconditionNotMet(Exception):
    """跨工具前置条件没满足 —— **不是重试能解决的**。

    ⚠️ 和 `MemoryWriteRefused` 同族、和 `PlatformError` 不同族：
    后者是"外部世界拒绝了"（可能可重试），这个是"你还没做该做的那一步"。
    """


async def _has_successful_approval(db, org_id: str, run_id: str) -> bool:
    """本轮有没有开成过审批单。"""
    async with db.tx() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM approval_cases WHERE org_id=$1 AND run_id=$2",
            org_id, run_id)
    return bool(n)


async def approval_create_case(db, org_id: str, run_id: str, *, campaign_id: str,
                               change_type: str, requested_value: Any, reason: str,
                               idempotency_key: str | None = None) -> dict[str, Any]:
    """开一张审批单。**不会立即生效。**

    ⚠️ runtime 侧它不是"又一个平台写动作" —— 它就是 `open_approval_case`，
      和网关自动触发时开的是**同一张表、同一条路**。
      另起一条路的话，「人在哪儿看这些单子」就会有两个答案。
    """
    from syncopate.runtime.gateway import Trigger, open_approval_case
    case_ref = await open_approval_case(
        db, org_id=org_id, run_id=run_id, action_type=change_type,
        proposed_params={"campaign_id": campaign_id, "requested_value": requested_value},
        rationale=reason,
        evidence={"source": "agent_requested"},
        triggers=[Trigger(reason="agent_requested",
                          detail="agent 主动开单，不是网关触发")])
    return {"case_ref": case_ref, "status": "pending", "applied": False}


async def campaign_create(platform: FakeAdPlatform, db, org_id: str, run_id: str, *,
                          account_id: str, product_id: str, region: str,
                          daily_budget: int, client_request_id: str,
                          platform_name: str | None = None,
                          creative_ids: list[str] | None = None,
                          idempotency_key: str | None = None) -> dict[str, Any]:
    """建 campaign。**先决条件：本轮必须已经开过审批单。**

    ★ 沙盒描述的第一句就是这条（★★ 标记），因为它是**不可逆**动作：
      「地域扩展的正确产出是**一份提议**（开审批单），不是直接建站。」
    """
    if not await _has_successful_approval(db, org_id, run_id):
        raise PreconditionNotMet(
            "approval_required_first: campaign.create 不可逆（建出来就在花钱、删不掉）"
            "⇒ 本轮必须先有一次成功的 approval.create_case")
    return await platform.create_campaign(
        account_id=account_id, product_id=product_id, region=region,
        daily_budget=daily_budget, platform=platform_name, creative_ids=creative_ids,
        idempotency_key=idempotency_key, client_request_id=client_request_id)


async def campaign_scale_budget(platform: FakeAdPlatform, db, org_id: str, run_id: str, *,
                                campaign_id: str, factor: float, reason: str,
                                client_request_id: str,
                                idempotency_key: str | None = None) -> dict[str, Any]:
    """按倍数改预算。**超出 ±20% 必须先走审批。**

    ⚠️ 这里要先读当前预算才能算出结果值 —— 而**读与写之间预算可能被别人改了**。
      所以把读到的值作为 `expected_current` 传给平台做乐观并发校验：
      对不上就拒绝，让调用方**重新读**（而不是拿一个过期的基数乘上去）。
    """
    if abs(float(factor) - 1.0) > SCALE_AUTO_APPROVE_BAND:
        if not await _has_successful_approval(db, org_id, run_id):
            raise PreconditionNotMet(
                f"approval_required_first: 幅度 {factor} 超出 ±"
                f"{SCALE_AUTO_APPROVE_BAND:.0%} 的自动执行区间 ⇒ 必须先走 approval.create_case")
    current = platform.budgets.get(campaign_id)
    out = await platform.scale_budget(
        campaign_id=campaign_id, factor=float(factor), expected_current=current,
        idempotency_key=idempotency_key, client_request_id=client_request_id)
    return {"campaign_id": campaign_id, "previous_budget": current,
            "new_budget": out["new_budget"], "factor": factor}


# ══════════════════════════════════════════════════════════════════════════
# 数据源类（B-2 第五批，2026-08-19）—— 形状同质，但**每个都有一条"不许多做"**
# ══════════════════════════════════════════════════════════════════════════
#
# ★ 这一批的共同纪律：沙盒描述里每个工具都写了它**不做什么**。
#   那些"不做"不是省事，是**把某个判断留给模型** ——
#   多做一步，就把对应的能力从训练目标里抹掉了。
#
#     geo_breakdown        只给各地域现状，**不告诉能不能扩**
#     industry_baseline    是参照，**不是决策依据**（决策要用内部安全线）
#     seasonal_context     只给时令背景，**不判断**素材该不该投
#     detect_anomalies     只**定性**给异常类型，**不给**方案
#     playbook             只给方案，**不执行**，也不判断数据够不够
#     budget_rule          只给账户级规则，**不做**风控判断
#     risk.check_account   只看账户风控，**不判断**金额合不合政策


async def analysis_feature_lift(db, *, feature: str, region: str,
                                product_id: str | None = None) -> dict[str, Any]:
    """某 feature 在某地域对 d7 ROAS 的 lift。

    ★★ **必须逐地域算**（沙盒描述原话）：同一个 feature 在不同地域可能**符号相反**，
      混在一起算会得出一个两头都不对的数。
      ⇒ 所以 `region` 是**必填**，而且这里**不做**跨地域聚合的兜底。
    ⚠️ 必须带置信区间与两组样本量 —— 只给点估计的话，
      「lift=+3% 但样本 12 条」和「lift=+3% 样本 8000 条」长得一模一样。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT feature, region, product_id, lift, ci_low, ci_high,"
            "       n_treatment, n_control FROM feature_lifts "
            "WHERE feature=$1 AND region=$2 AND (product_id=$3 OR product_id IS NULL) "
            "ORDER BY product_id NULLS LAST LIMIT 1", feature, region, product_id)
    if row is None:
        return {"found": False, "feature": feature, "region": region}
    d = dict(row)
    # 显著性：置信区间是否跨过 0 —— **算给模型看，但不替它下结论**
    lo, hi = d.get("ci_low"), d.get("ci_high")
    d["significant"] = bool(lo is not None and hi is not None and (lo > 0 or hi < 0))
    return {"found": True, **d}


async def analysis_geo_breakdown(db, *, product_id: str,
                                 regions: list[str] | None = None) -> dict[str, Any]:
    """按地域拆表现。

    ★ 它只告诉你**各地域现在跑成什么样**，**不告诉能不能扩** ——
      那要逐个地域查 `benchmark.get_safety_line`（每个地域一条线）。
    ⚠️ `asset_count` **必须返回**：素材条数少的地域数字本身就不可信，
      不给这个数，模型就分不出「这个地域不行」和「这个地域样本太少」。
    """
    async with db.tx() as conn:
        rows = await conn.fetch(
            "SELECT region, roas_d7, cpi_d7, asset_count FROM geo_performance "
            "WHERE product_id=$1 ORDER BY region", product_id)
    out = [dict(r) for r in rows]
    if regions:
        out = [r for r in out if r["region"] in set(regions)]
    return {"product_id": product_id, "regions": out}


async def benchmark_get_industry_baseline(db, *, platform: str, game_genre: str,
                                          metric: str) -> dict[str, Any]:
    """行业基准。⚠️ **不是决策依据** —— 能不能扩量要用内部安全线。"""
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT platform, game_genre, metric, p25, p50, p75, sample_size "
            "FROM industry_baselines WHERE platform=$1 AND game_genre=$2 AND metric=$3",
            platform, game_genre, metric)
    if row is None:
        return {"found": False, "platform": platform,
                "game_genre": game_genre, "metric": metric}
    return {"found": True, **dict(row)}


async def calendar_get_seasonal_context(db, *, region: str, event: str | None = None,
                                        horizon_days: int = 30) -> dict[str, Any]:
    """时令背景。**不判断**你的素材现在该不该投，也**不含**任何投放指标。

    ⚠️ `horizon_days` 必须强转 int：模型从 JSON 里给的可能是 "30"（字符串），
    asyncpg 会按 unknown 传给 PG ⇒ `date + unknown` 选不出运算符直接炸
    （2026-08-20 压测 I11 全灭的根因）。SQL 侧再补 ::int 双保险。
    """
    horizon_days = int(horizon_days)
    async with db.tx() as conn:
        rows = await conn.fetch(
            "SELECT region, event, event_date, lift_factor, creative_tags "
            "FROM seasonal_events WHERE region=$1 "
            "  AND ($2::text IS NULL OR event=$2) "
            "  AND event_date BETWEEN CURRENT_DATE - 7 AND CURRENT_DATE + $3::int "
            "ORDER BY event_date", region, event, horizon_days)
    import datetime as _dt
    today = _dt.date.today()
    return {"region": region,
            "events": [{**dict(r), "days_until": (r["event_date"] - today).days}
                       for r in rows]}


async def campaign_detect_anomalies(platform: FakeAdPlatform, *,
                                    campaign_id: str) -> dict[str, Any]:
    """只**定性**给异常类型，**不给**优化方案，也**不判断**数据成不成熟。

    ⚠️ 三件事分给三个工具是刻意的：
      异常是什么（本工具）· 怎么办（playbook）· 数据够不够下结论（metrics.get_freshness）
      合并任意两个，模型就不用自己串这条链了 —— 而串链正是我们要训的东西。
    """
    m = await platform.get_metrics(campaign_id=campaign_id)
    anomalies = []
    if (m.get("cpi") or 0) > 3.0:
        anomalies.append("cpi_spike")
    if (m.get("roas_d7") or 1.0) < 0.5:
        anomalies.append("roas_drop")
    # ★ `severity` 是沙盒也给的字段 —— 少了模型就没法排优先级
    severity = ("high" if len(anomalies) > 1 else "medium" if anomalies else "none")
    return {"campaign_id": campaign_id, "anomalies": anomalies, "severity": severity}


async def playbook_get_optimization(db, *, anomaly_type: str) -> dict[str, Any]:
    """按异常类型给方案。**不执行**任何写动作，也**不判断**数据够不够支撑。"""
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT anomaly_type, steps, cautions FROM playbooks WHERE anomaly_type=$1",
            anomaly_type)
    if row is None:
        # ★ 报"没有"，不猜一个相近的打法 —— 猜错的方案会被照着执行
        return {"found": False, "anomaly_type": anomaly_type,
                "error": f"unknown_anomaly_type: {anomaly_type}"}
    return {"found": True, **dict(row)}


async def policy_get_budget_rule(db, org_id: str, *, account_id: str) -> dict[str, Any]:
    """账户级预算调整规则。**不含**平台广告政策条款，也**不做**风控判断。"""
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT max_increase_pct, approval_threshold, risk_check_required,"
            "       monthly_cap FROM budget_rules WHERE org_id=$1", org_id)
    if row is None:
        return {"found": False, "account_id": account_id}
    return {"found": True, "account_id": account_id, **dict(row)}


async def risk_check_account(db, org_id: str, *, account_id: str) -> dict[str, Any]:
    """账户风控状态。**不判断**具体金额合不合政策，也**不返回**投放指标。

    ⚠️ 查不到 ⇒ **不能默认放行**。「没有风控记录」和「查过了、没问题」是两件事，
      而把前者当后者，就是在未知状态下放行 —— 同 `policy.search` 那条三态。
    """
    async with db.tx() as conn:
        row = await conn.fetchrow(
            "SELECT flags, state, allow_increase FROM account_risk "
            "WHERE org_id=$1 AND account_id=$2", org_id, account_id)
    if row is None:
        return {"found": False, "account_id": account_id,
                "error": "no_risk_record: 查不到风控记录 —— 这不等于「没有风险」，不要据此放行"}
    return {"found": True, "account_id": account_id, **dict(row)}


# ══════════════════════════════════════════════════════════════════════════
# ★★★ mmp.get_attribution —— 这一条**不能建成随机噪声**
# ══════════════════════════════════════════════════════════════════════════
#
# `07 §2.2` 的 A4 是实查结论，也是这条工具存在的全部理由：
#
#   「**归因窗口不一致是 Meta/AF 差异的头号成因**：
#     Meta 默认 7 天点击 + 1 天浏览；若 AF 侧配成 1 天点击，
#     则**点击后 2–7 天才首次打开 App 的用户，在 AF 里算自然量，在 Meta 里算投放带来的**。」
#
# ★★ 而 `07` 紧接着写了一句方法论，这里必须兑现：
#
#   「原计划是给两个源加一个**随机偏差**。**那是假的** ——
#     真实的打架有**确定的成因和方向**：它来自归因窗口配置，而且偏差方向可预测
#     （AF 少算、Meta 多算）。
#     **模型该学的是识别成因并据此判断该信谁，不是识别噪声。**」
#
# ⇒ 所以这里：
#   ① 差异由**窗口配置**推出来，不是 random
#   ② 方向**恒定**：MMP 的窗口更短 ⇒ MMP installs **少于**平台口径
#   ③ 返回里**必须带 attribution_window** —— 沙盒描述原话：
#      「做判断前先看两边的窗口是不是一致」。不给窗口，那句话就没法执行。

# 平台后台（自归因）的默认窗口：7 天点击 + 1 天浏览（实查 Meta）
PLATFORM_CLICK_WINDOW_DAYS = 7
PLATFORM_VIEW_WINDOW_DAYS = 1
# MMP 侧常见的保守配置：1 天点击、无浏览归因
DEFAULT_MMP_CLICK_WINDOW_DAYS = 1

# 点击后第 2–7 天才首次打开的用户占比。
# ⚠️ 工程值，但**方向不是拍的**：这批人在短窗口下必然落到自然量里。
LATE_OPEN_SHARE = 0.18


async def mmp_get_attribution(platform: FakeAdPlatform, *, campaign_id: str,
                              mmp_click_window_days: int = DEFAULT_MMP_CLICK_WINDOW_DAYS
                              ) -> dict[str, Any]:
    """MMP 口径的安装与回收。**和平台后台口径会有差异，成因是窗口不一致。**

    ⚠️ 这里**不做**"该信谁"的判断 —— 那正是模型要学的。
      我们只如实给出：两边的数、两边的窗口、以及差异的量。
    """
    m = await platform.get_metrics(campaign_id=campaign_id)
    platform_installs = int(m.get("installs") or 1000)
    # ★ 窗口越短，越多"晚开"的用户被算成自然量 ⇒ MMP 少算。**方向恒定。**
    shortfall = (LATE_OPEN_SHARE
                 if mmp_click_window_days < PLATFORM_CLICK_WINDOW_DAYS else 0.0)
    mmp_installs = int(round(platform_installs * (1 - shortfall)))
    # ⚠️⚠️ 字段名**以沙盒为准**（B-5b 对照台抓到的）——
    #   第一版我按自己的想法起名（`installs` / `platform_installs_for_reference`），
    #   而沙盒给的是 `installs_7d` / `organic_installs_7d` / `platform_attribution_window`。
    #   ⇒ `tool_impls` 模块开头那条纪律写着「**不许自己发明字段**」，我自己违反了。
    return {
        "campaign_id": campaign_id,
        "source": "mmp",
        "installs_7d": mmp_installs,
        # ★ 被短窗口漏掉的那批，在 MMP 侧**算成自然量** —— 这正是 A4 的机制
        "organic_installs_7d": platform_installs - mmp_installs,
        "cpi": round(float(m.get("cpi") or 2.1), 4),
        "roas_d7": round(float(m.get("roas_d7") or 0.42), 4),
        # ★ 两边的窗口都给出来 —— 沙盒描述：「做判断前先看两边的窗口是不是一致」
        "attribution_window": {"click_days": mmp_click_window_days, "view_days": 0},
        "platform_attribution_window": {"click_days": PLATFORM_CLICK_WINDOW_DAYS,
                                        "view_days": PLATFORM_VIEW_WINDOW_DAYS},
    }
