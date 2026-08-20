"""M9.4 · 假广告平台：可注入故障的外部系统替身。

★ 为什么不接真 Meta API（2026-08-14 定）

真 API 会**真的烧钱**，而 M9 要验的是**我们这侧的正确性**。真接入留到 M10 影子模式。
但假平台必须**如实建模真实世界的坏脾气**，否则 runtime 的降级路径就是假的。

★★★ 一条从沙盒继承过来、这里必须兑现的纪律：**超时分两种**

    请求没发出去    重试是安全的
    到了但回包丢了  **重试 = 重复扣款**

⚠️ 而模型/客户端看到的现象**一模一样**（都是超时）。沙盒里这条靠
`side_effect_applied` 建模（`EnvSnapshot.failures` 的注释：
"构造不出后者，模型学到的就是'超时=没做成'，那是错的"）。
这里必须同样建出来 —— 而且**错误文本要逐字相同**，否则 runtime 就能靠文本区分，
那它学到的东西在真平台上不成立。

⇒ 区分只能靠**幂等键**：重试时带同一个键，平台告诉你"这个键我见过"。
这就是三层幂等第三层存在的全部理由。

★ 实查过的事实（记在 `core/tool_registry.py`）：**Meta Marketing API 本身没有幂等机制**。
所以这里的 `_seen_keys` 是我们**希望平台有**的东西；真接入时这层保证得由
`runtime/db.py` 的 `record_tool_call` 兑现。假平台把它建出来，是为了让
runtime 的代码路径和真实接入时一致。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class PlatformError(Exception):
    """平台侧错误。`retriable` 是**平台说的**，不是我们猜的。

    ★ `code` / `subcode` 照真实 API 的形状给（`07 §2.1` 实查）：
    Meta 的改动频次超限是 `613` + 子码 `1487632`，限流是 HTTP 429 带 `retry_after`。
    ⚠️ **子码不能省** —— 真实世界里 `613` 是一大类错误的总称，
    只看主码分不出"改太频繁"和"参数不合法"，而这两种的正确应对完全相反。
    """

    def __init__(self, message: str, *, code: str, retriable: bool,
                 subcode: str | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retriable = retriable
        self.subcode = subcode
        self.retry_after = retry_after


# ★ 超时的错误文本**只有一份**。两种超时（没发出去 / 回包丢了）用同一句话，
# 因为真实世界里客户端确实分不出来。谁想分辨，只能靠幂等键回查。
TIMEOUT_MESSAGE = "upstream_timeout: 平台在 30s 内没有响应"


@dataclass
class FaultPlan:
    """故障剧本。**由调用方声明，不随机** —— 和沙盒 `EnvSnapshot.failures` 同一条纪律：
    随机故障会让"这次失败"和"模型做得不同"混在一起，压测的归因就糊了。"""

    # 第 n 次调用超时（1-indexed）
    timeout_at: set[int] = field(default_factory=set)
    # ★ 超时的那次，副作用**到底发生了没有**。这是两种超时的唯一区别。
    side_effect_applied: bool = False
    # ★ 数据有多老（天）。默认 7 = 已收敛；调小就能造出「D7 未收敛」的局面。
    data_age_days: int = 7
    rate_limit_at: set[int] = field(default_factory=set)
    server_error_at: set[int] = field(default_factory=set)
    latency_seconds: float = 0.0


# ══════════════════════════════════════════════════════════════════════════
# 真实 API 的两条硬机制（B-1，2026-08-19）—— 实查依据见 `07 §2.1`
# ══════════════════════════════════════════════════════════════════════════
#
# ★★ 为什么这两条必须建出来，而不是"以后再说"
#
# `07 §2.1` 的结论原话：**M4 + M5 合起来是最要命的组合** ——
# 平台**没有幂等机制**，而改动次数**有硬上限**。
# ⇒ 一次超时后盲目重试，可能同时造成「多改一次预算」和「耗尽当小时配额」。
# ⇒ 如果假平台不建这两条，runtime 的重试策略在真实世界里是**没被验过**的。
#
# ⚠️ 数值全部来自实查，不是拍的：
#   BUC 积分制    读 1 分 / 写 3 分；开发档 60 分、标准档 9000 分；衰减 300 秒；
#                 **按广告账户共享额度**（不是按 campaign）
#   改动频次      每个 ad set **每小时最多 4 次**预算改动；
#                 超了报 `613` / 子码 `1487632`，并**封禁该 ad set 一小时**
READ_POINTS = 1
WRITE_POINTS = 3
BUC_WINDOW_SECONDS = 300.0
BUC_STANDARD_TIER = 9_000
BUC_DEVELOPMENT_TIER = 60

BUDGET_CHANGES_PER_HOUR = 4
BUDGET_CHANGE_WINDOW_SECONDS = 3600.0
META_TOO_MANY_CHANGES_CODE = "613"
META_TOO_MANY_CHANGES_SUBCODE = "1487632"

# ── 分页（实查 P1-3）───────────────────────────────────────────────────────
#
# ⚠️ **要的比给的多，平台只给上限那么多，而且不报错。**
# 真实 API 就是这么干的 —— 这正是"以为拿到全部了"这个错误的来源：
# agent 传 limit=1000 拿回 25 条，**不看 paging 就会以为账户里只有 25 条**。
# ⇒ 建出来才能让模型学会"看 paging，不看 len(data)"。
MAX_PAGE_SIZE = 25

# ── 异步任务（实查 P1-1 / P1-2）───────────────────────────────────────────
#
# ★★ `07 §P1-2` 原话：`poll_review` 现在是阻塞等待，不是轮询
#   ——「**这把"什么时候该查"这个决策从模型手里拿走了**」。
# ⇒ 生产侧必须是：提交立刻返回 id + `pending`，由 agent 自己决定何时再查。
# ⚠️ 而且**每次查都要扣 BUC 积分** ⇒ "死循环狂查"会自然地把配额烧掉，
#   不需要我们额外加一条"不许频繁轮询"的规则。**代价内建，不靠规训。**
DEFAULT_JOB_SETTLE_SECONDS = 480.0     # 素材审核：真实 2–4 小时，取下界的保守值


@dataclass
class _Budget:
    """BUC 积分桶。**按账户**共享，不是按 campaign（实查 M6）。"""

    limit: int
    window_seconds: float
    spent: int = 0
    window_started: float = 0.0

    def charge(self, points: int, now: float) -> float | None:
        """扣分。够扣返回 None；不够返回还要等多少秒。"""
        if now - self.window_started >= self.window_seconds:
            self.spent = 0
            self.window_started = now
        if self.spent + points > self.limit:
            return self.window_seconds - (now - self.window_started)
        self.spent += points
        return None


@dataclass
class FakeAdPlatform:
    """内存态的假平台。**不是 mock**：它有真实的状态，写动作真的改数字。"""

    budgets: dict[str, int] = field(default_factory=dict)
    faults: FaultPlan = field(default_factory=FaultPlan)
    calls: int = 0
    # 平台侧记住见过的幂等键 → 原结果。真 Meta 没有这个，见模块 docstring。
    _seen_keys: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ★ 时钟可注入：默认真实时间，测试里换成可控的。
    # ⚠️ 不用真实 sleep 去测限流窗口 —— 那会让测试慢且偶发红，
    #   而**一个会偶发红的测试就是一把不可信的尺子**（记录在案）。
    clock: Callable[[], float] = time.monotonic
    # BUC 档位。默认标准档 —— 现有编排一条 run 只花几分，够用；
    # 压测（B-6）要造限流场景时换成开发档 60 分。
    buc_limit: int = BUC_STANDARD_TIER
    _buc: _Budget | None = None
    # campaign_id → 这一小时里的改动时间戳
    _budget_changes: dict[str, list[float]] = field(default_factory=dict)
    # campaign_id → 封禁到什么时候（实查 M5：超限会封禁该 ad set 一小时）
    _blocked_until: dict[str, float] = field(default_factory=dict)
    # 账户里的 campaign 记录（分页 / 显式字段用）。key = campaign_id
    campaigns: dict[str, dict[str, Any]] = field(default_factory=dict)
    # demo 的默认账户（get_metrics 回传，模型据此查风控）
    account_id: str = "ACC_DEMO"
    # 异步任务：job_id → {status, settle_at, result, error}
    _jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _job_seq: int = 0
    # 素材库：asset_id → {campaign_id, name, asset_type, tags, metrics, review_job}
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    _asset_seq: int = 0
    _campaign_seq: int = 0

    async def create_campaign(self, *, account_id: str, product_id: str, region: str,
                              daily_budget: int, platform: str | None = None,
                              creative_ids: list[str] | None = None,
                              idempotency_key: str | None = None,
                              client_request_id: str | None = None) -> dict[str, Any]:
        """建一条 campaign。**不可逆：建出来就开始花钱，删不掉。**

        ⚠️ 返回**只表示提交成功，不代表已开始跑量**（沙盒描述原话）——
          所以这里给的是 `status: submitted`，不是 `running`。
          给 `running` 就等于替模型断言了一件平台还没做完的事。
        """
        self.calls += 1
        if idempotency_key and idempotency_key in self._seen_keys:
            return {**self._seen_keys[idempotency_key], "deduped_by_platform": True}
        self._charge(WRITE_POINTS)
        self._campaign_seq += 1
        cid = f"CMP_N{self._campaign_seq:04d}"
        self.campaigns[cid] = {"name": f"{product_id}-{region}", "status": "SUBMITTED",
                               "daily_budget": daily_budget, "product_id": product_id,
                               "region": region, "account_id": account_id}
        self.budgets[cid] = daily_budget
        out = {"campaign_id": cid, "status": "submitted", "daily_budget": daily_budget}
        if idempotency_key:
            self._seen_keys[idempotency_key] = out
        return out

    async def scale_budget(self, *, campaign_id: str, factor: float,
                           expected_current: int | None = None,
                           idempotency_key: str | None = None,
                           client_request_id: str | None = None) -> dict[str, Any]:
        """按倍数改预算。**带乐观并发校验。**

        ★★ `factor` 是**相对量** ⇒ 必须"读当前 → 乘 factor → 写回"，
          而两次操作之间预算可能被别人改了（另一条 run、运营手动、自动规则）。
          ⇒ 调用方把读到的值作为 `expected_current` 传回来；对不上就**拒绝**。

        ⚠️ 不做这个校验的后果不是报错，是**乘错基数**：
          以为在 1000 上提 20%，实际在别人刚改成的 5000 上提 —— 悄悄多花 4 倍。
        ⚠️ 冲突**不可重试**（`retriable=False`）：基数已经变了，
          重试只会拿同一个过期的期望值再撞一次。正确应对是**重新读**。
        """
        current = self.budgets.get(campaign_id)
        if current is None:
            raise PlatformError(f"unknown_campaign: {campaign_id}",
                                code="400", retriable=False)
        if expected_current is not None and expected_current != current:
            raise PlatformError(
                f"concurrent_modification: 期望 {expected_current}，实际 {current} "
                f"—— 预算在你读到之后被改过了，请重新读再算",
                code="409", retriable=False)
        return await self.update_budget(
            campaign_id=campaign_id, new_budget=int(round(current * factor)),
            idempotency_key=idempotency_key, client_request_id=client_request_id)

    # ── 素材（B-2 第三批）─────────────────────────────────────────────
    async def upload_creative(self, *, campaign_id: str, creative_name: str,
                              asset_type: str, duration_seconds: float | None = None,
                              idempotency_key: str | None = None,
                              client_request_id: str | None = None,
                              review_seconds: float | None = None) -> dict[str, Any]:
        """上传素材 ⇒ **进审核队列**，立刻返回 asset_id + 一个审核任务。

        ★★ **上传成功 ≠ 审核通过**（沙盒描述里的原话）。
          所以这里返回的是 `asset_id` + `review_job_id`，**不含审核结论** ——
          给一个结论字段就等于把"要不要等审核"这个决策从模型手里拿走了。
        ⚠️ 真实审核 2–4 小时；这里用异步任务表达，默认 480 秒（取下界的保守值）。
        """
        self.calls += 1
        if idempotency_key and idempotency_key in self._seen_keys:
            return {**self._seen_keys[idempotency_key], "deduped_by_platform": True}
        self._charge(WRITE_POINTS)
        self._asset_seq += 1
        asset_id = f"AST_{self._asset_seq:04d}"
        wait = DEFAULT_JOB_SETTLE_SECONDS if review_seconds is None else review_seconds
        job_id = self._new_job("review", settle_after=wait,
                               result={"asset_id": asset_id, "review": "approved"})
        self.assets[asset_id] = {"asset_id": asset_id, "campaign_id": campaign_id,
                                 "name": creative_name, "asset_type": asset_type,
                                 "duration_seconds": duration_seconds,
                                 "review_job_id": job_id,
                                 "tags": {}, "metrics": {}}
        out = {"asset_id": asset_id, "review_job_id": job_id, "status": "in_review"}
        if idempotency_key:
            self._seen_keys[idempotency_key] = out
        return out

    # ── 异步任务（实查 P1-1 / P1-2）─────────────────────────────────────
    def _new_job(self, kind: str, *, settle_after: float,
                 result: dict[str, Any] | None = None,
                 error: str | None = None) -> str:
        self._job_seq += 1
        job_id = f"job_{kind}_{self._job_seq}"
        self._jobs[job_id] = {"kind": kind, "settle_at": self.clock() + settle_after,
                              "result": result or {}, "error": error}
        return job_id

    async def get_job(self, *, job_id: str) -> dict[str, Any]:
        """查一个异步任务的状态。

        ★ **不阻塞** —— 没到点就如实返回 `pending`，由调用方决定何时再查。
        ⚠️ 每次查都扣积分（读 1 分）：狂查会把配额烧掉。**代价内建，不靠规训。**
        ⚠️ 认不出 job_id 就报错，**不猜**（不要返回一个"看起来在跑"的 pending —
          那会让 agent 永远等一个不存在的任务）。
        """
        self.calls += 1
        self._charge(READ_POINTS)
        job = self._jobs.get(job_id)
        if job is None:
            raise PlatformError(f"unknown_job: {job_id}", code="400", retriable=False)
        if self.clock() < job["settle_at"]:
            return {"job_id": job_id, "status": "pending"}
        if job["error"]:
            return {"job_id": job_id, "status": "failed", "error": job["error"]}
        return {"job_id": job_id, "status": "succeeded", "result": job["result"]}

    # ── 分页 + 显式字段（实查 M3 / P1-3）────────────────────────────────
    async def list_campaigns(self, *, account_id: str, fields: list[str],
                             after: str | None = None,
                             limit: int = MAX_PAGE_SIZE) -> dict[str, Any]:
        """列 campaign。**必须显式说要哪些字段**（实查 M3）。

        ⚠️ `fields` 不给就报错，**不给一个"默认全给"** ——
          真实 API 不会替你猜，而"本地能跑、上线拿不到字段"是最难查的一类。
        ⚠️ 要的比上限多，只给上限那么多，**而且不报错**（实查 P1-3）：
          这正是"以为拿到全部了"的来源 ⇒ 判据是**看 paging，不是看 len(data)**。
        """
        self.calls += 1
        self._charge(READ_POINTS)
        if not fields:
            raise PlatformError("missing_fields: 必须显式指定 fields",
                                code="400", retriable=False)
        keys = sorted(self.campaigns)
        unknown = [f for f in fields if f not in {"id", "name", "daily_budget", "status"}]
        if unknown:
            raise PlatformError(f"unknown_fields: {unknown}", code="400", retriable=False)
        start = keys.index(after) + 1 if after in keys else 0
        page = keys[start:start + min(limit, MAX_PAGE_SIZE)]
        data = [{f: self._campaign_field(cid, f) for f in fields} for cid in page]
        has_more = start + len(page) < len(keys)
        return {"data": data,
                "paging": {"cursors": {"after": page[-1] if page else None},
                           "has_next": has_more}}

    def _campaign_field(self, campaign_id: str, field_name: str) -> Any:
        row = self.campaigns.get(campaign_id, {})
        if field_name == "id":
            return campaign_id
        if field_name == "daily_budget":
            # ★ read-after-write：写过的值要读得到（`07 §P0-1` 实测过的坑）
            return self.budgets.get(campaign_id, row.get("daily_budget", 50_000))
        return row.get(field_name)

    # ── 两条硬机制 ───────────────────────────────────────────────────────
    def _charge(self, points: int) -> None:
        """BUC 积分制（实查 M6）。耗尽 ⇒ 429 + `retry_after`，**可重试**。"""
        if self._buc is None:
            self._buc = _Budget(limit=self.buc_limit,
                                window_seconds=BUC_WINDOW_SECONDS,
                                window_started=self.clock())
        wait = self._buc.charge(points, self.clock())
        if wait is not None:
            raise PlatformError(
                f"rate_limited: 账户 BUC 额度耗尽，{wait:.0f}s 后重试",
                code="429", retriable=True, retry_after=wait)

    def _check_change_frequency(self, campaign_id: str) -> None:
        """每小时最多改 4 次预算（实查 M5）。超了 613/1487632 并封禁一小时。

        ⚠️ **这条不可重试**：`retriable=False`。它和限流长得像但性质相反 ——
        限流是"等一下再来"，这条是"你已经被封了，重试只会更糟"。
        ⇒ 真实世界里把它当限流去重试，是把一小时的封禁续成两小时。
        """
        now = self.clock()
        until = self._blocked_until.get(campaign_id)
        if until is not None and now < until:
            raise PlatformError(
                f"too_many_budget_changes: {campaign_id} 已被封禁至 {until - now:.0f}s 后",
                code=META_TOO_MANY_CHANGES_CODE, subcode=META_TOO_MANY_CHANGES_SUBCODE,
                retriable=False)
        hits = [t for t in self._budget_changes.get(campaign_id, [])
                if now - t < BUDGET_CHANGE_WINDOW_SECONDS]
        if len(hits) >= BUDGET_CHANGES_PER_HOUR:
            self._blocked_until[campaign_id] = now + BUDGET_CHANGE_WINDOW_SECONDS
            self._budget_changes[campaign_id] = hits
            raise PlatformError(
                f"too_many_budget_changes: {campaign_id} 一小时内已改 {len(hits)} 次",
                code=META_TOO_MANY_CHANGES_CODE, subcode=META_TOO_MANY_CHANGES_SUBCODE,
                retriable=False)
        hits.append(now)
        self._budget_changes[campaign_id] = hits

    async def update_budget(self, *, campaign_id: str, new_budget: int,
                            idempotency_key: str | None = None,
                            client_request_id: str | None = None) -> dict[str, Any]:
        """
        ⚠️⚠️ `client_request_id` **必须接住**（2026-08-19 被工具对齐判据抓到）。

        沙盒的 `campaign.update_budget` 把它列为**必填参数** —— 模型是被这样训出来的。
        而 runtime 这侧原本不接它，后果有两层：

            ① `invoke(**arguments)` 直接 TypeError —— 模型一按训练学的方式调就炸
            ② 更隐蔽的那层：`derive_idempotency_key` 是对**全部参数**做哈希的。
               没有这个参数时，「用户有意连续两次把预算调成同一个值」
               会推出**同一个键** ⇒ **第二次被当成重放挡掉，而且看起来像成功。**

        ⇒ 这个参数正是为区分「有意的第二次」和「重放」而存在的。
          它进 arguments、进哈希，两侧的幂等语义才真的一致。
        """
        self.calls += 1
        n = self.calls

        if self.faults.latency_seconds:
            await asyncio.sleep(self.faults.latency_seconds)

        # 幂等键命中 ⇒ 直接返回原结果，**不重复执行**
        # ⚠️ 这一步必须在扣分和频次检查**之前**：重放没有真的改动世界，
        #   既不该消耗 BUC 配额，也不该算作"这一小时又改了一次"。
        #   放在后面的话，一次重试就会白白吃掉一次改动额度 —— 而额度只有 4 次。
        if idempotency_key and idempotency_key in self._seen_keys:
            return {**self._seen_keys[idempotency_key], "deduped_by_platform": True}

        # ★ 真实 API 的两条硬机制。⚠️ 频次检查在扣分**之后**：
        #   真实世界里请求已经打过去了才会被判定超频，配额是照扣的。
        self._charge(WRITE_POINTS)
        self._check_change_frequency(campaign_id)

        if n in self.faults.rate_limit_at:
            raise PlatformError("rate_limited: 请求过于频繁", code="rate_limited", retriable=True)
        if n in self.faults.server_error_at:
            raise PlatformError("server_error: 平台内部错误", code="server_error", retriable=True)

        if n in self.faults.timeout_at:
            # ★★★ 关键：副作用**先按剧本决定要不要真的发生**，再抛同一句超时。
            # side_effect_applied=True 就是"到了但回包丢了" —— 这时重试会重复扣款。
            if self.faults.side_effect_applied:
                self.budgets[campaign_id] = new_budget
                if idempotency_key:
                    self._seen_keys[idempotency_key] = {
                        "campaign_id": campaign_id, "new_budget": new_budget}
            raise PlatformError(TIMEOUT_MESSAGE, code="timeout", retriable=True)

        self.budgets[campaign_id] = new_budget
        # ★ 异步任务也在这里生成：效果**只有一处实现**，
        #   `submit_budget_change` 是同一条路的"不等结果"入口，不是第二份实现。
        job_id = self._new_job("budget", settle_after=0.0,
                               result={"campaign_id": campaign_id, "new_budget": new_budget})
        result = {"campaign_id": campaign_id, "new_budget": new_budget, "change_id": job_id}
        if idempotency_key:
            self._seen_keys[idempotency_key] = result
        return result

    async def submit_budget_change(self, *, campaign_id: str, new_budget: int,
                                   idempotency_key: str | None = None,
                                   settle_after: float = 0.0,
                                   validate_only: bool = False) -> dict[str, Any]:
        """提交一次预算改动，**立刻返回 `change_id` + `pending`**（实查 P1-1）。

        ★ `validate_only` 是**真 API 提供的 dry-run**（实查 M7）：只校验不生效。
          它值得建出来 —— 有了它，"先验证再提交"才成为模型可以学的策略；
          没有它，模型面对不确定只能赌。

        ⚠️ 校验失败**不可重试**：参数不合法，重试一百次还是不合法。
          把它标成可重试会让 `ToolRuntime` 白白重试三次，还各扣一次积分。
        """
        if new_budget <= 0:
            raise PlatformError("invalid_parameter: new_budget 必须为正",
                                code="100", retriable=False)
        if validate_only:
            self.calls += 1
            self._charge(READ_POINTS)      # dry-run 按读计价：它不改世界
            return {"campaign_id": campaign_id, "valid": True, "validate_only": True}
        done = await self.update_budget(campaign_id=campaign_id, new_budget=new_budget,
                                        idempotency_key=idempotency_key)
        if settle_after:
            # 需要"真的要等一会儿"时，把任务的到点时间往后推
            self._jobs[done["change_id"]]["settle_at"] = self.clock() + settle_after
        return {"change_id": done["change_id"], "status": "pending",
                "campaign_id": campaign_id}

    @classmethod
    def from_fixture(cls, path: str = "data/demo/platform_state.json") -> "FakeAdPlatform":
        """从 fixture 建一个带 demo 数据的假平台。

        ★ 文件缺失 ⇒ **返回空平台并打一行告警**，不静默给默认值：
          "没有数据"和"有数据"必须能分辨（查不到时模型的正确行为是说查不到，
          而我们要知道那是环境空的还是模型不会）。
        """
        import json as _json
        import pathlib as _pl

        f = _pl.Path(path)
        if not f.is_file():
            print(f"[platform] ⚠️ 未找到 {path} ⇒ 空平台（agent 将查不到任何 campaign）",
                  flush=True)
            return cls()
        state = _json.loads(f.read_text(encoding="utf-8"))
        campaigns = {k: v for k, v in state.get("campaigns", {}).items()
                     if not k.startswith("_")}
        budgets = {k: v.get("daily_budget", 0) for k, v in campaigns.items()}
        print(f"[platform] 已加载 {len(campaigns)} 个 demo campaign（{path}）", flush=True)
        return cls(campaigns=campaigns, budgets=budgets,
                   account_id=state.get("account_id", "ACC_DEMO"))

    async def get_freshness(self, *, campaign_id: str) -> dict[str, Any]:
        """数据成熟度。★ **归因延迟是第一性约束**（设计 §0.3）：D7 才知对错，
        D1 数据极易被误当结论。

        真实世界里这个信号来自 MMP 的归因窗口，沙盒里由 case 声明；
        这里由剧本给（`data_age_days`），**不随机** —— 同失败注入那条纪律。
        """
        self.calls += 1
        self._charge(READ_POINTS)
        # ★ 逐 campaign 的数据年龄优先（训练侧每个 case 自己声明 data_age_days，
        #   runtime 若只有一个全局值，用户说"前天刚上的"也拿不到 immature ——
        #   B-5 那条：两侧不一致 ⇒ 训练时的最优策略在线上不成立）。
        row = self.campaigns.get(campaign_id, {})
        age = row.get("data_age_days", self.faults.data_age_days)
        maturity = "mature" if age >= 7 else ("partial" if age >= 3 else "immature")
        return {"campaign_id": campaign_id, "data_age_days": age, "maturity": maturity,
                "d7_available": age >= 7}

    async def get_metrics(self, *, campaign_id: str) -> dict[str, Any]:
        self.calls += 1
        self._charge(READ_POINTS)
        if self.faults.latency_seconds:
            await asyncio.sleep(self.faults.latency_seconds)
        if self.calls in self.faults.timeout_at:
            raise PlatformError(TIMEOUT_MESSAGE, code="timeout", retriable=True)
        # ⚠️⚠️ 字段名**以沙盒为准**（B-5b 的对照台 2026-08-19 抓到少了 9 个）——
        #   模型是照沙盒训的，按名字取数；少一个不会报错，只会让它**自己编一个**。
        row = self.campaigns.get(campaign_id, {})
        m = row.get("metrics", {})
        # ★ product_id / region / account_id 必须返回：模型要拿它们去查安全线、
        #   查风控、查地域表现。不给的话它只能**编一个**，然后查不到
        #   （2026-08-20 实测：查不到就一路 no_data，看起来像"模型不会"）。
        return {"campaign_id": campaign_id,
                "name": row.get("name", campaign_id),
                "status": row.get("status", "ACTIVE"),
                "platform": row.get("platform", "meta"),
                "game_genre": row.get("game_genre", "casual"),
                "product_id": row.get("product_id", "GAME_PUZZLE"),
                "region": row.get("region", "华东"),
                "account_id": row.get("account_id", self.account_id),
                "daily_budget": self.budgets.get(campaign_id, row.get("daily_budget", 50_000)),
                "spend_7d": m.get("spend_7d", 31_500),
                "impressions": m.get("impressions", 1_200_000),
                "installs_7d": m.get("installs_7d", 15_000),
                "frequency": m.get("frequency", 2.4),
                "ctr": m.get("ctr", 0.021),
                "cpi": m.get("cpi", 2.1),
                "roas_d7": m.get("roas_d7", 0.42)}
