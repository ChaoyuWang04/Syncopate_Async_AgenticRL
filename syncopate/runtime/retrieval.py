"""M9+ · Runtime 侧检索服务：三态契约 + 精确的生效期计算。

★★★ 这个模块存在的第一理由不是"能检索"，是**契约由 runtime 定义**

在它之前，检索的契约实际上是沙盒（`domains/adcampaign/corpus.py`）单方面定的：
阈值、`no_match` 的语义、取代关系怎么算 —— 而 runtime 这边什么都没有。
「沙盒是 runtime 的子集，且契约由 runtime 定义」这条纪律是**反着的**。
设计与取舍全文见 `docs/syncopate/12-rag-runtime-design.md`。

★★★ 三种状态，不是两种

    ok           查到了       hits 非空
    no_match     查了，确实没有  语义是「**没有政策限制这件事**」  ⇒ 可带 caveat 继续 / 转人工
    unavailable  查不了       语义是「**我们不知道有没有限制**」  ⇒ **绝对不能继续**

⚠️ 现象一模一样（都是"没拿到东西"），语义正好相反。合并成一个信号的话，
**一次故障看起来就是"没有政策限制" ⇒ agent 放行** —— 那正是压测场景④要抓的灾难。

★ 这和「超时的两种形态」是同一个形状（见 `runtime/platform.py`）：
超时那两种**在真实世界里就是分不开的**，所以错误文本必须逐字相同；
而检索这两种**我们完全分得开**（一个是返回 0 行，一个是连接失败）——
**分得开却不分，是自己制造了一个致命的歧义。**

★ 打分**直接复用沙盒那个函数**（`overlap_score`），不是图省事：
两边各写一份迟早会漂移，而漂移了没有任何东西会响。
⚠️ **但阈值不能沿用** —— 候选集不一样，操作点必然不同，见下面 RUNTIME_MATCH_THRESHOLD。
为什么不用 PG 全文：默认配置不懂中文，装 zhparser 等于引入一个
"版本变了检索结果就变"的依赖 —— 正是沙盒侧刻意避免的。
成立范围与改的判据见设计文档 §5（语料 > 10⁴ 条或 P95 > 100ms 时才换）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ★ 和沙盒**同一个打分函数**。别在这里复制一份实现 —— 两边各写一份迟早漂移，
#   而漂移了没有任何东西会响。
from syncopate.domains.adcampaign.corpus import overlap_score
from syncopate.runtime.db import Database

# ★★★ 阈值**不能沿用沙盒的 0.35** —— 这是实测出来的，不是偏好
#
# 打分函数两边是同一个，但**候选集不是**：
#
#     沙盒     每条 case 自带 1–2 篇、手写、构造上不会互撞      ⇒ 0.35 够用
#     runtime  整个语料库一起参与打分，篇数只会越来越多         ⇒ 撞得上
#
# ⇒ **相同的打分函数 + 不同的候选集 = 操作点必然不同。** 不是调参，是结构性的：
#   误召回概率随语料条数单调上升。
#
# 实测（`scripts/calibrate_runtime_retrieval.py`，种子语料 4 条）：
#
#     应命中 族内最低 0.667   应留空 最高 0.400（「量子计算加速广告投放」撞上
#                                              东南亚博彩条款，共享 投/放/广/告）
#     ⇒ 可用区间 (0.400, 0.667]，取中点
#
# ⚠️ **4 条语料 + 我自己写的 11 条查询，远不够定案**（同 corpus.py 当年那句）。
#   真实查询要等 M10 影子模式。**语料一变就要重跑那个脚本。**
RUNTIME_MATCH_THRESHOLD = 0.53


class RetrievalStatus(str, Enum):
    OK = "ok"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RetrievalResult:
    """★ `hits` 在 no_match 和 unavailable 下**都是空的** —— 区别在 `status`。

    调用方**不许**用 `if not result.hits` 来判断"没有政策限制"：
    那句话会把服务故障读成放行信号。要判断得看 `status`，
    所以这里刻意**不提供** `__bool__` / `__len__`，逼你去看状态。
    """

    status: RetrievalStatus
    hits: list[dict[str, Any]]
    query: str
    latency_ms: int
    error: str | None = None

    @property
    def usable(self) -> bool:
        """真的查到了可引用的东西。"""
        return self.status is RetrievalStatus.OK

    @property
    def blocks_decision(self) -> bool:
        """★ 这次检索是否**禁止**继续做决定。

        `no_match` 不禁止（确实没有相关政策，可以带 caveat 走）；
        `unavailable` 禁止（我们不知道有没有政策）。
        """
        return self.status is RetrievalStatus.UNAVAILABLE


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value: Any) -> datetime | None:
    """ISO 字符串 → datetime。

    ⚠️ asyncpg **不接受字符串喂给 timestamptz**（`$n::timestamptz` 那个 cast 也救不了，
    它按参数推断类型在客户端就编码失败了）。而语料是 JSON，时间天然是字符串
    ⇒ 转换必须发生在这一层，不能指望数据库。
    """
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _clause_text(row: dict[str, Any]) -> str:
    return " ".join(filter(None, [row.get("title"), row.get("body"),
                                  row.get("section_path")]))


def _insight_text(row: dict[str, Any]) -> str:
    bits = [row.get("claim") or ""]
    bits += [str(v) for v in (row.get("scope_json") or {}).values()]
    return " ".join(bits)


def _rank(rows: list[dict[str, Any]], query: str, *, text_of, top_k: int,
          key_field: str) -> list[dict[str, Any]]:
    """打分 + 阈值 + 截断。

    ★ 排序键带上 id：分数相同时按 id 排，避免行顺序影响结果 ——
    沙盒那边栽过一次「prompt 内容取决于 dict 插入顺序」。
    ★ **不做"总是返回 top-k"**：低于阈值就是没查到。否则「检索为空时不编答案」
    这项验收在构造上永远触发不了（同 BM25 被淘汰的理由：它是排序器不是判定器）。
    """
    scored = []
    for row in rows:
        score = overlap_score(query, text_of(row))
        if score >= RUNTIME_MATCH_THRESHOLD:
            scored.append((-round(score, 4), str(row[key_field]), row, round(score, 4)))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [{**row, "score": score} for _, _, row, score in scored[:top_k]]


class RetrievalService:
    """PG 里的语料 + 词法打分。**SQL 做过滤，Python 做打分**（设计文档 §5）。"""

    def __init__(self, db: Database, *, timeout_seconds: float = 2.0) -> None:
        self.db = db
        self.timeout_seconds = timeout_seconds

    # -- 半结构化：政策条款 ------------------------------------------------

    async def search_policy(self, *, org_id: str, query: str, platform: str | None = None,
                            region: str | None = None, top_k: int = 3,
                            now: datetime | None = None) -> RetrievalResult:
        """检索政策条款。

        ★ `expired` / `superseded_by` 是**算出来的，不是查出来的**（设计文档 §4）：
        生效期是 SQL 里的时间比较，取代关系是**从新条款反查**
        （新版本声明 `supersedes=<旧 id>`，不在旧条款上回写）。
        ⇒ **即使检索是模糊的，"哪一版现行"也是精确的。**
        §14「过期检出率」那项验收因此不依赖检索质量。
        """
        started = time.perf_counter()
        at = now or _now()
        try:
            async with self.db.tx() as conn:
                rows = [dict(r) for r in await conn.fetch(
                    """
                    SELECT clause_id, title, body, section_path, platform, region,
                           valid_from, valid_to, version, supersedes, source_doc,
                           (valid_to IS NOT NULL AND valid_to < $2) AS expired
                      FROM policy_clauses
                     WHERE (scope = 'global' OR scope = $1)
                       AND ($3::text IS NULL OR platform IS NULL OR platform = $3)
                       AND ($4::text IS NULL OR region   IS NULL OR region   = $4)
                    """, org_id, at, platform, region)]
                # 取代关系：一次查完整个作用域的反向映射，避免 N+1
                succ = {r["supersedes"]: r["clause_id"] for r in await conn.fetch(
                    "SELECT clause_id, supersedes FROM policy_clauses "
                    " WHERE (scope='global' OR scope=$1) AND supersedes IS NOT NULL", org_id)}
        except Exception as exc:                       # noqa: BLE001
            # ★★ 这里**绝不能**退化成 no_match。见模块 docstring。
            return RetrievalResult(
                status=RetrievalStatus.UNAVAILABLE, hits=[], query=query,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"retrieval_unavailable: {type(exc).__name__}: {str(exc)[:200]}")

        hits = _rank(rows, query, text_of=_clause_text, top_k=top_k, key_field="clause_id")
        for hit in hits:
            hit["superseded_by"] = succ.get(hit["clause_id"])
            for col in ("valid_from", "valid_to"):
                if isinstance(hit.get(col), datetime):
                    hit[col] = hit[col].isoformat()
        return RetrievalResult(
            status=RetrievalStatus.OK if hits else RetrievalStatus.NO_MATCH,
            hits=hits, query=query,
            latency_ms=int((time.perf_counter() - started) * 1000))

    # -- 非结构化：复盘结论 ------------------------------------------------

    async def search_claims(self, *, org_id: str, query: str, top_k: int = 3,
                            include_inactive: bool = False) -> RetrievalResult:
        """检索历史复盘结论。

        ★ `status` 一起返回，**不在这里过滤掉 superseded/refuted**：
        「查到的历史结论已经被推翻了」本身就是要让上层看见的局面
        （那正是 `memory.conflict_resolve` 的题面）。默认只给 active，
        要看全部得显式要 —— **但绝不是悄悄把它们藏起来**。
        """
        started = time.perf_counter()
        try:
            async with self.db.tx() as conn:
                rows = [dict(r) for r in await conn.fetch(
                    """
                    SELECT claim_id, claim, scope_json, evidence, confidence,
                           source_doc, status, superseded_by, recorded_at
                      FROM insights
                     WHERE (scope = 'global' OR scope = $1)
                       AND ($2::bool OR status = 'active')
                    """, org_id, include_inactive)]
        except Exception as exc:                       # noqa: BLE001
            return RetrievalResult(
                status=RetrievalStatus.UNAVAILABLE, hits=[], query=query,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"retrieval_unavailable: {type(exc).__name__}: {str(exc)[:200]}")

        hits = _rank(rows, query, text_of=_insight_text, top_k=top_k, key_field="claim_id")
        for hit in hits:
            if isinstance(hit.get("recorded_at"), datetime):
                hit["recorded_at"] = hit["recorded_at"].isoformat()
        return RetrievalResult(
            status=RetrievalStatus.OK if hits else RetrievalStatus.NO_MATCH,
            hits=hits, query=query,
            latency_ms=int((time.perf_counter() - started) * 1000))


# --------------------------------------------------------------------------
# 入库
# --------------------------------------------------------------------------


async def upsert_policy_clauses(db: Database, rows: list[dict[str, Any]], *,
                                scope: str = "global") -> int:
    """幂等入库。语料是**增量追加**的，所以按 (scope, clause_id) upsert。"""
    async with db.tx() as conn:
        for row in rows:
            await conn.execute(
                """
                INSERT INTO policy_clauses (scope, clause_id, title, body, section_path,
                                            platform, region, valid_from, valid_to,
                                            version, supersedes, source_doc)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (scope, clause_id) DO UPDATE SET
                    title=EXCLUDED.title, body=EXCLUDED.body,
                    section_path=EXCLUDED.section_path, platform=EXCLUDED.platform,
                    region=EXCLUDED.region, valid_from=EXCLUDED.valid_from,
                    valid_to=EXCLUDED.valid_to, version=EXCLUDED.version,
                    supersedes=EXCLUDED.supersedes, source_doc=EXCLUDED.source_doc,
                    ingested_at=now()
                """,
                scope, row["clause_id"], row["title"], row["body"],
                row.get("section_path") or "", row.get("platform"), row.get("region"),
                _ts(row.get("valid_from")), _ts(row.get("valid_to")),
                row.get("version") or "v1", row.get("supersedes"),
                row.get("source_doc") or "")
    return len(rows)


async def upsert_insights(db: Database, rows: list[dict[str, Any]], *,
                          scope: str = "global") -> int:
    async with db.tx() as conn:
        for row in rows:
            await conn.execute(
                """
                INSERT INTO insights (scope, claim_id, claim, scope_json, evidence,
                                      confidence, source_doc, status, superseded_by,
                                      recorded_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (scope, claim_id) DO UPDATE SET
                    claim=EXCLUDED.claim, scope_json=EXCLUDED.scope_json,
                    evidence=EXCLUDED.evidence, confidence=EXCLUDED.confidence,
                    source_doc=EXCLUDED.source_doc, status=EXCLUDED.status,
                    superseded_by=EXCLUDED.superseded_by,
                    recorded_at=EXCLUDED.recorded_at, ingested_at=now()
                """,
                scope, row["claim_id"], row["claim"],
                row.get("scope") or row.get("scope_json") or {},
                row.get("evidence") or "", row.get("confidence") or "medium",
                row.get("source_doc") or "", row.get("status") or "active",
                row.get("superseded_by"), _ts(row.get("recorded_at")))
    return len(rows)
