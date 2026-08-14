"""M8 · RAG v1 的语料层：半结构化政策条款 + 非结构化复盘结论。

★★★ 为什么沙盒里不能用真向量库

GRPO 把同一条 case **并发跑 8 遍**比谁好谁坏。近似最近邻、共享索引、任何跨
rollout 的可写状态，都会让「这次召回了、下次没召回」变成组内 reward 的差异来源
—— 而那个差异会被当成"模型做得不同"记进 advantage。这就是踩过的那条：
**RL 里任何跨 rollout 的随机性都是污染**（见 EnvSnapshot.failures 的注释）。

⇒ 语料和 `safety_lines` / `benchmarks` 一样，**逐 case 落在 readonly_tables 里**，
检索是 `(query, 本 case 的语料表, reference_now)` 的**纯函数**。
「同一个问题第二次问答案不同」不靠跨 rollout 状态，靠造一对只有语料不同的 case。

⚠️ **由此产生一个必须显式记账的训推缺口**（设计文档 §33）：
线上是真向量库 + rerank，沙盒是确定性词法打分。**处理办法不是把它藏起来**，
而是让沙盒实现**同一份契约**——会返回空、会返回过期的、会返回只沾边的。
按「沙盒不要比真实世界友好」的原则，把不完美如实建模，而不是给个干净的召回。

★ 合成语料不是"写一堆文档"，是**造出可判定的结构**。三条轴必须成对出现：

    版本对   同一主题的 v1（valid_to 已过期）+ v2（生效中）   → 过期检出率
    空洞     明确无对应 chunk 的查询                         → 无检索幻觉率
    矛盾对   status=active 的旧结论 + 与之冲突的新证据        → conflict_resolve 的题面

前两条是 §14 点名的两项验收（都要求趋近 0）；第三条是遗留清单里挂着的那个缺口——
「现在没有任何一道题考『查到的历史结论和现在的数据矛盾了怎么办』」。

★ 复盘结论按设计文档 §13 的 schema 落，**`status` 字段不能省**：
它是 M12 飞轮的物理接口 —— 飞轮跑出相反结论时把旧条目标 `refuted` 而不是删掉，
**你需要知道"我们曾经这么以为"**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# --------------------------------------------------------------------------
# 分词与打分：确定性，且刻意不完美
# --------------------------------------------------------------------------

# 中英混排，按「连续 ASCII 词」+「单个 CJK 字」切。
# 选单字而不是词组，是因为不想引入分词器依赖 —— 分词器版本变了检索结果就变，
# 那等于把一个隐藏的非确定性源引进来（同 flash-attn 垫片那条教训：依赖要能钉死）。
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")

# 停用词：不去掉的话「的/是/在」会让任意两条文本都有相似度，
# 「检索为空」这个我们**需要**它发生的情形就永远不会发生。
_STOP = frozenset("的 了 在 是 和 与 或 有 我 你 它 这 那 个 会 要 能 对 从 把 被 中 上 下".split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t not in _STOP]


def overlap_score(query: str, document: str) -> float:
    """查询词在文档里的覆盖率，[0,1]。

    ★★ 选型是**实测**出来的，不是推的（2026-08-14，17 查询 × 10 条款的自建评测集，
    脚本见 `docs/infra_exp/` 对应报告；四种方案同集对照）：

        方案                最佳点      命中/12   误召回   正确留空/5
        词法覆盖率           0.35–0.40    11        2        4       ← 采用
        CJK 二元组           0.05         11        2        4
        BM25（按查询归一）    任意          11        6        0
        Qwen3-0.6B 向量      0.70          9        6        0

    **BM25 出局的真实机制**（此处原先写的是"奖励长文档偶然重合"，实测推翻）：
    归一化后 top1 恒为 1.0 ⇒ **任何阈值都必然返回至少一条**，
    它结构上产生不了"查不到" —— **BM25 是排序器，不是判定器**。
    而「无检索幻觉率」这项验收要求"查不到"必须能发生。

    **向量出局的真实机制**：Qwen3-0.6B 是 causal decoder 不是检索模型，
    mean-pool 向量各向异性强，任意两段文本余弦都落在 0.6–0.9，
    **相关与不相关之间没有分离带** ⇒ 高阈值全漏、低阈值全中。
    ⚠️ 这**不代表"embedding 不行"**，只代表手头这个模型不行；
    真正的 bge/e5 类检索模型大概率可用，代价是联网依赖 + 版本会影响检索结果。

    ⚠️⚠️ **已知且不打算用词法方案解决的局限：词汇失配。**
    「每天预算最多能加多少」vs「单日预算上调不得超过前一日的 20%」
    （每天/单日、加/上调、最多/不得超过）表层几乎无共同词，
    换分词、换 BM25、调阈值**都在错误的维度上使劲**，实测四种方案全军覆没。
    ⇒ 造题时的纪律见下面 MATCH_THRESHOLD 的注释。
    """
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(document))
    return len(q & d) / len(q)


# ★ 召回阈值。低于它就是**没查到**，而不是"返回一个最像的"。
#
# 这是本模块最重要的一个常数：真检索系统总能返回 top-k，哪怕全是噪声。
# 我们**必须**保留"确实查不到"这个状态，否则「检索为空时不编答案」这条
# 验收（§14，要求趋近 0）在构造上就不可能被触发 —— 又一个"机制建好了但没接上"。
#
# 0.35 是**扫出来的**，不是拍的：0.30 起误召回从 2 涨到 3，0.45 起开始漏召回，
# 0.35–0.40 是平台期（命中 11/12、误召回 2、正确留空 4/5）。取下沿留余量。
#
# ⚠️ **标定基准只有 17 条查询 × 10 条条款，而且是我自己写的。**
# 比第一版的"两个字符串"强，但**远不够定案**。真模板出来后必须用真实的
# gold 查询重标定 —— 这个数直接决定"检索为空"发生的频率，
# 定高了空洞遍地、定低了空洞永不发生，**两种都会让那项验收失去意义**。
MATCH_THRESHOLD = 0.35

# ★★★ 造题纪律（由词汇失配这条实测局限直接推出，见 overlap_score 的说明）
#
# 词法检索**教不会**"怎么组织查询词"，而真向量库不需要这个技能 ——
# 如果让 reward 依赖"换几种说法才查得到"，就会训出一个**只在沙盒里有用的错技能**。
# 这是训推不一致里最坏的一种：不是能力没学到，是学了个错的。
#
#   ✅ 该训的：拿到结果之后怎么办 —— 用现行版还是过期版、查不到时转人工还是硬答
#   ❌ 不该训的：怎么把查询词凑成文档里的原词
#
# ⇒ 造题时**必须验证 gold 的自然查询确实能命中**（造数据脚本里加断言），
#   **不许设计"要换三次说法才查得到"的题**。
#   §14 的两项验收都只考"怎么处理结果"，不考"怎么组织查询" —— 这是对的。


# --------------------------------------------------------------------------
# 两类语料的 schema
# --------------------------------------------------------------------------


@dataclass
class PolicyClause:
    """半结构化：平台广告政策 / 广告法 / 内部 SOP 的**一条条款**。

    按条款切并保留章节路径（设计文档 §13 第 3、4、5 项）。带生效期字段，
    因此"这条是不是过期了"是纯计算，不靠模型判断。
    """

    clause_id: str
    title: str
    body: str
    section_path: str                    # 如 "Meta 广告政策 / 4. 预算与竞价 / 4.2 单日涨幅"
    platform: str | None = None
    region: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None          # None = 长期有效
    version: str = "v1"
    supersedes: str | None = None        # ★ 指向被它取代的旧条款 clause_id
    source_doc: str = ""

    def is_expired(self, now: datetime) -> bool:
        return self.valid_to is not None and _parse(self.valid_to) < now

    def searchable(self) -> str:
        return " ".join(filter(None, [self.title, self.body, self.section_path]))

    def to_row(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id, "title": self.title, "body": self.body,
            "section_path": self.section_path, "platform": self.platform, "region": self.region,
            "valid_from": self.valid_from, "valid_to": self.valid_to, "version": self.version,
            "supersedes": self.supersedes, "source_doc": self.source_doc,
        }


@dataclass
class InsightClaim:
    """非结构化：复盘纪要抽出来的**一条结论**（设计文档 §13 第 6 项的理想形态）。

    ★ 按"一条结论"切，不按段落切 —— 每个 chunk 必须能独立成为一条可引用的论断。
    """

    claim_id: str
    claim: str
    scope: dict[str, Any] = field(default_factory=dict)      # region / product / period
    evidence: str = ""                                        # "复盘会议 2026-07-15，样本 N=42"
    confidence: str = "medium"                                # low / medium / high
    source_doc: str = ""
    status: str = "active"                                    # active / superseded / refuted
    superseded_by: str | None = None
    recorded_at: str | None = None

    def searchable(self) -> str:
        bits = [self.claim, *(str(v) for v in self.scope.values())]
        return " ".join(bits)

    def to_row(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "claim": self.claim, "scope": dict(self.scope),
            "evidence": self.evidence, "confidence": self.confidence,
            "source_doc": self.source_doc, "status": self.status,
            "superseded_by": self.superseded_by, "recorded_at": self.recorded_at,
        }


def _parse(ts: str) -> datetime:
    from syncopate.domains.adcampaign.memory import parse_time

    return parse_time(ts)


# --------------------------------------------------------------------------
# 检索：纯函数
# --------------------------------------------------------------------------


@dataclass
class Hit:
    key: str
    score: float
    row: dict[str, Any]


def search_rows(
    rows: dict[str, Any], query: str, *, text_of, top_k: int = 3,
    threshold: float = MATCH_THRESHOLD, filters: dict[str, Any] | None = None,
) -> list[Hit]:
    """在一张语料表里检索。**同一份输入永远给同一份输出。**

    排序键刻意带上 `key`：分数相同时按 id 排，避免 dict 顺序影响结果 ——
    栽过一次「prompt 内容取决于 dict 插入顺序」（去重守卫和泄漏检测跑在两个空间）。
    """
    filters = filters or {}
    hits: list[Hit] = []
    for key, row in rows.items():
        if any(row.get(f) not in (None, want) for f, want in filters.items() if want is not None):
            continue
        score = overlap_score(query, text_of(row))
        if score >= threshold:
            hits.append(Hit(key=key, score=round(score, 4), row=row))
    hits.sort(key=lambda h: (-h.score, h.key))
    return hits[:top_k]
