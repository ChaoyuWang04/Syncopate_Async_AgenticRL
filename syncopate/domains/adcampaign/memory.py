"""记忆机制：分 lane、带 TTL、写入需提案。

★ 存储在哪：**不引数据库，就用现有的两个地方**

    ①「本次 rollout 开始前记忆库里有什么」 → env_snapshot 的只读表 `memory`
    ②「本次 rollout 里 agent 提议写什么」   → Sandbox 的 audit_log

为什么不能用 SQLite / 全局存储：GRPO 会把同一条 case **并发跑 8 遍**。任何跨 rollout
的可写状态都会让 A 读到 B 写的东西 → 轨迹不再独立 → 组内比较失效 → 而且不可复现。
verl 还会重放、resume、多 epoch 跑同一批数据，持久状态是灾难。

★ 时间从哪来：`env.reference_now`。每条 case 自己声明"现在几点"，TTL 过滤就是
纯计算。所以"同一个问题第二次问答案不同"**不是靠跨 rollout 的状态**，而是靠
造一对 case——用户消息和世界完全相同，只有 memory 不同，正确答案就不同。
这既保留了 memory 的必要性，又完全确定可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# --------------------------------------------------------------------------
# lane 定义
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Lane:
    name: str
    ttl_days: int
    write_policy: str      # "system" 只有系统能写 / "proposal" 需提案 / "proposal_review" 提案+强审核
    description: str


LANES: dict[str, Lane] = {
    # 当前会话已查到的东西。不落盘，所以不在这张表里出现，列出来是为了让概念完整。
    "working": Lane("working", 0, "free", "当前会话上下文"),
    # 单次投放动作的完整记录：谁在什么时候把哪条素材投到哪个地域、结果如何。
    # 由系统在动作完成后自动写，agent 不能自己往里塞——否则它可以伪造"历史"。
    "episodic": Lane("episodic", 30, "system", "单次投放动作的完整记录"),
    # 素材/受众的稳定属性：视觉标签、历史 IPM、适配地域。
    "semantic": Lane("semantic", 90, "proposal", "素材与受众的稳定属性"),
    # 干预效果经验：对某个 campaign 用过什么优化方案、有没有效。
    "business": Lane("business", 60, "proposal", "优化干预的历史效果"),
    # 风控标记：预算调整频次、异常操作、账户风险分。改这个必须先过风控。
    "risk": Lane("risk", 180, "proposal_review", "账户与操作的风险标记"),
}

WRITABLE_LANES = [name for name, lane in LANES.items() if lane.write_policy != "system"]

# 提案入库的门槛（老师课件里的 confidence ≥ 0.7 + evidence_refs ≥ 2）
MIN_CONFIDENCE = 0.7
MIN_EVIDENCE_REFS = 2

# 未脱敏的用户标识特征。写提案里出现就算隐私违规。
PII_HINTS = ("@", "+86", "手机号", "email", "身份证", "phone")


# --------------------------------------------------------------------------
# 记录
# --------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    record_id: str
    lane: str
    subject: dict[str, Any]          # {account_id, campaign_id, creative_id, region, platform...}
    content: dict[str, Any]
    created_at: str                  # ISO8601
    confidence: float = 1.0
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "active"           # active / superseded / invalidated

    def expires_at(self) -> datetime:
        return parse_time(self.created_at) + timedelta(days=LANES[self.lane].ttl_days)

    def is_expired(self, now: datetime) -> bool:
        return now > self.expires_at()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "lane": self.lane, "subject": self.subject,
            "content": self.content, "created_at": self.created_at,
            "expires_at": self.expires_at().isoformat(), "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs), "status": self.status,
        }


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_record(raw: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        record_id=raw["record_id"], lane=raw["lane"], subject=raw.get("subject", {}),
        content=raw.get("content", {}), created_at=raw["created_at"],
        confidence=float(raw.get("confidence", 1.0)),
        evidence_refs=list(raw.get("evidence_refs", [])),
        status=raw.get("status", "active"),
    )


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------


def search(
    records: dict[str, Any],
    now: datetime,
    *,
    lane: str | None = None,
    subject: dict[str, Any] | None = None,
    include_expired: bool = False,
    top_k: int = 5,
) -> list[MemoryRecord]:
    """按 lane + subject 过滤，默认剔除过期和已失效的记录。

    subject 是**子集匹配**：给 {"account_id": "ACC_11"} 会匹配到所有该账户的记录，
    不要求 subject 完全相等。
    """
    hits: list[MemoryRecord] = []
    for raw in records.values():
        record = build_record(raw)
        if lane and record.lane != lane:
            continue
        if record.status != "active":
            continue
        if not include_expired and record.is_expired(now):
            continue
        if subject and any(record.subject.get(k) != v for k, v in subject.items() if v is not None):
            continue
        hits.append(record)
    # 新的排前面——决策时优先看最近发生的事
    hits.sort(key=lambda r: parse_time(r.created_at), reverse=True)
    return hits[:top_k]


def expired_ids(records: dict[str, Any], now: datetime) -> set[str]:
    """已过 TTL 的记录 id。`stale_memory_cap` 用它判断模型有没有拿过期信息做决策。"""
    return {rid for rid, raw in records.items() if build_record(raw).is_expired(now)}


# --------------------------------------------------------------------------
# 写提案的合法性
# --------------------------------------------------------------------------


@dataclass
class ProposalIssues:
    """一条写提案的问题清单。空 = 干净。

    分成 hard / soft 两类是有意的：
      hard —— 真实 API 会直接 403 的（往系统专属 lane 写、lane 名不存在），工具直接报错
      soft —— 属于"纪律"的（证据不足、没先过风控），**工具照做，由 cap 封顶**

    后者不硬拦是因为：如果工具直接拒绝，模型学到的是"报错了就换一个"，
    而不是"为什么要先查"。让它做成、然后拿低分，信号才是对的。
    """

    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)


def check_proposal(
    *,
    lane: str,
    content: dict[str, Any],
    confidence: float,
    evidence_refs: list[str],
    risk_reviewed: bool,
) -> ProposalIssues:
    issues = ProposalIssues()
    spec = LANES.get(lane)
    if spec is None:
        issues.hard.append(f"unknown_lane: {lane}")
        return issues
    if spec.write_policy == "system":
        issues.hard.append(f"lane_is_system_managed: {lane}")
    if spec.write_policy == "free":
        issues.hard.append(f"lane_not_persisted: {lane}")

    if confidence < MIN_CONFIDENCE:
        issues.soft.append(f"low_confidence: {confidence} < {MIN_CONFIDENCE}")
    if len(evidence_refs) < MIN_EVIDENCE_REFS:
        issues.soft.append(f"insufficient_evidence: {len(evidence_refs)} < {MIN_EVIDENCE_REFS}")
    if spec.write_policy == "proposal_review" and not risk_reviewed:
        issues.soft.append(f"lane_requires_review: {lane} 需先调 risk.check_account")

    blob = " ".join(str(v) for v in content.values())
    if any(hint in blob for hint in PII_HINTS):
        issues.soft.append("pii_not_redacted")
    return issues
