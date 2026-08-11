"""世界构造器：拼出一份 EnvSnapshot。

造 case 的人只声明「这个 campaign 的 CPI 涨了 30%、账户是 standard 等级、有风险标记」，
表结构和默认值由这里补全。好处是 case 定义能保持一两行，且改表结构只改这一处。

一条原则：**世界里的数字要能推出正确答案**。比如 cpi/cpi_baseline 的比值决定
detect_anomalies 会不会报 cpi_spike；daily_budget + 账户等级决定 approved_budget 是多少。
不要在 case 里另写一份「预期答案」，那样两边会不同步。
"""

from __future__ import annotations

from typing import Any

import functools
import json
from datetime import timedelta
from pathlib import Path

from syncopate.core.schemas import EnvSnapshot
from syncopate.domains.adcampaign.memory import LANES, parse_time
from syncopate.domains.adcampaign.policies import BUDGET_POLICIES, PLATFORM_POLICIES

_EXTERNAL_PATH = Path(__file__).resolve().parents[3] / "data" / "external" / "ingested.json"


@functools.lru_cache(maxsize=1)
def load_external() -> dict[str, Any]:
    """加载离线 ingest 的产物（安全线 / 素材目录 / 时令日历）。

    缓存是必要的：造几百条 case 时会被调几百次，每次读盘没意义。
    文件不存在时返回空表——这样单元测试不依赖外部数据也能跑。
    """
    if not _EXTERNAL_PATH.exists():
        return {}
    return json.loads(_EXTERNAL_PATH.read_text(encoding="utf-8"))

# 各表的默认值。case 只需覆盖它关心的字段。
_CAMPAIGN_DEFAULTS: dict[str, Any] = {
    "platform": "Meta",
    "game_genre": "puzzle",
    "status": "active",
    # ★ 开投至今多少天。它决定每个指标收敛没有（见 maturity.py）。
    # 默认 30 = 早已收敛，这样存量 case 的世界一个字节都不变——
    # 新增机制不该悄悄改掉已经测过基线的那批 case。
    "started_days_ago": 30,
    "daily_budget": 50_000,      # 分。真实 Meta API 就是最小货币单位
    "spend_7d": 3200.0,
    "installs_7d": 1280,
    "cpi": 2.10,
    "cpi_baseline": 2.10,
    "roas_d7": 0.45,
    "roas_d7_baseline": 0.45,
    "ctr": 0.021,
    "ctr_baseline": 0.021,
    "frequency": 2.4,
    "impressions": 1_520_000,
}

_CREATIVE_DEFAULTS: dict[str, Any] = {
    "asset_type": "video",
    "ctr": 0.020,
    "ipm": 6.2,
    "spend_7d": 900.0,
    "frequency": 2.5,
    "status": "active",
}

_ACCOUNT_DEFAULTS: dict[str, Any] = {
    "tier": "standard",
    "status": "active",
    "risk_flag": False,
    "risk_reason": None,
    # ★ 预算类字段统一用**分**（和 daily_budget 同口径）。
    # 只改 daily_budget 不改这两个，monthly_cap 的约束会把预算掐到 1/100 ——
    # 正是这次要消灭的那种单位混用 bug。
    # 注意 spend_7d 留在"元"：它是 insights 指标，Meta 那边也是十进制字符串，
    # **真实世界的单位就是混的**，这一点如实建模。
    "monthly_cap": 6_000_000,
    "spend_mtd": 1_800_000,
}

# 行业基准表：**全组合覆盖**，key 形如 "Meta|puzzle|cpi"。
#
# ⚠️ 早期版本只硬编码了 5 条，生成器一组合出 25 种平台×品类就大面积
# `benchmark_not_found`。这类"参数组合出一个不自洽的世界"的错，
# 只有真跑 gold 才发现得了——所以生成器必须逐条实跑验证。
_PLATFORM_FACTOR = {"Meta": 1.00, "Google": 0.85, "TikTok": 0.70, "AppLovin": 0.78, "Unity": 0.75}
_GENRE_FACTOR = {"casual": 0.80, "puzzle": 1.00, "hyper_casual": 0.45, "rpg": 1.85, "strategy": 1.60}
_METRIC_BASE = {
    "cpi": (2.20, "USD"), "roas_d7": (0.42, "ratio"),
    "ctr": (0.021, "ratio"), "retention_d1": (0.34, "ratio"),
}


def _default_benchmarks() -> dict[str, dict[str, Any]]:
    """确定性地生成全部 平台 × 品类 × 指标 的基准值。

    数值由「平台系数 × 品类系数 × 指标基准」算出，不是随机数——
    重放和复现都要求它稳定。ROAS/CTR/留存这类"越高越好"的指标取倒数关系，
    这样 rpg（获客贵）自然就是 CPI 高、ROAS 低。
    """
    table: dict[str, dict[str, Any]] = {}
    for platform, pf in _PLATFORM_FACTOR.items():
        for genre, gf in _GENRE_FACTOR.items():
            for metric, (base, unit) in _METRIC_BASE.items():
                factor = pf * gf if metric == "cpi" else (1.0 / (pf * gf)) ** 0.5
                value = round(base * factor, 4)
                table[f"{platform}|{genre}|{metric}"] = {
                    "metric": metric, "value": value,
                    "p25": round(value * 0.72, 4), "p75": round(value * 1.32, 4),
                    "unit": unit,
                }
    return table


_DEFAULT_BENCHMARKS: dict[str, dict[str, Any]] = _default_benchmarks()


class WorldBuilder:
    """链式构造一份 env。

        env = (WorldBuilder("CASE_1")
               .account("ACC_01", tier="standard")
               .campaign("CMP_1024", account_id="ACC_01", cpi=2.80)
               .build())
    """

    def __init__(self, case_id: str, *, reference_now: str = "2026-08-01T00:00:00+00:00") -> None:
        self.case_id = case_id
        # ★ 时间由 case 自己声明。TTL 过滤、时令判断全基于它，
        #   所以"同一个请求在不同时间点有不同的正确答案"是可复现的。
        self.reference_now = reference_now
        external = load_external()
        self._tables: dict[str, dict[str, Any]] = {
            "campaigns": {}, "creatives": {}, "accounts": {},
            "benchmarks": dict(_DEFAULT_BENCHMARKS), "review_outcomes": {},
            # ---- 记忆库基线：只读，本次 rollout 的写动作永远不改它 ----
            "memory": {},
            # ---- 离线资料：安全线 / 素材目录 / 时令日历 ----
            "safety_lines": dict(external.get("safety_lines", {})),
            "creative_catalog": dict(external.get("creative_catalog", {})),
            "seasonal_events": {e["event"]: e for e in external.get("seasonal_events", [])},
        }
        self._policies: list[dict[str, Any]] = [*BUDGET_POLICIES, *PLATFORM_POLICIES]
        self._memory_seq = 0

    def account(self, account_id: str, **overrides: Any) -> WorldBuilder:
        self._tables["accounts"][account_id] = {
            "account_id": account_id, "name": overrides.pop("name", f"Account {account_id}"),
            **_ACCOUNT_DEFAULTS, **overrides,
        }
        return self

    def campaign(self, campaign_id: str, *, account_id: str = "ACC_01", **overrides: Any) -> WorldBuilder:
        self._tables["campaigns"][campaign_id] = {
            "campaign_id": campaign_id, "account_id": account_id,
            "name": overrides.pop("name", f"Campaign {campaign_id}"),
            **_CAMPAIGN_DEFAULTS, **overrides,
        }
        return self

    def creative(self, creative_id: str, *, campaign_id: str, **overrides: Any) -> WorldBuilder:
        self._tables["creatives"][creative_id] = {
            "creative_id": creative_id, "campaign_id": campaign_id,
            "name": overrides.pop("name", creative_id.lower()),
            **_CREATIVE_DEFAULTS, **overrides,
        }
        return self

    def benchmark(self, platform: str, genre: str, metric: str, **row: Any) -> WorldBuilder:
        self._tables["benchmarks"][f"{platform}|{genre}|{metric}"] = {"metric": metric, **row}
        return self

    def review_outcome(self, creative_name: str, *, review_status: str, reject_reason: str | None = None) -> WorldBuilder:
        """预设某个素材名的审核结果。不设则默认通过。"""
        self._tables["review_outcomes"][creative_name] = {
            "review_status": review_status, "reject_reason": reject_reason
        }
        return self

    def memory(self, lane: str, *, days_ago: float, subject: dict[str, Any],
               content: dict[str, Any], confidence: float = 0.95,
               evidence_refs: list[str] | None = None) -> WorldBuilder:
        """往记忆库基线里放一条记录。

        `days_ago` 相对 reference_now 倒推——这样"这条记忆过期没有"完全由
        case 声明的时间决定，改一个数就能让同一条记忆从有效变过期。
        """
        if lane not in LANES:
            raise ValueError(f"未知 lane: {lane}")
        self._memory_seq += 1
        created = parse_time(self.reference_now) - timedelta(days=days_ago)
        record_id = f"MEM_{lane[:3].upper()}_{self._memory_seq:04d}"
        self._tables["memory"][record_id] = {
            "record_id": record_id, "lane": lane, "subject": subject, "content": content,
            "created_at": created.isoformat(), "confidence": confidence,
            "evidence_refs": list(evidence_refs or [f"EP_{self._memory_seq:05d}"]),
            "status": "active",
        }
        return self

    def build(self) -> EnvSnapshot:
        return EnvSnapshot(
            case_id=self.case_id,
            reference_now=self.reference_now,
            readonly_tables={k: v for k, v in self._tables.items()},
            policies=list(self._policies),
        )
