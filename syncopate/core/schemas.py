"""四件套的数据结构定义。

一条 case = 四个文件，缺一不可：

    case.json      考题     —— 用户说了什么、已知哪些线索
    env.json       世界     —— 这一刻数据库长什么样（只读表 + 政策库）
    verifier.json  评分标准 —— 必须查什么、必须做什么、终答必须给出哪些字段
    gold.json      标准答案 —— 一条走得通的最优轨迹

为什么 RL 要在运行时读它们：SFT 的答案是死的（照抄 gold），
RL 的答案是活的（模型自己乱走），所以必须现场重放世界、现场对照标准。

与老师包最大的区别：`required_answer_fields` 取代了 `required_response_points`。
老师那套要 LLM judge 判「这句话说清楚了没有」；我们要求终答是结构化的，
每个字段的真值都能从 env 里解析出来，**纯规则可判**。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "syncopate_v1"


# --------------------------------------------------------------------------
# 1. Case —— 考题
# --------------------------------------------------------------------------


@dataclass
class CaseMetadata:
    """case 的分类标签。两条轴是正交的，都要标。

    signal_class 是「这条 case 的 reward 长什么样」（决定它在 RL 里起什么作用）；
    bucket 是「这条 case 想修哪种错」（决定它为什么值得造出来）。
    """

    # ---- 轴 1：reward 信号形态。决定这条 case 在 GRPO 里能不能产生梯度 ----
    # all_high     组内 rollout 全高分 → advantage≈0 → 弱信号，作为反例存在
    # graded       组内分数分层 → advantage 拉得开 → RL 主力
    # long_tail    正确率不低但耗时极长 → 用来暴露 staleness / 阻塞问题
    # all_low      链路太长基本做不完 → 需要 curriculum 拆分
    # high_risk    有前置检查的写动作，漏检直接碰 cap
    # tool_missing 故意抽掉必需工具，看模型怎么绕；后续用 SFT 补
    signal_class: str

    # ---- 轴 2：policy weakness（来自 AdCampaignAgent 的 RL_Data_Design.md）----
    # tool_confusion / critical_args / sequential_dependency /
    # parallel_grouping / clarify_boundary / reject_boundary
    bucket: str

    # ---- 辅助标签 ----
    topology: str = "standard"      # standard / sequential / parallel / clarify / reject
    difficulty: str = "L2"          # L1..L5
    entry_mode: str = "id_given"    # id_given（关键 id 直接给）/ must_discover（要自己查）
    primary_intent: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Case:
    """考题本体。

    context 和 entities 的区别是这套数据最容易搞错的地方：

        context  —— 模型**看得到**的信息，会渲染进 prompt
        entities —— 真值实体表，只给 verifier 和 gold 用

    当 entry_mode == "must_discover" 时，关键 id 只在 entities 里、不在 context 里，
    模型必须靠读工具自己查出来。这是「调查先于行动」的强制手段。
    """

    case_id: str
    user_message: str
    context: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    metadata: CaseMetadata = field(default_factory=lambda: CaseMetadata("graded", "tool_confusion"))
    max_steps: int = 10
    # 本 case 允许模型看到的工具子集；None = 用全量菜单。
    # tool_missing 类 case 靠这个字段故意抽掉必需工具。
    tool_menu: list[str] | None = None
    version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# 2. EnvSnapshot —— 世界
# --------------------------------------------------------------------------


@dataclass
class EnvSnapshot:
    """这一刻的世界状态。**只读**——工具读它，但写动作不改它。

    写动作的结果去 Sandbox 的台账里，这样「世界原样」和「本次 rollout 做了什么」
    永远分得开，重放和归因才可能。
    """

    case_id: str
    # reference_now 让「最近 7 天」这类相对时间可复现。
    reference_now: str = "2026-08-01T00:00:00+00:00"
    # 只读表：campaigns / creatives / accounts / benchmarks / attribution ...
    readonly_tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 政策库：条件规则，verifier 用它算「本 case 的正确决策应该是什么」
    policies: list[dict[str, Any]] = field(default_factory=list)
    version: str = SCHEMA_VERSION

    def table(self, name: str) -> dict[str, Any]:
        return self.readonly_tables.get(name, {})

    def row(self, table: str, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        return self.table(table).get(key)


# --------------------------------------------------------------------------
# 3. VerifierSpec —— 评分标准
# --------------------------------------------------------------------------


@dataclass
class AnswerField:
    """终答里必须给出的一个字段。

    value_source 是「真值从哪来」的引用式，由 resolve_value_source 解析：

        "literal:creative_fatigue"   字面值
        "campaign.cpi"               env.readonly_tables['campaigns'][entities['campaign_id']]['cpi']
        "decision.new_budget"        policy 规则套上 case 事实算出来的期望决策
        None                         只要求「说了」，不校验值

    这就是我们不需要 LLM judge 的原因：每个字段都能算出真值来比对。
    """

    key: str
    value_source: str | None = None
    description: str = ""
    # ★ 这个字段的值必须由哪个读工具的 observation 背书。
    # 不声明的话，模型可以直接猜一个高频词蒙对（比如不查审核就写 "approved"）——
    # value_source 只管「值对不对」，管不了「你凭什么知道」。
    # 声明后由 false_claim_cap 负责封顶。老师包里的 false_promise_cap 治的是同一种病。
    evidence_tool: str | None = None


@dataclass
class SideEffectReq:
    """必须真实发生的一个写动作。

    required_args 里的值同样可以是 value_source 引用式，
    比如 {"new_budget": "decision.new_budget"} 表示「改的数必须等于政策算出来的数」。
    """

    tool: str
    required_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierSpec:
    """评分标准。verifier 只认这份文件，不认 gold。

    （gold 只是「一条走得通的路」，不是唯一解。RL 的意义就在于允许模型走别的路。）
    """

    # 期望的顶层行为：tool_call 正常执行 / clarify 先问清 / reject 拒答
    expected_behavior: str = "tool_call"

    # ---- evidence 子分：该查的读工具 ----
    required_read_tools: list[str] = field(default_factory=list)

    # ---- outcome 子分 ----
    allowed_write_tools: list[str] = field(default_factory=list)
    required_side_effects: list[SideEffectReq] = field(default_factory=list)
    forbidden_side_effects: list[str] = field(default_factory=list)
    required_answer_fields: list[AnswerField] = field(default_factory=list)

    # ---- policy 子分：是否需要走政策库 ----
    policy_required: bool = False
    expected_policy_id: str | None = None

    # ---- efficiency 子分 ----
    max_steps: int = 10

    # ---- caps：本 case 启用的封顶规则名（域相关，定义在 domains/*/rules.py）----
    # None = 启用全部已注册规则（默认从严：忘了声明会被 gold 验证当场抓出来）
    # []   = 明确不启用任何 cap
    # 这两者必须分开——早期版本把空列表也当成「全部」，结果玩具 case 被别的域的
    # 规则打中，排查了半天。默认从严是对的，但「明确关掉」也得有办法表达。
    active_caps: list[str] | None = None

    version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# 4. GoldPath —— 标准答案
# --------------------------------------------------------------------------


@dataclass
class GoldPath:
    """一条走得通的最优轨迹。SFT 直接抄它；RL 只拿它做健全性检查。"""

    actions: list[dict[str, Any]] = field(default_factory=list)   # [{tool, arguments}]
    final_answer: dict[str, Any] = field(default_factory=dict)    # 结构化终答
    expected_reward_min: float = 0.95
    version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# 打包 / 读写
# --------------------------------------------------------------------------

# 四件套在磁盘上的固定布局。所有读写都走这张表，不在别处硬编码路径。
ARTIFACT_LAYOUT: dict[str, tuple[str, str]] = {
    "case": ("cases", ".json"),
    "env": ("env_snapshots", ".env.json"),
    "verifier": ("verifier_specs", ".verifier.json"),
    "gold": ("gold_paths", ".gold.json"),
}


@dataclass
class CaseBundle:
    """四件套的内存表示。造 case、跑 rollout、算分都传这个对象。"""

    case: Case
    env: EnvSnapshot
    verifier: VerifierSpec
    gold: GoldPath | None = None

    @property
    def case_id(self) -> str:
        return self.case.case_id

    # ---- 落盘 ----

    def write(self, batch_dir: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        payloads = {"case": self.case, "env": self.env, "verifier": self.verifier, "gold": self.gold}
        for kind, obj in payloads.items():
            if obj is None:
                continue
            subdir, suffix = ARTIFACT_LAYOUT[kind]
            path = batch_dir / subdir / f"{self.case_id}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(asdict(obj), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths[kind] = path
        return paths

    # ---- 读盘 ----

    @classmethod
    def read(cls, batch_dir: Path, case_id: str) -> CaseBundle:
        def load(kind: str) -> dict[str, Any] | None:
            subdir, suffix = ARTIFACT_LAYOUT[kind]
            path = batch_dir / subdir / f"{case_id}{suffix}"
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        case_raw = load("case")
        if case_raw is None:
            raise FileNotFoundError(f"case not found: {batch_dir}/cases/{case_id}.json")
        gold_raw = load("gold")
        return cls(
            case=_build_case(case_raw),
            env=EnvSnapshot(**(load("env") or {"case_id": case_id})),
            verifier=_build_verifier(load("verifier") or {}),
            gold=GoldPath(**gold_raw) if gold_raw else None,
        )


def _build_case(raw: dict[str, Any]) -> Case:
    raw = dict(raw)
    raw["metadata"] = CaseMetadata(**raw.get("metadata", {"signal_class": "graded", "bucket": "tool_confusion"}))
    return Case(**raw)


def _build_verifier(raw: dict[str, Any]) -> VerifierSpec:
    raw = dict(raw)
    raw["required_side_effects"] = [SideEffectReq(**item) for item in raw.get("required_side_effects", [])]
    raw["required_answer_fields"] = [AnswerField(**item) for item in raw.get("required_answer_fields", [])]
    return VerifierSpec(**raw)
