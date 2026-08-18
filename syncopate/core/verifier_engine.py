"""判分引擎：把一条轨迹变成一个 [0,1] 的标量 reward。

结构和老师包一样是「四个子分加权 + 若干 cap 封顶」，但有三处刻意的不同：

1. **没有 LLM judge**。老师的终答是自然语言客服话术，只能靠 LLM 判「说清楚了没有」；
   我们要求终答是结构化的，每个字段的真值都能从 env + policy 算出来 → 纯规则可判。
   好处：零成本、不可 hack、**完全可复现**（同一条轨迹每次算分一模一样）。

2. **cap 自带责任步号**。老师那边 9 个 cap 里有 6 个内部算出了 step 却在 return
   时丢掉了。我们让每个 cap 直接返回 `steps`，`cap_steps` 从第一天就有——
   这是后面做步级信用分配的地基（docs/syncopate/02 §3）。

3. **cap 规则是注册进来的，不是写死的**。引擎只管「怎么封顶」，
   「什么算违规」由 domains/*/rules.py 声明。换场景不用改引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import CaseBundle, VerifierSpec
from syncopate.core.trajectory import Trajectory

# 四个子分的权重，合计 1.0。
# 老师那套有第五个 communication(0.05)，纯靠 judge，我们直接砍掉，把权重给 outcome。
WEIGHTS: dict[str, float] = {
    "outcome": 0.50,     # 该做的写动作做了吗 + 终答字段说对了吗
    "policy": 0.20,      # 决策符合政策库吗
    "evidence": 0.20,    # 该查的查了吗
    "efficiency": 0.10,  # 有没有绕路 / 空转
}


# --------------------------------------------------------------------------
# cap 注册表
# --------------------------------------------------------------------------


@dataclass
class CapHit:
    """一次 cap 命中。steps 是责任步号——这是我们相对老师包最重要的增量。"""

    name: str
    ceiling: float           # 命中后 reward 的上限（由 CapRegistry 按注册值填入）
    reason: str
    steps: list[int] = field(default_factory=list)
    # ★ 2026-08-18：让**同一条规则**按违规的严重程度给不同的上限。
    # 起因：`unauthorized_write_cap` 把"越权开一张审批单"（无外部副作用）
    # 和"没打招呼就改预算"（不可逆、立即花钱）罚得一样重（都 0.30）。
    # ⚠️ 不用「ceiling 传 0 表示不覆盖」那种写法 —— 0.0 是合法上限
    # （multi_tool_per_step_cap / prompt_injection_cap 就是 0.0），
    # 用哨兵值会静默地把它们改掉。所以另开一个 None 字段。
    ceiling_override: float | None = None


# cap 检测器签名：(bundle, trajectory, sandbox) -> CapHit | None
CapDetector = Callable[[CaseBundle, Trajectory, Sandbox], "CapHit | None"]


class CapRegistry:
    """cap 规则表。域实现用 @CAPS.rule(...) 注册。"""

    def __init__(self) -> None:
        self._rules: dict[str, tuple[float, CapDetector]] = {}

    def rule(self, *, name: str, ceiling: float) -> Callable[[CapDetector], CapDetector]:
        def decorator(func: CapDetector) -> CapDetector:
            self._rules[name] = (ceiling, func)
            return func

        return decorator

    def names(self) -> list[str]:
        return sorted(self._rules)

    def evaluate(
        self, bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox, enabled: list[str] | None = None
    ) -> list[CapHit]:
        """跑全部（或指定的）cap 规则，返回所有命中。"""
        selected = self.names() if enabled is None else [n for n in enabled if n in self._rules]
        hits: list[CapHit] = []
        for name in selected:
            ceiling, detector = self._rules[name]
            hit = detector(bundle, trajectory, sandbox)
            if hit is not None:
                hit.name = name
                hit.ceiling = ceiling if hit.ceiling_override is None else hit.ceiling_override
                hits.append(hit)
        return hits


CAPS = CapRegistry()


# --------------------------------------------------------------------------
# 真值解析
# --------------------------------------------------------------------------


def resolve_value_source(source: str, bundle: CaseBundle, decision: dict[str, Any] | None) -> Any:
    """把 value_source 引用式解析成真值。这是「不用 LLM 也能判对错」的核心。

    支持四种写法：
        literal:creative_fatigue     字面值
        entity:campaign_id           case.entities 里的实体
        campaigns.cpi                只读表里那一行的字段（行由 entities 定位）
        decision.new_budget          政策规则套上 case 事实算出的期望决策
    """
    if not source:
        return None
    if source.startswith("literal:"):
        return source[len("literal:") :]
    if source.startswith("entity:"):
        return bundle.case.entities.get(source[len("entity:") :])
    if source.startswith("decision."):
        return (decision or {}).get(source[len("decision.") :])

    # 形如 "campaigns.cpi"：表名 + 字段名，行主键从 entities 里按约定取
    if "." in source:
        table, _, field_path = source.partition(".")
        key = bundle.case.entities.get(_PRIMARY_KEY_OF.get(table, ""))
        row = bundle.env.row(table, key)
        return _get_nested(row, field_path)
    return None


# 表名 -> 该表在 entities 里的主键字段名
_PRIMARY_KEY_OF = {
    "campaigns": "campaign_id",
    "creatives": "creative_id",
    "accounts": "account_id",
    "ad_groups": "ad_group_id",
}


def _get_nested(row: dict[str, Any] | None, dotted: str) -> Any:
    current: Any = row
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _as_bool(value: Any) -> bool:
    """把 "true"/"false"/"1"/"0" 这类字面量按字面意思转成布尔，其余走 Python 真值。"""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0", ""}:
            return False
    return bool(value)


def values_equal(actual: Any, expected: Any) -> bool:
    """宽松比较：数字按数值比（1 == 1.0），其余去空格小写后比字符串。"""
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        # ⚠️ `literal:false` 走到这里时是字符串 "false"，而 bool("false") 是 True——
        # 于是「期望 false」永远判成期望 True，`literal:true` 只是碰巧对。
        # 实测抓到点：I02 的 can_decide=False 被判为不匹配，outcome 卡在 0.67。
        return _as_bool(actual) == _as_bool(expected)
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except (TypeError, ValueError):
        return str(actual).strip().lower() == str(expected).strip().lower()


# --------------------------------------------------------------------------
# 四个子分
# --------------------------------------------------------------------------


def score_evidence(spec: VerifierSpec, trajectory: Trajectory) -> tuple[float, dict[str, Any]]:
    """该查的读工具查了几个。

    注意这里给读工具**显式定价**：读工具不翻转任何谓词，但它绝不是「白走一步」。
    「调查先于行动」这条守则就是靠这个子分兑现的。
    """
    required = spec.required_read_tools
    if not required:
        return 1.0, {"required": [], "hit": []}
    called = set(trajectory.called_tools())
    hit = [tool for tool in required if tool in called]
    return len(hit) / len(required), {"required": required, "hit": hit, "missing": sorted(set(required) - called)}


def score_outcome(
    spec: VerifierSpec, bundle: CaseBundle, trajectory: Trajectory, sandbox: Sandbox, decision: dict[str, Any] | None
) -> tuple[float, dict[str, Any]]:
    """写动作 + 终答字段。这两块合起来就是「任务到底做成了没有」。

    三种组合分别用不同公式，和老师的 calculate_outcome 同构：
        既要写又要答 -> 0.75*write + 0.25*answer
        只要写       -> write
        只要答       -> answer（clarify / reject / 纯查询类走这条）
    """
    write_score, write_detail = _score_writes(spec, bundle, sandbox, decision)
    answer_score, answer_detail = _score_answer_fields(spec, bundle, trajectory, decision)
    detail = {"write": write_detail, "answer": answer_detail}

    has_write = bool(spec.required_side_effects)
    has_answer = bool(spec.required_answer_fields)
    if has_write and has_answer:
        return 0.75 * write_score + 0.25 * answer_score, detail
    if has_write:
        return write_score, detail
    return answer_score, detail


def _score_writes(
    spec: VerifierSpec, bundle: CaseBundle, sandbox: Sandbox, decision: dict[str, Any] | None
) -> tuple[float, list[dict[str, Any]]]:
    """每个 required_side_effect 都要真实发生，且关键参数对得上。"""
    if not spec.required_side_effects:
        return 1.0, []
    passed, detail = 0, []
    for requirement in spec.required_side_effects:
        records = sandbox.records_for(requirement.tool)
        args_ok, mismatches = False, {}
        for record in records:
            mismatches = {}
            for key, source in requirement.required_args.items():
                expected = resolve_value_source(source, bundle, decision) if isinstance(source, str) else source
                if not values_equal(record.arguments.get(key), expected):
                    mismatches[key] = {"actual": record.arguments.get(key), "expected": expected}
            if not mismatches:
                args_ok = True
                break
        ok = bool(records) and args_ok
        passed += int(ok)
        detail.append({"tool": requirement.tool, "executed": bool(records), "args_ok": args_ok,
                       "mismatches": mismatches, "passed": ok})
    return passed / len(spec.required_side_effects), detail


def _score_answer_fields(
    spec: VerifierSpec, bundle: CaseBundle, trajectory: Trajectory, decision: dict[str, Any] | None
) -> tuple[float, list[dict[str, Any]]]:
    """终答字段：说了吗 + 说对了吗。

    ★ 这里是防 reward hacking 的关键。老师的 heuristic 兜底写的是
      `covered = bool(final_text.strip())` —— 只要吐一句非空的话，所有信息点全算覆盖，
      模型秒学会说废话刷分。我们要求字段必须存在**且**值等于算出来的真值。
    """
    if not spec.required_answer_fields:
        return 1.0, []
    if not trajectory.parse_ok:
        return 0.0, [{"error": "final_answer_unparseable"}]

    passed, detail = 0, []
    for field_spec in spec.required_answer_fields:
        stated = trajectory.final_answer.get(field_spec.key)
        present = stated is not None and str(stated).strip() != ""
        expected = resolve_value_source(field_spec.value_source, bundle, decision) if field_spec.value_source else None
        ok = present and (expected is None or values_equal(stated, expected))
        passed += int(ok)
        detail.append({"key": field_spec.key, "present": present, "stated": stated,
                       "expected": expected, "passed": ok})
    return passed / len(spec.required_answer_fields), detail


def score_efficiency(spec: VerifierSpec, trajectory: Trajectory) -> tuple[float, dict[str, Any]]:
    """步数超出「理论最少步数」就线性扣分，每多一步扣 5%。

    expected = 必查读工具数 + 必做写动作数。gold 走的就是这个步数。
    """
    expected = len(spec.required_read_tools) + len(spec.required_side_effects)
    actual = trajectory.num_steps
    overshoot = max(0, actual - expected)
    return max(0.0, 1.0 - 0.05 * overshoot), {"expected": expected, "actual": actual, "overshoot": overshoot}


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


@dataclass
class ScoreResult:
    reward: float                                  # 封顶后的最终 reward，进 GRPO 的就是它
    raw_reward: float                              # 封顶前的加权和
    subscores: dict[str, float]
    cap_hits: list[CapHit] = field(default_factory=list)
    cap_steps: dict[str, list[int]] = field(default_factory=dict)   # ★ cap -> 责任步号
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": round(self.reward, 6),
            "raw_reward": round(self.raw_reward, 6),
            "subscores": {k: round(v, 6) for k, v in self.subscores.items()},
            "active_caps": [h.name for h in self.cap_hits],
            "cap_reasons": {h.name: h.reason for h in self.cap_hits},
            "cap_steps": self.cap_steps,
            "details": self.details,
        }


def score_trajectory(
    bundle: CaseBundle,
    trajectory: Trajectory,
    sandbox: Sandbox,
    *,
    policy_scorer: Callable[[CaseBundle, Trajectory, Sandbox], tuple[float, dict[str, Any]]] | None = None,
    decision_fn: Callable[[CaseBundle], dict[str, Any] | None] | None = None,
    caps: CapRegistry = CAPS,
) -> ScoreResult:
    """纯函数：(四件套, 轨迹, 沙盒) -> 分数。不碰网络、不碰模型、可完全复现。

    policy_scorer 和 decision_fn 由域实现注入——政策怎么算是广告域的知识，
    引擎不该知道。不注入时 policy 子分记满分（适合还没建政策库的早期 case）。
    """
    spec = bundle.verifier
    decision = decision_fn(bundle) if decision_fn else None

    # ---- 顶层行为不对，直接零分。clarify 该问却动手、reject 该拒却执行，都属于这类 ----
    if trajectory.behavior != spec.expected_behavior:
        return ScoreResult(
            reward=0.0,
            raw_reward=0.0,
            subscores={name: 0.0 for name in WEIGHTS},
            cap_hits=[CapHit("behavior_mismatch", 0.0,
                             f"expected={spec.expected_behavior} actual={trajectory.behavior}", [])],
            cap_steps={"behavior_mismatch": []},
            details={"behavior": {"expected": spec.expected_behavior, "actual": trajectory.behavior}},
        )

    evidence, evidence_detail = score_evidence(spec, trajectory)
    outcome, outcome_detail = score_outcome(spec, bundle, trajectory, sandbox, decision)
    efficiency, efficiency_detail = score_efficiency(spec, trajectory)
    if policy_scorer and spec.policy_required:
        policy, policy_detail = policy_scorer(bundle, trajectory, sandbox)
    else:
        policy, policy_detail = 1.0, {"skipped": True}

    subscores = {"outcome": outcome, "policy": policy, "evidence": evidence, "efficiency": efficiency}
    raw_reward = sum(subscores[name] * weight for name, weight in WEIGHTS.items())

    # ---- caps：取所有命中里最狠的那个上限 ----
    # spec.active_caps 为 None 表示全启用，空列表表示全关闭——原样透传，不要用 `or None`
    hits = caps.evaluate(bundle, trajectory, sandbox, enabled=spec.active_caps)
    reward = raw_reward
    if hits:
        reward = min(raw_reward, min(hit.ceiling for hit in hits))

    return ScoreResult(
        reward=round(max(0.0, min(1.0, reward)), 6),
        raw_reward=round(raw_reward, 6),
        subscores=subscores,
        cap_hits=hits,
        cap_steps={hit.name: hit.steps for hit in hits},
        details={
            "outcome": outcome_detail,
            "policy": policy_detail,
            "evidence": evidence_detail,
            "efficiency": efficiency_detail,
        },
    )
