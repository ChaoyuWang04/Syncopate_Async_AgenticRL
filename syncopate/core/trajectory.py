"""一条 rollout 的完整记录。

verifier 只吃这个对象 + 四件套，不碰模型、不碰网络，是个纯函数。
纯函数意味着：同一条轨迹每次算分结果完全一样。这对我们研究 reward 方差很重要——
如果打分本身带随机性（比如 LLM judge），测出来的方差就分不清是策略的还是噪声的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """模型发起的一次工具调用。

    step 是 1-indexed 的 assistant 轮号，**在这里显式存下来**，
    这样「第 k 步做了什么」是免费可得的，不需要任何事后推断。
    """

    step: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """工具返回给模型的东西。靠 tool_call_id 和 Action 一一对应。"""

    tool_call_id: str
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trajectory:
    case_id: str
    rollout_id: str
    namespace_id: str
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    # 顶层行为：模型最后选择了做什么
    #   tool_call —— 正常执行完并给出结论
    #   clarify   —— 信息不足，反问用户
    #   reject    —— 越权/越域，拒答
    #   defer     —— 数据还没收敛，明确说 X 天后再判（M1 新增）
    behavior: str = "tool_call"

    # 结构化终答。我们不要 LLM judge，靠的就是这个必须是可解析的 dict。
    final_answer: dict[str, Any] = field(default_factory=dict)
    # 原始终答文本，解析失败时留证据
    final_text: str = ""
    parse_ok: bool = True

    # ★★ 2026-08-18：截断的**原因**。`truncated` 只说"没走到终答"，
    # 而它有四个出口、三种成因，**修法方向完全不同甚至相反**：
    #     "tokens"       模型自己把 token 预算写满了      ⇒ 加 token 预算
    #     "observation"  工具返回塞不进剩余预算（不是模型的锅）⇒ 截断 observation
    #     "turns"        轮数用完还没给终答                ⇒ 缩链路 / 加轮数 / 查为什么打转
    # ⚠️ 此前四个出口共用一个布尔值 ⇒ 数据里**根本不存在**这个区分，只能事后猜；
    #   2026-08-18 就因此按错的假设（`num_steps>=8`，而真实上限逐 case 是 4–14）
    #   得出过一个整个反了的结论。
    # ⚠️ `truncated` 保留（= `truncation_reason is not None`）：下游消费者很多，
    #   而且历史 dump 里没有新字段，必须照读。判据在 check_pipeline_invariants。
    truncation_reason: str | None = None

    # 每个 token 属于第几步。token→step 的映射表，步级信用分配要用。
    token_trace: dict[str, Any] = field(default_factory=dict)

    truncated: bool = False   # 撞上 max_steps 被截断

    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return max((a.step for a in self.actions), default=0)

    @property
    def business_actions(self) -> list["Action"]:
        """排除 session.* 信令后的动作 —— 「模型有没有对世界动手」的唯一口径。

        信令族是**零副作用**的：它们只是把"我要等/我要问/我要拒/我要报数"变成
        一个可被编排的动作。把它们算成"动手了"，等于换个契约就凭空触发
        `acted_when_should_not` 这类 cap（R1 门槛⑤ 实测：clarify/reject 全线误判）。
        ⚠️ v14 轨迹里没有 session.* action ⇒ 与 actions 恒等，旧判据逐位不变。
        """
        from syncopate.core.contract import SESSION_TOOL_NAMES
        return [a for a in self.actions if a.name not in SESSION_TOOL_NAMES]

    @property
    def num_business_steps(self) -> int:
        """★ 只数**业务**工具占用的步数，排除 session.* 信令族。

        效率子分问的是「办这件事有没有绕路」。v15 里 `session.report` 是**契约要求的
        报数动作**，不是绕路；把它算进去等于换个契约就凭空扣 5% ——
        实测（R1 门槛⑤ 判分对拍）：不排除的话 120/120 条 gold 的 efficiency 全部变化。
        ⚠️ v14 轨迹里没有 session.* action ⇒ 本属性与 num_steps 恒等，**旧分不动**。
        """
        return max((a.step for a in self.business_actions), default=0)

    def called_tools(self) -> list[str]:
        return [a.name for a in self.actions]

    def steps_by_tool(self, tool: str) -> list[int]:
        return [a.step for a in self.actions if a.name == tool]

    def observation_for(self, tool_call_id: str) -> Observation | None:
        for obs in self.observations:
            if obs.tool_call_id == tool_call_id:
                return obs
        return None

    def multi_tool_steps(self) -> list[int]:
        """同一步发了多个工具调用的步号。

        老师那套算出来之后只 return 了个布尔值，把「哪一步」扔掉了
        （docs/syncopate/02 §3）。我们从一开始就留着。
        """
        counts: dict[int, int] = {}
        for action in self.actions:
            counts[action.step] = counts.get(action.step, 0) + 1
        return sorted(step for step, count in counts.items() if count > 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "rollout_id": self.rollout_id,
            "namespace_id": self.namespace_id,
            "behavior": self.behavior,
            "actions": [vars(a) for a in self.actions],
            "observations": [vars(o) for o in self.observations],
            "final_answer": self.final_answer,
            "final_text": self.final_text,
            "parse_ok": self.parse_ok,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "num_steps": self.num_steps,
        }
