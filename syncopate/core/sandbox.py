"""有状态沙盒：记录本次 rollout 真正改变了什么。

这是整个项目最关键的一块。没有它，「模型说它改了预算」和「模型真的改了预算」
就区分不开——而这个区分正是可验证 reward 的全部基础。

三条设计规则：

1. **世界只读，改动入账**。EnvSnapshot 永远不被修改；所有写动作进 Sandbox 的台账。
2. **命名空间隔离**。同一个 case 会被并发跑 N 条 rollout，各自的写动作绝不能串台。
3. **审计日志是唯一真相**。老师那套里覆盖式台账（后写覆盖先写）会**物理抹掉**
   前一次写的记录，连带它的 step 号一起丢——这是谓词翻转历史的真实丢失点。
   我们反过来：`audit_log` 永远 append-only，台账只是它的一个派生视图。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from syncopate.core.schemas import EnvSnapshot


@dataclass
class WriteRecord:
    """一条写动作记录。

    step 和 tool_call_id 是**从第一天就存下来**的，不是事后补的。
    有了它们，「哪一步碰了 cap」才归因得到（见 docs/syncopate/02 §3）。
    """

    tool: str
    fact_key: str            # 这个写工具翻转的谓词，如 "budget_updated"
    arguments: dict[str, Any]
    result: dict[str, Any]
    step: int                # 第几个 assistant 轮，1-indexed
    tool_call_id: str
    ok: bool = True
    object_key: str | None = None   # 被写对象的主键，如 campaign_id。查重复写用
    # 这次写改了世界的哪张表哪一行的哪些字段。读工具的叠加视图靠它。
    mutation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "fact_key": self.fact_key,
            "arguments": self.arguments,
            "result": self.result,
            "step": self.step,
            "tool_call_id": self.tool_call_id,
            "ok": self.ok,
            "object_key": self.object_key,
            "mutation": self.mutation,
        }


class Sandbox:
    """一条 rollout 的写动作账本。

    用法：
        sandbox = Sandbox(env, namespace_id="run:case:rollout3")
        sandbox.record_write(tool="campaign.update_budget", fact_key="budget_updated", ...)
        sandbox.facts()                       -> {"budget_updated"}
        sandbox.records_for("campaign.update_budget")  -> [WriteRecord, ...]
    """

    def __init__(self, env: EnvSnapshot, namespace_id: str) -> None:
        self.env = env
        self.namespace_id = namespace_id
        # 唯一真相：append-only，任何写动作（成功或失败）都留痕
        self.audit_log: list[WriteRecord] = []
        # 每个工具被调用了几次。失败剧本按「该工具的第几次调用」匹配 ——
        # 按 step 匹配的话，模型多插一次读工具就会错开，剧本形同虚设。
        self.call_counts: dict[str, int] = {}
        # 出现在工具返回里的对象主键。★ 防注入用（设计文档 §37 的 param_source）：
        # 「拿工具返回里读来的 id 去做写动作」是一条可判定的规则。
        self.ids_seen_in_output: set[str] = set()

    def note_call(self, tool: str) -> int:
        """记一次调用，返回这是该工具的第几次（1-indexed）。"""
        self.call_counts[tool] = self.call_counts.get(tool, 0) + 1
        return self.call_counts[tool]

    # ---------------------------------------------------------------- 写入

    def record_write(
        self,
        *,
        tool: str,
        fact_key: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        step: int,
        tool_call_id: str,
        ok: bool = True,
        object_key: str | None = None,
        mutation: dict[str, Any] | None = None,
    ) -> WriteRecord:
        record = WriteRecord(
            tool=tool,
            fact_key=fact_key,
            arguments=copy.deepcopy(arguments),
            result=copy.deepcopy(result),
            step=step,
            tool_call_id=tool_call_id,
            ok=ok,
            object_key=object_key,
            mutation=copy.deepcopy(mutation) if mutation else None,
        )
        self.audit_log.append(record)
        return record

    # ---------------------------------------------------------------- 查询

    def records_for(self, tool: str, *, only_ok: bool = True) -> list[WriteRecord]:
        """某个写工具的全部记录，按发生顺序。重复写检测直接数这个列表长度。"""
        return [r for r in self.audit_log if r.tool == tool and (r.ok or not only_ok)]

    def facts(self, *, only_ok: bool = True) -> set[str]:
        """本次 rollout 真实翻转的谓词集合。dry-run 和失败的写不算。"""
        return {r.fact_key for r in self.audit_log if r.ok or not only_ok}

    def executed_write_tools(self, *, only_ok: bool = True) -> set[str]:
        return {r.tool for r in self.audit_log if r.ok or not only_ok}

    def current_state(self, fact_key: str, object_key: str | None = None) -> WriteRecord | None:
        """覆盖式语义的派生视图：同一对象被写多次时返回**最后一次**。

        注意这只是个视图——历史全在 audit_log 里，一条都没丢。
        """
        matches = [
            r
            for r in self.audit_log
            if r.fact_key == fact_key and r.ok and (object_key is None or r.object_key == object_key)
        ]
        return matches[-1] if matches else None

    # ---------------------------------------------------------------- 叠加视图
    #
    # ★ 世界只读、改动入账，但**读的时候必须把账本叠加回去**。
    # 实测缺口：改预算 500 → 900 之后再查，读到的还是 500 ——
    # 因为读工具只看 env、不看账本。真实平台（Meta 明确支持 read-after-write）
    # 改完再读会读到新值。读不到的话，模型学不会「改完要确认」，
    # 还可能因为一直读到旧值而反复改同一个对象。

    def find_by_request_id(self, tool: str, request_id: str) -> WriteRecord | None:
        """幂等查重：这个 client_request_id 之前提交过吗。

        只认**成功**的记录 —— 上次失败了就该允许重试。
        """
        for record in self.audit_log:
            if (record.tool == tool and record.ok
                    and str(record.arguments.get("client_request_id")) == request_id):
                return record
        return None

    def mutations_for(self, table: str, key: str) -> dict[str, Any]:
        """(表, 行) 上已生效的字段改动，按发生顺序叠加 —— 后写的覆盖先写的。"""
        merged: dict[str, Any] = {}
        for record in self.audit_log:
            m = record.mutation
            if record.ok and m and m.get("table") == table and m.get("key") == key:
                merged.update(m.get("fields") or {})
        return merged

    def mutated_keys(self, table: str) -> set[str]:
        return {r.mutation["key"] for r in self.audit_log
                if r.ok and r.mutation and r.mutation.get("table") == table}

    def duplicate_writes(self) -> dict[str, list[int]]:
        """返回 {tool: [重复发生的 step 号]}。同一 (tool, object_key) 写了 >1 次即算重复。

        这是 duplicate_side_effect cap 的归因来源——直接给出「哪几步重复了」。
        """
        seen: dict[tuple[str, str | None], list[int]] = {}
        for r in self.audit_log:
            if not r.ok:
                continue
            seen.setdefault((r.tool, r.object_key), []).append(r.step)
        return {tool: steps for (tool, _), steps in seen.items() if len(steps) > 1}

    # ---------------------------------------------------------------- 导出

    def export(self) -> dict[str, Any]:
        """落盘用的完整快照。rollout 结束后交给 verifier 和 artifact writer。"""
        return {
            "namespace_id": self.namespace_id,
            "audit_log": [r.to_dict() for r in self.audit_log],
            "facts": sorted(self.facts()),
            "executed_write_tools": sorted(self.executed_write_tools()),
        }
