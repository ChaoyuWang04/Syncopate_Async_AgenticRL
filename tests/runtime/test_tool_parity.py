"""工具对齐账本（B-5）：缺口必须**被登记**，不能是静默的。

★★★ 为什么这份判据要在补工具**之前**立

`09 §1-③`：**沙盒是 runtime 的子集，契约由 runtime 定义。**
同一个工具两边行为不一致，**训出来的策略在线上就不成立** —— 而且不会报错。

`[实测 2026-08-19]` 训练侧 **30** 个工具，runtime 真正实现的只有 **2** 个。
⚠️ 而 `tools.WRITE_TOOLS` 里登记了 **8** 个写工具的权限与幂等 ——
   **其中 7 个根本没有实现。** 登记表填了，实现没有。
⇒ 这正是本项目的第一失效形状：**机制在，但没接上。**
⇒ 而它此前**没有任何东西在喊**，因为"没实现"不报错 ——
  它只是让某个工具在 runtime 里不存在，在训练里存在。

判据形状是「**某集合应当完整**」（守则①）：非黑即白，不需要阈值。
"""

from __future__ import annotations

import pytest

from syncopate.runtime.tool_parity import (
    IMPLEMENTED, KNOWN_GAPS, coverage_report, sandbox_tools, signature_mismatch,
)


def test_every_sandbox_tool_is_either_implemented_or_ledgered():
    """★★ 核心判据：沙盒的每个工具，要么实现了，要么**写下来说没实现**。

    ⚠️ 缺口本身**不算失败** —— 它是被承认的债。
      但**必须写下来才算被承认**：没写的那些，才是真正危险的。
    """
    r = coverage_report()
    assert not r["unledgered"], (
        f"这些沙盒工具在账本里一个字都没提：{r['unledgered']}\n"
        f"⇒ 要么实现它，要么写进 KNOWN_GAPS 说清楚在等谁")


def test_the_ledger_has_no_stale_entries():
    """账本里有、沙盒却没有 ⇒ 名字写错了，或者沙盒删了工具。

    ★ 这条挡的是"名字写错"——写错的名字会让一个工具**看起来已登记**，
      而它实际从没被覆盖到。同 `select_sft_ckpt` 那次匹配错对象的形状。
    """
    r = coverage_report()
    assert not r["stale_ledger"], f"账本里这些沙盒没有：{r['stale_ledger']}"


def test_a_tool_cannot_be_both_implemented_and_a_gap():
    r = coverage_report()
    assert not r["both"], f"同时出现在两边：{r['both']}"


def test_the_ledger_covers_exactly_the_sandbox():
    r = coverage_report()
    assert len(IMPLEMENTED) + len(KNOWN_GAPS) == r["sandbox_total"]


def test_every_gap_says_why_not_just_the_name():
    """★ 只写工具名不够 —— 那会变成一张永远不会被清空的名单。

    写清楚归属，才知道它在等谁（`observed-needs-an-owner` 那条记忆）。
    """
    thin = [k for k, v in KNOWN_GAPS.items() if len(v.strip()) < 8]
    assert not thin, f"这些缺口只写了名字没写原因：{thin}"


def _bare_worker():
    """只为拿到绑定表 —— 不连库、不起服务。

    ⚠️ 用 `__new__` 跳过 `__init__`，所以**依赖要手工补齐**；
       漏一个就是 AttributeError（一次就抓到了 retrieval）。
    """
    from syncopate.runtime.db import Database
    from syncopate.runtime.platform import FakeAdPlatform
    from syncopate.runtime.retrieval import RetrievalService
    from syncopate.runtime.worker import Worker
    w = Worker.__new__(Worker)
    w.db = Database()
    w.platform = FakeAdPlatform()
    w.retrieval = RetrievalService(w.db)
    from syncopate.runtime.worker import WorkerConfig
    w.config = WorkerConfig()
    return w


# ── 已实现的那些，签名必须对得上 ────────────────────────────────────────

def test_implemented_tools_accept_the_sandbox_required_params():
    """★★ 声称实现了，就要接得住沙盒 spec 的**必填参数**。

    接不住的话调用直接 TypeError —— 是硬错，但要跑到那一步才看得见。
    ⇒ 这里静态地把它挡在前面。
    """
    w = _bare_worker()
    problems = [signature_mismatch(name, b.invoke)
                for name, b in w._bindings().items()]
    assert not [p for p in problems if p], [p for p in problems if p]


def test_the_binding_table_and_the_ledger_agree():
    """★ 绑定表（真的有实现）和账本的 IMPLEMENTED **必须一致**。

    两份名单各写各的，就会出现"账本说实现了、绑定表里没有"——
    而那种不一致**恰恰是本模块要防的东西**，不能自己先犯。
    """
    assert set(_bare_worker()._bindings()) == IMPLEMENTED


def test_the_judge_can_actually_fail():
    """★ 判据必须**有能力失败** —— 否则只是个永远绿的装饰。"""
    async def _no_params():
        ...

    assert signature_mismatch("campaign.update_budget", _no_params) is not None
    assert signature_mismatch("不存在的工具", _no_params) is not None


def test_write_tools_must_accept_the_idempotency_key():
    """写工具接不住 `idempotency_key` ⇒ **幂等保护在这个工具上静默失效**。

    ⚠️ 反面教材就在隔壁：`WRITE_TOOLS` 漏登记一个写工具，代价不是报错，
       而是那个写动作悄悄没有幂等保护。同一个形状。
    """
    async def _write_without_key(*, campaign_id: str, new_budget: int,
                                 client_request_id: str):
        ...

    msg = signature_mismatch("campaign.update_budget", _write_without_key)
    assert msg is not None and "idempotency_key" in msg


# ── ★ 把"登记了但没实现"这件事本身钉住 ──────────────────────────────────

def test_registered_for_permission_is_not_the_same_as_implemented():
    """★★ `WRITE_TOOLS` 里有名字 ≠ 这个工具存在。

    `[实测 2026-08-19]` 8 个登记了权限/幂等的写工具里，**7 个没有实现**。
    ⇒ 这条测试不是要把它判红（那是已知的债），
      是要**让"登记≠实现"这个区别有一处写下来** ——
      否则下一个人看到 `WRITE_TOOLS` 有 8 个，会以为 runtime 支持 8 个写工具。
    """
    from syncopate.runtime.tools import WRITE_TOOLS
    registered_but_absent = set(WRITE_TOOLS) - IMPLEMENTED
    assert registered_but_absent <= set(KNOWN_GAPS), (
        f"这些写工具登记了权限却既没实现、也没登记成缺口："
        f"{registered_but_absent - set(KNOWN_GAPS)}")
