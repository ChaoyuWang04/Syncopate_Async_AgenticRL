"""线上「当前投放任务」context 的**唯一**构造函数（decider 与训练建库共用）。

★ 09-02 Chaoyu 裁定：一次真实请求只带 **account_id**（租户是谁）；用户若在界面选了某个 campaign 才带
  campaign_id。除此之外模型一无所知——有哪些 campaign 调 campaign.list 翻页、指标调 get_metrics，
  工具就是为这个存在的。真实账户下有几万个 campaign 且每天在变，**不可能塞进提示词**。
⛔ 08-20 之前这里把 demo 租户的 7 条 campaign 清单整段塞进 context（当时是给演示环境打的补丁：
  线上只给写死的 CMP_1，模型只能编 product_id），被当成产品形状留了两周，W2 又把它复制进训练数据。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

PLATFORM_STATE = "data/demo/platform_state.json"


def demo_context(path: str = PLATFORM_STATE, campaign_id: str | None = None) -> dict[str, Any]:
    f = pathlib.Path(path)
    acc = "ACC_DEMO"
    if f.is_file():
        acc = json.loads(f.read_text(encoding="utf-8")).get("account_id", acc)
    ctx: dict[str, Any] = {"account_id": acc}
    if campaign_id:
        ctx["campaign_id"] = campaign_id          # 用户在界面上选中的那条（可选）
    return ctx
