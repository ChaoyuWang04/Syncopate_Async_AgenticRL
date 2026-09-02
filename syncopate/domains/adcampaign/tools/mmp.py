"""MMP（AppsFlyer 口径）—— 第二个数据源，以及它和平台口径为什么会打架。

★★★ 差异不是噪声，是**有确定成因和方向的**

原计划是给两个源加一个随机偏差来"模拟数据源打架"。查完 AppsFlyer 官方文档
发现那是假的。真实的头号成因是**归因窗口配置不一致**：

    Meta 默认       7 天点击 + 1 天浏览
    AppsFlyer 若配   1 天点击

  ⇒ 点击广告后 **2–7 天**才首次打开 App 的用户，
    在 AppsFlyer 里算**自然量**，在 Meta 里算**投放带来的**。

所以差异的方向是可预测的（Meta 报得多、AF 报得少），而且**可以被解释**。
模型该学的是「识别成因并据此判断信谁」，不是「识别噪声」。
给随机偏差的话，模型只能学会"两个数不一样时取平均"——那是错的。

★★ 为什么以 MMP 为准（附录 A7）

平台后台有**自归因**偏向：它既是投放方也是记账方。MMP 是第三方，
口径统一且跨平台可比。行业通行做法是「以 MMP 为准做决策，但两个都必须查」——
差异本身就是信号（配置错了、或者有作弊），不查就发现不了。
"""

from __future__ import annotations

from typing import Any

from syncopate.core.tool_registry import REGISTRY, ToolContext, ToolResult

# 各归因窗口能"认领"到的转化占比。以 Meta 默认（7 天点击 + 1 天浏览）为 1.00。
# 窗口越短，认领到的越少，剩下的会被算成自然量。
WINDOW_COVERAGE: dict[str, float] = {
    "7d_click_1d_view": 1.00,     # Meta 默认
    "7d_click": 0.94,
    "1d_click_1d_view": 0.74,
    "1d_click": 0.68,             # AppsFlyer 常见的保守配置
}

# Meta 侧固定用它的默认窗口
PLATFORM_WINDOW = "7d_click_1d_view"

# 差异超过这个比例就必须标注 caveat 并降 confidence（附录 A7）
DISCREPANCY_THRESHOLD = 0.15


def coverage(window: str) -> float:
    return WINDOW_COVERAGE.get(window, 1.0)


def discrepancy(mmp_window: str) -> float:
    """MMP 相对平台口径少认领了多少（0.32 = 少 32%）。"""
    return 1.0 - coverage(mmp_window) / coverage(PLATFORM_WINDOW)


@REGISTRY.tool(
    name="mmp.get_attribution",
    description=(
        "查 MMP（第三方归因平台）口径下的安装与回收数据，返回里带 attribution_window。它和平台后台口径（campaign.get_metrics，自归因）会有差异，最常见成因是两边归因窗口不一致，做判断前先看窗口是否一致。不含平台侧花费和曝光。"
    ),
    parameters={
        "type": "object",
        "properties": {"campaign_id": {"type": "string"}},
        "required": ["campaign_id"],
    },
    kind="read",
    api_ref="appsflyer:GET /export/{app_id}/partners_report",
)
def get_attribution(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    row = ctx.row("campaigns", args.get("campaign_id"))
    if row is None:
        return ToolResult(ok=False, error=f"campaign_not_found: {args.get('campaign_id')}")
    window = row.get("mmp_attribution_window", PLATFORM_WINDOW)
    ratio = coverage(window) / coverage(PLATFORM_WINDOW)

    platform_installs = float(row.get("installs_7d") or 0.0)
    installs = int(round(platform_installs * ratio))
    spend = float(row.get("spend_7d") or 0.0)
    # 花费两边一样（钱是平台扣的），但安装数少了 ⇒ CPI 变高、ROAS 变低。
    # 这就是"同一条 campaign，两个源给出不同结论"的具体形态。
    return ToolResult(ok=True, data={
        "campaign_id": row["campaign_id"],
        "source": "mmp",
        "attribution_window": window,
        "platform_attribution_window": PLATFORM_WINDOW,
        "installs_7d": installs,
        "cpi": round(spend / installs, 4) if installs else None,
        "roas_d7": round(float(row.get("roas_d7") or 0.0) * ratio, 4),
        # ★ 差额去哪了：被算成自然量了，不是丢了
        "organic_installs_7d": int(round(platform_installs - installs)),
    })
