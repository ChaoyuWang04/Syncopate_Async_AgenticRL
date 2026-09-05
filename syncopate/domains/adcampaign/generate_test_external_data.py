"""生成测试用的外部资料：安全线 Excel + 素材图 + 离线视觉标签。

模拟真实的业务资料流：

    运营每周更新 Excel（产品 × 地域 的安全线）
    素材库里躺着一堆图/视频
    离线跑一次 VLM 给素材打标签
              ↓
        ingest 脚本转成 env 的只读表
              ↓
        rollout 时用工具查（不在线跑视觉模型）

★ 为什么不做在线 RAG：安全线是拿来做**判断**的（`cpi > ceiling` 就告警），
数值必须精确。检索出文本让模型自己读数一定会读错，而且不可验证——
那样就又得上 LLM judge，我们好不容易才把它去掉。

    python -m syncopate.domains.adcampaign.generate_test_external_data
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "external"

# 产品轴：一个发行商旗下的几款游戏
PRODUCTS = [
    ("PUZ_QUEST", "puzzle"),
    ("MERGE_FARM", "casual"),
    ("IDLE_HERO", "rpg"),
    ("TAP_RUSH", "hyper_casual"),
    ("WAR_THRONE", "strategy"),
]
# 地域轴
REGIONS = ["US", "GB", "DE", "JP", "BR"]
PLATFORMS = ["Meta", "Google", "TikTok", "AppLovin", "Unity"]

CURRENT_WEEK = "2026-W32"

# ★ M2：安全线是**有有效期**的资料，不是永久真理。
#
# 运营每周更新一次，一份表的有效期就是那一周（周一到周日）+ 一点宽限。
# 没有有效期，「模型拿着上上周的线做决策」这类错误就构造不出来，
# 而设计文档 §14 把「过期检出率」列为本业务 RAG 最重要的两项之一
# （理由：政策/安全线错了是合规事故，不是分数低一点的问题）。
#
# ⚠️ 有效期只是**一个普通字段**，工具不会替模型判断过没过期 ——
# 真实世界里没人会在返回里塞一个 `expired: true`。模型必须自己拿它和今天比。
# 这是「沙盒不能比真实世界更友好」那条纪律的又一次应用。
WEEK_DATES = {
    "2026-W30": ("2026-07-20", "2026-07-26"),
    "2026-W31": ("2026-07-27", "2026-08-02"),
    "2026-W32": ("2026-08-03", "2026-08-09"),
}
# 宽限期：过了有效期不是立刻作废，运营常常晚一两天才更新。
# 给 3 天，让「刚过期」和「过期两周」成为两种不同难度的题。
GRACE_DAYS = 3
# 除了当周，再生成一份两周前的旧表 —— 有它才能构造「拿着旧线做决策」的 case。
STALE_WEEK = "2026-W30"
# 旧周的线整体松一档。松紧方向和真实业务一致：产品早期跑得差、线定得松，后来收紧。
# **数值必须真的不同**，否则用旧线和用新线得出同一个结论，这类题就成了摆设。
_WEEK_DRIFT = {"2026-W30": 1.18, "2026-W31": 1.09, "2026-W32": 1.00}

# 地域的获客成本系数（成熟市场贵、新兴市场便宜）
_REGION_COST = {"US": 1.00, "GB": 0.88, "DE": 0.82, "JP": 1.35, "BR": 0.35}
# 品类系数
_GENRE_COST = {"puzzle": 1.00, "casual": 0.80, "rpg": 1.85, "hyper_casual": 0.45, "strategy": 1.60}


def safety_lines(week: str = CURRENT_WEEK) -> list[dict]:
    """产品 × 地域 的安全线，带有效期。

    ★ 不同周的数值必须**真的不一样**，否则「用了旧线」和「用了新线」得出同一个结论，
    这类题就成了摆设 —— 判据分辨不出模型有没有真的看有效期。
    这里让旧周的线整体松一档（`_WEEK_DRIFT`），松紧方向和真实业务一致：
    产品早期跑得差、线定得松，后来收紧。
    """
    start, end = WEEK_DATES[week]
    valid_to = (date.fromisoformat(end) + timedelta(days=GRACE_DAYS)).isoformat()
    drift = _WEEK_DRIFT[week]
    rows = []
    for product, genre in PRODUCTS:
        for region in REGIONS:
            factor = _REGION_COST[region] * _GENRE_COST[genre]
            cpi_ceiling = round(2.20 * factor * drift, 2)
            # ROAS 和成本反向：贵的市场回收要求相对低一点
            roas_floor = round(0.42 / math.sqrt(factor) / drift, 3)
            rows.append({
                "product_id": product,
                "genre": genre,
                "region": region,
                "week": week,
                "valid_from": start,
                "valid_to": valid_to,               # ★ 过了这天就不该再拿它当依据
                "d7_cpi_ceiling": cpi_ceiling,      # 超过这条线要告警
                "d7_roas_floor": roas_floor,        # 低于这条线要告警
                "d1_retention_floor": round(0.34 - 0.03 * math.log(factor + 1), 3),
                "daily_budget_cap": int(3000 * factor * drift / 100) * 100,
                "updated_by": "ua_ops",
            })
    return rows


def write_excel(rows: list[dict], path: Path, week: str = CURRENT_WEEK) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    # ★ 钉死内嵌时间戳（2026-08-19 影子重建抓到的）：openpyxl 默认把「现在」写进
    #   docProps/core.xml ⇒ 同样的内容每次重跑字节都不同，「逐字节可复现」在 xlsx
    #   这一环静默断掉。产物的一切都必须由输入决定，时钟不是输入。
    from datetime import datetime
    workbook.properties.created = datetime(2026, 1, 1)
    workbook.properties.modified = datetime(2026, 1, 1)
    sheet = workbook.active
    sheet.title = f"safety_lines_{week}"
    headers = list(rows[0])
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([row[h] for h in headers])
    for index, header in enumerate(headers, start=1):
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = max(14, len(header) + 2)
    workbook.save(path)


# --------------------------------------------------------------------------
# 素材 + 离线视觉标签
# --------------------------------------------------------------------------

# 模拟 VLM 离线打标的产物。真实流程是 图/视频首帧 -> Qwen3-VL -> 结构化属性，
# 这里直接给结果，因为**在线 rollout 不该跑视觉模型**（每步都过一遍 VLM 训练会慢到没法用）。
_THEMES = [
    ("halloween", ["pumpkin", "dark_palette", "spooky"], "orange"),
    ("christmas", ["snow", "gift_box", "warm_light"], "red"),
    ("summer", ["beach", "bright_palette", "water"], "cyan"),
    ("generic", ["ui_showcase", "neutral_palette"], "blue"),
    ("lunar_new_year", ["lantern", "red_envelope", "festive"], "red"),
]
_HOOKS = ["before_after", "gameplay_first_3s", "fail_then_win", "ugc_testimonial", "tutorial"]

# ==========================================================================
# ★★★ M3 · feature 归因的地基：把"规律"埋进素材表现的生成方式里
# ==========================================================================
#
# 归因任务和前面几类的根本不同：前面的"正确做法"是**流程性**的（先查什么再查什么），
# 判据可以逐步核对；归因的结论是**一个判断**（"真人出镜在美国有效"），
# 这句话对不对没法从流程里推出来。
#
# 两条路：(A) 沙盒里预埋真实规律，工具把它算出来；(B) 上 LLM judge。
# 选 A —— 项目一开始就定了不用 judge（终答必须是可解析的 dict，false_claim_cap
# 那套全靠这个）。而且 A 让标准答案**可推导**，不需要人来标。
#
# ★ 但"预埋"不等于"把答案写进表里"。沿用同文件 detect_anomalies 的做法：
#   **真算，不读预置答案** —— 规律埋在 roas 的生成公式里，
#   `analysis.feature_lift` 从素材表现里重新算出来。
#   于是改一条素材的数值，归因结论就自然跟着变，不用手工维护一张"正确答案表"。

# 归因看的五个 feature（都是二值的，从视觉标签派生）
FEATURES = ["real_person", "before_after", "dark_palette", "fast_cut", "ugc_style"]

# ★★ feature × 地域 的真实效应（对 d7 ROAS 的相对提升）。
#
# 刻意做成**跨地域不一致**：real_person 在 US 强正、在 JP 负、在 BR 几乎没用。
# 这样"按地域分别归因"就成了必须做的事，而不是可以偷懒平均掉的 ——
# 一个把五个地域混在一起算的模型，会得出 real_person 微弱正向的结论，
# 而那个结论在 JP 是**反的**，照着它扩量就是真金白银的亏损。
_FEATURE_EFFECT: dict[tuple[str, str], float] = {
    ("real_person", "US"): 0.26, ("real_person", "GB"): 0.19,
    ("real_person", "DE"): 0.04, ("real_person", "JP"): -0.21, ("real_person", "BR"): 0.02,
    ("before_after", "US"): 0.12, ("before_after", "GB"): 0.14,
    ("before_after", "DE"): 0.17, ("before_after", "JP"): 0.15, ("before_after", "BR"): 0.11,
    ("dark_palette", "US"): -0.09, ("dark_palette", "GB"): -0.06,
    ("dark_palette", "DE"): 0.01, ("dark_palette", "JP"): 0.22, ("dark_palette", "BR"): -0.03,
    ("fast_cut", "US"): 0.03, ("fast_cut", "GB"): 0.02,
    ("fast_cut", "DE"): 0.01, ("fast_cut", "JP"): 0.05, ("fast_cut", "BR"): 0.28,
    ("ugc_style", "US"): 0.15, ("ugc_style", "GB"): 0.13,
    ("ugc_style", "DE"): 0.12, ("ugc_style", "JP"): 0.09, ("ugc_style", "BR"): 0.16,
}

# ★ 每个 (feature, 地域) 在素材库里有多少条 —— 决定"样本够不够下结论"。
#
# 这是 M3 的第二个教学目标，和第一个同等重要：
#   样本足 + 效应大  ⇒ 该下结论
#   样本不足        ⇒ **该拒绝下结论**，哪怕算出来的 lift 看着很漂亮
# 设计文档给 feature_lift 的原话是"让模型学不会拿 3 个样本下结论"。
#
# 所以刻意留几个**样本稀少但效应巨大**的格子（JP 的 real_person 只有 4 条，
# 效应 -0.21）—— 只看 lift 的模型会一头撞上去，看样本量的才躲得开。
_SPARSE_CELLS = {("real_person", "JP"), ("dark_palette", "BR"), ("fast_cut", "DE")}
# 稀疏格子留几条。4 条：足够算出一个数（不是空的），又明显不够下结论。
SPARSE_COUNT = 4

# 素材库规模。250 条 / 5 地域 = 每地域 50 条，一个 feature 出现率 ~40% ⇒ 约 20 条样本，
# 够做归因；稀疏格子压到 4 条，明确不够。
CATALOG_SIZE = 250
# 少于这么多条素材就不该下归因结论。和 maturity.MIN_SAMPLE_INSTALLS 是同一个思路的
# 另一个维度：那个管"时间够不够"，这个管"样本够不够"。
MIN_FEATURE_SAMPLE = 12
# 地域基准 ROAS（不带任何 feature 的素材大致落在这里）
_REGION_BASE_ROAS = {"US": 0.52, "GB": 0.50, "DE": 0.48, "JP": 0.61, "BR": 0.38}


def _features_for(index: int, theme: str, hook: str) -> dict[str, bool]:
    """这条素材天然带哪几个 feature（不考虑稀疏格子，那个在第二遍处理）。"""
    return {
        "real_person": index % 5 in (0, 2),
        "before_after": hook == "before_after" or index % 7 == 1,
        "dark_palette": theme in ("halloween", "lunar_new_year"),
        "fast_cut": index % 4 == 1,
        "ugc_style": hook == "ugc_testimonial" or index % 11 == 3,
    }


def _thin_sparse_cells(rows: list[dict]) -> None:
    """把 `_SPARSE_CELLS` 里的 (feature, 地域) 削到恰好 SPARSE_COUNT 条。**原地改**。

    ★ 为什么要第二遍，不能在 _features_for 里按 index 取模搞定

    地域是按 `(index*3)//5 % 5` 分配的 —— 同一个地域的 index 是**分段连续**的，
    再叠一层取模，落到某一格的条数完全不可控。第一版就是这么写的，实测
    `real_person|JP` 变成 **0 条**、`dark_palette|BR` 剩 1 条。

    而 0 条意味着那道题**根本不存在** —— 我们想要的恰恰是
    「样本少（4 条）但算出来的 lift 很唬人（-0.21）」这个陷阱：
    只看 lift 的模型会一头撞上去，看样本量的才躲得开。
    ⇒ 需要精确控制的量，就别靠取模碰运气，直接数出来。
    """
    for feature, region in sorted(_SPARSE_CELLS):
        carriers = [r for r in rows if r["region"] == region and feature in r["_features"]]
        for row in carriers[SPARSE_COUNT:]:
            row["_features"].discard(feature)


def _roas_for(region: str, features: dict[str, bool], index: int) -> float:
    """★ 规律埋在这里：ROAS = 地域基准 × Π(1 + 该 feature 在该地域的效应) × 个体扰动。

    个体扰动是**确定性**的（由 index 算出），不是随机数 ——
    数据必须可复现，而且 GRPO 是组内比较，任何随机性都会污染 advantage。
    扰动幅度 ±8%，明显小于我们要检出的效应（0.12–0.28），
    但足以让"样本量小的时候噪声淹没信号"这件事真的发生 —— 这正是要教的东西。
    """
    value = _REGION_BASE_ROAS[region]
    for feature, on in features.items():
        if on:
            value *= 1.0 + _FEATURE_EFFECT.get((feature, region), 0.0)
    jitter = 1.0 + 0.08 * math.sin(index * 2.399963)      # 黄金角，铺得均匀且确定
    return round(value * jitter, 4)


def creative_catalog() -> list[dict]:
    """素材库 + 离线标签 + 历史表现。"""
    items = []
    for index in range(CATALOG_SIZE):
        theme, tags, color = _THEMES[index % len(_THEMES)]
        product, genre = PRODUCTS[index % len(PRODUCTS)]
        region = REGIONS[(index * 3) // 5 % len(REGIONS)]
        platform = PLATFORMS[(index * 2 + index // 5) % len(PLATFORMS)]
        # 让时令素材在对应节日附近表现好——这是"万圣节素材出量"要能被推出来的依据
        ipm = round(4.0 + (6.0 if theme == "halloween" else 0.0) + (index % 5) * 0.6, 2)
        hook = _HOOKS[index % len(_HOOKS)]
        features = _features_for(index, theme, hook)
        items.append({
            # 第二遍（_thin_sparse_cells）要改它，roas 也要等它定下来才能算 ——
            # 下划线开头，落盘前删掉
            "_features": {f for f, on in features.items() if on},
            "_index": index,
            "creative_id": f"CRV_{9000 + index}",
            "creative_name": f"{theme}_{_HOOKS[index % len(_HOOKS)]}_v{1 + index % 3}",
            "product_id": product,
            "genre": genre,
            "region": region,
            "platform": platform,
            "asset_type": "video" if index % 4 else "image",
            "duration_seconds": 15 + (index % 4) * 10,
            # ---- 离线视觉标签（模拟 VLM 输出）----
            "visual_tags": [theme, *tags],
            "hook_type": hook,
            "dominant_color": color,
            "text_overlay_ratio": round(0.15 + (index % 5) * 0.08, 2),
            # ---- 历史表现（用于"这条素材以前跑过且低于安全线"的提醒）----
            "ipm": ipm,
            "ctr": round(0.010 + ipm * 0.0015, 4),
            "d7_cpi": round(3.4 - ipm * 0.12, 2),
            "week_launched": CURRENT_WEEK if index % 3 == 0 else "2026-W29",
            "image_path": f"data/external/creatives/CRV_{9000 + index}.png",
        })

    # ---- 第二遍：削稀疏格子，然后才能算 roas（它依赖最终的 feature 集合）----
    _thin_sparse_cells(items)
    for row in items:
        features = {f: f in row["_features"] for f in FEATURES}
        # ★ M3 归因的输入：五个二值 feature，摊平成独立字段而不是塞进 visual_tags ——
        # 归因要按 feature 分组，从一个混着主题词的标签数组里挑"哪些算 feature"
        # 是模型不该承担的负担。
        row["features"] = sorted(row["_features"])
        row["has_face"] = features["real_person"]
        # ★ 归因的被解释变量。规律埋在 _roas_for 里，工具**重新算**出来，不读预置答案
        row["roas_d7"] = _roas_for(row["region"], features, row["_index"])
        del row["_features"], row["_index"]
    return items


def write_placeholder_images(items: list[dict]) -> None:
    """生成占位图。真实项目里这里是素材文件本体，图内容不参与在线推理，
    只在离线打标时被 VLM 读一次。"""
    from PIL import Image, ImageDraw

    palette = {"orange": (232, 122, 32), "red": (198, 48, 48), "cyan": (32, 168, 198),
               "blue": (52, 84, 200)}
    for item in items:
        path = ROOT / item["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (360, 640), palette.get(item["dominant_color"], (90, 90, 90)))
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 340, 200], outline=(255, 255, 255), width=3)
        draw.text((32, 40), item["creative_id"], fill=(255, 255, 255))
        draw.text((32, 70), item["visual_tags"][0], fill=(255, 255, 255))
        draw.text((32, 100), item["hook_type"], fill=(255, 255, 255))
        draw.text((32, 130), f"IPM {item['ipm']}", fill=(255, 255, 255))
        image.save(path)


# --------------------------------------------------------------------------
# 时令日历
# --------------------------------------------------------------------------


def seasonal_calendar() -> list[dict]:
    """节日/时令表。lift_factor 是该时令素材的历史出量放大倍数。"""
    return [
        {"event": "halloween", "start": "2026-10-18", "end": "2026-11-02",
         "regions": ["US", "GB", "DE"], "lift_factor": 1.40, "matching_tags": ["halloween", "pumpkin"]},
        {"event": "black_friday", "start": "2026-11-20", "end": "2026-12-01",
         "regions": ["US", "GB", "DE", "BR"], "lift_factor": 1.75, "matching_tags": ["discount", "gift_box"]},
        {"event": "christmas", "start": "2026-12-10", "end": "2026-12-27",
         "regions": ["US", "GB", "DE"], "lift_factor": 1.55, "matching_tags": ["christmas", "snow", "gift_box"]},
        {"event": "lunar_new_year", "start": "2027-01-28", "end": "2027-02-12",
         "regions": ["JP", "BR"], "lift_factor": 1.30, "matching_tags": ["lunar_new_year", "lantern"]},
        {"event": "summer_peak", "start": "2026-06-15", "end": "2026-08-31",
         "regions": ["US", "GB", "DE", "JP", "BR"], "lift_factor": 1.15, "matching_tags": ["summer", "beach"]},
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ★ 两周都生成：当周 + 两周前的旧版。有旧版才能构造「拿着过期的线做决策」。
    for week in (STALE_WEEK, CURRENT_WEEK):
        lines = safety_lines(week)
        excel_path = OUT / "safety_lines" / f"{week}.xlsx"
        write_excel(lines, excel_path, week)
        tag = "（当周）" if week == CURRENT_WEEK else "（旧版，用来造过期题）"
        print(f"[OK] 安全线 Excel  -> {excel_path.relative_to(ROOT)}  ({len(lines)} 行 = "
              f"{len(PRODUCTS)} 产品 × {len(REGIONS)} 地域) 有效至 {lines[0]['valid_to']} {tag}")

    items = creative_catalog()
    (OUT / "creative_tags.json").write_text(
        json.dumps({"generated_by": "offline_vlm_stub", "week": CURRENT_WEEK, "items": items},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    write_placeholder_images(items)
    print(f"[OK] 素材标签      -> data/external/creative_tags.json  ({len(items)} 条)")
    print(f"[OK] 占位图        -> data/external/creatives/  ({len(items)} 张)")

    (OUT / "seasonal_calendar.json").write_text(
        json.dumps({"events": seasonal_calendar()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] 时令日历      -> data/external/seasonal_calendar.json  ({len(seasonal_calendar())} 个事件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
