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

    python scripts/make_test_external_data.py
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def creative_catalog() -> list[dict]:
    """素材库 + 离线标签 + 历史表现。"""
    items = []
    for index in range(30):
        theme, tags, color = _THEMES[index % len(_THEMES)]
        product, genre = PRODUCTS[index % len(PRODUCTS)]
        region = REGIONS[(index * 3) // 5 % len(REGIONS)]
        platform = PLATFORMS[(index * 2 + index // 5) % len(PLATFORMS)]
        # 让时令素材在对应节日附近表现好——这是"万圣节素材出量"要能被推出来的依据
        ipm = round(4.0 + (6.0 if theme == "halloween" else 0.0) + (index % 5) * 0.6, 2)
        items.append({
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
            "hook_type": _HOOKS[index % len(_HOOKS)],
            "dominant_color": color,
            "has_face": index % 3 == 0,
            "text_overlay_ratio": round(0.15 + (index % 5) * 0.08, 2),
            # ---- 历史表现（用于"这条素材以前跑过且低于安全线"的提醒）----
            "ipm": ipm,
            "ctr": round(0.010 + ipm * 0.0015, 4),
            "d7_cpi": round(3.4 - ipm * 0.12, 2),
            "week_launched": CURRENT_WEEK if index % 3 == 0 else "2026-W29",
            "image_path": f"data/external/creatives/CRV_{9000 + index}.png",
        })
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
