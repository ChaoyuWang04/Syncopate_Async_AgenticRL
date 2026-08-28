#!/usr/bin/env python
"""U 路 P0-1/P0-2 · 两个冻结考场的生成器（`24 §4-P0`）。

    .venv/bin/python scripts/u_make_exams.py        # 落 data/u_route/{talk,context}_exam.jsonl

设计（生成后**冻结**，改内容=新版本号，不许就地编辑）：
- talk_exam    100 条 = 闲聊/能力询问/客套/跨轮承接 各 25 —— 说人话小考场（盲评 0/1/2）
- context_exam 100 段 = L1–L4 各 25 —— 承接考场（机判，规则随题携带）
真对象来自 data/demo/platform_state.json（CMP_1–6），保证 L2/L3 可机判且答案存在。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("data/u_route")
STATE = json.load(open("data/demo/platform_state.json"))
CAMPS = STATE["campaigns"]              # CMP_1..CMP_6，含 name/daily_budget/metrics
rng = random.Random(1428)

# ── talk_exam ────────────────────────────────────────────────────────────────

CHAT = [
    "今天心情不太好，投放的事先不聊，随便聊两句吧",
    "你平时都是怎么工作的？好奇问问",
    "我刚开完一个特别长的会，脑子很累",
    "周末快到了，你觉得做投放的人该怎么休息",
    "最近行业里都在聊 AI 投放，你怎么看",
    "说实话，我有点担心这个季度的指标完不成",
    "我是新来的运营，之前没做过手游买量，紧张",
    "你觉得买量这行未来会被自动化取代吗",
    "刚被老板批了一顿，缓一缓再干活",
    "给我讲个投放圈的冷知识吧",
    "你有名字吗？我该怎么称呼你",
    "如果预算无限，你最想怎么花",
    "我总觉得数据看多了人会麻木，你呢",
    "夸夸我今天的工作吧，我需要一点鼓励",
    "你会不会觉得我问的问题太基础了",
    "咖啡和茶你选哪个？开个玩笑",
    "帮我想一句今天发工作群的早安语",
    "做投放最有成就感的瞬间是什么",
    "我在考虑要不要转岗做数据分析，你怎么看",
    "今天是我入职一周年，感慨一下",
    "你累不累？一直回答问题",
    "用一句话形容一下'买量'这份工作",
    "我马上要向老板汇报了，帮我壮壮胆",
    "聊点别的，你对游戏本身感兴趣吗",
    "假如你是投手，你最怕遇到什么",
]
CAPABILITY = [
    "你都能帮我做什么？",
    "你能直接帮我改预算吗，还是只能建议",
    "你查数据的范围有多大，能看历史多久的",
    "如果我让你做一件你做不了的事会发生什么",
    "你和普通的报表工具有什么区别",
    "你能帮我盯着账户，异常的时候提醒我吗",
    "素材相关的事你能做哪些",
    "你怎么保证不会乱花我的钱",
    "哪些操作需要我人工确认，哪些你可以直接做",
    "你能跨账户操作吗",
    "你的数据是实时的吗，有延迟吗",
    "你能帮我写投放周报吗",
    "如果两个需求冲突了你听谁的",
    "你能记住我之前告诉过你的事吗",
    "风控相关的检查你会自动做吗",
    "你可以帮我做竞品分析吗",
    "预算超了你会拦我吗，怎么拦",
    "你能解释你每一步为什么这么做吗",
    "我可以让你同时处理几件事吗",
    "你出错了怎么办，会有记录吗",
    "新建一个 campaign 你能全程搞定吗",
    "你能预测下周的 ROAS 吗",
    "安全线是什么，你怎么用它",
    "你能撤销一个已经执行的操作吗",
    "第一次用，给我一个快速上手的建议",
]
COURTESY = [
    "谢谢你，刚才帮大忙了",
    "辛苦了，今天先到这里",
    "不好意思，我刚才口气不太好",
    "太棒了，就是这个效果",
    "麻烦你了，一会儿还要再查一次",
    "抱歉打扰，现在方便吗",
    "好的收到，明白了",
    "你刚才的解释很清楚，赞",
    "先这样，我去开个会",
    "下班啦，明天见",
    "新年快乐！哦不对，还没到",
    "刚才是我理解错了，不怪你",
    "这个答案很有帮助，谢谢",
    "嗯嗯，继续",
    "等我五分钟，马上回来",
    "我回来了，继续刚才的",
    "没事了，问题解决了",
    "你反应真快",
    "今天效率很高，多亏你",
    "拜拜",
    "对不起，刚才网断了",
    "早上好，开工",
    "午饭吃了吗？开个玩笑，继续干活",
    "这次汇报很顺利，谢啦",
    "行，就按你说的办",
]
# 跨轮承接（两轮：第一轮铺垫，第二轮省略式追问——判"是否接住语境"，盲评）
FOLLOWUP_TERMS = [
    ("ROI", "ROAS"), ("CPM", "CPC"), ("CTR", "CVR"), ("CPI", "LTV"),
    ("留存率", "付费率"), ("归因窗口", "回收周期"), ("素材疲劳", "频次控制"),
    ("A/B 测试", "增量测试"), ("竞价", "预算平滑"), ("深度转化", "浅层转化"),
    ("自然量", "买量"), ("大盘", "细分市场"), ("冷启动", "学习期"),
]
FOLLOWUP = []
for a, b in FOLLOWUP_TERMS:
    FOLLOWUP.append([f"能解释一下{a}是什么意思吗", f"那{b}呢"])
FOLLOWUP += [
    ["什么是买量里的'跑量'", "跑不动的时候一般怎么办"],
    ["投放里说的'起量'是什么意思", "起不来呢"],
    ["什么叫'素材衰退'", "多久算正常"],
    ["解释一下'学习期'", "学习期内能调预算吗（概念上说说就行）"],
    ["什么是'出价策略'", "手动和自动各适合什么场景"],
    ["'频次'高了会怎样", "多少算高"],
    ["什么是'再营销'", "和拉新比优先级怎么排"],
    ["'安卓和 iOS 买量'差别大吗", "预算该怎么分（大概原则）"],
    ["什么叫'混合变现'", "对买量策略有什么影响"],
    ["'留存曲线'怎么看", "次留和七留哪个更重要"],
    ["什么是'防作弊'", "常见的作弊长什么样"],
    ["'素材标签'有什么用", "打错了有影响吗"],
]
rng.shuffle(FOLLOWUP)
FOLLOWUP = FOLLOWUP[:25]

# ── context_exam ─────────────────────────────────────────────────────────────

DEF_WORDS = "是指|指的是|表示|衡量|用于|含义|意思是|定义|计算|即|指"
L1_TERMS = [("ROI", "ROAS"), ("CPM", "CPC"), ("CTR", "CVR"), ("CPI", "CPA"),
            ("留存", "流失"), ("曝光", "点击"), ("转化", "深度转化"), ("竞价", "底价"),
            ("频次", "触达"), ("素材", "创意"), ("回收", "毛利"), ("拉新", "促活"),
            ("自然量", "归因"), ("学习期", "冷启动"), ("起量", "衰退"),
            ("增量测试", "AB 测试"), ("预算平滑", "预算前置"), ("出价", "扣费"),
            ("大盘", "竞对"), ("深度事件", "浅层事件"), ("防作弊", "无效流量"),
            ("再营销", "召回"), ("付费率", "ARPU"), ("LTV", "回本周期"), ("买量", "投放")]
L1 = []
for i, (a, b) in enumerate(L1_TERMS[:25]):
    L1.append({
        "id": f"L1_{i:02d}", "level": "L1",
        "turns": [f"{a}是什么意思？", rng.choice([f"那{b}呢", f"{b}呢？", f"那{b}又是什么"])],
        "judge": {"type": "definition_followup", "term": b,
                  "def_regex": DEF_WORDS},
        "note": "第二轮必须续接'解释概念'意图：不调数据类工具、回复是定义性文字且提到该词",
    })

METRIC_WORDS = [("消耗", "spend"), ("安装量", "installs"), ("ROAS", "roas"),
                ("CPI", "cpi"), ("点击率", "ctr"), ("频次", "frequency")]
L2, L3 = [], []
cids = list(CAMPS.keys())
for i in range(25):
    cid = cids[i % len(cids)]
    m1, _ = METRIC_WORDS[i % len(METRIC_WORDS)]
    m2, _ = METRIC_WORDS[(i + 1) % len(METRIC_WORDS)]
    ref = rng.choice(["它", "这个 campaign", "那它", "这条计划"])
    L2.append({
        "id": f"L2_{i:02d}", "level": "L2", "campaign": cid,
        "turns": [f"帮我查一下 {cid} 最近的{m1}", f"{ref}的{m2}呢？"],
        "judge": {"type": "same_object_tool", "campaign": cid,
                  "tools": ["campaign.get_metrics", "metrics.get_freshness",
                            "creative.get_metrics_by_asset"]},
        "note": "第二轮必须再次对同一 campaign 调数据类工具（指代解析正确）",
    })
for i in range(25):
    cid = cids[i % len(cids)]
    pct = rng.choice([10, 15, 20])
    old = CAMPS[cid]["daily_budget"]
    L3.append({
        "id": f"L3_{i:02d}", "level": "L3", "campaign": cid,
        "turns": [f"查一下 {cid} 的近况，日预算现在是多少",
                  f"按刚才说的，把日预算上调 {pct}% 提个方案"],
        "judge": {"type": "budget_proposal", "campaign": cid,
                  "old_budget": old, "factor": 1 + pct / 100,
                  "tol": 0.02,
                  "tools": ["campaign.scale_budget", "campaign.update_budget",
                            "approval.create_case"]},
        "note": "第二轮的提案参数须= 上文预算×(1+pct)（±2%）；C 档被审批拦停算过（提案对即可）",
    })

L4_FIELDS = [("素材名称", "creative_name", "新春烟花版视频"),
             ("投放地域", "region", "华东"),
             ("目标产品", "product_id", "GAME_PUZZLE"),
             ("日预算", "daily_budget", "30000"),
             ("平台", "platform", "meta")]
L4 = []
for i in range(25):
    fname, fkey, fval = L4_FIELDS[i % len(L4_FIELDS)]
    ask = rng.choice([
        "帮我新建一条 campaign 跑消消乐",
        "上传一个新素材然后跑审核",
        "给三国志开一条新计划",
    ]) if fkey != "daily_budget" else "帮我新建一条 campaign 跑消消乐"
    L4.append({
        "id": f"L4_{i:02d}", "level": "L4",
        "turns": [ask, f"{fname}用「{fval}」"],
        "judge": {"type": "clarify_then_proceed", "field_value": fval},
        "note": "第一轮预期 clarify（缺字段）；第二轮给了字段后必须继续办（不许再问同一件事），"
                "且后续调用/提案参数里带上该值",
    })

def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}  {len(rows)} 条")

talk = ([{"id": f"chat_{i:02d}", "cat": "chat", "turns": [p]} for i, p in enumerate(CHAT)]
        + [{"id": f"cap_{i:02d}", "cat": "capability", "turns": [p]} for i, p in enumerate(CAPABILITY)]
        + [{"id": f"court_{i:02d}", "cat": "courtesy", "turns": [p]} for i, p in enumerate(COURTESY)]
        + [{"id": f"fu_{i:02d}", "cat": "followup", "turns": t} for i, t in enumerate(FOLLOWUP)])
assert len(talk) == 100, len(talk)
ctx = L1 + L2 + L3 + L4
assert len(ctx) == 100, len(ctx)
dump(OUT / "talk_exam.jsonl", talk)
dump(OUT / "context_exam.jsonl", ctx)
