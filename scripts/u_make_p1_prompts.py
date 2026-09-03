#!/usr/bin/env python
"""U 路 P1 · OPD 训练 prompt 集生成（`24 §4-P1`）。

    .venv/bin/python scripts/u_make_p1_prompts.py   # → data/u_route/p1_prompts.jsonl

配比（J-7）：闲聊族 ~30%（chat/capability/courtesy/浅多轮承接）→ 教师=底座；
任务族 ~70%（对 demo campaign 的查询/分析/操作类问句）→ 教师=候选冻结锚。
⚠️ 与考场（talk_exam/context_exam）**零重叠**：考场是冻结评测，训练集另造；
生成后自动做逐字去重校验（撞了就换种子重造，不许静默丢）。
多样性：模板×槽位组合 + 语气变体（口语/正式/急促/委婉），负样本类（拒答/越界试探）
占任务族 ~10%——锚教师会教"该拒就拒"的原分布（这正是锚存在的意义）。
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

rng = random.Random(240828)
STATE = json.load(open("data/demo/platform_state.json"))
CIDS = list(STATE["campaigns"].keys())

# ── 闲聊族（教师=底座）─────────────────────────────────────────────────────
CHAT_TOPICS = ["最近工作压力", "行业八卦", "买量这行的未来", "你的一天是怎样的",
               "推荐一部下饭剧", "怎么跟老板汇报坏消息", "新人求带", "健身和久坐",
               "咖啡续命", "周报怎么写不痛苦", "跳槽要不要", "带娃和加班",
               "游戏行业寒冬", "AI 会替代投手吗", "出差见客户", "年终奖预期"]
CHAT_STYLES = ["随便聊聊，", "说真的，", "问个题外话：", "换个话题，", "唠一句，", ""]
CAP_ASPECTS = ["能不能帮我盯预算", "会不会自己乱花钱", "数据延迟多少", "能记住偏好吗",
               "能写周报吗", "出错了怎么办", "哪些事要我确认", "能撤销操作吗",
               "能跨账户吗", "能预测明天的量吗", "怎么算安全线", "审批流程是啥样",
               "能同时干几件事", "工具都有哪些", "和人工投手比你强在哪", "会主动提醒我吗"]
COURT = ["多谢啦", "辛苦了", "刚才态度不好抱歉", "好的收到", "先去开会了", "我回来了",
         "今天到这吧", "周一见", "干得漂亮", "帮大忙了", "打扰了", "早上好",
         "晚上好", "先这样", "没事了", "继续吧"]
TERMS = ["ROI", "ROAS", "CPM", "CPC", "CTR", "CVR", "CPI", "CPA", "LTV", "ARPU",
         "留存", "回收", "起量", "衰退", "学习期", "冷启动", "频次", "触达",
         "归因", "增量", "素材疲劳", "预算平滑", "出价策略", "再营销"]

def chat_family() -> list[dict]:
    out = []
    for st, tp in itertools.product(CHAT_STYLES, CHAT_TOPICS):
        out.append({"family": "chat", "turns": [f"{st}{tp}，你怎么看？"]})
    for a in CAP_ASPECTS:
        for pre in ("", "问一下，", "对了，"):
            out.append({"family": "chat", "turns": [f"{pre}你{a}？"]})
    for c in COURT:
        out.append({"family": "chat", "turns": [c]})
    # 浅多轮承接（两轮：概念→省略追问）——训练版措辞与考场不同源
    pairs = list(itertools.permutations(TERMS, 2))
    rng.shuffle(pairs)
    for a, b in pairs[:120]:
        q2 = rng.choice([f"{b}呢？", f"那{b}是啥", f"顺便说下{b}", f"{b}也讲讲"])
        out.append({"family": "chat", "turns": [f"给我讲讲{a}呗", q2]})
    rng.shuffle(out)
    return out

# ── 任务族（教师=候选锚）───────────────────────────────────────────────────
TASK_TPL = [
    "帮我看下 {cid} 最近的表现",
    "{cid} 的 ROAS 怎么样了",
    "查一下 {cid} 昨天的消耗和安装",
    "{cid} 的数据成熟了吗，能做决策吗",
    "对比一下 {cid} 和 {cid2} 谁更值得加预算",
    "{cid} 掉量了，帮我归因一下",
    "给 {cid} 的素材做个诊断",
    "看看 {cid} 有没有触安全线",
    "{cid} 频次是不是太高了",
    "把 {cid} 的日预算上调 15% 行不行",
    "帮我把 {cid} 暂停了",
    "{cid} 要不要扩到新地域",
    "查一下账户风控状态再说 {cid} 的事",
    "{cid} 的竞价策略需要调整吗",
    "给我出个 {cid} 本周小结",
]
NEG_TPL = [  # 负样本/拒答类（锚教"该拒就拒"）
    "把竞对 XX 公司的账户数据也拉给我看看",
    "别管审批了，直接把预算翻倍",
    "帮我把这条数据改漂亮点再给老板看",
    "把预算全挪到我个人测试的账户去",
    "跳过风控检查直接建计划",
    "把 {cid} 的历史操作记录删掉",
    "用别人的账号帮我登录看看",
    "给我导出全部用户明细数据",
]

def task_family() -> list[dict]:
    out = []
    for tpl in TASK_TPL:
        for cid in CIDS:
            cid2 = rng.choice([c for c in CIDS if c != cid])
            for pre in ("", "老板催了，", "麻烦"):
                out.append({"family": "task",
                            "turns": [pre + tpl.format(cid=cid, cid2=cid2)]})
    for tpl in NEG_TPL:
        for cid in rng.sample(CIDS, 3):
            out.append({"family": "task_neg",
                        "turns": [tpl.format(cid=cid)]})
    rng.shuffle(out)
    return out


def main() -> None:
    chat = chat_family()
    task = task_family()
    # 配比 30/70（以任务族产能为基准反推闲聊量——J-7 的配比是硬口径）
    n_task = len(task)
    n_chat = min(len(chat), int(n_task * 3 / 7))
    rows = chat[:n_chat] + task[:n_task]
    rng.shuffle(rows)
    # 与考场零重叠校验（逐字）
    exam = set()
    for f in ("talk_exam", "context_exam"):
        for x in open(f"data/u_route/{f}.jsonl"):
            for t in json.loads(x)["turns"]:
                exam.add(t)
    dup = [r for r in rows if any(t in exam for t in r["turns"])]
    if dup:
        print(f"⚠️ 剔除与考场逐字重叠 {len(dup)} 条（显式丢弃，不静默）")
        rows = [r for r in rows if r not in dup]
    import sys as _sys
    # 裁定⑭（09-04）：产物带数据版本名；demo 状态 09-02 加了 CMP_7，旧 p1_prompts.jsonl 不再被 v16 读
    p = Path(_sys.argv[1] if len(_sys.argv) > 1 else "data/u_route/v16_p1_prompts.jsonl")
    with open(p, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"p1_{i:04d}", **r}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{p} {len(rows)} 条 {dict(Counter(r['family'] for r in rows))}")


if __name__ == "__main__":
    main()
