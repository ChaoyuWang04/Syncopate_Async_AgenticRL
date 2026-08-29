#!/usr/bin/env python
"""U 路 P2 · v14 SFT 数据集构建（`24 §4-P2`）。

    .venv/bin/python scripts/u_build_v14.py     # → data/sft/v14/{train,val}.parquet

构成（seed 固定，可复现）：
  ① v13 train 419 行原样继承（影子链已验的资产不重造）
  ② 多轮 L2 行（P1 红旗正枪）：查询类 case 的第二轮指代追问——历史前缀+追问进
     user_message，gold=[get_metrics(同 campaign), 终答(summary+reply)]，
     **走 build_sft_sample 真回放**（工具真执行=回放断言天然满足）
  ③ 闲聊行（离线蒸馏）：chat_gold.jsonl（教师文本+契约装壳），直接 tokenize
  ④ CoT 行：cot_traces.jsonl（8B think 末答对才留），单步案 ctx+<think>…</think>+gold
     re-tokenize，mask=prompt 0 / 生成段 1
  val = v13 val 84 行原样（SFT 选点用）。
范围注记：L3/L4 多轮行刻意不造——该行为归 P3 RL（gold 派生要碰写权限语义，
  教错比不教危险）；C-6（RL val 并入）属 RL 数据，在 P3 前的 rl v14 生成时做。
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
rng = random.Random(1409)

METRIC_ASKS = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("ROAS", "roas_d7"),
               ("CPI", "cpi"), ("点击率", "ctr"), ("频次", "frequency")]
REF_FORMS = ["它的{m}呢？", "这条计划的{m}呢", "那{m}怎么样", "顺便看下它的{m}"]


async def build_l2_rows(tokenizer, registry, max_rows: int = 200) -> list[dict]:
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    bundles = load_bundles(Path("data/batches/v13"))
    split_ids = json.load(open("data/splits/v13/splits.json")) if \
        Path("data/splits/v13/splits.json").exists() else None
    rows, skipped = [], 0
    cands = []
    for cid, b in bundles.items():
        if b.gold and b.gold.actions and \
                b.gold.actions[0]["tool"] == "campaign.get_metrics" and \
                b.case.context.get("campaign_id"):
            cands.append(b)
    rng.shuffle(cands)
    print(f"L2 候选 case {len(cands)}")
    for b in cands:
        if len(rows) >= max_rows:
            break
        camp = b.case.context["campaign_id"]
        mname, mkey = METRIC_ASKS[len(rows) % len(METRIC_ASKS)]
        ask = rng.choice(REF_FORMS).format(m=mname)
        b2 = copy.deepcopy(b)
        prev_sum = (b.gold.final_answer or {}).get("summary") or \
                   (b.gold.final_answer or {}).get("conclusion") or "已给出结论"
        b2.case.user_message = (
            f"[上一轮] 用户：{b.case.user_message}\n"
            f"[上一轮] 助手：{str(prev_sum)[:120]}\n\n{ask}")
        b2.case.case_id = f"{b.case_id}_MT2"
        b2.gold.actions = [{"tool": "campaign.get_metrics",
                            "arguments": {"campaign_id": camp}}]
        b2.gold.final_answer = {
            "summary": f"{camp} {mkey} 已查",
            "reply": f"{camp} 的{mname}指标已经取到，详见上方数据；如需对比或进一步分析随时说。"}
        try:
            cfg = None
            row = await build_sft_row(b2, tokenizer=tokenizer, registry=registry,
                                      index=90000 + len(rows), split="train",
                                      config=cfg)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠️ {b.case_id} 回放失败：{str(e)[:120]}")
            continue
        row["bucket"] = "multiturn"
        rows.append(row)
    print(f"L2 行 {len(rows)}（回放失败丢 {skipped}——显式计数）")
    return rows


GLOSSARY = {
    "ROI": "投资回报率，衡量投入产出比的指标，计算方式是收益除以成本",
    "CPI": "单次安装成本，指平均每带来一个应用安装所花的钱",
    "CPM": "千次曝光成本，指广告每展示一千次所花的费用",
    "CPC": "单次点击成本，指用户每点击一次广告所花的费用",
    "CVR": "转化率，指从点击到完成目标行为（如安装、付费）的比例",
    "转化": "指用户完成了我们期望的目标行为，比如安装、注册或付费",
    "留存": "指用户在安装后的第 N 天仍然活跃的比例，是衡量质量的核心指标",
    "LTV": "用户生命周期价值，指一个用户在整个生命周期内预计带来的总收入",
    "ARPU": "每用户平均收入，用总收入除以活跃用户数得到",
    "出价": "指在竞价系统里为一次展示或转化愿意支付的价格",
    "定向": "指投放时圈定目标人群的条件，比如地区、年龄、兴趣",
    "素材": "指广告展示给用户的创意内容，包括图片、视频和文案",
    "归因": "指判断一次转化应该算在哪个渠道或广告头上的规则",
    "自然量": "指不靠付费广告、用户自发下载带来的量",
    "冷启动": "指新计划刚投放时系统还没学到足够数据的探索阶段",
    "学习期": "指投放系统为新计划积累转化数据、模型逐步稳定的阶段",
    "放量": "指在效果达标后逐步提高预算扩大投放规模的操作",
    "付费率": "指活跃用户中产生付费行为的比例",
    "次留": "即次日留存率，指安装次日仍活跃的用户比例",
    "买量": "指通过付费广告渠道获取新用户的投放行为",
    # ---- v14.2 扩表（08-29）：r2 考场证明模型背词条不泛化规则（词表内 5/5 过、
    #      词表外 20 挂 10）⇒ 把领域标准词汇补全。作者预裁定维持：术语可与考场重叠
    #      （能力是「概念⇒答」不是背题），原句仍逐字去重。
    "ROAS": "广告支出回报率，用广告带来的收入除以广告花费，衡量每块钱广告费换回多少收入",
    "CPA": "单次转化成本，指平均每带来一个目标转化行为所花的钱",
    "CTR": "点击率，用点击次数除以曝光次数得到，衡量素材吸引力",
    "曝光": "指广告被展示给用户看到一次，是计费和转化漏斗的最上层",
    "点击": "指用户看到广告后主动点了它，是从曝光走向转化的中间行为",
    "频次": "指同一个用户在一段时间内看到同一条广告的平均次数",
    "触达": "指广告实际覆盖到的去重用户数，触达乘频次约等于总曝光",
    "竞价": "指多条广告为同一次展示机会实时出价、价高者得的分配机制",
    "底价": "指平台为一次展示设定的最低成交价，出价低于它就没有参与资格",
    "扣费": "指广告实际被收钱的动作，按计费方式（如点击或千次曝光）结算",
    "创意": "指一条具体的广告内容组合，即素材加文案加落地页的整体",
    "流失": "指老用户不再活跃或卸载离开产品，与留存相对",
    "拉新": "指以获取新用户为目标的投放或运营动作",
    "促活": "指让已安装但不活跃的用户重新活跃起来的运营或投放动作",
    "召回": "指把已经流失的老用户重新拉回产品的投放或运营动作",
    "再营销": "指对接触过产品的人群（如点过广告、装过应用）再次定向投放",
    "投放": "指把广告实际推到渠道上跑量的整个执行过程，含预算、出价、定向与素材",
    "大盘": "指整个市场或渠道的总体量与价格水平，用来对照自家表现",
    "竞对": "指和我们抢同一批用户的竞争对手产品或它们的投放动作",
    "毛利": "指收入减去直接成本后剩下的利润，不含摊到头上的间接费用",
    "回收": "指投放花出去的钱通过用户付费逐步赚回来的过程",
    "回本周期": "指累计回收追平投放成本所需要的天数，越短现金流越健康",
    "衰退": "指计划或素材效果持续走低、量价齐跌的阶段",
    "起量": "指计划通过学习期后消耗与转化明显放大的阶段",
    "AB测试": "指把流量分成对照组和实验组各跑一个版本，用数据对比谁更好",
    "增量测试": "指留出不投放的对照人群，测广告带来的真实增量效果",
    "预算平滑": "指系统把每日预算均匀分配到全天，避免早上就花光",
    "预算前置": "指把预算刻意往时段或周期的前面倾斜，抢先跑量",
    "深度转化": "指比安装更靠后的高价值行为，如付费、留存或通关关键关卡",
    "深度事件": "指转化漏斗后段的高价值行为事件，常用作出价优化目标",
    "浅层事件": "指漏斗前段量大但价值低的行为，如展示、点击、打开",
    "无效流量": "指作弊、重复或机器产生的虚假点击与安装，会被平台过滤",
    "防作弊": "指识别并过滤虚假流量的机制，保护预算不被刷量吃掉",
    "智能出价": "指把出价交给平台模型按转化概率自动调整的出价方式",
    "目标成本": "指投放时设定的期望单次转化成本，系统围绕它自动调价",
    "频控": "指限制单个用户看到同一广告次数的规则，防骚扰也防浪费",
    "受众包": "指按条件圈好的一组目标用户集合，可直接用于定向",
    "相似人群": "指以种子用户为模板扩展出的高相似度潜在用户群",
    "素材疲劳": "指同一素材跑久了用户看腻、点击率与转化持续下滑的现象",
    "一方数据": "指我们自己产品直接收集的用户数据，区别于平台或第三方数据",
    "排期": "指广告按约定时间段和位置投放的计划安排",
}
L1_FORMS = ["那{b}呢？", "{b}又是什么", "那{b}是什么意思", "顺便讲下{b}", "{b}呢",
            "什么是{b}？", "{b}是啥意思", "再说说{b}"]


async def build_l1_rows(tokenizer, registry, max_rows: int = 100) -> list[dict]:
    """概念追问行（L1 正枪，v14.1 增补）：第二轮省略式追问概念 ⇒ gold=零动作纯文字
    定义（与 L2 的「数据追问⇒调工具」构成判别对照，教会二者的分界）。"""
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    rng1 = random.Random(1410)
    bundles = load_bundles(Path("data/batches/v13"))
    scaffolds = [b for b in bundles.values() if b.gold and not b.gold.actions]
    # 考场逐字去重（术语可重叠——能力是「概念⇒答」不是背题；原句不得出现）
    exam_turns = set()
    for f in ("talk_exam.jsonl", "context_exam.jsonl"):
        for x in open(f"data/u_route/{f}"):
            exam_turns.update(json.loads(x)["turns"])
    terms = list(GLOSSARY)
    pairs = [(a, b) for a in terms for b in terms if a != b]
    rng1.shuffle(pairs)
    rows, skipped = [], 0
    for a, b in pairs:
        if len(rows) >= max_rows:
            break
        ask = rng1.choice(L1_FORMS).format(b=b)
        if ask in exam_turns:
            continue
        sc = copy.deepcopy(rng1.choice(scaffolds))
        sc.case.user_message = (
            f"[上一轮] 用户：{a}是什么意思？\n"
            f"[上一轮] 助手：{a} 指{GLOSSARY[a][:60]}。\n\n{ask}")
        sc.case.case_id = f"L1C_{len(rows)+skipped:04d}"
        sc.gold.actions = []
        sc.gold.final_answer = {"reply": f"{b} 指{GLOSSARY[b]}。"}
        try:
            row = await build_sft_row(sc, tokenizer=tokenizer, registry=registry,
                                      index=93000 + len(rows), split="train",
                                      config=None)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠️ {sc.case.case_id} 回放失败：{str(e)[:120]}")
            continue
        row["bucket"] = "multiturn_l1"
        rows.append(row)
    print(f"L1 概念行 {len(rows)}（回放失败丢 {skipped}）")
    return rows


def build_chat_rows(tokenizer, start_idx: int) -> list[dict]:
    sys.path.insert(0, "scripts")
    from probe_opd_divergence import render_prompt_text
    rows = []
    for i, x in enumerate(open("data/u_route/chat_gold.jsonl")):
        d = json.loads(x)
        prompt = render_prompt_text(tokenizer, d["prompt"], tools=None)
        ids_p = tokenizer(prompt, add_special_tokens=False).input_ids
        ids_g = tokenizer(d["gold_json"] + "<|im_end|>",
                          add_special_tokens=False).input_ids
        rows.append({
            "case_id": f"CHATG_{i:04d}", "input_ids": ids_p + ids_g,
            "loss_mask": [0] * len(ids_p) + [1] * len(ids_g),
            "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
            "supervised_tokens": len(ids_g), "split": "train",
            "index": start_idx + i, "signal_class": "graded",
            "behavior": "answer", "bucket": "chat_distill",
        })
    print(f"闲聊行 {len(rows)}")
    return rows


def build_cot_rows(tokenizer, start_idx: int) -> list[dict]:
    ASST = "<|im_start|>assistant"
    df = pd.concat([pd.read_parquet("data/sft/v13/train.parquet"),
                    pd.read_parquet("data/sft/v13/val.parquet")])
    by_id = {r.case_id: r for _, r in df.iterrows()}
    rows = []
    for i, x in enumerate(open("data/u_route/cot_traces.jsonl")):
        d = json.loads(x)
        src = by_id.get(d["case_id"])
        if src is None:
            continue
        ids = list(src.input_ids)
        full = tokenizer.decode(ids[: src.total_length])
        cut = full.rfind(ASST)
        head_end = full.find("\n", cut) + 1
        ctx = full[:head_end]
        new_tail = f"<think>\n{d['think']}\n</think>\n\n{d['gold_tail']}"
        ids_p = tokenizer(ctx, add_special_tokens=False).input_ids
        ids_g = tokenizer(new_tail, add_special_tokens=False).input_ids
        rows.append({
            "case_id": f"{d['case_id']}_COT{d['sample_idx']}",
            "input_ids": ids_p + ids_g,
            "loss_mask": [0] * len(ids_p) + [1] * len(ids_g),
            "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
            "supervised_tokens": len(ids_g), "split": "train",
            "index": start_idx + i, "signal_class": "graded",
            "behavior": d["behavior"], "bucket": "cot_distill",
        })
    print(f"CoT 行 {len(rows)}")
    return rows


async def main() -> int:
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    registry = build_domain().registry
    registry.latency_scale = 0.0

    t13 = pd.read_parquet("data/sft/v13/train.parquet")
    v13 = pd.read_parquet("data/sft/v13/val.parquet")
    l2 = await build_l2_rows(tokenizer, registry)
    chat = build_chat_rows(tokenizer, 91000)
    cot = build_cot_rows(tokenizer, 92000)
    l1 = await build_l1_rows(tokenizer, registry, max_rows=250)  # v14.2：词表 61·规则要靠量教
    new = pd.DataFrame(l2 + chat + cot + l1)
    train = pd.concat([t13, new], ignore_index=True)
    out = Path("data/sft/v14"); out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    v13.to_parquet(out / "val.parquet")
    manifest = {
        "version": "v14.2", "seed": 1409,
        "sources": {"v13_train": len(t13), "l2_multiturn": len(l2),
                    "chat_distill": len(chat), "cot_distill": len(cot),
                    "l1_concept": len(l1)},
        "total": len(train),
        "supervised_tokens": int(train.supervised_tokens.sum()),
        "behavior_counts": train.behavior.value_counts().to_dict(),
        "bucket_counts": train.bucket.value_counts().to_dict(),
    }
    json.dump(manifest, open(out / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    # 健全性断言：所有新行监督 token > 0、mask 与长度一致
    for r in l2 + chat + cot + l1:
        assert r["supervised_tokens"] > 0, r["case_id"]
        assert len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"], r["case_id"]
    print("✅ v14 构建完成，健全性断言全过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
