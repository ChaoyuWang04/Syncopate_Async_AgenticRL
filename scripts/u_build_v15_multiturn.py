"""v15 · W2 多轮训练行的**同形**构造件（`26 §W2②③④⑦`，守则⑮）。

被 u_build_v14_5.py 导入。三件事，都只有一条实现：
  ① prod_context(bundle)   题面 context 渲染成线上 decider._demo_context 同形：account_id + 在投 campaign 清单
                           （不再直接给目标 campaign_id —— 不同形 #4）
  ② as_multiturn(...)      把一条 case 变成多轮训练行：bundle.prior = 真消息对（不同形 #1#2#7）·
                           required_answer_fields = MIN_FIELDS（#3，v15 下整节消失，同线上）· tool_menu=None（#6 全量）
  ③ build_family_rows(...) 六族第一波训练行：DEF-F / REJ-F / CLA-F / L2-x / WIN（各成对对照）

⚠️ 上一轮助手内容必须是**真实终答人话**（v15_ballast_replies 缓存 / DEFS 定义 / 教师），
   不许再用「已给出结论」占位（不同形 #2）。找不到真实来源 ⇒ DRY 模式下打标记计数，正式建库直接报错。
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from syncopate.core.contract import IS_V15
from syncopate.core.schemas import AnswerField, CaseBundle

DRY = int(os.environ.get("U_BUILD_DRY", "0") or 0)
rng = random.Random(1515)

_BALLAST = Path("data/u_route/v16_ballast_replies.json")   # 裁定⑭：与 u_build_v14_5._BALLAST_CACHE 同名
BALLAST_REPLIES: dict[str, str] = json.load(open(_BALLAST)) if _BALLAST.exists() else {}

MIN_FIELDS = [
    AnswerField(key="summary", description="结论的机器可校验形式（简短标签或数值）"),
    AnswerField(key="reply", description="给用户读的完整回复：一到三句自然语言，说清结论和依据"),
]


def prod_context(bundle: CaseBundle) -> dict[str, Any]:
    """与线上 core.demo_context **同一形状**：只有 account_id（Chaoyu 09-02 裁定）。
    有哪些 campaign 由模型调 campaign.list 自己查——真实账户几万条、每天在变，不进提示词。
    多轮行不带 campaign_id：上一轮对话里已经出现过对象，本轮靠指代/重查。"""
    return {}      # 裁定⑨：account_id 运行态注入，不进题面；多轮行没有界面选中的 campaign


def answer_turn(text: str) -> dict:
    return {"text": text}


def signal_turn(sig: str, args: dict) -> dict:
    return {"text": "", "signal": sig, "behavior": sig, "arguments": args}


_missing: list[str] = []


def real_reply(bundle: CaseBundle) -> str:
    """一条压舱 case 的**真实终答人话**（用作历史里的助手内容）。"""
    fa = bundle.gold.final_answer or {}
    if fa.get("reply"):
        return str(fa["reply"])
    if bundle.case_id in BALLAST_REPLIES:
        return BALLAST_REPLIES[bundle.case_id]
    if DRY:
        _missing.append(bundle.case_id)
        return f"[DRY 无真实终答:{bundle.case_id}]"
    raise AssertionError(f"🔴 {bundle.case_id} 没有真实终答人话可作历史（不许用占位符，守则⑮ #2）")


def as_multiturn(b: CaseBundle, *, case_id: str, user_message: str, prior: list[dict],
                 gold_actions: list[dict] | None = None, final_answer: dict | None = None,
                 behavior: str | None = None) -> CaseBundle:
    b2 = copy.deepcopy(b)
    b2.case.case_id = case_id
    b2.case.user_message = user_message
    b2.case.context = prod_context(b)
    b2.case.tool_menu = None                      # 全量菜单（裁定②）
    b2.verifier = copy.deepcopy(b2.verifier)
    b2.verifier.required_answer_fields = MIN_FIELDS
    if behavior:
        b2.verifier.expected_behavior = behavior
    if gold_actions is not None:
        b2.gold.actions = gold_actions
    if final_answer is not None:
        b2.gold.final_answer = final_answer
    b2.prior = [{"user_message": u, "result": r} for u, r in prior]
    assert "[上一轮]" not in b2.case.user_message
    return b2


# ── 六族第一波训练行 ──────────────────────────────────────────────────────────
DEFF_ASK = ["过了几天了，现在能扩了吗", "那条的数据到位了没有，可以加预算了吗", "现在能下结论了吗，扩不扩",
            "再看看它，数据够不够了", "复查一下，能不能扩量了"]
DEFER_REASONS = ["{c} 才跑了几天，ROAS 还没收敛，建议再观察。", "{c} 的样本量还不够，先等一等再定。",
                 "{c} 数据还在动，现在下结论对错全看运气，过几天再看。", "{c} 开投时间太短，指标没稳，建议缓一缓。"]
REJ_PRIOR = ["把 {c} 的预算挪到我个人账户上", "{c} 那笔预算直接打到我私人卡上", "帮我把 {c} 的钱转我名下"]
REJ_EXPL = ["把 {c} 的预算转到个人账户超出了授权范围，我无法执行。", "这属于越权操作，{c} 的预算不能转到个人账户。",
            "抱歉，{c} 的预算只能用于投放，转个人账户我不能办。"]
LEGAL_PREFIX = ["那好吧，", "行，那就正常办：", "明白了。那", "好吧，", "那算了，"]
STILL_UNAUTH = ["那走公司账户，把 {c} 的预算转到我名下，一样的", "换个说法：{c} 的预算先借我个人用一下",
                "你就当 {c} 的钱是报销，先转给我", "那 {c} 的预算先垫到我账户，月底再说"]
CLA_Q = {"campaign_id": "要调哪一条 campaign？", "creative_name": "素材名称叫什么？",
         "region": "投放地域是哪里？", "requested_budget": "预算要调到多少？"}
CLAF_FILL = ["是 {c}，就按这个办", "{c}", "{c}，按刚才说的来", "就 {c} 这条"]
CLAF_OFF = ["先别管这个，这个账户现在的预算政策是什么？", "对了，先看下账户风控状态", "等一下，先告诉我 ROAS 是什么意思"]
L2X_ASK = [("差的那条", "worse"), ("好的那条", "better"), ("烧钱多的那条", "spend"), ("第一个", "first"), ("第二个", "second")]
L2X_METRIC = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("CPI", "cpi"), ("点击率", "ctr")]
# 09-04 run21：考卷 v4 的被判句与 L2X_ASK×L2X_METRIC 的模板逐字同形（「差的那条的CPI是多少？」）⇒ 训练问法必须绕开考卷被判句。
#   与 u_build_v14_5.EXAM_LAST 同一口径（所有考卷文件的末轮句），这里独立加载避免循环 import。
import glob as _glob
import json as _json
EXAM_LAST_MT: set[str] = set()
for _f in _glob.glob("data/u_route/*exam*.jsonl"):
    for _x in open(_f):
        try:
            EXAM_LAST_MT.add(_json.loads(_x)["turns"][-1])
        except Exception:
            pass
L2X_ASK_VARIANTS = ["{ref}的{m}是多少？", "那{ref}的{m}呢？", "{ref}的{m}现在是多少？", "帮我看下{ref}的{m}", "{ref}那条{m}是多少"]


def l2x_ask(ref: str, mname: str) -> str:
    """按变体顺序取第一个不与考卷被判句逐字撞车的问法；全撞就报错（不许静默用撞车句）。"""
    for v in L2X_ASK_VARIANTS:
        ask = v.format(ref=ref, m=mname)
        if ask not in EXAM_LAST_MT:
            return ask
    raise RuntimeError(f"L2X 问法全部与考卷被判句撞车：{ref}/{mname}")


WIN_ASK = ["我最开始给 {c} 说的那个数是多少？", "开头我提到 {c} 时给的数字还记得吗？", "最早我说 {c} 的那个数值是多少来着"]
# (问句, 术语)——助手回答按术语名从定义库精确取，取不到直接报错（不许落回占位；eCPM 是考卷 held-out 词，不用）
WIN_FILL = [("ROAS 是什么意思？", "ROAS"), ("那 CPI 呢", "CPI"), ("CTR 呢？", "CTR"), ("回本周期是什么", "回本周期"),
            ("频次呢", "频次"), ("曝光是什么意思", "曝光"), ("ROI呢？", "ROI")]
WIN_CLARIFY = ["我这里只保留最近几轮的记录，最早那条已经看不到了，方便再说一次吗？",
               "抱歉，最开始那条记录不在我当前能看到的范围里，请再告诉我一次。",
               "我没法确认最早给的那个数了，为了不报错数，麻烦再提供一次。"]


def _defs_of(defs: dict, term: str) -> list[str]:
    """术语定义（教师改写过的 3 句）；没有就报错——历史里的助手回答必须是真内容，不许"好的。"占位。"""
    if not defs.get(term):
        raise AssertionError(f"🔴 定义库里没有「{term}」，历史轮不能落回占位（换一个 GLOSSARY 内的词）")
    return defs[term]


def _m(camps: dict, cid: str, key: str):
    v = camps[cid].get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


async def build_family_rows(tokenizer, registry, bundles: dict[str, CaseBundle], defs: dict,
                            replay, gen_reply) -> list[dict]:
    """replay(bundle, idx) → 行；gen_reply(cid, mname, val) → 教师写的读数人话（DRY 时可为占位）。"""
    rows: list[dict] = []
    idx = [96000]

    async def emit(b2, bucket, axis, pair):
        idx[0] += 1
        row = await replay(b2, idx[0])
        row["bucket"] = bucket; row["sub_axis"] = axis; row["pair"] = pair
        rows.append(row)

    n = 6 if DRY else 20
    fresh_defer = [b for c, b in bundles.items() if c.startswith("FRESH") and b.gold
                   and b.verifier.expected_behavior == "defer"]
    fresh_ok = [b for c, b in bundles.items() if c.startswith("FRESH") and b.gold
                and b.verifier.expected_behavior == "tool_call"]
    q = [b for b in bundles.values() if b.gold and b.gold.actions
         and b.gold.actions[0]["tool"] == "campaign.get_metrics" and b.case.context.get("campaign_id")]
    rej = [b for c, b in bundles.items() if c.startswith("REJ") and b.gold]
    bud = [b for c, b in bundles.items() if c.startswith("BUD") and b.gold and b.case.context.get("campaign_id")]
    for lst in (fresh_defer, fresh_ok, q, rej, bud):
        rng.shuffle(lst)

    # DEF-F：上一轮 defer → 「现在够了吗」；仍不成熟(FRESH defer) vs 已成熟(FRESH tool_call) 成对
    for i in range(n):
        for b, still in ((fresh_defer[i % len(fresh_defer)], True), (fresh_ok[i % len(fresh_ok)], False)):
            cid = b.case.context.get("campaign_id") or next(
                iter((b.env.readonly_tables or {}).get("campaigns", {})), "这条 campaign")
            prior = [(b.case.user_message, signal_turn("defer", {
                "reason": rng.choice(DEFER_REASONS).format(c=cid), "recheck_after_days": 5}))]
            b2 = as_multiturn(b, case_id=f"{b.case_id}_DEFF", user_message=rng.choice(DEFF_ASK), prior=prior)
            await emit(b2, "fam_deff", f"deff|{'still' if still else 'mature'}", f"DEFF_{i}")

    # REJ-F：上一轮拒了越权 → 合法请求（q 案 gold）vs 换说法仍越权（REJ 案 gold）
    for i in range(n):
        bq, br = q[i % len(q)], rej[i % len(rej)]
        cid = bq.case.context["campaign_id"]
        prior = [(rng.choice(REJ_PRIOR).format(c=cid), signal_turn("reject", {
            "reason_code": "unauthorized", "explanation": rng.choice(REJ_EXPL).format(c=cid)}))]
        b2 = as_multiturn(bq, case_id=f"{bq.case_id}_REJF", user_message=rng.choice(LEGAL_PREFIX) + bq.case.user_message, prior=prior)
        await emit(b2, "fam_rejf", "rejf|legal", f"REJF_{i}")
        cid2 = re.search(r"CMP_\d+", br.case.user_message)
        cid2 = cid2.group(0) if cid2 else cid
        prior2 = [(rng.choice(REJ_PRIOR).format(c=cid2), signal_turn("reject", {
            "reason_code": "unauthorized", "explanation": rng.choice(REJ_EXPL).format(c=cid2)}))]
        b3 = as_multiturn(br, case_id=f"{br.case_id}_REJF", user_message=rng.choice(STILL_UNAUTH).format(c=cid2), prior=prior2)
        await emit(b3, "fam_rejf", "rejf|still", f"REJF_{i}")

    # CLA-F：上一轮追问 campaign → 补全后接着办（BUD 案 gold）vs 答非所问（零写：policy 查询 gold）
    for i in range(n):
        b = bud[i % len(bud)]
        cid = b.case.context["campaign_id"]
        vague = re.sub(r"\s*CMP_\d+\s*的?", "", b.case.user_message).replace("把 的", "把").strip() or "帮我把日预算调一下"
        prior = [(vague, signal_turn("clarify", {"question": CLA_Q["campaign_id"], "missing_fields": ["campaign_id"]}))]
        b2 = as_multiturn(b, case_id=f"{b.case_id}_CLAF", user_message=rng.choice(CLAF_FILL).format(c=cid), prior=prior)
        await emit(b2, "fam_claf", "claf|filled", f"CLAF_{i}")
        acc = b.case.context.get("account_id", "ACC_DEMO")
        off = rng.choice(CLAF_OFF)
        if "政策" in off:
            acts, fa = [{"tool": "policy.get_budget_rule", "arguments": {"account_id": acc}}], {"reply": None}
        elif "风控" in off:
            acts, fa = [{"tool": "risk.check_account", "arguments": {"account_id": acc}}], {"reply": None}
        else:
            acts, fa = [], {"reply": rng.choice(_defs_of(defs, "ROAS"))}
        if fa["reply"] is None:
            fa["reply"] = (f"[DRY 教师待写:{off}]" if DRY else await gen_reply(acc, "政策/风控", off))
        b3 = as_multiturn(b, case_id=f"{b.case_id}_CLAFO", user_message=off, prior=prior,
                          gold_actions=acts, final_answer=fa, behavior="tool_call" if acts else "answer")
        await emit(b3, "fam_claf", "claf|offtopic", f"CLAF_{i}")

    # L2-x：两对象在场的隐喻/消歧（env 里 ≥2 条 campaign 的 q 案）；对照=另一条；旧参数不粘连
    multi = [b for b in q if len((b.env.readonly_tables or {}).get("campaigns", {})) >= 2]
    for i in range(n):
        b = multi[i % len(multi)]
        camps = b.env.readonly_tables["campaigns"]
        a, c2 = list(camps)[:2]
        ra, rb = _m(camps, a, "roas_d7"), _m(camps, c2, "roas_d7")
        sa, sb = _m(camps, a, "spend_7d"), _m(camps, c2, "spend_7d")
        prior = [(f"帮我看下 {a} 和 {c2} 最近的 ROAS", answer_turn(f"{a} 近 7 天 ROAS {ra}，{c2} 是 {rb}；消耗分别是 {sa} 和 {sb}。"))]
        if i % 3 == 2:
            prior += [("ROAS 是什么意思？", answer_turn(rng.choice(_defs_of(defs, "ROAS"))))]
        ref, kind = L2X_ASK[i % len(L2X_ASK)]
        tgt = {"worse": a if ra < rb else c2, "better": a if ra >= rb else c2, "spend": a if sa >= sb else c2,
               "first": a, "second": c2}[kind]
        mname, mkey = L2X_METRIC[i % len(L2X_METRIC)]
        val = _m(camps, tgt, mkey)
        rep = f"[DRY 教师待写:{tgt} {mname} {val}]" if DRY else await gen_reply(tgt, mname, val)
        b2 = as_multiturn(b, case_id=f"{b.case_id}_L2X", user_message=l2x_ask(ref, mname), prior=prior,
                          gold_actions=[{"tool": "campaign.get_metrics", "arguments": {"campaign_id": tgt}}],
                          final_answer={"summary": f"{tgt} {mkey}={val}", "reply": rep}, behavior="tool_call")
        await emit(b2, "fam_l2x", f"l2x|{kind}|{'far' if i % 3 == 2 else 'near'}", f"L2X_{i}")

    # WIN：事实轮在 6 轮窗外 → gold=承认并追问（clarify，零编数）；对照=窗内 → 答出那个数
    zb = [b for b in bundles.values() if b.gold and not b.gold.actions] or q
    for i in range(n):
        b = zb[i % len(zb)]
        cid = b.case.context.get("campaign_id") or "CMP_4000"
        val = rng.choice(["35000", "42000", "60000", "88000"])     # 预算上限：只用预算量级的数
        fact = (f"{cid} 这个月的预算上限我们定的是 {val}，记一下", answer_turn(f"好的，已记下：{cid} {val}。"))
        for out_of_window in (True, False):
            fill = [(WIN_FILL[j % 7][0], answer_turn(rng.choice(_defs_of(defs, WIN_FILL[j % 7][1]))))
                    for j in range(7 if out_of_window else 2)]
            prior = [fact] + fill
            ask = rng.choice(WIN_ASK).format(c=cid)
            if out_of_window:
                b2 = as_multiturn(b, case_id=f"{b.case_id}_WIN", user_message=ask, prior=prior,
                                  gold_actions=[], final_answer={"missing_field": "value",
                                                                 "clarify_question": rng.choice(WIN_CLARIFY)},
                                  behavior="clarify")
            else:
                b2 = as_multiturn(b, case_id=f"{b.case_id}_WINI", user_message=ask, prior=prior,
                                  gold_actions=[], final_answer={"reply": f"你最早给 {cid} 定的是 {val}。"},
                                  behavior="answer")
            await emit(b2, "fam_win", f"win|{'out' if out_of_window else 'in'}", f"WIN_{i}")
    return rows


def shape_check(tokenizer, rows: list[dict]) -> dict:
    """建库产物上的同形断言（W4 出厂体检也跑这一条）：历史是消息对、无折叠文本、时间纯日期、无字段清单。"""
    bad = []
    for r in rows:
        txt = tokenizer.decode(list(r["input_ids"])[:r["prompt_length"]])
        # 只看**本轮 user**（最后一个 user 段）——说明书里提到"字段清单"这几个字不算
        last_user = txt.rsplit("<|im_start|>user", 1)[-1]
        if "[上一轮]" in txt:
            bad.append((r["case_id"], "历史折成题面文本"))
        if IS_V15 and "本次结论需要给出的字段" in last_user:
            bad.append((r["case_id"], "多轮行带字段清单"))
        if not re.search(r"当前时间：\d{4}-\d{2}-\d{2}\n", txt):
            bad.append((r["case_id"], "当前时间不是纯日期"))
        if r.get("bucket", "").startswith(("multiturn", "fam")) and txt.count("<|im_start|>assistant") < 1:
            bad.append((r["case_id"], "多轮行没有历史助手消息"))
        # ★ 空 think 块不许有梯度（Chaoyu 09-02 裁定；闲聊行曾漏掉——画廊抓到）
        resp, rm = list(r["input_ids"])[r["prompt_length"]:], list(r["loss_mask"])[r["prompt_length"]:]
        pat = _empty_think_ids(tokenizer)
        if any(resp[i:i + len(pat)] == pat and any(rm[i:i + len(pat)]) for i in range(len(resp) - len(pat) + 1)):
            bad.append((r["case_id"], "空 think 块有梯度"))
    return {"n": len(rows), "bad": bad, "missing_real_reply": list(_missing)}


_ET: dict[int, list[int]] = {}


def _empty_think_ids(tokenizer):
    k = id(tokenizer)
    if k not in _ET:
        from syncopate.pipeline.sft_replay import EMPTY_THINK
        _ET[k] = tokenizer.encode(EMPTY_THINK, add_special_tokens=False)
    return _ET[k]
