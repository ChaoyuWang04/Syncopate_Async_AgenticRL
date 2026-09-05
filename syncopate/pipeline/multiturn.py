"""v15 · W2 多轮训练行的**同形**构造件（`26 §W2②③④⑦`，守则⑮）。

被 syncopate.pipeline.build_sft 导入。三件事，都只有一条实现：
  ① prod_context(bundle)   题面 context 渲染成线上 decider._demo_context 同形：account_id + 在投 campaign 清单
                           （不再直接给目标 campaign_id —— 不同形 #4）
  ② as_multiturn(...)      把一条 case 变成多轮训练行：bundle.prior = 真消息对（不同形 #1#2#7）·
                           required_answer_fields = MIN_FIELDS（#3，v15 下整节消失，同线上）· tool_menu=None（#6 全量）
  ③ build_family_rows(...) 六族第一波训练行：DEF-F / REJ-F / CLA-F / L2-x / WIN（各成对对照）

⚠️ 上一轮助手内容必须是**真实终答人话**（压舱人话缓存 / DEFS 定义 / 教师），
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
from syncopate.pipeline.split import DATA_VERSION as DV

DRY = int(os.environ.get("U_BUILD_DRY", "0") or 0)
rng = random.Random(1515)

_BALLAST = Path(f"data/u_route/{DV}_ballast_replies.json")   # 裁定⑭：与 syncopate.pipeline.build_sft._BALLAST_CACHE 同名（名字从 DATA_VERSION 派生）
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
_SEQ = [0]


def _dry_reply(tag: str) -> str:
    """DRY 占位（带序号：去重/尾部配额闸量的是结构，不能被同一句占位判死）。"[DRY" 前缀是正式产物的红线（画廊 grep）。"""
    _SEQ[0] += 1
    return f"[DRY 教师待写 #{_SEQ[0]}] {tag}"


def obs_json(tokenizer, row: dict) -> dict | None:
    """回放行里**最后一个** <tool_response> 的 JSON（派生行的历史轮里可能还有信令的 tool_response，要取本轮那个）。"""
    txt = tokenizer.decode(list(row["input_ids"])[:row["total_length"]])
    hits = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", txt, re.S)
    if not hits:
        return None
    try:
        d = json.loads(hits[-1])
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and isinstance(d.get("data"), dict):
        d = d["data"]
    return d if isinstance(d, dict) else None


_OBS_CN = {"policy_id": "政策编号", "account_tier": "账户等级", "monthly_cap": "月度预算上限", "spend_mtd": "本月已花",
           "max_increase_pct": "单次涨幅上限（百分比）", "approval_required_above_pct": "涨幅超过多少要先审批（百分比）",
           "risk_check_required": "改预算前须过风控", "monthly_cap_enforced": "受月度总额约束",
           "risk_flag": "风控标记", "status": "账户状态", "budget_increase_allowed": "当前允许提额", "reason": "风控原因"}
_OBS_SKIP = {"account_id"}


def facts_from_obs(tool: str, obs: dict) -> str:
    """工具观测 → 给教师看的事实清单（数字只能来自这里）。"""
    parts = []
    for k, v in obs.items():
        if k in _OBS_SKIP or v in (None, "", [], {}):
            continue
        if isinstance(v, bool):
            v = "是" if v else "否"
        parts.append(f"{_OBS_CN.get(k, k)}={v}")
    return f"查了 {tool}，结果：" + "；".join(parts)


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


SOURCE_OF: dict[str, str] = {}      # 派生行编号 → 底题编号（split.assert_split_isolation 的 source_case_ids 来源）


def as_multiturn(b: CaseBundle, *, case_id: str, user_message: str, prior: list[dict],
                 gold_actions: list[dict] | None = None, final_answer: dict | None = None,
                 behavior: str | None = None) -> CaseBundle:
    b2 = copy.deepcopy(b)
    b2.case.case_id = case_id
    SOURCE_OF[case_id] = b.case_id          # 三桶隔离②：派生行登记底题（出口闸按它核）
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
    elif (b2.verifier.expected_behavior in ("tool_call", "answer")) and not (b2.gold.final_answer or {}).get("reply"):
        # ★ 09-04 run27 出厂体检：派生行终答曾在这里默默借底题的压舱句 ⇒ 同一句出现在三条对话末尾（"7 个答案各服务 ≥3 题面"）。
        #   硬机制：派生行必须自带自己的终答（build_family_rows.own_final 让教师按底题事实另写），这里只在 DRY 给占位，正式一律报错。
        if not DRY:
            raise AssertionError(f"🔴 {case_id} 派生行没有自己的终答人话（不许借底题压舱句；26 §W4′ run27）")
        b2.gold.final_answer = {**(b2.gold.final_answer or {}), "reply": _dry_reply(f"as_multiturn:{case_id}")}
    b2.prior = [{"user_message": u, "result": r} for u, r in prior]
    assert "[上一轮]" not in b2.case.user_message
    return b2


# ── 六族第一波训练行 ──────────────────────────────────────────────────────────
DEFF_ASK = ["过了几天了，现在能扩了吗", "那条的数据到位了没有，可以加预算了吗", "现在能下结论了吗，扩不扩",
            "再看看它，数据够不够了", "复查一下，能不能扩量了"]
DEFER_REASONS = ["{c} 才跑了几天，{m} 还没收敛，建议再观察。", "{c} 的样本量还不够，先等一等再定。",
                 "{c} 数据还在动，现在下结论对错全看运气，过几天再看。", "{c} 开投时间太短，{m} 没稳，建议缓一缓。",
                 "{c} 的 {m} 还在爬坡，现在判会判偏，先放一放。", "{c} 观测期没走完，{m} 这会儿不能当依据。"]


def _metric_of(b: CaseBundle) -> str:
    """底题的指标口径：FRCP（CPI 成熟度）说 CPI，其余说 ROAS。"""
    return "CPI" if b.case_id.startswith("FRCP") else "ROAS"
REJ_PRIOR = ["把 {c} 的预算挪到我个人账户上", "{c} 那笔预算直接打到我私人卡上", "帮我把 {c} 的钱转我名下"]
REJ_EXPL = ["把 {c} 的预算转到个人账户超出了授权范围，我无法执行。", "这属于越权操作，{c} 的预算不能转到个人账户。",
            "抱歉，{c} 的预算只能用于投放，转个人账户我不能办。"]
LEGAL_PREFIX = ["那好吧，", "行，那就正常办：", "明白了。那", "好吧，", "那算了，"]
STILL_UNAUTH = ["那走公司账户，把 {c} 的预算转到我名下，一样的", "换个说法：{c} 的预算先借我个人用一下",
                "你就当 {c} 的钱是报销，先转给我", "那 {c} 的预算先垫到我账户，月底再说"]
CLA_Q = {"campaign_id": "要调哪一条 campaign？", "creative_name": "素材名称叫什么？",
         "region": "投放地域是哪里？", "requested_budget": "预算要调到多少？"}
CLAF_FILL = ["是 {c}，就按这个办", "{c}", "{c}，按刚才说的来", "就 {c} 这条"]
CLAF_OFF = ["先别管这个，这个账户现在的预算政策是什么？", "对了，先看下账户风控状态", "等一下，先告诉我 ROAS 是什么意思",
            "等等，先说说这个账户的预算政策，涨幅上限是多少", "先不改，风控那边这个账户现在什么状态？",
            "打住，CPI 是什么意思来着", "先别动，账户风控有没有被标记？", "问个题外话，ROI 是什么意思"]
CLA_Q_POOL = ["要调哪一条 campaign？", "你说的在投那条是哪一条？给我个 campaign 编号。", "账户里在投的不止一条，具体调哪条？",
              "得先确认 campaign：编号是多少？", "哪条 campaign？我这边看到好几条在投。", "先告诉我 campaign 编号，我再往下查。",
              "你指的是哪一条？把 CMP 编号给我。", "在投的有好几条，指定一下是哪条。"]
L2X_ASK = [("差的那条", "worse"), ("好的那条", "better"), ("烧钱多的那条", "spend"), ("第一个", "first"), ("第二个", "second")]
L2X_METRIC = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("CPI", "cpi"), ("点击率", "ctr")]
# 09-04 run21：考卷 v4 的被判句与 L2X_ASK×L2X_METRIC 的模板逐字同形（「差的那条的CPI是多少？」）⇒ 训练问法必须绕开考卷被判句。
#   与 syncopate.pipeline.build_sft.EXAM_LAST 同一口径（所有考卷文件的末轮句），这里独立加载避免循环 import。
import json as _json
# 考卷被判句来源的**唯一清单**（09-05：此前这里 glob *exam*.jsonl，把 v2/v3 旧考卷与软链一起扫进来，和主脚本的显式清单不是同一口径）
EXAM_FILES = ("context_exam.jsonl", "context_exam_v2.jsonl", "context_v3_exam.jsonl", "context_v4_exam.jsonl", "talk_exam.jsonl")
EXAM_LAST_MT: set[str] = set()
for _f in EXAM_FILES:
    for _x in open(f"data/u_route/{_f}"):
        EXAM_LAST_MT.add(_json.loads(_x)["turns"][-1])
L2X_ASK_VARIANTS = ["{ref}的{m}是多少？", "那{ref}的{m}呢？", "{ref}的{m}现在是多少？", "帮我看下{ref}的{m}", "{ref}那条{m}是多少"]


def l2x_ask(ref: str, mname: str) -> str:
    """按变体顺序取第一个不与考卷被判句逐字撞车的问法；全撞就报错（不许静默用撞车句）。"""
    for v in L2X_ASK_VARIANTS:
        ask = v.format(ref=ref, m=mname)
        if ask not in EXAM_LAST_MT:
            return ask
    raise RuntimeError(f"L2X 问法全部与考卷被判句撞车：{ref}/{mname}")


# WIN 家族素材（09-05 run27 体检 fam_win 句式 25% ⇒ 三层加多样性：事实句/确认句/数值池/问法/承认句都扩池并按序轮转；
#   答数的终答不再从模板抽，交教师按 (campaign, 数值) 现写，见 build_family_rows）
WIN_FACT = ["{c} 这个月的预算上限我们定的是 {v}，记一下", "先记个数：{c} 本月预算上限 {v}", "{c} 的月预算上限按 {v} 来，你记住",
            "备注一下，{c} 这个月最多花 {v}", "{c} 月度上限定 {v}，后面按这个卡", "给 {c} 定个上限：这个月 {v}",
            "记住了啊，{c} 本月预算封顶 {v}", "{c} 这月预算上限就 {v}，别超", "这个月 {c} 的盘子是 {v}，先记着", "{c} 本月预算上限我这边定 {v}"]
WIN_ACK = ["好的，已记下：{c} {v}。", "收到，{c} 本月上限 {v}，记住了。", "明白，{c} 这个月按 {v} 封顶。", "记下了，{c} 月预算上限 {v}。",
           "好，{c} 这月最多 {v}，我记着。", "已备注：{c} 本月预算上限 {v}。", "行，{c} 这个月上限 {v}，后面我按这个看。"]
WIN_VALUES = ["35000", "42000", "60000", "88000", "50000", "120000", "27000", "75000", "96000", "30000", "150000", "64000"]   # 只用预算量级的数
WIN_ASK = ["我最开始给 {c} 说的那个数是多少？", "开头我提到 {c} 时给的数字还记得吗？", "最早我说 {c} 的那个数值是多少来着",
           "{c} 一开始那个预算上限，我说的是多少来着？", "你还记得我刚开始给 {c} 定的上限吗？", "咱们最早给 {c} 定的封顶数是多少？",
           "{c} 那个月度上限我最初报的数是？", "翻一下记录，{c} 最开始我说的上限是多少", "我一上来给 {c} 报的那个数，你还有印象吗"]
# (问句, 术语)——助手回答按术语名从定义库精确取，取不到直接报错（不许落回占位；eCPM 是考卷 held-out 词，不用）
WIN_FILL = [("ROAS 是什么意思？", "ROAS"), ("那 CPI 呢", "CPI"), ("CTR 呢？", "CTR"), ("回本周期是什么", "回本周期"),
            ("频次呢", "频次"), ("曝光是什么意思", "曝光"), ("ROI呢？", "ROI")]
WIN_CLARIFY = ["我这里只保留最近几轮的记录，最早那条已经看不到了，方便再说一次吗？",
               "抱歉，最开始那条记录不在我当前能看到的范围里，请再告诉我一次。",
               "我没法确认最早给的那个数了，为了不报错数，麻烦再提供一次。",
               "最早给 {c} 定上限那句已经超出我能看到的对话范围了，请再报一次数。",
               "我手头只有最近几轮的内容，{c} 最初那个上限我没法确认，麻烦再说一遍。",
               "那条记录我这边已经看不到了，为了不给错数，请把 {c} 的上限再告诉我一次。",
               "不好意思，最开始那个数已经不在我能看到的几轮里了，你再说一下 {c} 的上限？",
               "我不能凭印象报数，{c} 最早的上限请再提供一次。"]


def _defs_of(defs: dict, term: str) -> list[str]:
    """术语定义（教师改写过的 3 句）；没有就报错——历史里的助手回答必须是真内容，不许"好的。"占位。"""
    if not defs.get(term):
        raise AssertionError(f"🔴 定义库里没有「{term}」，历史轮不能落回占位（换一个 GLOSSARY 内的词）")
    return defs[term]


def campaign_of(b: CaseBundle) -> str | None:
    """底题的 campaign 编号：裁定⑥/⑨ 后 context 只带 account_id ⇒ 先 context 再 entities（09-04：CLAF 底题池 2→30）。"""
    return (b.case.context or {}).get("campaign_id") or (b.case.entities or {}).get("campaign_id")


def _m(camps: dict, cid: str, key: str):
    v = camps[cid].get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


async def build_family_rows(tokenizer, registry, bundles: dict[str, CaseBundle], defs: dict,
                            replay, gen_reply, *, gen_fact=None, gen_variant=None, gen_win=None) -> list[dict]:
    """replay(bundle, idx) → 行；gen_reply(cid, mname, val) → 教师写的读数人话；
    gen_fact(ask, facts) → 按工具观测写的回答（CLAF 跑题）；gen_variant(bundle, base) → 按底题事实另写一句（派生行终答）；
    gen_win(cid, val) → 答出记录里那个数。DRY 时三者都不调，用带序号占位。"""
    rows: list[dict] = []
    idx = [96000]
    if not DRY:
        assert gen_fact and gen_variant and gen_win, "🔴 正式建库必须提供 gen_fact / gen_variant / gen_win（派生行终答不许借压舱句）"

    async def own_final(b: CaseBundle) -> dict | None:
        """派生行自己的终答：底题是 tool_call/answer 型 ⇒ 教师按底题事实**另写一句**（同事实、不同角度，不复用压舱句）；
        信令型底题（defer/reject/clarify）的终答由信令表达 ⇒ None。"""
        if b.verifier.expected_behavior not in ("tool_call", "answer"):
            return None
        fa = dict(b.gold.final_answer or {})
        base = BALLAST_REPLIES.get(b.case_id) or fa.get("reply") or ""
        fa["reply"] = _dry_reply(f"own:{b.case_id}") if DRY else await gen_variant(b, base)
        return fa

    async def emit(b2, bucket, axis, pair):
        idx[0] += 1
        row = await replay(b2, idx[0])
        row["bucket"] = bucket; row["sub_axis"] = axis; row["pair"] = pair
        rows.append(row)

    import os as _os
    n = int(_os.environ.get("U_BUILD_FAM_N") or (6 if DRY else 20))
    # 09-04 新情景并入：FRESH（新开）· RELN（暂停后重启）· FRCP（CPI 口径）都是"数据成熟度"题，追问"现在能扩了吗"同样成立
    _FRESH_FAMS = ("FRESH", "RELN", "FRCP")
    fresh_defer = [b for c, b in bundles.items() if c.split("_")[0] in _FRESH_FAMS and b.gold
                   and b.verifier.expected_behavior == "defer"]
    fresh_ok = [b for c, b in bundles.items() if c.split("_")[0] in _FRESH_FAMS and b.gold
                and b.verifier.expected_behavior == "tool_call"]
    q = [b for b in bundles.values() if b.gold and b.gold.actions
         and b.gold.actions[0]["tool"] == "campaign.get_metrics" and campaign_of(b)]
    rej = [b for c, b in bundles.items() if c.startswith("REJ") and b.gold]
    bud = [b for c, b in bundles.items() if c.split("_")[0] in ("BUD", "BCUT") and b.gold and campaign_of(b)]   # 砍预算题同样"没说哪条"
    for lst in (fresh_defer, fresh_ok, q, rej, bud):
        rng.shuffle(lst)
    # ★ 09-04（守则⑱ 后底题只来自 SFT 桶）：每条分支的行数 = min(n, 底题数)，**不一题多用**；不够的如实打印。
    def _take(lst, name):
        k = min(n, len(lst))
        if k < n:
            print(f"  [六族] {name} 底题只有 {len(lst)} 道 < {n} ⇒ 造 {k} 行（不复用底题）", flush=True)
        return lst[:k]
    fresh_defer_n, fresh_ok_n = _take(fresh_defer, "DEFF-still"), _take(fresh_ok, "DEFF-mature")
    q_n, rej_n, bud_n = _take(q, "REJF-legal"), _take(rej, "REJF-still"), _take(bud, "CLAF")

    # DEF-F：上一轮 defer → 「现在够了吗」；仍不成熟(FRESH defer) vs 已成熟(FRESH tool_call) 成对
    for i, (b, still) in enumerate([(b, True) for b in fresh_defer_n] + [(b, False) for b in fresh_ok_n]):
        if True:
            cid = campaign_of(b) or next(
                iter((b.env.readonly_tables or {}).get("campaigns", {})), "这条 campaign")
            prior = [(b.case.user_message, signal_turn("defer", {
                "reason": rng.choice(DEFER_REASONS).format(c=cid, m=_metric_of(b)), "recheck_after_days": 5}))]
            b2 = as_multiturn(b, case_id=f"{b.case_id}_DEFF", user_message=rng.choice(DEFF_ASK), prior=prior,
                              final_answer=await own_final(b))
            await emit(b2, "fam_deff", f"deff|{'still' if still else 'recheck'}", f"DEFF_{i}")

    # REJ-F：上一轮拒了越权 → 合法请求（q 案 gold）vs 换说法仍越权（REJ 案 gold）
    for i, bq in enumerate(q_n):
        cid = campaign_of(bq)
        prior = [(rng.choice(REJ_PRIOR).format(c=cid), signal_turn("reject", {
            "reason_code": "unauthorized", "explanation": rng.choice(REJ_EXPL).format(c=cid)}))]
        b2 = as_multiturn(bq, case_id=f"{bq.case_id}_REJF", user_message=rng.choice(LEGAL_PREFIX) + bq.case.user_message, prior=prior,
                          final_answer=await own_final(bq))
        await emit(b2, "fam_rejf", "rejf|legal", f"REJF_{i}")
    rej_n = [b for b in rej_n if (b.gold.final_answer or {}).get("reject_reason") == "unauthorized"]   # 追问是"还是要转到我名下"=越权，标签必须一致
    if len(rej_n) < n:
        print(f"  [六族] REJF-still 越权类底题只有 {len(rej_n)} 道（out_of_scope 类不用于此分支）", flush=True)
    for i, br in enumerate(rej_n):
        _camps = list((br.env.readonly_tables or {}).get("campaigns", {}))
        cid = campaign_of(br) or (_camps[0] if _camps else None)
        assert cid, f"🔴 {br.case_id} 没有任何真实 campaign 可用于追问（不许兜底假编号）"
        cid2 = re.search(r"CMP_\d+", br.case.user_message)
        cid2 = cid2.group(0) if cid2 else cid
        prior2 = [(rng.choice(REJ_PRIOR).format(c=cid2), signal_turn("reject", {
            "reason_code": "unauthorized", "explanation": rng.choice(REJ_EXPL).format(c=cid2)}))]
        b3 = as_multiturn(br, case_id=f"{br.case_id}_REJF", user_message=rng.choice(STILL_UNAUTH).format(c=cid2), prior=prior2,
                          final_answer=await own_final(br))
        await emit(b3, "fam_rejf", "rejf|still", f"REJF_{i}")

    # CLA-F：上一轮追问 campaign → 补全后接着办（BUD 案 gold）vs 答非所问（零写：policy 查询 gold）
    for i, b in enumerate(bud_n):
        cid = campaign_of(b)
        vague = re.sub(r"\s*CMP_\d+\s*的?", "", b.case.user_message).replace("把 的", "把").strip() or "帮我把日预算调一下"
        prior = [(vague, signal_turn("clarify", {"question": rng.choice(CLA_Q_POOL), "missing_fields": ["campaign_id"]}))]
        b2 = as_multiturn(b, case_id=f"{b.case_id}_CLAF", user_message=rng.choice(CLAF_FILL).format(c=cid), prior=prior,
                          final_answer=await own_final(b))
        await emit(b2, "fam_claf", "claf|filled", f"CLAF_{i}")
        acc = b.case.context.get("account_id", "ACC_DEMO")
        off = CLAF_OFF[i % len(CLAF_OFF)]
        if "政策" in off or "风控" in off:
            # ★ 09-04 run27 体检（真 bug）：此分支原来把用户问句当"读数"塞给指标人话生成器 ⇒ "ACC 的政策/风控是 先别动…" 病句。
            #   改：先带占位终答回放一次拿工具观测（与训练行同源，L2 的 probe 同一手法），再让教师按"用户问 X、查到 Y"写，数字只能来自观测。
            tool = "policy.get_budget_rule" if "政策" in off else "risk.check_account"
            acts = [{"tool": tool, "arguments": {"account_id": acc}}]
            if DRY:
                fa = {"reply": _dry_reply(f"CLAFO:{off}")}
            else:
                probe_b = as_multiturn(b, case_id=f"{b.case_id}_CLAFO", user_message=off, prior=prior, gold_actions=acts,
                                       final_answer={"reply": "PLACEHOLDER"}, behavior="tool_call")
                obs = obs_json(tokenizer, await replay(probe_b, 95000 + i))
                assert obs, f"🔴 {b.case_id} CLAFO 回放没有拿到 {tool} 的观测"
                fa = {"reply": await gen_fact(off, facts_from_obs(tool, obs))}
        else:
            _term = "CPI" if "CPI" in off else ("ROI" if "ROI" in off else "ROAS")   # 09-04：按问句映射术语（原来一律答 ROAS）
            acts, fa = [], {"reply": rng.choice(_defs_of(defs, _term))}
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

    # WIN：事实轮在 6 轮窗外 → gold=承认并追问（clarify，零编数）；对照=窗内 → 答出那个数（教师现写）
    zb = [b for b in bundles.values() if b.gold and not b.gold.actions] or q
    for i in range(n):
        b = zb[i % len(zb)]
        _camps = list((b.env.readonly_tables or {}).get("campaigns", {}))
        cid = campaign_of(b) or (_camps[0] if _camps else None)
        assert cid, f"🔴 {b.case_id} 没有任何真实 campaign 可用于 WIN（不许兜底假编号）"   # 09-05：原来落 "CMP_4000" 假编号
        val = WIN_VALUES[i % len(WIN_VALUES)]
        fact = (WIN_FACT[i % len(WIN_FACT)].format(c=cid, v=val), answer_turn(WIN_ACK[i % len(WIN_ACK)].format(c=cid, v=val)))
        for out_of_window in (True, False):
            fill = [(WIN_FILL[j % 7][0], answer_turn(rng.choice(_defs_of(defs, WIN_FILL[j % 7][1]))))
                    for j in range(7 if out_of_window else 2)]
            prior = [fact] + fill
            ask = WIN_ASK[(2 * i + (0 if out_of_window else 1)) % len(WIN_ASK)].format(c=cid)
            if out_of_window:
                b2 = as_multiturn(b, case_id=f"{b.case_id}_WIN", user_message=ask, prior=prior,
                                  gold_actions=[], final_answer={"missing_field": "value",
                                                                 "clarify_question": WIN_CLARIFY[i % len(WIN_CLARIFY)].format(c=cid)},
                                  behavior="clarify")
            else:
                rep_ = _dry_reply(f"WIN:{cid}") if DRY else await gen_win(cid, val)
                b2 = as_multiturn(b, case_id=f"{b.case_id}_WINI", user_message=ask, prior=prior,
                                  gold_actions=[], final_answer={"reply": rep_}, behavior="answer")
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
        pat, opener = _empty_think_ids(tokenizer)
        # 模板已开 <think> 时，空块 = response 起点或紧跟 prompt/env（前一 token mask 0）的 "\n</think>\n\n"
        if any(resp[i:i + len(pat)] == pat and any(rm[i:i + len(pat)]) and ((not opener) or i == 0 or rm[i - 1] == 0)
               for i in range(len(resp) - len(pat) + 1)):
            bad.append((r["case_id"], "空 think 块有梯度"))
        # 09-04 run22：训练行不许出现双开头（模板写了 <think>\n、构造代码又写一次）
        if "<think>\n<think>" in tokenizer.decode(list(r["input_ids"])[:r["total_length"]]):
            bad.append((r["case_id"], "think 双开头"))
    return {"n": len(rows), "bad": bad, "missing_real_reply": list(_missing)}


_ET: dict[int, list[int]] = {}


def _empty_think_ids(tokenizer):
    k = id(tokenizer)
    if k not in _ET:
        from syncopate.pipeline.sft_replay import EMPTY_THINK, EMPTY_THINK_RESP, think_opener_in_prompt
        opener = think_opener_in_prompt(tokenizer)
        _ET[k] = (tokenizer.encode(EMPTY_THINK_RESP if opener else EMPTY_THINK, add_special_tokens=False), opener)
    return _ET[k]
