#!/usr/bin/env python
"""v14.5 · S4 数据集构建（24 §4-P2 最终设计的执行件；取代 u_build_v14.py）。

    .venv/bin/python scripts/u_build_v14_5.py   # → data/sft/v14_5/{train,val}.parquet

总原则：程序造事实 · 教师穿语言 · 判据把关。
教师走 vLLM（4B 语言层 @:8210 · 8B CoT @:8211），由 u_p2_v145_chain.sh 负责起停。
桶与门槛：24 §4-P2（份额±3pp · 密度闸 · OOV 断言 · 泄漏断言 · 冻结校验 · sub_axis 全量）。
L2 读数用两次回放法：先占位回放取真实观测值 → 教师带真值写 reply → 二次回放定稿。
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

rng = random.Random(1455)
T4B = "http://127.0.0.1:8210/v1"
T8B = "http://127.0.0.1:8211/v1"
SICK = re.compile(r"指指|的的|是是|了了")
PERSONA_LEAK = re.compile(r"我(每天)?(喝|吃|睡觉|跑步|锻炼|健身)|我的身体")
MIN_FIELDS = [
    {"key": "summary", "description": "结论的机器可校验形式（简短标签或数值）"},
    {"key": "reply", "description": "给用户读的完整回复：一到三句自然语言，说清结论和依据"},
]
OOV = json.load(open("data/u_route/oov_holdout_terms.json"))["terms"]
PATTERNS = json.load(open("data/u_route/ellipsis_patterns.json"))["templates"]
SUB_TRAIN = [t["template"] for t in PATTERNS
             if t["split"] == "train" and t["kind"] == "substitution"
             and t["template"].count("{X}") == 1
             and not re.search(r"\{X\}[中年里在]", t["template"])]
CONT_TRAIN = [t["template"] for t in PATTERNS
              if t["split"] == "train" and t["kind"] == "continuation"][:8]
from u_build_v14 import GLOSSARY  # noqa: E402  61 词（定义要点作教师素材，不再直拼 gold）

STYLES = ["亲切口语", "简洁专业", "轻松幽默"]

EXAM_LAST = set()          # 考卷被判轮句（训练构造时逐字规避）
for _fn in ("context_exam.jsonl", "context_exam_v2.jsonl", "talk_exam.jsonl"):
    for _x in open(f"data/u_route/{_fn}"):
        EXAM_LAST.add(json.loads(_x)["turns"][-1])


_SEED = [0]


async def teach(client, base, prompt, sys_prompt="", max_tokens=200, temp=0.8):
    _SEED[0] += 1     # ⚠️ vllm serve 默认 seed=0 ⇒ 同请求采样确定性；逐请求换 seed 才有多样性
    r = await client.post(f"{base}/chat/completions", json={
        "model": "t", "temperature": temp, "top_p": 0.95, "max_tokens": max_tokens,
        "seed": _SEED[0],
        "messages": ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
        + [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def clean_reply(t: str) -> str:
    t = re.sub(r"^[\"“」『]|[\"”」『]$", "", t.strip())
    return t.replace("\n", " ").strip()


def has_oov(t: str) -> bool:
    """教师产文里偶然出现 OOV held-out 词 ⇒ 该条重生成（保 OOV=0 建库断言）。"""
    return any(x in t for x in OOV)


def value_forms(v):
    forms = {str(v)}
    try:
        f = float(v)
        forms |= {f"{f:g}", f"{f:.2f}".rstrip("0").rstrip("."), f"{f:.1f}"}
        if f < 1:
            forms |= {f"{f*100:g}", f"{f*100:.1f}".rstrip("0").rstrip("."), f"{f*100:.0f}"}
        if f == int(f):
            iv = int(f)
            forms |= {str(iv), f"{iv:,}"}
            if iv >= 10000:
                forms.add(f"{iv/10000:g}万")
    except (ValueError, TypeError):
        pass
    return [x for x in forms if x]


# ═══════════ Stage A · 教师语言层（4B）═══════════

async def gen_defs(client) -> dict[str, list[str]]:
    """61 词 × 3 版定义（修「指指」：教师产完整句，禁模板拼接）。"""
    out = {}
    for term, hint in GLOSSARY.items():
        versions = []
        angles = [
            f"用一句自然的中文向运营同事解释投放术语「{term}」（参考要点：{hint}）。"
            f"要求：以「{term}」开头、一句话说完、不要列条目。直接输出这句话。",
            f"换一种说法，用大白话向新人解释「{term}」是什么（背景知识：{hint}）。"
            f"不要照抄背景知识的原句，一句话，以「{term}」开头，直接输出。",
            f"用一个具体例子帮同事理解「{term}」（它的含义：{hint}）。"
            f"一句话说完，以「{term}」开头，不要照抄含义原文，直接输出。",
            f"如果同事把「{term}」理解错了，你会怎么一句话纠正并讲清它？"
            f"（正确含义：{hint}）以「{term}」开头，直接输出这句话。",
            f"向老板汇报时顺口解释一下「{term}」（含义：{hint}），一句话，"
            f"以「{term}」开头，措辞和书面定义不同，直接输出。",
        ]
        for i in range(5):
            if len(versions) >= 3:
                break
            t = await teach(client, T4B, angles[i], temp=0.9)
            t = clean_reply(t)
            if (term in t and 12 <= len(t) <= 90 and not SICK.search(t)
                    and not has_oov(t) and t not in versions and "{" not in t):
                versions.append(t)
        assert len(versions) >= 2, f"定义生成不足：{term}"
        out[term] = versions
    return out


async def gen_chat(client, bank) -> list[dict]:
    """chat 契约壳素材：80 条（其中 ~16 条带第二轮 continuation 追问）。"""
    out = []
    for i, item in enumerate(bank[:96]):
        if len(out) >= 80 + 10:            # 10 条留给 val held-out
            break
        style = STYLES[i % 3]
        sysp = (f"你是手游买量投放团队的 AI 助手，风格{style}。用一到三句中文回应同事，"
                f"不列条目，不提工具或系统，办不到的事坦诚说明边界。")
        rep = clean_reply(await teach(client, T4B, item["prompt"], sysp, 220))
        if not (20 <= len(rep) <= 400) or "<|" in rep or "{" in rep \
                or SICK.search(rep) or PERSONA_LEAK.search(rep) or has_oov(rep):
            continue
        summ = clean_reply(await teach(
            client, T4B, f"用不超过 12 个字概括这句话的主旨（直接输出概括）：{rep}", "", 30))[:20]
        if not summ or rep.startswith(summ):
            summ = f"{style}回应"
        row = {"prompt": item["prompt"], "reply": rep, "summary": summ,
               "style": style, "source": item["source"], "turns": 1}
        # 20% 补第二轮：真实 continuation 追问（"还有吗"族）→ 教师续答
        if i % 5 == 0 and CONT_TRAIN:
            fu = rng.choice(CONT_TRAIN)
            rep2 = clean_reply(await teach(
                client, T4B,
                f"上一轮同事问：「{item['prompt']}」你答：「{rep}」现在对方追问：「{fu}」"
                f"请自然地补充回应（一到三句，不重复原话）。", sysp, 220))
            if 20 <= len(rep2) <= 400 and not SICK.search(rep2) and "{" not in rep2:
                row.update({"turns": 2, "followup": fu, "reply2": rep2})
        out.append(row)
    assert len(out) >= 85, f"chat 素材不足 {len(out)}"
    return out


L2_TAILS: Counter = Counter()
TAIL_CAP = 10          # 200 行里同尾 ≤5%，给 10% 密度闸留一半余量


def _tail_ok(rep: str) -> bool:
    return L2_TAILS[rep[-10:]] < TAIL_CAP


def _tail_note(rep: str) -> None:
    L2_TAILS[rep[-10:]] += 1


ANGLES_L2 = ["可附一句简短观察", "顺带说一句这个数是高是低",
             "用口头汇报的口气说", "顺带提示下一步可以看什么"]


async def gen_l2_reply(client, cid, mname, val) -> str:
    for k in range(4):
        rep = clean_reply(await teach(
            client, T4B,
            f"你查到了 {cid} 的{mname}是 {val}。用一两句自然中文把这个结果告诉用户，"
            f"必须包含数值 {val}（可换算写法），{ANGLES_L2[k % 4]}，不要用固定套话，"
            f"收尾方式不要千篇一律。",
            temp=0.9, max_tokens=120))
        clean = rep.replace(",", "").replace("，", "")
        if any(f in clean for f in value_forms(val)) and 10 <= len(rep) <= 160 \
                and not SICK.search(rep) and not has_oov(rep) and _tail_ok(rep):
            _tail_note(rep)
            return rep
    rep = f"{cid} 的{mname}是 {val}，需要进一步对比随时说。"   # 兜底（计数）
    _tail_note(rep)
    return rep


# ═══════════ Stage B · CoT 难例（8B 逐步 think + 承诺闸）═══════════

ASST = "<|im_start|>assistant"


async def gen_cot(client, tok, max_rows=100) -> list[dict]:
    from u_teacher_probe import gold_values
    hard_ids = set(json.load(open("_audit/triage/cand_v13r2_e1/卡死.json")))
    hard_ids |= set(json.load(open("_audit/triage/cand_v13r2_e1/死格.json")))
    # ⚠️ triage id 是冻结评测集的 case（与训练集刻意隔离，交集恒空）⇒ 映射到模板族：
    #    在训练集中选同族 case，优先长轨迹（多步 = 思考有用武之地）
    pref = Counter(x.split("_")[0] for x in hard_ids)
    hard_pref = {p for p, c in pref.items() if c >= 3}
    print(f"[CoT] 难例模板族：{dict(pref)} → 选族 {sorted(hard_pref)}")
    df = pd.concat([pd.read_parquet("data/sft/v13/train.parquet"),
                    pd.read_parquet("data/sft/v13/val.parquet")])
    cases = [r for _, r in df.iterrows()
             if str(r.case_id).split("_")[0] in hard_pref]
    cases.sort(key=lambda r: -int(r.total_length))     # 长轨迹（多步）优先
    capped, percnt = [], Counter()
    for r in cases:
        p = str(r.case_id).split("_")[0]
        if percnt[p] >= 30:
            continue
        percnt[p] += 1
        capped.append(r)
    cases = capped
    rng.shuffle(cases)
    print(f"[CoT] 难例池 {len(cases)}（族内限额 30）")
    out, tried = [], 0
    reject_step = Counter()

    async def one_step_think(ctx: str):
        _SEED[0] += 1
        r = await client.post(f"{T8B}/completions", json={
            "model": "t", "prompt": ctx + "<think>\n", "max_tokens": 1100,
            "seed": _SEED[0], "temperature": 0.7, "top_p": 0.95})
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    async def step_rejection_sample(ctx: str, gold_kind, gold_act, n=6):
        """R1 式 rejection sampling：并发 n 条思考路径，收教师自己也选了 gold 动作的
        那条（思行一致构造性成立——不是给 gold 编理由，是从真决策里选对的）。"""
        gens = await asyncio.gather(*[one_step_think(ctx) for _ in range(n)],
                                    return_exceptions=True)
        for gen in gens:
            if isinstance(gen, Exception) or "</think>" not in gen:
                continue
            think, post = gen.split("</think>", 1)
            think = think.strip()
            cjk = len(re.findall(r"[一-鿿]", think)) / max(1, len(think))
            if not think or len(think) > 4096 or cjk < 0.5:
                continue
            if first_action(post) == (gold_kind, gold_act):
                return think
        return None

    def first_action(text: str):
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        if m:
            try:
                return ("tool", json.loads(m.group(1)).get("name"))
            except json.JSONDecodeError:
                return ("tool", None)
        m = re.search(r'"behavior"\s*:\s*"(\w+)"', text)
        return ("behavior", m.group(1)) if m else (None, None)

    for r in cases:
        if len(out) >= max_rows:
            break
        ids = list(r.input_ids)
        full = tok.decode(ids[:r.total_length])
        segs = full.split(ASST)
        if len(segs) < 2:
            continue
        # v2 设计（08-29 第二迭代）：think 只插**终答步**——上下文已含全部工具观测，
        # 思考集中在「看完数据怎么判」（难例的难点=终判：cap 权衡/预算数学/证据取舍）；
        # 中间步保持原样。理由：gold 工具序非唯一合理序，逐步全等在深轨迹指数衰减
        # （6 采样实测 104 案全灭：弃1步45·2步48）；终答行为匹配率高且思行一致依旧
        # 构造性成立。全轨迹自主 rollout 版（R1 正统）记 P3 后增强项。
        si_last = len(segs) - 1
        ctx = ASST.join(segs[:si_last]) + ASST + "\n"
        g_kind, g_act = first_action(segs[si_last])
        tried += 1
        think = await step_rejection_sample(ctx, g_kind, g_act, n=8)
        if think is None:
            reject_step[si_last] += 1
            continue
        thinks = {si_last: think}
        vals = gold_values(segs[-1])
        # 末答闸沿用：终段 gold 值仍是原 gold（我们只插 think 不改答案）
        new_segs = [segs[0]]
        for si in range(1, len(segs)):
            body = segs[si]
            if si in thinks:
                if "<think>\n\n</think>" in body[:30]:
                    body = body.replace("<think>\n\n</think>",
                                        f"<think>\n{thinks[si]}\n</think>", 1)
                else:
                    body = f"\n<think>\n{thinks[si]}\n</think>" + body
            new_segs.append(body)
        new_full = ASST.join(new_segs)
        cut = new_full.find(ASST)          # prompt = 首个 assistant 头之前
        head = new_full[:cut + len(ASST) + 1]
        tail = new_full[len(head):]
        # ★ mask 按段落构造：<|im_start|>user…<|im_end|>（工具返回=环境消息）必须为 0
        #   ——「把工具返回算进 loss = 教模型复述环境」是 collate 明令禁止的
        ids_p = tok(head, add_special_tokens=False).input_ids
        ids_all, mask = list(ids_p), [0] * len(ids_p)
        for part in re.split(r"(<\|im_start\|>user.*?<\|im_end\|>)", tail, flags=re.S):
            if not part:
                continue
            pids = tok(part, add_special_tokens=False).input_ids
            flag = 0 if part.startswith("<|im_start|>user") else 1
            ids_all += pids
            mask += [flag] * len(pids)
        out.append({"case_id": f"{r.case_id}_COT5", "input_ids": ids_all,
                    "loss_mask": mask,
                    "prompt_length": len(ids_p), "total_length": len(ids_all),
                    "supervised_tokens": sum(mask),
                    "behavior": r.behavior, "bucket": "cot_hard",
                    "sub_axis": f"{r.case_id.split('_')[0]}|laststep{si_last}",
                    "signal_class": "graded", "split": "train",
                    "index": 95000 + len(out), "n_vals": len(vals)})
        print(f"  [CoT] 收 {r.case_id}（{len(thinks)} 步）→ {len(out)}/{max_rows}", flush=True)
    print(f"[CoT] 保留 {len(out)}，尝试步数 {tried}，弃于第 N 步分布 {dict(reject_step)}")
    return out


# ═══════════ Stage C · 结构桶（回放）═══════════

async def build_l2_l1(tokenizer, registry, client):
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_bundles
    bundles = load_bundles(Path("data/batches/v13"))
    q_bundles = [b for b in bundles.values()
                 if b.gold and b.gold.actions
                 and b.gold.actions[0]["tool"] == "campaign.get_metrics"
                 and b.case.context.get("campaign_id")]
    z_bundles = [b for b in bundles.values() if b.gold and not b.gold.actions]
    rng.shuffle(q_bundles)
    rng.shuffle(z_bundles)
    METRICS = [("消耗", "spend_7d"), ("安装量", "installs_7d"), ("ROAS", "roas_d7"),
               ("CPI", "cpi"), ("点击率", "ctr"), ("频次", "frequency")]

    async def replay(b, idx):
        return await build_sft_row(b, tokenizer=tokenizer, registry=registry,
                                   index=idx, split="train", config=None)

    def obs_value(row_dict, key):
        txt = tokenizer.decode(list(row_dict["input_ids"])[:row_dict["total_length"]])
        m = re.search(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", txt, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1)).get(key)
        except json.JSONDecodeError:
            return None

    defs = DEFS  # 全局（Stage A 产物）
    l2_rows, l1_rows, skipped, fallback = [], [], 0, 0
    # ---- L2 ~200 + 10 val：句式×工具×对象 ----
    obj_seq = (["same"] * 60 + ["switch"] * 25 + ["compare"] * 15)
    tool_seq = (["campaign.get_metrics"] * 70 + ["mmp.get_attribution"] * 30)
    i = 0
    for b in q_bundles:
        if len(l2_rows) >= 210:
            break
        cid = b.case.context["campaign_id"]
        mname, mkey = METRICS[i % len(METRICS)]
        obj = obj_seq[i % 100]
        tool = tool_seq[i % 100]
        pat = rng.choice(SUB_TRAIN)
        i += 1
        b2 = copy.deepcopy(b)
        prev = (b.gold.final_answer or {}).get("summary") or "已给出结论"
        cid2 = cid
        if obj == "switch":
            m2 = re.match(r"(.*?)(\d+)$", cid)
            cid2 = f"{m2.group(1)}{int(m2.group(2)) + 1}" if m2 else cid
        if tool == "campaign.get_metrics":
            ask = pat.replace("{X}", f"{cid2+' 的' if obj=='switch' else '它的'}{mname}") \
                if "{X}" in pat else pat
            if obj == "compare":
                ask = f"对比下它和 {cid2} 的{mname}" if cid2 != cid else f"它的{mname}前后对比下"
        else:
            ask = pat.replace("{X}", "它的归因情况") if "{X}" in pat else "它的归因呢"
        if ask in EXAM_LAST:               # 与考卷被判句逐字撞车 ⇒ 换模板重构
            alt = rng.choice([p for p in SUB_TRAIN if p != pat] or SUB_TRAIN)
            ask = alt.replace("{X}", f"这条的{mname}")
            if ask in EXAM_LAST:
                ask = f"顺手把{mname}也拉一下"
        b2.case.user_message = (f"[上一轮] 用户：{b.case.user_message}\n"
                                f"[上一轮] 助手：{str(prev)[:120]}\n\n{ask}")
        b2.case.case_id = f"{b.case_id}_MT5"
        if tool == "campaign.get_metrics":
            acts = [{"tool": tool, "arguments": {"campaign_id": cid2}}]
            if obj == "compare" and cid2 != cid:
                acts.append({"tool": tool, "arguments": {"campaign_id": cid}})
            b2.gold.actions = acts
            b2.gold.final_answer = {"summary": f"{cid2} {mkey} 查询",
                                    "reply": "PLACEHOLDER"}
            try:
                probe = await replay(copy.deepcopy(b2), 90000)
            except Exception:
                skipped += 1
                continue
            val = obs_value(probe, mkey)
            if val is None:
                skipped += 1
                continue
            rep = await gen_l2_reply(client, cid2, mname, val)
            if rep.endswith("随时说。"):
                fallback += 1
            b2.gold.final_answer = {"summary": f"{cid2} {mkey}={val}", "reply": rep}
        else:
            b2.gold.actions = [{"tool": tool, "arguments": {"campaign_id": cid}}]
            rep = None
            for k in range(4):
                cand_rep = clean_reply(await teach(
                    client, T4B,
                    f"你刚核对了 {cid} 的 MMP 归因数据。用一两句自然中文告诉用户核对结论"
                    f"（{rng.choice(['口径基本一致', '和平台数有小差异要再看', '归因窗口正常', '部分安装未被归因'])}），"
                    f"{ANGLES_L2[k % 4]}，收尾方式不要千篇一律。", temp=0.95, max_tokens=100))
                if 10 <= len(cand_rep) <= 160 and not SICK.search(cand_rep) \
                        and not has_oov(cand_rep) and _tail_ok(cand_rep):
                    rep = cand_rep
                    break
            if rep is None:
                rep = f"{cid} 的归因数据核对完毕，细节口径我再核一轮。"
            _tail_note(rep)
            b2.gold.final_answer = {"summary": f"{cid} 归因已核", "reply": rep}
        try:
            row = await replay(b2, 91000 + len(l2_rows))
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠️ L2 回放失败 {b.case_id}: {str(e)[:100]}")
            continue
        row["bucket"] = "multiturn_l2"
        row["sub_axis"] = f"{tool.split('.')[-1]}|{obj}|{pat[:6]}"
        l2_rows.append(row)

    # ---- L1 ~150 + 10 val：75 概念史 + 75 查询史（与 L2 成对照对）----
    terms = list(GLOSSARY)
    li = 0
    all_forms = SUB_TRAIN + ["那{X}呢", "{X}又是什么", "什么是{X}？", "再说说{X}"]
    while len(l1_rows) < 160 and li < 600:
        li += 1
        b_src = rng.choice(z_bundles) if li % 2 == 0 else rng.choice(q_bundles)
        kind = "concept_hist" if li % 2 == 0 else "query_hist"
        a, t2 = rng.sample(terms, 2)
        pat = rng.choice(all_forms)
        ask = pat.replace("{X}", t2)
        if ask in EXAM_LAST:
            continue                        # 撞被判句直接换对（li 循环量足够）
        b2 = copy.deepcopy(b_src)
        if kind == "concept_hist":
            hist = (f"[上一轮] 用户：{a}是什么意思？\n"
                    f"[上一轮] 助手：{rng.choice(DEFS[a])}\n\n{ask}")
        else:
            prev = (b_src.gold.final_answer or {}).get("summary") or "已给出结论"
            hist = (f"[上一轮] 用户：{b_src.case.user_message}\n"
                    f"[上一轮] 助手：{str(prev)[:120]}\n\n{ask}")
        b2.case.user_message = hist
        b2.case.case_id = f"L1F_{li:04d}"
        b2.gold.actions = []
        d = rng.choice(DEFS[t2])
        b2.gold.final_answer = {"summary": f"{t2} 释义", "reply": d}
        b2.verifier = copy.deepcopy(b2.verifier)
        b2.verifier.required_answer_fields = MIN_FIELDS       # 去标签泄漏
        try:
            row = await replay(b2, 92000 + len(l1_rows))
        except Exception:
            skipped += 1
            continue
        row["bucket"] = "multiturn_l1"
        row["sub_axis"] = f"{kind}|{t2}|{pat[:6]}"
        l1_rows.append(row)
    print(f"[L2] {len(l2_rows)}（回放丢 {skipped}·读数兜底 {fallback}） [L1] {len(l1_rows)}")
    return l2_rows, l1_rows


def build_chat_rows(tokenizer, chat_mat):
    from probe_opd_divergence import render_prompt_text
    rows = []
    for i, c in enumerate(chat_mat):
        if c["turns"] == 1:
            user = c["prompt"]
            gold = {"behavior": "answer",
                    "answer": {"summary": c["summary"], "reply": c["reply"]}}
        else:
            user = (f"[上一轮] 用户：{c['prompt']}\n[上一轮] 助手：{c['reply'][:120]}"
                    f"\n\n{c['followup']}")
            gold = {"behavior": "answer",
                    "answer": {"summary": c["summary"], "reply": c["reply2"]}}
        prompt = render_prompt_text(tokenizer, user, tools=None)
        gtext = json.dumps(gold, ensure_ascii=False) + "<|im_end|>"
        ids_p = tokenizer(prompt, add_special_tokens=False).input_ids
        ids_g = tokenizer(gtext, add_special_tokens=False).input_ids
        rows.append({"case_id": f"CHAT5_{i:04d}", "input_ids": ids_p + ids_g,
                     "loss_mask": [0] * len(ids_p) + [1] * len(ids_g),
                     "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
                     "supervised_tokens": len(ids_g), "split": "train",
                     "index": 93000 + i, "signal_class": "graded",
                     "behavior": "answer", "bucket": "chat_shell",
                     "sub_axis": f"{c['style']}|{c['source']}|t{c['turns']}"})
    return rows


# ═══════════ 门禁 ═══════════

def density_gate(rows, tokenizer, name):
    reps = []
    for r in rows:
        txt = tokenizer.decode(list(r["input_ids"])[r["prompt_length"]:r["total_length"]])
        m = re.search(r'"reply"\s*:\s*"([^"]+)"', txt)
        if m:
            reps.append(m.group(1))
    if not reps:
        return
    tails = Counter(x[-10:] for x in reps if len(x) >= 10)
    top = (tails.most_common(1) or [("", 0)])[0]
    sick = sum(1 for x in reps if SICK.search(x))
    grams = set()
    tot = 0
    for x in reps:
        gs = {x[i:i+3] for i in range(len(x)-2)}
        grams |= gs
        tot += len(gs)
    dist3 = len(grams) / max(1, tot)
    print(f"  [密度:{name}] 最高频尾 {top[1]}/{len(reps)}={top[1]/len(reps):.0%} "
          f"({top[0]!r}) · 病句 {sick} · distinct3={dist3:.2f}")
    assert top[1] / len(reps) <= 0.10, f"🔴 {name} 话术密度超标"
    assert sick == 0, f"🔴 {name} 病句 {sick} 条"


async def main() -> int:
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    global DEFS
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B")
    registry = build_domain().registry
    registry.latency_scale = 0.0
    bank = [json.loads(x) for x in open("data/u_route/chat_bank_v2.jsonl")]

    cache_d = Path("data/u_route/v145_defs.json")
    cache_c = Path("data/u_route/v145_chat_mat.json")
    async with httpx.AsyncClient(timeout=180) as client:
        if cache_d.exists():
            DEFS = json.load(open(cache_d))
            print(f"[A1] 定义缓存命中（{len(DEFS)} 词）")
        else:
            print("[A1] 定义改写 61×3 …", flush=True)
            DEFS = await gen_defs(client)
            json.dump(DEFS, open(cache_d, "w"), ensure_ascii=False)
        if cache_c.exists():
            chat_mat = json.load(open(cache_c))
            print(f"[A3] chat 缓存命中（{len(chat_mat)} 条）")
        else:
            print("[A3] chat 素材 …", flush=True)
            chat_mat = await gen_chat(client, bank)
            json.dump(chat_mat, open(cache_c, "w"), ensure_ascii=False)
        cache_l = Path("data/u_route/v145_l2l1_rows.json")
        if cache_l.exists():
            _c = json.load(open(cache_l))
            l2, l1 = _c["l2"], _c["l1"]
            print(f"[C] L2/L1 缓存命中（{len(l2)}/{len(l1)}）")
        else:
            print("[C] L2/L1 回放构建 …", flush=True)
            l2, l1 = await build_l2_l1(tokenizer, registry, client)
            json.dump({"l2": l2, "l1": l1}, open(cache_l, "w"))
        cache_cot = Path("data/u_route/v145_cot_rows.json")
        if cache_cot.exists():
            cot = json.load(open(cache_cot))
            print(f"[B] CoT 缓存命中（{len(cot)} 行）")
        else:
            print("[B] CoT 难例（8B）…", flush=True)
            cot = await gen_cot(client, tokenizer, max_rows=60)
            json.dump(cot, open(cache_cot, "w"))

    # held-out val 切分（每桶尾部拿走）
    l2, l2v = l2[:200], l2[200:210]
    l1, l1v = l1[:150], l1[150:160]
    chat_rows = build_chat_rows(tokenizer, chat_mat)
    chat_rows, chatv = chat_rows[:80], chat_rows[80:90]

    t13 = pd.read_parquet("data/sft/v13/train.parquet")
    v13v = pd.read_parquet("data/sft/v13/val.parquet")
    # ★ CoT 预算截断必须发生在装配之前（第 5 次发射的教训：截断放在份额计算之后
    #   = 截了个寂寞——train 里还是全量、闸读的还是旧份额）
    non_cot_tok = int(t13.supervised_tokens.sum()) + \
        sum(r["supervised_tokens"] for r in l2 + l1 + chat_rows)
    budget = int(non_cot_tok * 0.19 / 0.81)
    acc, kept = 0, []
    for r in cot:
        if acc + r["supervised_tokens"] > budget:
            break
        acc += r["supervised_tokens"]
        kept.append(r)
    if len(kept) < len(cot):
        print(f"[CoT] 预算截断 {len(cot)}→{len(kept)}（sup-tok 预算 {budget}）")
    cot = kept
    assert len(cot) >= 40, f"🔴 CoT 桶下限闸：仅 {len(cot)} 行（要 ≥40）"
    new_rows = l2 + l1 + chat_rows + cot
    train = pd.concat([t13, pd.DataFrame(new_rows)], ignore_index=True)
    valrows = l2v + l1v + chatv
    for r in valrows:
        r["split"] = "val"
    val = pd.concat([v13v, pd.DataFrame(valrows)], ignore_index=True)

    # ── 门禁 ──
    assert len(t13) == 419, "冻结校验失败"
    tok_by = {"v13": int(t13.supervised_tokens.sum()),
              "l2": sum(r["supervised_tokens"] for r in l2),
              "l1": sum(r["supervised_tokens"] for r in l1),
              "chat": sum(r["supervised_tokens"] for r in chat_rows),
              "cot": sum(r["supervised_tokens"] for r in cot)}
    total = sum(tok_by.values())
    share = {k: v / total for k, v in tok_by.items()}
    print("sup-tok 份额:", {k: f"{v:.1%}" for k, v in share.items()})
    bands = {"v13": (0.52, 0.66), "l2": (0.10, 0.17), "l1": (0.03, 0.09),
             "chat": (0.01, 0.07), "cot": (0.05, 0.20)}
    for k, (lo, hi) in bands.items():
        assert lo <= share[k] <= hi, f"🔴 份额闸：{k}={share[k]:.1%} ∉ [{lo:.0%},{hi:.0%}]"
    density_gate(l2, tokenizer, "L2")
    density_gate(l1, tokenizer, "L1")
    density_gate(chat_rows, tokenizer, "chat")
    # OOV 断言（口径修订 08-29 第7次发射）：held-out 的语义=「这些词的**定义**不被教」
    # （保 L1-oov 考规则泛化），非「字串全语料绝迹」——v13 轨迹/8B think 里的自然词用
    # 不构成定义教学。教学面（L1/chat gold 段 + DEFS 词典）必须 0；其余桶记数上报。
    teach_hits, ambient_hits = 0, 0
    for r in new_rows + valrows:
        txt = tokenizer.decode(list(r["input_ids"])[:r["total_length"]])
        n = sum(1 for t in OOV if t in txt)
        if r.get("bucket") in ("multiturn_l1", "chat_shell"):
            gold = tokenizer.decode(
                list(r["input_ids"])[r["prompt_length"]:r["total_length"]])
            teach_hits += sum(1 for t in OOV if t in gold)
        ambient_hits += n
    for vs in DEFS.values():
        teach_hits += sum(1 for v in vs for t in OOV if t in v)
    print(f"[OOV] 教学面命中 {teach_hits}（必须 0）· 全语料自然词用 {ambient_hits}（上报不判）")
    assert teach_hits == 0, f"🔴 OOV 定义教学泄漏 {teach_hits} 次"
    # 考场泄漏：考卷第二轮句子逐字不得出现在训练 user 文本
    # 泄漏闸口径（第8次发射修订）：**被判轮**逐字必 0（防背答案）；铺垫轮是公共
    # 自然句式（「X是什么意思？」），逐字禁会禁掉整类表达 ⇒ 记数上报不判死
    first_turns = set()
    for fn in ("context_exam.jsonl", "context_exam_v2.jsonl", "talk_exam.jsonl"):
        for x in open(f"data/u_route/{fn}"):
            first_turns.update(json.loads(x)["turns"][:-1])
    leak_last, leak_first = 0, 0
    for r in new_rows:
        txt = tokenizer.decode(list(r["input_ids"])[:r["prompt_length"]])
        leak_last += sum(1 for t in EXAM_LAST if len(t) >= 8 and t in txt)
        leak_first += sum(1 for t in first_turns if len(t) >= 8 and t in txt)
    print(f"[泄漏] 被判轮命中 {leak_last}（必须 0）· 铺垫轮 {leak_first}（上报）")
    assert leak_last == 0, f"🔴 考场被判句泄漏 {leak_last}"
    for r in new_rows + valrows:
        assert r["supervised_tokens"] > 0 and \
            len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"], r["case_id"]

    out = Path("data/sft/v14_5")
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    val.to_parquet(out / "val.parquet")
    axes = Counter(r.get("sub_axis", "?").split("|")[0] for r in new_rows)
    manifest = {"version": "v14.5", "seed": 1455,
                "sources": {"v13_train": len(t13), "multiturn_l2": len(l2),
                            "multiturn_l1": len(l1), "chat_shell": len(chat_rows),
                            "cot_hard": len(cot)},
                "total": len(train), "val": len(val),
                "sup_tok_share": {k: round(v, 4) for k, v in share.items()},
                "axis_counts": dict(axes),
                "gates": "份额±带宽 · 密度 · OOV=0 · 泄漏=0 · 冻结419 全过"}
    json.dump(manifest, open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("✅ v14.5 构建完成，全部门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
