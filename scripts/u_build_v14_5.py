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
import os
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
# ★ 裁定⑬（09-03）：人话教师与思考教师同一个 Qwen3.8-27B 端点；B200/Modal 上由 stack_probe.p_build_v16 起在 :8210，两条 URL 同指
T4B = os.environ.get("SYNCOPATE_TEACHER_LANG_URL", "http://127.0.0.1:8210/v1")    # 人话（enable_thinking=False）
T8B = os.environ.get("SYNCOPATE_TEACHER_THINK_URL", "http://127.0.0.1:8210/v1")   # 思考（/completions 续写 <think>）；09-04：与人话同一端点（裁定⑬），默认不再分端口
SICK = re.compile(r"指指|的的|是是|了了")
PERSONA_LEAK = re.compile(r"我(每天)?(喝|吃|睡觉|跑步|锻炼|健身)|我的身体")
# ⛔ 2026-08-30：这里原本是**字典**列表，而 `required_answer_fields` 全项目都按对象属性读
#   （`.key` / `.value_source`）。v14 从来没有代码路径读过它 ⇒ 类型错了两个月没人知道；
#   v15 的 `_machine_fields` 一读就 AttributeError，被 L1 循环的 `except Exception` 吞掉
#   ⇒ **L1 桶 0/150 行，静默**。典型的「类型不一致只在新路径上才显形」+「异常被吞」。
#   ⇒ 修法：用真正的 AnswerField；同时把 L1 循环的裸 except 改成记录首个异常（见下）。
from syncopate.core.schemas import AnswerField  # noqa: E402

# MIN_FIELDS 唯一定义在 u_build_v15_multiturn（09-02 收口副本；两处各写一份会漂）
from u_build_v15_multiturn import MIN_FIELDS  # noqa: E402
# ★ v15 契约分支（Chaoyu 08-29 立项，25 号）：**同一份脚本、两个契约**，不复制第二份。
#   副本会漂——R0 已经为「spec 三份副本」付过一次学费（25 §7⑥）。
from syncopate.core.contract import IS_V15  # noqa: E402

# v14.5 的教师物料里**与契约无关**的部分（reply / think 文本）可直接复用，
# 省掉几小时教师生成。⚠️ 但不复用 summary：v15 已废除该字段，且它正是
# 08-29 真人实测发现③「summary 被『X 释义』模板污染」的病灶。
# ⛔ 09-04 Chaoyu 裁定⑭（26 §6）：**v16 不许混进任何旧版本产物**——教师换 27B 后，v14.5/v15 时代的 4B/8B 物料
#   （reply / think / 定义 / 闲聊素材）一律不复用，全部由 27B 重生成。run16 的 64 行 CoT 候选里 ~60 条"1 步思考"
#   就是从 v15_materials.json 的 cot_think 静默复用来的（前任 09-04 核对确认），不是 27B 的贡献。
#   物料表保留空结构（消费者代码不改形状），文件本身不再读。
MATERIALS = {"l2_replies": {}, "l1_replies": {}, "cot_think": {}}
# 难例模板族：来自 v13 考场 triage（_audit/triage/cand_v13r2_e1 的卡死/死格 ≥3 条的族）。裁定⑭ 后不再读那份 v13 产物，
# 只保留推导出的族名常量；⚠️ 欠账：v16 第一遍考场跑完后按 v16 读数重定（26 §7）。
HARD_FAMILIES = ("BUD", "DIA", "FAIL", "RAG", "SCALE")

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
# ⚠️ 考卷 v3 也要进泄漏闸 —— 新增的 REJ 题（业务内越权）一旦漏进训练集，
#   考场就从"测能力"退化成"测记忆"（考卷审计第③条的同族）。
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR
from u_build_v15_multiturn import (DRY, answer_turn, as_multiturn, build_family_rows,  # noqa: E402
                                   real_reply, shape_check)

for _fn in ("context_exam.jsonl", "context_exam_v2.jsonl", "context_v3_exam.jsonl",
            "context_v4_exam.jsonl", "talk_exam.jsonl"):
    for _x in open(f"data/u_route/{_fn}"):
        EXAM_LAST.add(json.loads(_x)["turns"][-1])


_SEED = [0]


OFFLINE = os.environ.get("SYNCOPATE_TEACHER_OFFLINE", "0") == "1"   # 09-04：离线全量建库——教师材料只能来自缓存，缺一条就红（本机验闸用）


async def teach(client, base, prompt, sys_prompt="", max_tokens=200, temp=0.8):
    if OFFLINE:
        raise RuntimeError(f"🔴 离线模式（SYNCOPATE_TEACHER_OFFLINE=1）缓存缺失，需要教师现写：{prompt[:80]!r}")
    if DRY:
        _SEED[0] += 1     # 占位带序号：DRY 下去重/尾部配额闸也能过（裁定⑭后 DRY 不再有任何缓存可命中）
        return f"[DRY 教师待写 #{_SEED[0]}] 这条的数据核对完毕，细节口径我再核一轮。"
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
    if DRY:   # 裁定⑭后没有定义缓存可命中；DRY 不调教师 ⇒ 直接造带 "[DRY" 的占位（正式产物 grep "[DRY" = 0 的红线照旧）
        return {term: [f"{term}[DRY 定义待写 1]：{hint[:30]}", f"{term}[DRY 定义待写 2]：{hint[:30]}"] for term, hint in GLOSSARY.items()}
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
            if len(versions) >= 5:      # 09-04：3→5 版（250 行 L1 / 61 词 ⇒ 3 版必有同一定义服务 ≥3 题面；出厂体检"预设答案"闸）
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
    if DRY:   # 同上：DRY 不调教师，占位素材只演练结构
        return [{"prompt": item["prompt"], "reply": f"[DRY 闲聊待写 #{i}] 这条的数据核对完毕，细节口径我再核一轮。",
                 "summary": "DRY 占位", "style": STYLES[i % 3], "source": item["source"], "turns": 1} for i, item in enumerate(bank[:90])]
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
        if not summ or rep.startswith(summ) or has_oov(summ):
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
            if 20 <= len(rep2) <= 400 and not SICK.search(rep2) and "{" not in rep2 and not has_oov(rep2):
                row.update({"turns": 2, "followup": fu, "reply2": rep2})
        out.append(row)
    assert len(out) >= 85, f"chat 素材不足 {len(out)}"
    return out


L2_TAILS: Counter = Counter()
# ⛔ 08-30 体检实测：19 条以「可能需要进一步优化投放策略。」收尾（配额本该是 10）。
#   两个原因，都得治：① 配额是按 200 行标的，现在全库 900+ 行还在用同一个绝对数；
#   ② **复用物料那条路根本没过这道闸** —— 缓存里的句子直接进库，配额只管新生成的。
#   ⇒ 配额收到 4（全库 922 行 ⇒ ≤0.4%，体检闸 2%），且复用也要过闸。
# ⚠️ 这个计数器现在是**全库共用**（L2 + 压舱人话），不再只属于 L2。
TAIL_CAP = 4


def _tail_ok(rep: str) -> bool:
    return L2_TAILS[rep[-10:]] < TAIL_CAP


def _tail_note(rep: str) -> None:
    L2_TAILS[rep[-10:]] += 1


ANGLES_L2 = ["可附一句简短观察", "顺带说一句这个数是高是低",
             "用口头汇报的口气说", "顺带提示下一步可以看什么"]


async def gen_l2_reply(client, cid, mname, val) -> str:
    if DRY:
        _SEED[0] += 1     # 占位带序号：DRY 走密度/尾部配额闸时量的是结构，不能被同一句占位判死
        return f"{cid} 的{mname}是 {val}。[DRY 教师待写 #{_SEED[0]}]"
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
# ⛔ 09-02 Chaoyu 裁定：**不缩短 CoT**——教师思考 p50 691 字/6 段，没有证据说明多出来的是啰嗦而不是必要推演；
#   "空块占 98%" 的比例问题改由 sft_replay 把空 think 块 mask 掉解决，不靠砍思考长度换行数。
#   这里保留原采样约束（900 token 上限、≤4096 字、不限段数）；W3① 撤回。
# ★ 09-04 Chaoyu 裁定（26 §W4′ S3-diag 读数后）：① **撤中文闸**——CoT 语言不限（27B 全英文思考，语言与目标无关）；
#   ② 教师 CoT 必须完整：token 上限 900 → **2048**（实测英文 p90 326/max 894、中文引子臂 max 1688）；
#   ③ 字数闸撤（4096 字对英文 ≈1000 token，会把长思考按字数砍掉，违背"不缩短 CoT"）；唯一长度闸 = token 上限；
#   ④ 注册截断闸：finish_reason=length 的样本占比 ≤ 3%（gen_cot_v15 末尾断言），红了 = 上限不够、重新注册。
THINK_MAX_TOKENS, THINK_MAX_CHARS, THINK_MAX_SEGS = 2048, 10 ** 6, 10 ** 6
THINK_TRUNC_RATE_MAX = 0.03


async def gen_cot(client, tok, max_rows=100) -> list[dict]:
    from u_teacher_probe import gold_values
    # 难例族 = HARD_FAMILIES 常量（裁定⑭：不读 v13 triage 产物）；在训练集中选同族 case，优先长轨迹
    hard_pref = set(HARD_FAMILIES)
    print(f"[CoT] 难例模板族：{sorted(hard_pref)}")
    df = pd.concat([pd.read_parquet(f"{DEFAULT_SFT_DIR}/train.parquet"),
                    pd.read_parquet(f"{DEFAULT_SFT_DIR}/val.parquet")])
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


# ═══════════ Stage B-v15 · 难例逐步思考 ═══════════


async def gen_cot_v15(client, tokenizer, registry, max_rows=60, target=0.60):
    """v15 难例 CoT：**逐步**接受的 rejection sampling（Chaoyu 08-30 裁定的做法）。

    和 v14.5 的差别，以及为什么必须差：
      v14.5 只在**终答步**插 think —— 因为它要求「整条轨迹逐步全等」，
      6 采样实测 104 案全灭（弃 1 步 45 · 2 步 48）。
      但 v15 里**每个** assistant 轮都有 think 块 ⇒ 只插终答步 = 全库非空占比 1.8%，
      而门槛⑤⒝ 要难例桶 ≥60%。
    ⇒ 改成**逐步接受**：哪一步教师独立选中了 gold 动作，就收哪一步的思考；
      收不到的步留显式空块。不要求整条全等 —— 那条路已经实测走不通。
    ★ 思行一致仍然是**构造性**的：只收「教师自己也选了 gold 动作」的那条思考，
      不是给 gold 编理由（Goodhart 那条线不能越）。
    """
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_split_bundles

    hard_pref = set(HARD_FAMILIES)          # 裁定⑭：族名常量，不读 v13 triage 产物
    # ★ 09-04 裁定⑩：候选池 = 当前切分的 sft 桶（不再读上一版 parquet——v16 之前根本没有 parquet）
    sft_ids = json.load(open(f"{DEFAULT_SPLIT_DIR}/sft_cases.json"))["case_ids"]
    bundles = load_split_bundles(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")   # 三桶隔离①：只装 SFT 桶
    cands = [c for c in sft_ids if c.split("_")[0] in hard_pref and c in bundles]
    cands.sort(key=lambda c: -len(bundles[c].gold.actions))     # 长轨迹优先（多步=思考有用武之地）
    percnt, capped = Counter(), []
    for c in cands:
        fam = c.split("_")[0]
        if percnt[fam] >= 40:
            continue
        percnt[fam] += 1
        capped.append(c)
    rng.shuffle(capped)
    print(f"[CoT-v15] 难例池 {len(capped)}（族 {sorted(hard_pref)}，族内限额 30）")

    async def one_think(ctx: str):
        if OFFLINE:
            raise RuntimeError("🔴 离线模式：CoT 缓存缺失，需要教师采样（先在云上建一次把 v16_cot_rows.json 存进缓存）")
        _SEED[0] += 1
        r = await client.post(f"{T8B}/completions", json={
            # ★ 09-02 W3①（26 §W3）：think 做轻——上限 900→350 token；段数/字数闸在 step_sample
            "model": "t", "prompt": ctx + "<think>\n", "max_tokens": THINK_MAX_TOKENS,
            "seed": _SEED[0], "temperature": 0.7, "top_p": 0.95})
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return ch["text"], ch.get("finish_reason")

    def first_action(text: str):
        # 线格式无关（Qwen3.5 教师吐的是 XML `<function=…>`，旧缓存是 JSON）：走 parsing_v15 的解析器
        from syncopate.core.parsing_v15 import parse_tool_calls
        if "<tool_call>" in text:
            calls, malformed = parse_tool_calls(text)
            return calls[0]["name"] if calls else None
        return "__text__"           # v15 的纯文本终答也是一种"动作"

    # ★ 09-04 先量后动（守则⑤）：run16 采样 892 步只命中 12（1%），过滤链每个丢弃原因都是静默 continue，
    #   日志里分不出是 900 token 没写完 / 英文思考 / 动作不符。⇒ 每个原因计数 + 思考长度分布，末尾打一行；
    #   **不改任何阈值**（THINK_MAX_* 与 900 上限原样）。
    drop = Counter()
    think_lens: list[int] = []

    async def step_sample(ctx: str, want: str, n=8):
        gens = await asyncio.gather(*[one_think(ctx) for _ in range(n)],
                                    return_exceptions=True)
        for g in gens:
            if isinstance(g, Exception):
                drop["exception"] += 1; continue
            g, fin = g
            if "</think>" not in g:
                # 分两类：length = 真截断（进截断闸）；stop = 老师写完想法直接进动作没写 </think>（格式失败，与上限无关）
                drop["no_close_truncated(length)" if fin == "length" else "no_close_eos(格式)"] += 1
                think_lens.append(len(g)); continue
            think, post = g.split("</think>", 1)
            think = think.strip()
            think_lens.append(len(think))
            cjk = len(re.findall(r"[一-鿿]", think)) / max(1, len(think))
            n_seg = len([p for p in re.split(r"\n\s*\n|\n", think) if p.strip()])
            # ★ W3① 画像闸：≤THINK_MAX_CHARS 字、≤THINK_MAX_SEGS 段、中文 ≥0.5；命中判据「首动作名相等」不放宽
            if not think:
                drop["empty_think"] += 1; continue
            if len(think) > THINK_MAX_CHARS:
                drop["too_long_chars"] += 1; continue
            if n_seg > THINK_MAX_SEGS:
                drop["too_many_segs"] += 1; continue
            drop["cjk_below_0.5(只记录不拦)"] += int(cjk < 0.5)     # 裁定：语言不限
            got_action = first_action(post)
            if got_action == want:
                drop["hit"] += 1
                return think
            drop["action_mismatch"] += 1
            drop[f"mismatch:{want}->{got_action}"] += 1
        return None

    out, tried, hit, trimmed = [], 0, 0, 0
    sem = asyncio.Semaphore(3)

    inc = Path("data/u_route/v16_cot_partial.jsonl")
    done = {}
    if inc.exists():
        for line in inc.open():
            r = json.loads(line)
            done[r["case_id"]] = r
        print(f"[CoT-v15] 增量缓存命中 {len(done)} 行（重启不从零开始）")

    async def one_row(cid):
        if f"{cid}_COT15" in done:
            return done[f"{cid}_COT15"]
        async with sem:
            try:
                r = await _row(cid)
            except Exception as e:       # 单行失败不许带走整批（gather 会连坐）
                print(f"  ⚠️ CoT 行失败 {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                return None
            if r:
                with inc.open("a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return r

    async def _row(cid):
        nonlocal tried, hit, trimmed
        b = bundles[cid]
        # ★ CoT 桶取的也是 v13 case ⇒ 同样没有 reply。体检实测：这 20 行**100%**
        #   以同一句兜底话收尾（㉖ 那次只修了压舱桶，漏了这里）。用同一份教师人话。
        if IS_V15:
            _rep = await ballast_replies(client, bundles, [cid])
            b = copy.deepcopy(b)
            if cid in _rep:
                b.gold.final_answer = dict(b.gold.final_answer or {})
                b.gold.final_answer["reply"] = _rep[cid]
            # ★ W3②（26 §W3）：触发显性化——难例行题面加多步诊断问法，让"该想"在题面上可学
            #   （探针实测：族内 65.5% → 显性化后 88.5%，_audit/v15_w3/trigger_probe.json）
            from u_build_v15_cot import explicit_hard_prompt
            b.case.user_message = explicit_hard_prompt(b.case.user_message, cid)
        base = await build_sft_row(b, tokenizer=tokenizer, registry=registry,
                                   index=0, split="train", config=None)
        full = tokenizer.decode(list(base["input_ids"])[:base["total_length"]])
        segs = full.split(ASST)
        n_steps = len(segs) - 1
        if n_steps < 2:
            return None
        # ★ session.report 步**不参与采样**：它是机械序列化（把数字填进 schema），
        #   不需要思考，空块正是对的。而且实测教师（裸 8B，没学过我们的信令契约）
        #   在这一步 0/6 命中 —— 强行采样只会白烧几十分钟。
        eligible = [i for i in range(n_steps)
                    if first_action(segs[i + 1]) != "session.report"]
        want_n = max(1, int(round(n_steps * target)))
        if len(eligible) < want_n:
            return None
        # 裁定⑭：不复用任何旧物料的思考（run16 的"1/10 步有思考"几乎全是这里混进来的 8B 旧料）
        thinking: dict[int, str] = {}
        # ★ 整行的步**一次性并发**采样：每一步的上下文只是前缀，事先就全知道，
        #   步与步之间没有依赖。顺序跑的话一行要几分钟（实测），并发后是一轮的事。
        todo = [si for si in eligible if si not in thinking]
        tried += len(todo)
        res = await asyncio.gather(*[
            step_sample(ASST.join(segs[:si + 1]) + ASST + "\n",
                        first_action(segs[si + 1])) for si in todo])
        for si, th in zip(todo, res):
            if th:
                thinking[si] = th
                hit += 1
        # ⛔ 2026-08-30：这里原本"单行不足 60% 就丢"——**比判据本身还严**。
        #   门槛⑤⒝ 判的是**难例桶的聚合占比**，不是逐行占比 ⇒ 逐行丢会把已经生成好的
        #   思考白扔掉（实测收率掉到 ~25%，会撞 CoT 桶下限 40 行）。
        #   ⇒ 改为：拿到思考的行**都留**，在装配时按覆盖率降序选到**聚合 ≥60%**。
        #     判据没放宽（聚合口径一字未动），放宽的是我自己多加的那一道。
        if not thinking:
            return None
        # ★ 预算内自适应裁剪（门槛⑤⒟：think-on 下 gold 回放截断率 **=0**）
        #   ⛔ 2026-08-30 实案：插了 5–6 段教师推理后，长轨迹撑破 8192 的 response 预算
        #     （truncation_reason="observation"）⇒ 建库整个崩掉。
        #   ⇒ **让数据适配预算，不是反过来**：撑破就丢掉最长的那一段 think 再试，
        #     直到装得下。丢的是覆盖率（聚合口径还有余量），保住的是"零截断"这条硬判据。
        #   ⚠️ 裁剪次数要**报出来**——静默裁剪等于偷偷降覆盖率。
        row = None
        while thinking:
            try:
                row = await build_sft_row(b, tokenizer=tokenizer, registry=registry,
                                          index=95000, split="train", config=None,
                                          thinking=thinking)
                break
            except ValueError as e:
                if "被截断" not in str(e):
                    raise
                longest = max(thinking, key=lambda k: len(thinking[k]))
                del thinking[longest]
                trimmed += 1
        if row is None or not thinking:
            return None
        row["case_id"] = f"{cid}_COT15"
        row["bucket"] = "cot_hard"
        row["sub_axis"] = f"{cid.split('_')[0]}|steps{n_steps}|think{len(thinking)}"
        row["_think"], row["_blocks"] = len(thinking), n_steps
        print(f"  [CoT-v15] 收 {cid}（{len(thinking)}/{n_steps} 步有思考）", flush=True)
        return row

    got = [x for x in await asyncio.gather(*[one_row(c) for c in capped]) if x]
    # ★ **不在这里定选谁** —— 选择要同时满足两个约束（聚合 ≥60% 覆盖率 + token 预算），
    #   而 token 预算要等非 CoT 桶算完才知道 ⇒ 交给 main 的 `_pick_cot()` 统一做。
    #   ⛔ 2026-08-30：先按覆盖率挑 60 行、再让预算去砍，砍出来只剩 19 行（差 1 行撞下限）——
    #     **两个约束分两处做，就会互相打架**。
    for i, r in enumerate(got):
        r["index"] = 95000 + i
    print(f"[CoT-v15] 候选 {len(got)} 行（选择交给 main 的预算环节，两个约束一起解）")
    out = got
    print(f"[CoT-v15] 预算裁剪 {trimmed} 段 think（撑破 response 预算 {__import__('syncopate.train.rollout_budget', fromlist=['MAX_RESPONSE_LENGTH']).MAX_RESPONSE_LENGTH} 的长轨迹）")
    print(f"[CoT-v15] 保留 {len(out)} 行 · 采样步数 {tried} · 命中 {hit}"
          f"（命中率 {hit/max(1,tried):.0%}）")
    # 丢弃原因分布（先量后动：这一行是 09-04 诊断的读数位置）
    _main = {k: v for k, v in drop.items() if not k.startswith("mismatch:")}
    _mm = sorted(((k, v) for k, v in drop.items() if k.startswith("mismatch:")), key=lambda kv: -kv[1])[:8]
    _tl = sorted(think_lens)
    _q = (lambda q: _tl[min(len(_tl) - 1, int(q * len(_tl)))]) if _tl else (lambda q: 0)
    print(f"[CoT-diag] 采样 {sum(_main.values())} 条 ⇒ {dict(_main)} · 动作不符 top: {_mm}")
    print(f"[CoT-diag] think 字数 p50/p90/max = {_q(0.5)}/{_q(0.9)}/{_tl[-1] if _tl else 0}（含未写完的按已吐字数计）")
    _n = sum(v for k, v in _main.items() if k != "exception")
    _trunc = drop.get("no_close_truncated(length)", 0) / max(1, _n)
    print(f"[CoT-trunc] 教师 CoT 截断率 {_trunc:.1%}（{drop.get('no_close_truncated(length)', 0)}/{_n}，闸 ≤{THINK_TRUNC_RATE_MAX:.0%}，上限 {THINK_MAX_TOKENS} tok）")
    assert _n == 0 or _trunc <= THINK_TRUNC_RATE_MAX, (
        f"🔴 教师 CoT 截断率 {_trunc:.1%} > {THINK_TRUNC_RATE_MAX:.0%} —— THINK_MAX_TOKENS={THINK_MAX_TOKENS} 不够，抬上限并重新注册")
    return out


# ═══════════ Stage C · 结构桶（回放）═══════════

async def build_l2_l1(tokenizer, registry, client):
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_split_bundles
    bundles = load_split_bundles(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")   # 三桶隔离①：EVAL/RL 的题不进内存
    # ★ 09-04（v16 首次全量重建暴露）：L2/L1 的历史轮是「上一轮助手的真实终答人话」，所以源 case 只能是
    #   终答型（tool_call / answer）——defer/clarify/reject 收场的 case 没有"人话终答"，历史该是信令自己的话，
    #   那是④族（DEF-F/REJ-F/CLA-F）的事，不进这里。此前 v13 时代靠累积的压舱缓存碰巧盖住，v16 重编号+缓存作废后
    #   FRESH_0125（defer）撞到 real_reply 断言。与下面 `_need` 的过滤保持同一条件。
    _TERMINAL_OK = ("tool_call", "answer")
    q_bundles = [b for b in bundles.values()
                 if b.gold and b.gold.actions
                 and b.gold.actions[0]["tool"] == "campaign.get_metrics"
                 and b.case.context.get("campaign_id")
                 and b.verifier.expected_behavior in _TERMINAL_OK]
    z_bundles = [b for b in bundles.values() if b.gold and not b.gold.actions
                 and b.verifier.expected_behavior in _TERMINAL_OK]
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
    if not DRY:
        # ★ 历史里的上一轮助手内容必须是**真实终答人话**（不同形 #2）：源 case 若不在压舱人话缓存里，
        #   先让教师写好（同一份 ballast_replies 缓存，与冻结桶共用），不许落到占位符
        from u_build_v15_multiturn import BALLAST_REPLIES as _BR
        _need = [b.case_id for b in q_bundles + z_bundles
                 if b.case_id not in _BR and b.verifier.expected_behavior in ("tool_call", "answer")]
        if _need:
            _BR.update(await ballast_replies(client, {b.case_id: b for b in q_bundles + z_bundles}, _need))
    l2_rows, l1_rows, skipped, fallback, reused = [], [], 0, 0, 0
    l1_err: list[str] = []
    # ---- L2 ~200 + 10 val：句式×工具×对象 ----
    obj_seq = (["same"] * 60 + ["switch"] * 25 + ["compare"] * 15)
    tool_seq = (["campaign.get_metrics"] * 70 + ["mmp.get_attribution"] * 30)
    i = 0
    # ★ v15 下 L2 要更多行：换契约后每轮多一个 think 块、每条多一个 report 步，
    #   而**v13 压舱石桶轨迹最长、涨得最多** ⇒ 同一份数据的份额会系统性偏移，
    #   实测 L2 从带内掉到 8.3%（带宽 [10%,17%]）。
    #   ⇒ 抬的是**行数**不是带宽 —— 带宽表达的是「数据追问该拿多少梯度预算」，
    #     这个设计意图与契约无关，不该因为换了承载通道就放宽（守则③）。
    l2_cap = DRY if DRY else (290 if IS_V15 else 210)
    for b in q_bundles:
        if len(l2_rows) >= l2_cap:
            break
        cid = b.case.context["campaign_id"]
        mname, mkey = METRICS[i % len(METRICS)]
        obj = obj_seq[i % 100]
        tool = tool_seq[i % 100]
        pat = rng.choice(SUB_TRAIN)
        i += 1
        b2 = copy.deepcopy(b)
        # ★ 09-02（不同形 #8）：switch/compare 的另一条必须**真实存在于 env**，此前 cid+1 是
        #   题面与 gold 指向两个不同对象的 bug；env 只有一条 campaign 时退回 same
        cid2 = cid
        others = [c for c in (b.env.readonly_tables or {}).get("campaigns", {}) if c != cid]
        if obj in ("switch", "compare"):
            if others:
                cid2 = rng.choice(others)
            else:
                obj = "same"
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
        # ★ 09-02（不同形 #1#2#3#4#6#7）：历史 = 真消息对（上一轮助手 = 真实终答人话），
        #   题面 context = 线上同形（账户 + 在投清单），字段清单 MIN_FIELDS，菜单全量
        b2 = as_multiturn(b, case_id=f"{b.case_id}_MT5", user_message=ask,
                          prior=[(b.case.user_message, answer_turn(real_reply(b)))])
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
            # 裁定⑭：人话一律现调 27B 教师，不复用 v14.5 物料（MATERIALS 已空）
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
                print(f"  ⚠️ L2 回放失败 {b.case_id}: {type(e).__name__}: {str(e)[:160]}")
            continue
        row["bucket"] = "multiturn_l2"
        row["sub_axis"] = f"{tool.split('.')[-1]}|{obj}|{pat[:6]}"
        l2_rows.append(row)

    # ---- L1 ~150 + 10 val：75 概念史 + 75 查询史（与 L2 成对照对）----
    terms = list(GLOSSARY)
    li = 0
    all_forms = SUB_TRAIN + ["那{X}呢", "{X}又是什么", "什么是{X}？", "再说说{X}"]
    # ★ 08-30：L1 行去掉了那一步"把人话写进 session.report"（㉗）之后，每行的监督
    #   token 少了一整轮 ⇒ 份额从 4.6% 掉到 2.8%，撞穿 [3%,9%] 的下沿。
    #   ⛔ 处理方式是**补行数**，不是放宽带宽：带宽表达的是"L1 该占多少教学份量"，
    #     那个意图没变；变的是"一行值多少 token"。按旧口径标定的**行数**才是失效的那个数。
    while len(l1_rows) < (DRY if DRY else 250) and li < 1500:
        li += 1
        b_src = rng.choice(z_bundles) if li % 2 == 0 else rng.choice(q_bundles)
        kind = "concept_hist" if li % 2 == 0 else "query_hist"
        a, t2 = rng.sample(terms, 2)
        pat = rng.choice(all_forms)
        ask = pat.replace("{X}", t2)
        if ask in EXAM_LAST:
            continue                        # 撞被判句直接换对（li 循环量足够）
        if kind == "concept_hist":
            prior = [(f"{a}是什么意思？", answer_turn(rng.choice(DEFS[a])))]
        else:
            if b_src.verifier.expected_behavior not in ("tool_call", "answer"):
                continue
            prior = [(b_src.case.user_message, answer_turn(real_reply(b_src)))]
        d = rng.choice(DEFS[t2])
        b2 = as_multiturn(b_src, case_id=f"L1F_{li:04d}", user_message=ask, prior=prior,
                          gold_actions=[], final_answer={"summary": f"{t2} 释义", "reply": d},
                          behavior="answer")
        try:
            row = await replay(b2, 92000 + len(l1_rows))
        except Exception as e:
            skipped += 1
            if not l1_err:
                l1_err.append(f"{type(e).__name__}: {str(e)[:200]}")
            continue
        row["bucket"] = "multiturn_l1"
        row["sub_axis"] = f"{kind}|{t2}|{pat[:6]}"
        l1_rows.append(row)
    print(f"[L2] {len(l2_rows)}（回放丢 {skipped}·读数兜底 {fallback}·物料复用 {reused}）"
          f" [L1] {len(l1_rows)}")
    if l1_err:
        print(f"  ⚠️ L1 首个回放异常：{l1_err[0]}")
    return l2_rows, l1_rows


def _chat_prompt(tokenizer, user: str, prior: list[tuple[str, str]]) -> str:
    """chat 行的 prompt：与 rollout_loop.build_messages / decider._messages **同一组函数**
    （load_system_prompt 含 v15 尾段 · step_user 纯日期 · 线上同形 context · 历史消息对 · 全量菜单）。
    ⛔ 09-02 之前走 probe_opd_divergence.render_prompt_text：v14 尾段的 system.txt + ACC_DEMO 假 context
       + summary 字段清单 + 历史折进题面 —— chat 行与线上四处不同形。"""
    from syncopate.core.prior_turns import render_prior_messages
    from syncopate.domains.adcampaign import build_domain
    from syncopate.prompts import load_system_prompt, render_prompt
    from syncopate.core.demo_context import demo_context as _demo_context
    from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS
    msgs = [{"role": "system", "content": load_system_prompt()}]
    msgs += render_prior_messages([{"user_message": u, "result": {"text": a}} for u, a in prior], tokenizer)
    msgs.append({"role": "user", "content": render_prompt("step_user.txt", {
        "reference_now": "2026-08-20", "context": _demo_context(), "user_message": user,
        "answer_fields": []})})
    return tokenizer.apply_chat_template(msgs, tools=build_domain().registry.menu(None),
                                         add_generation_prompt=True, tokenize=False, **CHAT_TEMPLATE_KWARGS)


def build_chat_rows(tokenizer, chat_mat):
    rows = []
    for i, c in enumerate(chat_mat):
        if c["turns"] == 1:
            user, prior = c["prompt"], []
            reply = c["reply"]
        else:
            user, prior = c["followup"], [(c["prompt"], c["reply"])]
            reply = c["reply2"]
        prompt = _chat_prompt(tokenizer, user, prior)
        if IS_V15:
            # v15：闲聊没有机器可核字段 ⇒ 不发 session.report，终答就是一句人话。
            # ★ 但 think 段必须显式写出来（门槛⑤⒜=100%）——闲聊属"简单题"，
            #   填**空块**正是在教「这种题不用想」（N3 按需思考的负样本那一半）。
            from syncopate.pipeline.sft_replay import EMPTY_THINK, EMPTY_THINK_RESP, think_opener_in_prompt
            # 09-04 run22：模板已在生成提示里写了 "<think>\n" 时，这里只能写收尾（否则双开头）——与 sft_replay.attach_think 同规则
            gtext = (EMPTY_THINK_RESP if think_opener_in_prompt(tokenizer) else EMPTY_THINK) + reply + "<|im_end|>"
        else:
            gold = {"behavior": "answer",
                    "answer": {"summary": c["summary"], "reply": reply}}
            gtext = json.dumps(gold, ensure_ascii=False) + "<|im_end|>"
        ids_p = tokenizer(prompt, add_special_tokens=False).input_ids
        ids_g = tokenizer(gtext, add_special_tokens=False).input_ids
        mask = [0] * len(ids_p) + [1] * len(ids_g)
        if IS_V15:
            # ★ 09-02（Chaoyu 在画廊里抓到的）：闲聊行不走 build_sft_sample，空 think 块此前全 1 有梯度。
            #   用同一份 _mask_empty_think（不另抄一份规则）。
            from syncopate.pipeline.sft_replay import _mask_empty_think
            _mask_empty_think(tokenizer, ids_p + ids_g, mask, start=len(ids_p))
        rows.append({"case_id": f"CHAT5_{i:04d}", "input_ids": ids_p + ids_g,
                     "loss_mask": mask,
                     "prompt_length": len(ids_p), "total_length": len(ids_p) + len(ids_g),
                     "supervised_tokens": sum(mask), "split": "train",
                     "index": 93000 + i, "signal_class": "graded",
                     "behavior": "answer", "bucket": "chat_shell",
                     "sub_axis": f"{c['style']}|{c['source']}|t{c['turns']}"})
    return rows


# ═══════════ 门禁 ═══════════

_TC_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_TH_RE = re.compile(r"<think>.*?</think>", re.S)


def _final_text_v15(txt: str) -> str:
    """v15 的"人话终答" = **最后一个 assistant 轮**里剥掉 think/tool_call 之后的文本。

    ⛔ 2026-08-30：初版没剥 chat 模板标记，于是"最高频收尾 10 字"量到的是
      `>assistant`（`<|im_start|>assistant` 的尾巴）—— L1 密度闸报 33% 超标，
      **量的是模板不是话术**。仪器错，不是数据错（「归因之前先查输入」同族）。
    """
    tail = txt.rsplit("<|im_start|>assistant", 1)[-1]
    body = _TC_RE.sub("", _TH_RE.sub("", tail))
    for mark in ("<|im_end|>", "<|im_start|>", "user\n", "assistant\n"):
        body = body.replace(mark, "")
    return body.strip()


def density_gate(rows, tokenizer, name):
    reps, reports = [], []
    for r in rows:
        txt = tokenizer.decode(list(r["input_ids"])[r["prompt_length"]:r["total_length"]])
        if IS_V15:
            t = _final_text_v15(txt)
            if t:
                reps.append(t)
            # ★ v14.6 唯一在册的修正项：密度闸此前**只查 reply 没查 summary**，
            #   于是「X 释义」模板外溢成万能 summary 没被闸住（08-29 真人实测发现③）。
            #   v15 里 summary 没了，接它班的是 session.report ⇒ 一起查。
            for m in re.finditer(r'"name":\s*"session\.report",\s*"arguments":\s*(\{.*?\})',
                                 txt, re.S):
                reports.append(m.group(1))
        else:
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
    # 比例闸只在样本 ≥20 时判（DRY 每桶 6 行 ⇒ 1/6 就是 17%，量的是样本量不是复读）；正式建库 L2/L1/chat 都 ≥80
    assert len(reps) < 20 or top[1] / len(reps) <= 0.10, f"🔴 {name} 话术密度超标"
    assert sick == 0, f"🔴 {name} 病句 {sick} 条"
    if reports:
        rtails = Counter(reports)
        rtop = rtails.most_common(1)[0]
        print(f"  [密度:{name}/report] 最高频参数组 {rtop[1]}/{len(reports)}="
              f"{rtop[1]/len(reports):.0%}")
        assert rtop[1] / len(reports) <= 0.10, (
            f"🔴 {name} 的 session.report 参数模板化超标（{rtop[0][:80]}）")


# ── 压舱桶的终答人话：**教师生成**，不是模板拼接（`25 §7㉙`）──────────────────
#
# ⛔ 三次同一种病，第三次才看清（08-30）：
#   ㉖ v13 压舱 419 行的 gold **没有 reply**（v14 时代终答是 JSON 壳，机器字段就是答案），
#      于是 v15 拼壳时用了一句常量兜底 ⇒ 41.8% 的行以同一句空话收尾。
#   ㉖的修法（把字段渲染成中文句）只是把常量换成**模板** ⇒ 体检器实测：
#      rag_policy 桶 30.3% 同句式、reasoning 21.7%、32 个答案各自服务 ≥3 个不同题面。
#   ⇒ 一般化：**旧契约里不存在的字段，必须有真实来源；"拼一个"和"常量"是同一类错误。**
#     真实来源 = 教师按题面 + 事实清单说一句人话（L2/chat 桶一直是这么做的）。
#
# 三道过滤缺一不可：① 长度/病句 ② 句式去重（抹掉数字后不许撞） ③ **禁编数**
#   —— ③ 是最容易漏的：教师顺手编一个没出现过的数字，就等于在教模型幻觉。
_BALLAST_CACHE = Path("data/u_route/v16_ballast_replies.json")
_BALLAST_FALLBACK = [0]   # 裁定⑭：缓存名带数据版本，旧名读不到
ANGLES_BALLAST = [
    "先说结论再补一句依据", "从用户关心的那个点切入", "口语一点，像同事口头汇报",
    "先点出关键数字再说结论", "简短直接，一句话说完", "带一句下一步建议",
    "把前提交代清楚再给结论", "语气平实，不要套话",
]
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# ⚠️ 枚举值要**先翻译成行话再喂教师** —— 让它自己猜 snake_case 就会猜错，
#   而猜错的是**业务事实**：实测 `only_4_creatives` 被译成"只有 4 个创作者"
#   （其实是 4 条素材）。错的人话会当成 gold 训进去，比说得干巴巴贵得多。
_TERM_CN = {
    "real_person": "真人出镜", "before_after": "前后对比开场", "dark_palette": "暗色调",
    "fast_cut": "快切节奏", "ugc_style": "UGC 风格",
    "cpi_spike": "CPI 冲高", "roas_drop": "ROAS 下滑", "creative_fatigue": "素材疲劳",
    "no_expansion": "不扩量", "escalated": "上报审批", "approved": "已通过",
    "pending": "审核中", "rejected": "已驳回",
    "policy_not_found": "没有查到对应政策条款",
    # 09-04 run23：DIA 题 8/8 兜底——枚举没译名 ⇒ 教师写"点击率下滑"被「事实里没有的指标名」拦（事实里只有 ctr_decline）
    "ctr_decline": "点击率下滑", "rebalance_budget": "重新分配预算", "refresh_creative": "换一批新素材",
    "narrow_targeting": "收窄定向", "rotate_creative": "轮换素材",
}
_ONLY_N = re.compile(r"only_(\d+)_creatives")


# 教师最容易顺手编的东西：一个**事实里没有的指标名**。
# （实测："提升 0.2667" 被写成"转化率提升了 0.2667" —— 那个 lift 是对 d7 ROAS 的。）
_METRIC_WORDS = ["转化率", "点击率", "留存", "安装量", "曝光", "客单价", "利润",
                 "CTR", "IPM", "CPI", "ROAS", "ARPU", "LTV"]
# 叠字病句：中文有合法叠词（看看/试试），所以只查**虚词**叠字。
_DUP = re.compile(r"([的了是和与在就都也很更值])\1")


def _facts_line(fa: dict) -> str:
    """给教师看的事实清单（**不是**给模型看的答案）。"""
    from syncopate.pipeline.sft_replay import _CONCLUSION_CN, _FIELD_CN

    out = []
    for k, v in (fa or {}).items():
        if k in ("summary", "reply") or v in (None, ""):
            continue
        name = _FIELD_CN.get(k, k)
        if k == "conclusion":
            out.append(f"{name}={_CONCLUSION_CN.get(str(v), str(v))}")
        elif isinstance(v, str) and _ONLY_N.fullmatch(v):
            out.append(f"{name}=只有 {_ONLY_N.fullmatch(v).group(1)} 条素材，样本不足")
        elif isinstance(v, str) and v in _TERM_CN:
            out.append(f"{name}={_TERM_CN[v]}")
        elif isinstance(v, (list, tuple)):
            out.append(f"{name}={'、'.join(map(str, v))}" if v else "")
        else:
            out.append(f"{name}={v}")
    return "；".join(x for x in out if x)


def _no_invented_numbers(text: str, allowed: str) -> bool:
    """文本里出现的每个数字都必须在事实清单或题面里有 —— 否则就是教师在编。"""
    return all(n in allowed for n in _NUM_RE.findall(text))


def _no_invented_metrics(text: str, allowed: str) -> bool:
    """不许出现事实/题面里没有的指标名 —— 编一个指标名 = 把结论安到别的指标上。"""
    return all(w in allowed for w in _METRIC_WORDS if w in text)


async def ballast_replies(client, bundles, case_ids: list[str]) -> dict[str, str]:
    """case_id → 一句自然中文终答。带缓存，断了不从零开始。"""
    cache = json.loads(_BALLAST_CACHE.read_text()) if _BALLAST_CACHE.exists() else {}
    # ★ 09-04：缓存里的机器字段兜底句每轮自动剔除重试（不靠人手改缓存）；闸按**全库**兜底占比算
    from syncopate.pipeline.sft_replay import _prose_from_fields as _pff
    _stale_fb = [k for k, v in cache.items() if bundles.get(k) is not None
                 and v == _pff(dict(bundles[k].gold.final_answer or {}))]
    for k in _stale_fb:
        cache.pop(k)
    if _stale_fb:
        print(f"[压舱-兜底] 缓存里 {len(_stale_fb)} 条兜底句剔除重试", flush=True)
    used = {re.sub(r"\d+(\.\d+)?", "§", v) for v in cache.values()}
    # ⚠️ 只有"要给结论"的行才用得上人话；defer/clarify/reject 的终答是信令，
    #   给它们生成 = 白烧教师额度，还会往缓存里塞永远用不到的条目。
    case_ids = [c for c in case_ids
                if bundles.get(c) is not None
                and bundles[c].verifier.expected_behavior in ("tool_call", "answer")]
    todo = [c for c in case_ids if c not in cache]
    if todo:
        print(f"[压舱人话] 教师生成 {len(todo)} 条（缓存已有 {len(cache)}）", flush=True)
    for i, cid in enumerate(todo):
        b = bundles[cid]
        fa = dict(b.gold.final_answer or {})
        facts = _facts_line(fa)
        ask = str(b.case.user_message or "")[:300]
        allowed = facts + " " + ask
        got = ""
        # ★ 09-04 run22 体检根因：5 次尝试 + 「句式不许与已有重复」在同事实族（51 条 FRESH 同为 partial/4 天）上把
        #   措辞用尽 ⇒ 89/1021 条落到机器字段兜底句（"can_decide否、数据成熟度 partial…"）⇒ 六族 30% 同句、38 个预设答案。
        #   改：10 次尝试；后 5 次放开"句式不重复"（少量复读 ≪ 兜底句同句）；兜底占比进闸（≤2%）。
        for k in range(10):
            cand = clean_reply(await teach(
                client, T4B,
                f"用户问：{ask}\n\n你已经查完了，事实是：{facts or '（无额外数据）'}\n\n"
                f"用一到两句自然中文把结论说给用户听，{ANGLES_BALLAST[(i + k) % len(ANGLES_BALLAST)]}。"
                f"⚠️ 只能用上面给的数字，不许出现别的数字；不要列清单、不要 JSON、不要写『结论如下』。",
                temp=0.95 if k < 5 else 1.05, max_tokens=110))
            key = re.sub(r"\d+(\.\d+)?", "§", cand)
            if (12 <= len(cand) <= 160 and not SICK.search(cand)
                    and not _DUP.search(cand) and _tail_ok(cand)
                    and "{" not in cand and (key not in used or k >= 5)
                    and _no_invented_numbers(cand, allowed)
                    and _no_invented_metrics(cand, allowed)):
                got = cand
                break
        if not got:                       # 兜底：宁可留模板，也要**记账**（见下方闸）
            from syncopate.pipeline.sft_replay import _prose_from_fields
            got = _prose_from_fields(fa)
            _BALLAST_FALLBACK[0] += 1
            print(f"  [压舱-兜底] {cid} 事实={facts[:90]} 问={ask[:40]!r}", flush=True)
        cache[cid] = got
        _tail_note(got)
        used.add(re.sub(r"\d+(\.\d+)?", "§", got))
        if (i + 1) % 25 == 0:
            _BALLAST_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            print(f"  [压舱人话] {i + 1}/{len(todo)}", flush=True)
    if todo:
        _BALLAST_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        _fb = _BALLAST_FALLBACK[0] / max(1, len(cache))
        print(f"[压舱-兜底] 本轮 {_BALLAST_FALLBACK[0]} 条兜底 / 全库 {len(cache)} = {_fb:.1%}（闸 ≤2%）", flush=True)
        assert _fb <= 0.02, f"🔴 压舱人话兜底 {_fb:.1%} > 2% —— 教师写不出人话的题太多，查事实清单/过滤器"
    return cache


async def _replay_frozen(tokenizer, registry, parquet_path: str, base_index: int,
                         client=None):
    """把冻结桶的 case **按当前契约重放**成行（v15 用）。

    ★ 冻结的是**语义**不是字节：换壳之后逐字节冻结在物理上不可能，
      所以判据改成「同一批 case、同样的工具动作序、同样的机器字段」
      —— 全量 419 条已由 scripts/v15_r2_migrate.py 证过（25 §R2①）。
    """
    from syncopate.pipeline.build_dataset import build_sft_row
    from syncopate.pipeline.split import load_split_bundles
    df = pd.read_parquet(parquet_path)
    bundles = load_split_bundles(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")   # 三桶隔离①：只装 SFT 桶
    # ★ v15：这批 case 的 gold **没有 reply**（v14 终答是 JSON 壳）⇒ 终答人话要有真实来源。
    #   不给 client 就会落回模板兜底 —— 那正是 ㉖/㉙ 的病根，所以这里**要求**给。
    replies = {}
    if IS_V15 and DRY:
        # DRY 不调教师：占位人话（画廊/产物里 "[DRY" 是正式建库的红线判据，不会漏进真产物）
        replies = {str(c): f"[DRY 压舱人话:{c}]" for c in df.case_id}
    elif IS_V15:
        # ⚠️ 调用点可能在 `async with httpx.AsyncClient(...)` 块**之外**（实测崩过一次：
        #   "Cannot send a request, as the client has been closed"）⇒ 客户端关了就自己开一个。
        #   ⛔ 但**绝不许**因为拿不到客户端就落回模板兜底 —— 那正是 ㉖/㉙ 的病根。
        ids = list(map(str, df.case_id))
        if client is None or client.is_closed:
            async with httpx.AsyncClient(timeout=180) as _c:
                replies = await ballast_replies(_c, bundles, ids)
        else:
            replies = await ballast_replies(client, bundles, ids)
    rows = []
    for i, cid in enumerate(df.case_id):
        b = bundles.get(cid)
        if b is None or not b.gold:
            raise AssertionError(f"🔴 冻结桶的 case 找不到 bundle：{cid}")
        if str(cid) in replies:
            b = copy.deepcopy(b)
            b.gold.final_answer = dict(b.gold.final_answer or {})
            b.gold.final_answer["reply"] = replies[str(cid)]
        row = await build_sft_row(b, tokenizer=tokenizer, registry=registry,
                                  index=base_index + i,
                                  split=str(df.iloc[i]["split"]), config=None)
        rows.append(row)
    print(f"[冻结桶] {parquet_path} → v15 重放 {len(rows)} 行")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["case_id", "split", "supervised_tokens", "total_length"])


async def main() -> int:
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    global DEFS
    # 学生权重在（Modal）用学生自己的分词器；本机 DRY 无权重 ⇒ 同词表的 Qwen3.5-0.8B（model_paths.TEST_TOKENIZER）
    _tok_path = STUDENT_MODEL if Path(STUDENT_MODEL, "tokenizer.json").exists() else TEST_TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(_tok_path)
    if DRY:
        print(f"[DRY] 结构演练模式：每桶 {DRY} 行、不调教师、不写缓存、不落 parquet（tokenizer={_tok_path}）")
    registry = build_domain().registry
    registry.latency_scale = 0.0
    bank = [json.loads(x) for x in open("data/u_route/chat_bank_v2.jsonl")]

    # 裁定⑭：v16 的定义/闲聊素材由 27B 教师重生成，缓存名带版本；v14.5 分支保留旧名（legacy）
    cache_d = Path("data/u_route/v16_defs.json" if IS_V15 else "data/u_route/v145_defs.json")
    cache_c = Path("data/u_route/v16_chat_mat.json" if IS_V15 else "data/u_route/v145_chat_mat.json")
    async with httpx.AsyncClient(timeout=180) as client:
        if cache_d.exists():
            DEFS = json.load(open(cache_d))
            print(f"[A1] 定义缓存命中（{len(DEFS)} 词）")
        else:
            print("[A1] 定义改写 61×3 …", flush=True)
            DEFS = await gen_defs(client)
            if not DRY:                       # DRY 占位不许落成缓存（09-04：本机 DRY 曾把 "[DRY 定义待写" 写进 v16_defs.json）
                json.dump(DEFS, open(cache_d, "w"), ensure_ascii=False)
        if cache_c.exists():
            chat_mat = json.load(open(cache_c))
            print(f"[A3] chat 缓存命中（{len(chat_mat)} 条）")
        else:
            print("[A3] chat 素材 …", flush=True)
            chat_mat = await gen_chat(client, bank)
            if not DRY:
                json.dump(chat_mat, open(cache_c, "w"), ensure_ascii=False)
        cache_l = Path("data/u_route/v16_l2l1_rows.json" if IS_V15
                       else "data/u_route/v145_l2l1_rows.json")
        if DRY:
            # ⚠️ 08-31 前的缓存是折叠文本形状，W2 之后**必须重建**；DRY 只演练构建路径
            l2, l1 = await build_l2_l1(tokenizer, registry, client)
        elif cache_l.exists():
            _c = json.load(open(cache_l))
            l2, l1 = _c["l2"], _c["l1"]
            print(f"[C] L2/L1 缓存命中（{len(l2)}/{len(l1)}）")
        else:
            print("[C] L2/L1 回放构建 …", flush=True)
            l2, l1 = await build_l2_l1(tokenizer, registry, client)
            json.dump({"l2": l2, "l1": l1}, open(cache_l, "w"))
        # ★ 09-02 W2⑦：六族第一波训练行（DEF-F/REJ-F/CLA-F/L2-x/WIN，各成对）
        from syncopate.pipeline.build_dataset import build_sft_row as _bsr
        from syncopate.pipeline.split import load_bundles as _lb
        from syncopate.pipeline.split import load_split_bundles as _lsb
        _bundles = _lsb(Path(DEFAULT_BATCH_DIR), Path(DEFAULT_SPLIT_DIR), "sft")   # 三桶隔离①（09-04 事故：六族行 48 条底题来自 EVAL）

        async def _replay_fam(b, idx):
            return await _bsr(b, tokenizer=tokenizer, registry=registry, index=idx, split="train", config=None)
        cache_f = Path("data/u_route/v16_fam_rows.json")
        if not DRY and cache_f.exists():
            fam = json.load(open(cache_f))
            print(f"[F] 家族行缓存命中（{len(fam)}）")
        else:
            async def _gen(cid, mname, val):
                return await gen_l2_reply(client, cid, mname, val)
            fam = await build_family_rows(tokenizer, registry, _bundles, DEFS, _replay_fam, _gen)
            if not DRY:
                json.dump(fam, open(cache_f, "w"))
        print(f"[F] 家族行 {len(fam)}：{dict(Counter(r['bucket'] for r in fam))}")
        cache_cot = Path("data/u_route/v16_cot_rows.json" if IS_V15
                         else "data/u_route/v145_cot_rows.json")
        if DRY:
            cot = []      # 旧缓存是 W2 之前的形状（裁剪菜单/ISO 时间/空块有梯度），DRY 不许混进画廊；W4 重采样
            print("[B] DRY：CoT 不用旧缓存（需教师重采样）")
        elif cache_cot.exists():
            cot = json.load(open(cache_cot))
            print(f"[B] CoT 缓存命中（{len(cot)} 行）")
        else:
            print("[B] CoT 难例（27B 教师逐步拒绝采样）…", flush=True)
            cot = await (gen_cot_v15(client, tokenizer, registry, max_rows=60) if IS_V15
                         else gen_cot(client, tokenizer, max_rows=60))
            json.dump(cot, open(cache_cot, "w"))

    # ★ 桶下限闸放在**这里**（用数据的地方），不放在生产者内部。
    #   ⛔ 2026-08-30 实案：闸写在 build_l2_l1 里，结果上一轮把 L1=0 的坏结果**写进了缓存**，
    #     下一轮缓存一命中就绕过了闸 —— 判据必须长在「实际会被用的那份数据」上。
    if not DRY:
        assert len(l1) >= 150, f"🔴 L1 桶下限闸：仅 {len(l1)} 行（要 ≥150）—— 缓存也算数"
        assert len(l2) >= (280 if IS_V15 else 200), f"🔴 L2 桶下限闸：仅 {len(l2)} 行"

    # held-out val 切分（每桶尾部拿走）
    _l2_train = 280 if IS_V15 else 200
    l2, l2v = l2[:_l2_train], l2[_l2_train:_l2_train + 10]
    # ★ 08-30：L1 训练条数跟着"一行值多少 token"走 —— 去掉 report 步之后每行变短，
    #   150 行只够 2.7%（带宽下沿 3%）。⛔ 这里是**第二处**按旧口径写死的行数：
    #   上面把生成上限从 160 提到 250，如果这里不跟着改，多造的 100 行会被直接切掉
    #   （而且不报错 —— 份额闸只会说"还是 2.7%"，不会说"你多造的被扔了"）。
    _l1_train = 240 if IS_V15 else 150
    l1, l1v = l1[:_l1_train], l1[_l1_train:_l1_train + 10]
    chat_rows = build_chat_rows(tokenizer, chat_mat)
    chat_rows, chatv = chat_rows[:80], chat_rows[80:90]

    # ★ 09-04 裁定⑩（v16 全部重来）：压舱桶不再来自任何旧 parquet——直接取 **当前切分的 sft_cases**，
    #   val 留出 = 每 6 条取 1（与 `syncopate data build --val-every 6` 同口径），按当前契约重放成行。
    #   （v13 时代"沿用旧 parquet 的 419 行再迁移"那条路随裁定⑩作废。）
    _sft_ids = json.load(open(f"{DEFAULT_SPLIT_DIR}/sft_cases.json"))["case_ids"]
    _val_ids = _sft_ids[5::6]
    _train_ids = [c for c in _sft_ids if c not in set(_val_ids)]
    if DRY:
        _train_ids, _val_ids = _train_ids[:DRY], _val_ids[:max(1, DRY // 3)]
    _fz = Path("_audit/v16"); _fz.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": _train_ids, "split": ["train"] * len(_train_ids)}).to_parquet(_fz / "_frozen_train.parquet")
    pd.DataFrame({"case_id": _val_ids, "split": ["val"] * len(_val_ids)}).to_parquet(_fz / "_frozen_val.parquet")
    if IS_V15:
        t13 = await _replay_frozen(tokenizer, registry, str(_fz / "_frozen_train.parquet"), 0, client=client)
        v13v = await _replay_frozen(tokenizer, registry, str(_fz / "_frozen_val.parquet"), 80000, client=client)
    else:
        t13 = pd.read_parquet(f"{DEFAULT_SFT_DIR}/train.parquet")
        v13v = pd.read_parquet(f"{DEFAULT_SFT_DIR}/val.parquet")
    # ★ CoT 预算截断必须发生在装配之前（第 5 次发射的教训：截断放在份额计算之后
    #   = 截了个寂寞——train 里还是全量、闸读的还是旧份额）
    non_cot_tok = int(t13.supervised_tokens.sum()) + \
        sum(r["supervised_tokens"] for r in l2 + l1 + chat_rows + fam)
    # ★ 09-04 全量自检：这里原来写死 0.19/0.81（CoT 带宽 ≤20% 时代的数）——Chaoyu 08-31 裁定④ 已把上沿抬到 30%
    #   （不顶满），份额闸也早是 (0.05, 0.30)，预算公式没跟着改 ⇒ 又一个"两处各写各的"。目标取 28%（上沿 30% 留 2pp）。
    COT_SHARE_TARGET = 0.28
    budget = int(non_cot_tok * COT_SHARE_TARGET / (1 - COT_SHARE_TARGET))
    acc, kept = 0, []
    if IS_V15:
        # ★ 两个约束一起解：① token 预算（份额带宽 5–20% 的直接来源）
        #                    ② 聚合非空 think ≥60%（门槛⑤⒝-难）
        #   做法：按 token 升序（同样预算装更多 case = 最大化难例覆盖面），
        #   但一行只有在**不把聚合压到 60% 以下**时才收；被跳过的用高覆盖率行回填。
        # ⛔ 2026-08-30：初版用「按 token 升序、压破 60% 就跳过」的贪心 —— 顺序依赖的
        #   次优解，便宜的行往往覆盖率低、一路被跳过，只装进 12 行。
        # ⇒ 改成两步：① 先**最大化行数**装满预算（覆盖面是行数下限想要的东西）
        #             ② 再用「高覆盖率换低覆盖率」的交换把聚合拉回 ≥60%（不减行数）
        #   surplus = 非空块数 − 0.6×总块数；聚合 ≥60% ⟺ Σsurplus ≥ 0。
        # 先保可行（Σsurplus ≥ 0 ⟺ 聚合 ≥60%），再在可行前提下最大化行数：
        #   扫 kp = 取多少条"高 surplus 性价比"行作为底子，剩下预算塞最便宜的行，
        #   只要不破可行性就收。kp 全扫一遍取行数最多的那个 —— 这就是可行上界。
        sur = lambda r: r.get("_think", 0) - 0.60 * r.get("_blocks", 0)
        pos = sorted([r for r in cot if sur(r) > 0],
                     key=lambda r: -sur(r) / max(1, r["supervised_tokens"]))
        neg = sorted([r for r in cot if sur(r) <= 0], key=lambda r: r["supervised_tokens"])
        sel = []
        for kp in range(len(pos) + 1):
            base = pos[:kp]
            tok = sum(r["supervised_tokens"] for r in base)
            if tok > budget:
                break
            su, cur = sum(map(sur, base)), list(base)
            for r in neg:
                if tok + r["supervised_tokens"] <= budget and su + sur(r) >= 0:
                    tok += r["supervised_tokens"]; su += sur(r); cur.append(r)
            if len(cur) > len(sel):
                sel = cur
        acc = sum(r["supervised_tokens"] for r in sel)
        swaps = 0
        ne = sum(r.get("_think", 0) for r in sel)
        bl = sum(r.get("_blocks", 0) for r in sel)
        kept = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sel]
        print(f"[CoT-v15] 预算 {budget} 内选中 {len(kept)} 行（可行上界搜索）· "
              f"聚合非空 think {ne}/{bl} = {ne/max(1,bl):.1%}（门槛 ≥60%）")
        assert bl == 0 or ne / bl >= 0.60, (
            f"🔴 CoT 聚合覆盖率 {ne/max(1,bl):.1%} < 60% —— 预算与覆盖率无法同时满足，"
            f"停下来报 Chaoyu，不许自己放宽")
    else:
        for r in cot:
            if acc + r["supervised_tokens"] > budget:
                break
            acc += r["supervised_tokens"]
            kept.append(r)
    if len(kept) < len(cot):
        print(f"[CoT] 预算截断 {len(cot)}→{len(kept)}（sup-tok 预算 {budget}）")
    cot = kept
    # ⚠️ 行数下限按契约分家（Chaoyu 08-30 裁定）：v15 的 CoT 行带 4–6 段教师推理，
    #   监督 token 中位 2160（v14 时代只有几百）⇒「≥40 行」与「token 带宽 ≤20%」
    #   在 v15 下**数学上不可兼**（40 行 = 28.2% 份额）。
    #   裁定：**保 token 带宽（它是"梯度预算"的直接表达，且 24 §P2 明写配比口径=监督
    #   token 不是行数），行数下限按新的行重量重标定 40→20**。
    #   代价如实记：难例覆盖面从 40 个 case 降到 20 个，但每个 case 的思考密度高了 4–6 倍。
    #   ⚠️ 二次修正（Chaoyu 08-30）：20 也不可行 —— 三条判据（token 带宽 ≤20% ·
    #     行数下限 · 难例桶覆盖率 ≥60%）**两两相容、三条一起不可行**。
    #     穷举出的可行上界 = **19 行**（预算 35366 内，其中高覆盖行 9，surplus 恰好 0）。
    #     ⇒ 行数下限取实测上界 19。三条里只有行数下限是"拍的"，另外两条各有来源
    #       （token 带宽=梯度预算 · 覆盖率=N3 按需思考）。
    _cot_floor = 19 if IS_V15 else 40
    if not DRY:
        assert len(cot) >= _cot_floor, f"🔴 CoT 桶下限闸：仅 {len(cot)} 行（要 ≥{_cot_floor}）"
    new_rows = l2 + l1 + chat_rows + fam + cot
    train = pd.concat([t13, pd.DataFrame(new_rows)], ignore_index=True)
    valrows = l2v + l1v + chatv
    for r in valrows:
        r["split"] = "val"
    val = pd.concat([v13v, pd.DataFrame(valrows)], ignore_index=True)

    # ── 门禁 ──
    # ★ 09-04 run17 实案：这里原来写死 419（v13 冻结桶行数）——v16 按"每 6 取 1"派生是 420 ⇒ 按旧单位标定的阈值又一次不报错地失效。
    #   判据改成「两个东西应当相同」：重放行数 == 冻结清单行数（派生，不写数）。
    assert len(t13) == len(_train_ids), f"🔴 冻结校验失败：重放 {len(t13)} 行 != 冻结清单 {len(_train_ids)} 行"   # DRY 也核（09-04：419 那次 DRY 跳过了）
    tok_by = {"v13": int(t13.supervised_tokens.sum()),
              "l2": sum(r["supervised_tokens"] for r in l2),
              "l1": sum(r["supervised_tokens"] for r in l1),
              "chat": sum(r["supervised_tokens"] for r in chat_rows),
              "fam": sum(r["supervised_tokens"] for r in fam),
              "cot": sum(r["supervised_tokens"] for r in cot)}
    total = sum(tok_by.values())
    share = {k: v / total for k, v in tok_by.items()}
    print("sup-tok 份额:", {k: f"{v:.1%}" for k, v in share.items()})
    # ★ 09-02：新增 fam 桶后带宽重定（26 §W2⑦「份额闸按监督 token 重新定带宽」）：
    #   v13 压舱 0.48–0.62（原 0.52–0.66，让出 4pp 给六族）· fam 0.04–0.12；其余不动。
    #   ⚠️ 数字待 W4 首次实测回填（先按行数估：fam ≈200 行 × ~700 tok ≈ 8–10%）
    bands = {"v13": (0.48, 0.62), "l2": (0.10, 0.17), "l1": (0.03, 0.09),
             "chat": (0.01, 0.07), "fam": (0.04, 0.12), "cot": (0.05, 0.30)}
    # ★ 09-04 全量自检：五桶带宽是 v15 行重下标定的（27B 人话更长、think 上限 2048 ⇒ 行重全变）。按上面那行注释的原计划
    #   「首次实测回填」：v16 首建**只报读数**（manifest 记 bands_mode=report_only），Chaoyu 按读数批准带宽后
    #   U_BUILD_BANDS_STRICT=1 变硬闸。⚠️ CoT 上沿 30% 是裁定④，**始终硬**（守则：空门槛不等于通过）。
    _strict = os.environ.get("U_BUILD_BANDS_STRICT", "0") == "1"
    for k, (lo, hi) in bands.items():
        if DRY:
            continue
        ok_band = lo <= share[k] <= hi
        if k == "cot":
            assert share[k] <= hi, f"🔴 CoT 份额 {share[k]:.1%} > 上沿 {hi:.0%}（裁定④）"
        if _strict:
            assert ok_band, f"🔴 份额闸：{k}={share[k]:.1%} ∉ [{lo:.0%},{hi:.0%}]"
        elif not ok_band:
            print(f"  [份额-报告] {k}={share[k]:.1%} ∉ [{lo:.0%},{hi:.0%}]（首建只报，待回填带宽）")
    print(f"[份额] 模式={'硬闸' if _strict else '首建报告（U_BUILD_BANDS_STRICT=1 变硬闸）'}")
    # ★ 同形体检（守则⑮）：建库产物上再跑一遍（W2⑥ 的测试在真产物上复跑）
    sc = shape_check(tokenizer, l2 + l1 + chat_rows + fam)
    print(f"[同形] 多轮行 {sc['n']} · 不同形 {len(sc['bad'])} · 缺真实终答 {len(sc['missing_real_reply'])}")
    for cid, why in sc["bad"][:10]:
        print(f"   ✗ {cid}: {why}")
    assert not sc["bad"], f"🔴 同形体检 {len(sc['bad'])} 处不同形"
    if DRY:
        _dry = pd.DataFrame(list(t13.to_dict("records")) + l2 + l1 + chat_rows + fam + cot)
        from u_build_v15_multiturn import SOURCE_OF as _SRC
        from syncopate.pipeline.split import assert_split_isolation as _asi, base_case_id as _bcid
        _dry["source_case_ids"] = [[_SRC.get(str(c)) or _bcid(str(c))] for c in _dry["case_id"]]
        _asi(_dry, Path(DEFAULT_SPLIT_DIR), "sft")          # 三桶隔离③：DRY 也走出口闸
        _dp = Path("_audit/v16/dry_rows.parquet"); _dp.parent.mkdir(parents=True, exist_ok=True)
        _dry.to_parquet(_dp)
        print(f"[DRY] 演练产物 → {_dp}（给 scripts/v15_data_gallery.py 看终态）· 不提前返回，继续走全部结构闸")
        print(f"[DRY] 行数 {dict((k, len(v)) for k, v in [('l2', l2), ('l1', l1), ('chat', chat_rows), ('fam', fam), ('cot', cot)])}"
              f" · 缺真实终答的历史 {sc['missing_real_reply'][:5]}")
        # ★ 09-04 Chaoyu：结构闸必须本机就能验（419 冻结行数 / 考场泄漏 / think 双开头都是 DRY 提前返回才漏到云上的）
    if IS_V15:
        # ── 闸：人话不许出现在机器通道 + 信令自由文本不许是同一句（`25 §7㉗㉘`）──
        #
        # ⛔ 这两条都是**考场炸出来之后**才补的（08-30）。此前的密度闸只量终答尾巴，
        #   `session.report` 的参数和信令的 explanation 是**闸的盲区** ——
        #   于是 150 行人话进了机器通道、全库只有 3 句拒绝话，两样都没人看见。
        #   ⇒ 判据要跟着"模型实际会读到的每一段文本"走，不是只跟着终答走。
        import re as _re

        from syncopate.core.contract import PROSE_FIELDS
        _rp = _re.compile(r'<tool_call>\s*(\{"name": "session\.(?:report|defer|clarify|reject)".*?\})\s*</tool_call>', _re.S)
        prose_in_report, sig_lines = 0, Counter()
        for r in new_rows + valrows:
            txt = tokenizer.decode(list(r["input_ids"])[:r["total_length"]])
            for m in _rp.findall(txt):
                try:
                    call = json.loads(m)
                except Exception:
                    continue
                args = call.get("arguments") or {}
                if call["name"] == "session.report":
                    prose_in_report += len(set(args) & set(PROSE_FIELDS))
                else:
                    for k in ("explanation", "question", "reason"):
                        if args.get(k):
                            sig_lines[str(args[k])] += 1
        top = sig_lines.most_common(1)
        top_share = (top[0][1] / sum(sig_lines.values())) if sig_lines else 0.0
        print(f"[人话通道] report 里的人话字段 {prose_in_report}（必须 0）· "
              f"信令自由文本 {len(sig_lines)} 种/{sum(sig_lines.values())} 次 · "
              f"最高频 {top_share:.0%}"
              + (f" ('{top[0][0][:24]}')" if top else ""))
        assert prose_in_report == 0, f"🔴 人话进了机器通道 {prose_in_report} 处"
        assert top_share <= 0.35, f"🔴 信令话术复读 {top_share:.0%}（≤35%）—— 会长成万能出口"
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
            for t in OOV:
                if t in gold:
                    teach_hits += 1
                    _i = gold.find(t)
                    print(f"   ✗ OOV 教学面：{r.get('bucket')} {r.get('case_id')} 词「{t}」…{gold[max(0, _i - 50):_i + 30]!r}")
        ambient_hits += n
    for term, vs in DEFS.items():
        for v in vs:
            for t in OOV:
                if t in v:
                    teach_hits += 1
                    print(f"   ✗ OOV 教学面：DEFS[{term}] 词「{t}」：{v[:80]!r}")
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
        for t in EXAM_LAST:
            if len(t) >= 8 and t in txt:
                leak_last += 1
                print(f"   ✗ 考场被判句泄漏：{r.get('bucket')} {r.get('case_id')} 句「{t[:40]}」")   # 09-04 run21：闸红要说是谁
        leak_first += sum(1 for t in first_turns if len(t) >= 8 and t in txt)
    print(f"[泄漏] 被判轮命中 {leak_last}（必须 0）· 铺垫轮 {leak_first}（上报）")
    assert leak_last == 0, f"🔴 考场被判句泄漏 {leak_last}"
    for r in new_rows + valrows:
        assert r["supervised_tokens"] > 0 and \
            len(r["input_ids"]) == len(r["loss_mask"]) == r["total_length"], r["case_id"]

    out = Path("_audit/v16/dry") if DRY else Path(DEFAULT_SFT_DIR)   # DRY 产物落审计目录；正式目录只许正式建库写
    # ★ 三桶隔离②③（09-04）：每行登记底题；落盘只走唯一带闸写盘函数（底题桶 == 产物桶，越桶直接抛）
    from u_build_v15_multiturn import SOURCE_OF as _SRC
    from syncopate.pipeline.split import base_case_id as _bcid
    from syncopate.pipeline.build_dataset import write_split_checked
    train_records = train.to_dict("records"); val_records = val.to_dict("records")
    for r in train_records + val_records:
        cid = str(r.get("case_id", ""))
        r["source_case_ids"] = [_SRC.get(cid) or _bcid(cid)]
    axes = Counter(r.get("sub_axis", "?").split("|")[0] for r in new_rows)
    manifest = {"version": "v15" if IS_V15 else "v14.5", "seed": 1455,
                "sources": {"v13_train": len(t13), "multiturn_l2": len(l2),
                            "multiturn_l1": len(l1), "chat_shell": len(chat_rows),
                            "fam": dict(Counter(r["bucket"] for r in fam)), "cot_hard": len(cot)},
                "render": {"prior": "message_pairs", "time": "date_only", "menu": "full_34",
                           "answer_fields": "min_fields_v15", "since": "2026-09-02 W2"},
                "total": len(train), "val": len(val),
                "sup_tok_share": {k: round(v, 4) for k, v in share.items()},
                "axis_counts": dict(axes),
                "bands_mode": "strict" if os.environ.get("U_BUILD_BANDS_STRICT", "0") == "1" else "report_only_first_v16",
                "cot_share_target": 0.28,
                "gates": "份额（见 bands_mode）· CoT≤30% 硬 · 密度 · OOV=0 · 泄漏=0 · 冻结桶行数派生相等"}
    manifest = write_split_checked(out, train_records, val_records, split_dir=Path(DEFAULT_SPLIT_DIR), pool="sft", manifest_extra=manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    # ── 出厂体检（`25 §7㉙`，Chaoyu 08-30：「不能走完完整训练之后再返工来做检查」）──
    #   ⚠️ 必须在**落盘之后**跑真产物，不是跑内存里的中间态 ——
    #     ㉖ 那次就是闸写在生产者里，缓存命中时整条闸被绕过去了。
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "v15_data_audit", Path(__file__).resolve().parent / "v15_data_audit.py")
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _audit = _mod.audit
    if DRY:
        print(f"[DRY] 出厂体检跳过（依赖教师真文本；结构闸 + 出口写盘 + 隔离已全部走过）· 产物 {out}/train.parquet")
        return 0
    rep = _audit(out / "train.parquet")
    if not rep["ok"]:
        print("🔴 出厂体检未通过 ⇒ 不许进训练（见上面的 🔴 行）")
        return 1
    print(f"✅ {'v15' if IS_V15 else 'v14.5'} 构建完成，全部门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
