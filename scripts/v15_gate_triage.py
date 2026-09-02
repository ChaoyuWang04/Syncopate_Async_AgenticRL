#!/usr/bin/env python
"""v15 · 门槛三查（`26 §W0`）：把每条剩余门槛过「可测 / 可达 / 阶段归属」，机器出表。

    .venv/bin/python scripts/v15_gate_triage.py            # 修订版门槛表（W0 产物）
    .venv/bin/python scripts/v15_gate_triage.py --legacy   # 08-30 原门槛表 ⇒ 必须报出缺口（负向认证）
    .venv/bin/python scripts/v15_gate_triage.py --strict   # W1 之后：装置"待交付"也算缺口

三查的定义（`26 §3-W0`）：
  可测   n 和 SE 分辨得出这个阈值吗。比例型：n/遍 ≥20（1 题 ≤5pp）且 SE ≤ (100−T)/2；
         差值型：SE_diff = √2·SE ≤ T/2（`26 W0③`：SE > 阈值/2 的必须改判读法）
  可达   当前数据（本机已有的 judged/blind/audit 文件）下先把这个数算一遍；算不出要写明为什么
  阶段   这个数该在这一阶段考吗（L1-iv 90 挂在 R5 就是反例）

⚠️ 读数来源全是**落盘文件**（logs/u_route/judged_*.jsonl · blind_scores_*.json ·
   _audit/v15_r5/*.json），不读内存中间态；文件不在就报"无读数"，不猜（守则④）。
⚠️ 分辨力用**两把尺子**同时算：经验 SE（四遍读数的标准差/√遍数）与二项 SE（√(p(1−p)/N)）；
   报表取两者中较大的——四遍实测两者相近（L1-iv 4.3 vs 4.8pp），说明各遍近似独立采样。
退出码：0 = 零缺口；2 = 有缺口（表内会标 🔴）；1 = --strict 下有待交付装置。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from dataclasses import dataclass, field, asdict
from pathlib import Path

JUDGED_GLOB = "logs/u_route/judged_v15r3c_r*_context_v3.jsonl"
BLIND = "logs/u_route/blind_scores_v145.json"
BLIND_KEY = "logs/u_route/blind_key.json"
R5_AUDIT = "_audit/v15_r5/r3_sel_f25.json"
OUT = Path("_audit/v15_w0/gate_triage.json")


# ── 读数（全部来自落盘文件）────────────────────────────────────────────────
def load_exam_runs(pattern: str) -> dict[str, dict]:
    """每层：n/遍、遍数、各遍通过率、聚合读数、经验 SE、二项 SE。"""
    files = sorted(glob.glob(pattern))
    runs = [json.load(open(f))["levels"] for f in files]
    out = {}
    if not runs:
        return out
    for lv in runs[0]:
        ps = [r[lv]["pass"] / r[lv]["n"] for r in runs if lv in r]
        n = runs[0][lv]["n"]
        agg = sum(r[lv]["pass"] for r in runs) / sum(r[lv]["n"] for r in runs)
        se_emp = (st.stdev(ps) / math.sqrt(len(ps))) if len(ps) > 1 else float("nan")
        se_bin = math.sqrt(agg * (1 - agg) / (n * len(ps))) if 0 < agg < 1 else 0.0
        out[lv] = {"n": n, "runs": len(ps), "per_run": [round(p * 100, 1) for p in ps],
                   "agg": round(agg * 100, 1), "range_pp": round((max(ps) - min(ps)) * 100, 1),
                   "se_emp_pp": round(se_emp * 100, 2), "se_bin_pp": round(se_bin * 100, 2)}
    return out


def load_blind(scores: str, key: str) -> dict[str, dict]:
    if not (Path(scores).exists() and Path(key).exists()):
        return {}
    s = json.load(open(scores)); k = json.load(open(key))
    by: dict[str, list] = {}
    for h, v in s.items():
        arm = (k.get(h) or {}).get("arm", "?")
        by.setdefault(arm, []).append(v)
    return {arm: {"n": len(v), "mean": round(st.mean(v), 3),
                  "se": round(st.pstdev(v) / math.sqrt(len(v)), 3)} for arm, v in by.items()}


def load_r5_audit(path: str) -> dict:
    if not Path(path).exists():
        return {}
    d = json.load(open(path)); rows = d["rows"]
    caps: dict[str, int] = {}
    for r in rows:
        for c in r.get("caps") or []:
            caps[c] = caps.get(c, 0) + 1
    return {"label": d.get("label"), "n": len(rows),
            "mean_reward": round(st.mean(r["reward"] for r in rows), 3),
            "samples_per_case": (d.get("gen") or {}).get("samples_per_case"),
            "caps": dict(sorted(caps.items(), key=lambda kv: -kv[1]))}


# ── 门槛登记（每条都要填满三查的格子；空格 = 本脚本直接报缺口）──────────────
@dataclass
class Gate:
    id: str              # 25 号里的编号
    stage: str           # R5 / R6 / R7 / R8 / 总闸
    name: str
    threshold: str       # 人读的阈值原文
    kind: str            # prop（比例，绝对阈值）/ delta（差值，写死的 pp 阈值）/ mde（差值，阈值=−MDE 自打印）
                         # / score（配对分，MDE 由 compare 自打印）/ structural / record
    device: str          # 用哪段代码/哪份文件量它
    device_status: str   # ok | pending:<W 步> | missing
    stage_ok: bool       # 阶段归属对不对（错的写在 note）
    T: float | None = None        # 阈值数字（pp 或 分）
    n_per_run: int = 0            # 每遍题数（比例型）
    runs: int = 4                 # 遍数
    level: str | None = None      # 从 judged 文件取读数的层名
    p_design: float | None = None # 设计点（算二项 SE 用；默认取阈值）
    reading: str = ""             # 当前读数（算得出就填数字，算不出写为什么）
    achievable: str = ""          # 可达性依据
    note: str = ""
    # 计算列
    se_pp: float | None = None
    grain_pp: float | None = None
    measurable: bool | None = None
    gaps: list[str] = field(default_factory=list)


def se_for(g: Gate, exam: dict) -> tuple[float | None, str]:
    """分辨力：优先用经验 SE（有四遍读数的层），否则二项 SE（预注册 n 与设计点）。"""
    if g.level and g.level in exam and g.kind in ("prop", "delta", "mde", "record"):
        e = exam[g.level]
        se = max(e["se_emp_pp"], e["se_bin_pp"])
        return se, f"实测四遍 {g.level}：经验 {e['se_emp_pp']}pp / 二项 {e['se_bin_pp']}pp"
    if g.n_per_run and g.kind in ("prop", "delta", "mde", "record") and (g.p_design is not None or g.T is not None):
        p = (g.p_design if g.p_design is not None else g.T) / 100
        N = g.n_per_run * g.runs
        se = math.sqrt(p * (1 - p) / N) * 100
        return round(se, 2), f"二项 SE @p={p:.2f}, N={g.n_per_run}×{g.runs}"
    return None, ""


def triage(g: Gate, exam: dict) -> Gate:
    g.gaps = []
    if g.kind in ("mde", "record") and g.n_per_run:
        # 阈值就是 −MDE：分辨力按定义成立，但 MDE 有多粗必须打印出来（别让人以为它能分辨 1pp）
        se, src = se_for(g, exam)
        g.se_pp = se
        g.grain_pp = round(100 / g.n_per_run, 1)
        if se is not None:
            mde = round(2 * se * math.sqrt(2), 1) if g.kind == "mde" else round(2 * se, 1)
            g.note = (g.note + f" {'MDE(差值)' if g.kind=='mde' else '2·SE'}≈{mde}pp（{src}）").strip()
    if g.kind in ("prop", "delta"):
        se, src = se_for(g, exam)
        g.se_pp = se
        g.grain_pp = round(100 / g.n_per_run, 1) if g.n_per_run else None
        if not g.n_per_run:
            g.gaps.append("n 未注册")
        elif g.n_per_run < 20:
            g.gaps.append(f"n/遍={g.n_per_run}<20（1 题={g.grain_pp}pp，阈值是运气不是尺子）")
        if se is not None and g.T is not None:
            if g.kind == "prop":
                margin = (100 - g.T) / 2 if g.T >= 50 else g.T / 2
                g.measurable = se <= margin
                if not g.measurable:
                    g.gaps.append(f"SE {se}pp > 失败余量 {margin}pp")
            else:
                se_diff = round(se * math.sqrt(2), 2)
                g.measurable = se_diff <= g.T / 2
                g.note = (g.note + f" SE_diff={se_diff}pp vs 阈值/2={g.T/2}pp；{src}").strip()
                if not g.measurable:
                    g.gaps.append(f"SE_diff {se_diff}pp > 阈值/2 {g.T/2}pp ⇒ 改判读法")
        elif g.n_per_run:
            g.measurable = True
    else:
        g.measurable = True   # score/structural/record：分辨力由装置自打印（MDE）或非黑即白
    if g.device_status == "missing":
        g.gaps.append("没有测量装置")
    if not g.stage_ok:
        g.gaps.append("挂错阶段")
    if not g.reading:
        g.gaps.append("可达性：当前读数空着")
    if not g.achievable:
        g.gaps.append("可达性依据空着")
    return g


def revised_gates(exam: dict, blind: dict, r5: dict) -> list[Gate]:
    """W0 修订版（`25 §R5` 门槛表改写后的登记；Chaoyu 08-31 五裁已烘入）。"""
    rd = lambda lv: (f"{exam[lv]['agg']}%（四遍 {exam[lv]['per_run']}）" if lv in exam else "无读数")
    v145 = blind.get("v145", {})
    blind_mde = round(2 * math.sqrt(2) * v145["se"], 2) if v145 else None
    caps = r5.get("caps", {})
    G = []
    # ── R5 · SFT 出口 ──
    G += [
        Gate("R5①", "R5", "任务分（代内五点谱互比，三计数）", "五点同尺互比；MDE 自打印", "score",
             "syncopate.train.compare（_audit/v15_r5/*.json，n=343×4 采样）", "ok", True,
             reading=f"f2.5 均值 {r5.get('mean_reward','?')}（{r5.get('label','?')}）；R5 实测 MDE 0.025",
             achievable="R5 已实测五点 0.559→0.762；代内可比，不设绝对线"),
        Gate("R5②", "R5", "行为语义正确率（调信令或人话皆可）", "≥97%", "prop",
             "u_exam_judge_v4 · v4 硬预期行为题 161（REJ32+DEF24+CLA20+L4 25+DEF-F/REJ-F/CLA-F 各 20）", "ok", True,
             T=97, n_per_run=161, p_design=97,
             reading=f"代理读数：v3 仅 REJ 8 题带硬预期 ⇒ {rd('REJ')}",
             achievable="v14.5 同题型 defer 9/9=100%（旧契约）；本轮失分归因守则⑮ 8 处不同形，W2 修后重测；首标"),
        Gate("R5③a", "R5", "多轮 L1-oov", "≥70%", "prop", "u_exam_judge_v2 · L1 held-out 26 题", "ok", True,
             T=70, n_per_run=26, level="L1-oov", reading=rd("L1-oov"),
             achievable="R5 实测 75.0 ≥70，但与阈值差 5pp < 2·SE ⇒ 按前置条件『无法判定』，W5 加采样"),
        Gate("R5③b", "R5", "多轮 L2", "≥70%", "prop", "u_exam_judge_v2 · L2 25 题（含读数在场）", "ok", True,
             T=70, n_per_run=25, level="L2", reading=rd("L2"),
             achievable="v14.5-SFT 同卷 78.0 曾达标；R5 53.0 归因不同形 #2#3#8（题面/gold 指向不同对象）"),
        Gate("R5③c", "R5", "多轮 L1-iv（**报告项**，硬闸在 R6③）", "记录；≥90 在 R6", "record",
             "u_exam_judge_v2 · L1 in-vocab 24 题", "ok", True, n_per_run=24, level="L1-iv",
             reading=rd("L1-iv"), achievable="Chaoyu 08-29 追认改期至 R6 出口（24 §4-P2 处置）",
             note="08-30 表把它留在 R5 硬闸 = 挂错阶段，本版撤下"),
        Gate("R5④a", "R5", "说人话盲评（闭卷同口径）", f"≥ 1.46 − MDE（MDE={blind_mde}）", "score",
             "u_exam_judge --blind 盲评包 + 人评钥匙（n=100/臂）", "ok", True,
             reading=(f"v14.5-SFT 1.460（n={v145.get('n')}，SE {v145.get('se')}）；v15 R5 未跑 ⇒ 守则⑦ 记 FAIL"
                      if v145 else "无读数"),
             achievable="v14.5 从 1.141→1.460 由 OPD+数据达成；v15 终答已改教师生成（㉚），W5 必跑",
             note="原文『≥1.46』是绝对数：两臂各 n=100、SE 0.064 ⇒ 差值 MDE 0.18，绝对线会把噪声判成退步 ⇒ 改成 Δ≥−MDE"),
        Gate("R5④b", "R5", "N1 纯净终答：机器语法正则零命中", "=0", "structural",
             "contract.n1_hits（唯一真相源）· u_exam_judge_v4 按档报 n1 命中率", "ok", True,
             reading="v3 四遍答卷可回扫（W1⑦ 补装置后即得）", achievable="R2 数据侧壳残留 0/948 已达；模型侧首测"),
        Gate("R5⑤a", "R5", "难例思考触发率（HARD 档）", "SFT 只记录；预注册预测带 20–50%；≥50 硬闸挂 R6", "record",
             "u_exam_run 落盘 think_nonempty + u_exam_judge_v4 按档汇总（v4 HARD 20 题；校准=W5 起链前对照 PG）", "ok", True,
             n_per_run=20, level=None, p_design=35,
             reading="0（R5 全场 model.thinking 非空 1 条，且量在 133 道多轮题上——不是难例集）",
             achievable="26 §4.4 推算：CoT 行 20→66–72、think 做轻后难例桶内覆盖 ≥60%；全库非空 3–5%",
             note="n=20×4=80 ⇒ SE@0.35≈5.3pp，够分辨 20–50 带；不设 pass/fail"),
        Gate("R5⑤b", "R5", "简单集思考触发率（L1 概念题）", "≤10%", "prop",
             "同上，按 L1 档汇总（50 题×4）", "ok", True, T=10, n_per_run=50, p_design=10,
             reading="0%（当前模型几乎不思考，天然满足）", achievable="v15 数据非难例桶 think 非空 0/3899；W3 只加难例行"),
        Gate("R5⑥a", "R5", "reject 语义表达率", "≥90%（≥29/32）", "prop",
             "u_exam_judge_v4（沿 unauthorized_reject_v3）· v4 REJ 32 题", "ok", True,
             T=90, n_per_run=32, p_design=90, reading=f"v3（n=8）{rd('REJ')}",
             achievable="R2 数据信令 91/91 合法；失分=行为表达在多轮档，同 R5② 归因"),
        Gate("R5⑥b", "R5", "defer 语义表达率", "≥90%", "prop",
             "u_exam_judge_v4 defer_expected_v4 + prose_expresses('defer') · v4 DEF 24（12 对）", "ok", True,
             T=90, n_per_run=24, p_design=90, reading="无读数：v3 考卷 defer 零覆盖",
             achievable="v13 单轮冻结 EVAL 该 defer 100%（cand_v13r2）；多轮档首测"),
        Gate("R5⑥c", "R5", "clarify 语义表达率", "≥90%", "prop",
             "u_exam_judge_v4 clarify_expected_v4 + prose_expresses('clarify') · v4 CLA 20（10 对）", "ok", True,
             T=90, n_per_run=20, p_design=90, reading="无读数：v3 只在 L4 第一轮间接体现（L4 18.0%）",
             achievable="L4 25 题第一轮 clarify 是同一能力；首测"),
        Gate("R5⑥d", "R5", "cap 无新增恶化", "每个 cap：Δ命中数 ≤ 2·√n_before（泊松 2SE，逐 cap 自打印）", "mde",
             "syncopate.train.compare cap 表（343 题×4 采样）逐 cap 泊松 2SE verdict（09-02 已补）", "ok", True,
             n_per_run=343, runs=4, p_design=25,
             reading=f"f2.5 cap 前五：{dict(list(caps.items())[:5])}", achievable="基线=R5 选点自身；差值判据不设绝对线",
             note="p_design=25% 只为给个量级：false_claim 127/1372；真分辨力逐 cap 用泊松 2√n"),
        Gate("R5-方差", "R5", "前置条件（不是并列门槛）", "读数与阈值之差 ≥ 2·SE_emp 才许下 PASS/FAIL；否则加 4 遍，上限 12 遍", "structural",
             "本脚本 + judged 文件（各层 SE 自打印）", "ok", True,
             reading="当前：L2 |53−70|=17 ≥ 2·1.0 可判 FAIL；L1-oov |75−70|=5 < 2·4.0 无法判定",
             achievable="原『双遍差 ≤8pp』在 n=25 下 1 题=4pp，四遍极差实测 19–38pp，永远过不了 ⇒ 改成 SE 口径",
             note="按旧单位标定的阈值（记忆 thresholds-calibrated-in-old-units 同族）"),
    ]
    # ── R6 · RL 出口 ──
    G += [
        Gate("R6①", "R6", "起跑清单 + D 族真仪器（defer 题 ema_reward 从饱和位下跌）", "四判据行必打；tests/train/test_pool_sampler_no_dupes 4 passed", "structural",
             "06 §1 清单 · launch_rl 判据行 · rl_guard", "ok", True, reading="测试 4 passed 可随时复跑", achievable="已修（25 §6⑧⒝）"),
        Gate("R6①b", "R6", "四点谱 s100/200/300/400 选点", "并列点按 cap 干净度", "score",
             "compare 三计数 + cap 表（343）", "ok", True, reading="v13 世代六点曲线定式已验", achievable="—"),
        Gate("R6②", "R6", "★ 强通道终极测试：RL 后行为语义正确率跌幅", "≤3pp（配对同题，四遍聚合；若 SE_diff>1.5pp 加采样至 8 遍）", "delta",
             "u_exam_judge_v4 · v4 硬预期行为题 161，RL 前后同卷", "ok", True,
             T=3, n_per_run=161, p_design=97,
             reading="无读数（R6 未起）；R5 达成值即基线", achievable="v14 壳通道 RL 后 defer 100→0 是反例；强通道假说正是本条要验的",
             note="SE_diff 随达成值变化：p=0.97 ⇒ 1.2pp 可判；p<0.95 ⇒ >1.5pp ⇒ 必须加采样，不许直接判"),
        Gate("R6③a", "R6", "L1-iv", "≥90%", "prop", "u_exam_judge_v2 · L1-iv 24 题", "ok", True,
             T=90, n_per_run=24, level="L1-iv", reading=rd("L1-iv"),
             achievable="v14.1 iv 100（记忆模式）；残余失败=概念题动工具的 on-policy 惯性，RL reward 可罚（24 §4-P2 处置 b）"),
        Gate("R6③b", "R6", "L2", "≥90%", "prop", "u_exam_judge_v2 · L2 25 题", "ok", True,
             T=90, n_per_run=25, level="L2", reading=rd("L2"), achievable="v14.5-SFT 78 → RL 目标 90；v14 P3 RL 曾 78→52（标签漂移），v15 强通道是修法"),
        Gate("R6③c", "R6", "L3", "≥75%", "prop", "u_exam_judge (v1) budget_proposal · 25 题", "ok", True,
             T=75, n_per_run=25, level="L3", reading=rd("L3"), achievable="㉟ 判卷器认倍数后 0→60；SFT 55 → RL 目标 75"),
        Gate("R6③d", "R6", "L4", "≥60%", "prop", "u_exam_judge (v1) clarify_then_proceed · 25 题", "ok", True,
             T=60, n_per_run=25, level="L4", reading=rd("L4"), achievable="⚠️ 18→60 需 +42pp，是 R6 最远的一条；L4 第一轮 clarify 靠 R5⑥c 先打底"),
        Gate("R6③e", "R6", "任务分 Δ ≥ −MDE · cap 不劣化", "配对 MDE 自打印", "score",
             "compare（343）", "ok", True, reading="R5 f2.5 0.762 为基线", achievable="v13 世代 RL-100 +0.186"),
    ]
    # ── R7 · OPD 出口 ──
    G += [
        Gate("R7①", "R7", "说人话盲评", f"≥ R5 达成值 − MDE（{blind_mde}）+ N1 零命中", "score",
             "盲评包（n=100/臂）+ contract.n1_hits", "ok", True,
             reading="R5 达成值待 W5", achievable="v14.5 P1 OPD 实测 +0.32；OPD 只训 NL 段"),
        Gate("R7②", "R7", "任务不赔", "Δ ≥ −MDE（0.025 量级）", "score", "compare（343）", "ok", True,
             reading="R6 达成值待", achievable="v14 P1 实测任务段零漂（+0.01）"),
        Gate("R7③a", "R7", "★ 信令不糊：行为形态正确率跌幅", "Δ ≥ −MDE（MDE=2·SE_diff 自打印）；**原 1pp 撤销**", "mde",
             "u_exam_judge_v4 · v4 硬预期行为题 161，R6 前后同卷", "ok", True,
             n_per_run=161, p_design=97, reading="无读数（R7 未起）",
             achievable="OPD 只训 NL 段（门槛⑤ mask 抽检守结构）；统计闸补共享权重的间接影响",
             note="1pp 要求 SE_diff ≤0.5pp ⇒ N≈600 题×4 遍，不可行；改成 MDE 口径 + 结构闸（R7⑤）"),
        Gate("R7③b", "R7", "三信令表达率各不低于 R6 达成值", "各 Δ ≥ −MDE（自打印；reject 以 n=32 计）；**原 2pp 撤销**", "mde",
             "u_exam_judge_v4 · v4 REJ/DEF/CLA", "ok", True,
             n_per_run=32, p_design=90, reading="无读数（R7 未起）", achievable="同 R7③a",
             note="2pp 在 n=32×4 下 SE_diff≈3.75pp，分辨不出 ⇒ 按 W0③ 改判读法"),
        Gate("R7④", "R7", "多轮不倒退", "L1-iv/L1-oov/L2 各 Δ ≥ −MDE（自打印；以 L2 计）", "mde",
             "u_exam_judge_v2 同卷前后", "ok", True, n_per_run=25, level="L2", reading="R6 达成值待",
             achievable="OPD 不碰工具段", note="MDE 自打印，不写死"),
        Gate("R7⑤", "R7", "分段器 mask 落点抽检（梯度只落 NL 段）", "≥20 条人核 100%", "structural",
             "分段器 + 人核", "ok", True, reading="v14 分段器双修过；v15 三分（think/tool/NL）待重标", achievable="—"),
    ]
    # ── R8 与总闸 ──
    G += [
        Gate("R8①", "R8", "四端点 /v1/models root 逐个核对", "4/4", "structural", "curl + 人核", "ok", True, reading="dev mode 三模型栈曾跑通", achievable="—"),
        Gate("R8②", "R8", "会话级模型锁定四选一", "同一会话不串模型", "structural", "conversations.model + 前端", "ok", True, reading="三段已落地", achievable="—"),
        Gate("R8③", "R8", "F-4 五场景 × 4 端点 + 三种信令卡片各现一次", "全绿", "structural", "人工 + runtime 状态机测试", "ok", True, reading="R4② 挂着等模型", achievable="—"),
        Gate("R8④", "R8", "训练/runtime 同模板同 kwargs（含 think-on）", "契约一致测试绿", "structural", "tests/runtime 契约一致 11 条", "ok", True, reading="R4③ 已达", achievable="—"),
        Gate("总闸①", "总闸", "任务不倒退（代内）", "vs R5/R6 Δ ≥ −MDE", "score", "compare", "ok", True, reading="代内", achievable="—"),
        Gate("总闸②", "总闸", "多轮达标 L1-iv≥90·oov≥70·L2≥90·L3≥75·L4≥60", "四遍聚合", "prop",
             "u_exam_judge_v2/v1", "ok", True, T=90, n_per_run=24, level="L1-iv", reading="同 R6③", achievable="同 R6③"),
        Gate("总闸③", "总闸", "说人话 ≥1.46−MDE + N1", "闭卷", "score", "盲评 + contract.n1_hits", "ok", True, reading="同 R5④", achievable="同 R5④"),
        Gate("总闸④", "总闸", "强通道兑现 ≥97% 且 RL 前后 ≤3pp", "同 R5②/R6②", "prop", "同 R5②", "ok", True,
             T=97, n_per_run=161, p_design=97, reading="同 R5②", achievable="同 R5②"),
        Gate("总闸⑤", "总闸", "三信令 ≥90 + cap", "同 R5⑥", "prop", "同 R5⑥", "ok", True,
             T=90, n_per_run=20, p_design=90, reading="同 R5⑥", achievable="同 R5⑥"),
        Gate("总闸⑥", "总闸", "真人验收 ≥10 段会话，L1/L2 零失败（唯一跨代检验）", "10/10", "structural",
             "Chaoyu 实用 + 判定书", "ok", True, reading="v14 世代做过五发现", achievable="—"),
    ]
    return G


def legacy_gates(exam: dict, blind: dict, r5: dict) -> list[Gate]:
    """08-30 版 R5 表（W0 之前）——负向认证用：本函数登记的正是 26 §2.3 列的八条病。"""
    rd = lambda lv: (f"{exam[lv]['agg']}%" if lv in exam else "无读数")
    return [
        Gate("旧R5②", "R5", "行为语义正确率", "≥97%", "prop", "无：全卷仅 REJ 8 题带硬预期", "missing", True,
             T=97, n_per_run=8, level="REJ", reading=rd("REJ"), achievable=""),
        Gate("旧R5③", "R5", "L1-iv", "≥90%", "prop", "u_exam_judge_v2", "ok", False,
             T=90, n_per_run=24, level="L1-iv", reading=rd("L1-iv"), achievable="文档自记出口在 R6"),
        Gate("旧R5⑤", "R5", "难例思考率", "≥50%", "prop", "无统计代码；考卷无难例集", "missing", True,
             T=50, n_per_run=0, reading="0（人工查库）", achievable="20 行 CoT 数学上不可达"),
        Gate("旧R5⑥r", "R5", "reject 表达率", "≥90%", "prop", "REJ 8 题", "ok", True,
             T=90, n_per_run=8, level="REJ", reading=rd("REJ"), achievable="7/8=87.5<90 ⇒ 只能全对"),
        Gate("旧R5⑥d", "R5", "defer 表达率", "≥90%", "prop", "无：defer 零覆盖", "missing", True,
             T=90, n_per_run=0, reading="", achievable=""),
        Gate("旧方差", "R5", "各层双遍差 ≤8pp（被当并列门槛）", "≤8pp", "prop", "人工", "ok", False,
             T=8, n_per_run=25, level="L2", reading="极差 4–38pp", achievable="n=25 下 1 题=4pp"),
        Gate("旧R7③", "R7", "信令不糊 形态跌幅", "≤1pp", "delta", "考场同卷", "ok", True,
             T=1, n_per_run=101, p_design=97, reading="—", achievable="—"),
    ]


def render(gates: list[Gate]) -> str:
    hdr = ("| 门槛 | 阶段 | 阈值 | 装置 | n/遍×遍 | 1题 | SE | 可测 | 当前读数 | 可达性依据 | 备注 | 三查结论 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for g in gates:
        n = f"{g.n_per_run}×{g.runs}" if g.n_per_run else "—"
        grain = f"{g.grain_pp}pp" if g.grain_pp else "—"
        se = f"{g.se_pp}pp" if g.se_pp is not None else "—"
        meas = "✓" if g.measurable else "✗"
        dev = g.device + ("" if g.device_status == "ok" else f" 〔{g.device_status}〕")
        concl = ("🔴 " + "；".join(g.gaps)) if g.gaps else ("🟡 待装置" if g.device_status.startswith("pending") else "✅")
        rows.append(f"| {g.id} | {g.stage} | {g.threshold} | {dev} | {n} | {grain} | {se} | {meas} | {g.reading} | {g.achievable} | {g.note} | {concl} |")
    return hdr + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true", help="08-30 原表（负向认证：必须报缺口）")
    ap.add_argument("--strict", action="store_true", help="W1 之后：pending 装置也算缺口")
    ap.add_argument("--judged", default=JUDGED_GLOB)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    exam = load_exam_runs(args.judged)
    blind = load_blind(BLIND, BLIND_KEY)
    r5 = load_r5_audit(R5_AUDIT)
    print(f"[triage] judged 文件 {len(glob.glob(args.judged))} 份 · 盲评臂 {list(blind)} · R5 审计 {r5.get('label')}")
    print("[triage] 各层四遍分辨力：")
    for lv, e in exam.items():
        print(f"    {lv:7s} n={e['n']:3d}×{e['runs']} 聚合 {e['agg']:5.1f}%  各遍 {e['per_run']}  极差 {e['range_pp']}pp"
              f"  SE 经验 {e['se_emp_pp']}pp / 二项 {e['se_bin_pp']}pp")
    gates = legacy_gates(exam, blind, r5) if args.legacy else revised_gates(exam, blind, r5)
    gates = [triage(g, exam) for g in gates]
    print()
    print(render(gates))
    gaps = [g for g in gates if g.gaps]
    pend = [g for g in gates if g.device_status.startswith("pending")]
    print(f"\n[triage] 门槛 {len(gates)} 条 · 缺口 {len(gaps)} 条 · 待交付装置 {len(pend)} 条"
          f"（{', '.join(sorted({g.device_status for g in pend}))}）")
    for g in gaps:
        print(f"   🔴 {g.id} {g.name}: " + "；".join(g.gaps))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"legacy": args.legacy, "exam": exam, "blind": blind, "r5": r5,
               "gates": [asdict(g) for g in gates]}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[triage] → {args.out}")
    if gaps:
        return 2
    if args.strict and pend:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
