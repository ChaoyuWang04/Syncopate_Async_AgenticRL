#!/usr/bin/env python
"""v15 · W2⑤ —— 工具说明书全量修剪（Chaoyu 09-02 看过三条样例后批准）。

    SYNCOPATE_CONTRACT=v15 .venv/bin/python scripts/v15_w2_trim_tool_desc.py [--check]

规则（固定，逐条同一标准）：每条描述只留「做什么 · 输入 · 返回什么 · 本工具独有的硬规则」；
跨工具的通用纪律（查不到不编 / 过期不当依据 / 冲突记录 / 低置信不写 / 幂等重试）**只在 system.txt 写一次**；
交叉引用「X 不在这，在 Y」删（菜单是全量的，模型看得见全部工具）；原理解释删；星号圆点删。

★ 判据（不许无声消失）：
  ① 旧描述里的每个**硬事实**（数字、工具名、枚举值、字段名）必须出现在 新描述 ∪ 参数说明 ∪ system.txt 中，
     缺一个脚本就报错；② token 前后对照落盘 _audit/v15_w2/tool_desc_trim.json；
  ③ 被删的每句话归到四个筐之一（system_prompt / cross_ref / rationale / rewritten）逐条登记。
实施：用 ast 定位每个 @REGISTRY.tool(description=…) 的字面量位置**就地替换源文件**——注册表仍是唯一真相源，
训练与线上共用同一份，改一处两侧同变。--check 只核对不写。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

TOOLS_DIR = Path("syncopate/domains/adcampaign/tools")

NEW: dict[str, str] = {
 "campaign.get_metrics": "查询单个 campaign 的投放大盘指标：花费、安装、CPI、ROAS、CTR、频次、曝光。不含单条素材明细，不判断数据是否收敛。",
 "creative.get_metrics_by_asset": "按素材粒度查表现：逐条素材的 CTR、IPM、花费、曝光频次、疲劳分。不含 campaign 层汇总和视觉标签。",
 "campaign.detect_anomalies": "诊断 campaign 是否存在指标异常，返回异常类型列表（如 cpi_spike / roas_drop / creative_fatigue）。只定性给出类型，不给方案、不判断数据成熟度；要拿优化方案先用它确定类型。",
 "benchmark.get_industry_baseline": "查询行业基准值（不是自己的投放数据，也不是内部安全线），按 平台+游戏品类+指标 定位，用于判断自己的数据在行业里是高是低。只作参考，不是扩量的决策依据。",
 "analysis.feature_lift": "算某个素材 feature 在某个地域对 d7 ROAS 的提升幅度（lift），带 95% 置信区间、两组样本量和显著性判定。feature 取值：real_person / before_after / dark_palette / fast_cut / ugc_style。必须逐地域分别算，不同地域符号可能相反；样本量少于 12 条素材时不论 lift 多大都不能据此下结论，应如实说明样本不足。不返回素材清单和标签。",
 "analysis.geo_breakdown": "按地域拆某个产品的投放表现：各地域的 d7 ROAS、d7 CPI、素材条数。地域扩展前用它挑候选地域。它只反映各地域现状，能不能扩要逐个地域查安全线；素材条数少的地域数字不可信。",
 "benchmark.get_safety_line": "查询本产品在指定地域的内部投放安全线（每周更新）：d7 CPI 上限、d7 ROAS 下限、d1 留存下限、日预算上限，带 valid_from / valid_to。判断是否超标、能否加预算以这条线为准，不用行业基准代替。查不到返回 safety_line_not_found。",
 "calendar.get_seasonal_context": "查询当前时间点附近的时令活动（万圣节、黑五、圣诞等），返回距离天数、出量放大倍数和对应的素材标签。只给时令背景，不判断素材该不该投，不含投放指标。",
 "creative.get_asset_tags": "读取单条素材的视觉标签与历史表现（离线视觉分析产出）：主题标签、开头钩子类型、主色、是否有人脸、文字占比，以及历史 IPM / CTR / d7 CPI 和投放地域。不做跨素材归因，不含 campaign 层数据。",
 "creative.search_similar": "按视觉标签检索素材库，可按地域、平台过滤并设 IPM 下限，结果按 IPM 从高到低。用于找当前表现好的同主题替代素材。只检索现有素材，不生成新素材，不判断是否适合当前 campaign。",
 "creative.upload": "上传一条素材到指定 campaign，上传后进入平台审核队列。上传成功不等于审核通过，审核结果要用 creative.poll_review 查。",
 "creative.poll_review": "查询已上传素材的审核结果，立刻返回当前状态，不替你等待。审核通常需要 480 秒，未出结果时返回 pending 并告知还差多久；状态没变前重复查没有意义，先用 system.wait 等够再查。",
 "system.wait": "等待指定秒数后继续。收到 429 / 限流且带 retry_after 时，先等不少于 retry_after 秒再重试。单次上限 600 秒；要等更久说明这件事不该在本次会话里做完。不要用它等数据成熟（那是几天的量级，应当用 session.defer）。",
 "policy.search": "按关键词检索平台广告政策 / 广告法 / 内部 SOP 的条款（半结构化，可按平台和地域过滤）。每条结果带 valid_from / valid_to、expired 标记（true = 已被新版本取代）和 superseded_by；查不到返回空 hits（不是报错）。关键词匹配而非语义理解，换个说法可能查不到。",
 "insight.search_claims": "检索历史复盘沉淀的结论（如「某类素材在某地域表现更好」）。返回的是经验不是实时数据，下决策仍需用实时指标核实。每条带 status（active 现行 / superseded 已被取代，superseded_by 指向新结论 / refuted 已被推翻）、confidence 与 evidence 样本量；非 active 的结论会一并返回，因为「老结论已被推翻」本身就是要报告的信息。查不到返回空 hits（不是报错）。",
 "memory.search": "检索我们自己写下的历史记忆，按分区（lane）和主体过滤，自动剔除已过 TTL 的记录。分区：episodic 历史投放动作 / semantic 素材与受众属性 / business 优化干预效果 / risk 风控标记。涉及重复投放、频繁调预算、历史干预是否有效时必须先查。不含平台政策和团队复盘结论。",
 "memory.read": "按 record_id 读取一条记忆的完整内容，含置信度、证据引用和过期时间。只按 id 取一条，不做检索，不校验它现在还成不成立。",
 "memory.write_proposal": "提交一条记忆写入提案（不会立即入库，需经审核）。要求 confidence ≥ 0.7 且 evidence_refs 至少 2 条；写 risk 分区前必须先调 risk.check_account。episodic 分区由系统维护，不可写入。",
 "memory.invalidate": "提议把一条已失效的记忆标记为作废（如素材已下线、政策已变更），需经审核。只是提议，不立即生效，不删除原记录。",
 "memory.conflict_resolve": "两条记忆互相矛盾时提议处置方式：supersede 用新的取代旧的，merge 合并。record_ids 至少两条。只是提议，不自动执行，不判断哪条是对的。",
 "metrics.get_freshness": "查询某个指标的观测条件：这条 campaign 开投了几天、该指标通常几天收敛、累计样本量多少、预期区间。只给事实不给结论，数据现在能不能用由你判断。涉及扩量、砍量、归因结论之前先查它。",
 "mmp.get_attribution": "查 MMP（第三方归因平台）口径下的安装与回收数据，返回里带 attribution_window。它和平台后台口径（campaign.get_metrics，自归因）会有差异，最常见成因是两边归因窗口不一致，做判断前先看窗口是否一致。不含平台侧花费和曝光。",
 "playbook.get_optimization": "根据已确认的异常类型返回对应的优化方案。anomaly_type 必须是 campaign.detect_anomalies 实际返回过的类型。只给方案，不执行写动作，不判断当前数据够不够支撑。",
 "policy.get_budget_rule": "查询账户级预算调整政策：单次涨幅上限、需要审批的阈值、是否强制风控、月度总额约束。改预算前必须先查。不含平台侧广告政策条款，不做风控判断。",
 "risk.check_account": "账户风控检查：是否有风险标记、是否处于冻结 / 受限状态、是否允许提额。改预算前必须先过。只看账户风控状态，不判断金额合不合政策，不含投放指标。",
 "campaign.update_budget": "调整 campaign 的日预算。不可逆写操作，立即生效。调用前必须已查政策并通过风控。new_budget 单位是最小货币单位（分）：900 元填 90000。每个 campaign 每小时最多改 4 次，超出会被平台冻结一小时。返回只表示提交成功，最终结果要再查一次。",
 "campaign.list": "列出账户下的 campaign（id、名称、状态、日预算、产品、地域）。用户没给 campaign_id 时先用它定位。分页返回，每页最多 3 条，next_cursor 非空表示还有下一页，把它传回来继续取；需要「全部 campaign」的结论必须翻到 next_cursor 为空为止。不含效果指标。",
 "approval.create_case": "为超出自动执行范围的变更创建审批单（不会立即生效）。政策判定 requires_approval、或风控 / 记忆显示该操作过于频繁时，走审批而不是直接执行写动作。",
 "campaign.create": "新建一条 campaign 并投放。不可逆：建出来就开始花钱，删不掉。本轮没有一次成功的 approval.create_case 之前不要调用；地域扩展的正确产出是开审批单的提议，不是直接建站。跨地域铺开时每个地域建一条，每条单独确认对应地域的安全线。返回只表示提交成功，不代表已开始跑量。",
 "campaign.scale_budget": "按倍数扩量或缩量（factor=1.3 表示提到原来的 1.3 倍）。update_budget 是设成绝对值，这个表达「在现状基础上加 / 减多少」，扩量决策用这个。幅度在 ±20% 以内可以直接执行，超出必须先走 approval.create_case。扩量前必须已确认数据收敛、离安全线有空间、风控放行。",
}

# 参数说明只去重复（描述里已有的单位/含义）与冗长句；其余不动
PARAM_NEW: dict[str, dict[str, str]] = {
 "campaign.update_budget": {"new_budget": "新的日预算（分）", "client_request_id": "本次请求唯一标识"},
 "campaign.create": {"client_request_id": "本次请求唯一标识"},
 "insight.search_claims": {"active_only": "只要 active 结论；默认 false（连已被取代 / 推翻的一起给并标 status）"},
 "metrics.get_freshness": {"metric": "指标，默认 roas_d7（最慢收敛）"},
}

# 搬进 system.txt 的通用纪律（只写一次）
SYSTEM_ADD = {
 "## 检索结果的使用规则": [
  "- 查不到、返回空列表或 `not_found` 时，不要自己编数据、编政策、编经验；按实际数据判断，或用 `session.clarify` 向用户确认，或用 `approval.create_case` 转人工。关键词检索连续两次查不到就转人工，不要无限换词重试。",
  "- 安全线、政策等带有效期的资料：`valid_to` 早于今天、或标记 `expired` / `superseded` / `refuted` 的，不能当决策依据；有 `superseded_by` 就改引用新版本，没有就转人工补录并在终答说明，不要照旧线执行、也不要自己估一个数或拿行业基准顶替。",
  "- 历史经验与刚查到的实际数据矛盾时，用 `memory.conflict_resolve` 记录冲突，终答里说明两者并建议以数据为准。低置信度或小样本的结论不足以支撑写动作。",
 ],
 "## 高风险写动作规则": [
  "- 写工具必须传 `client_request_id`（自己生成、不复用）；超时后带同一个 id 重试是安全的（会去重），不确定是否已提交时先查证当前值再决定。",
 ],
}

FACT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}|\d+(?:\.\d+)?|±?\d+%")


def facts(text: str) -> set[str]:
    return {m for m in FACT_RE.findall(text) if m.lower() not in {"the", "d7", "d1"}}


def classify(sent: str) -> str:
    if re.search(r"(不含|不返回|不包含|不给|不判断|不查|不做|不生成)[^。\n]*(那在|那是|那个在|那属于|另一回事|那要)", sent):
        return "cross_ref"
    if re.search(r"不要(自己)?(编|估|猜)|不要拿.*顶替|转人工|走 clarify|不能拿它当依据|不能当依据|绝对能|绝对不能|无限换词|conflict_resolve|低置信|client_request_id|重试", sent):
        return "system_prompt"
    if re.search(r"因为|本身就是|看着唬人|噪声会淹没|对错完全是运气|那是你的判断|别混用|别当成", sent):
        return "rationale"
    return "rewritten"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只核对不写")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from syncopate.domains.adcampaign import build_domain
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    reg = build_domain().registry
    old = {t["function"]["name"]: t for t in reg.menu(None)}
    sys_txt = Path("syncopate/prompts/system.txt").read_text(encoding="utf-8")
    sys_new = "\n".join(line for sec in SYSTEM_ADD.values() for line in sec)
    report = {"tools": {}, "removed_sentences": [], "system_prompt_added_tokens": len(tok.encode(sys_new))}
    tot_old = tot_new = 0
    errors = []
    for name, spec in old.items():
        f = spec["function"]; od = f["description"]
        nd = NEW.get(name, od)
        params = f.get("parameters", {}).get("properties", {}) or {}
        pdesc = " ".join(PARAM_NEW.get(name, {}).get(k, v.get("description", "")) for k, v in params.items())
        # ① 硬事实不丢
        pool = nd + " " + pdesc + " " + sys_txt + " " + sys_new + " " + name
        # 交叉引用里的工具名不算丢失：菜单是全量的，模型看得见那个工具（只豁免注册表里真实存在的名字）
        xref_names = set(old) | {"benchmark"}
        missing = sorted(x for x in facts(od) if x not in pool and x not in xref_names)
        if missing:
            errors.append(f"{name}: 硬事实丢失 {missing}")
        # ③ 删句登记
        for s in re.split(r"(?<=[。；\n])", od):
            s = s.strip(" ·★*")
            if s and s not in nd:
                report["removed_sentences"].append({"tool": name, "bucket": classify(s), "sentence": s})
        # ② token
        t_old = len(tok.encode(json.dumps(spec, ensure_ascii=False)))
        spec2 = json.loads(json.dumps(spec, ensure_ascii=False)); spec2["function"]["description"] = nd
        for k, v in PARAM_NEW.get(name, {}).items():
            spec2["function"]["parameters"]["properties"][k]["description"] = v
        t_new = len(tok.encode(json.dumps(spec2, ensure_ascii=False)))
        tot_old += t_old; tot_new += t_new
        report["tools"][name] = {"tok_old": t_old, "tok_new": t_new, "desc_old": od, "desc_new": nd}
    report["total_old"] = tot_old; report["total_new"] = tot_new
    print(f"[trim] 工具块 {tot_old} → {tot_new}（省 {tot_old - tot_new}；system.txt +{report['system_prompt_added_tokens']}）")
    from collections import Counter
    print("[trim] 删句归筐：", dict(Counter(r["bucket"] for r in report["removed_sentences"])))
    if errors:
        print("🔴 " + "\n🔴 ".join(errors))
        return 2
    Path("_audit/v15_w2").mkdir(parents=True, exist_ok=True)
    json.dump(report, open("_audit/v15_w2/tool_desc_trim.json", "w"), ensure_ascii=False, indent=1)
    if args.check:
        return 0
    # ── 就地改源文件（ast 定位字面量）──
    changed = 0
    for py in sorted(TOOLS_DIR.glob("*.py")):
        src = py.read_text(encoding="utf-8"); tree = ast.parse(src); lines = src.splitlines(keepends=True)
        edits = []   # (start_line, start_col, end_line, end_col, new_text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "tool"):
                    continue
                kw = {k.arg: k.value for k in dec.keywords}
                nm = kw.get("name")
                if not isinstance(nm, ast.Constant) or nm.value not in NEW:
                    continue
                d = kw["description"]
                edits.append((d.lineno, d.col_offset, d.end_lineno, d.end_col_offset, json.dumps(NEW[nm.value], ensure_ascii=False)))
                # 参数说明
                for k, v in PARAM_NEW.get(nm.value, {}).items():
                    for sub in ast.walk(kw["parameters"]):
                        if isinstance(sub, ast.Dict):
                            for kk, vv in zip(sub.keys, sub.values):
                                if isinstance(kk, ast.Constant) and kk.value == k and isinstance(vv, ast.Dict):
                                    for k2, v2 in zip(vv.keys, vv.values):
                                        if isinstance(k2, ast.Constant) and k2.value == "description":
                                            edits.append((v2.lineno, v2.col_offset, v2.end_lineno, v2.end_col_offset, json.dumps(v, ensure_ascii=False)))
        # ⚠️ ast 的 col_offset 是 **UTF-8 字节偏移**，不是字符偏移——中文描述按字符切会切错位
        #   （第一版就是这么把 external_tools.py 切坏的），所以一律按字节切。
        blines = [ln.encode("utf-8") for ln in lines]
        for (l1, c1, l2, c2, new) in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
            before = b"".join(blines[: l1 - 1]) + blines[l1 - 1][:c1]
            after = blines[l2 - 1][c2:] + b"".join(blines[l2:])
            bsrc = before + new.encode("utf-8") + after
            blines = bsrc.splitlines(keepends=True)
            changed += 1
        py.write_bytes(b"".join(blines))
    # ── system.txt：并入既有章节（不新开节）──
    for sec, add in SYSTEM_ADD.items():
        i = sys_txt.index(sec)
        j = sys_txt.index("\n## ", i + 1)
        block = sys_txt[i:j].rstrip("\n") + "\n" + "\n".join(add) + "\n"
        sys_txt = sys_txt[:i] + block + sys_txt[j:]
    Path("syncopate/prompts/system.txt").write_text(sys_txt, encoding="utf-8")
    print(f"[trim] 改写 {changed} 处字面量 · system.txt 并入 {sum(len(v) for v in SYSTEM_ADD.values())} 条 · 报告 _audit/v15_w2/tool_desc_trim.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
