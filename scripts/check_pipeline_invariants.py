#!/usr/bin/env python
"""管线前提检查 —— 「两个东西应当相同 / 某集合应当完整」这类断言的集中执行处。

    python scripts/check_pipeline_invariants.py                     # 全跑
    python scripts/check_pipeline_invariants.py --only merge rank   # 只跑某几组

    退出码  0 = 全过
            1 = **有判据被违反**（真问题，有行动）
            2 = **只有无法判定的**（探针自己没跑成，多半是环境/依赖 —— 不等于通过，
                                    但也不是判据查出来的问题。见 `run_checks` 的注释）

★★★ 为什么要有这个文件（2026-08-18，E21 之后建）

E21（三个 trainer rank 没有同步梯度）不是"查出来"的，是**一句顺手写的断言炸出来的**：
「DDP 下各 rank 的 LoRA 应该相同」。那条断言原本只是为了防止读错分片。

⇒ 由此定的纪律：**凡是"我假设 X 成立"的地方，把它写成断言。**
⇒ 并且断言要写在「**两个东西应当相同**」的地方，而不是「这个数应该在某范围里」——
   前者非黑即白、不需要阈值、不会因基线漂移而失效；后者就是本项目一直在踩的
   「空门槛 / 门槛太宽」（见 .claude/memory/blank-thresholds-are-not-passes.md）。

⚠️ 每条检查都必须**有能力失败**。加新检查时，先在一个已知会违反的输入上确认它会红，
   否则你只是加了一个永远绿的装饰（同一条记忆的第 ⑤ 点：判据为空时先怀疑解析器）。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR
from syncopate.train.rollout_loop import chat_template_ids

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[str, str, object]] = []


def check(group: str, name: str):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def _hf_tensor(model_dir: Path, key: str):
    from safetensors import safe_open
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            if key in fh.keys():
                return fh.get_tensor(key)
    return None


def _lora_delta(adapter_dir: Path, weight_key: str):
    """返回该层的 ΔW_eff = (alpha/r)·B@A；adapter 里没有这一层就返回 None。"""
    from safetensors import safe_open
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    stem = weight_key[: -len(".weight")]
    for f in sorted(adapter_dir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            ka = [k for k in fh.keys() if stem + ".lora_A" in k]
            if not ka:
                continue
            A = fh.get_tensor(ka[0]).float()
            B = fh.get_tensor(ka[0].replace("lora_A", "lora_B")).float()
            return scale * (B @ A)
    return None


PROBE_LAYERS = [
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.20.self_attn.v_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
]

# ── 组 merge：「合并」到底有没有把增量放进权重 ────────────────────────────────

@check("merge", "合并模型的权重必须真的不同于它的基座")
def _merge_actually_changed(log):
    """🔴 2026-08-18 抓到过：models/Qwen3-4B-rl-v13-s110 与 SFT 模型逐位相同。

    根因不是 bug，是 verl 的 LoRA merger **本来就**产出「未改的基座 + 独立 adapter」。
    危险在于 launch_rl 没有加载 adapter 的入口 ⇒ 拿它当下一轮 RL 的起点会静默丢掉整轮。
    """
    pairs = [("models/Qwen3-4B-sft-v13-e1", STUDENT_MODEL, "SFT 合并模型")]
    ok = True
    for merged, base, label in pairs:
        md, bd = ROOT / merged, ROOT / base
        if not md.exists() or not bd.exists():
            log(f"  ⏭ {label}：模型不在本机，跳过")
            continue
        for key in PROBE_LAYERS:
            m, b = _hf_tensor(md, key), _hf_tensor(bd, key)
            if m is None or b is None:
                continue
            d = (m.float() - b.float()).norm().item()
            if d == 0.0:
                log(f"  🔴 {label} 的 {key} 与基座**逐位相同** ⇒ 增量没进权重")
                ok = False
            else:
                log(f"  ✅ {label} {key}: ‖merged−base‖={d:.6f}")
            break
    return ok


@check("merge", "凡是目录里带 lora_adapter/ 的模型，都不是「合并后的模型」")
def _merged_dir_has_no_adapter(log):
    """判据：一个可以直接当 RL 起点的模型目录，不应该同时存在 lora_adapter/。
    存在就说明增量还在外面 —— 而 launch_rl 读不到它。"""
    ok = True
    for d in sorted((ROOT / "models").glob("*")):
        if not d.is_dir():
            continue
        if (d / "lora_adapter").exists():
            log(f"  🔴 {d.name}/ 里有 lora_adapter/ ⇒ 主权重**不含**增量，不能当 RL 起点")
            ok = False
    if ok:
        log("  ✅ models/ 下没有「看起来已合并、其实没合并」的目录")
    return ok


@check("merge", "bf16 存储必须留得住增量（增量太小就不该合并）")
def _bf16_can_hold_delta(log):
    """★ 2026-08-18 量出来的硬约束：合并的损失来自**存储精度**，不是累加精度。

        kept     = quantize_bf16(W + Δ) − W          实际留在权重里的增量
        保真残差 = ‖kept − Δ‖ / ‖Δ‖                  ★ 判据用这个
        幅度比   = ‖kept‖ / ‖Δ‖                      仅供参考（幅度对不代表方向对）

    ⚠️ 两者必须分开看：RL 那一档幅度比 0.68 看着还行，**保真残差却是 0.87**
       —— 幅度留住了，方向被舍入噪声打乱了。**只报幅度比会得出错误的安心结论。**

    实测（layers.0.self_attn.k_proj）：
        SFT  Δ 占基座 0.42%   ⇒ 保真残差 0.36
        RL   Δ 占基座 0.056%  ⇒ 保真残差 0.87   ⇒ **合并会毁掉它，必须保持 adapter 形态**

    ★ 损失来自**存储精度**，不是累加精度 —— 在 fp32 里相加再存 bf16 一样丢。
    """
    import torch
    ok = True
    cases = [
        ("SFT", "models/Qwen3-4B", "checkpoints/sft/v13/epoch1", 0.5),  # 残差 ≤0.5 才算合并有意义
        ("RL", "models/Qwen3-4B-sft-v13-e1", "models/Qwen3-4B-rl-v13-s110/lora_adapter", None),
    ]
    for label, base_rel, ad_rel, floor in cases:
        base_dir, ad_dir = ROOT / base_rel, ROOT / ad_rel
        if not base_dir.exists() or not ad_dir.exists():
            log(f"  ⏭ {label}：产物不在本机，跳过")
            continue
        key = PROBE_LAYERS[0]
        W = _hf_tensor(base_dir, key)
        d = _lora_delta(ad_dir, key)
        if W is None or d is None:
            log(f"  ⏭ {label}：探针层不在 adapter 里，跳过")
            continue
        W = W.float()
        kept = (W + d).to(torch.bfloat16).float() - W
        resid = ((kept - d).norm() / d.norm()).item()      # ★ 判据
        mag = (kept.norm() / d.norm()).item()
        rel = (d.norm() / W.norm()).item()
        verdict = "ℹ️" if floor is None else ("✅" if resid <= floor else "🔴")
        log(f"  {verdict} {label}: Δ 占基座 {rel*100:.4f}%  →  保真残差 {resid:.2f}"
            f" · 幅度比 {mag:.2f}" + ("" if floor is None else f"（门槛 残差≤{floor}）"))
        if floor is not None and resid > floor:
            ok = False
        if floor is None and resid > 0.5:
            log("     ⇒ 残差过半 ⇒ **不要合并**，保持 adapter 形态（这是记录不是失败）")
    return ok


# ── 组 truncation：截断的三种成因不许再合并 ────────────────────────────────

@check("truncation", "★ 每个设 truncated 的出口都必须同时写 truncation_reason")
def _truncation_reason_always_set(log):
    """🔴 2026-08-18：`truncated` 有四个出口、三种成因，而**修法方向相反**
    （加 token 预算 / 截断 observation / 缩链路）。此前合并成一个布尔值
    ⇒ 数据里根本不存在这个区分，只能事后猜 —— 而按错的假设猜过一次，结论整个反了。

    ⚠️ 判据只能是**源码扫描**：「将来有人加了第五个出口却忘了标因」这件事
    没有行为可测（它还没发生），只有扫源码看得见。同 M9 §6 那条。
    """
    import re as _re
    ok = True
    for rel in ["syncopate/train/rollout_loop.py", "syncopate/core/runner.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(src):
            if "truncated = True" not in line or line.strip().startswith("#"):
                continue
            # 紧随其后的 3 行内必须出现 truncation_reason（允许夹注释）
            window = "\n".join(src[i + 1: i + 5])
            if "truncation_reason" not in window:
                log(f"  🔴 {rel}:{i + 1} 设了 truncated 但**没标 truncation_reason**")
                ok = False
    if ok:
        log("  ✅ 所有设 truncated 的出口都标了原因")

    # 反过来也查：cap 不许再用裸 truncated 当判据
    rules_src = (ROOT / "syncopate/domains/adcampaign/rules.py").read_text(encoding="utf-8")
    if _re.search(r"if not trajectory\.truncated:", rules_src):
        log("  🔴 rules.py 里还有 cap 拿裸 `truncated` 当判据 —— 它有三种成因，报错会编数字")
        ok = False
    else:
        log("  ✅ 没有 cap 再拿裸 truncated 当判据")
    return ok


# ── 组 docfacts：操作文档里不许留未标注的陈旧断言 ─────────────────────────

# (标签, 正则, 现在为什么是错的)。⚠️ 只扫**操作文档**（照着做的），
# 不扫 archive / 设计文档 / E 报告 —— 那些是记录，「推翻的预期不删」是明文约定。
STALE_ASSERTIONS = [
    # 只查 `EVAL 278` —— 它**任何版本都不对**。`EVAL 254`(v12) / `EVAL 198`(v11)
    # 是那两版的真实数字、按版本标注了，不是陈旧。第一版把它们也扫进来 ⇒ 又一次「判据太宽会错杀」。
    ("三桶数字", r"EVAL 278", "任何版本都不对；v13 实测 EVAL 343"),
    ("两堵墙", r"lr 被夹在两堵墙", "那堵墙是 E22 的 bug，不是异步的代价"),
    ("分池没接上", r"动态分池在 fully_async 下没接上|fully_async 下动态分池没接上", "2026-08-17 已接上"),
    ("本机无产物", r"本机上没有任何训练产物", "已重训，产物在"),
    ("旧长度默认", r"--max-prompt-length 3584", "已统一成 5120/2048"),
]
# 上下文里有这些词 ⇒ 是在描述历史，不是在规定现在。
# ⚠️ 这一条本身就是「判据太宽会错杀」的补丁：第一版没有它，
#    把「此前是 X，已改成 Y」也判成了陈旧断言（见 memory blank-thresholds ⑦）。
# ⚠️ 这张表每漏一个词就误伤一次 —— 2026-08-18 补过三轮。
#    ★ 更稳的做法是要求**显式标记**（⛔）而不是猜词，但那要改所有描述点；
#      折中：词表 + ⛔，且**误伤时优先补词表，不要放宽正则**。
_HIST = __import__("re").compile(
    r"此前|原(写法|猜想|版|值|来|文)|旧(版|值|口径|的|文档)|改成|已改|已接上|作废|⛔"
    r"|推翻|教训|留作|留在这里|过期|重写|修复前|核心叙事|不是当前|定格|快照|历史")

# ⚠️ infra 2026-08-19 改组：ONBOARDING/00-INFRA-HANDOFF → 00-START/01-TASKS/02-DECISIONS。
#    路径不存在时会被静默跳过（`p.exists()`）—— 所以改名必须跟着改这里，否则扫描范围悄悄缩水。
OPERATIONAL_DOCS = ["docs/syncopate", "docs/infra_exp/00-START.md",
                    "docs/infra_exp/01-TASKS.md", "docs/infra_exp/02-DECISIONS.md"]


@check("docfacts", "★ 操作文档里不许留**未标注**的陈旧断言")
def _no_unmarked_stale_assertions(log):
    """🔴 Chaoyu 2026-08-18：**docs 里的当前文档必须所有事实都正确、0 干扰信息 ——
    没做就是没做，等待做就是等待做。**

    ⚠️ 但这和本项目另一条约定冲突：「设计文档里**推翻的预期不删**」。
    ⇒ 解法不是二选一，是**分开放**：
        操作文档（照着做的）→ 必须 0 错误，本判据扫它
        历史记录（回头看的）→ `docs/archive/` 与设计文档 / E 报告，标清楚是快照
    """
    import re as _re
    ok = True
    targets = []
    for t in OPERATIONAL_DOCS:
        p = ROOT / t
        targets += sorted(p.rglob("*.md")) if p.is_dir() else ([p] if p.exists() else [])
    for doc in targets:
        lines = doc.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            for tag, pat, why in STALE_ASSERTIONS:
                if _re.search(pat, line) and not _HIST.search("\n".join(lines[max(0, i - 3):i + 3])):
                    log(f"  🔴 {doc.relative_to(ROOT)}:{i + 1} [{tag}] {why}")
                    log(f"     {line.strip()[:78]}")
                    ok = False
    if ok:
        log(f"  ✅ 扫了 {len(targets)} 份操作文档，没有未标注的陈旧断言")
    return ok


# ── 组 quarantine：作废的数字不许在没有警示的情况下出现 ───────────────────

QUARANTINE_DOC = "docs/syncopate/21-invalidated-numbers.md"
QUARANTINE_BANNER = "21-invalidated-numbers"     # 指回链接
# ★ 就地标记：出现作废数字的**那一行**必须带它。见下面那条判据的长注释。
QUARANTINE_MARK = "⛔"


def _quarantined_tokens() -> list[tuple[str, str]]:
    """从 21 号文档的 ```quarantine``` 块里读作废清单 —— **唯一来源**。

    ⚠️ 刻意不在这里另立一份名单：同一件事两份实现，两份最后都会不准
    （本项目的 `syncopate-docs-map` 明确记过这条）。
    """
    text = (ROOT / QUARANTINE_DOC).read_text(encoding="utf-8")
    block = text.split("```quarantine", 1)
    if len(block) < 2:
        return []
    body = block[1].split("```", 1)[0]
    out = []
    for line in body.splitlines():
        if "|" not in line:
            continue
        token, what = (x.strip() for x in line.split("|")[:2])
        if token:
            out.append((token, what))
    return out


@check("quarantine", "★ 每个作废数字必须**就地**标记（不是靠文档顶部的横幅）")
def _quarantined_numbers_are_flagged(log):
    """🔴 2026-08-18：两个基石级 bug（E21 梯度不同步 · 权重从没推给 rollout）
    把 08-14 至 08-18 之间所有 RL 训练的实测数字都污染了。

    ⚠️ **作废的数字被将来引用，比没有数字更危险** —— 它看起来精确、有出处、有上下文，
    而它量的是一个坏掉的系统。本项目已经为此付过一次钱
    （「M7 已验收」被下游引用了两天，见 blank-thresholds-are-not-passes）。

    ⇒ 判据不是"把数字删掉"（删了就没人知道当初错在哪），
      而是**每个作废数字必须自己喊出来**。集合包含型判据，不需要阈值。

    ⛔⛔ **2026-08-19 改了标记的位置：从"文档顶部横幅"下沉到"数字行内"。**

    原设计是每份含作废数字的文档顶部挂一条 ⛔ 横幅。问题是它长成了**17 份一模一样的横幅**
    ——而**假警报会训练人忽略这条判据，那比没有判据更糟**（守则③）。
    更要命的是它**不精确**：横幅在第 3 行，数字在第 160 行，
    读到数字的人根本不会把两者联系起来。

    ⇒ 改成 `⛔(21)` **就地标在那一行**：警告出现在**数字旁边**，而不是 200 行之外。

    ⚠️⚠️ **不要误读成"作废解除了"**：截至 2026-08-19，
      21 号清单 **22 条里 21 条仍是「待重测」** —— 基石 bug 修好了，
      但那些数字**是在坏系统上量的，不会因为 bug 被修而变有效**。
      这次改的是**警告的形式**，不是警告的内容。
    """
    tokens = _quarantined_tokens()
    if not tokens:
        log("  ⛔ 21 号文档里读不到 ```quarantine``` 块 —— 判据自己坏了")
        return False
    log(f"  作废清单共 {len(tokens)} 条（来源：{QUARANTINE_DOC}）")

    scope = sorted(set((ROOT / "docs").rglob("*.md")))
    ok = True
    offenders: dict[str, list[str]] = {}
    for doc in scope:
        rel = str(doc.relative_to(ROOT))
        if rel == QUARANTINE_DOC or ".ipynb_checkpoints" in rel:
            continue
        text = doc.read_text(encoding="utf-8")
        hits = [t for t, _ in tokens if t in text]
        if not hits:
            continue
        # ★ 逐行查：出现作废数字的**那一行**必须自己带标记
        # ⚠️ archive/ 整个目录已有总声明「里面的一切都不是当前状态」⇒ 不重复标
        if "/archive/" in rel:
            continue
        # ⚠️ infra_exp/ 是另一条线的文档（`交界`）⇒ **只报告不判红**。
        #   主线单方面改他们的文档，比"标记不一致"更糟（两条线共用一个仓库）。
        if "/infra_exp/" in rel:
            todo = [str(i) for i, line in enumerate(text.splitlines(), 1)
                    if any(t in line for t, _ in tokens)
                    and QUARANTINE_MARK not in line and "作废" not in line]
            if todo:
                log(f"  ℹ️ {rel}: {len(todo)} 行还没就地标记 —— **infra 的文档，交给他们**")
            continue
        unmarked = []
        for i, line in enumerate(text.splitlines(), 1):
            if any(t in line for t in (t for t, _ in tokens)) and QUARANTINE_MARK not in line:
                # 这一行本身在解释"某数字已作废"就不算违反
                if "作废" in line or QUARANTINE_BANNER in line:
                    continue
                unmarked.append(f"{i}")
        if unmarked:
            offenders[rel] = unmarked
            ok = False
    if offenders:
        for rel, lines in sorted(offenders.items()):
            log(f"  🔴 {rel}: {len(lines)} 行含作废数字但**没有就地标记** —— 行 {lines[:6]}")
        log(f"     ⇒ 在那一行末尾加 `{QUARANTINE_MARK}(21)`。"
            f"**标在数字旁边**，不是文档顶部 —— 横幅隔 200 行没人会联系起来")
    else:
        log(f"  ✅ 扫了 {len(scope)} 份文档，作废数字都**就地**标了")
    return ok


# ── 组 budget：训练与评测必须跑在同一份长度预算上 ──────────────────────────

@check("budget", "★ 训练与评测的 rollout 契约必须一致（长度预算 + 采样参数）")
def _train_eval_budget_match(log):
    """🔴 2026-08-18 抓到：评测硬编码 5120/2048，而训练传 3584/1536 ——
    **宽 43% / 33%**。后果：同一条轨迹在评测里跑得完、在训练里可能被截断
    ⇒ 两边跑在不同的输入分布上，而我们拿评测分数去判训练有没有用。

    判据是「两个数应当相等」，不需要阈值。
    """
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
    log(f"  共用预算（rollout_budget.py）: prompt={MAX_PROMPT_LENGTH} response={MAX_RESPONSE_LENGTH}")

    ok = True
    # ① 源码里不该再有硬编码的长度 / 采样常量
    import re as _re
    from syncopate.train.rollout_budget import SAMPLING_TOP_K, SAMPLING_TOP_P
    log(f"  共用采样参数: top_p={SAMPLING_TOP_P} top_k={SAMPLING_TOP_K}"
        "（对齐方向：**评测跟训练**，见 rollout_budget.py）")
    for rel in ["syncopate/train/eval_local.py", "syncopate/train/staleness.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        hard = _re.findall(r"max_response_length=(\d+)", src)
        # ⚠️ 只查采样参数的**字面量赋值**；注释里引用旧值（0.95/20）是刻意保留的历史记录
        hard += [m for m in _re.findall(r"^\s+top_[pk]=([\d.]+)[,\s]", src, _re.M)]
        # ★ 2026-08-19 补：**单轮生成上限**也是契约的一部分（E23「评测跟训练」）。
        #   256 的字面量默认曾让 base 评测截断率 0.203、E27 B 臂整跑作废。
        hard += [m for m in _re.findall(r"max_new_tokens\s*=\s*(\d+)\b", src)
                 if m not in ("0",)]
        if hard:
            log(f"  🔴 {rel} 里还有硬编码的长度/采样常量：{hard}")
            log("     ⇒ 单轮上限/长度一律从 rollout_budget 推（MAX_RESPONSE_LENGTH）")
            ok = False
    # ② 最近一次真跑用的值（overrides.yaml 是"上次到底怎么跑通的"唯一可信来源）
    ovs = sorted((ROOT / "outputs").glob("*/*/.hydra/overrides.yaml"),
                 key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    for ov in ovs:
        txt = ov.read_text(encoding="utf-8")
        got = {}
        for key, want in (("data.max_prompt_length", MAX_PROMPT_LENGTH),
                          ("data.max_response_length", MAX_RESPONSE_LENGTH)):
            for line in txt.splitlines():
                if line.strip().lstrip("- ").startswith(key + "="):
                    got[key] = int(line.split("=")[1])
        stamp = "/".join(ov.parts[-4:-2])
        bad = [k for k, v in got.items()
               if v != (MAX_PROMPT_LENGTH if "prompt" in k else MAX_RESPONSE_LENGTH)]
        if bad:
            # ⚠️⚠️ 旧版这里打的是 🟡 且附一句「这是修复**之前**的跑，属预期」——
            #   而那句话是**无条件**的：它没有任何东西能判断"之前/之后"。
            #   2026-08-19 02:55 有一跑（修复**之后**）用了 3584/1536，
            #   被这句话原样放过了。判据写满了、真的在跑、也真的报了个数，
            #   只是它量的不是我们以为的那件事（`19 §2.1` 第二层）。
            # ⇒ 改成 🔴。它会一直红到**下一次干净的跑**为止 —— 这不是误报，
            #   而是一句真话：「最近一次跑不在当前契约上，别拿它当参照」。
            log(f"  🔴 最近一次跑（{stamp}）用的是 {got} ≠ 共用预算")
            log("     ⇒ 要么是修复前的历史跑（那就**别拿它当任何比较的一端**），")
            log("       要么是有人绕开固定管线手起的 —— 两种都不该沉默放过")
            log("     ⇒ 下一次从固定脚本起的跑会让这条自动转绿")
            ok = False
        else:
            log(f"  ✅ 最近一次跑（{stamp}）与共用预算一致")
    # ③ 见下面独立的 contract 组：**脚本一律不许显式传契约参数**
    #    （此前这里只比"传的值对不对"，那挡不住"值现在对、下次改契约时漂掉"）
    return ok


# ══════════════════════════════════════════════════════════════════════════
# 组 contract：**跑训练只能走固定管线，参数不许各抄一份**
# ══════════════════════════════════════════════════════════════════════════

# ★ 契约参数 → 命令行开关的映射。**这是唯一的耦合点**，加契约参数时改这里。
#   值本身不在这里写 —— 唯一来源是 `syncopate/train/rollout_budget.py`。
CONTRACT_FLAGS = (
    "--max-prompt-length", "--max-response-length",
    "--temperature", "--top-p", "--top-k",
)

# ⚠️ 逃生口：确实要做契约参数的 A/B 时，在**同一行或上一行**写这个标记 + 理由。
#   为什么留：判据太宽会制造假警报，而假警报会训练人忽略这条判据（守则③）。
#   为什么要求写理由：让"我就改一下"变成一个**留痕**的动作。
CONTRACT_OVERRIDE_MARK = "CONTRACT-OVERRIDE:"


@check("contract", "★ 跑训练的脚本一律不许显式传契约参数（长度预算 + 采样参数）")
def _no_script_overrides_contract(log):
    """★★ 为什么判据是"一律不许传"，而不是"传的值要对"

    2026-08-18 的实况：15 个启动脚本**全都**走 `launch_rl`（唯一入口这条纪律是守住的），
    但**每个脚本各自抄了一份参数**。于是抄着抄着漂成了两套：
    11 个是 5120/2048，4 个还停在 3584/1536。

    ⇒ 关键在于：那 11 个当时**是对的**。旧判据只比"值对不对"，所以它们一路绿灯 ——
      而它们的问题不是值错，是**存在一份副本**。副本在下一次改契约时必然漏掉几个。
    ⇒ 所以判据要写在「**只该有一份**」上，而不是「这一份的值对不对」上（守则①）。

    ⚠️ 这条是源码扫描：「将来有人又抄一份」这件事**还没发生**，没有行为可测，
       只有扫源码看得见（同 `truncation` 组那条）。
    """
    import re as _re
    ok = True
    pat = _re.compile("|".join(_re.escape(f) for f in CONTRACT_FLAGS))
    offenders, waived = [], []
    for sh in sorted((ROOT / "scripts").glob("*.sh")):
        lines = sh.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#") or not pat.search(line):
                continue
            context = line + (lines[i - 1] if i else "")
            (waived if CONTRACT_OVERRIDE_MARK in context else offenders).append(
                f"{sh.name}:{i + 1}")
    if offenders:
        log(f"  🔴 {len(offenders)} 处脚本显式传了契约参数：{', '.join(offenders[:8])}"
            + (" …" if len(offenders) > 8 else ""))
        log(f"     ⇒ 删掉即可 —— launch_rl / eval_local 的默认值已从 "
            f"rollout_budget.py 取，不传就是对的")
        log(f"     ⇒ 确实要做契约参数的 A/B，在同行或上一行写 "
            f"`# {CONTRACT_OVERRIDE_MARK} <理由>`")
        ok = False
    else:
        log(f"  ✅ 扫了 {len(list((ROOT / 'scripts').glob('*.sh')))} 个脚本，"
            f"没有一处显式传契约参数")
    if waived:
        log(f"  ℹ️ {len(waived)} 处已显式声明覆盖（留痕，不算违反）：{', '.join(waived)}")
    return ok


@check("contract", "★ SFT 的长度上限必须来自 rollout 契约，且当前版本数据一条都不超")
def _sft_length_is_contractual(log):
    """🔴 2026-08-19 抓到（虚惊但三个隐患都真）：sft.py 曾是 `--max-length` 默认 4096
    + 静默 `[:max_length]` 切片，而 v13 数据 92.6% 超过 4096 —— 实跑靠手传 6656
    侥幸躲过，`08` 文档示范命令的 6144 会无声截掉 46 条。
    与 RL 侧 prompt 截断（defer 崩塌归因翻案）同族：截掉的没人看见、没人计数。

    判据两半：① 源码里不许再出现独立的长度参数（守则⑨：该一致的值这里就不该有）
              ② 当前 DATA_VERSION 的 SFT parquet 一条都不许超过契约预算
    """
    ok = True
    # ① 源码扫描。⚠️ 只查 add_argument 调用 —— 第一版查裸字符串 "--max-length"，
    #    把注释里的历史记录也扫进来当场误伤（守则③：判据太宽会制造假警报）。
    src = (ROOT / "syncopate/train/sft.py").read_text(encoding="utf-8")
    if 'add_argument("--max-length"' in src:
        log("  🔴 sft.py 又长出了 --max-length 参数 —— 长度上限只能从 rollout_budget 推")
        ok = False
    if "SFT_MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" not in src:
        log("  🔴 sft.py 里找不到 SFT_MAX_LENGTH = prompt + response 的契约推导")
        ok = False
    if ok:
        log("  ✅ sft.py 的长度上限来自 rollout_budget（无独立参数、无静默切片）")
    # ② 数据层
    from syncopate.pipeline.split import DATA_VERSION
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
    budget = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
    import pyarrow.parquet as pq
    for name in ("train", "val"):
        p = ROOT / f"data/sft/{DATA_VERSION}/{name}.parquet"
        if not p.exists():
            log(f"  ⏭ {p.relative_to(ROOT)} 不存在")
            continue
        lens = [len(x) for x in pq.read_table(p, columns=["input_ids"])["input_ids"].to_pylist()]
        over = sum(1 for n in lens if n > budget)
        if over:
            log(f"  🔴 {p.relative_to(ROOT)}: {over}/{len(lens)} 条超过预算 {budget}"
                f"（最长 {max(lens)}）—— sft.py 会硬报错拒绝启动，先修数据或契约")
            ok = False
        else:
            log(f"  ✅ {p.relative_to(ROOT)}: {len(lens)} 条全部 ≤ {budget}（最长 {max(lens)}）")
    return ok


@check("contract", "★ 起训练只能走固定入口（sft / launch_rl），不许另起一套")
def _only_canonical_entrypoints(log):
    """`08 §4`：入口只有两个，不许再写第三个 —— 临时脚本会**静默地和主路径长出差异**。

    ⚠️ 这条此前只写在文档里，没有任何东西在执行它。
    """
    import re as _re
    ok = True
    # 直接调训练框架 = 绕过了我们自己的入口（那里挂着守卫、默认值、契约）
    bypass = _re.compile(r"python[0-9.]*\s+-m\s+verl\.|torchrun.*verl|"
                         r"python[0-9.]*\s+.*verl/trainer/")
    bad = []
    for sh in sorted((ROOT / "scripts").glob("*.sh")):
        for i, line in enumerate(sh.read_text(encoding="utf-8").splitlines()):
            if line.lstrip().startswith("#"):
                continue
            if bypass.search(line):
                bad.append(f"{sh.name}:{i + 1}")
    if bad:
        log(f"  🔴 {len(bad)} 处直接调了训练框架，绕过 launch_rl：{', '.join(bad)}")
        log("     ⇒ launch_rl 上挂着契约默认值、起手断言、守卫 —— 绕过去这些全都不生效")
        ok = False
    else:
        log("  ✅ 没有脚本绕过 syncopate.train.sft / syncopate.train.launch_rl")
    return ok


# ── 组 rank：「多个副本应当相同」───────────────────────────────────────────

@check("rank", "ckpt 的各 rank LoRA 必须逐位相同（DDP 副本）")
def _ckpt_ranks_identical(log):
    """🔴 E21 就是这条炸出来的。只查**未瘦身**的 ckpt（瘦身会把三份压成一份）。"""
    import torch
    ok = True
    cands = [a for a in (ROOT / "checkpoints/grpo").glob("*/global_step_*/actor")
             if len(list(a.glob("model_world_size_*_rank_*.pt"))) >= 2]
    if not cands:
        # ★ 全量分片被回收之后，判据改读**指纹**（extract_ckpt_fingerprint.py 的产物）。
        #   ⇒ 「为了省空间把证据删了」这件事，被指纹这一步挡住了。
        fps = sorted((ROOT / "checkpoints/grpo").glob("*/global_step_*/actor/rank_fingerprint.json"))
        if not fps:
            log("  ⏭ 既没有多 rank ckpt 也没有指纹可查")
            return True
        def _run_time(f: Path) -> float:
            d = json.loads(f.read_text())
            if d.get("run_mtime"):
                return float(d["run_mtime"])
            marker = f.parents[2] / "dispatched.jsonl"     # 这一跑的产物，没被清理动过
            return marker.stat().st_mtime if marker.exists() else 0.0
        newest_fp = max(fps, key=_run_time)
        d = json.loads(newest_fp.read_text())
        tag = f"{newest_fp.parents[2].name}/{newest_fp.parents[1].name}"
        pw = d.get("pairwise", {})
        bad = [k for k, v in pw.items() if not v.get("identical")]
        if bad:
            log(f"  🔴 {tag}（读指纹）: rank 对 {bad} 不一致 ⇒ 梯度没同步（E21）")
            return False
        log(f"  ✅ {tag}（读指纹）: {len(pw)} 对 rank 全部逐位相同")
        older = [f for f in fps if f is not newest_fp]
        if older:
            n_bad = sum(1 for f in older
                        if any(not v.get("identical")
                               for v in json.loads(f.read_text()).get("pairwise", {}).values()))
            log(f"  ℹ️ 另有 {len(older)} 份历史指纹，其中 {n_bad} 份是 E21 修复前的（注定红）")
        return True
    # ⚠️ 只判**最新**那个：历史产物是 E21 修复之前跑的，注定红，红了也没有行动可做。
    #    判据要指向「现在还坏不坏」，不是「以前坏过」。
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    for actor in [newest]:
        ranks = sorted(actor.glob("model_world_size_*_rank_*.pt"))
        found = True
        sd0 = torch.load(ranks[0], map_location="cpu", weights_only=False)
        sd1 = torch.load(ranks[1], map_location="cpu", weights_only=False)
        keys = [k for k in sd0 if "lora_" in k]
        bad = [k for k in keys if not torch.equal(sd0[k], sd1[k])]
        tag = f"{actor.parent.parent.name}/{actor.parent.name}"
        if bad:
            log(f"  🔴 {tag}: {len(bad)}/{len(keys)} 个 LoRA 张量跨 rank 不同 ⇒ 梯度没同步（E21）")
            ok = False
        else:
            log(f"  ✅ {tag}: {len(keys)} 个 LoRA 张量在 rank0/rank1 上逐位相同")
        del sd0, sd1
    older = [a for a in cands if a is not newest]
    if older:
        log(f"  ℹ️ 另有 {len(older)} 个更早的多 rank ckpt 未查"
            "（E21 修复前的产物注定红，红了也没有行动可做）")
    return ok


@check("rank", "任何「读一个 rank 就代表全部」的代码，必须先比过两个 rank")
def _rank_readers_have_guard(log):
    """源码扫描。行为测试测不出「不可达的假设」，只能扫源码（同 M9 §6 那条）。"""
    targets = {
        "scripts/rl_ckpt_to_adapter.py": True,
        "scripts/rl_ckpt_drift.py": True,
        "scripts/prune_rl_ckpts.py": True,
    }
    ok = True
    for rel in targets:
        p = ROOT / rel
        if not p.exists():
            log(f"  ⏭ {rel} 不存在")
            continue
        src = p.read_text(encoding="utf-8")
        reads_rank = "rank_0" in src or "rank_*" in src or "_rank_" in src
        has_guard = "assert_ranks_identical(" in src   # ★ 必须是**调用**共用函数，不是各写一份
        if reads_rank and not has_guard:
            log(f"  🔴 {rel} 读单个 rank 但**没有跨 rank 断言**")
            ok = False
        elif reads_rank:
            log(f"  ✅ {rel} 有跨 rank 断言")
    return ok


# ── 组 audit：评测与配对 ─────────────────────────────────────────────────

@check("audit", "审计的 case 集合必须与 split 的 eval 桶完全相同")
def _audit_covers_eval_bucket(log):
    sp = ROOT / f"{DEFAULT_SPLIT_DIR}/eval_cases.json"
    if not sp.exists():
        log(f"  ⏭ 没有 {DEFAULT_SPLIT_DIR}")
        return True
    want = set(json.loads(sp.read_text())["case_ids"])
    ok = True
    for p in sorted((ROOT / "_audit").glob("v13_*.json")):
        rows = json.loads(p.read_text()).get("rows")
        if rows is None:
            continue
        got = [r["case_id"] for r in rows]
        dup = len(got) - len(set(got))
        miss, extra = want - set(got), set(got) - want
        if dup or miss or extra:
            log(f"  🔴 {p.name}: 重复 {dup} · 漏 {len(miss)} · 多 {len(extra)}")
            ok = False
        else:
            log(f"  ✅ {p.name}: {len(want)} 条，不漏不重")
    return ok


@check("audit", "★ 配对比较的两份审计，必须跑在同一个起点模型上")
def _paired_audits_same_base(log):
    """🔴 2026-08-18 抓到：基线跑在 `裸基座 + SFT adapter`，
    而 RL 的起点是 `合并后的 SFT 模型`（合并丢了 36% 的增量）⇒ 两端基座不是同一个模型。
    00-START §2.4 明写配对比较要求「同一起点模型 + 同一推理引擎」。"""
    pairs = [("v13_sft_e1", "v13_rl_s110")]
    ok = True
    for a, b in pairs:
        pa, pb = ROOT / f"_audit/{a}.json", ROOT / f"_audit/{b}.json"
        if not (pa.exists() and pb.exists()):
            log(f"  ⏭ {a}/{b} 缺失")
            continue
        la = json.loads(pa.read_text())["label"]
        lb = json.loads(pb.read_text())["label"]
        base_a = la.split(" + ")[0].strip()
        base_b = lb.split(" + ")[0].strip()
        if base_a != base_b:
            log(f"  🔴 {a} 的基座是 `{base_a}`，{b} 的基座是 `{base_b}` ⇒ **不可配对**")
            ok = False
        else:
            log(f"  ✅ {a} / {b} 同基座 `{base_a}`")
    return ok


# ── 组 rollout：GRPO 组结构 ─────────────────────────────────────────────

@check("rollout", "每次更新的 GRPO 组：题目不重复、每组恰好 rollout_n 条")
def _grpo_groups_wellformed(log):
    """⚠️ 口径更正（2026-08-19）：重复的根因不是「有放回抽样」——`Pool.sample`
    一直是无放回的；重复来自**采样批与训练批的边界错位**（两次独立调用之间可重复）。
    已在采样器层结构性修掉（排除上一批 ⇒ 任何 ≤ 批宽的窗口无重复）。

    ⇒ 判据分两档：带 `case_id` 的 dump（修复后的新仪器）**零容忍**；
      旧 dump（没有 case_id）保留 10% 的历史容差 —— 那些跑已经跑完，红了也没有行动。
    """
    import collections
    files = sorted(glob.glob(str(ROOT / "checkpoints/grpo/*/rollout_dumps/*.jsonl")))
    if not files:
        log("  ⏭ 没有 rollout_dumps")
        return True
    by_run = collections.defaultdict(list)
    for f in files:
        by_run[Path(f).parents[1].name].append(f)
    ok = True
    for run, fs in sorted(by_run.items()):
        bad, instrumented = 0, False
        for f in fs:
            rows = [json.loads(l) for l in open(f)]
            if rows and "case_id" in rows[0]:
                instrumented = True
                g = collections.Counter(r["case_id"] for r in rows)
            else:
                g = collections.Counter(hash(r["input"]) for r in rows)
            if len(set(g.values())) != 1:
                bad += 1
        rate = bad / max(1, len(fs))
        if instrumented and bad:
            log(f"  🔴 {run}: {bad}/{len(fs)} 步有重复题 —— 修复后的跑**零容忍**"
                "（排除窗口失效了，去查 DynamicPoolSampler）")
            ok = False
        elif rate > 0.10:
            log(f"  🔴 {run}: {bad}/{len(fs)} 步（{rate:.0%}）同一条题被抽了不止一次")
            ok = False
        elif bad:
            log(f"  🟡 {run}: {bad}/{len(fs)} 步（{rate:.0%}）—— 修复前的历史跑"
                "（采样批/训练批边界错位），低于 10% 容忍")
    return ok


@check("rollout", "同一条 case 的所有 rollout 必须吃同一份失败注入剧本（Q4）")
def _failure_injection_deterministic(log):
    """`22 §13`：失败注入必须确定性、由 case 声明 —— GRPO 是组内比较，
    同组吃到不同剧本，reward 差异分不清「策略不同」还是「运气不同」。
    机制上剧本存在 bundle、匹配无随机源 ⇒ 应当恒同；此前**没有任何守卫**。

    判据：dump 里同一 case_id 的 failures_fp 集合大小必须为 1（两个东西应当相同）。
    """
    import collections
    files = sorted(glob.glob(str(ROOT / "checkpoints/grpo/*/rollout_dumps/*.jsonl")))
    fp_by_case: dict[str, set] = collections.defaultdict(set)
    n_rows = 0
    for f in files:
        for line in open(f):
            row = json.loads(line)
            if "failures_fp" in row and "case_id" in row:
                fp_by_case[row["case_id"]].add(row["failures_fp"])
                n_rows += 1
    if not fp_by_case:
        log("  ⏭ 还没有带 failures_fp 的 dump（2026-08-19 新仪器）——"
            "下一次 RL 跑会自动落数据，这条随之生效")
        return True
    bad = {cid: fps for cid, fps in fp_by_case.items() if len(fps) != 1}
    if bad:
        for cid, fps in sorted(bad.items())[:5]:
            log(f"  🔴 {cid}: 出现 {len(fps)} 份不同的失败剧本指纹 {sorted(fps)}")
        log(f"     共 {len(bad)} 条 case 违反 —— 失败注入不再确定性，advantage 被污染")
        return False
    log(f"  ✅ {len(fp_by_case)} 条 case / {n_rows} 条 rollout：每条 case 的剧本指纹唯一")
    return True


# ── 组 data：训练数据与渲染器同源、且在预算之内 ────────────────────────────

@check("data", "★ RL parquet 的 prompt 必须与渲染器同源、且零截断（Q7）")
def _rl_prompts_same_source_and_fit(log):
    """Q7（`18 §8`）：「RL 看到的题面 = 现在的渲染器渲出来的题面，且一个 token 不截」。

    这正是 defer 崩塌归因翻案那次事故的数据侧探针：3584 预算下 prompt 100% 被左截断，
    「调查先于任何结论（含拒绝）」整段消失（存活 0/659），模型训练时看不见"可以拒绝"。

    判据两半（都是「两个东西应当相同」）：
      ① 每行的 prompt_trace_hash == 用当前渲染器重渲的 hash（同源，抓渲染器漂移）
      ② 每行 prompt 按**运行时的同一条路径**分词后 ≤ MAX_PROMPT_LENGTH（零截断）

    ⚠️ ②必须带 tools 一起渲：运行时是 `apply_chat_template(messages, tools=...)`，
      工具 spec 占 ~2.7k token。第一版没带 tools，量出"最长 1471"——而实测真实
      prompt 是 4170–4210。**量的不是运行时那条路径 = 量错了对象**（守则形状）。
    """
    from syncopate.core.schemas import CaseBundle
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.split import DATA_VERSION
    from syncopate.prompts import stable_hash
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH
    from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS, build_messages

    import pandas as pd
    ok = True
    tok = None
    registry = build_domain().registry
    tok_dir = ROOT / STUDENT_MODEL
    if tok_dir.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        log("  ⏭ models/Qwen3-4B 不在本机 ⇒ 只验同源，跳过分词长度那半")

    for name in ("train", "val"):
        p = ROOT / f"data/rl/{DATA_VERSION}/{name}.parquet"
        if not p.exists():
            log(f"  ⏭ {p.relative_to(ROOT)} 不存在")
            continue
        frame = pd.read_parquet(p)
        drift, missing, over, max_len = [], 0, [], 0
        for _, row in frame.iterrows():
            extra = row["extra_info"]
            batch_dir = Path(extra["batch_dir"])
            if not batch_dir.exists():
                missing += 1
                continue
            bundle = CaseBundle.read(batch_dir, extra["case_id"])
            prompt = build_messages(bundle, bundle.case.tool_menu)
            if stable_hash(prompt)[:16] != extra["prompt_trace_hash"]:
                drift.append(extra["case_id"])
            if tok is not None:
                # ★ 与 rollout_loop.run_rollout 同一条渲染路径（messages + tools）
                tools = registry.menu(bundle.case.tool_menu)
                n = len(chat_template_ids(tok, prompt, tools=tools, add_generation_prompt=True,
                                                **CHAT_TEMPLATE_KWARGS))
                max_len = max(max_len, n)
                if n > MAX_PROMPT_LENGTH:
                    over.append((extra["case_id"], n))
        tag = str(p.relative_to(ROOT))
        if missing:
            log(f"  ⏭ {tag}: {missing} 条的 batch_dir 不在本机，跳过")
        if drift:
            log(f"  🔴 {tag}: {len(drift)} 条与当前渲染器**渲出不同的 prompt**"
                f"（前 5：{drift[:5]}）—— 数据是旧渲染器的产物，重建或钉住版本")
            ok = False
        else:
            log(f"  ✅ {tag}: {len(frame) - missing} 条 prompt 与当前渲染器同源")
        if over:
            log(f"  🔴 {tag}: {len(over)} 条分词后超过 MAX_PROMPT_LENGTH="
                f"{MAX_PROMPT_LENGTH}（最长 {max(n for _, n in over)}）—— 会被截断")
            ok = False
        elif tok is not None:
            log(f"  ✅ {tag}: 分词后最长 {max_len} ≤ {MAX_PROMPT_LENGTH}，零截断")
    return ok


@check("data", "★ 每条 SFT 样本的监督段必须以终答收尾（gold 回放完整走完）")
def _sft_samples_end_with_final_answer(log):
    """🔴 2026-08-19 抓到：build_dataset 造 SFT 数据时轮数上限落在默认 8
    （没跟 case.max_steps），v13 有 131/503 条 gold 回放在第 8 轮被无声掐断
    ⇒ **最终结论从没进过训练数据**，模型学到的是「调满 8 次工具然后停」。

    判据：最后一个 loss_mask==1 的连续段，解码后必须是终答
    （render_final_answer 的 ```json {"behavior": ...}``` 块）。
    构造侧已加硬断言（sft_replay），这条守数据侧 —— 两头都看着。
    """
    from syncopate.pipeline.split import DATA_VERSION
    tok_dir = ROOT / STUDENT_MODEL
    if not tok_dir.exists():
        log("  ⏭ models/Qwen3-4B 不在本机（要它的 tokenizer 解码）")
        return True
    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    ok = True
    for name in ("train", "val"):
        p = ROOT / f"data/sft/{DATA_VERSION}/{name}.parquet"
        if not p.exists():
            log(f"  ⏭ {p.relative_to(ROOT)} 不存在")
            continue
        frame = pd.read_parquet(p, columns=["case_id", "input_ids", "loss_mask"])
        bad = []
        for _, row in frame.iterrows():
            ids, mask = list(row["input_ids"]), list(row["loss_mask"])
            # 最后一个 mask==1 的连续段
            end = max((i for i, m in enumerate(mask) if m == 1), default=-1)
            if end < 0:
                bad.append(row["case_id"])
                continue
            start = end
            while start > 0 and mask[start - 1] == 1:
                start -= 1
            tail = tok.decode(ids[start:end + 1])
            if '"behavior"' not in tail:
                bad.append(row["case_id"])
        if bad:
            log(f"  🔴 {p.relative_to(ROOT)}: {len(bad)}/{len(frame)} 条的监督段"
                f"**不以终答收尾**（前 5：{bad[:5]}）—— gold 回放被掐断，"
                "模型在这些题上从没被教过下结论 ⇒ 数据要重建")
            ok = False
        else:
            log(f"  ✅ {p.relative_to(ROOT)}: {len(frame)} 条全部以终答收尾")
    return ok


def run_checks(groups_to_run) -> tuple[list[str], list[tuple[str, Exception]]]:
    """跑检查，把结果分成**两种失败**返回：`(违反, 无法判定)`。

    ★★ 为什么必须分开（2026-08-18 晚，当场踩的）

    这个脚本原本把「检查抛异常」和「检查真的查出违反」写进**同一个 `failed` 列表**，
    汇总只打一句「🔴 N 条不通过」。结果：用错解释器跑了一次（`/venv/main` 而不是项目
    `.venv`），4 条检查因 `ModuleNotFoundError: torch / safetensors / syncopate`
    变成"不通过"，和 3 条**真红**混报成「🔴 7 条不通过」。

    ⇒ 危险方向不是"假通过"，是**假失败**：任何没装依赖的环境跑它都会稳定红 4 条，
      而**假警报会训练人忽略这条判据 —— 那比没有判据更糟**（守则③）。
    ⇒ 但「无法判定」也**不许算过**（守则⑦：空着的门槛应读作"无法判定"）
      ⇒ 所以它仍然是非零退出码，只是**和"违反"分开计数、分开显示**。

    ★ 同一天在 `E22 §6.5.3` 记下的那条，说的正是这件事：
      **判据可以假通过，也可以假失败；打印本身要能区分"读不到"和"读到了是空"。**
    """
    groups = sorted({g for g, _, _ in CHECKS})
    violated: list[str] = []
    undetermined: list[tuple[str, Exception]] = []
    for group in groups:
        if group not in groups_to_run:
            continue
        print(f"\n══════ {group} ══════")
        for g, name, fn in CHECKS:
            if g != group:
                continue
            print(f"· {name}")
            lines: list[str] = []
            try:
                ok = fn(lines.append)
            except Exception as exc:  # 探针自己坏了要看得见，不能静默算过
                lines.append(f"  ⛔ 无法判定 —— 检查抛异常：{exc!r}")
                undetermined.append((name, exc))
            else:
                if not ok:
                    violated.append(name)
            for line in lines:
                print(line)
    return violated, undetermined


def _interpreter_note() -> str | None:
    """用的不是项目 `.venv` 时给一句提示。

    ⚠️ 只是**提示**，不是判据：脚本在别的解释器下跑是合法的（干净机器、CI）。
    这里做的只是把「无法判定」这个症状和它最常见的成因摆在同一屏上 ——
    2026-08-18 那次误报，从看到「🔴 7 条不通过」到发现是解释器不对，中间隔了一整轮。
    """
    expected = ROOT / ".venv"
    if not expected.is_dir():
        return None
    if Path(sys.prefix).resolve() == expected.resolve():
        return None
    return (f"   ℹ️ 当前解释器不是项目 .venv（{sys.prefix}）——"
            f"若上面是缺依赖，先换成 `source {expected}/bin/activate` 再跑")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="管线前提检查。退出码：0=全过 · 1=有判据被违反 · 2=只有无法判定的")
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些组")
    args = ap.parse_args(argv)

    run = args.only or sorted({g for g, _, _ in CHECKS})
    violated, undetermined = run_checks(run)

    print("\n" + "=" * 60)
    if violated:
        print(f"🔴 {len(violated)} 条判据被违反：")
        for name in violated:
            print(f"   - {name}")
    if undetermined:
        # ⚠️ 措辞刻意不叫"不通过" —— 它没查出问题，是**没查成**。
        print(f"⛔ {len(undetermined)} 条无法判定（探针自己没跑成，**不等于通过**）：")
        for name, exc in undetermined:
            print(f"   - {name}：{type(exc).__name__}")
        note = _interpreter_note()
        if note:
            print(note)
    if violated:
        return 1
    if undetermined:
        return 2
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
