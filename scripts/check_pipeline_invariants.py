#!/usr/bin/env python
"""管线前提检查 —— 「两个东西应当相同 / 某集合应当完整」这类断言的集中执行处。

    python scripts/check_pipeline_invariants.py                     # 全跑
    python scripts/check_pipeline_invariants.py --only merge rank   # 只跑某几组
    退出码 0 = 全过

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
    pairs = [("models/Qwen3-4B-sft-v13-e1", "models/Qwen3-4B", "SFT 合并模型")]
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


# ── 组 quarantine：作废的数字不许在没有警示的情况下出现 ───────────────────

QUARANTINE_DOC = "docs/syncopate/21-invalidated-numbers.md"
QUARANTINE_BANNER = "21-invalidated-numbers"     # 横幅里必须出现的指回链接


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


@check("quarantine", "★ 含作废数字的文档，顶部必须挂 ⛔ 横幅指回 21 号")
def _quarantined_numbers_are_flagged(log):
    """🔴 2026-08-18：两个基石级 bug（E21 梯度不同步 · 权重从没推给 rollout）
    把 08-14 至 08-18 之间所有 RL 训练的实测数字都污染了。

    ⚠️ **作废的数字被将来引用，比没有数字更危险** —— 它看起来精确、有出处、有上下文，
    而它量的是一个坏掉的系统。本项目已经为此付过一次钱
    （「M7 已验收」被下游引用了两天，见 blank-thresholds-are-not-passes）。

    ⇒ 判据不是"把数字删掉"（删了就没人知道当初错在哪），
      而是**含作废数字的文档必须自己喊出来**。集合包含型判据，不需要阈值。
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
        # 横幅只认"文档开头 40 行内出现指回链接"——挂在末尾没人看得见
        head = "\n".join(text.splitlines()[:40])
        if QUARANTINE_BANNER not in head:
            offenders[rel] = hits
            ok = False
    if offenders:
        for rel, hits in sorted(offenders.items()):
            log(f"  🔴 {rel}: 含 {len(hits)} 个作废数字但**顶部没有 ⛔ 横幅** —— {hits[:4]}")
    else:
        log(f"  ✅ 扫了 {len(scope)} 份文档，含作废数字的都挂了横幅")
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
        if hard:
            log(f"  🔴 {rel} 里还有硬编码的长度/采样常量：{hard}")
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
            log(f"  🟡 最近一次跑（{stamp}）用的是 {got} ≠ 共用预算 ——"
                " 这是修复**之前**的跑，属预期；下一跑起应当一致")
        else:
            log(f"  ✅ 最近一次跑（{stamp}）与共用预算一致")
    # ③ 批次脚本里显式覆盖的值
    import re as _re
    stale = []
    for sh in sorted((ROOT / "scripts").glob("*.sh")):
        txt = sh.read_text(encoding="utf-8")
        m1 = _re.search(r"--max-prompt-length\s+(\d+)", txt)
        m2 = _re.search(r"--max-response-length\s+(\d+)", txt)
        if m1 and m2 and (int(m1.group(1)) != MAX_PROMPT_LENGTH or int(m2.group(1)) != MAX_RESPONSE_LENGTH):
            stale.append(f"{sh.name}({m1.group(1)}/{m2.group(1)})")
    if stale:
        log(f"  🔴 {len(stale)} 个批次脚本仍显式传旧预算：{', '.join(stale[:6])}"
            + (" …" if len(stale) > 6 else ""))
        ok = False
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
    sp = ROOT / "data/splits/v13/eval_cases.json"
    if not sp.exists():
        log("  ⏭ 没有 data/splits/v13")
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
    05-handoff §2.4 明写配对比较要求「同一起点模型 + 同一推理引擎」。"""
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
        bad = 0
        for f in fs:
            g = collections.Counter(hash(json.loads(l)["input"]) for l in open(f))
            if len(set(g.values())) != 1:
                bad += 1
        rate = bad / max(1, len(fs))
        if rate > 0.10:
            log(f"  🔴 {run}: {bad}/{len(fs)} 步（{rate:.0%}）同一条题被抽了不止一次")
            ok = False
        elif bad:
            log(f"  🟡 {run}: {bad}/{len(fs)} 步（{rate:.0%}）—— 动态分池有放回抽样，低于 10% 视作正常")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="管线前提检查")
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些组")
    args = ap.parse_args(argv)

    groups = sorted({g for g, _, _ in CHECKS})
    run = args.only or groups
    failed = []
    for group in groups:
        if group not in run:
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
                lines.append(f"  ⛔ 检查抛异常：{exc!r}")
                ok = False
            for line in lines:
                print(line)
            if not ok:
                failed.append(name)
    print("\n" + "=" * 60)
    if failed:
        print(f"🔴 {len(failed)} 条不通过：")
        for f in failed:
            print(f"   - {f}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
