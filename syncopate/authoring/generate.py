"""按配额批量生成 case，并**逐条实跑 gold 验证**。

流程：

    YAML 配额  →  参数组合  →  模板  →  跑一遍 gold  →  过了才写盘
                                          ↑
                            没过的直接丢弃并报出来，不进数据集

最后一步是刻意的。老师包里 2737 条 gold 的分数全部恰好 = 1.0，是构造时**预烤**进
文件的，没有任何一条真跑过——所以"gold 走得通"这件事其实从未被验证。
生成器最容易犯的错就是参数组合出一个自相矛盾的世界（比如 Google 平台配 45 秒视频，
upload 会被规格校验打回），这种错**只有真跑才发现得了**。
"""

from __future__ import annotations

import asyncio
import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import hashlib

from syncopate.authoring.axes import axis_summary, params_for
from syncopate.authoring.templates import TEMPLATES
from syncopate.train.rollout_loop import build_messages
from syncopate.core.runner import PlannedCall, run_plan
from syncopate.core.schemas import CaseBundle
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain


@dataclass
class Rejected:
    case_id: str
    reason: str


def prompt_fingerprint(bundle: CaseBundle) -> str:
    """一条 case 的**内容指纹**：模型实际看到的 prompt + 它该给出的答案。

    ★ 为什么必须有这道守卫

    v2 实测出过 6 条泄漏：val 和 train 里有 6 对样本 **token 完全相同**，
    case_id 不同但内容一模一样。成因是 clarify/reject 的 prompt 只含
    `account_id`（index%7）和 `requested_budget`（index%6），参数周期太短，
    不同 index 必然撞车。而 case_id 层面的去重**完全看不到这种撞车**。

    更糟的是：泄漏的 6 条里有 3 条正好是评测用的 case，
    于是"边界行为学会了"这个结论是在训练数据的字面副本上测出来的。

    ⇒ 去重必须按**内容**，不能按 id。参数设计再怎么改都可能撞，
       但只要这道守卫在，撞了就会被丢弃并报出来。
    """
    payload = json.dumps({
        "messages": build_messages(bundle, bundle.case.tool_menu),
        "tools": sorted(bundle.case.tool_menu or []),
        "answer": bundle.gold.final_answer if bundle.gold else None,
        "behavior": bundle.verifier.expected_behavior,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GenerateResult:
    accepted: list[CaseBundle] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)

    def counts_by(self, attr: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for bundle in self.accepted:
            key = getattr(bundle.case.metadata, attr)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    @property
    def counts(self) -> dict[str, int]:
        # 按 signal_class 统计。注意 clarify / reject 的 signal_class 也是 graded
        # （它们的 reward 确实是分层的），所以想看边界类的量要用 counts_by("bucket")。
        return self.counts_by("signal_class")


async def verify_gold(bundle: CaseBundle, domain) -> tuple[bool, str]:
    """跑一遍 gold，返回 (是否通过, 失败原因)。"""
    if bundle.gold is None:
        return False, "no_gold"
    calls = [PlannedCall(tool=a["tool"], arguments=a.get("arguments", {}))
             for a in bundle.gold.actions]
    trajectory, sandbox = await run_plan(
        bundle, domain.registry, calls,
        final_answer=bundle.gold.final_answer,
        behavior=bundle.verifier.expected_behavior,
    )
    # ★ 声明过失败剧本的工具，报错是**预期之内**的 —— F 类 gold 的全部意义
    # 就是示范"失败之后怎么办"。这道检查写于"任何工具报错都是世界坏了"的年代，
    # 现在有些错误就是题目本身。
    injected = {f.get("tool") for f in bundle.env.failures}
    failed = [(o.tool, o.error) for o in trajectory.observations
              if not o.ok and o.tool not in injected]
    if failed:
        return False, f"工具报错 {failed}"
    result = score_trajectory(
        bundle, trajectory, sandbox,
        policy_scorer=domain.policy_scorer, decision_fn=domain.decision_fn, caps=domain.caps,
    )
    if result.cap_hits:
        return False, f"gold 命中 cap {[h.name for h in result.cap_hits]}"
    if result.reward < bundle.gold.expected_reward_min:
        return False, (f"reward {result.reward:.3f} < {bundle.gold.expected_reward_min} "
                       f"subscores={ {k: round(v,2) for k,v in result.subscores.items()} }")
    return True, ""


async def generate(spec: dict[str, Any]) -> GenerateResult:
    """按 spec 里的配额生成。

    每个 signal_class 独立编号，所以序号轴不会互相干扰；
    `entry_mode` 之类的次级变化由 params_for 用不同质数步长错开。
    """
    domain = build_domain()
    # 生成时把工具延迟归零：造数据不该等 480 秒（延迟是 RL 计时实验的属性，不是数据的）
    original_scale = domain.registry.latency_scale
    domain.registry.latency_scale = 0.0

    result = GenerateResult()
    seen: dict[str, str] = {}      # 内容指纹 -> 首次出现的 case_id，跨模板全局去重
    try:
        for template_name, quota in spec["quotas"].items():
            template = TEMPLATES.get(template_name)
            if template is None:
                raise ValueError(f"未知模板: {template_name}")
            index = 0
            produced = 0
            before = len(result.rejected)
            # 允许最多试 6 倍配额——gold 验证和内容去重都会刷掉一部分
            while produced < quota and index < quota * 6:
                bundle = template(params_for(index))
                index += 1
                fingerprint = prompt_fingerprint(bundle)
                if fingerprint in seen:
                    result.rejected.append(Rejected(bundle.case_id, f"内容重复于 {seen[fingerprint]}"))
                    continue
                ok, reason = await verify_gold(bundle, domain)
                if ok:
                    seen[fingerprint] = bundle.case_id
                    result.accepted.append(bundle)
                    produced += 1
                else:
                    result.rejected.append(Rejected(bundle.case_id, reason))
            if produced < quota:
                # ⚠️ 只报**本模板**的拒绝原因。早期版本报的是 result.rejected[:3]
                # （全局前三条，通常来自前一个模板），排查时会被带偏。
                mine = result.rejected[before:]
                reasons = collections.Counter(r.reason.split(" ")[0] for r in mine)
                raise RuntimeError(
                    f"{template_name} 只生成出 {produced}/{quota} 条（试了 {index} 个 index）。"
                    f"拒绝原因分布: {dict(reasons)}；样例: {[r.reason for r in mine[:3]]}"
                )
    finally:
        domain.registry.latency_scale = original_scale
    return result


def write_batch(result: GenerateResult, out_dir: Path, spec: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for bundle in result.accepted:
        bundle.write(out_dir)
        meta = bundle.case.metadata
        entries.append({
            "case_id": bundle.case_id,
            "signal_class": meta.signal_class,
            "bucket": meta.bucket,
            "topology": meta.topology,
            "difficulty": meta.difficulty,
            "expected_behavior": bundle.verifier.expected_behavior,
        })
    manifest = {
        "version": "manifest_v1",
        "domain": "adcampaign",
        "spec_name": spec.get("name", "unnamed"),
        "count": len(entries),
        "signal_class_counts": result.counts,
        "bucket_counts": result.counts_by("bucket"),
        "topology_counts": result.counts_by("topology"),
        "rejected_count": len(result.rejected),
        "entries": sorted(entries, key=lambda e: e["case_id"]),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def load_spec(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(spec_path: Path, out_dir: Path) -> GenerateResult:
    spec = load_spec(spec_path)
    result = asyncio.run(generate(spec))
    write_batch(result, out_dir, spec)
    return result
