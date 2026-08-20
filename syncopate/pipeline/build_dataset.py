"""四件套 -> 训练数据。一个模块同时出 RL 和 SFT 两种格式。

    RL  : prompt-only。只放首轮 messages + 指向四件套的路径。
          reward 是在线跑完 rollout 才算出来的，所以 env/verifier 必须留在磁盘上
          让 AgentLoop 运行时去读——不能提前烤进 parquet。

    SFT : 预分词。input_ids + loss_mask，由 `sft_replay` 回放 gold 产出。
          不走 verl 的 MultiTurnSFTDataset，原因见 sft_replay 的模块 docstring
          （整段渲染和增量拼接天生对不齐）。

两者共用同一份 batch 目录和同一套切分规则，保证 case 不会一边在 train 一边在 eval。

★★★ `split_dir` 不是可选装饰

M0 切出了三桶，但本模块**曾经完全没读它**——直接按整个 manifest 建数据集。
后果是 `data/sft/v2` 有 580 条，52 条冻结 EVAL 全在训练数据里，
而 `split_report.json` 还写着「三桶零重叠 ✅」。切分文件和训练数据是两回事，
**切得再干净，只要构造这一步不读它，泄漏照样发生**。

所以这里不给 `split_dir` 一个安静的默认值：要么给桶，要么显式 `full_batch=True`
说明你就是要全量。忘记声明会当场报错，不会安静地泄漏。
（同一条原则见 VerifierSpec.active_caps 的 None / [] 之分。）
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolRegistry
from syncopate.prompts import stable_hash
from syncopate.train.rollout_loop import RolloutConfig, build_messages

DATA_SOURCE = "syncopate_adcampaign"
AGENT_NAME = "syncopate_adcampaign"      # 必须和 verl_agent_loop.py 的 @register 一致


@dataclass
class BuildResult:
    out_dir: Path
    train_path: Path
    val_path: Path
    train_count: int
    val_count: int


def read_manifest(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    # 排序后再切分，保证多次构造得到完全一致的 train/val
    return sorted(manifest["entries"], key=lambda item: item["case_id"])


def split_of(index: int, val_every: int) -> str:
    """确定性切分：每 val_every 条取 1 条进 val。

    简单，但**可复现**——这比随机切分重要得多。老师包用的也是这个规则。
    """
    return "val" if val_every > 0 and index % val_every == 0 else "train"


# --------------------------------------------------------------------------
# RL：prompt-only
# --------------------------------------------------------------------------


def build_rl_row(bundle: CaseBundle, batch_dir: Path, index: int, split: str,
                 artifact_root: Path) -> dict[str, Any]:
    prompt = build_messages(bundle, bundle.case.tool_menu)
    meta = bundle.case.metadata
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        # verl 按行选 AgentLoop 实现；也可以用 rollout.agent.default_agent_loop 统一指定
        "agent_name": AGENT_NAME,
        "ability": "ad_campaign_tool_agent",
        "reward_model": {
            # 不是启用 verl 内置 reward model（train 脚本里 reward_model.enable=False），
            # 只是给样本标明 reward 口径
            "style": "syncopate_rule_verifier",
            "ground_truth": bundle.case_id,
        },
        "extra_info": {
            "index": index,
            "split": split,
            "case_id": bundle.case_id,
            # ★ AgentLoop 靠这个路径在运行时读回四件套
            "batch_dir": str(batch_dir.resolve()),
            "artifact_root": str(artifact_root.resolve()),
            "max_assistant_turns": bundle.case.max_steps,
            "need_tools_kwargs": False,
            # 分组统计用：可以按信号形态/弱点桶分别看 reward 分布和 advantage 方差
            "signal_class": meta.signal_class,
            "bucket": meta.bucket,
            "topology": meta.topology,
            "difficulty": meta.difficulty,
            "gold_reward_min": bundle.gold.expected_reward_min if bundle.gold else None,
            "prompt_trace_hash": stable_hash(prompt)[:16],
        },
    }


# --------------------------------------------------------------------------
# SFT：预分词
# --------------------------------------------------------------------------


async def build_sft_row(bundle: CaseBundle, tokenizer: Any, registry: ToolRegistry,
                        index: int, split: str, config: RolloutConfig) -> dict[str, Any]:
    from syncopate.pipeline.sft_replay import build_sft_sample

    sample = await build_sft_sample(bundle, tokenizer=tokenizer, registry=registry, config=config)
    if sample.supervised_tokens == 0:
        raise ValueError(f"{bundle.case_id}: 监督 token 为 0，gold 回放出了问题")
    return {
        "case_id": bundle.case_id,
        "input_ids": sample.input_ids,
        "loss_mask": sample.loss_mask,
        "prompt_length": sample.prompt_length,
        "total_length": sample.total_length,
        "supervised_tokens": sample.supervised_tokens,
        "split": split,
        "index": index,
        "signal_class": bundle.case.metadata.signal_class,
        # ★ 均衡采样按这一列分组。上一轮实测：clarify/reject 各占 8.7% 样本
        #   但只占 1.4% 监督 token，导致 SFT 把边界行为学没了。
        "behavior": bundle.verifier.expected_behavior,
        "bucket": bundle.case.metadata.bucket,
    }


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def build(
    *,
    batch_dir: Path,
    out_dir: Path,
    pool: str,
    val_every: int = 5,
    artifact_root: Path = Path("data/rollouts"),
    model_path: str | None = None,
    max_length: int = 8192,
    split_dir: Path | None = None,
    full_batch: bool = False,
) -> BuildResult:
    if (split_dir is None) == (not full_batch):
        raise ValueError(
            "必须二选一：给 split_dir（按三桶取 pool 对应的桶），"
            "或显式 full_batch=True（整批，会把冻结 EVAL 一起训进去）"
        )

    entries = read_manifest(batch_dir)
    excluded = 0
    if split_dir is not None:
        from syncopate.pipeline.split import load_bucket

        wanted = set(load_bucket(split_dir, pool))
        kept = [e for e in entries if e["case_id"] in wanted]
        if not kept:
            raise ValueError(
                f"{split_dir} 的 {pool} 桶和 {batch_dir} 一条都对不上——"
                f"多半是 batch 和 split 不是同一个版本"
            )
        excluded = len(entries) - len(kept)
        entries = kept

    bundles = [CaseBundle.read(batch_dir, entry["case_id"]) for entry in entries]

    if pool == "rl":
        rows = [
            build_rl_row(bundle, batch_dir, index, split_of(index, val_every), artifact_root)
            for index, bundle in enumerate(bundles)
        ]
        split_key = lambda row: row["extra_info"]["split"]  # noqa: E731
    elif pool == "sft":
        if model_path is None:
            raise ValueError("构造 SFT 数据需要 --model（要用它的 tokenizer 分词）")
        from transformers import AutoTokenizer

        from syncopate.domains.adcampaign import build_domain

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        registry = build_domain().registry
        # ★ 造 SFT 数据时把工具延迟归零。
        # gold 回放产出的是 token 序列，延迟一个字节都不影响它；但 poll_review 挂着
        # 480 秒真 sleep，不关掉的话构造 6 条数据要干等 8 分钟（我第一次就这么卡住了）。
        # 延迟是 RL 计时实验的属性，不是数据的属性——两者必须分开。
        registry.latency_scale = 0.0

        # ★★ 轮数上限必须逐 case 取 `case.max_steps`，和 RL 侧（verl_agent_loop
        #    从 extra_info.max_assistant_turns 读的就是它）同源。
        #
        # 🔴 2026-08-19 抓到：这里原来只传两个长度，`max_assistant_turns` 落在
        #    RolloutConfig 的默认值 8 上 —— 而 v13 有 131/503 条（26%）的 gold
        #    需要 >8 个 assistant 轮 ⇒ 回放在第 8 轮被掐断，**最终结论从没进过
        #    训练数据**，且 build_sft_row 不看 truncated，无声写盘。
        #    模型在这些 case 上学到的是「调满 8 次工具然后停」——
        #    与「GEO 类卡死（在里面打转）」的行为形状一致。
        # ⇒ 修法：轮数跟 case 走；外加 sft_replay 里的硬断言（gold 回放不许截断）。
        def _config_for(bundle: CaseBundle) -> RolloutConfig:
            return RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                                 max_prompt_length=max_length,
                                 max_response_length=max_length)

        async def _all() -> list[dict[str, Any]]:
            return [
                await build_sft_row(bundle, tokenizer, registry, index,
                                    split_of(index, val_every), _config_for(bundle))
                for index, bundle in enumerate(bundles)
            ]

        rows = asyncio.run(_all())
        split_key = lambda row: row["split"]  # noqa: E731
    else:
        raise ValueError(f"未知 pool: {pool}（可选 rl / sft）")

    train_rows = [row for row in rows if split_key(row) == "train"]
    val_rows = [row for row in rows if split_key(row) == "val"]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.parquet", out_dir / "val.parquet"
    _write_parquet(train_path, train_rows)
    _write_parquet(val_path, val_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps({
            "pool": pool,
            "data_source": DATA_SOURCE,
            "agent_name": AGENT_NAME if pool == "rl" else None,
            "source_batch": str(batch_dir.resolve()),
            "split_dir": str(split_dir.resolve()) if split_dir else None,
            "excluded_by_split": excluded,
            "count": len(rows),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "val_every": val_every,
            "columns": sorted(rows[0]) if rows else [],
            "signal_class_counts": _count_by(rows, "signal_class", pool),
        }, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BuildResult(out_dir, train_path, val_path, len(train_rows), len(val_rows))


def _count_by(rows: list[dict[str, Any]], key: str, pool: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row["extra_info"][key] if pool == "rl" else row[key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)
