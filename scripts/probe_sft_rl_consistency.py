#!/usr/bin/env python
"""Q5/Q6 探针：SFT 与 RL 看到的 token 序列同构 · loss_mask 只落在模型生成段。

    python scripts/probe_sft_rl_consistency.py                # 抽样 48 条（确定性等距）
    python scripts/probe_sft_rl_consistency.py --all          # 全量（约 30 分钟，CPU）

退出码与 check_pipeline_invariants 同口径：0=过 · 1=违反 · 2=没跑成。

★ 两条判据（`18 §8` 的 Q5/Q6），都是「两个东西应当相同」：

  Q5  parquet 里的 (input_ids, loss_mask) 必须 == 用**当前代码**回放 gold 重建的结果。
      SFT 数据由 sft_replay 回放产出（构造上与 RL 同构）——但那是**建数据那天**的代码。
      渲染器/工具/模板此后任何漂移，都会让「SFT 教的」和「RL 看的」悄悄岔开，
      而两边各自都跑得通、无任何报错。逐 token 比对是唯一能抓到岔开的判据。

  Q6  loss_mask==1 的 token 序列必须 == gold 各步文本分词后的拼接。
      即：监督**恰好**覆盖模型该生成的 token —— 工具返回一个都不进 loss
      （把工具返回算进 loss = 教模型复述环境）、模型该说的一个都不漏。
      ⚠️ E26 刚在 verl 里发现过「梯度掩码被当成存在掩码」的先例，mask 语义
      是这条管线最容易被上游偷走的东西，值得独立验。

⚠️ 回放预算刻意给到 build_dataset.build 的同款（8192/8192）——比对的是「同一台机器
   两次跑同一段代码」，预算必须同源，否则截断差异会污染逐 token 比对。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Q5/Q6：SFT↔RL token 同构 + mask 落点")
    ap.add_argument("--data", default=None, help="默认 data/sft/<DATA_VERSION>")
    ap.add_argument("--model", default="models/Qwen3-4B")
    ap.add_argument("--sample", type=int, default=48,
                    help="确定性等距抽样条数（--all 时忽略）")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    import pandas as pd

    from syncopate.core.schemas import CaseBundle
    from syncopate.domains.adcampaign import build_domain
    from syncopate.pipeline.sft_replay import build_sft_sample, gold_script
    from syncopate.pipeline.split import DATA_VERSION
    from syncopate.train.rollout_loop import RolloutConfig
    from transformers import AutoTokenizer

    data_dir = ROOT / (args.data or f"data/sft/{DATA_VERSION}")
    model_dir = ROOT / args.model
    if not model_dir.exists():
        print(f"⛔ 没跑成：{model_dir} 不在本机（要它的 tokenizer）")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    registry = build_domain().registry
    registry.latency_scale = 0.0          # 延迟是计时实验的属性，不是数据的属性
    # 每个 assistant 轮收尾的 token（模型该学会自己停）——从真实分词取，不硬编码
    eot_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    frames = []
    for name in ("train", "val"):
        p = data_dir / f"{name}.parquet"
        if p.exists():
            frame = pd.read_parquet(p)
            frame["_file"] = name
            frames.append(frame)
    if not frames:
        print(f"⛔ 没跑成：{data_dir} 下没有 parquet")
        return 2
    rows = pd.concat(frames).sort_values("case_id").reset_index(drop=True)

    if not args.all and len(rows) > args.sample:
        stride = len(rows) / args.sample
        rows = rows.iloc[[int(i * stride) for i in range(args.sample)]]

    # batch_dir：SFT parquet 不带路径，按版本约定取
    batch_dir = ROOT / f"data/batches/{DATA_VERSION}"
    if not batch_dir.exists():
        print(f"⛔ 没跑成：{batch_dir} 不在本机（回放要读四件套）")
        return 2

    q5_bad, q6_bad, checked = [], [], 0
    for _, row in rows.iterrows():
        cid = row["case_id"]
        bundle = CaseBundle.read(batch_dir, cid)
        # 轮数上限跟 case 走（与修复后的 build_dataset 同源）。
        # ⚠️ 对修复前建的数据，这会让被掐断的样本在 Q5 红 —— 那是数据的病，不是探针的
        config = RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                               max_prompt_length=8192, max_response_length=8192)
        sample = asyncio.run(build_sft_sample(
            bundle, tokenizer=tokenizer, registry=registry, config=config))
        got_ids, got_mask = list(row["input_ids"]), list(row["loss_mask"])
        checked += 1

        # ── Q5：parquet == 当前代码的回放，逐 token ──
        if got_ids != sample.input_ids or got_mask != sample.loss_mask:
            first = next((i for i, (a, b) in enumerate(zip(got_ids, sample.input_ids))
                          if a != b), min(len(got_ids), len(sample.input_ids)))
            q5_bad.append((cid, len(got_ids), len(sample.input_ids), first))
            continue                      # 序列都不同了，Q6 没有比的基础

        # ── Q6：mask==1 的 token == gold 各步文本（+ 每轮的 <|im_end|>\n）拼接 ──
        supervised = [t for t, m in zip(got_ids, got_mask) if m == 1]
        expected: list[int] = []
        for step_text in gold_script(bundle):
            expected += tokenizer.encode(step_text, add_special_tokens=False) + eot_ids
        if supervised != expected:
            q6_bad.append((cid, len(supervised), len(expected)))

    print(f"检查 {checked} 条（{'全量' if args.all else f'等距抽样，全量 {sum(len(f) for f in frames)}'}）")
    if q5_bad:
        print(f"🔴 Q5 违反 {len(q5_bad)} 条：parquet 与当前回放**逐 token 不同**")
        for cid, a, b, first in q5_bad[:5]:
            shape = ("parquet 是回放的前缀 ⇒ 旧数据被轮数上限掐断（缺终答），数据要重建"
                     if first == a < b else "中途分歧 ⇒ 渲染器/工具/模板漂了")
            print(f"   {cid}: parquet {a} tok vs 回放 {b} tok，首个分歧位 {first} —— {shape}")
    else:
        print("✅ Q5：parquet 与当前代码的回放逐 token 相同（SFT↔RL 同构仍由构造保证）")
    if q6_bad:
        print(f"🔴 Q6 违反 {len(q6_bad)} 条：监督 token ≠ gold 各步的分词拼接")
        for cid, a, b in q6_bad[:5]:
            print(f"   {cid}: mask 覆盖 {a} tok vs 期望 {b} tok")
        print("   ⇒ 工具返回混进了 loss，或模型该说的被漏掉 —— mask 语义被动过")
    else:
        print("✅ Q6：loss_mask 恰好覆盖模型生成段（工具返回零泄漏）")
    return 1 if (q5_bad or q6_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
