#!/usr/bin/env python3
"""R0-b · bf16 合并到底让策略偏了多少？—— 用 **logprob** 度量，不是用权重范数

★ 为什么要有这一步（E22 §6）：
`--lora-merge` 是 E22 的止血 —— 但它是**在 bf16 里做合并**。
主线量过这一级增量在 bf16 里的下场（`18 §3.3`）：

    RL 一轮的增量占基座 0.056%  →  幅度比 0.68、**保真残差 0.87**
    （幅度留住了，**方向被舍入噪声打乱**；损失来自**存储精度**，不是累加精度）

⇒ 「推出去了」不等于「推对了」。**但权重的保真残差不是我们真正关心的量** ——
我们关心的是**它让策略偏了多少**，而策略只通过 logprob 进入训练（IS 权重、advantage、ESS）。
⇒ 所以这里直接量 logprob。

★ 设计上刻意**不比 vLLM 与 trainer**，而是在**同一个引擎里**比两份权重：

    路径 A（trainer 手上的那份）  基座 + LoRA adapter，前向时按 W + BA 计算
    路径 B（推给 rollout 的那份） (W + BA) 存成 bf16 之后的那份

⇒ 两条路径**同引擎、同 dtype、同 kernel、同一批 prompt** ⇒ 差异**只可能**来自 bf16 合并。
⚠️ 若改成"比 vLLM 与 trainer"，差异里会混进
   **引擎数值实现差异 + 陈旧度 + 合并损失**三项，分不开 —— 那就是本项目栽过的「判据量错对象」。
   引擎那一项已有独立参考值：同版本下的 `log_ppl_diff` 地板 ≈ **3.4e-4**（主线实测）。

判据（三档，对应三条完全不同的路）：
    逐 token |Δlogprob| 的中位数与 p95，以及**序列级** Σ|Δ| ⇒
      ≲ 3.4e-4 量级        ✅ 合并损失在引擎噪声以下 ⇒ 止血够用，可以开始重跑
      明显更大但不随位移增长 🟡 能用，但每份结论要带上这个误差
      随位移单调增长         🔴 止血不够 ⇒ 必须走上游方案①（单独推 adapter，不做 bf16 合并）

用法：
    python scripts/probe_merge_logprob_fidelity.py <ckpt 的 actor 目录> [--n 16]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR


def load_lora_from_ckpt(actor_dir: Path) -> dict[str, torch.Tensor]:
    """从 verl 的 RL ckpt 里读出 LoRA 张量（跨 rank 一致性由 ckpt_guards 保证）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from syncopate.train.ckpt_guards import assert_ranks_identical

    # ★ 沿用 E21 之后的纪律：读单个 rank 之前先确认各 rank 一致（共用实现，三处同一份）
    n_cmp = assert_ranks_identical(actor_dir)
    print(f"  跨 rank 一致性断言：比了 {n_cmp} 个张量（0 = 单 rank，没得比）")
    cands = sorted(glob.glob(str(actor_dir / "model_world_size_*_rank_0.pt"))) or \
            sorted(glob.glob(str(actor_dir / "*lora*.pt")))
    if not cands:
        raise SystemExit(f"★ {actor_dir} 里没有找到权重文件")
    sd = torch.load(cands[0], map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # ⚠️ verl 存的键带适配器名（`...lora_A.default.weight`），
    #    而 PEFT 的 `set_peft_model_state_dict` 期望**不带**（它自己会插入 adapter_name）。
    #    ⇒ 不去掉的话 504 个张量一个都装不上，而 load_state_dict(strict=False) **不会报错**
    #      （2026-08-18 实测撞上，靠探针自检拦下）。
    out = {}
    for k, v in sd.items():
        if "lora_" not in k.lower():
            continue
        out[k.replace(".default.weight", ".weight")] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("actor_dir", type=Path)
    ap.add_argument("--base", default="models/Qwen3-4B-sft-v13-e1")
    ap.add_argument("--prompts", default=f"{DEFAULT_RL_DIR}/val.parquet")
    ap.add_argument("--n", type=int, default=16, help="用多少条 prompt")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    import pandas as pd
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    df = pd.read_parquet(args.prompts).head(args.n)
    col = "prompt" if "prompt" in df.columns else df.columns[0]

    def render(x):
        if isinstance(x, str):
            return x
        try:                                   # verl 的 parquet 存的是 chat 列表
            return tok.apply_chat_template(list(x), tokenize=False, add_generation_prompt=True)
        except Exception:                      # noqa: BLE001
            return str(x)

    texts = [render(v) for v in df[col].tolist()]

    lora = load_lora_from_ckpt(args.actor_dir)
    if not lora:
        raise SystemExit("★ ckpt 里没有 LoRA 张量 —— 判据无效，先查 ckpt 是不是对的")
    print(f"  读到 {len(lora)} 个 LoRA 张量，{len(texts)} 条 prompt")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0")
    peft_model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    incompat = set_peft_model_state_dict(peft_model, lora)
    n_missing_lora = len([k for k in getattr(incompat, "missing_keys", []) if "lora_" in k])
    print(f"  装载 adapter：missing 里还有 {n_missing_lora} 个 lora_ 键（应为 0）")
    if n_missing_lora:
        raise SystemExit("★ LoRA 键没对上 ⇒ 判据无效。先修键名映射，别看下面的数")

    @torch.no_grad()
    def logprobs(m) -> torch.Tensor:
        out = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
            lg = m(ids).logits[:, :-1].float().log_softmax(-1)
            out.append(lg.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).flatten().cpu())
        return torch.cat(out)

    # ★★ 先做「探针有能力失败」的自检（主线 18 §1-③）：
    #    若 adapter 根本没装上，下面 A 与 B 会**平凡地**相等 ⇒ 读成"通过"。
    #    所以先确认 adapter 确实改变了 logprob。
    with peft_model.disable_adapter():
        lp_base = logprobs(peft_model)
    lp_a = logprobs(peft_model)          # 路径 A：基座 + adapter（trainer 手上的那份）
    d_adapter = (lp_a - lp_base).abs().median().item()
    print(f"  自检：adapter 让 logprob 中位偏移 {d_adapter:.3e}")
    if d_adapter == 0.0:
        raise SystemExit("★ adapter 没有生效（A 与裸基座逐位相同）⇒ 判据无效，不许当通过读")

    # 路径 B：把 adapter 合并进 bf16 权重（推给 rollout 的那份）
    merged = peft_model.merge_and_unload()          # PEFT 就地合并，dtype 仍是 bf16
    lp_b = logprobs(merged)

    d = (lp_a - lp_b).abs()
    n_tok = d.numel()
    print("\n" + "=" * 88)
    print("  R0-b · bf16 合并造成的 logprob 偏移（同引擎、同 dtype、只差合并这一步）")
    print("=" * 88)
    print(f"    token 数            {n_tok}")
    print(f"    |Δlogprob| 中位数    {d.median().item():.3e}")
    print(f"    |Δlogprob| p95      {d.quantile(0.95).item():.3e}")
    print(f"    |Δlogprob| 最大      {d.max().item():.3e}")
    print(f"    平均每条序列 Σ|Δ|    {d.sum().item() / len(texts):.3e}")
    print(f"\n    参考地板（vLLM↔FSDP 同版本，主线实测）  3.4e-4")
    med = d.median().item()
    if med <= 3.4e-4:
        print("    ✅ 合并损失在引擎噪声以下 ⇒ 止血够用，可以开始重跑")
    elif med <= 3.4e-3:
        print("    🟡 比引擎噪声大一个量级以内 ⇒ 能用，但每份结论要带上这个误差")
    else:
        print("    🔴 显著大于引擎噪声 ⇒ 止血不够，走上游方案①（单独推 adapter）")
    print("=" * 88)
    print("  ⚠️ 本探针只隔离了**合并**这一项。引擎差异与陈旧度是另外两项，不在这里量。")


if __name__ == "__main__":
    main()
