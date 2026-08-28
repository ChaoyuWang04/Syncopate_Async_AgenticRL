#!/usr/bin/env python
"""U 路 P0-4 · OPD 自建路径 spike（`24 §4-P0`）：一个真 batch 验通三件事——
① 机制：掩码反向 KL 只在 NL 段产生梯度（零掩码对照=梯度逐位为零）
② 数值：损失有限、一步小步长后同 batch KL 下降（方向正确）
③ 工程：单步耗时（student fwd+bwd + teacher fwd）可接受

    CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/u_opd_spike.py

学生 = merged SFT + candidate LoRA（只训 LoRA）@ cuda:0；教师 = 裸底座 @ cuda:1。
样本 = 说人话考场前 4 条 prompt + 学生现役端点(:8100)的真实回复（on-policy 味道）。
"""

from __future__ import annotations

import json
import sys
import time

import httpx
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

STUDENT_BASE = "models/Qwen3-4B-sft-v13r2-e1"
ADAPTER = "checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25"
TEACHER = "models/Qwen3-4B"


def get_samples(n=4) -> list[tuple[str, str]]:
    """(用户消息, 学生回复文本)——回复来自现役 candidate 端点（真 on-policy 输出）。"""
    items = [json.loads(x) for x in open("data/u_route/talk_exam.jsonl")][:n]
    from probe_opd_divergence import render_prompt_text  # 复用契约渲染（含工具菜单）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    out = []
    with httpx.Client(base_url="http://127.0.0.1:8100", timeout=120) as c:
        for it in items:
            prompt = render_prompt_text(tok, it["turns"][0], tools=None)
            r = c.post("/v1/completions", json={
                "model": "candidate", "prompt": prompt, "max_tokens": 200,
                "temperature": 0.7})
            r.raise_for_status()
            out.append((prompt, r.json()["choices"][0]["text"]))
    return out


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    print("加载学生（merged+LoRA, cuda:0）…", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
    student = PeftModel.from_pretrained(student, ADAPTER, is_trainable=True)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  可训练参数 {n_train/1e6:.1f}M（应≈LoRA r32 量级）")
    print("加载教师（裸底座, cuda:1）…", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map={"": 1}).eval()

    samples = get_samples(4)
    print(f"取到 {len(samples)} 条学生真实回复（示例：{samples[0][1][:60]!r}）")

    def batch_loss(mask_on: bool, do_backward: bool = False,
                   eval_only: bool = False) -> float:
        """逐样本算 KL；do_backward=逐样本立即 backward（不攥四张图——首版 OOM 学费）；
        logits_to_keep 只算回复段（全位置 logits 是 1.3GB/样本的冤枉钱）。"""
        total, n_tok = 0.0, 0
        for prompt, reply in samples:
            ids_p = tok(prompt, return_tensors="pt").input_ids
            ids_r = tok(reply, add_special_tokens=False, return_tensors="pt").input_ids
            ids = torch.cat([ids_p, ids_r], dim=1)
            kr = ids_r.shape[1]
            ctx = torch.no_grad() if eval_only else torch.enable_grad()
            with ctx:
                s_out = student(ids.to(0), logits_to_keep=kr + 1).logits[0, :-1]
            with torch.no_grad():
                t_out = teacher(ids.to(1), logits_to_keep=kr + 1).logits[0, :-1].to(0)
            m = torch.ones(kr, device="cuda:0") if mask_on \
                else torch.zeros(kr, device="cuda:0")
            ls = torch.log_softmax(s_out.float(), -1)
            lt = torch.log_softmax(t_out.float(), -1)
            kl = (ls.exp() * (ls - lt)).sum(-1)          # 逐 token 反向 KL(学生‖教师)
            loss = (kl * m).sum()
            if do_backward and not eval_only:
                loss.backward()                          # 逐样本释放计算图
            total += float(loss.item()); n_tok += int(m.sum().item())
            del s_out, t_out, ls, lt, kl, loss
        return total / max(n_tok, 1)

    # ── 判据① 零掩码对照：梯度必须逐位为零 ────────────────────────────────
    student.zero_grad(set_to_none=True)
    loss0 = batch_loss(mask_on=False, do_backward=True)
    g0 = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    nonzero0 = sum(int(g.abs().sum() > 0) for g in g0)
    print(f"[判据①a] 零掩码：loss={loss0:.6f}，非零梯度张量 {nonzero0}/{len(g0)}"
          f" —— {'✅ 全零' if nonzero0 == 0 and loss0 == 0 else '🔴'}")

    # ── 正常掩码：梯度只在 LoRA、有限、KL>0 ─────────────────────────────
    student.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss1 = batch_loss(mask_on=True, do_backward=True)
    dt = time.perf_counter() - t0
    grads = [(n, p.grad) for n, p in student.named_parameters() if p.requires_grad]
    n_with = sum(int(g is not None and torch.isfinite(g).all()) for _, g in grads)
    frozen_touched = sum(int(p.grad is not None)
                         for n, p in student.named_parameters() if not p.requires_grad)
    print(f"[判据①b] 掩码开：KL={loss1:.4f}（应>0）· LoRA 梯度有限 {n_with}/{len(grads)}"
          f" · 冻结参数带梯度 {frozen_touched}（应=0）· 单步 {dt:.1f}s")

    # ── 判据② 方向：一小步后同 batch KL 应下降 ──────────────────────────
    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=5e-5)
    opt.step()
    loss2 = batch_loss(mask_on=True, eval_only=True)
    print(f"[判据②] 一步后 KL {loss1:.4f} → {loss2:.4f} "
          f"—— {'✅ 下降' if loss2 < loss1 else '🔴 未降'}")

    ok = (nonzero0 == 0 and loss0 == 0 and n_with == len(grads)
          and frozen_touched == 0 and loss1 > 0 and loss2 < loss1)
    print("SPIKE-" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
