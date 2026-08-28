"""U 路 P1 · On-Policy Distillation 训练器（`docs/syncopate/24 §4-P1`，固定管线家族）。

    torchrun --nproc_per_node=2 -m syncopate.train.opd \
        --prompts data/u_route/p1_prompts.jsonl --epochs 3 --out checkpoints/opd/p1_r1

四卡布局（充分利用，零跨卡通信除 DDP 梯度）：
    rank0: 学生 @cuda:0(物理GPU0) · 教师+锚 @物理GPU1
    rank1: 学生 @cuda:1(物理GPU3) · 教师+锚 @物理GPU2
  —— 每 rank 自带一对教师/锚（4B bf16 ×2 = 16GB/卡），无共享争抢。
  启动令：CUDA_VISIBLE_DEVICES=0,3,1,2 torchrun ...（教师卡 OPD_AUX_GPUS=2,3=物理1,2）

机制（P0-4 spike 三判据的生产版，判据行全部常驻）：
  on-policy：学生自己 generate（契约渲染+契约采样参数，多轮取最后一轮回复）
  掩码：segment_text 的 reply 值白名单（P0-3 修复版）——[opd-mask] 每批非零断言
  双教师路由：chat→底座 · task/task_neg→候选冻结锚——[opd-route] 计数入 wandb
  损失：逐 token 反向 KL(学生‖教师)，逐样本 backward + logits_to_keep（显存两课）
  零泄漏断言：每 --probe-every 步跑一次零掩码对照，LoRA 梯度必须逐位为零

存储：adapter-only（peft save_pretrained，E29 口径），滚动保留最近 3 份 + final。
wandb：project=syncopate（与 sft/launch_rl 同族默认开），指标前缀 opd/*。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

STUDENT_BASE = "models/Qwen3-4B-sft-v13r2-e1"
ADAPTER = "checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25"
TEACHER_BASE = "models/Qwen3-4B"


def log(msg: str) -> None:
    print(f"[opd r{os.environ.get('RANK', '0')}] {msg}", flush=True)


def build_prompt(tok, turns: list[str], replies: list[str]) -> str:
    """契约渲染（复用 O-1 的 render_prompt_text）；多轮时把已完成轮以摘要形式垫底
    （与 F-5 壳层同思路：历史进 prompt）。"""
    from probe_opd_divergence import render_prompt_text
    if len(turns) == 1:
        return render_prompt_text(tok, turns[0], tools=None)
    hist = "\n".join(f"[上一轮] 用户：{t}\n[上一轮] 助手：{r[:200]}"
                     for t, r in zip(turns[:-1], replies))
    return render_prompt_text(tok, f"{hist}\n\n{turns[-1]}", tools=None)


@torch.no_grad()
def gen_batch(student, tok, prompts: list[str], max_new: int, temp: float,
              top_p: float, top_k: int) -> list[str]:
    """学生 on-policy 采样（左 pad 批量生成）。"""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(student.device)
    out = student.generate(**enc, max_new_tokens=max_new, do_sample=True,
                           temperature=temp, top_p=top_p,
                           top_k=(top_k if top_k > 0 else 0) or None,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    texts = []
    for i in range(len(prompts)):
        texts.append(tok.decode(out[i][enc.input_ids.shape[1]:],
                                skip_special_tokens=True))
    return texts


def kl_step(student, aux, tok, prompt: str, reply: str, aux_dev: str,
            zero_mask: bool = False) -> tuple[float, int]:
    """单样本：掩码反向 KL + 立即 backward。返回 (sum_kl, masked_tokens)。"""
    from probe_opd_divergence import segment_text
    ids_p = tok(prompt, return_tensors="pt").input_ids
    r_ids, r_labs = segment_text(tok, reply)
    if not r_ids:
        return 0.0, 0
    mask = torch.tensor([1.0 if (l == "text" and not zero_mask) else 0.0
                         for l in r_labs])
    if mask.sum() == 0 and not zero_mask:
        return 0.0, 0
    ids = torch.cat([ids_p, torch.tensor([r_ids])], dim=1)
    kr = len(r_ids)
    s_out = student(ids.to(student.device), logits_to_keep=kr + 1).logits[0, :-1]
    with torch.no_grad():
        t_out = aux(ids.to(aux_dev), logits_to_keep=kr + 1).logits[0, :-1]
    ls = torch.log_softmax(s_out.float(), -1)
    lt = torch.log_softmax(t_out.float().to(student.device), -1)
    kl = (ls.exp() * (ls - lt)).sum(-1)
    m = mask.to(student.device)
    loss = (kl * m).sum()
    if loss.requires_grad:
        loss.backward()
    return float(loss.item()), int(m.sum().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/u_route/p1_prompts.jsonl")
    ap.add_argument("--out", default="checkpoints/opd/p1_r1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8, help="每 rank 每步样本数")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--probe-every", type=int, default=20, help="零掩码对照断言间隔")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    aux_gpus = os.environ.get("OPD_AUX_GPUS", "2,3").split(",")
    aux_dev = f"cuda:{aux_gpus[rank]}"
    # ⚠️ 卡号是 CUDA_VISIBLE 重映射后的索引：启动令 CUDA_VISIBLE_DEVICES=0,3,1,2 下
    #   可见 0=物理0(rank0 学生) 1=物理3(rank1 学生) 2=物理1(rank0 教师) 3=物理2(rank1 教师)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syncopate.train.rollout_budget import (SAMPLING_TEMPERATURE,
                                                SAMPLING_TOP_K, SAMPLING_TOP_P)

    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log("加载学生…")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16, device_map={"": rank})
    student = PeftModel.from_pretrained(student, ADAPTER, is_trainable=True)
    log("加载教师（底座）与锚（候选冻结）…")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_BASE, torch_dtype=torch.bfloat16,
        device_map={"": aux_dev}).eval()
    anchor_base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16,
        device_map={"": aux_dev})
    anchor = PeftModel.from_pretrained(anchor_base, ADAPTER,
                                       is_trainable=False).eval()

    rows = [json.loads(x) for x in open(args.prompts)]
    rows = rows[rank::world]               # rank 分片
    trainables = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=args.lr)

    wb = None
    if rank == 0 and not args.no_wandb:
        import wandb
        wb = wandb.init(project="syncopate", name=f"u_opd_{Path(args.out).name}",
                        config=vars(args))

    step = 0
    saved: list[Path] = []
    for ep in range(args.epochs):
        import random
        random.Random(100 + ep).shuffle(rows)
        for i in range(0, len(rows), args.batch):
            batch = rows[i: i + args.batch]
            t0 = time.time()
            # ① on-policy 采样（逐轮：多轮 prompt 先生成前轮回复垫底）
            samples = []                   # (prompt, reply, family)
            for r in batch:
                replies: list[str] = []
                for tI in range(len(r["turns"])):
                    prm = build_prompt(tok, r["turns"][: tI + 1], replies)
                    rep = gen_batch(student, tok, [prm], args.max_new,
                                    SAMPLING_TEMPERATURE, SAMPLING_TOP_P,
                                    SAMPLING_TOP_K)[0]
                    replies.append(rep)
                samples.append((prm, replies[-1], r["family"]))
            # ② 掩码反 KL（双教师路由），逐样本 backward
            opt.zero_grad(set_to_none=True)
            kl_chat = kl_task = 0.0
            n_chat = n_task = tok_chat = tok_task = 0
            for prm, rep, fam in samples:
                aux = teacher if fam == "chat" else anchor
                s, m = kl_step(student, aux, tok, prm, rep, aux_dev)
                if fam == "chat":
                    kl_chat += s; tok_chat += m; n_chat += 1
                else:
                    kl_task += s; tok_task += m; n_task += 1
            total_masked = tok_chat + tok_task
            # ⚠️ 判据分两层（首跑 rank1 全 task 批被误杀的学费）：
            #   chat 样本有回复却零掩码 = 分段器病 ⇒ 停机；
            #   全批只有 task 且回复=工具 JSON（无 reply 可蒸）= 合法 ⇒ 跳步记数
            if n_chat > 0 and tok_chat == 0:
                log(f"[opd-mask] 🔴 step {step} chat 样本掩码为零——分段器/生成出问题，停机自查")
                raise RuntimeError("chat-zero mask batch")
            # 跳步必须集体决定（单 rank 跳而对端进 all_reduce = 死锁）
            gm = torch.tensor([float(total_masked)], device=f"cuda:{rank}")
            dist.all_reduce(gm)
            if gm.item() == 0:
                if rank == 0:
                    log(f"[opd-mask] step {step} 全局无可蒸 token（全 task 工具回复），集体跳步")
                    if wb:
                        wb.log({"opd/skipped_steps": 1}, step=step)
                opt.zero_grad(set_to_none=True)
                step += 1
                dist.barrier()
                continue
            # DDP 梯度手动 allreduce（模型未包 DDP——逐样本 backward 与 PEFT 包装更省心）
            for p in trainables:
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
                p.grad = g
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step()
            step += 1
            dt = time.time() - t0
            if rank == 0:
                m_chat = kl_chat / max(tok_chat, 1)
                m_task = kl_task / max(tok_task, 1)
                log(f"step {step} ep{ep} kl_chat/tok={m_chat:.4f} "
                    f"kl_task/tok={m_task:.4f} masked={total_masked} "
                    f"[opd-route] chat={n_chat} task={n_task} {dt:.1f}s")
                if wb:
                    wb.log({"opd/kl_chat_per_tok": m_chat,
                            "opd/kl_task_per_tok": m_task,
                            "opd/masked_tokens": total_masked,
                            "opd/route_chat": n_chat, "opd/route_task": n_task,
                            "opd/step_time_s": dt, "opd/epoch": ep}, step=step)
            # ③ 零泄漏对照断言（守则②：假设写成断言）
            if step % args.probe_every == 0:
                opt.zero_grad(set_to_none=True)
                kl_step(student, teacher, tok, samples[0][0], samples[0][1],
                        aux_dev, zero_mask=True)
                bad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                          for p in trainables)
                assert bad == 0, f"[opd-zero] 零掩码对照有 {bad} 张量带梯度"
                if rank == 0:
                    log(f"[opd-zero] step {step} 对照通过（0/{len(trainables)}）")
                opt.zero_grad(set_to_none=True)
            # ④ adapter-only 滚动存档
            if rank == 0 and step % args.save_every == 0:
                pth = Path(args.out) / f"step_{step}"
                student.save_pretrained(pth)
                saved.append(pth)
                while len(saved) > 3:
                    shutil.rmtree(saved.pop(0), ignore_errors=True)
                log(f"[opd-ckpt] {pth}（adapter-only，保留最近 3）")
            dist.barrier()
    if rank == 0:
        pth = Path(args.out) / "final"
        student.save_pretrained(pth)
        log(f"[opd-ckpt] final → {pth}")
        if wb:
            wb.finish()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
