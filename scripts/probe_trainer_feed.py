#!/usr/bin/env python3
"""E25 · trainer 到底是「活重」还是「没喂饱」—— 隔离出训练侧的前向/反向,单卡可跑。

★ 为什么单独写一个探针（2026-08-19）
   生产里 trainer 的一步被三件事混在一起（update_actor / old_log_prob / ref），
   而我们想测的两个开关（gradient_checkpointing / micro_batch）只作用在计算本身。
   用真跑去测要 4 卡 + Ray + vLLM，每档 15 分钟且**不能并行**；
   隔离出来之后 1 卡一档、4 档可以同时跑在 4 张卡上。

★ 保真度自检（这份脚本的判据）
   生产实测：update_actor 17.56 s / 每 global step，48 条序列分 3 卡 ⇒ **每卡 16 条**。
   ⇒ 本脚本 baseline（gc=on, micro_batch=1, 16 条）应当落在 17.56 s 附近。
     **落不上就说明这个探针不保真，本次所有相对结论作废** —— 不许只报相对值。

⚠️ 与生产的已知差异（读数时必须记得）：
   ① 没有 FSDP/DDP 包装（生产是 fsdp_size=1 = 全量复制，量级应接近，但不是同一件事）
   ② 用等长序列代替 rmpad 打包（等长 ⇒ 没有 padding 浪费，与 rmpad 的效果同向）
   ③ 损失用的是 PG 形状的代理损失，不是 verl 的真损失（只影响反向的最后一小段）
"""
from __future__ import annotations
import argparse, json, os, time
import torch

def build_batch(n_seq, prompt_len, resp_lens, vocab, device):
    seqs = []
    for i in range(n_seq):
        L = prompt_len + resp_lens[i]
        ids = torch.randint(0, vocab, (L,), device=device)
        seqs.append(ids)
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-4B-sft-v13-e1")
    ap.add_argument("--gc", choices=["on", "off"], default="on")
    ap.add_argument("--micro-batches", default="1,2,4,8")
    ap.add_argument("--n-seq", type=int, default=16)       # = 生产每卡 48/3
    ap.add_argument("--prompt-len", type=int, default=4196) # = 实测 prompt_length/mean
    ap.add_argument("--resp-len", type=int, default=654)    # = 实测 response_length/mean
    ap.add_argument("--varlen", action="store_true", help="response 用真实分布而不是定长")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import LoraConfig, get_peft_model

    dev = "cuda"
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    vocab = cfg.vocab_size
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    if a.gc == "on":
        # ⚠️ use_reentrant=True（默认）下，若输入不 requires_grad，整段 backward 会被跳过
        #    ⇒ LoRA 梯度全是 None，而只打一行 UserWarning。这正是 E21 那个形状。
        #    ⇒ 两道保险：非 reentrant + enable_input_require_grads，并在下面**断言**。
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    if a.varlen:
        g = torch.Generator().manual_seed(0)
        rl = (torch.randn(a.n_seq, generator=g) * 250 + a.resp_len).clamp(30, 1536).int().tolist()
    else:
        rl = [a.resp_len] * a.n_seq

    results = []
    for mb in [int(x) for x in a.micro_batches.split(",")]:
        if a.n_seq % mb != 0:
            continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            fb_times, fo_times = [], []
            for it in range(a.warmup + a.iters):
                torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                for s in range(0, a.n_seq, mb):
                    lens = rl[s:s+mb]
                    L = a.prompt_len + max(lens)
                    ids = torch.randint(0, vocab, (mb, L), device=dev)
                    am = torch.ones_like(ids)
                    for j, ln in enumerate(lens):
                        am[j, a.prompt_len+ln:] = 0
                    out = model(input_ids=ids, attention_mask=am).logits
                    lp = torch.log_softmax(out[:, a.prompt_len-1:-1].float(), -1)
                    tgt = ids[:, a.prompt_len:]
                    sel = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    m = am[:, a.prompt_len:].float()
                    loss = -(sel * m).sum() / m.sum() / (a.n_seq // mb)
                    loss.backward()
                    del out, lp, sel
                # ★ 断言：反向必须真的产生了梯度（成本≈0，但它挡住"量到一个空操作"）
                if it == 0:
                    tp = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                    ng = [n for n, p in tp if p.grad is None]
                    nz = sum(1 for _, p in tp if p.grad is not None and p.grad.abs().sum() > 0)
                    assert not ng, f"[判据] {len(ng)}/{len(tp)} 个可训练参数梯度为 None（首个 {ng[0]}）"
                    assert nz > 0, "[判据] 所有梯度都是 0 —— 反向是空操作"
                    print(f"[判据] 梯度检查 ✓ 可训练张量 {len(tp)} 个，非零梯度 {nz} 个", flush=True)
                opt.step()
                torch.cuda.synchronize(); t1 = time.time()

                with torch.no_grad():
                    for s in range(0, a.n_seq, mb):
                        lens = rl[s:s+mb]
                        L = a.prompt_len + max(lens)
                        ids = torch.randint(0, vocab, (mb, L), device=dev)
                        am = torch.ones_like(ids)
                        for j, ln in enumerate(lens):
                            am[j, a.prompt_len+ln:] = 0
                        model(input_ids=ids, attention_mask=am)
                torch.cuda.synchronize(); t2 = time.time()
                if it >= a.warmup:
                    fb_times.append(t1-t0); fo_times.append(t2-t1)
            import statistics as st
            r = dict(micro_batch=mb, gc=a.gc, varlen=a.varlen,
                     fwd_bwd_s=round(st.median(fb_times), 3),
                     fwd_only_s=round(st.median(fo_times), 3),
                     peak_gb=round(torch.cuda.max_memory_allocated()/2**30, 2), ok=True)
        except torch.cuda.OutOfMemoryError:
            r = dict(micro_batch=mb, gc=a.gc, varlen=a.varlen, ok=False, err="OOM")
            torch.cuda.empty_cache()
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
        if not r["ok"]:
            break
    if a.out:
        json.dump(dict(args=vars(a), results=results), open(a.out, "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
