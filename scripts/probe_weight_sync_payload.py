#!/usr/bin/env python3
"""0-B 第一问 · 权重同步**到底推的是什么**？（基座 / LoRA / 两者）

★ 为什么先问这个，而不是直接问"两边一不一样"：
读发送侧代码时看到一条可疑分支 ——

    engine_workers.py:698（**disaggregated / fully_async 这条路**）
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   ← **不传任何参数**
        await self.checkpoint_engine.send_weights(per_tensor_param, ...)

    而默认参数是 `base_sync_done=False`，它在 LoRA 下会走：
    fsdp_utils.collect_lora_params(base_sync_done=False)
        for name, param in model.state_dict().items():
            if any(x in name for x in ["_flat_param", "lora_"]): continue   ← **跳过 lora_**

    对照 naive（colocate）那条路（engine_workers.py:711-731）**调了两次**：
        base_sync_done=False → 先推基座；base_sync_done=True → 再推 adapter

⇒ **[推断，未验证]** fully_async 下可能每次只推基座、从不推 LoRA
   ⇒ 若成立，rollout 用来生成数据的策略**永远停在起点**，而 trainer 一直在走远。

⚠️ **但有一条反证**：E12 记的是「实际只推 132 MB」（≈ 66M×2B = LoRA 的大小，不是基座的 8 GB）。
   两条证据打架 ⇒ **不许再推，必须量。**

本脚本用**小模型 + 单卡 FSDP** 离线复现发送侧那一步，不依赖 Ray / 不起训练：
直接调 verl 自己的 `collect_lora_params`，看两种 `base_sync_done` 各吐出什么。

判据（"某集合应当完整"型，不设阈值）：
    推出去的张量集合里，**必须包含 LoRA 的增量**（名字含 lora_A/lora_B，或已被合并进基座）
    否则 rollout 侧拿到的就不是当前策略。
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def summarize(tag: str, params) -> dict:
    items = list(params.items()) if hasattr(params, "items") else list(params)
    n_bytes = sum(v.numel() * v.element_size() for _, v in items)
    lora_names = [k for k, _ in items if "lora_" in k.lower()]
    print(f"\n  【{tag}】")
    print(f"    张量个数      {len(items)}")
    print(f"    总字节        {n_bytes / 2**20:,.1f} MiB")
    print(f"    含 'lora_' 的  {len(lora_names)} 个")
    print(f"    名字样例      {[k for k, _ in items[:3]]}")
    if lora_names:
        print(f"    lora 样例     {lora_names[:3]}")
    return {"n": len(items), "mib": n_bytes / 2**20, "lora": len(lora_names)}


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "30041")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.utils.fsdp_utils import collect_lora_params

    # 小号 Qwen3（同一个模型家族，同一套模块名）—— 秒级构建，不下载权重
    cfg = AutoConfig.from_pretrained("models/Qwen3-4B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.tie_word_embeddings = True
    base = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).cuda()

    # 和正式跑同一套 target_modules（launch_rl 的 dense 默认）
    peft_model = get_peft_model(base, LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    # ★ 把 lora_B 设成非零，否则"有没有推 LoRA"在数值上看不出来（B 零初始化 ⇒ 增量恒为 0）
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)

    # PEFT 默认把 LoRA 建成 fp32，而基座是 bf16 ⇒ FSDP 拒绝混合 dtype 打平。
    # 本探针只问"哪些名字被推出去"，与 dtype 无关 ⇒ 统一成 bf16。
    peft_model = peft_model.to(torch.bfloat16)

    module = FSDP(peft_model, sharding_strategy=ShardingStrategy.NO_SHARD,
                  use_orig_params=True, device_id=0)

    print("\n" + "=" * 96)
    print("  发送侧 collect_lora_params 各分支吐出什么（LoRA + merge=False，与正式跑一致）")
    print("=" * 96)

    # ① fully_async 走的那条：get_per_tensor_param() 不传参 ⇒ base_sync_done=False
    a = summarize("① base_sync_done=False　← **fully_async / disaggregated 实际走的分支**",
                  collect_lora_params(module=module, layered_summon=False, base_sync_done=False))
    # ② colocate 第二次调用走的那条
    b = summarize("② base_sync_done=True 　← colocate 会再调一次，推 adapter",
                  collect_lora_params(module=module, layered_summon=False, base_sync_done=True))

    print("\n" + "=" * 96)
    print(f"  判据：fully_async 那一路（①）推出去的张量里，含 'lora_' 的有 {a['lora']} 个")
    if a["lora"] == 0:
        print("  🔴 **零个** ⇒ 该分支只推基座，**LoRA 增量根本没被推给 rollout**")
        print("     ⇒ 若 fully_async 确实只调用①一次，rollout 的策略会永远停在起点。")
        print("     ⚠️ 下一步必须在**真实跑**里确认调用次数与分支（本脚本只证明了分支的行为）。")
    else:
        print("  ✅ 含 LoRA ⇒ 该分支本身没问题，0-B 的重点回到「两侧内容是否一致」。")
    print(f"  对照：①{a['n']} 个张量 / {a['mib']:,.1f} MiB　②{b['n']} 个 / {b['mib']:,.1f} MiB")
    print("=" * 96)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
