#!/usr/bin/env python
"""E26 §6.3 脱 Ray 最小复现：Adam `expected dtype c10::BFloat16 for 'end' but got dtype float`。

真实训练里 16 次尝试都在用「起训练」当调试循环（一轮 ~5 分钟，还要抢 4 张卡）。
这里把问题降维成：**单进程 + 迷你 Qwen3 + 真实的接线路径**，一轮 ~30 秒。

镜像自真实跑（outputs/2026-08-19/10-04-29/.hydra/config.yaml）：
  model_dtype=bf16 · FSDP1 · use_orig_params=False · NO_SHARD（E21 修复后的形态）
  MixedPrecision(param=bf16, reduce=fp32, buffer=fp32) · LoRA all-linear r32
  gradient_checkpointing(use_reentrant=False) · FA2 · torch.optim.AdamW
  前向路径 = verl_patches._patch_prefix_grouper 装出来的那个 forward_step 本体（不是复刻）

三臂：
  control   root FSDP forward + autocast + logits 损失      —— 已知必然通过的对照
  ours      打补丁后的 forward_step（当前卡在 AdamW 的那条） —— 预期复现 fp32 梯度
  rootfwd   同 ours 的数学，但 hidden 走 **根 FSDP 模块** 拿   —— 隔离「绕过根」这一个变量

判据（全部只在终态读）：
  ① 对照计数：params_with_grad 必须 > 0（=0 说明这臂根本没跑到 backward）
  ② mismatch 表：所有 (param.dtype != grad.dtype) 的参数逐个打印
  ③ AdamW.step() 的结果：OK / 原样的 RuntimeError
  ④ log_probs 的 fp32 求和 + 前 4 个值 —— 跨臂/跨修复比对等价性用
"""

import argparse
import os
import sys

os.environ.setdefault("SYNCOPATE_PREFIX_GROUPER", "1")

import torch
import torch.distributed as dist


def build_model(device, tie=False, rmpad=False):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=4096,
        tie_word_embeddings=tie,      # 真实模型 Qwen3-4B 是绑定的
        attn_implementation="flash_attention_2",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)

    # ★ 与真实顺序一致：monkey_patch 在 LoRA/FSDP 之前、对 HF 模型施加
    #   真实跑：use_remove_padding=True + use_fused_kernels=True
    from verl.models.transformers import monkey_patch as _mp
    _mp.apply_monkey_patch(model, use_remove_padding=rmpad, use_fused_kernels=rmpad,
                           fused_kernels_backend="torch" if rmpad else None)

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=32, target_modules="all-linear"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(module=model, config=None, is_lora=True),
        device_id=device,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
        sync_module_states=True,
        use_orig_params=False,
    )
    return fsdp_model


def build_micro_batch(device, varlen=False, seed=1):
    """2 组 × 4 条同题面，P=64 / R=32。varlen=True 时镜像真实批：
    prompt 左填充（组内一致）、response 右填充（逐条不同）。"""
    from tensordict import TensorDict

    torch.manual_seed(seed)
    P, R, G, N = 64, 32, 2, 4
    prompts = torch.randint(3, 1000, (G, P)).repeat_interleave(N, dim=0)
    responses = torch.randint(3, 1000, (G * N, R))
    am = torch.ones(G * N, P + R, dtype=torch.long)
    if varlen:
        pad_left = torch.randint(0, P // 2, (G,)).repeat_interleave(N)   # 组内同 prompt ⇒ 同左 pad
        resp_len = torch.randint(4, R + 1, (G * N,))
        for i in range(G * N):
            am[i, :pad_left[i]] = 0
            am[i, P + resp_len[i]:] = 0
            prompts[i, :pad_left[i]] = 0
            responses[i, resp_len[i]:] = 0
    return TensorDict({
        "prompts": prompts,
        "responses": responses,
        "response_mask": am[:, P:].clone(),
        "attention_mask": am,
    }, batch_size=[G * N]).to(device)


def fake_loss(model_output, data, dp_group=None):
    lp = model_output["log_probs"]
    vals = lp.values() if lp.is_nested else lp
    return vals.to(torch.float32).mean(), {}


def scan_grads(fsdp_model):
    total, mismatch = 0, []
    for name, p in fsdp_model.named_parameters():
        if p.grad is None:
            continue
        total += 1
        if p.grad.dtype != p.dtype:
            mismatch.append((name, str(p.dtype), str(p.grad.dtype)))
    return total, mismatch


class FakeEngine:
    """forward_step 用到的 self 只有这四样。"""
    def __init__(self, module):
        self.module = module
        self.pad_token_id = 0
        self._autocast_dtype = torch.bfloat16

    def get_data_parallel_group(self):
        return dist.group.WORLD


def run_control(fsdp_model, mb):
    input_ids = torch.cat([mb["prompts"], mb["responses"]], dim=1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = fsdp_model(input_ids=input_ids, attention_mask=mb["attention_mask"], use_cache=False)
        logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
        loss = -logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).mean()
    loss.backward()
    return loss, None


def run_ours(fsdp_model, mb):
    """真实卡点路径：直接调用补丁装出来的 forward_step 本体。"""
    from verl.workers.engine.fsdp import transformer_impl as _ti
    fs = _ti.FSDPEngineWithLMHead.forward_step
    eng = FakeEngine(fsdp_model)
    loss, meta = fs(eng, mb, fake_loss, forward_only=False)
    loss.backward()
    return loss, meta


def run_rootfwd(fsdp_model, mb):
    """同 ours 的打包数学，但 hidden 从**根 FSDP 模块的 forward** 里拿（hook 捕获），
    logits_to_keep=1 让 lm_head 只投影最后一个位置（防止真实尺寸下物化全量 logits）。"""
    from verl.trainer.ppo import prefix_grouper_utils as _pgu
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    orig_pg_forward = _pgu.pg_forward

    def pg_forward_via_root(model, prefix_grouper, concat_input_ids, attention_mask, position_ids,
                            completion_ids, completion_mask, *, temperature=1.0,
                            padding_mode="right", include_prefix_last=1,
                            calculate_entropy=False, entropy_fn=None):
        pg_forward_via_root._ran = True
        base = model
        for _ in range(6):
            inner = getattr(base, "model", None)
            if inner is None or inner is base:
                break
            base = inner
        captured = []
        h = base.register_forward_hook(lambda m, i, o: captured.append(o[0] if isinstance(o, tuple) else o.last_hidden_state))
        try:
            out = model(input_ids=concat_input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=False,
                        prefix_grouper=prefix_grouper, logits_to_keep=1)
        finally:
            h.remove()
        hidden = captured[0]
        # ★ 锚：让损失流经**根 FSDP forward 的输出**。FSDP1 把 pre-backward hook 注册在根
        #   输出张量上，final callback（梯度归约/收尾）由它排队 —— 损失不经过根输出，
        #   整套 post-backward 机制静默不跑（探针①实测：梯度不跨 rank 归约）。
        anchor = out.logits.float().sum() * 0.0
        _, _, suffix_h, suffix_mask_raw = prefix_grouper.split_output(
            hidden, include_prefix_last=include_prefix_last)
        completion_right = prefix_grouper.convert_padding(
            completion_ids, completion_mask, padding_mode=padding_mode)
        hh = suffix_h[:, :-1].contiguous()
        B, T, D = hh.shape
        lm_w = None
        m = model
        for _ in range(6):
            fn = getattr(m, "get_output_embeddings", None)
            if callable(fn):
                emb = fn()
                if emb is not None and hasattr(emb, "weight"):
                    lm_w = emb.weight
                    break
            m = getattr(m, "module", None) or getattr(m, "base_model", None) or getattr(m, "model", None)
        log_probs, entropy = FusedLinearForPPO()(
            hidden_states=hh.view(B * T, D), vocab_weights=lm_w,
            input_ids=completion_right.reshape(B * T), temperature=temperature)
        return log_probs.view(B, T) + anchor, entropy.view(B, T), suffix_mask_raw[:, 1:]

    # ★ 先空跑一次 forward_only：补丁的 forward_step 在**第一次调用**时会把闭包里的
    #   _pending 弹进 _pgu（覆盖 pg_forward）——不排干的话我们的覆盖会被它顶掉，
    #   这臂就静默变回 ours（之前的 rootfwd 结果全部因此作废）。
    from verl.workers.engine.fsdp import transformer_impl as _ti
    with torch.no_grad():
        _ti.FSDPEngineWithLMHead.forward_step(FakeEngine(fsdp_model), mb, None, forward_only=True)

    _pgu.pg_forward = pg_forward_via_root
    try:
        loss, meta = run_ours(fsdp_model, mb)
        assert getattr(pg_forward_via_root, "_ran", False), "rootfwd 覆盖没有生效（又被顶掉了？）"
        return loss, meta
    finally:
        _pgu.pg_forward = orig_pg_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "ours", "rootfwd"], required=True)
    ap.add_argument("--tie", action="store_true", help="tied embeddings（真实模型是）")
    ap.add_argument("--rmpad", action="store_true", help="monkey_patch 开 rmpad+fused_kernels（真实跑是）")
    ap.add_argument("--varlen", action="store_true", help="变长+padding 的批（真实批是）")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积几个 micro-batch")
    ap.add_argument("--statedict-before", action="store_true",
                    help="前向前先做一次 SHARDED state_dict 收集（镜像 adapter 同步首推）")
    ap.add_argument("--rank-data", action="store_true",
                    help="每 rank 喂不同数据 + 校验梯度是否真的被 all-reduce（E21 判据）")
    ap.add_argument("--probe-root", action="store_true", help="打印哪些 FSDP 单元自封了 _is_root")
    ap.add_argument("--probe-hooks", action="store_true",
                    help="数 post-backward hook 的注册/触发/归约次数（找哪一环没接上）")
    args = ap.parse_args()

    hook_counts = {"register": 0, "fire": 0, "skip_sync": 0, "reduce": 0,
                   "final_queued": 0, "final_ran": 0, "prebw": 0}
    if args.probe_hooks:
        import torch.distributed.fsdp._runtime_utils as _ru
        _orig_reg = _ru._register_post_backward_hook
        _orig_hook = _ru._post_backward_hook
        _orig_reduce = _ru._reduce_grad_no_shard
        _orig_qfinal = _ru._register_post_backward_final_callback
        _orig_final = _ru._post_backward_final_callback
        _orig_prebw = _ru._pre_backward_hook

        def _qfinal(*a, **k):
            hook_counts["final_queued"] += 1
            return _orig_qfinal(*a, **k)

        def _final(*a, **k):
            hook_counts["final_ran"] += 1
            return _orig_final(*a, **k)

        def _prebw(*a, **k):
            hook_counts["prebw"] += 1
            return _orig_prebw(*a, **k)

        _ru._register_post_backward_final_callback = _qfinal
        _ru._post_backward_final_callback = _final
        _ru._pre_backward_hook = _prebw

        _orig_regpre = _ru._register_pre_backward_hooks

        def _regpre(state, module, outputs, handle):
            if getattr(state, "_is_root", None):
                n = [0]
                def _cnt(t):
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        n[0] += 1
                    return t
                from torch.distributed.utils import _apply_to_tensors
                _apply_to_tensors(_cnt, outputs)
                if dist.get_rank() == 0:
                    print(f"[repro] 探针④ 根单元注册 pre-backward：requires_grad 的输出张量 {n[0]} 个 "
                          f"（module={type(module).__name__}）", flush=True)
            return _orig_regpre(state, module, outputs, handle)

        _ru._register_pre_backward_hooks = _regpre

        def _reg(*a, **k):
            hook_counts["register"] += 1
            return _orig_reg(*a, **k)

        def _hook(state, handle, flat_param, *a):
            hook_counts["fire"] += 1
            if not state._sync_gradients:
                hook_counts["skip_sync"] += 1
            return _orig_hook(state, handle, flat_param, *a)

        def _red(*a, **k):
            hook_counts["reduce"] += 1
            return _orig_reduce(*a, **k)

        _ru._register_post_backward_hook = _reg
        _ru._post_backward_hook = _hook
        _ru._reduce_grad_no_shard = _red

    # 单进程直接跑 = world 1；torchrun --nproc-per-node=N 跑 = 多 rank
    # （真实跑是 3 个 trainer rank —— NO_SHARD 的梯度 all-reduce 只在 world>1 时发生，
    #   而 fp32 升格如果来自 reduce_dtype，world=1 下会整个短路 ⇒ 两种都要能跑）
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ★ 装真实补丁（不是复刻）：先替换 wrapper 工厂，再由 apply_monkey_patch 触发 _wire_engine
    from syncopate.train import verl_patches
    verl_patches._patch_prefix_grouper()

    fsdp_model = build_model(local_rank, tie=args.tie, rmpad=args.rmpad)
    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=3e-5)

    if args.statedict_before:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
        if dist.get_world_size() > 1:
            FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())
        _sd = fsdp_model.state_dict()
        print(f"[repro] state_dict 收集完成（{len(_sd)} 个键）", flush=True)
        del _sd

    run = {"control": run_control, "ours": run_ours, "rootfwd": run_rootfwd}[args.arm]
    loss = meta = None
    for k in range(args.accum):
        seed = 1 + k + (dist.get_rank() * 100 if args.rank_data else 0)
        mb = build_micro_batch(torch.device("cuda", local_rank), varlen=args.varlen, seed=seed)
        loss, meta = run(fsdp_model, mb)

    # ── 探针①：梯度有没有真的跨 rank 归约（E21 的判据形状：归约后应逐位相同）──
    if args.rank_data:
        gsum = torch.zeros(1, device="cuda")
        for p in fsdp_model.parameters():
            if p.grad is not None:
                gsum += p.grad.float().sum()
        gathered = [torch.zeros_like(gsum) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, gsum)
        if dist.get_rank() == 0:
            vals = [g.item() for g in gathered]
            same = all(abs(v - vals[0]) < 1e-9 for v in vals)
            print(f"[repro] 探针① 各 rank 梯度和 {vals} ⇒ "
                  f"{'✅ 逐位相同（归约发生了）' if same else '🔴 不同（梯度没有跨 rank 归约！）'}",
                  flush=True)

    if args.probe_hooks and dist.get_rank() == 0:
        print(f"[repro] 探针③ post-backward：注册 {hook_counts['register']} · 触发 {hook_counts['fire']} · "
              f"因 _sync_gradients=False 跳过 {hook_counts['skip_sync']} · 真正归约 {hook_counts['reduce']} · "
              f"pre-backward {hook_counts['prebw']} · final排队 {hook_counts['final_queued']} · "
              f"final执行 {hook_counts['final_ran']}", flush=True)

    # ── 探针②：每个 FSDP 单元的 _is_root（谁自封了根）──
    if args.probe_root and dist.get_rank() == 0:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        roots = []
        for name, m in fsdp_model.named_modules():
            if isinstance(m, FSDP) and getattr(m, "_is_root", None):
                roots.append(name or "<outermost>")
        print(f"[repro] 探针② _is_root=True 的单元（{len(roots)} 个）: {roots[:6]}", flush=True)

    if dist.get_rank() != 0:
        # 非 0 rank 只陪跑（NO_SHARD all-reduce 需要它们），判据只在 rank 0 读
        try:
            opt.step()
        except RuntimeError:
            pass
        dist.destroy_process_group()
        return

    if meta is not None:
        lp = meta["model_output"]["log_probs"]
        vals = lp.values() if lp.is_nested else lp
        v = vals.to(torch.float32)
        print(f"[repro] log_probs fp32 sum={v.sum().item():.6f} 前4值={[round(x, 6) for x in v[:4].tolist()]}")

    total, mismatch = scan_grads(fsdp_model)
    print(f"[repro] arm={args.arm} loss={loss.item():.6f} params_with_grad={total}（对照计数，必须>0）")
    for name, pd, gd in mismatch:
        print(f"[repro]   MISMATCH {name}: param={pd} grad={gd}")
    print(f"[repro] mismatched={len(mismatch)}/{total}")

    try:
        opt.step()
        print(f"[repro] 终态 arm={args.arm}: AdamW.step() OK · mismatched={len(mismatch)}")
    except RuntimeError as e:
        print(f"[repro] 终态 arm={args.arm}: AdamW.step() RuntimeError: {e}")
        sys.exit(2)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
