"""U 路 P1 · On-Policy Distillation 训练器（现行说明见
`docs/syncopate/04-TRAINING.md`；历史方案见
`docs/archive/syncopate/pre-consolidation-v16/24-unified-conversation-training.md`）。

    CUDA_VISIBLE_DEVICES=0,1 OPD_AUX_GPUS=1 python -m torch.distributed.run \
        --nproc_per_node=1 -m syncopate.train.opd --base <本轮合并 SFT> \
        --adapter <本轮 RL adapter> --out checkpoints/opd/<run>

当前 B200×2 布局：学生在 GPU0；教师和冻结锚在 GPU1。历史双 rank
布局已经归档，不能套到当前默认入口。

机制（P0-4 spike 三判据的生产版，判据行全部常驻）：
  on-policy：学生自己 generate（契约渲染+契约采样参数，多轮取最后一轮回复）
  掩码：v15 think / tool / 纯自然语言三分；只蒸纯自然语言——[opd-mask] 每批非零断言
  双教师路由：chat→底座 · task/task_neg→候选冻结锚——[opd-route] 计数入 wandb
  损失：逐 token 反向 KL(学生‖教师)，逐样本累积总和、更新前除以全局有效 token 数
  零泄漏断言：每 --probe-every 步跑一次零掩码对照，LoRA 梯度必须逐位为零

存储：adapter-only（peft save_pretrained，E29 口径），滚动保留最近 3 份 + final。
wandb：project=syncopate（与 sft/launch_rl 同族默认开），指标前缀 opd/*。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_OPD_PROMPTS
from syncopate.train.token_objective import assert_replicated_parameters, finish_token_window
from syncopate.train.opd_tokens import (ByteLevelTokenMapper, GeneratedSample,
    TokenAuditWriter, capture_generated_sample, positions_from_mask)
from syncopate.train.opd_updates import OptimizerUpdateAudit

sys.path.insert(0, ".")

STUDENT_BASE = STUDENT_MODEL   # v16：学生底座（LoRA 另挂）
# 裁定⑭（09-04）：不再写死 v13 产物 cand_v13r2_e1；adapter 由 --adapter 传（v16 的 SFT/RL 产物），冒烟可不传 = 底座 + 新建 LoRA
TEACHER_BASE = TEACHER_MODEL


def log(msg: str) -> None:
    print(f"[opd r{os.environ.get('RANK', '0')}] {msg}", flush=True)


def training_completed(real_steps: int, target_real_steps: int) -> bool:
    """候选/冒烟都至少要有一次真实更新；短跑还必须达到登记的真实更新数。"""
    return real_steps > 0 and (target_real_steps <= 0 or real_steps >= target_real_steps)


def prioritize_smoke_routes(rows: list[dict]) -> list[dict]:
    """Put one task and one chat row first so a tiny smoke covers both routes.

    Candidate ordering is untouched.  The previous one-step smoke happened to draw
    four task-only batches before its first chat row, spending most of the run on
    legal zero-mask skips and never exercising the chat/teacher route until attempt
    five.  With a batch of two this deterministic prefix exercises anchor + teacher
    immediately; remaining chat rows are tried before task-only rows so a short
    smoke does not spend its whole attempt budget on valid zero-NL tool calls.
    """
    task_i = next((i for i, row in enumerate(rows) if row.get("family") == "task"), None)
    chat_i = next((i for i, row in enumerate(rows) if row.get("family") == "chat"), None)
    if task_i is None or chat_i is None:
        raise ValueError("OPD smoke 需要至少一条 task 和一条 chat，才能覆盖双教师路由")
    # First batch covers both route labels.  If that chat sample has no valid v15
    # prose, subsequent batches draw chat rows first instead of burning attempts on
    # task-only tool calls that correctly have a zero NL mask.
    selected = [rows[task_i], rows[chat_i]]
    chat_rest = [row for i, row in enumerate(rows)
                 if i not in {task_i, chat_i} and row.get("family") == "chat"]
    other_rest = [row for i, row in enumerate(rows)
                  if i not in {task_i, chat_i} and row.get("family") != "chat"]
    return selected + chat_rest + other_rest


def prior_result(reply: str, *, thinking_enabled: bool | None = None) -> dict:
    """Turn a generated terminal v15 response into Runtime's prior-result shape."""
    from syncopate.core.parsing_v15 import parse_step_v15
    from syncopate.train.opd_render import v15_char_labels
    from syncopate.train.rollout_budget import ENABLE_THINKING

    if thinking_enabled is None:
        thinking_enabled = ENABLE_THINKING
    parsed = parse_step_v15(reply, implicit_think_open=thinking_enabled)
    if thinking_enabled and parsed.kind == "error" and parsed.error == "empty_final_text":
        raise ValueError("上一轮思考段没有闭合，不能作为多轮历史回灌")
    if parsed.kind == "final_text":
        if not any(label == "text" for label in v15_char_labels(
                reply, implicit_think_open=thinking_enabled)):
            raise ValueError("上一轮没有合格的 v15 自然语言终答")
        return {"text": parsed.text}
    if parsed.kind == "signal":
        return {"text": parsed.text, "signal": parsed.signal,
                "arguments": dict(parsed.signal_args)}
    raise ValueError(f"上一轮没有形成可回灌终态：{parsed.kind}")


def build_prompt(tok, turns: list[str], replies: list[str], tools) -> str:
    """Use the same user template, real message-pair history, and tool menu as Runtime."""
    from syncopate.train.opd_render import render_prompt_text
    prior = [{"user_message": turn, "result": prior_result(reply)}
             for turn, reply in zip(turns[:-1], replies)]
    return render_prompt_text(tok, turns[-1], tools=tools, prior=prior)


@torch.no_grad()
def gen_batch(student, tok, prompts: list[str], max_new: int, temp: float,
              top_p: float, top_k: int, *, mapper: ByteLevelTokenMapper | None = None
              ) -> list[GeneratedSample]:
    """Current text-only, single-beam sampling; retain its exact padded inputs."""
    from syncopate.train.rollout_budget import ENABLE_THINKING

    if student.config.is_encoder_decoder or student.generation_config.num_beams != 1:
        raise ValueError("OPD raw-token capture only supports current decoder-only single-beam sampling")
    if student.generation_config.num_return_sequences != 1:
        raise ValueError("OPD requires one generated sequence per prompt")
    mapper = mapper if mapper is not None else ByteLevelTokenMapper(tok)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(student.device)
    if set(enc) != {"input_ids", "attention_mask"}:
        raise ValueError("OPD expects text-only input_ids and attention_mask")
    positions = torch.tensor([positions_from_mask(tuple(mask)) for mask in enc.attention_mask.tolist()],
                             dtype=torch.long, device=student.device)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    if pad_id is None:
        raise ValueError("OPD requires an explicit padding token ID")
    eos_ids = student.generation_config.eos_token_id
    eos_ids = () if eos_ids is None else (eos_ids,) if isinstance(eos_ids, int) else tuple(eos_ids)
    out = student.generate(**enc, position_ids=positions, max_new_tokens=max_new, do_sample=True,
                           temperature=temp, top_p=top_p,
                           top_k=(top_k if top_k > 0 else 0) or None,
                           eos_token_id=list(eos_ids) or None, pad_token_id=pad_id)
    if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != len(prompts):
        raise ValueError("unexpected generate output interface")
    records = []
    for i in range(len(prompts)):
        records.append(capture_generated_sample(mapper,
            prompt_ids=enc.input_ids[i].tolist(), prompt_attention_mask=enc.attention_mask[i].tolist(),
            prompt_position_ids=positions[i].tolist(), generated_ids=out[i].tolist(),
            eos_token_ids=eos_ids, pad_token_id=pad_id, implicit_think_open=ENABLE_THINKING))
    return records


def generation_runtime_evidence(student=None) -> dict:
    """Record the installed position/cache API, not an assumption from an old HF release."""
    import hashlib
    import inspect
    import importlib.metadata
    from transformers.generation.utils import GenerationMixin

    methods = {}
    for name in ("prepare_inputs_for_generation", "_prepare_position_ids_for_generation",
                 "_update_model_kwargs_for_generation"):
        source = inspect.getsource(getattr(GenerationMixin, name))
        methods[name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "source": source}
    evidence = {"versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch")},
        "position_policy": "text_only_cumsum_attention_minus_one_pad_zero",
        "generation_methods": methods}
    if student is not None:
        active = {}
        base = student.get_base_model() if hasattr(student, "get_base_model") else student
        for name, model in (("student", student), ("base", base)):
            source = inspect.getsource(model.prepare_inputs_for_generation)
            active[name] = {"class": type(model).__name__, "source": source,
                            "sha256": hashlib.sha256(source.encode()).hexdigest()}
        evidence["active_prepare_inputs"] = active
        if student is not base:
            evidence["versions"]["peft"] = importlib.metadata.version("peft")
    return evidence


def kl_step(student, aux, sample: GeneratedSample, aux_dev: str, *,
            zero_mask: bool = False, audit: dict | None = None) -> tuple[float, int]:
    """Score the actual sampled IDs under identical student/aux masks and positions.

    No string/tokenizer API exists here: decoding and re-encoding can change both
    BPE identity and normalization, even when the visible answer looks identical.
    """
    if not isinstance(sample, GeneratedSample):
        raise TypeError("kl_step requires a GeneratedSample, never prompt/reply strings")
    sample.validate()
    r_ids, r_labs = sample.response_ids, sample.alignment.labels
    active = [int(label == "text" and not zero_mask) for label in r_labs]
    if audit is not None:
        audit.update(student_forward=0, aux_forward=0, backward=0, reply_tokens=len(r_ids),
            response_ids=list(r_ids), response_labels=list(r_labs), response_mask=active,
            masked_tokens=sum(active), alignment_warnings=list(sample.alignment.warnings), loss_sum=0.0)
    if not r_ids:
        return 0.0, 0
    mask = torch.tensor(active, dtype=torch.float32)
    if mask.sum() == 0 and not zero_mask:
        return 0.0, 0
    kr = len(r_ids)
    inputs = {name: torch.tensor([getattr(sample, name)], dtype=torch.long)
              for name in ("input_ids", "attention_mask", "position_ids")}
    student_inputs = {name: value.to(student.device) for name, value in inputs.items()}
    s_out = student(**student_inputs, logits_to_keep=kr + 1).logits[0, :-1]
    if audit is not None:
        audit['student_forward'] += 1
        audit.update({f"student_{name}": value[0].detach().cpu().tolist()
                      for name, value in student_inputs.items()})
    aux_inputs = {name: value.to(aux_dev) for name, value in inputs.items()}
    with torch.no_grad():
        t_out = aux(**aux_inputs, logits_to_keep=kr + 1).logits[0, :-1]
    if audit is not None:
        audit['aux_forward'] += 1
        audit.update({f"aux_{name}": value[0].detach().cpu().tolist()
                      for name, value in aux_inputs.items()})
    if s_out.shape != t_out.shape or s_out.ndim != 2 or s_out.shape[0] != kr:
        raise ValueError("student/aux response logits do not have identical token/vocabulary dimensions")
    ls = torch.log_softmax(s_out.float(), -1)
    lt = torch.log_softmax(t_out.float().to(student.device), -1)
    kl = (ls.exp() * (ls - lt)).sum(-1)
    m = mask.to(student.device)
    loss = (kl * m).sum()
    if loss.requires_grad:
        loss.backward()
        if audit is not None:
            audit['backward'] += 1
    total = float(loss.item())
    if audit is not None:
        audit['loss_sum'] = total
    return total, int(m.sum().item())


def zero_mask_control(student, aux, sample: GeneratedSample, aux_dev: str) -> dict:
    """空回复/没有反向不能冒充零梯度；辅助模型也不得收到梯度。"""
    student.zero_grad(set_to_none=True)
    evidence = {}
    total, count = kl_step(student, aux, sample, aux_dev, zero_mask=True, audit=evidence)
    gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    evidence['gradients'] = len(gradients)
    finite = [bool(torch.isfinite(g).all()) for g in gradients]
    evidence['nonfinite_gradient_tensors'] = sum(not good for good in finite)
    evidence['zero_gradient_tensors'] = sum(good and bool((g == 0).all()) for good, g in zip(finite, gradients))
    evidence['aux_gradient_tensors'] = sum(p.grad is not None for p in aux.parameters())
    executed = all(evidence[key] == 1 for key in ('student_forward', 'aux_forward', 'backward'))
    if not executed or not gradients or evidence['reply_tokens'] <= 0:
        raise ValueError('[opd-zero] 对照没有完整执行前向和反向，不能判通过')
    if total != 0 or count != 0 or evidence['zero_gradient_tensors'] != len(gradients):
        raise ValueError('[opd-zero] 零掩码仍有非零或非有限梯度')
    if evidence['aux_gradient_tensors']:
        raise ValueError('[opd-zero] 辅助模型不应收到梯度')
    return evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=DEFAULT_OPD_PROMPTS)
    ap.add_argument("--out", default="checkpoints/opd/p1_r1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8, help="每 rank 每步样本数")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-new", type=int, default=2048)   # 09-04：think-on 学生要先想再答，200 会把回复截没（B200 上不缺）
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--probe-every", type=int, default=20, help="零掩码对照断言间隔")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--adapter", default="", help="学生起点 LoRA（v16 SFT/RL 产物）；空 = 底座上新建 r=32 LoRA（冒烟）")
    ap.add_argument("--base", default=STUDENT_BASE, help="学生底座；RL adapter 训在合并 SFT 模型之上时要传那个合并模型（评测同底）")
    ap.add_argument("--lora-targets", default=None, help="新建 LoRA 的 target_modules 正则；默认同 sft.py 的 attn_shared")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="冒烟：跑满 N 次真实 optimizer update 才算完成（0=不限；跳步不计）")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="最多尝试多少个 batch；防止一直没有可蒸 token 时无限跑。"
                         "max-steps>0 且未传时自动取 max(10N,N+4)")
    ap.add_argument("--seed", type=int, default=100,
                    help="采样与 smoke 排序的基础随机种子；每个 rank 使用 seed+rank")
    args = ap.parse_args()

    from syncopate.core.contract import IS_V15
    if not IS_V15:
        raise SystemExit("🔴 当前 OPD 只支持 v15 纯自然语言契约；请显式设置 SYNCOPATE_CONTRACT=v15")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    import random
    effective_seed = args.seed + rank
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    log(f"[opd-seed] base={args.seed} rank={rank} effective={effective_seed}")
    completion_marker = Path(args.out) / "completion.json"
    run_token = uuid.uuid4().hex[:16] if rank == 0 else ""
    if rank == 0:
        # Clear only the success marker, before any expensive model load.  A crash
        # in this invocation can therefore never make a previous final look current.
        completion_marker.unlink(missing_ok=True)
        log(f"[opd-run] token={run_token} base={args.base} adapter={args.adapter} "
            f"out={args.out}")
    dist.barrier()
    run_tokens = [run_token]
    dist.broadcast_object_list(run_tokens, src=0)
    run_token = run_tokens[0]
    aux_gpus = os.environ.get("OPD_AUX_GPUS", "2,3").split(",")
    # ⚠️ 卡号是 CUDA_VISIBLE 重映射后的索引：启动令 CUDA_VISIBLE_DEVICES=0,3,1,2 下
    #   可见 0=物理0(rank0 学生) 1=物理3(rank1 学生) 2=物理1(rank0 教师) 3=物理2(rank1 教师)
    # 卡数回退（08-29 审计补）：可见卡不足教师独立卡时，教师/锚与学生同挤一张
    # 小模型单设备诊断分支；不能据此推断当前学生、教师和锚能同时放进一张卡。
    n_vis = torch.cuda.device_count()
    if rank < len(aux_gpus) and int(aux_gpus[rank]) < n_vis:
        aux_dev = f"cuda:{aux_gpus[rank]}"
    else:
        aux_dev = f"cuda:{rank}"
        log(f"⚠️ 可见 GPU {n_vis} 张不足教师独立卡 ⇒ 教师/锚与学生同卡 cuda:{rank}（回退档）")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syncopate.train.rollout_budget import (ENABLE_THINKING,
                                                SAMPLING_TEMPERATURE,
                                                SAMPLING_TOP_K, SAMPLING_TOP_P)

    STUDENT_BASE_ = args.base
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE_)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import syncopate.domains.adcampaign  # noqa: F401  注册与 Runtime 相同的工具
    from syncopate.core.tool_registry import REGISTRY
    from syncopate.prompts import load_system_prompt, prompt_hash
    tools = REGISTRY.menu(None)
    contract_hash = prompt_hash(load_system_prompt(), tools)
    log(f"[opd-prompt] full_menu={len(tools)} answer_fields=0 "
        f"history=message_pairs hash={contract_hash}")
    log("加载学生…")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE_, dtype=torch.bfloat16, device_map={"": rank})
    if args.adapter:
        student = PeftModel.from_pretrained(student, args.adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        from syncopate.train.sft import LORA_TARGETS_DEFAULT
        tm = args.lora_targets or os.environ.get("SYNCOPATE_LORA_TARGETS", LORA_TARGETS_DEFAULT)
        if tm == "attn_shared":
            tm = r"^(?!.*\.experts\.).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkvz|in_proj_ba|in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
        student = get_peft_model(student, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=tm, task_type="CAUSAL_LM"))
        log(f"⚠️ 无 --adapter ⇒ 底座上新建 LoRA（冒烟档；target={tm[:60]}…）")
    _tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log(f"可训练 {_tr/1e6:.1f}M")
    log("加载教师（底座）与锚（候选冻结）…")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_BASE, dtype=torch.bfloat16,
        device_map={"": aux_dev}).eval()
    # ★ 裁定⑬前置（09-04 核实）：学生/教师 vocab 逐项相同（248077，diff 0）才允许逐 token KL；不同只能走文本级
    _ttok = AutoTokenizer.from_pretrained(TEACHER_BASE)
    assert _ttok.get_vocab() == tok.get_vocab(), "🔴 学生/教师 vocab 不同 ⇒ 逐 token 蒸馏非法（裁定⑬）"
    log("[opd-vocab] 学生/教师 vocab 逐项相同 ✓（教师侧用学生模板渲染的同一串 token id）")
    anchor_base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE_, dtype=torch.bfloat16,
        device_map={"": aux_dev})
    anchor = (PeftModel.from_pretrained(anchor_base, args.adapter, is_trainable=False).eval()
              if args.adapter else anchor_base.eval())
    token_mapper = ByteLevelTokenMapper(tok)
    runtime_evidence = generation_runtime_evidence(student)
    runtime_evidence.update(student_class=type(student).__name__, base=args.base,
        tokenizer_class=type(tok).__name__, enable_thinking=ENABLE_THINKING,
        tokenizer=token_mapper.evidence())
    token_audit = TokenAuditWriter(Path(args.out), run_token=run_token, rank=rank, runtime=runtime_evidence)

    rows = [json.loads(x) for x in open(args.prompts)]
    # ★ 等长分片（08-29 审计修复）：原 rows[rank::world] 在特定条数下各 rank 批次数
    #   不等 ⇒ 集合通信次数错位 = 死锁（P1 的 419 条恰好 27/27 躲过；209 条就是
    #   14/13 卡死）。DistributedSampler 同法：循环补齐到等长再切块。
    n_per = math.ceil(len(rows) / world)
    rows = (rows * world)[: n_per * world][rank * n_per:(rank + 1) * n_per]
    trainables = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=args.lr)
    update_audit = OptimizerUpdateAudit(Path(args.out), run_token=run_token, rank=rank,
        named_parameters=student.named_parameters(), optimizer=opt)

    wb = None
    if rank == 0 and not args.no_wandb:
        import wandb
        wb = wandb.init(project="syncopate", name=f"u_opd_{Path(args.out).name}",
                        config=vars(args))

    attempted_steps = 0
    real_steps = 0
    skipped_steps = 0
    max_attempts = args.max_attempts
    if args.max_steps and max_attempts <= 0:
        max_attempts = max(args.max_steps * 10, args.max_steps + 4)
    saved: list[Path] = []
    _stop = False
    for ep in range(args.epochs):
        if _stop:
            break
        random.Random(args.seed + ep).shuffle(rows)
        epoch_rows = prioritize_smoke_routes(rows) if args.max_steps else rows
        if rank == 0 and args.max_steps:
            log("[opd-smoke-order] 首批固定覆盖 task+chat 双路由")
        for i in range(0, len(epoch_rows), args.batch):
            if _stop:
                break
            attempted_steps += 1
            batch = epoch_rows[i: i + args.batch]
            t0 = time.time()
            # ① on-policy 采样（逐轮：多轮 prompt 先生成前轮回复垫底）
            samples = []                   # (GeneratedSample | None, generation_id, family)
            invalid_history = 0
            for sample_index, r in enumerate(batch):
                replies: list[str] = []
                sample, generation_id = None, None
                if not r["turns"]:
                    raise ValueError("OPD source row has no prompt turns")
                for tI in range(len(r["turns"])):
                    try:
                        prm = build_prompt(tok, r["turns"][: tI + 1], replies, tools)
                    except ValueError:
                        invalid_history += 1
                        sample, generation_id = None, None
                        break
                    sample = gen_batch(student, tok, [prm], args.max_new,
                                    SAMPLING_TEMPERATURE, SAMPLING_TOP_P,
                                    SAMPLING_TOP_K, mapper=token_mapper)[0]
                    generation_id = token_audit.generation(sample, attempt=attempted_steps,
                        sample_index=sample_index, turn_index=tI, family=r["family"],
                        final_turn=tI == len(r["turns"]) - 1)
                    if sample.alignment.warnings:
                        log(f"[opd-token-warn] generation={generation_id} "
                            f"reasons={dict(Counter(sample.alignment.warnings))}")
                    replies.append(sample.response_text)
                samples.append((sample, generation_id, r["family"]))
            # ② 掩码反 KL（双教师路由），逐样本 backward
            opt.zero_grad(set_to_none=True)
            kl_chat = kl_task = 0.0
            n_chat = n_task = tok_chat = tok_task = 0
            chat_no_nl = 0
            chat_conservative_zero = 0
            for sample, generation_id, fam in samples:
                aux = teacher if fam == "chat" else anchor
                if sample is None:
                    s, m = 0.0, 0
                else:
                    evidence = {}
                    s, m = kl_step(student, aux, sample, aux_dev, audit=evidence)
                    token_audit.kl(generation_id, attempt=attempted_steps, family=fam,
                                   zero_mask=False, audit=evidence)
                if fam == "chat" and m == 0:
                    # A mixed-boundary token or uncertain UTF-8 range can legally
                    # contain visible NL while all original token masks are zero.
                    # Identity/length/interface faults already fail in validate().
                    from syncopate.train.opd_render import v15_char_labels
                    if any(label == "text" for label in v15_char_labels(
                            sample.response_text if sample is not None else "",
                            implicit_think_open=ENABLE_THINKING)):
                        chat_conservative_zero += 1
                    else:
                        chat_no_nl += 1
                if fam == "chat":
                    kl_chat += s; tok_chat += m; n_chat += 1
                else:
                    kl_task += s; tok_task += m; n_task += 1
            total_masked = tok_chat + tok_task
            if chat_conservative_zero:
                log(f"[opd-token-warn] attempt={attempted_steps} "
                    f"chat_conservative_zero={chat_conservative_zero}（原始 token 边界保守掩码）")
            # 跳步必须集体决定（单 rank 跳而对端进 all_reduce = 死锁）
            gm = torch.tensor([float(total_masked)], device=f"cuda:{rank}")
            dist.all_reduce(gm)
            if gm.item() == 0:
                skipped_steps += 1
                if rank == 0:
                    log(f"[opd-route] chat={n_chat} task={n_task} chat_masked={tok_chat} "
                        f"task_masked={tok_task} invalid_history={invalid_history}")
                    log(f"[opd-mask] attempt {attempted_steps} 全局无 v15 自然语言 token，集体跳步")
                    if wb:
                        wb.log({"opd/skipped_steps": skipped_steps,
                                "opd/attempted_steps": attempted_steps,
                                "opd/real_steps": real_steps}, step=attempted_steps)
                if max_attempts and attempted_steps >= max_attempts:
                    log(f"[max-attempts] 已尝试 {attempted_steps} 个 batch，"
                        f"只有 {real_steps} 次真实更新；停止并判失败")
                    _stop = True
                opt.zero_grad(set_to_none=True)
                dist.barrier()
                continue
            # 与 SFT 共用完整 DP 副本的 token 均值目标；不能用各 rank 梯度的平均
            # 代替有效 token 平均。零 token 的 rank 也参与，整个窗口为零已在上面集体跳过。
            objective = finish_token_window(trainables, local_tokens=total_masked,
                local_loss_sum=kl_chat + kl_task, device=student.device)
            torch.nn.utils.clip_grad_norm_(trainables, 1.0, error_if_nonfinite=True)
            with update_audit.step(step=real_steps + 1, attempt=attempted_steps):
                opt.step()
            real_steps += 1
            if args.max_steps and real_steps >= args.max_steps:
                log(f"[max-steps] 已完成 {real_steps} 次真实更新，停止（冒烟）")
                _stop = True
            elif max_attempts and attempted_steps >= max_attempts:
                log(f"[max-attempts] 已尝试 {attempted_steps} 个 batch，停止")
                _stop = True
            dt = time.time() - t0
            if rank == 0:
                m_chat = kl_chat / max(tok_chat, 1)
                m_task = kl_task / max(tok_task, 1)
                log(f"[opd-mask] attempt {attempted_steps} 全局可蒸 token={objective['supervised_tokens']}")
                log(f"[opd-objective] step {real_steps} kl_per_token={objective['loss']:.8g} "
                    f"global_tokens={objective['supervised_tokens']}")
                log(f"step {real_steps} attempt={attempted_steps} ep{ep} kl_chat/tok={m_chat:.4f} "
                    f"kl_task/tok={m_task:.4f} masked={total_masked} "
                    f"[opd-route] chat={n_chat} task={n_task} chat_masked={tok_chat} "
                    f"task_masked={tok_task} invalid_history={invalid_history} {dt:.1f}s")
                if wb:
                    wb.log({"opd/kl_chat_per_tok": m_chat,
                            "opd/kl_task_per_tok": m_task,
                            "opd/chat_no_nl": chat_no_nl,
                            "opd/chat_conservative_zero": chat_conservative_zero,
                            "opd/masked_tokens": total_masked,
                            "opd/global_masked_tokens": objective["supervised_tokens"],
                            "opd/kl_per_token_global": objective["loss"],
                            "opd/route_chat": n_chat, "opd/route_task": n_task,
                            "opd/step_time_s": dt, "opd/epoch": ep,
                            "opd/attempted_steps": attempted_steps,
                            "opd/real_steps": real_steps,
                            "opd/skipped_steps": skipped_steps}, step=attempted_steps)
            # ③ 零泄漏对照断言（守则②：假设写成断言）
            if args.probe_every > 0 and real_steps % args.probe_every == 0:
                opt.zero_grad(set_to_none=True)
                probe = next(((sample, generation_id, fam) for sample, generation_id, fam in samples
                              if sample is not None and sample.response_ids), None)
                if probe is None:
                    raise ValueError('[opd-zero] 本地没有非空回复，不能声称对照通过')
                sample, generation_id, fam = probe
                evidence = zero_mask_control(student, teacher if fam == 'chat' else anchor,
                                             sample, aux_dev)
                token_audit.kl(generation_id, attempt=attempted_steps, family=fam,
                              zero_mask=True, audit=evidence)
                if rank == 0:
                    log(f"[opd-zero] step {real_steps} 对照通过（0/{len(trainables)}） "
                        f"student_forward={evidence['student_forward']} aux_forward={evidence['aux_forward']} "
                        f"backward={evidence['backward']} gradients={evidence['gradients']}")
                opt.zero_grad(set_to_none=True)
            # ④ adapter-only 滚动存档
            if rank == 0 and real_steps % args.save_every == 0:
                pth = Path(args.out) / f"step_{real_steps}"
                student.save_pretrained(pth)
                saved.append(pth)
                while len(saved) > 3:
                    shutil.rmtree(saved.pop(0), ignore_errors=True)
                log(f"[opd-ckpt] {pth}（adapter-only，保留最近 3）")
            dist.barrier()
        # ★ epoch 末：rank 间学生权重一致性硬断言（08-29 审计补，与 sft.py 同款防线）
        #   手动梯度同步 + 各 rank 同步后独立 opt.step()，权重应逐位一致；
        #   发散 = 同步静默失效（「adapter 没推送」家族），宁可停机不带病训完。
        fingerprint = assert_replicated_parameters(student.named_parameters())
        if rank == 0:
            log(f"[opd-sync] ep{ep} 完整可训练参数指纹一致 "
                f"sha256={fingerprint['sha256']} tensors={fingerprint['tensors']}")
    local_token_summary = token_audit.summary()
    token_summaries = [None] * world
    dist.all_gather_object(token_summaries, local_token_summary)
    log(f"[opd-token-summary] {json.dumps(local_token_summary, ensure_ascii=False, sort_keys=True)}")
    local_update_summary = update_audit.summary()
    update_summaries = [None] * world
    dist.all_gather_object(update_summaries, local_update_summary)
    log(f"[opd-update-summary] {json.dumps(local_update_summary, ensure_ascii=False, sort_keys=True)}")
    completed = training_completed(real_steps, args.max_steps)
    if rank == 0:
        log(f"[opd-summary] attempted={attempted_steps} real={real_steps} "
            f"skipped={skipped_steps} target_real={args.max_steps or 'all'} "
            f"status={'pass' if completed else 'fail'}")
        if completed:
            pth = Path(args.out) / "final"
            student.save_pretrained(pth)
            completion = {
                "schema_version": 1,
                "status": "pass",
                "run_token": run_token,
                "base": args.base,
                "adapter": args.adapter,
                "prompts": args.prompts,
                "seed": args.seed,
                "world_size": world,
                "probe_every": args.probe_every,
                "prompt_hash": contract_hash,
                "attempted_steps": attempted_steps,
                "skipped_steps": skipped_steps,
                "real_steps": real_steps,
                "target_real_steps": args.max_steps or None,
                "objective": "global_masked_token_mean_reverse_kl",
                "parameter_fingerprint": fingerprint,
                "token_audits": token_summaries,
                "update_audits": update_summaries,
            }
            completion_marker.parent.mkdir(parents=True, exist_ok=True)
            marker_tmp = completion_marker.with_suffix(".json.tmp")
            marker_tmp.write_text(
                json.dumps(completion, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            marker_tmp.replace(completion_marker)
            log(f"[opd-ckpt] final → {pth}")
        else:
            log("🔴 没有完成要求的真实 optimizer update；不写 final，拒绝把空跑冒充成功")
        if wb:
            wb.finish()
    dist.barrier()
    dist.destroy_process_group()
    # 0 real update / missing final is a health failure, not an observable quality
    # warning.  Exit 3 so smoke/observe cannot continue into eval on a stale final.
    return 0 if completed else 3


if __name__ == "__main__":
    raise SystemExit(main())
