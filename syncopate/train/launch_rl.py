"""GRPO 启动器：拼 verl 的 Hydra override 并起子进程。

★ 相对老师那套 64 卡配置，单卡 5090 有三处**必须反着来**的改动：

1. **关掉 offload**。老师开 `param_offload=True` + `optimizer_offload=True`，
   把负担丢给 CPU 内存——前提是"CPU 内存富余、每卡显存紧张"。本机恰好相反：
   32GB 显存充裕，**30GB 系统内存才是瓶颈**（Ray + vLLM + trainer 三个进程本身就要 8-10GB）。
   开着 offload 会被 OOM killer 干掉。

2. **上 LoRA**。4B 全参 AdamW 要 48GB 优化器状态，装不下。
   r=32 挂全部线性层 = 66M 可训练参数（占 1.64%），优化器状态 0.79GB。
   ⚠️ 只挂注意力是老习惯，容量差 2.8 倍，必须带上 MLP 的 gate/up/down。

3. **vLLM 显存份额调大**。老师给 0.2（64 卡分摊），单卡要给到 0.4 左右。

用法：
    python -m syncopate.train.launch_rl --dry-run          # 只打印命令，不启动
    python -m syncopate.train.launch_rl --steps 10         # 真跑 10 步冒烟
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_overrides(args: argparse.Namespace) -> list[str]:
    model = str((ROOT / args.model).resolve())
    max_model_len = args.max_prompt_length + args.max_response_length

    overrides = [
        # ---- 算法：GRPO，不要 critic，不要内置 reward model ----
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "critic.enable=False",
        # reward 由我们的 AgentLoop 内部算好直接返回，不走 verl 的 reward model
        "reward_model.enable=False",

        # ---- 数据 ----
        f"data.train_files={ROOT / args.train_file}",
        f"data.val_files={ROOT / args.val_file}",
        "data.prompt_key=prompt",
        "data.return_raw_chat=True",          # AgentLoop 要拿原始 messages
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.dataloader_num_workers=0",

        # ---- 模型 ----
        f"actor_rollout_ref.model.path={model}",
        # remove_padding 需要 flash-attn（verl 的 unpad_input 直接 import flash_attn.bert_padding）。
        # sm_120 上 flash-attn 目前得自己编译，冒烟阶段先关掉；
        # 它是真实的效率优化（见 docs/learning-notes/04），正式跑长序列前应该装上再打开。
        f"actor_rollout_ref.model.use_remove_padding={str(args.remove_padding)}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",

        # ---- ★ 改动 1：关掉 offload（本机瓶颈是内存不是显存）----
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",

        # ---- actor ----
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size}",
        f"actor_rollout_ref.actor.use_kl_loss={str(args.use_kl_loss)}",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.micro_batch_size}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.micro_batch_size}",

        # ---- rollout：vLLM async（AgentLoop 必需）----
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        # ★ 改动 3：单卡要给 vLLM 更大份额（老师 64 卡时是 0.2）
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_util}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        # calculate_log_probs=True 才有 rollout_is_* 那套 TIS 诊断指标
        "actor_rollout_ref.rollout.calculate_log_probs=True",

        # ---- multi-turn AgentLoop 接线 ----
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={args.max_turns}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={args.max_turns}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=syncopate_adcampaign",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={ROOT / 'configs/verl_agent_loop.yaml'}",
        f"actor_rollout_ref.rollout.agent.num_workers={args.agent_workers}",

        # ---- trainer ----
        f"trainer.default_local_dir={ROOT / args.save_path}",
        f"trainer.project_name={args.project}",
        f"trainer.experiment_name={args.experiment}",
        f"trainer.logger=[{args.logger}]",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.steps}",
        "trainer.save_freq=-1",
        f"trainer.val_before_train={str(args.val_before_train)}",
        f"trainer.test_freq={args.test_freq}",
        "trainer.resume_mode=disable",
    ]

    # ---- ★ 改动 2：LoRA ----
    if args.lora_rank > 0:
        overrides += [
            f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={args.lora_rank * 2}",
            # 挂全部线性层。只挂注意力容量差 2.8 倍——这是最常见的 LoRA 配置错误。
            "actor_rollout_ref.model.target_modules=all-linear",
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=True",
        ]

    # ---- TIS / rollout correction：主线研究要用的诊断指标 ----
    if args.rollout_correction:
        overrides += [
            f"algorithm.rollout_correction.rollout_is={args.rollout_is}",
            f"algorithm.rollout_correction.rollout_is_threshold={args.rollout_is_threshold}",
            "algorithm.rollout_correction.rollout_is_batch_normalize=false",
        ]

    return overrides + list(args.extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syncopate GRPO 启动器（单卡 5090 降配）")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/rl/v3/train.parquet")
    parser.add_argument("--val-file", default="data/rl/v3/val.parquet")
    parser.add_argument("--save-path", default="checkpoints/grpo/smoke")
    parser.add_argument("--project", default="syncopate")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    parser.add_argument("--experiment", default="smoke")
    parser.add_argument("--logger", default="console,wandb",
                        help="verl 会把 rollout_corr/* 和 critic/* 全套指标上报")

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rollout-n", type=int, default=8, help="GRPO 组大小")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--rollout-gpu-util", type=float, default=0.40)
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = 全参；4B 必须给 32")
    parser.add_argument("--lr", default="1e-6")
    parser.add_argument("--remove-padding", default="False", choices=["True", "False"],
                        help="需要 flash-attn；sm_120 上要自己编译")
    parser.add_argument("--use-kl-loss", default="True")
    parser.add_argument("--val-before-train", default="False")
    parser.add_argument("--test-freq", type=int, default=-1)

    parser.add_argument("--rollout-correction", action="store_true", default=True)
    parser.add_argument("--rollout-is", default="sequence", choices=["token", "sequence"])
    parser.add_argument("--rollout-is-threshold", type=float, default=2.0)

    parser.add_argument("--latency-scale", type=float, default=0.01,
                        help="工具真实延迟的缩放。★ 长尾比例是异步对照实验的核心自变量。"
                             "1.0=真实（poll_review 睡 480 秒，约 18%% 的 step 会变成 8 分钟）；"
                             "0.01=保留长尾相对结构、成本降两个数量级；0=没有长尾")
    parser.add_argument("--async-verifier", default="1", choices=["0", "1"],
                        help="1=verifier 走线程池；0=复现老师那套的阻塞行为（对照组）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs="*", help="额外的 Hydra override")
    args = parser.parse_args(argv)

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *build_overrides(args)]
    if args.dry_run:
        print(" \\\n  ".join(cmd))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["SYNCOPATE_ASYNC_VERIFIER"] = args.async_verifier
    env["SYNCOPATE_RUN_ID"] = args.experiment
    env["SYNCOPATE_LATENCY_SCALE"] = str(args.latency_scale)
    # 这两个开关决定实验的物理含义，必须打印出来——静默的默认值是最难查的那种错
    print(f"[实验设定] latency_scale={args.latency_scale}  async_verifier={args.async_verifier}"
          f"  rollout_is={args.rollout_is}(阈值 {args.rollout_is_threshold})")
    env["WANDB_MODE"] = args.wandb_mode
    env.setdefault("WANDB_PROJECT", args.project)
    # ⚠️ 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
    # 老师的脚本里有这一行（64 卡场景下能省显存碎片），但它和 vLLM colocate 用的
    # 内存池直接冲突，engine 启动就会挂：
    #   AssertionError: Expandable segments are not compatible with memory pool
    # 参见 pytorch/pytorch#147851。抄配置时最容易连这种坑一起抄过来。
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "1")
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
