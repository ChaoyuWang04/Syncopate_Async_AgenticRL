"""GRPO 启动器：拼 verl 的 Hydra override 并起子进程。

★ 相对老师那套 64 卡配置，单卡 5090 有三处**必须反着来**的改动：

1. **关掉 offload**（两个都关）。老师那套开 `param_offload` + `optimizer_offload`，
   前提是"CPU 内存富余、每卡显存紧张"；本机 30GB 内存才是瓶颈，开着会被 OOM killer 干掉。
   ⚠️ commit cf813f0 的标题写着「param_offload 必须开」，但**真正跑通 50 步的那次用的是
   False**——让它跑通的是 `max_num_seqs` 1024→32。2026-08-12 实测开 True 反而在
   vLLM 第一次 `sleep()` 时静默杀掉 VllmWorker。**别照着 commit 标题改默认值。**

★★ 2026-08-12 实测的显存账（wake_up OOM 连挂三次换来的）：

   **`max_num_seqs` 不是显存旋钮。** `gpu_memory_utilization` 是按比例**预分配**的：
   0.42 × 31.36 = 13.2GB，vLLM 无论并发上限是 32 还是 20 都照拿不误。
   降 `max_num_seqs` 只限并发、不改分配量 —— 实测 32→20 对 OOM 毫无影响。
   （cf813f0 里 1024→32 有效，是因为那是调度结构和 block table 的开销，不是 KV 池。）

   本机的真实账本（Qwen3-4B：36 层 / 8 KV 头 / head_dim 128 ⇒ **KV = 144 KB/token**）：

       FSDP 建好后 actor 常驻      10.49 GB   （实测，日志里 "After FSDP"）
       vLLM 按 0.42 预分配         13.2  GB   （权重 7.6 + KV 池 5.6 = 41k token）
       update_weights 时推权重     ~7.6  GB   （聚合出的一份全量 bf16）
       ------------------------------------------------
       合计                        31.3  GB   ≈ 31.36 的物理上限 ⇒ **贴着墙，必挂**

   ⇒ 真正的旋钮是 **`--rollout-gpu-util`**。给 vLLM 少分一点，就是给 wake_up 那一刻
   的权重聚合腾地方。0.34 ⇒ vLLM 10.7GB、KV 池 3.1GB（22k token ≈ 3 条满长序列），
   总占用 28.8GB，留 2.5GB 余量。代价是 rollout 并发下降、变慢，但结果不受影响。

   ⚠️ 别把 `--rollout-gpu-util` 降到 KV 池装不下**一条**满长序列以下
   （max_model_len 7168 × 144KB = 0.98GB），否则会退化成 §5 表里第一行的
   「vLLM 分不到 KV cache」。

★★★ 更隐蔽的一条：**actor 的显存会随训练步数单调爬升，是碎片不是泄漏**

   2026-08-12 第五次尝试跑到 **step 24** 才挂（前四次是第一步就挂）。实测：

       step   allocated   reserved   差值(碎片)
         1      16.35      19.22       2.87
        10      17.01      20.38       3.36
        15      18.72      20.58       1.86
        24      18.72      21.08       2.36   ← 然后 wake_up OOM

   **`allocated` 从 step 15 起就不涨了，`reserved` 还在爬** —— PyTorch 缓存分配器
   的 reserved 只增不减。到 step 24 剩给 vLLM 的只有 31.36−21.08 = 10.28GB，
   而 0.42 要 13.2GB。

   ⇒ **配 `--rollout-gpu-util` 不能按第一步的显存算，要按跑几十步之后的峰值算。**
   按 21.08 反推上限是 0.328；实际取 0.30 留余量。
   ⇒ 这也解释了前四次：它们不是"某个参数配错"，是这条路本来就贴着墙走 ——
   prompt 更长的时候第一步就撞，prompt 裁短之后能撑 24 步，但墙还在那儿。

   ⚠️ 碎片的标准解法 `expandable_segments:True` 在这里**用不了**，原因见下面
   `env.pop("PYTORCH_CUDA_ALLOC_CONF")` 那段：它和 vLLM 的 colocate 内存池冲突。

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
        f"actor_rollout_ref.actor.fsdp_config.param_offload={args.param_offload}",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={args.param_offload}",

        # ---- ★ 改动 4：FSDP 用 bf16 存参数，不存 fp32 主权重 ----
        #
        # verl 默认 model_dtype=fp32：4.02B × 4 字节 = 16 GB。实测第一次冒烟就死在这：
        #     After FSDP, device memory used/total 19.35/31.36
        #     vLLM: No available memory for the cache blocks
        # trainer 先占掉 19 GB，vLLM 连 KV cache 都分不到。
        #
        # 而我们用 LoRA —— **98.4% 的参数是冻结的**，给它们存一份 fp32 主权重是纯浪费：
        # 主权重的意义是让优化器更新不被 bf16 的舍入吃掉，而冻结参数根本不更新。
        # 需要 fp32 的只有 LoRA 那 66M，它们的优化器状态本来就是 fp32。
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={args.fsdp_dtype}",
        f"actor_rollout_ref.ref.fsdp_config.model_dtype={args.fsdp_dtype}",

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
        # ★ vLLM 默认 max_num_seqs=1024，而 colocate 单卡上我们同时最多只有
        # train_batch_size × rollout_n 条 rollout（bs=8 时是 64）。
        # 多出来的 960 个槽位不是白占的：调度结构和 block table 都按它预留，
        # 而这块显存在 sleep/wake 循环里每轮都要重新申请 —— 实测 wake_up 就死在这。
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        # ---- H0 观测仪：vLLM 周期性统计（吞吐 / prefix cache 命中率 / preemption 次数）----
        # 纯打日志不改行为。「训练脚本的默认值必须是跑完就有记录」的推理侧版本：
        # KV 池会不会踢人、踢多凶，这三个数字以前从来没人看过。
        f"actor_rollout_ref.rollout.disable_log_stats={str(args.no_engine_stats)}",
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
        # ★★★ 每步把**实际进入训练**的样本 dump 成 JSONL，连同 reward_extra_info。
        #
        # 两个作用，缺一不可：
        # 1. verl 的 compute_data_metrics 只认 __num_turns__ / tool_call_counts，
        #    我们的 cap/* 和 subscore/* 训练时**根本不上 wandb**。这份 dump 是唯一的来源。
        # 2. 它是「下发分布 vs 训练分布」漂移的度量基准：
        #    dump 里是**训练到的**，我们 artifact 里是**下发过的**，两者的差就是漂移。
        #    sync 下这个差恒为 0（有 barrier）；async 下短任务先回、长任务被
        #    partial_rollout 切断，差就出来了 —— 而长链正是 agentic 的核心能力。
        f"trainer.rollout_data_dir={ROOT / args.save_path / 'rollout_dumps'}",
        f"trainer.project_name={args.project}",
        f"trainer.experiment_name={args.experiment}",
        f"trainer.logger=[{args.logger}]",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.steps}",
        # ★ 冒烟时 -1（不存）没问题，正式跑必须存 —— 否则两件事都做不了：
        #   1. 跑完没有 ckpt 可以在冻结 EVAL 上重评，等于白跑
        #   2. staleness 研究要的是**同一次训练里相隔 k 步的两个 policy**，
        #      没有中途 ckpt 就凑不出 k>0 的样本对（单卡跑不了真异步，只能这么合成）
        f"trainer.save_freq={args.save_freq}",
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

    # ---- Ostinato A1 实验入口：KV cache 量化（fp8_e4m3 / fp8_e5m2）----
    # 经 engine_kwargs.vllm 透传。容量红利：同一块 KV 池装 2× token ⇒ 驱逐/preemption
    # 减半。不设默认值 —— 开不开必须是显式决定，且开了要跑 EVAL 128 配对回归验精度。
    if args.kv_cache_dtype:
        overrides.append(
            f"actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype={args.kv_cache_dtype}")

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
    parser.add_argument("--rollout-gpu-util", type=float, default=0.55,
                        help="vLLM 的显存份额。★ 它是**占总显存的比例**，而且必须把 vLLM "
                             "自己的模型权重(4B bf16≈8GB)也装进这个预算里，剩下的才是 KV cache")
    parser.add_argument("--object-store-gb", type=int, default=2,
                        help="Ray 对象存储上限(GB)。默认按 RAM 的 30%% 预留，在 30GB 机器上会把 vLLM 和 trainer 挤死")
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="vLLM 并发槽位。默认 1024 远超实际需要"
                             "（colocate 单卡上就是 train_batch_size × rollout_n）")
    # ⚠️ 别被 commit cf813f0 的标题（「param_offload 必须开」）带偏：
    # 真正跑通 50 步的那一次（2026-08-11 12:46）用的是 **False**。让它跑通的是
    # `max_num_seqs` 从 1024 降到 32，不是 offload。2026-08-12 实测 True 反而在
    # vLLM 第一次 `sleep()` 时静默杀掉 VllmWorker。⇒ 保持 False，要开先单独验。
    parser.add_argument("--param-offload", default="False", choices=["True", "False"],
                        help="rollout 期间把 actor 参数挪到 CPU。默认 False（实测跑通的那次就是 False）")
    parser.add_argument("--fsdp-dtype", default="bf16", choices=["bf16", "fp32"],
                        help="FSDP 持有参数的精度。LoRA 下 98.4%% 的参数冻结，"
                             "fp32 主权重是纯浪费（4B 要 16GB，实测直接把 vLLM 挤死）")
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = 全参；4B 必须给 32")
    parser.add_argument("--lr", default="1e-6")
    parser.add_argument("--remove-padding", default="False", choices=["True", "False"],
                        help="需要 flash-attn；sm_120 上要自己编译")
    parser.add_argument("--no-engine-stats", action="store_true",
                        help="关掉 vLLM 周期性统计日志（默认开：吞吐/prefix cache 命中率/preemption）")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="vLLM KV cache 精度（fp8_e4m3/fp8_e5m2）。KV 池容量 ×2 的免费杠杆；"
                             "默认不动。开了必须配 EVAL 128 配对回归验精度（Ostinato A1）")
    parser.add_argument("--use-kl-loss", default="True")
    parser.add_argument("--val-before-train", default="False")
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--save-freq", type=int, default=10,
                        help="每 N 步存一次 ckpt。★ 正式跑不能是 -1：跑完没 ckpt 就没法重评，"
                             "而且 staleness 研究要的就是同一次训练里相隔 k 步的两个 policy")

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
    # 下发侧记账，和 trainer.rollout_data_dir 的训练侧 dump 配对算分布漂移
    env["SYNCOPATE_DISPATCH_LOG"] = str(ROOT / args.save_path / "dispatched.jsonl")
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
    # ★ Ray 的对象存储默认占 RAM 的 30%（本机 /dev/shm 有 16GB 可用，它就敢要 9GB）。
    #
    # 实测：本机 30.9GB 内存，Ray 报 29.78GB 用满后直接杀 worker，**param_offload
    # 关着也照样爆**。我们的 batch 很小，对象存储根本用不了那么多——
    # 它只是按比例预留，然后把真正需要内存的 vLLM 和 trainer 挤死。
    env.setdefault("RAY_object_store_memory", str(args.object_store_gb * 1024**3))
    # 内存吃紧时 Ray 会提前杀 worker。放宽一点，让真正的分配失败自己暴露出来，
    # 而不是被 Ray 的保护机制提前打断（真 OOM 有堆栈，被 Ray 杀掉只有一句话）
    env.setdefault("RAY_memory_usage_threshold", "0.97")
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
