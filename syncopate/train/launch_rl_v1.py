"""v16 · verl 0.9 V1 统一 trainer 的 GRPO 启动器（B200 新栈；26 §W4′ S6）。

    python -m syncopate.train.launch_rl_v1 --steps 2 --gpus 2 --experiment rl_v16_smoke

★ 为什么另起一个薄壳而不是改 `launch_rl.py`：那份是 4×5090 / verl 0.8 的产物（DDP·无 P2P·offload 账本·
  20 处补丁），1200 行里大半前提在 B200+0.9 上都不成立（26 §W4′ S1-4 分诊：删 6 停 4 改 1 留 3）。
  这份只放 V1 trainer 真会读的键；契约参数（长度/采样）仍**只**从 `rollout_budget` 取（守则⑨），
  数据集采样器仍走 `main_ppo_pool`（动态分池；0.9 下补丁改挂 `trainer.ppo.utils`）。

verl 0.9 事实（09-04 容器 dump `/vol/_audit/v16/verl09_dump{,2}.json`）：
  · `trainer.use_v1=true`，`trainer.v1.trainer_mode ∈ {sync, colocate_async, separate_async}`；入口 `main_ppo.TaskRunnerV1`（Ray actor）
  · `create_rl_sampler` 定义在 `trainer/ppo/utils.py`，`v1/trainer_base.py:68` 导入时绑名
  · LoRA：`model.lora_rank/lora_alpha/target_modules(str|list)/exclude_modules(str 正则)`；rollout `load_format=safetensors`
  · reward：`reward.reward_model.enable`；AgentLoopOutput.reward_score 非空时直接当 rm_scores（agent_loop.py:142）
  · agent loop：`rollout.agent.{default_agent_loop, agent_loop_config_path, num_workers}` 与 0.8 同名，`_target_` 注册照旧
判据（S6 冒烟，跑前注册）：每步退出码 0 · `[pool] 动态分池启用` 行在 worker 侧出现 · actor loss/grad_norm 有限 ·
  reward 非全 0 · vLLM 权重同步行出现（checkpoint engine）· LoRA-only ckpt 落盘且能被 peft 读。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from syncopate.core.model_paths import STUDENT_MODEL
from syncopate.pipeline.split import DEFAULT_RL_DIR

ROOT = Path(__file__).resolve().parents[2]


def build_overrides(a: argparse.Namespace) -> list[str]:
    from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
    model = str((ROOT / a.model).resolve())
    max_model_len = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
    ov = [
        # ── 算法 ──
        "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False",
        "critic.enable=False", "reward.reward_model.enable=False",
        # ── 数据（契约长度只从 rollout_budget 来）──
        f"data.train_files={ROOT / a.train_file}", f"data.val_files={ROOT / a.val_file}",
        "data.prompt_key=prompt", "data.return_raw_chat=True", "data.filter_overlong_prompts=False", "data.truncation=error",
        f"data.train_batch_size={a.train_batch_size}", f"data.val_batch_size={a.val_batch_size}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}", f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.dataloader_num_workers=0", f"data.seed={a.seed}",
        # ── 模型 + LoRA（S1-7：默认排除专家，专家层作开关）──
        f"actor_rollout_ref.model.path={model}", "actor_rollout_ref.model.use_remove_padding=True",
        f"+actor_rollout_ref.model.override_config.attn_implementation={a.attn_implementation}",
        f"actor_rollout_ref.model.lora_rank={a.lora_rank}", f"actor_rollout_ref.model.lora_alpha={a.lora_rank * 2}",
        f"actor_rollout_ref.model.target_modules={a.target_modules}",
        *( [f"actor_rollout_ref.model.exclude_modules={a.exclude_modules}"] if a.exclude_modules else [] ),
        # ── actor（FSDP2；B200 不需要 offload）──
        f"actor_rollout_ref.actor.strategy={a.strategy}", f"actor_rollout_ref.ref.strategy={a.strategy}",
        f"actor_rollout_ref.actor.optim.lr={a.lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={a.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={a.micro_batch_size}",
        "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_model_len}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_model_len}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_model_len}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={a.micro_batch_size}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={a.micro_batch_size}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False", "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={a.gpus}", f"actor_rollout_ref.ref.fsdp_config.fsdp_size={a.gpus}",
        f"actor_rollout_ref.actor.use_prefix_grouper={str(a.prefix_grouper)}",
        # ── rollout（vLLM 0.28 async + 我们的 agent loop）──
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={a.rollout_gpu_util}",
        f"actor_rollout_ref.rollout.max_num_seqs={a.max_num_seqs}",
        f"actor_rollout_ref.rollout.enforce_eager={a.enforce_eager}",
        f"actor_rollout_ref.rollout.free_cache_engine={a.free_cache_engine}",
        f"actor_rollout_ref.rollout.n={a.rollout_n}", "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.multi_turn.enable=True", "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={a.max_turns}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={a.max_turns}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=syncopate_adcampaign",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={ROOT / 'configs/verl_agent_loop.yaml'}",
        f"actor_rollout_ref.rollout.agent.num_workers={a.agent_workers}",
        f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={a.weight_sync_bucket_mb}",
        "actor_rollout_ref.rollout.top_p=1.0", "actor_rollout_ref.rollout.top_k=-1",
        # ── trainer（V1）──
        "trainer.use_v1=True", f"trainer.v1.trainer_mode={a.mode}",
        f"trainer.n_gpus_per_node={a.gpus}", "trainer.nnodes=1", "trainer.total_epochs=1",
        f"trainer.total_training_steps={a.steps}", f"trainer.save_freq={a.save_freq}",
        "trainer.val_before_train=False", "trainer.test_freq=-1", "trainer.resume_mode=disable",
        f"trainer.default_local_dir={ROOT / a.save_path}", f"trainer.rollout_data_dir={ROOT / a.save_path / 'rollout_dumps'}",
        f"trainer.project_name={a.project}", f"trainer.experiment_name={a.experiment}", f"trainer.logger=[{a.logger}]",
        "trainer.balance_batch=True",
        # ── worker 钩子：补丁与动态分池在 Ray worker（含 TaskRunnerV1）进程里生效 ──
        "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=syncopate.train.verl_patches.setup_worker",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL={a.vllm_log_level}",
    ]
    if a.save_lora_only:
        ov.append("+actor_rollout_ref.actor.checkpoint.save_lora_only=True")   # 0.9：字段在 CheckpointConfig 数据类里、不在 yaml 默认 ⇒ 要 + 追加（rl_cfg 实测）
    return ov + list(a.extra)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Syncopate GRPO 启动器（v16 · verl 0.9 V1 · B200）")
    # ★ 2026-09-04（Chaoyu：默认值必须是"直接跑就健康"的正式值）：两套注册档位，smoke 只验机制、candidate 才是训练。
    #   candidate 的数字来源：步数下限 400（守则⑩，真正该停看 pool_readout 的零梯度率平台）· save-freq 25（E29 口径）·
    #   组大小 8（GRPO 默认）· 模型 = 合并后的 SFT 模型（RL 起点不许是裸底座，launch_rl_v1 断言 lora_adapter 不在目录里）。
    p.add_argument("--profile", default="candidate", choices=["smoke", "candidate"])
    p.add_argument("--model", default=None, help="candidate 默认 models/<学生名>-sft-<DATA_VERSION>（合并后）；smoke 默认学生底座")
    p.add_argument("--train-file", default=f"{DEFAULT_RL_DIR}/train.parquet")
    p.add_argument("--val-file", default=f"{DEFAULT_RL_DIR}/val.parquet")
    p.add_argument("--save-path", default=None)
    p.add_argument("--project", default="syncopate-b200")
    p.add_argument("--experiment", default=None)
    p.add_argument("--logger", default="console,wandb")
    p.add_argument("--mode", default="sync", choices=["sync", "colocate_async", "separate_async"])
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--steps", type=int, default=None, help="smoke 2 · candidate 400（下限，见守则⑩）")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rollout-n", type=int, default=None, help="smoke 4 · candidate 8")
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=2)
    p.add_argument("--ppo-mini-batch-size", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=14)
    p.add_argument("--agent-workers", type=int, default=1)
    p.add_argument("--rollout-gpu-util", type=float, default=0.45)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--enforce-eager", default="False", choices=["True", "False"])
    p.add_argument("--free-cache-engine", default="True", choices=["True", "False"])
    p.add_argument("--weight-sync-bucket-mb", type=int, default=512)
    p.add_argument("--strategy", default="fsdp2", choices=["fsdp", "fsdp2"])
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--exclude-modules", default=r".*\.experts\..*", help="S1-7：默认排除 MoE 专家；空串=不排除（LoRA 挂全部专家 ≈2554M）")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--prefix-grouper", default="False", choices=["True", "False"])
    p.add_argument("--lr", default="3e-5")
    p.add_argument("--save-freq", type=int, default=None, help="smoke 2 · candidate 25")
    p.add_argument("--save-lora-only", default="True", choices=["True", "False"])
    p.add_argument("--vllm-log-level", default="INFO")
    p.add_argument("--latency-scale", type=float, default=0.0, help="冒烟 0；正式异步对照实验用 0.01/1.0")
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cfg-only", action="store_true", help="只让 Hydra 合成配置（--cfg job），不起 Ray/GPU：键名判据")
    p.add_argument("extra", nargs="*")
    a = p.parse_args(argv)
    a.prefix_grouper = a.prefix_grouper == "True"; a.save_lora_only = a.save_lora_only == "True"
    from syncopate.pipeline.split import DATA_VERSION as _DV
    _student = Path(STUDENT_MODEL).name
    _prof = {"smoke": dict(model=STUDENT_MODEL, steps=2, rollout_n=4, save_freq=2, save_path=f"checkpoints/grpo/{_DV}_smoke", experiment=f"rl_{_DV}_smoke"),
             "candidate": dict(model=f"models/{_student}-sft-{_DV}", steps=400, rollout_n=8, save_freq=25, save_path=f"checkpoints/grpo/{_DV}_cand", experiment=f"rl_{_DV}_cand")}[a.profile]
    for k, v in _prof.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.profile == "candidate" and not Path(a.model).exists():
        raise SystemExit(f"🔴 candidate 档要的 RL 起点 {a.model} 不存在：先跑 sft-train → sft-select → merge（scripts/v16_pipeline.sh）")
    print(f"[rl-v1] profile={a.profile} model={a.model} steps={a.steps} n={a.rollout_n} save_freq={a.save_freq} save_path={a.save_path}", flush=True)

    from syncopate.train.rollout_budget import MAX_RESPONSE_LENGTH, THINK_ON
    from syncopate.core.contract import IS_V15
    assert IS_V15 and THINK_ON, "v16 训练路径要求 SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1"
    print(f"[think-train] 契约=v15 · think-on · MAX_RESPONSE_LENGTH={MAX_RESPONSE_LENGTH}", flush=True)
    if (Path(a.model) / "lora_adapter").exists():
        raise SystemExit(f"🔴 {a.model} 里有 lora_adapter/ ⇒ 不是合并后的模型（docs/syncopate/18 §3）")

    entry = "verl.trainer.main_ppo" if a.no_pool else "syncopate.train.main_ppo_pool"
    cmd = [sys.executable, "-m", entry, *build_overrides(a)]
    if a.cfg_only:
        cmd += ["--cfg", "job"]
    if a.dry_run:
        print(" \\\n  ".join(cmd)); return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["SYNCOPATE_RUN_ID"] = a.experiment
    env["SYNCOPATE_LATENCY_SCALE"] = str(a.latency_scale)
    env["SYNCOPATE_ASYNC_VERIFIER"] = "1"
    save = ROOT / a.save_path; save.mkdir(parents=True, exist_ok=True)
    (save / "run_purpose.json").write_text(json.dumps({"purpose": "probe", "steps_requested": a.steps, "stack": "verl0.9-v1"}, ensure_ascii=False))
    env["SYNCOPATE_DISPATCH_LOG"] = str(save / "dispatched.jsonl")
    env["SYNCOPATE_POOL_STATE"] = str(save / "pool_state.json")
    env["SYNCOPATE_POOL"] = "0" if a.no_pool else "1"
    env["SYNCOPATE_POOL_BATCH"] = str(a.ppo_mini_batch_size)
    env["SYNCOPATE_RL_MODE"] = "colocate"          # main_ppo_pool 的分发键：V1 三种 trainer_mode 都走 main_ppo 入口
    env["SYNCOPATE_PREFIX_GROUPER"] = "1" if a.prefix_grouper else "0"
    # 0.8 时代的 worker 补丁在 0.9 上按 S1-4 分诊：删/停用的这里显式关掉（留 3：pool sampler · DDP/grad 探针（按需开）· lora-only ckpt 走上游）
    for k in ("SYNCOPATE_FSDP_DDP_FIX", "SYNCOPATE_FIX_POSTPROC_CONCAT", "SYNCOPATE_LORA_ADAPTER_SYNC", "SYNCOPATE_CKPT_LORA_ONLY"):
        env.setdefault(k, "0")
    env["WANDB_MODE"] = a.wandb_mode; env.setdefault("WANDB_PROJECT", a.project)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "1")
    print(f"[rl-v1] mode={a.mode} gpus={a.gpus} steps={a.steps} n={a.rollout_n} lora r={a.lora_rank} exclude={a.exclude_modules!r} "
          f"pool={'关' if a.no_pool else '开'} save_lora_only={a.save_lora_only}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
