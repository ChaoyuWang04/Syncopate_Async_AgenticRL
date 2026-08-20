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
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from syncopate.train.rollout_budget import (  # noqa: E402
    MAX_PROMPT_LENGTH as BUDGET_PROMPT,
    MAX_RESPONSE_LENGTH as BUDGET_RESPONSE,
)


def _assert_model_is_merged(model_path: str) -> None:
    """★ 2026-08-18（E21 同族排查，见 docs/syncopate/18 §3）：给错宁可报错。

    verl 的 LoRA merger 产出的是「**未改的**基座 + 独立的 `lora_adapter/`」——
    主权重里**不含**上一轮 RL 学到的东西。而本入口没有加载 adapter 的路径
    ⇒ 拿这样一个目录当起点会**静默丢掉整轮 RL**（形状对、能加载、指标正常、零报错）。
    """
    from pathlib import Path as _P
    if (_P(model_path) / "lora_adapter").exists():
        raise SystemExit(
            f"🔴 {model_path} 里有 lora_adapter/ ⇒ 它不是合并后的模型，主权重不含增量。\n"
            "   拿它当 RL 起点会静默丢掉上一轮 RL —— 见 docs/syncopate/18 §3。\n"
            "   ⚠️ 且 RL 那一级的增量合并进 bf16 保真残差 0.87（同文 §3.3），"
            "先决定接续方案，别直接合并。"
        )


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
        # ⚠️ fully_async 里 `train_batch_size` **必须是 0**（`fully_async_rollouter.py:414`
        # 有硬断言）。它的运转方式根本没有"一步取固定条数"这回事：rollouter 按
        # gen_batch_size=1 连续产样本，trainer 攒够 `ppo_mini_batch_size × require_batches`
        # 就训一步。⇒ 这个键在两种模式下语义不同，不能共用一个值。
        f"data.train_batch_size={0 if args.mode == 'fully_async' else args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.dataloader_num_workers=0",

        # ---- 模型 ----
        f"actor_rollout_ref.model.path={model}",
        # remove_padding 需要 flash-attn（verl 的 unpad_input 直接 import flash_attn.bert_padding）。
        f"actor_rollout_ref.model.use_remove_padding={str(args.remove_padding)}",
        # ★ 2026-08-13 起默认 flash_attention_2 —— 装上了真轮子
        # （mjun0812/flash-attention-prebuild-wheels 的 2.8.3+cu128torch2.9-cp312，
        #   cuobjdump 验过 sm_120 kernel 在内；varlen 与逐序列 SDPA 对拍差 7e-3 bf16 量级，
        #   跨序列隔离精确为 0）。此前写死 sdpa 是垫片时代的产物。
        # ⚠️ sdpa 回退仍保留（--attn-implementation sdpa），但要知道它的三笔账：
        #   rmpad 路径下 transformers 4.57 恒物化 [1,1,L,L] mask（连单序列都物化，
        #   masking_utils.find_packed_sequence_indices 永不返回 None）+ SDPA 有显式
        #   mask 时用不了内部 flash 后端 + 打包时跨序列象限白算 ⇒ 见 dynamic_bsz 注释。
        f"+actor_rollout_ref.model.override_config.attn_implementation={args.attn_implementation}",

        # ---- ★★ 融合 logprob/熵 kernel：RL 侧的「稀疏投影」----
        #
        # 同一个病：算 log_prob 要先物化 logits [micro_bs × seq_len × 151936]，
        # 5120 token 的序列 fp32 就是 3.1 GB，前向反向各一份。verl 的融合实现
        # 边算边归约，logits 从不落地（dense_common 分支覆盖 Qwen3）。
        #
        # 省下来的显存不是白省的 —— 它直接变成 `--rollout-gpu-util` 的额度，
        # 也就是 vLLM 的 KV 池（现在只有 ~1.8 GB / 12.5k token，是 preemption 的根源）。
        # **这是 Ostinato 的因果链在训练侧的第一个可执行落点。**
        #
        # ⚠️⚠️ **2026-08-13 实测：本机现在开不了，不是没试过。**
        #
        #     AssertionError @ verl/workers/utils/padding.py:131
        #     no_padding_2_padding: sequence_offsets[-1] != values.shape[0]
        #
        # 根因：`dense_common.forward_with_torch_backend` 返回的 log_probs/entropy
        # 形状跟着 hidden_states 走 —— `use_remove_padding=False` 时是 **padded [B, T]**，
        # 而 `_compute_old_log_prob` 无条件按 **unpadded 扁平 [总token数]** 去还原。
        # ⇒ **verl 0.8 的融合 kernel 路径假定了 `use_remove_padding=True`。**
        #
        # 而 remove_padding 需要 flash-attn，sm_120 上要自己编译（已知缺口 🟡）。
        # ⇒ 解锁顺序是「先装 flash-attn 打开 remove_padding，再开这个」，
        #   或者自己写一版不依赖 unpad 契约的融合 kernel（Ostinato K1 的 RL 变体）。
        #
        # ⚠️ 即使解锁了也不设默认开：数学等价但浮点路径不同，logprob 有 1e-6 级差异
        # ⇒ TIS 的 importance ratio 跟着动，要配 EVAL 配对回归才能上。
        f"actor_rollout_ref.model.use_fused_kernels={str(args.fused_kernels)}",
        f"actor_rollout_ref.model.fused_kernel_options.impl_backend={args.fused_kernels_backend}",

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

        # ⛔ **不要**打开 verl 的 `actor.use_prefix_grouper`（2026-08-19 实测）：
        #   它会启用 `ray_trainer._balance_batch` 的组感知划分，而那条路径在 verl 0.8.0
        #   里**本身是坏的** —— `workers/utils/padding.py` 假设张量是 nested：
        #        AttributeError: 'NoneType' object has no attribute 'is_nested'
        #   （上游 #7202 对 prefix_grouper_utils.py 的改动正是 "nested→padded"，
        #     我们撞的就是它修的那个 bug；而 #7202 已被维护者关闭。）
        #   ⇒ 我们改为**在 micro-batch 内自己重排**（见 verl_patches._patch_prefix_grouper），
        #     不依赖 verl 的划分，也就不需要这个配置项。
        #
        # ★ 但**必须关掉 `trainer.balance_batch`**：它按序列长度排序来均衡各卡负载，
        #   顺手把同一道题的 8 条答案打散 —— 实测一个 8 条的 micro-batch 里装的是
        #   5 个不同题目的碎片（组大小 [2,2,2,1,1]）⇒ 打包退化成「5 个前缀各算一次」，
        #   比不打包还慢。关掉之后 GRPO 天然的「每题连续 n 条」顺序才保得住。
        *( ["trainer.balance_batch=False"]
           if os.environ.get("SYNCOPATE_PREFIX_GROUPER") == "1" else [] ),
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.entropy_coeff=0",

        # ---- ★★★ 训练侧的两个大旋钮（2026-08-13 在 4 卡上实测出来的）----
        #
        # 背景：分卡异步跑通后的第一份耗时分解（1 训练卡 + 1 rollout 卡，稳态）：
        #     step 116.0s = update_actor 86.3 (74%) + ref 17.3 (15%)
        #                 + update_weights 12.3 (11%) + gen 4.4 (4%)
        # ⇒ **瓶颈已经不是 rollout 了**，是训练侧的前向/反向。
        #
        # ① `fsdp_size`：**在这台机器上必须是 1**。
        #    create_device_mesh(world_size, fsdp_size)（fsdp/utils.py:40）：
        #        fsdp_size <= 0 或 >= world_size  ⇒ 一维 mesh，全部参数切分（FULL_SHARD）
        #        否则                             ⇒ (world_size//fsdp_size, fsdp_size)，
        #                                            维度 ["ddp", "fsdp"]
        #    取 1 ⇒ fsdp 维长度为 1 = **不切分**。                                    ← 事实
        #
        #    ⛔⛔ **2026-08-18 更正（E21）**：这里原本还有一句
        #        「只在 ddp 维 all-reduce 梯度 = DDP」                              ← **推断，且是错的**
        #    上面三句是读码读出来的事实，那第四句是推的，排版一模一样 ⇒ 没人质疑它。
        #    实际行为：(N,1) 网格 ⇒ HYBRID_SHARD ⇒ PyTorch 见分片维=1 自动降级成 NO_SHARD，
        #    **却把归约留在那个大小为 1 的组上** ⇒ 空操作 ⇒ 三个 rank 各训各的 LoRA，
        #    只打一行 UserWarning，训练照常跑完、指标全正常。**静默失效两个月。**
        #    ⇒ 已由 `verl_patches._patch_fsdp_degenerate_mesh` 拦住（默认开启）。
        #    ★ 纪律：推断句一律标 `[推断，未验证]`，别和事实排在一起。
        #
        #    ⚠️ 实测代价：trainer 3 卡跑 FULL_SHARD 每步 1182s，1 卡不切分 198s ——
        #    **多两张卡慢 6 倍**。因为 5090 没有 P2P，卡间只有 6.4 GB/s（经主机中转），
        #    而 FULL_SHARD 每层前向反向都要把 8GB 权重 all-gather 回来。
        #    ⛔ 原来这里写「LoRA 下 DDP 只同步 66M 梯度（约 260MB ⇒ 40ms），三个数量级的差距」——
        #    那 260MB 是**算出来的**（66M×4B），从没量过；而 E21 之下这段流量**根本不存在**。
        #    ⇒ 量级方向大概率仍成立，但**具体数字要实测一次 NCCL 流量之后才能引用**（重测队列 R3）。
        #
        #    ⚠️ **`ulysses_sequence_parallel_size` 必须保持 1**（verl 默认值，我们不设它）：
        #    修复后 FSDP 按**默认进程组**（world 个 rank）除，而 verl 按 `dp_size = world // sp` 乘
        #    ⇒ 只有 sp=1 时两者才抵消。开 SP 之前先重做 0-A。
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={args.fsdp_size}",
        f"actor_rollout_ref.ref.fsdp_config.fsdp_size={args.fsdp_size}",

        # ② `use_dynamic_bsz`：按 **token 预算**打包，而不是按固定条数。
        #
        # ⛔⛔ **在本机是倒退，默认关掉了。** 原猜想是「序列才 4.1k token，
        # 一条一条算喂不饱 5090，打包成 16k 应该更快」。实测（同为 step 1，1 张训练卡）：
        #        静态 micro_batch=1   update_actor  84.5 s
        #        dynamic 16384        update_actor 184.9 s   ← **慢 2.19×**
        #    3 卡上同样的比值（187.8 / 84.5 ≈ 2.2），说明和卡数无关。
        #
        # 根因（2026-08-13 读码+CPU 实验证实，比原推断多两层）：垫片时代被迫走 sdpa，而
        # verl rmpad 传 attention_mask=None + 打包 position_ids —— 这套约定是为 flash-attn
        # varlen 设计的。落到 sdpa 上，transformers 4.57 的 masking_utils 走「物化 mask」路：
        #   ① 跨序列象限白算再 mask 掉：(16k)² ≈ 3.2×Σ(5k)²；
        #   ② mask 本身 16k² bool = 256MB/前向；
        #   ③ SDPA 拿到显式 mask 后用不了内部 flash 后端（要求 attn_mask=None），落到
        #      efficient/math。三笔叠加 ⇒ 2.2×。
        # 两个附带发现：
        #   🍀 正确性是运气好——4.57 的 find_packed_sequence_indices 把块对角 mask 建对了，
        #      旧版 transformers 会静默跨序列泄漏；
        #   🔴 该函数**永不返回 None**（单序列也返回全零张量）⇒ allow_is_causal_skip 恒
        #      False ⇒ **不打包的 rmpad 基线同样在物化 mask 的慢路径上**。
        # ⇒ 已装真 flash-attn 2.8.3（attn_implementation 默认已切 flash_attention_2），
        #    varlen 按 cu_seqlens 分段、不物化 mask ⇒ 垫片那条根因**已经消失**。
        #
        # ✅✅ **2026-08-19 重测完毕（E25），结论：仍然关，但理由整个换了。**
        #    旧理由「打包会让注意力退化成 O(总长²)」——**已随垫片一起作废，别再引用**。
        #    新理由是实测出来的：**打包在我们这个长度上根本没有收益可拿。**
        #        每条序列 = 4196 题面 + 654 回答 ≈ **4850 token**
        #        ⇒ 一次前向已经是 [4850 × 2560] 量级的 GEMM，**GPU 早就吃饱了**
        #        ⇒ micro_batch 1 → 2 实测：定长**慢 1.0%**、变长**慢 6.3%**，多花 4.2 GB 显存
        #        ⇒ 变长更慢是因为 mb>1 要 pad 到 max(lens)，**mb=1 反而等价于完美打包**
        #    ⇒ ★ **喂饱 GPU 的单位是 token，不是序列条数。**「batch=1」在这里不代表批次小。
        #    ⇒ B20 那个「FA2 下打包只有 +4~5%」也由此得到解释：收益本来就接近零。
        #    详见 docs/infra_exp/E25-trainer-feed.md §4.1
        #
        # ⚠️ 它同时管住 ref（`log_prob_use_dynamic_bsz` 默认跟随这个开关），所以 ref 一起变慢。
        f"actor_rollout_ref.actor.use_dynamic_bsz={str(args.dynamic_bsz)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={args.max_token_len_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={args.max_token_len_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={args.max_token_len_per_gpu}",

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
        # ---- ★★★ H1a：要不要每步把 vLLM 的权重搬去 CPU 再搬回来 ----
        #
        # `free_cache_engine=True`（verl 默认）= 每步 sleep/wake：
        #   sleep    vLLM 的 7.6 GB 权重 → CPU，KV 池丢弃
        #   wake_up  7.6 GB CPU → GPU，KV 池重建   ← **实测就死在这一步**
        #
        # 2026-08-13 实测：这条路在 v11 上**起不来**（6 次冒烟 3 次 wake_up OOM，
        # `CUDA Error: out of memory at cumem_allocator.cpp:112`）。根因不是显存总量不够
        # （10.49 + 9.41 = 19.9 GB，离 31.36 还远），而是 wake_up 要**一次性映射
        # 7.6 GB 连续物理页**，而 PyTorch 缓存分配器占着的 reserved 不还给驱动。
        #
        # ⚠️ 注意别和「权重同步」搞混（源码查证过，我自己先搞混过一次）：
        # verl 的同步**本来就是 adapter-only**（首次推基座，之后 TensorLoRARequest
        # 只推 132 MB，见 vllm_rollout/utils.py:262）。**搬 7.6 GB 的是 sleep/wake 本身。**
        #
        # `False` ⇒ `sleep()` 开头直接 return（vllm_async_server.py:626），
        # 权重常驻 GPU、永不搬运，OOM 的触发点整个消失。
        # 代价：vLLM 的 gpu_util 份额变成**常驻**，actor 的可用空间被压到
        # 31.36 − 9.41 ≈ 21.9 GB —— 而 v8 实测 actor 峰值 21.08 GB，**贴边**。
        # ⇒ 所以这个开关必须和「融合 logprob kernel」（去掉 3.1 GB 的 logits 物化）
        #   配套，否则只是把 OOM 从 wake_up 挪到训练侧。
        f"actor_rollout_ref.rollout.free_cache_engine={str(args.free_cache_engine)}",
        # ---- ★ 让 vLLM 的引擎统计真的打出来（H0 的核心观测）----
        #
        # 坑：`disable_log_stats=False` 传对了也看不到任何统计行 ——
        # **verl 把 `VLLM_LOGGING_LEVEL` 硬编码成 `WARN`**（constants_ppo.py:34），
        # 而吞吐 / prefix cache 命中率 / preemption / KV 池大小全是 **INFO 级**。
        # ⇒ 两个开关必须同时开，只开一个等于没开。
        # `main_ppo.py:77` 用 OmegaConf.merge 合并，我们的值覆盖它的默认值。
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL={args.vllm_log_level}",
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
        f"trainer.n_gpus_per_node={args.trainer_gpus}",
        "trainer.nnodes=1",
        # ⚠️ fully_async 分支会按 rollout_steps 重算 epochs —— 这里只给其余模式发基础值，
        #   不发两份让 hydra last-wins（重复覆写 = 读配置的人要靠顺序猜谁赢）。
        *( ["trainer.total_epochs=1"] if args.mode != "fully_async" else [] ),
        f"trainer.total_training_steps={args.steps}",
        # ★ 冒烟时 -1（不存）没问题，正式跑必须存 —— 否则两件事都做不了：
        #   1. 跑完没有 ckpt 可以在冻结 EVAL 上重评，等于白跑
        #   2. staleness 研究要的是**同一次训练里相隔 k 步的两个 policy**，
        #      没有中途 ckpt 就凑不出 k>0 的样本对（单卡跑不了真异步，只能这么合成）
        f"trainer.save_freq={args.save_freq}",
        f"trainer.val_before_train={str(args.val_before_train)}",
        f"trainer.test_freq={args.test_freq}",
        # ★ 默认 disable（每次都是新跑，防止误吃旧 ckpt）；断点续跑显式传 --resume auto
        #   （E29 的续跑等价性验证要走这条路）。
        f"trainer.resume_mode={args.resume}",
    ]

    # ---- ★★★ 分卡异步：rollout 和 training 各占各的卡（2026-08-13 上 4 卡后新增）----
    #
    # verl 的两条异步路径**都要求 rollout 和 training 不同卡**，这是第二研究目标
    # （异步 agentic RL）在单卡上一直卡着的原因：
    #     one_step_off_policy/ray_trainer.py:89   assert not self.hybrid_engine
    #     fully_async_policy                      trainer_pool 和 rollout 是两个独立资源池
    #
    # ⚠️ 这**不是一个开关**，是换了一整套 trainer：不同的 hydra 根配置
    # （`one_step_off_ppo_trainer` / `fully_async_ppo_trainer`）、不同的 worker 栈
    # （`separation.engine_workers.DetachActorWorker`，不是 colocate 那套 FSDP worker）、
    # 不同的入口 main。⇒ 由 `main_ppo_pool` 按 SYNCOPATE_RL_MODE 分发。
    # ★★★ 分片时自动切 NCCL 协议（2026-08-17 A8/A12 实测）
    #
    # NCCL 在 **3 个 rank** 上给 all_gather 选了坏协议：实测 2卡 51.0 / 4卡 37.9 /
    # **3卡 3.2 GB/s**（差 12×），而同样 3 卡上 all_reduce/reduce_scatter/broadcast 都正常。
    # 后果：3 卡 ZeRO-3 的 update_actor 47.94 s = DDP 的 6.02×。
    # ⇒ `NCCL_PROTO=LL128` 把 3 卡 all_gather 拉回 22.2 GB/s（6.9×），
    #   实跑 ZeRO-3 **47.94 → 14.40 s（3.33×）**，比值 6.02× → 1.81×。
    #
    # ⚠️ **不能全局开**：LL128 让 all_reduce −30%、broadcast −41%。
    #    而 **DDP 走的正是 all_reduce** ⇒ 只在 fsdp_size>1（真分片）时才开。
    if args.fsdp_size and args.fsdp_size > 1:
        overrides += [
            '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PROTO="LL128"',
        ]
        print(f"[rl] 分片模式（fsdp_size={args.fsdp_size}）⇒ 自动 NCCL_PROTO=LL128 "
              f"（3 rank 上 all_gather 会塌 12×，见 infra_exp/README §7.3）")

    # ★★ worker 钩子：**所有模式**都要挂（2026-08-18 修）。
    #
    # 原来它只写在分卡分支里 ⇒ **colocate 下 `verl_patches.setup_worker` 从不执行**。
    # 后果：2026-08-18 的 A17（对齐补丁，跑在 colocate 上）**两臂时间一模一样、
    # 判据行 0 条** —— 补丁压根没进 worker 进程，实验白跑一轮。
    # ★ 同一族的第五例（判据看起来在工作，量的却不是那件事）：
    #   这次是「机制只在一半的模式下接上」，而实验恰好跑在没接上的那一半。
    # ⚠️ colocate 下 worker 同样是独立的 Ray actor 进程，一样够不着 driver 的补丁。
    overrides += [
        "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="
        "syncopate.train.verl_patches.setup_worker",
    ] if args.mode == "colocate" else []

    # ★ 权重同步 bucket：**所有模式**都要（colocate 也走 NCCL checkpoint engine，
    #   见上面那段更正）。放在分支之前，避免再次只接一半。
    overrides += [
        f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes"
        f"={args.weight_sync_bucket_mb}",
    ]

    if args.mode != "colocate":
        overrides += [
            # ⚠️ 上游 bug：`one_step_off_ppo_trainer.yaml` 的 searchpath 写的是
            #     hydra.searchpath: [file://verl/trainer/config]
            # 这是**相对 CWD** 的路径 —— 只有在 verl 源码仓库根目录下跑才找得到。
            # 我们从自己的仓库根跑，于是 `Could not load 'ppo_trainer'`。
            # （同目录的 `fully_async_ppo_trainer.yaml` 写的是正确的 `pkg://verl.trainer.config`，
            #   所以这是 one_step_off 独有的疏漏，不是我们配错了。）
            # ⇒ 命令行覆盖成 pkg://，对两个模式都正确。
            "hydra.searchpath=[pkg://verl.trainer.config]",
            # colocate 的反面。ray_trainer.py:89 那条 assert 就是查它。
            "actor_rollout_ref.hybrid_engine=False",
            # ★★ 让 verl_patches 在 **Ray worker 进程**里也生效。
            #
            # `main_ppo_pool` 里的 `verl_patches.apply()` 只覆盖 driver（TaskRunner）进程。
            # 而 P2 要补的 `save_model_to_cpu` 活在 `WorkerDict` 这个 Ray actor 里，
            # 那个进程从不 import 我们的包 ⇒ 2026-08-14 实测：driver 侧判据行照常打印，
            # 断言照常在 worker 里触发。**判据行打出来了不等于补丁在需要它的进程里生效。**
            # ⇒ 用 Ray 的 worker 启动钩子（字符串形式，模块路径；PYTHONPATH 已由本脚本注入）。
            "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="
            "syncopate.train.verl_patches.setup_worker",
            # 顶层 `rollout` 节是异步配置独有的（不是 actor_rollout_ref.rollout）
            f"rollout.n_gpus_per_node={args.rollout_gpus}",
            "rollout.nnodes=1",
            # ---- ★★ 权重同步的缓冲区 ----
            # ⚠️⚠️ 2026-08-17 更正：这一条**以前只写在这个分卡分支里**，注释还写着
            # 「分卡模式独有的一笔显存开销」——**那句话是错的**。实测 colocate 同样走
            # NCCL checkpoint engine、同样在 `nccl_checkpoint_engine.py` 里分配 bucket，
            # 于是 `--weight-sync-bucket-mb` 在 colocate 下被**静默忽略**、吃 verl 的
            # 默认 2048 MB，第一次权重同步直接 OOM。⇒ 已挪到下面对所有模式生效。
            # （又一次「机制在但没接上」：参数在、help 在、就是这条路径没接。）
            #
            # 分卡之后 trainer 每隔几步要把权重推给 rollout 卡，走 NCCL checkpoint engine。
            # 它会在 **rollout 那张卡上**分配一个 bucket 大小的暂存区
            # （`nccl_checkpoint_engine.py:142 prepare`）。
            #
            # ⚠️ 2026-08-13 实测：这笔钱**必须提前从 vLLM 的份额里扣掉**，否则第一次
            # 权重同步就 OOM —— 而它发生在第一步训练之后，前面所有东西都建好了才炸，
            # 是最贵的一种失败。0.85 那次的账：
            #     vLLM 0.85 × 31.37 = 26.66 GB 常驻
            #     CheckpointEngineWorker 自己 2.58 GB + bucket 2.00 GB = 4.58 GB
            #     31.37 − 26.66 = 4.71 GB 可用 ⇒ 差 0.13 GB，贴着墙 ⇒ 挂
            # ⇒ 默认降到 0.75（vLLM 23.5 GB，留 7.8 GB），并把 bucket 做成显式参数。

            # ---- ★★★ staleness 修正：**必须显式设成 decoupled** ----
            #
            # ⚠️ verl 的两个异步配置都把 `bypass_mode` 设成了 True（全局默认是 False）。
            # 它不是"跳过修正"，是两种修正模式之一：
            #     True  bypass    两个 policy：old_log_prob := rollout_log_prob，
            #                     修正靠 PPO ratio r = π_θ/π_rollout 本身 + clip
            #     False decoupled 三个 policy：额外重算 old_log_prob，**显式 IS 权重**
            #
            # 🔴 **选 decoupled 的真正理由不是修正强弱，是「有没有刹车」**：
            # verl 自己的注释（`separation/ray_trainer.py:590`）写着
            #     "Compute rollout correction: IS weights, rejection sampling, and metrics
            #      Only runs in decoupled mode ... In bypass mode, this is skipped"
            # ⇒ **bypass 模式下 `rollout_corr/*` 一整套指标（ESS / IS ratio 分布 /
            #    超界比例）根本不产出**。2026-08-13 实测：one_step_off 跑完三步，
            #    把整个 step 的指标键全列出来核对，一个 rollout_corr 都没有。
            #
            # 而 `06-rl-run-protocol.md` 的停止条件 P6 是「**ESS/N 跌破 0.3 立即停**」。
            # 没有这个数 = 没有刹车，而 fully_async 的 staleness 比 one_step_off 大得多，
            # 正是最需要刹车的场景。
            # ⇒ 代价：每步多一次 actor 前向算 old_log_prob。
            #
            # ⛔ **这里原来写「约 +6–10 s。值得。」——那是估算，2026-08-14 实测推翻了它**：
            #    M7 fully_async 全程 37 步均值 **old_log_prob = 76.3 s，占 step 的 25.7%**
            #    （update_actor 98.1 / old_log_prob 76.3 / param_sync 55.8 / ref 39.2 / gen 18.6）
            #    ⇒ **低估了 8–13 倍。** 详见 docs/infra_exp/E12-weight-sync.md §4.4 与 §6-②。
            #
            # 代价要如实标价：76.3 s × 37 步 = 47 分钟，占全程 224 分钟的 21%。
            #
            # ⛔ **但别把它理解成「花 76 秒买一个 ESS 刹车」——那个说法是错的。**
            # 读码查实（experimental/separation/ray_trainer.py:503-530）：
            #     bypass    old_log_prob := rollout_log_prob，走 compute_policy_loss_bypass_mode()
            #     decoupled 重算 old_log_prob 当 **proximal anchor π_old**，走标准 PPO loss + IS 修正
            #               源码注释：「π_old ... serves as stable reference during mini-batch updates」
            # ⇒ `old_log_prob` **不是指标，是损失函数的一部分**（= AReaL 的 π_prox）。
            #   AReaL 消融：decoupled 下 η≤4 安全，朴素 PPO 超 η=1 就崩。
            #   ⇒ 这 76.3 s 买的是「异步在高陈旧度下还能训」本身，ESS 指标只是副产品。
            #
            # ⇒ ⛔ **因此「ESS 降频采样（每 N 步算一次 old_log_prob）」不可行** ——
            #   那不是少测几次，是**在 decoupled PPO 和 bypass PPO 两个目标函数之间横跳**。别做。
            # ⇒ ✅ 该查的是：old_log_prob(76.3s) ≈ ref(39.2s) 的 1.95×，而两者都是同批数据的纯前向。
            #   这一倍差从哪来？→ 挂 E01/E12 查。另外 E11 稀疏 logprob 直接命中它。
            #
            # ★ 教训：**注释里的估算数字会被后人当实测引用。** 估算就标「估算」，
            #   拿到实测后回填。（同一天在 E11 已经犯过一次：拿配置上限当实际值。）
            f"algorithm.rollout_correction.bypass_mode={args.bypass_mode}",
        ]
        if args.mode == "fully_async":
            # ⚠️ fully_async 的 rollout 计划是**另一套键**，和 actor_rollout_ref.rollout.* 平行。
            # 它按「攒够 ppo_mini_batch_size × require_batches 条样本就训一步」运转，
            # 而 rollouter 自己按 total_rollout_steps 收工。
            # ⇒ 要跑满 N 个训练步，rollout 侧至少要产出 N × mini_batch × require_batches 条。
            rollout_steps = args.steps * args.ppo_mini_batch_size * args.require_batches
            overrides += [
                f"rollout.n={args.rollout_n}",
                # `fully_async_rollouter.py:415` 有硬断言 assert gen_batch_size == 1
                "data.gen_batch_size=1",
                f"rollout.total_rollout_steps={rollout_steps}",
                f"async_training.require_batches={args.require_batches}",
                f"async_training.staleness_threshold={args.staleness_threshold}",
                f"async_training.trigger_parameter_sync_step={args.sync_every}",
                # ⚠️ partial_rollout=True 会把被打断的长轨迹续跑。它直接关系到
                # 「异步会不会系统性丢掉长任务」这个问题（分布漂移那条线），
                # 所以做成显式参数，别让它藏在 verl 的默认值里。
                f"async_training.partial_rollout={args.partial_rollout}",
                # rollouter 的收工点是 min(total_rollout_steps, len(dataloader)×total_epochs)
                # ⇒ epoch 数必须够，否则 rollout 提前停、训练步数达不到。
                f"trainer.total_epochs={max(1, -(-rollout_steps // 590) + 1)}",
            ]

    # ---- ★ 兜底：把「靠 verl 默认值才成立」的前提**显式钉死** ----
    # 纪律（2026-08-18）：**默认值必须是对的那个**，而"对"不能依赖第三方的默认值不变。
    overrides += [
        # ① Ulysses SP 必须为 1 —— E21 修复后 FSDP 按**默认进程组**（world）除，
        #    而 verl 按 dp_size = world // sp 乘，**只有 sp=1 时两者才抵消**（0-A / E21 §4.7.5）。
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1",
        # ② 采样不截尾 —— rollout 报的是 `processed_logprobs`（截尾后），而 trainer 侧的
        #    截尾器在 verl 里**是注释掉的**（torch_functional.py:672）⇒ 一开截尾两边就不是
        #    同一个分布族，每 token 系统性偏 Z、序列级 Z^694（E23 §2）。
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
    ]

    # ---- ★ 改动 2：LoRA ----
    if args.lora_rank > 0 and args.lora_merge:
        # ⚠️ `lora.merge` 这个键**已经在默认配置里**（默认 False）⇒ 直接覆盖，不能加 `+`
        #    （加 `+` 会报 "An item is already at ..."，2026-08-18 撞过一次）
        overrides += ["actor_rollout_ref.model.lora.merge=True"]
    if args.lora_rank > 0:
        overrides += [
            f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={args.lora_rank * 2}",
            # 挂全部线性层。只挂注意力容量差 2.8 倍——这是最常见的 LoRA 配置错误。
            #
            # ⚠️⚠️ **但这条只对 dense 成立。MoE 上 all-linear 是灾难**（2026-08-14 实测，E07 §4.5.3）：
            #   Qwen3-30B-A3B 的 Linear 总数 18,673，其中 **18,432（98.7%）在专家里**
            #   （gate/up/down × 128 专家 × 48 层）。挂 all-linear 的后果：
            #       可训练参数 1696 M（dense 4B 是 66M，**26×**）
            #       LoRA 张量 37,346 个（dense 是 504 个，**74×**）
            #       每步权重同步 3.39 GB（dense 0.13 GB）
            #   而 E13 已证明**逐张量拷贝是瓶颈**：504 个要 0.037 s，
            #   37,346 个线性外推 ≈ 2.7 s，proximal anchor 每步拷三趟 ⇒ **光快照就 ~8 s/步**。
            #   更根本：top-8/128 ⇒ 每个专家只见 1/16 的 token，绝大多数梯度极稀疏。
            #
            # ⇒ 做成显式参数。**dense 保持 all-linear（默认不变），MoE 请显式传 attn-router。**
            f"actor_rollout_ref.model.target_modules={args.target_modules}",
            "actor_rollout_ref.rollout.load_format=safetensors",
            # ★★ `layered_summon`：**在这台机器上很可能是净亏损，做成显式参数以便对照**
            #
            # 它的作用（`fsdp_utils.layered_summon_lora_params`）：逐层 summon 参数、
            # 取出该层的 LoRA、拷到 CPU，**并且每层结束都调一次 `empty_cache()`**：
            #     for 36 层: with FSDP.summon_full_params(layer): layer.state_dict() ... ; empty_cache()
            #
            # 它是为**真正分片的 FSDP** 设计的——分片时一次性 gather 整个模型会爆显存，
            # 所以宁可逐层来。**但我们跑的是 `--fsdp-size 1`（DDP，不分片）**，
            # 参数本来就是完整的，整体 summon 几乎免费，而逐层路径要付
            # 36×(summon + 全量 state_dict + empty_cache)。
            #
            # 实测线索（E12，2026-08-14）：trainer 侧 `update_weights` 耗时
            #     首次(推 8 GB 基座) 67.1 s  vs  稳态(只推 132 MB LoRA) 69.9 s
            # ⇒ **数据量差 60× 而耗时不变 ⇒ 成本与传输量无关，就在"取参数"这一步。**
            f"actor_rollout_ref.rollout.layered_summon={args.layered_summon}",
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
            # ★ 种子要真的传下去 —— 加了参数却不传，就是"机制在但没接上"
            f"data.seed={args.seed}",
            # ⛔ 2026-08-19 infra 修：**这一行会让 launch_rl 每次启动都死**。
            #    verl 的 `SamplingConfig` 字段只有 [_target_, temperature, top_k, top_p,
            #    do_sample, n]，`rollout.yaml` 顶层也没有 seed ⇒ Hydra struct 模式直接拒绝：
            #        Key 'seed' is not in struct / full_key: ...rollout.val_kwargs.seed
            #    ⇒ 100% 触发、无法起跑。`data.seed` 那条是有效的，保留。
            #    ★ 教训：**新增的 config 覆盖必须起一次跑才算落地** —— 它不是写完就成立的，
            #      而这条的失败发生在**任何训练开始之前**，任何冒烟都会抓到。
            f"algorithm.rollout_correction.rollout_is_threshold={args.rollout_is_threshold}",
            "algorithm.rollout_correction.rollout_is_batch_normalize=false",
        ]

    return overrides + list(args.extra)


def _visible_gpu_count() -> int:
    """当前进程能看到几张卡。优先信 CUDA_VISIBLE_DEVICES —— 这条线上最容易搞错的
    就是「机器上有 4 张」和「这次跑能用几张」不是一回事。"""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([x for x in vis.split(",") if x.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True)
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0


def _resolve_topology(args: argparse.Namespace) -> None:
    """把「模式」翻译成卡怎么分、vLLM 拿多少显存，并且**当场校验**。

    ⚠️ 这里刻意做成硬失败而不是自动降级：分卡模式静默退回单卡，
    表现是「跑起来了、但测的根本不是异步」——本项目最怕的那种失效
    （交接文档 §15：机制在，但没接上）。
    """
    total = _visible_gpu_count()

    if args.mode == "colocate":
        if args.trainer_gpus is None:
            args.trainer_gpus = 1
        if args.rollout_gpu_util is None:
            args.rollout_gpu_util = 0.40
        args.rollout_gpus = 0          # colocate 没有独立的 rollout 池
        return

    # ---- ⛔ 启动即校验：mini-batch 必须能被 trainer 卡数整除 ----
    #   2026-08-18 实测：`--ppo-mini-batch-size 2 --rollout-n 8 --trainer-gpus 3`
    #   ⇒ 每个 mini-batch 16 条序列分给 3 个 rank ⇒ verl 在**跑起来两分钟后**才炸
    #      `AssertionError: 16 % 3 != 0`（队列 T6/T8 各白跑一次）。
    #   ★ 判据要放在**最早能判的地方** —— 这是纯算术，启动时就能判。
    if args.mode != "colocate" and args.lora_rank >= 0:
        _seqs = args.ppo_mini_batch_size * args.rollout_n
        if args.trainer_gpus and _seqs % args.trainer_gpus != 0:
            # ⚠️ 候选范围**不要依赖另一个参数** —— 第一版用了 train_batch_size，
            #    而它在这条路径上可能还没被解析成想要的值 ⇒ 建议列表打成空的，
            #    "报错了但没告诉你怎么改"比不报错好不了多少。
            _ok = [m for m in range(1, 17) if (m * args.rollout_n) % args.trainer_gpus == 0]
            raise SystemExit(
                f"★ 配置非法：--ppo-mini-batch-size {args.ppo_mini_batch_size} × "
                f"--rollout-n {args.rollout_n} = {_seqs} 条序列，"
                f"**不能被 trainer 卡数 {args.trainer_gpus} 整除**。\n"
                f"  verl 会在跑起来之后才报 `{_seqs} % {args.trainer_gpus} != 0`，白烧一次启动。\n"
                f"  ⇒ 本机可用的 --ppo-mini-batch-size：{_ok or '（无，请调 --rollout-n 或 --trainer-gpus）'}")

    # ---- ⛔ 启动即校验：`--lora-merge` 与 E22 修法①（推 adapter）互斥 ----
    #   两者都开会把 399 个基座张量当 adapter 喂给 add_lora。
    #   ★ 守卫要放在**启动时**，不能放在第一次权重同步里 —— 那要等 10 分钟才炸
    #     （2026-08-18 实测：放在 update_weights 里，vLLM 还没起来就先因别的原因失败了，
    #      守卫根本没机会触发）。**判据要在最早能判的地方判。**
    if args.lora_merge and args.lora_rank > 0 and args.mode != "colocate" \
            and os.environ.get("SYNCOPATE_LORA_ADAPTER_SYNC", "1") == "1":
        raise SystemExit(
            "★ `--lora-merge` 与 E22 修法①（默认开启的 adapter 推送）互斥。\n"
            "  修法①每次推 adapter 张量；而 merge=True 会让引擎吐出合并后的全量权重，\n"
            "  rollout 侧仍会按 adapter 装载 ⇒ 把整份基座当 adapter。\n"
            "  ⇒ 二选一：**去掉 `--lora-merge`（推荐）** —— R0-b 实测 bf16 合并会毁掉\n"
            "           adapter 一半的作用（logprob 偏移中位 1.717e-02 = adapter 自身作用的 50%）；\n"
            "           或设 SYNCOPATE_LORA_ADAPTER_SYNC=0 退回合并模式（不推荐）。")

    # ---- 分卡模式 ----
    if total and total < 2:
        raise SystemExit(
            f"--mode {args.mode} 要求 rollout 和 training 在不同 GPU 上，当前只看到 {total} 张卡。\n"
            "  （verl 的两条异步路径都有这个硬前提：one_step_off 有 `assert not hybrid_engine`，\n"
            "    fully_async 的 trainer_pool 和 rollout 是两个独立资源池。）")
    if args.trainer_gpus is None:
        args.trainer_gpus = max(1, total - args.rollout_gpus) if total else 1
    if args.rollout_gpu_util is None:
        # rollout 独占整张卡，没有 actor 和它抢 —— colocate 那套 0.40 的账在这里不成立。
        # 但**也不是想给多少给多少**：权重同步的 bucket 也在这张卡上（见 build_overrides
        # 里的账本）。0.85 实测第一次同步就 OOM ⇒ 0.75，留 7.8 GB 给同步和碎片。
        args.rollout_gpu_util = 0.75

    used = args.trainer_gpus + args.rollout_gpus
    if total and used > total:
        raise SystemExit(
            f"卡不够：trainer {args.trainer_gpus} + rollout {args.rollout_gpus} = {used}，"
            f"但只有 {total} 张。")
    if total and used < total:
        print(f"[拓扑] ⚠️ 只用了 {used}/{total} 张卡，剩下 {total - used} 张闲着", flush=True)


# 候选跑的最少步数（**下限，不是目标**）。理由见 --purpose 那一段。
MIN_CANDIDATE_STEPS = 400


def write_run_purpose(save_path: Path, *, purpose: str, steps: int) -> None:
    """把"这一跑是干什么用的"落盘。

    ★ 没有这个标记，晋级闸就只能靠**问人**"这跑是不是候选" ——
      而那是个手动步骤，一定会被忘（本项目第一失效形状）。
    """
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "run_purpose.json").write_text(
        json.dumps({"purpose": purpose, "steps_requested": steps},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syncopate GRPO 启动器（单卡 5090 降配）")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    # ★ 2026-08-19：默认值跟着 DATA_VERSION 走（单一来源，见 08 §4.0「不该有副本」）。
    #   此前写死 "v3" —— 目录早已不存在，谁忘了传就 FileNotFoundError（第七形态：
    #   默认值静默指向另一件事；今天的 e26 冒烟第 2 次启动就死在这上面）。
    from syncopate.pipeline.split import DATA_VERSION as _DV
    parser.add_argument("--train-file", default=f"data/rl/{_DV}/train.parquet")
    parser.add_argument("--val-file", default=f"data/rl/{_DV}/val.parquet")
    parser.add_argument("--save-path", default="checkpoints/grpo/smoke")
    parser.add_argument("--project", default="syncopate")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    # ★★ 2026-08-19：此前**没有 seed 参数** ⇒ 两次同配置的跑不可比。
    #
    # 这不是理论顾虑，是当场付的账：E17 的 KL 两臂只差 `--use-kl-loss` 一个变量，
    # 但因为没有固定种子，**它们其实是两次独立的随机跑**。
    # 于是逐模板出现 4 个"显著退化"、1 个"显著提升"（|t|>2），
    # 而 KL 惩罚项只占损失的 **0.0019%** —— 它没有能力造成那种量级的差异。
    # ⇒ 那组差异量的是**跑间方差**，不是 KL 的效应。
    #
    # ★ 副产品（留着当尺子）：同配置两次跑的**模板级差异可以到 ±0.14** ——
    #   低于这个幅度的模板级"差异"不该当信号看。
    parser.add_argument("--seed", type=int, default=1234,
                        help="固定随机种子。**默认固定**：两次跑之间不该多一个自由变量，"
                             "要做多种子对照就显式传不同的值（同 wandb 默认开、要关得显式）")
    parser.add_argument("--experiment", default="smoke")
    parser.add_argument("--logger", default="console,wandb",
                        help="verl 会把 rollout_corr/* 和 critic/* 全套指标上报")

    parser.add_argument("--steps", type=int, default=10)
    # ══════════════════════════════════════════════════════════════════
    # ★★★ 这一跑是干什么用的（2026-08-19 立）
    # ══════════════════════════════════════════════════════════════════
    #
    #   probe      精度/吞吐实验 —— 跑几步就够，**不受任何约束**
    #   candidate  **上线候选** —— 必须跑到"没东西可学了"，不是跑到步数就停
    #
    # ★★ 默认是 `probe`，这是刻意的：infra 一直在用 RL 跑短的精度实验，
    #   默认成 candidate 会**当场挡住他们**，而他们本来就不需要跑到没梯度。
    #
    # ⚠️⚠️ 那"主线忘了声明 candidate"怎么办？——**不靠记性**：
    #   约束不加在**起跑**上，加在**晋级**上（`scripts/candidate_gate.py`）：
    #   任何跑都随便跑，但**声称自己是上线候选**的跑必须过闸。
    #   ⇒ 忘了声明的后果是"晋级时被拦下"，不是"悄悄用了一个短跑当候选"。
    parser.add_argument("--purpose", default="probe", choices=["probe", "candidate"],
                        help="probe=实验（不受约束）· candidate=上线候选（受最少步数 + 完成判据约束）")
    parser.add_argument("--rollout-n", type=int, default=8, help="GRPO 组大小")
    parser.add_argument("--train-batch-size", type=int, default=6)   # 与 mini_batch 同批宽（fully_async 下 verl 强制 0，见 §data 处注释）
    parser.add_argument("--val-batch-size", type=int, default=2)
    # ★ 2026-08-19 默认 2 → 6：2×8=16 不能被 3 卡整除，默认起手就被起跑断言拦。
    #   6×8=48 是守卫 A4（≥24 条）标定的标准批；E20 更新次数实验要小批就显式传。
    parser.add_argument("--ppo-mini-batch-size", type=int, default=6)
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="每次前向算几条**序列**（不是几个题目）。★ 默认跟 PrefixGrouper 联动"
                             "（2026-08-20）：PG 开 ⇒ **8**（一组一批，E26 §6.6 实测最优，mb16 反慢 "
                             "5.7%%）；PG 关 ⇒ **1**（E25：1→2 定长慢 1.0%%/变长慢 6.3%%，4 OOM——"
                             "一条序列 ~4850 token，GPU 早吃饱了）。显式传值则不联动")
    # ★ 2026-08-18：默认值从 `rollout_budget` 取 —— **训练与评测共用一份**。
    #   显式传别的值是允许的，但 check_pipeline_invariants 会把不一致标红。
    parser.add_argument("--max-prompt-length", type=int, default=BUDGET_PROMPT)
    parser.add_argument("--max-response-length", type=int, default=BUDGET_RESPONSE)
    # ⚠️ verl 的 multi_turn.max_assistant_turns 我们的自定义 loop **不消费**（真上限 =
    #   每 case 的 max_steps，10/12/14，经 extra_info 进 RolloutConfig；实跑最大 12 步）。
    #   写成全库上限 14：防止这行"看着像上限"误导排查，也防未来 verl 启用时把长链切了。
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--rollout-gpu-util", type=float, default=None,
                        help="vLLM 的显存份额。★ 它是**占总显存的比例**，而且必须把 vLLM "
                             "自己的模型权重(4B bf16≈8GB)也装进这个预算里，剩下的才是 KV cache。"
                             "不传则按模式取：colocate 0.40（2026-08-13 实测：KV 池 4.19GB/"
                             "30528 token，actor 峰值 13.9GB / 上限 18.8GB）；"
                             "分卡模式 0.85（rollout 独占整张卡，没人和它抢）")

    # ---- ★★★ 分卡异步（4 卡才有的东西）----
    parser.add_argument("--mode", default="colocate",
                        choices=["colocate", "one_step_off", "fully_async"],
                        help="colocate=rollout 和 train 同卡（单卡时代唯一能跑的）；"
                             "one_step_off=分卡、落后一步；fully_async=分卡、两个独立池。"
                             "★ 后两个**要求至少 2 张卡**，是第二研究目标的入口。")
    parser.add_argument("--trainer-gpus", type=int, default=None,
                        help="训练池的卡数。不传：colocate 用 1，分卡模式用「总卡数 − rollout 卡数」")
    parser.add_argument("--rollout-gpus", type=int, default=1,
                        help="rollout 池的卡数（只在分卡模式下有意义）。"
                             "⚠️ agentic 负载的 rollout 很重，最优配比大概率不是 2+2，本身就是个实验")
    parser.add_argument("--weight-sync-bucket-mb", type=int, default=512,
                        # ⛔ 2026-08-18：默认从 verl 的 2048 改成 512。
                        #    2048 **已知会 OOM**（所有模式，不只分卡）：CheckpointEngine 会在
                        #    目标卡上分配一个 bucket 大小的暂存区（nccl_checkpoint_engine.py:142）。
                        #    ⇒ 「默认值必须是对的那个」——不能让人靠记忆去传一个救命参数。
                        help="分卡模式：权重从 trainer 推给 rollout 时的暂存区大小（MB）。"
                             "★ 它占的是 **rollout 卡**的显存，必须和 --rollout-gpu-util "
                             "一起算账，否则第一次权重同步才 OOM（见代码里的账本）")
    parser.add_argument("--bypass-mode", default="False", choices=["True", "False"],
                        help="staleness 修正的模式。★ **默认 False（decoupled）**：三个 policy、"
                             "显式 IS 权重，**而且只有这个模式产出 ESS 等 rollout_corr/* 指标** "
                             "—— 停止条件 P6「ESS/N<0.3 立即停」依赖它。"
                             "verl 的异步配置默认 True（bypass），那样跑等于没有刹车")
    parser.add_argument("--require-batches", type=int, default=1,
                        help="fully_async：攒够几个 mini-batch 才训一步")
    parser.add_argument("--staleness-threshold", type=float, default=0.1,
                        help="fully_async：允许的样本陈旧度上限")
    parser.add_argument("--sync-every", type=int, default=4,
                        help="fully_async：每几步把权重推给 rollout 池")
    parser.add_argument("--partial-rollout", default="True", choices=["True", "False"],
                        help="fully_async：权重同步时被打断的轨迹要不要续跑。"
                             "★ 关掉它，长轨迹会被系统性丢掉 —— 而长链正是 agentic 的核心能力，"
                             "这条直接关系到分布漂移那条研究线")
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
    parser.add_argument("--target-modules", default="all-linear",
                        # ⚠️ argparse 的 help 会走 %-格式化 —— 里面的百分号必须写成 %%，
                        #    否则 `--help` 直接抛 ValueError（2026-08-14 踩过）。
                        help="LoRA 挂在哪些线性层。**dense 用默认 all-linear**；"
                             "★ **MoE 千万别用 all-linear**（Qwen3-30B-A3B 上 98.7%% 的 Linear 在专家里，"
                             "可训练参数 26×、LoRA 张量 74×、每步同步 3.39 GB）—— "
                             "传 `[q_proj,k_proj,v_proj,o_proj,gate]` 之类的显式列表。见 E07 §4.5.3。")
    parser.add_argument("--layered-summon", default="True", choices=["True", "False"],
                        help="LoRA 权重同步时逐层 summon（每层 summon+state_dict+empty_cache）。"
                             "★ 默认 True 只是**保持现状以便对照**——它是为分片 FSDP 设计的，"
                             "而本机 --fsdp-size 1 不分片，很可能是净亏损。见 E12 与代码处注释。")
    parser.add_argument("--lora-merge", action="store_true",
                        help="权重同步前先把 LoRA 合并进基座（`model.lora.merge=True`）。"
                             "★★ 2026-08-18 实测（0-B）：**disaggregated 模式（fully_async / "
                             "one_step_off）不开这个就是错的** —— `engine_workers.py:698` 只调一次 "
                             "`get_per_tensor_param()`（`base_sync_done=False`），"
                             "`collect_lora_params` 会**显式跳过所有 lora_ 张量** ⇒ "
                             "每次同步推的是 8.4 GB **冻结基座**，adapter 一个字节都不推 ⇒ "
                             "rollout 的策略永远停在起点，训练学到的东西从不参与生成。"
                             "（colocate 那条路调两次、会推 adapter，**不受影响**。）"
                             "判据：`SYNCOPATE_SYNC_PAYLOAD=1` 打的『盯住层 ‖W‖』必须**不等于**起点模型。")
    parser.add_argument("--nvtx", action="store_true",
                        help="给 verl 的每个计时段套一层 NVTX range（E01/A5 的门槛）。"
                             "★ verl 自己那个 `marked_timer` 的 docstring 说会打 marker，"
                             "**函数体里一个都没有** ⇒ 不打这个，nsys 的 trace 切不开阶段。"
                             "只在采 profile 的那一跑开，平时别开。")
    parser.add_argument("--fsdp-size", type=int, default=1,
                        help="FSDP 的切分组大小。★ **本机默认 1 = 不切分 = DDP**。"
                             "-1 是 verl 的默认（全部切分 / FULL_SHARD），"
                             "在这台没有 P2P 的机器上实测慢 6 倍，别用")
    parser.add_argument("--dynamic-bsz", default="False", choices=["True", "False"],
                        help="按 token 预算打包 micro-batch。⛔ **默认关，而且 E25 已重测过**："
                             "我们每条序列已有 ~4850 token，GPU 本来就吃饱了，打包无收益可拿；"
                             "变长负载上 mb=1 等价于完美打包。⚠️ 旧理由（垫片时代的「慢 2.2×」）"
                             "**已作废，别再引用**。见 docs/infra_exp/E25-trainer-feed.md")
    parser.add_argument("--max-token-len-per-gpu", type=int, default=16384,
                        help="--dynamic-bsz 的预算：每个 micro-batch 最多多少 token。"
                             "★ 这是**显存旋钮**，调大 = 激活值更大 = 更快但更吃显存")
    parser.add_argument("--fsdp-dtype", default="bf16", choices=["bf16", "fp32"],
                        help="FSDP 持有参数的精度。LoRA 下 98.4%% 的参数冻结，"
                             "fp32 主权重是纯浪费（4B 要 16GB，实测直接把 vLLM 挤死）")
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = 全参；4B 必须给 32")
    # ★ 2026-08-19 默认 1e-6 → 3e-5：1e-6 是 M7 时代的遗留 —— 位移 0.0093% = 白训。
    #   协议 A-3 的标准值就是 3e-5（r1_seqis / +0.137 那批跑全是它）。默认值必须是对的兜底。
    parser.add_argument("--lr", default="3e-5")
    parser.add_argument("--remove-padding", default="True", choices=["True", "False"],
                        help="抠掉 pad token 再算（5120→~4200）。2026-08-13 晚起本机已装真"
                             "flash-attn 2.8.3（预编译轮子，sm_120 kernel 验证过），垫片已退役")
    # ★★★ 2026-08-17 的一整轮弯路，结论写在这里（详见 pyproject 的 [tool.uv.sources] 注释）：
    #
    # 社区那个 `cu128torch2.9cxx11abiFALSE` 轮子**前向三项全过、反向全错**
    #   flash_attn_func 反向        dq/dk/dv 全 nan
    #   flash_attn_varlen_func 反向 有限但**恒为 0**   ← verl 的 rmpad 走这条，★静默失败
    # 后果：每步 grad_norm=nan ⇒ verl 打 WARN 并 `optimizer.zero_grad()` **跳过更新**
    #       ⇒ RL 完全空转（实测每步都跳，模型一次没动过）。
    # ⇒ 教训：「import 成功 ≠ 契约满足」的下一层是 **「前向对 ≠ 反向对」**。
    #
    # 现在换成**官方 cu13torch2.9 轮子 + PyPI 的 CUDA 13 运行时**，反向与 fp32 参考
    # 对到 4–5 位有效数字，真实 RL 里 grad_norm 0.0147/0.0224（M7 整跑是 0.011–0.06，同区间）。
    # 实测 update_actor 16.78 s vs sdpa 26.01 s ⇒ **1.55×**，所以默认仍是 FA2。
    #
    # ⚠️ 换轮子/换机器后**先跑 `scripts/check_flash_attn_backward.py`**，它专门拦
    #    「反向恒为 0」这种没有 nan、没有报错、训练照常跑完的静默失败。
    #    判据不过就显式传 `--attn-implementation sdpa`（正确但慢 ~1.55×）。
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa"],
                        help="训练侧 attention 实现。默认 flash_attention_2（真轮子已装）。"
                             "sdpa 是垫片时代的回退项——注意 rmpad 路径下它恒物化 [1,1,L,L] "
                             "mask 且丢失 fused 后端（见 build_overrides 里的注释），只用于对照/排障")
    parser.add_argument("--no-pool", action="store_true",
                        help="退回 verl 原生的均匀采样（对照组）。默认启用动态分池："
                             "按组内 reward 方差加权，饱和的题降采样但不剔除")
    parser.add_argument("--pool-state", default=None,
                        help="池子状态的落盘路径；不给则放在 save-path 下")
    parser.add_argument("--attention-backend", default="TRITON_ATTN",
                        help="vLLM 的注意力后端。★ 本机默认的 FA2 起不来（PTX 工具链比驱动新），"
                             "TRITON_ATTN 是实测能用且最快的那个。驱动够新的机器上可以换回 "
                             "FLASH_ATTN")
    parser.add_argument("--vllm-log-level", default="INFO", choices=["INFO", "WARN", "ERROR"],
                        help="vLLM 日志级别。★ 默认 INFO：KV 池大小 / prefix cache 命中率 / "
                             "preemption 全是 INFO 级，verl 默认的 WARN 会把它们全吞掉")
    parser.add_argument("--free-cache-engine", default="False", choices=["True", "False"],
                        help="每步是否 sleep/wake vLLM（搬 7.6 GB 权重）。★ 默认 False —— "
                             "True 在 v11 上 wake_up OOM **起不来**（2026-08-13 实测 6 次冒烟 3 次挂）。"
                             "⚠️ 这是**单卡 colocate** 的产物：4 卡上 rollout 和 trainer 分卡后"
                             "这个问题不存在，届时可改回 True（详见 build_overrides 注释）")
    parser.add_argument("--fused-kernels", default="True", choices=["True", "False"],
                        help="融合 logprob/熵 kernel（不物化 [seq×151936] 的 logits）。"
                             "★ actor 峰值 −5.0 GB（2026-08-13 实测）。"
                             "⚠️ **必须和 --remove-padding True 一起开**：verl 的融合路径"
                             "假定 unpadded 扁平契约，单独开会 AssertionError@padding.py:131")
    parser.add_argument("--fused-kernels-backend", default="torch", choices=["torch", "triton"],
                        help="融合 kernel 的实现后端")
    parser.add_argument("--no-engine-stats", action="store_true",
                        help="关掉 vLLM 周期性统计日志（默认开：吞吐/prefix cache 命中率/preemption）")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="vLLM KV cache 精度（fp8_e4m3/fp8_e5m2）。KV 池容量 ×2 的免费杠杆；"
                             "默认不动。开了必须配 EVAL 128 配对回归验精度（Ostinato A1）")
    # ★★ 2026-08-20（Chaoyu 拍板）：KL **默认关**。E17 两臂：砍 KL 省 15.4%（= ref 整遍
    #   前向）、任务分 −0.009 < MDE（无差异）；cand_v13r2_e1 400 步 KL-off 长跑兑现
    #   （判据③ rollout_corr/kl 不吃 ref，全程中位 4e-4 在地板）。
    #   fabricated_safety_line_cap 仍是常驻观察（02 §1），每次 compare 必看。
    parser.add_argument("--use-kl-loss", default="False")
    parser.add_argument("--resume", default="disable", choices=["disable", "auto"],
                        help="verl 的 trainer.resume_mode。默认 disable（新跑不吃旧 ckpt）；"
                             "auto = 从 save-path 里最新 ckpt 续跑（E29 续跑验证 / 长跑恢复用）")
    parser.add_argument("--val-before-train", default="False")
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--save-freq", type=int, default=25,
                        help="每 N 步存一次 ckpt。★ 正式跑不能是 -1：跑完没 ckpt 就没法重评，"
                             "而且 staleness 研究要的就是同一次训练里相隔 k 步的两个 policy。\n"
                             "⚠️ 默认从 10 调到 25（2026-08-13）：verl 0.8.0 的 FSDP "
                             "checkpoint manager 存的是**全量** state_dict，LoRA 训练下 "
                             "97.1%% 是和基座逐字节相同的冻结权重 —— 一个 ckpt 8.5GB，"
                             "其中只有 252MB 是训练产物。12 个 ckpt 吃掉 98GB 才发现。\n"
                             "跑完用 scripts/prune_rl_ckpts.py 瘦身（只留 LoRA 权重）。")

    parser.add_argument("--rollout-correction", action="store_true", default=True)
    # ══════════════════════════════════════════════════════════════════════
    # ★★★ 2026-08-19：默认改回 **sequence**（这是第二次改，理由和第一次不同）
    # ══════════════════════════════════════════════════════════════════════
    #
    # ⛔ 2026-08-18 曾从 sequence 改成 token，依据是「序列级 IS 指数脆弱」，
    #    实测支撑是 `chi2_seq 64.19 vs chi2_token 0.065`（差 989×）。
    #    ⇒ **那批数字在作废清单里**（`21 §2.1`：E20 的全部 ESS/chi2 数字，B1+B2 污染）。
    #      它们量的是「trainer 的权重从没推给 rollout」那个**无界 bug** ——
    #      策略错位无限增长，序列级当然指数崩塌。
    #      **那不是序列级的性质，是那个 bug 的性质。**
    #
    # ✅ 干净基线上的实测（`seqis_long120`：序列级 · lr 3e-5 · **120 步**，
    #    约等于一个 epoch 的 88%）：
    #        ESS 前半均值 0.8768 · 后半均值 0.8734 · 线性斜率 **+0.00016/点**
    #        全程在 [0.78, 0.94] 震荡，**没有衰减趋势**
    #        离 A4 停机线（需掉到 0.500）有 **1.6 倍**余量
    #
    # ✅ 而行为维度上序列级明显更好（`compare` 的行为读数，同一份冻结 EVAL）：
    #        该 defer   序列 97%（= 起点，没掉）  vs  token **83%**
    #        误 defer   序列 0.1%                 vs  token 1.3%
    #        REJ 类     序列 −0.031               vs  token **−0.188**
    #        任务总分   两者**完全打平**（+0.000，MDE 0.016）
    #
    # ★★ 决定性的理由不是上面的数，是**可观测性的不对称**：
    #        选 sequence  ESS 在 [0.78, 0.94] 真实波动 ⇒ 它是一个**有读数**的仪表
    #        选 token     ESS ≈ 0.999 恒定          ⇒ 它是一个**永远不会响**的警报器
    #    而 token 级的 ESS≈1 **不是"漂移小"的证据，是算术结果** ——
    #    token 级按定义就不去连乘那些比值，所以它测不到序列级的错位；
    #    再叠一层：verl 报的 ESS 是 `clamp(0, 2.0)` 之后算的（`21 §3.3`）。
    #    ⇒ 选 token 等于主动选一个测不到东西的指标，再用"它没报警"来安心。
    #
    # ⚠️⚠️ **但不要据此说「ESS 这个指标没用」** —— 这是两回事：
    #    `[实测]` `seqis_long120` 的 `partial_ratio` **30 个点全是 0.0**
    #    ⇒ **没有任何一条轨迹跨越过权重版本边界**。
    #      trainer 的一步比 rollout 慢得多，rollout 每次都早早做完在等
    #      ⇒ **我们从来没有真正跑出过 fully_async 的陈旧度条件。**
    #    ⇒ π_rollout ≈ π_train ⇒ IS 修正本身就近乎恒等
    #      ⇒ 上面那个「任务分完全打平」**不是"两者一样好"，是"这个条件下 IS 几乎没参与"**。
    #    ⇒ **ESS 的作用没有被观测出来 ≠ ESS 没有作用。** 它只是还没面对它该检测的条件。
    #      真到了陈旧度起来的负载（长尾工具延迟 / rollout 更快 / sync_every 更大），
    #      这个默认值要重新审 —— 那时序列级**可能真的**会塌，而那正是它会告诉我们的。
    parser.add_argument("--rollout-is", default="sequence", choices=["token", "sequence"],
                        help="重要性采样的聚合口径。**默认 sequence**（2026-08-19，见源码里的长注释）："
                             "干净基线 120 步实测序列级 ESS 无衰减，而行为维度上它明显更好"
                             "（该 defer 97%% vs 83%%）；且序列级的 ESS **会动**，token 级恒 ≈1 "
                             "⇒ 后者等于一个永不报警的警报器。"
                             "⚠️ ESS/N 真的跌破 0.3 时，换成 token 是**逃生口**（06 §2.B）。"
                             "⚠️ 陈旧度条件至今没被跑出来过（partial_ratio 全程 0）⇒ 结论有范围。")
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

    # ★ think 开关只给评测探针（E27），训练路径启动即拦（最早能判的地方）。
    #   理由：SFT 是 enable_thinking=False 模板练的，开 think 会让「增量拼接 vs
    #   整段渲染」逐 token 不相等（rollout_loop.py:50 全文）⇒ 两阶段分布对不齐且不报错。
    #   think-on 的训练要等带思考的 SFT 数据，不是拨个开关的事。
    if os.environ.get("SYNCOPATE_THINK", "0") == "1":
        raise SystemExit(
            "🔴 SYNCOPATE_THINK=1 只允许用于评测探针（E27，scripts/run_e27_think_probe.sh）。\n"
            "   训练路径禁止开 thinking：SFT 是 think-off 模板练的，开了两阶段分布对不齐\n"
            "   （rollout_loop.py:50 有全文与测试）。请去掉该环境变量再起训练。")

    # ★ 候选跑的**最少**步数。⚠️ 它是**下限不是目标** ——
    #   真正的停止条件是「零梯度率不再创新高」（scripts/pool_readout.py）。
    #   [依据] e17a 跑 60 步时零梯度率仍在创新高（15%→52%），且 RL 桶只覆盖 22.7%；
    #   一个 epoch = 824/6 ≈ 137 步只让每条题被看**一次**，
    #   而分池的 WEIGHT_FLOOR=0.05 本身就预设了几十轮往返。
    #   ⇒ 400 步 ≈ 3 个 epoch，是"分池能开始起作用"的下限，不是"够了"。
    if args.purpose == "candidate" and args.steps < MIN_CANDIDATE_STEPS:
        raise SystemExit(
            f"🔴 --purpose candidate 要求至少 {MIN_CANDIDATE_STEPS} 步（本次 {args.steps}）。\n"
            f"   ⚠️ 而且步数是**下限不是目标**：真正该停的时候是"
            f"「零梯度率不再创新高」（scripts/pool_readout.py）。\n"
            f"   ⇒ 只是做实验的话用 --purpose probe（默认），不受任何约束。")
    _assert_model_is_merged(str((ROOT / args.model).resolve()))
    _resolve_topology(args)

    # ★★ 2026-08-20（Chaoyu 拍板）：PrefixGrouper **默认开**。证据链已闭合：
    #   E26 端到端 2.31× + cand_v13r2_e1 全程 400 步（PG on + mb8）四常驻判据全绿、
    #   候选 +0.186 晋级评测兜底到账（/MAINLINE-INFRA 五项确认①的对赌兑现）。
    #   显式 SYNCOPATE_PREFIX_GROUPER=0 可关（对照实验用）。
    #   ⚠️ 必须设在 build_overrides **之前**：第 210 行的 balance_batch=False
    #   联动读的是本进程环境变量，只塞进子进程 env 它不会生效。
    os.environ.setdefault("SYNCOPATE_PREFIX_GROUPER", "1")
    pg_on = os.environ["SYNCOPATE_PREFIX_GROUPER"] == "1"
    # ★ micro_batch 跟 PG 联动（E26 §6.6：PG 下最优 = mb8「一组一批」，mb1 无组可打包；
    #   PG 关时 E25 的结论不变：mb1 最优）。显式传值则两种情况都尊重。
    if args.micro_batch_size is None:
        args.micro_batch_size = 8 if pg_on else 1

    # ★ 走我们的薄壳而不是 verl 的入口：它只把训练集采样器换成动态分池，
    # 其余原样交给 verl（见 main_ppo_pool 的模块 docstring）。
    # --no-pool 退回 verl 原生的均匀采样，用于对照实验。
    entry = "verl.trainer.main_ppo" if args.no_pool else "syncopate.train.main_ppo_pool"
    cmd = [sys.executable, "-m", entry, *build_overrides(args)]
    if args.dry_run:
        print(" \\\n  ".join(cmd))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["SYNCOPATE_ASYNC_VERIFIER"] = args.async_verifier
    env["SYNCOPATE_RUN_ID"] = args.experiment
    env["SYNCOPATE_LATENCY_SCALE"] = str(args.latency_scale)
    # 下发侧记账，和 trainer.rollout_data_dir 的训练侧 dump 配对算分布漂移
    write_run_purpose(ROOT / args.save_path, purpose=args.purpose, steps=args.steps)
    env["SYNCOPATE_DISPATCH_LOG"] = str(ROOT / args.save_path / "dispatched.jsonl")
    # 动态分池的状态（per-case 的 ema_std / seen / last_seen_step），断点续跑要用
    env["SYNCOPATE_POOL_STATE"] = str(
        Path(args.pool_state) if args.pool_state else ROOT / args.save_path / "pool_state.json")
    env["SYNCOPATE_POOL"] = "0" if args.no_pool else "1"
    # ★ 采样器的批宽（= 去重窗口宽）。⚠️ 不能让它读 data.train_batch_size ——
    #   fully_async 下 verl 强制那个值为 0 ⇒ 窗口退化成 1，P4 的结构性去重名存实亡
    #   （e26ab 实跑 batch=1，零重复纯靠 659 条大池子的运气）。fit 批宽 = mini_batch。
    env["SYNCOPATE_POOL_BATCH"] = str(args.ppo_mini_batch_size)
    if args.nvtx:
        env["SYNCOPATE_NVTX"] = "1"
        print("[rl] NVTX 阶段标注已开 —— 判据是两侧各一行 `[verl-patch] NVTX 阶段标注 ✓`"
              "（driver 一行、worker 进程一行）。只有一行就说明作用域又漏了一半。")
    # 薄壳按它选 verl 的哪个 main（三套不同的 trainer，不是一个开关）
    env["SYNCOPATE_RL_MODE"] = args.mode
    # 这两个开关决定实验的物理含义，必须打印出来——静默的默认值是最难查的那种错
    print(f"[实验设定] latency_scale={args.latency_scale}  async_verifier={args.async_verifier}"
          f"  rollout_is={args.rollout_is}(阈值 {args.rollout_is_threshold})"
          f"  prefix_grouper={'开' if pg_on else '关'}(mb={args.micro_batch_size})"
          f"  kl={'开' if args.use_kl_loss == 'True' else '关'}")
    if args.mode == "colocate":
        print(f"[拓扑] colocate：rollout 和 train 共用 {args.trainer_gpus} 张卡"
              f"  vLLM 份额 {args.rollout_gpu_util}")
    else:
        print(f"[拓扑] {args.mode}：trainer {args.trainer_gpus} 卡 / rollout {args.rollout_gpus} 卡"
              f"  vLLM 份额 {args.rollout_gpu_util}")
    env["WANDB_MODE"] = args.wandb_mode
    env.setdefault("WANDB_PROJECT", args.project)
    # ⚠️ 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
    # 老师的脚本里有这一行（64 卡场景下能省显存碎片），但它和 vLLM colocate 用的
    # 内存池直接冲突，engine 启动就会挂：
    #   AssertionError: Expandable segments are not compatible with memory pool
    # 参见 pytorch/pytorch#147851。抄配置时最容易连这种坑一起抄过来。
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "1")
    # ★★★ vLLM 的注意力后端（2026-08-13 在 4×5090 / 驱动 570.195.03 上实测）
    #
    # 默认的 FlashAttention-2 在 sm_120 上走 PTX JIT，而那份 PTX 是用比本机驱动更新的
    # 工具链编的 ⇒ `CUDA error: the provided PTX was compiled with an unsupported toolchain`。
    # **模型能加载、引擎能起、第一次生成才炸** —— 前面所有东西都建好了才暴露。
    #     默认(FA2) ❌   FLASHINFER ❌   TRITON_ATTN ✅   FLEX_ATTENTION ✅
    # Triton 是本地 JIT，用的就是本机 CUDA 12.8，不存在错配。
    # ⚠️ setdefault：驱动够新的机器上导出这个变量就能换回 FA2（更快）。
    env.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
    # ★ Ray 的对象存储默认占 RAM 的 30%（本机 /dev/shm 有 16GB 可用，它就敢要 9GB）。
    #
    # 实测：本机 30.9GB 内存，Ray 报 29.78GB 用满后直接杀 worker，**param_offload
    # 关着也照样爆**。我们的 batch 很小，对象存储根本用不了那么多——
    # 它只是按比例预留，然后把真正需要内存的 vLLM 和 trainer 挤死。
    env.setdefault("RAY_object_store_memory", str(args.object_store_gb * 1024**3))
    # 内存吃紧时 Ray 会提前杀 worker。放宽一点，让真正的分配失败自己暴露出来，
    # 而不是被 Ray 的保护机制提前打断（真 OOM 有堆栈，被 Ray 杀掉只有一句话）
    env.setdefault("RAY_memory_usage_threshold", "0.97")
    # ★★★ 多卡必须设 NCCL_CUMEM_ENABLE=0（2026-08-13 在 4×5090 + Ray 上实测出来的）
    #
    # 症状：FSDP 初始化的第一次参数广播直接炸
    #     transport/shm.cc:590 NCCL WARN Cuda failure 217
    #     'peer access is not supported between these two devices'
    #
    # ⚠️ **根因不是"P2P 关了"那么简单，一开始我就是这么以为的，被自己的测量推翻**：
    # 裸 torchrun 跑 NCCL all_reduce **默认配置就能通**（每个进程看得见全部 4 张卡）。
    # 真正的触发条件是 **Ray 给每个 worker 只设一张卡的 CUDA_VISIBLE_DEVICES** ——
    # 进程根本看不见对端设备，NCCL 的 SHM 传输要用的 CUDA IPC 就开不了。
    #
    # 四种配置在「每进程只看得到自己那张卡」下的实测：
    #     默认                     ❌ 217
    #     NCCL_P2P_DISABLE=1       ❌ 217   ← 光关 P2P 没用，别照抄网上的偏方
    #     NCCL_SHM_DISABLE=1       ✅       退回 socket 传输
    #     NCCL_CUMEM_ENABLE=0      ✅       保留 SHM 传输，走老式 cudaMalloc 的 IPC
    #
    # 选后者的依据是带宽（2 卡 all-reduce bus bandwidth，实测）：
    #     SHM_DISABLE=1     1MB 1.11 GB/s   256MB 2.09 GB/s
    #     CUMEM_ENABLE=0    1MB 5.20 GB/s   256MB 6.44 GB/s   ← 快 3 倍
    #
    # ⚠️ 6.4 GB/s 是这台机器**卡间通信的天花板**（没有 NVLink、没有 P2P、经主机中转），
    #    记在账上：TP 每层两次 all-reduce 在这个带宽下大概率是负收益，而 LoRA 的
    #    DDP 只同步 132 MB ≈ 20 ms。这是通信画像的第一个实测点
    #    （`docs/distributed-training-design-v0.1.md` §2 / D7）。
    if args.trainer_gpus + args.rollout_gpus > 1:
        env.setdefault("NCCL_CUMEM_ENABLE", "0")
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
