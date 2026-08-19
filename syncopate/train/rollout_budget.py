"""rollout 的**契约参数**（长度预算 + 采样参数）—— 训练 / 评测 / 部署共用这一份。

★ 为什么单独一个模块（2026-08-18）

此前这三个数散在三处，各写各的：

    训练（launch_rl / 各 batch 脚本）   --max-prompt-length 3584  --max-response-length 1536
    评测（eval_local.py:528）           MAX_PROMPT_LENGTH 5120    **硬编码 2048**
    离线合成（staleness.py:123）        同上，也是硬编码 2048

⇒ **评测的预算比训练宽 43% / 33%**，后果是：
  ① 同一条轨迹在评测里跑得完、在训练里可能被截断 ⇒ 两边跑在不同的输入分布上；
  ② 训练的 reward 分布系统性偏向"更短的轨迹"，而我们拿评测分数去判训练有没有用；
  ③ 评测量到的截断率**不能直接套到训练** —— 两个数不同源。

这是记录在案的坑 #3（「prompt 被截断 ⇒ 训练和评测跑在不同输入分布上」）的同族，
只是这次断在 response 侧，而且是**两个入口各写各的常量**。

⇒ 修法与 `ckpt_guards` 同构：**提成一份值，所有路径都从这里取**。
⇒ 取值方向由 Chaoyu 定：**取宽的那一档**（放宽只会让原本被截断的轨迹跑完，
   不会新增截断；反过来收窄会改变已有基线）。

⚠️ 改这三个数会改变 `max_model_len`（`launch_rl` 自动算成 prompt+response）
   ⇒ 会动 vLLM 的 KV cache 布局。改完必须跑一次冒烟（队列 20 号文档的 G-3）。
"""

from __future__ import annotations

import os

# ── thinking 开关（E27 探针，2026-08-19）──────────────────────────────────
#
# ★ 为什么开关放在**契约模块**：开 thinking 同时动两个契约量（模板 kwarg + response
#   预算），按「这个值应该和那边一致 ⇒ 这里根本不该有第二份」的纪律，两处都从这里取。
#
# ★ 默认必须是 False（= 现行为，逐字节不变）：全管线共用同一份脚本，开关只给探针用。
#   ⛔ 训练路径禁止开（launch_rl 启动即拦）：SFT 是 think-off 模板练的，
#   开 think 会让「增量拼接 vs 整段渲染」逐 token 不相等（rollout_loop.py:50 有全文，
#   tests/train 有测试守着）⇒ think-on 训练要等带思考的 SFT 数据，不是拨个开关的事。
#
# ★ 预算为什么是 8192：rollout 循环是**增量拼 token、从不重渲染**（rollout_loop.py:239）
#   ⇒ 历史轮的 <think> 会留在上下文里并计入 response 预算。
#   上界估算：中位 4 个 assistant 轮 × Qwen3 单轮思考 ~0.5–1.5k + 动作/终答 ~0.3k
#   + 工具返回 ~0.6k ⇒ ~7k，取 8192 含余量。
#   ⚠️ think-on 跑完必查 truncation_reason="tokens" 的比例 —— 截断的思考连答案都
#   没有（E20 §7.12 那类翻案的同族），预算不够宁可加大重跑。
THINK_ON = os.environ.get("SYNCOPATE_THINK", "0") == "1"

# chat 模板的 enable_thinking —— rollout_loop.CHAT_TEMPLATE_KWARGS 从这里取
ENABLE_THINKING = THINK_ON

# 首轮 prompt 超过这个长度就左截断（不是"整条轨迹的上限"）
MAX_PROMPT_LENGTH = 5120

# 一条轨迹里**模型生成 + 工具返回**加起来的 token 预算
MAX_RESPONSE_LENGTH = 8192 if THINK_ON else 2048

if THINK_ON:
    # 判据行：think 模式必须显式可见，静默生效 = 下一个「机制在但没接上」
    print(f"[think-mode] SYNCOPATE_THINK=1 ⇒ enable_thinking=True · "
          f"MAX_RESPONSE_LENGTH 2048→{MAX_RESPONSE_LENGTH}（仅评测探针 E27；训练路径会拦）",
          flush=True)

# ⇒ launch_rl 会算 max_model_len = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = 7168
#   （`launch_rl.py:31` 的注释里算过：7168 × 144 KB ≈ 0.98 GB KV cache，可接受）

# ── 采样参数 ───────────────────────────────────────────────────────────────
#
# ★★ 2026-08-18：此前训练与评测**也不一致**（同 §长度预算，是同一个形状的第三例）：
#
#       评测 eval_local   temperature 1.0 · top_p **0.95** · top_k **20**
#       训练 verl 默认    temperature 1.0 · top_p **1.0**  · top_k **-1**（不截）
#
#   `eval_local` 的注释写着"采样参数逐项对齐 **HF 路径**" ——
#   它把 eval-vLLM 对齐到了 eval-HF，**但没有人把 eval 对齐到训练**。
#
#   后果实测：一步多调用的违规率 训练 18.8% / 评测 **0%**
#   ⇒ 那截被截掉的尾巴几乎全是格式违规 ⇒ **评测分数系统性高估了训练时的策略**。
#
# ★ 对齐方向（infra 2026-08-18 定，主线核过部署侧无约束）：**评测跟训练**，不是反过来。
#   理由：截尾会让「实际采样的分布」≠「算 logprob 的分布」
#   ⇒ 重要性采样的分母不再是真实的行为策略 ⇒ 直接污染 TIS / ESS（E20/E23 那条线）。
#   ⇒ **宁可让评测看见训练真实会产生的那条尾巴**，也不要为了评测好看去动训练分布。
#
# ⚠️ **唯一会翻案的条件**：部署侧硬性要求某组采样参数。
#   2026-08-18 核过：`syncopate/runtime/` 里的 `top_k` 全是**检索**的，
#   M9 至今没有接模型调用（`model_calls` 表建了但没有写入路径，见 `18 §6`）
#   ⇒ **当前无约束**。⚠️ 但这是"现在没有"，不是"永远没有" ——
#   **部署一旦接上模型调用，就以部署为准，三者再对齐一次。**
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = -1          # -1 = 不截断

__all__ = [
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
    "SAMPLING_TEMPERATURE", "SAMPLING_TOP_P", "SAMPLING_TOP_K",
    "THINK_ON", "ENABLE_THINKING",
]
