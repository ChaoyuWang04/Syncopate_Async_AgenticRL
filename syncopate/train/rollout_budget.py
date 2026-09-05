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
# ★ 当前 v16（v15 契约）默认必须是 True：SFT 数据已经带逐轮 think 形状，RL 也从
#   同一契约模块取开关，并通过共享的增量 rollout 路径消费。显式关掉只用于旧 v14
#   重放或预注册 A/B，不能静默改变正式训练形状。
#
# ★ 预算为什么是 8192：rollout 循环是**增量拼 token、从不重渲染**（rollout_loop.py:239）
#   ⇒ 历史轮的 <think> 会留在上下文里并计入 response 预算。
#   上界估算：中位 4 个 assistant 轮 × Qwen3 单轮思考 ~0.5–1.5k + 动作/终答 ~0.3k
#   + 工具返回 ~0.6k ⇒ ~7k，取 8192 含余量。
#   ⚠️ think-on 跑完必查 truncation_reason="tokens" 的比例 —— 截断的思考连答案都
#   没有（E20 §7.12 那类翻案的同族），预算不够宁可加大重跑。
# ★ v15 起 think-on 是**契约默认**（`25 §3.2` 修法 A）：think-off 时模板会把 think 块
#   预先关闭（`…assistant\n<think>\n\n</think>\n\n`）⇒ 模型**结构上没有思考的机会**，
#   两个方向都没有梯度。这不是"做得差"，是"不可能"。
#   ⚠️ 显式设了 SYNCOPATE_THINK 就以它为准（v14 历史重放要能关掉）。
_THINK_ENV = os.environ.get("SYNCOPATE_THINK")
if _THINK_ENV is None:
    from syncopate.core.contract import IS_V15
    THINK_ON = IS_V15
else:
    THINK_ON = _THINK_ENV == "1"

def assistant_turn_budget(max_steps: int) -> int:
    """一条轨迹允许的 assistant 轮数上限 —— **从契约派生，不许各处硬编码**。

    v15 比 v14 恰好多一步：机器字段走 `session.report` **单独一步**（`25 §3.1`，
    理由见 sft_replay._v15_tail —— 与信令挤一步会被判混合形态）。

    ⛔ 2026-08-30 实案：不加这一格，`case.max_steps` 用满的 case 在 v15 下 gold 回放
      直接被截断（SIG_LOW_001），而截掉的正是终答那一段。
      这是 P3-1「v13 有 131/503 条因轮数上限被无声掐断」的**同族第二次**。
    ⇒ 一般化（守则⑨）：预算必须从契约派生；契约变了，预算自动跟着变，
      而不是等某个调用方记得 +1。
    """
    from syncopate.core.contract import IS_V15
    return max_steps + (1 if IS_V15 else 0)


# chat 模板的 enable_thinking —— rollout_loop.CHAT_TEMPLATE_KWARGS 从这里取
ENABLE_THINKING = THINK_ON

# 首轮 prompt 超过这个长度就左截断（不是"整条轨迹的上限"）
#
# ⛔ 2026-08-30（Chaoyu 裁定 5120→5760）：v15 的 R2 数据实测 prompt max **5430**，
#   65 行（critical_args 桶）撑破 5120。根因是叠加的：v13 case 本身长 + 信令族 428 tok。
#   ★ 处置顺序按 `25 §6②`：先看能不能精简 schema —— 实测**整个信令块才 428**，
#     而满足"余量 ≥300"需要省 610 ⇒ 精简到底也不够，必须抬上限。
#   ★ 抬到 5760 的依据是**真实约束**：服务侧 max_model_len 14336。
#     5760 + 8192 = 13952 ≤ 14336（余量 384）⇒ **服务侧不用改**。
#   ⚠️ 余量仍然薄（数据侧 5760−5430=330）：R2 之后任何加长 system/工具描述的改动
#     都必须重量一次 prompt max —— 判据见 syncopate/pipeline/prompt_budget_gate.py 的 --prompt-budget。
# ★ 2026-09-02（Chaoyu 裁定：不爆显存就抬到线上真实形状）：5760 → **9216**。
#   依据（26 §W2⑤ 实测）：全量 34 工具菜单 = 线上形状（守则⑮ #6），工具描述修剪后最长 prompt 仍 7167，
#   多轮行再加最近 6 轮历史（每轮 ≤400 tok）⇒ 9216。9216 + 8192 = 17408 ⇒ 服务/RL max_model_len 18432
#   （logs/runtime/start_vllm.sh · scripts/v16/exam_chain.sh · decider.RUNTIME_MAX_MODEL_LEN 同步改）。
#   显存：Qwen3-4B 每 token KV ≈144 KB ⇒ 18432 一条 ≈2.65 GB；R6 起跑前按 25 §R6 V0⒠ 重测并发。
# ★ 2026-09-04（Chaoyu 裁定：上限是按 5090 显存定的数字，B200 上只要不爆显存就抬；教师 CoT 必须完整）：9216 → **12288**，
#   think-on response 8192 → **12288** ⇒ max_model_len 24576（服务/RL/eval 全部派生；stack_probe.SERVE_MAX_MODEL_LEN 有相等断言）。
#   依据：学生 Qwen3.6-35B-A3B 30/40 层是线性注意力（KV 不随长度涨），10 层全注意力 GQA ⇒ 每 token KV 极小；
#   SFT 实测 17408 上限下峰值 74 GB / 183 GB；RL 冒烟 response 均值 1.3k token，抬上限只是给尾巴留余量。
MAX_PROMPT_LENGTH = 12288

# 一条轨迹里**模型生成 + 工具返回**加起来的 token 预算
MAX_RESPONSE_LENGTH = 12288 if THINK_ON else 2048

if THINK_ON:
    # 判据行：think 模式必须显式可见，静默生效 = 下一个「机制在但没接上」
    _src = "SYNCOPATE_THINK=1" if _THINK_ENV == "1" else "契约 v15 默认"
    print(f"[think-mode] {_src} ⇒ enable_thinking=True · "
          f"MAX_RESPONSE_LENGTH 2048→{MAX_RESPONSE_LENGTH}"
          f"（v15 训练路径放行；v14 仍拦，见 launch_rl）", flush=True)

# ⇒ v16 think-on 下 max_model_len = 12288 + 12288 = 24576；训练、评测与服务均从
#   这里派生，不能在入口另抄一份。

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
