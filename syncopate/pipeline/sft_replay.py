"""SFT 样本 = 用同一个 rollout 循环回放 gold。

★ 为什么不用 `apply_chat_template(整段对话)` 构造 SFT 数据

Qwen3 的 chat template 对 assistant 轮的处理是不对称的：**只给最后一个 assistant 轮
加空的 `<think>\\n\\n</think>\\n\\n`，历史轮不加**（历史推理会被剥掉）。
而增量拼接时每一轮都是「当前最后一轮」，所以每轮都会带上这个前缀。

于是「整段渲染」和「增量拼接」**天生逐 token 不相等**，无论 enable_thinking 设什么。
这不是参数问题，是结构问题。

老师包踩的就是这个坑的变体（sft-truth-report T10）：SFT 侧硬编码
enable_thinking=False、RL 侧从不传，两阶段分布不一致且**没有任何报错**。

我们的解法是**只保留一条代码路径**：SFT 数据由 `run_rollout` 回放 gold 产出，
和 RL 跑出来的序列同构是构造保证的，不需要靠测试去碰运气验证。
代价是不能直接用 verl 的 MultiTurnSFTDataset（它自己做 chat template），
所以我们产出**预分词**的样本，配一个最小 SFT 训练脚本——单卡 LoRA 场景下这更简单也更可控。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from syncopate.core.contract import IS_V15
from syncopate.core.parsing import render_final_answer, render_tool_call
from syncopate.core.parsing_v15 import render_report, render_signal
from syncopate.train.rollout_budget import ENABLE_THINKING
from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolRegistry
from syncopate.train.rollout_loop import RolloutConfig, run_rollout


def gold_script(bundle: CaseBundle, behavior: str | None = None,
                thinking: dict[int, str] | None = None) -> list[str]:
    """把 gold 轨迹翻译成「模型每一步该输出什么文本」。

    这个函数是 SFT 和测试共用的——保证「SFT 教的格式」和「RL 解析的格式」
    是同一个东西，不会各自漂移。

    ★ behavior 必须来自 `bundle.verifier.expected_behavior`。

    早期版本这里写的是 `behavior: str = "tool_call"` 且调用方从不传值，
    结果 **clarify / reject 类 case 的监督目标里写的是 `"behavior": "tool_call"`**
    ——我们在教模型输出错误的标签。

    症状极具迷惑性：分组 val_loss 降到 **0.0000**（它确实完美学会了那个错误目标），
    但自回归生成时 behavior 恒为 tool_call，behavior_mismatch 命中率 100%。
    当时误判成"token 失衡把边界能力挤没了"，做了加权采样——那只是让它把错的学得更牢。

    教训：**loss 降到 0 只说明学到了标签，不说明标签是对的。**
    tests/train 里有一条测试专门比对监督目标和 expected_behavior。
    """
    assert bundle.gold is not None, f"{bundle.case_id} 没有 gold"
    resolved = behavior or bundle.verifier.expected_behavior
    steps = [render_tool_call(a["tool"], a.get("arguments", {})) for a in bundle.gold.actions]
    if IS_V15:
        steps.extend(_v15_tail(bundle, resolved))
    else:
        steps.append(render_final_answer(resolved, bundle.gold.final_answer))
    if ENABLE_THINKING:
        steps = attach_think(steps, thinking or {})
    return steps


def _machine_fields(bundle: CaseBundle) -> dict:
    """判分器真正会核对的字段（= 必填字段里非「只查存在」的那些）。

    ★ 这些字段**一律走 session.report**，不管本轮的行为是什么 —— 包括 defer/clarify/
      reject。理由（R1 门槛⑤ 实测逼出来的）：
        · 信令 schema 装不下它们。`defer` 要报 `data_maturity`，但
          `session.defer{reason, recheck_after_days}` 里没有这一格；
          而 R0 双臂数据是按现 schema 冻结的，改 schema 就得重建 R0。
        · 硬做字段名映射（missing_fields→missing_field、reason_code→reject_reason）
          等于在契约里再加一层翻译表 —— 多一处会漂的副本。
      ⇒ **分工写死：session.* 管编排语义（挂起/等补充/终止），session.report 管判分字段。**
      两者在 `recheck_after_days` 上有意重叠：编排要它挂复查、判分要它核数值，
      各取各的，不是副本。
    """
    fa = dict(bundle.gold.final_answer or {})
    return {f.key: fa[f.key] for f in bundle.verifier.required_answer_fields
            if f.key in fa and f.value_source != "any"}


def _v15_tail(bundle: CaseBundle, behavior: str) -> list[str]:
    """v15 终答段：机器字段 → session.report（单独一步）；行为 → 信令调用或纯文本。

    ⚠️ report 之所以**单独一步**（而不是和信令/收尾话挤一步）：解析器把
    「信令 + 别的工具同一步」判成混合形态（`25 §6③`），把「有 tool_call」读成轨迹继续。
    代价（多一步）已在判分侧抵消 —— `trajectory.num_business_steps` 排除 session.*
    （R1 门槛⑤ 实测：不排除的话 120/120 条 gold 的 efficiency 全变）。
    """
    fa = dict(bundle.gold.final_answer or {})
    machine = _machine_fields(bundle)
    head = [render_report(machine)] if machine else []
    if behavior == "defer":
        return head + [render_signal("session.defer", {
            "reason": str(fa.get("defer_reason") or "数据还不足以支撑结论"),
            "recheck_after_days": int(fa.get("recheck_after_days") or 5)})]
    if behavior == "clarify":
        mf = fa.get("missing_field") or "campaign_id"
        return head + [render_signal("session.clarify",
                                     {"question": f"请补充 {mf} 后我再继续。",
                                      "missing_fields": [mf]})]
    if behavior == "reject":
        rr = {"unauthorized": "unauthorized", "policy": "policy"}.get(
            fa.get("reject_reason"), "out_of_scope")
        expl = {"unauthorized": "该操作超出当前授权范围，无法执行。",
                "out_of_scope": "这超出投放助手的职责范围，无法处理。",
                "policy": "该请求与平台政策冲突，无法执行。"}[rr]
        return head + [render_signal("session.reject",
                                     {"reason_code": rr, "explanation": expl})]
    # tool_call / answer：机器字段已在 head 的 report 里，这里只剩一句人话。
    # ★ 人话字段（value_source=="any"，实测 60/4100 全是 CHAT 的 reply）**不进 report** ——
    #   否则等于逼模型把同一句人话在机器通道里再抄一遍（「summary 污染」同族）。
    return head + [str(fa.get("reply") or "已经按上面的结果处理完了。")]


EMPTY_THINK = "<think>\n\n</think>\n\n"


def attach_think(steps: list[str], thinking: dict[int, str]) -> list[str]:
    """★ think-on 下**每个** assistant 轮都要显式写出 think 段（`25 §3.2` 修法 B）。

    ⚠️ 只做 A（切 think-on）不做 B 的后果是实测过的：监督段直接以 <tool_call> 开头、
    一个 think 块都不出现 ⇒ 变成**主动训练"永不思考"**，比 think-off 更糟。

    `thinking` = {步号: 推理文本}；没给的步填**显式空块**（= 教"这步不用想"）。
    空块与非空块的比例就是 N3「按需思考」的旋钮。
    """
    out = []
    for i, body in enumerate(steps):
        content = (thinking.get(i) or "").strip()
        prefix = f"<think>\n{content}\n</think>\n\n" if content else EMPTY_THINK
        out.append(prefix + body)
    return out


class _ScriptedEngine:
    """按剧本吐 token 的假引擎。回放 gold 时代替真模型。"""

    def __init__(self, tokenizer: Any, script: list[str]) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        if not self.script:
            return []
        return self.tokenizer.encode(self.script.pop(0), add_special_tokens=False)


@dataclass
class SFTSample:
    """一条预分词的 SFT 样本。

    loss_mask 直接复用 rollout 的 response_mask：1=模型该学会生成的 token，
    0=环境插入的工具返回。prompt 段全部为 0（不监督 system/user）。
    """

    case_id: str
    input_ids: list[int]
    loss_mask: list[int]
    prompt_length: int

    @property
    def total_length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(self.loss_mask)


async def build_sft_sample(
    bundle: CaseBundle,
    *,
    tokenizer: Any,
    registry: ToolRegistry,
    config: RolloutConfig | None = None,
) -> SFTSample:
    """回放 gold，产出一条 SFT 样本。

    副作用是顺带验证了 gold 走得通——工具报错会体现在 observation 里，
    进而污染后续 token。所以构造 SFT 数据这一步本身就是一次 gold 健全性检查。
    """
    output = await run_rollout(
        bundle, registry=registry, tokenizer=tokenizer,
        generate=_ScriptedEngine(tokenizer, gold_script(bundle)),
        config=config or RolloutConfig(),
        rollout_id="gold", run_id="sft",
    )
    # ★ gold 回放**不许截断**——被截掉的一定是轨迹结尾（终答那段），
    #   而那正是最该学的。v13 实测 131/503 条因轮数上限用了默认 8（< case.max_steps）
    #   被无声掐断，最终结论从没进过训练数据。判据写在发生点，不靠调用方记得检查。
    if output.trajectory.truncated:
        raise ValueError(
            f"{bundle.case_id}: gold 回放被截断（原因 {output.trajectory.truncation_reason}，"
            f"需要 {len(bundle.gold.actions) + 1} 个 assistant 轮，"
            f"上限 {(config or RolloutConfig()).max_assistant_turns}）——"
            "SFT 样本必须是完整轨迹；轮数上限应取 case.max_steps（见 build_dataset）"
        )
    return SFTSample(
        case_id=bundle.case_id,
        input_ids=output.prompt_ids + output.response_ids,
        loss_mask=[0] * len(output.prompt_ids) + output.response_mask,
        prompt_length=len(output.prompt_ids),
    )
