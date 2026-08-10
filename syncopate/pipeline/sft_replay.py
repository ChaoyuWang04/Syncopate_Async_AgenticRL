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

from syncopate.core.parsing import render_final_answer, render_tool_call
from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolRegistry
from syncopate.train.rollout_loop import RolloutConfig, run_rollout


def gold_script(bundle: CaseBundle, behavior: str | None = None) -> list[str]:
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
    steps.append(render_final_answer(resolved, bundle.gold.final_answer))
    return steps


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
    return SFTSample(
        case_id=bundle.case_id,
        input_ids=output.prompt_ids + output.response_ids,
        loss_mask=[0] * len(output.prompt_ids) + output.response_mask,
        prompt_length=len(output.prompt_ids),
    )
