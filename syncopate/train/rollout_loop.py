"""框架无关的 agent 主循环。

verl / 裸 vLLM / 单元测试的假引擎都能驱动它——它只要求对方提供一个
`async generate(prompt_ids, sampling_params) -> list[int]`。

这样分层的三个好处：
  1. 不装 verl 也能跑通整条链路（单元测试用假引擎，秒级）
  2. 调试时可以直接打一个 vLLM server，不用起 Ray 集群
  3. 将来换框架（slime / AReaL）只动薄薄一层适配器

产出的 token 序列直接对齐 verl 的 `AgentLoopOutput` 契约：

    response_ids  = 模型生成 token + 工具 observation token，按真实发生顺序串联
    response_mask = 与之等长，1=模型自己生成（算梯度），0=环境插入（不算梯度）

★ mask 这件事是多轮 RL 最容易错的地方：把工具返回的 token 也算进梯度，
等于在训练模型「复述环境给它的东西」，会严重污染训练信号。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from syncopate.core.parsing import ParsedStep, parse_step
from syncopate.core.sandbox import Sandbox
from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolContext, ToolRegistry
from syncopate.core.trajectory import Action, Observation, Trajectory
from syncopate.prompts import load_prompt, prompt_hash, render_prompt


# Qwen3 的轮次结束符。模型通常自己会生成它并被 vLLM 当停止符吞掉，
# 所以增量拼接时要显式补回来（见 run_rollout 里的说明）。
ASSISTANT_TURN_END = "<|im_end|>"

# ★★★ prompt 预算的**唯一来源**。训练侧和评测侧必须用同一个值。
#
# 实测真实 prompt 是 4170–4210 token（system 规则书 + 22 个工具的 schema + 任务）。
# 之前 eval 硬编码 4096、RL 侧用 `max_model_len // 2` 算出 2304 ——
# 两边都在截，而且截得不一样，等于训练和评测跑在两个不同的输入分布上。
# 左截断砍掉的是 system 规则书的**开头**（工具规则 + clarify/reject/defer 枚举），
# 实测让 CLAR/REJ 的 reward 从 0.9 掉到恒等于 0。
# ★ 2026-08-18：这个值挪进了 `rollout_budget.py`，因为**训练和评测必须共用同一份**
#   （此前评测硬编码 2048 而训练传 1536 —— 两边跑在不同的输入分布上）。
#   这里保留同名 re-export，下游的 import 一个字都不用改。
from syncopate.train.rollout_budget import (  # noqa: E402,F401
    ENABLE_THINKING, MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH,
)

# ★ 默认显式关掉 thinking，SFT / RL / gold 回放三处必须完全一致。
#
# 老师包里 SFT 侧硬编码 enable_thinking=False，RL 侧**从不传**（走模板默认 = 允许
# thinking），两阶段不一致（sft-truth-report T10）。这不只是"浪费 token"的问题：
#
#   enable_thinking=False —— 空的 `<think>\n\n</think>\n\n` 出现在**生成提示里**
#                            （模型看到的前缀），和整段渲染的 assistant 段对得上
#   enable_thinking=True  —— 生成提示里没有，但整段渲染会给最后一个 assistant 轮
#                            补一个，于是增量拼接和整段渲染**逐 token 不相等**
#
# 也就是说，开着 thinking 的话 SFT 学到的序列和 RL 跑出来的序列天生对不齐，
# 而且不会有任何报错。tests/train 里有一条测试专门守着这件事。
#
# ★ 2026-08-19（E27）：值改从契约模块取 —— `SYNCOPATE_THINK=1` 时为 True，
#   **仅供评测探针**（预算随之 2048→8192，见 rollout_budget.py 的开关注释）；
#   默认（不设环境变量）与旧行为逐字节相同；训练路径由 launch_rl 启动即拦。
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": ENABLE_THINKING}


@dataclass
class Generation:
    """引擎的一次返回。log_probs 可选——没有它 TIS 诊断就是空的。"""

    token_ids: list[int]
    log_probs: list[float] | None = None


def _as_generation(result: Any) -> Generation:
    """兼容只返回 token 列表的简单引擎（单元测试的假引擎就是这种）。"""
    if isinstance(result, Generation):
        return result
    return Generation(token_ids=list(result))


class GenerateFn(Protocol):
    """引擎接口：给一串 prompt token，返回生成的 token（可带 logprob）。"""

    def __call__(self, prompt_ids: list[int],
                 sampling_params: dict[str, Any]) -> Awaitable[Any]: ...


# 再发一次生成请求所需的最小剩余预算。低于它就收工：
# ①<1 时引擎直接报错并杀死整个任务（实测，见循环里的注释）；
# ②只剩几十个 token 也吐不出完整的 tool_call/终答，白烧一次往返。
MIN_GENERATION_HEADROOM = 64


@dataclass
class RolloutConfig:
    max_assistant_turns: int = 8
    max_prompt_length: int = 8192
    max_response_length: int = 4096
    # 工具 observation 单条最长 token 数，超了截断——防止一个大 JSON 把上下文吃光
    max_observation_tokens: int = 1024


@dataclass
class RolloutOutput:
    """一条 rollout 的全部产物。"""

    trajectory: Trajectory
    sandbox: Sandbox
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    # rollout 侧（vLLM）算出的逐 token logprob。训练侧会重算一份，
    # 两者之比就是 TIS 的重要性采样权重——**没有它整条 rollout_corr 诊断链是空的**。
    response_logprobs: list[float]
    num_turns: int
    # token -> step 的映射表。步级信用分配要用它，落盘成本几乎为零。
    token_trace: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def build_messages(bundle: CaseBundle, tool_menu_names: list[str] | None) -> list[dict[str, str]]:
    """首轮 messages：system 规则书 + 本 case 的任务描述。

    SFT 和 RL 必须走同一个函数——两阶段 prompt 不一致是最难查的一类 bug
    （老师包 T10：`tool_schema_hash` 算了但从没比对过）。
    """
    # ★★★ context 必须**按 key 排序**渲染（模板里的 `| dictsort`）。
    #
    # 实测踩到过：内存里 context 的插入顺序是 account_id → requested_budget → campaign_id，
    # 而 CaseBundle.write 落盘时 sort_keys=True，读回来变成字母序。
    # 于是**同一条 case，生成时的 prompt 和训练时的 prompt 不一样**：
    #   · 生成器的内容去重守卫跑在"内存顺序"空间
    #   · split 的泄漏检测和实际训练跑在"磁盘顺序"空间
    # 两个空间不一致，直接漏掉了一对内容完全相同的 case（BUD_0108 / FAIL_0108）。
    #
    # 更根本的是：prompt 的内容不该取决于 dict 的插入顺序 ——
    # 那意味着换个构造写法，模型看到的题面就变了。
    system_text = load_prompt("system.txt")
    user_text = render_prompt("step_user.txt", {
        # ★ M2：把「今天是几号」给模型。
        #
        # 在此之前模型**根本不知道今天几号** —— 只能靠调 calendar.get_seasonal_context
        # 顺带读到，而那个工具只在 CRE 的菜单里。于是「这份安全线过期了没有」这道题
        # 对它是不公平的：没有比较的基准。
        #
        # ⚠️ 这和当初砍掉 metrics.get_freshness 的 `as_of` 参数不矛盾。
        # 那条禁的是「模型**自己声明**今天是哪天」（它可以填个假日期，把
        # premature_decision_cap 整个架空）；**读到**今天是哪天没有任何问题。
        # 区别是**只读**还是**可写**。真实世界里 agent 的运行时当然知道系统时间。
        #
        # 放在模板变量而不是塞进 context：context 参与内容哈希（去重 / 泄漏检测），
        # 往里加 key 会让所有已有 case 的指纹变化，两件事就纠缠在一起了。
        "reference_now": bundle.env.reference_now,
        "context": bundle.case.context,
        "user_message": bundle.case.user_message,
        "answer_fields": bundle.verifier.required_answer_fields,
    })
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def observation_message(tool_name: str, observation: dict[str, Any]) -> dict[str, str]:
    """工具返回渲染成一条 tool message。Qwen3 的模板认 role="tool"。"""
    import json

    return {"role": "tool", "name": tool_name,
            "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)}


async def run_rollout(
    bundle: CaseBundle,
    *,
    registry: ToolRegistry,
    tokenizer: Any,
    generate: GenerateFn,
    config: RolloutConfig | None = None,
    sampling_params: dict[str, Any] | None = None,
    rollout_id: str = "r0",
    run_id: str = "local",
) -> RolloutOutput:
    """跑一条完整 rollout：生成 → 解析 → 执行工具 → 回灌 observation → 循环。"""
    config = config or RolloutConfig()
    sampling_params = dict(sampling_params or {})
    tool_names = bundle.case.tool_menu  # None = 全量菜单
    tools = registry.menu(tool_names)

    namespace_id = f"{run_id}:{bundle.case_id}:{rollout_id}"
    sandbox = Sandbox(bundle.env, namespace_id=namespace_id)
    trajectory = Trajectory(case_id=bundle.case_id, rollout_id=rollout_id, namespace_id=namespace_id)

    messages = build_messages(bundle, tool_names)
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=True, **CHAT_TEMPLATE_KWARGS,
    )
    prompt_truncated = len(prompt_ids) - config.max_prompt_length
    if prompt_truncated > 0:
        # 左截断：砍掉的是 system 规则书的开头。
        # ⚠️ 这不是"侥幸幸存"——砍掉的正是输出格式和 behavior 枚举，
        # 模型会因此不知道 clarify/reject/defer 存在。必须当成事故指标盯着，
        # 不能静默发生（实测就是这么让整轮 RL 白跑的）。
        prompt_ids = prompt_ids[-config.max_prompt_length:]

    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    segments: list[dict[str, Any]] = []
    # 我们自己补的轮次结束符没有 rollout logprob，只能填占位值。
    # 这会给 TIS 带来一点系统性偏差，所以把数量记下来让它可量化。
    placeholder_logprobs = 0

    step = 0
    tool_errors = 0
    parse_errors = 0
    started = time.monotonic()
    generate_seconds = 0.0
    tool_seconds = 0.0

    while step < config.max_assistant_turns:
        # ★★ 预算耗尽就收工 —— 不能再发一次生成请求。
        #
        # 2026-08-13 实测炸过：response 预算用满（1536）时，送进引擎的上下文
        # 正好等于 max_model_len（3584+1536=5120），vLLM 直接抛
        # `Prompt length (5120) leaves no room to generate` 并**杀掉整个训练任务**。
        #
        # 为什么以前没炸而现在炸：v11 的 GEO(14 步)/ATTR(8-10 步) 轨迹变长了，
        # 才第一次把 1536 的 response 预算真正吃满。**这是长任务暴露的边界 bug，
        # 不是新引入的**——同一条循环在 eval 里不炸，只因为那边 max_model_len
        # 留了 4096 的多轮累积余量。
        #
        # 阈值不取 1 而取 64：只剩几个 token 时就算能发出去，也吐不出一个完整的
        # tool_call 或终答，白烧一次往返。截断由 trajectory.truncated 记账，
        # 和下面 `added == 0` 那条出口是同一套语义。
        if config.max_response_length - len(response_ids) < MIN_GENERATION_HEADROOM:
            trajectory.truncated = True
            trajectory.truncation_reason = "tokens"      # 生成之前预算就不够了
            break
        step += 1

        # ---- 1. 生成 ----
        gen_start = time.monotonic()
        generation = _as_generation(await generate(prompt_ids + response_ids, sampling_params))
        generate_seconds += time.monotonic() - gen_start
        new_ids = generation.token_ids
        new_logprobs = list(generation.log_probs or [])

        text = tokenizer.decode(new_ids, skip_special_tokens=False)

        # ★ 补齐 assistant 轮的结束符。
        # 这是增量拼 token 的经典坑：整段 apply_chat_template 会在 assistant 内容后面
        # 加 `<|im_end|>\n` 再接下一轮，而 vLLM 通常把停止符吞掉不返回。不补的话
        # SFT（整段渲染）和 RL（增量拼接）看到的 token 序列会差一小段，
        # 两阶段分布不一致——而且不会有任何报错，只会让指标莫名其妙地差。
        suffix_ids: list[int] = []
        if not text.rstrip().endswith(ASSISTANT_TURN_END):
            suffix_ids = tokenizer.encode(ASSISTANT_TURN_END + "\n", add_special_tokens=False)

        # ⚠️ 截断必须给结束符预留位置，否则补完会顶出预算。
        # 之前就是先截断再补，长度变成 max_response_length + 2，
        # verl 的 _postprocess 在 torch.cat 时直接报 size mismatch。
        budget = config.max_response_length - len(response_ids)
        kept = max(0, budget - len(suffix_ids))
        new_logprobs = new_logprobs[:kept]
        new_ids = new_ids[:kept] + suffix_ids
        if not new_ids:
            trajectory.truncated = True
            trajectory.truncation_reason = "tokens"      # 截到只剩结束符都放不下
            break

        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))     # 1 = 模型生成，算梯度
        # logprob 必须和 ids 等长：引擎没给的部分（含我们补的结束符）填 0.0 占位
        missing = len(new_ids) - len(new_logprobs)
        placeholder_logprobs += missing
        response_logprobs.extend(new_logprobs + [0.0] * missing)
        segments.append({"type": "assistant", "step": step, "token_count": len(new_ids), "mask": 1})
        parsed: ParsedStep = parse_step(text)

        # ---- 2. 终答：循环结束 ----
        if parsed.kind == "final":
            trajectory.behavior = parsed.behavior
            trajectory.final_answer = parsed.answer
            trajectory.final_text = text
            trajectory.parse_ok = True
            break

        # ---- 3. 解析失败：把错误喂回去让它自己修 ----
        if parsed.kind == "error":
            parse_errors += 1
            trajectory.parse_ok = False
            trajectory.final_text = text
            feedback = render_prompt("tool_error.txt", {"error": f"输出格式无法解析：{parsed.error}"})
            obs_ids = _append_message(
                tokenizer, {"role": "user", "content": feedback},
                response_ids, response_mask, response_logprobs, segments, step, config,
            )
            if obs_ids == 0:
                break
            continue

        # ---- 4. 执行工具 ----
        # ★★ 2026-08-18：一步多调用时，**只执行第一个**，其余当协议错误退回。
        #
        # 起因（实测 5280 条训练 rollout，见 `docs/syncopate/01 §P0-2`）：
        #   18.8% 的 rollout 一步发了多个 tool call，而 system.txt 第 8 行明确禁止；
        #   截尾采样（评测口径 top_p 0.95 / top_k 20）下这个数是 **0%**
        #   ⇒ 它是**采样尾巴**，不是模型不懂规矩。
        #
        # 为什么要拦在这里而不是只靠 cap 事后封顶：
        #   ① system.txt 自己说了危险在哪 ——「不要在没看到前一个 observation 的情况下
        #      就调用依赖它的工具」。**执行了才叫危险**，拦下来危害就没了。
        #   ② 事后封顶把整条轨迹打成 0 分，实测**吃掉 29% 的组内方差**
        #      —— GRPO 的梯度完全来自组内方差，等于近三成梯度在教"别进采样尾巴"。
        #
        # ⚠️ **仍然把没执行的那些记进 trajectory.actions**：
        #   `multi_tool_per_step_cap` 靠 `multi_tool_steps()` 按 step 计数来判，
        #   不记的话这条 cap 会直接失效 —— 那就是本项目第一失效形状（机制在但没接上）。
        # ⚠️ topology == "parallel" 的 case 本来就要求同一步发多个，不拦（同 cap 的豁免口径）。
        parallel_ok = bundle.case.metadata.topology == "parallel"
        for call_index, call in enumerate(parsed.tool_calls):
            tool_call_id = f"tc_{step}" if len(parsed.tool_calls) == 1 else f"tc_{step}_{call_index}"
            ctx = ToolContext(case=bundle.case, env=bundle.env, sandbox=sandbox,
                              step=step, tool_call_id=tool_call_id)

            if call_index > 0 and not parallel_ok:
                # 不执行，只把协议错误回灌 —— 模型能看见自己错在哪，下一步可以改
                result_ok, result_data, result_error = False, {}, (
                    "protocol_violation: 每步只能发起一个 tool call，"
                    f"本步的第 {call_index + 1} 个调用（{call['name']}）未执行。"
                    "请等上一个 observation 返回后再决定下一步。"
                )
            # 菜单外的工具直接拒绝——不能让模型靠调用隐藏工具绕过 tool_missing 类 case
            elif tool_names is not None and call["name"] not in tool_names:
                result_ok, result_data, result_error = False, {}, f"tool_not_available: {call['name']}"
            else:
                tool_start = time.monotonic()
                result = await registry.execute(call["name"], call["arguments"], ctx)
                tool_seconds += time.monotonic() - tool_start
                result_ok, result_data, result_error = result.ok, result.data, result.error

            trajectory.actions.append(
                Action(step=step, tool_call_id=tool_call_id, name=call["name"], arguments=call["arguments"])
            )
            trajectory.observations.append(
                Observation(tool_call_id=tool_call_id, tool=call["name"], ok=result_ok,
                            data=result_data, error=result_error)
            )
            tool_errors += (not result_ok)

            payload = result_data if result_ok else {"error": result_error}
            added = _append_message(
                tokenizer, observation_message(call["name"], payload),
                response_ids, response_mask, response_logprobs, segments, step, config,
            )
            if added == 0:
                trajectory.truncated = True
                # ★ 这一种**不是模型的锅**：工具返回太大塞不进剩余预算。
                #   混在 "tokens" 里会把"该截 observation"误判成"该加预算"。
                trajectory.truncation_reason = "observation"
                break
        else:
            continue
        break
    else:
        # while 正常走完 = 撞上 max_assistant_turns 还没给终答
        trajectory.truncated = True
        trajectory.truncation_reason = "turns"           # ★ 只有这一种才是真的"撞步数上限"

    # ★ 硬约束：这个长度是 verl 的批次契约，超一个 token 都会在 _postprocess 的
    # torch.cat 里炸（"Sizes of tensors must match"）。上面每处 append 都算过预算了，
    # 这里再兜一道底——这类越界只在真跑起来才暴露，代价太高。
    assert len(response_ids) == len(response_mask), "mask 和 ids 长度不一致"
    assert len(response_logprobs) == len(response_ids), "logprob 和 ids 长度不一致"
    assert len(response_ids) <= config.max_response_length, (
        f"response 超长: {len(response_ids)} > {config.max_response_length}"
    )

    return RolloutOutput(
        trajectory=trajectory,
        sandbox=sandbox,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        num_turns=step,
        token_trace={
            "segments": segments,
            "response_token_count": len(response_ids),
            "prompt_token_count": len(prompt_ids),
            "prompt_hash": prompt_hash(load_prompt("system.txt"), tools),
        },
        metrics={
            "num_steps": step,
            "tool_errors": tool_errors,
            "parse_errors": parse_errors,
            "truncated": trajectory.truncated,
            "truncation_reason": trajectory.truncation_reason,
            # >0 就是事故：system 规则书的开头被砍掉了
            "prompt_truncated_tokens": max(0, prompt_truncated),
            # ★ 把「生成耗时」和「工具耗时」分开记。长尾 case 的时间全在 tool_seconds 上，
            #   这两个数是后面做异步对照实验的基础观测量。
            "generate_seconds": round(generate_seconds, 4),
            "tool_seconds": round(tool_seconds, 4),
            "wall_seconds": round(time.monotonic() - started, 4),
            # TIS 偏差的可量化指标：占位 logprob 越多，重要性采样权重越不可信
            "placeholder_logprobs": placeholder_logprobs,
            "logprob_coverage": round(
                1.0 - placeholder_logprobs / max(1, sum(response_mask)), 4),
        },
    )


def _append_message(
    tokenizer: Any,
    message: dict[str, str],
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    segments: list[dict[str, Any]],
    step: int,
    config: RolloutConfig,
) -> int:
    """把一条环境消息（工具返回 / 报错反馈）追加进响应序列，mask 记 0。

    返回实际追加的 token 数；0 表示已经没有预算了。
    """
    ids: list[int] = tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, tokenize=True, **CHAT_TEMPLATE_KWARGS,
    )
    ids = ids[: config.max_observation_tokens]
    budget = config.max_response_length - len(response_ids)
    ids = ids[: max(0, budget)]
    if not ids:
        return 0
    response_ids.extend(ids)
    response_mask.extend([0] * len(ids))    # 0 = 环境插入，不算梯度
    # 环境 token 没有 rollout logprob；mask=0 会让它们在损失和 TIS 里都被剔除
    response_logprobs.extend([0.0] * len(ids))
    segments.append({"type": message["role"], "step": step, "token_count": len(ids), "mask": 0})
    return len(ids)
