"""verl AgentLoop 适配层。

**这个文件应该保持很薄。** 所有业务逻辑在 `rollout_loop.py` 里，那边不依赖 verl；
这里只做三件事：

    1. 把 verl 的 `server_manager.generate` 包装成核心循环要的 `generate(prompt_ids, params)`
    2. 从 parquet 的 `extra_info` 把四件套读回来
    3. 把 `RolloutOutput` 翻译成 `AgentLoopOutput`

将来换框架（slime / AReaL）只需要重写这一个文件。

★ 关于 verifier 的阻塞问题：老师的 adapter 在 `async def run()` 里**同步**调
`score_and_persist_rollout`（train/verl_agent_loop_adapter.py:288），一路下去是
`urllib.request.urlopen(timeout=120)`。那会卡死整个 AgentLoopWorker 的事件循环——
不是"评分排队"，是"评分把生成也堵住了"。

我们这里用 `ASYNC_VERIFIER` 开关把两种行为都保留下来，因为**这正是要测量的对象**：
同一批 case、同一个种子，只翻这个开关，量每 step 的墙钟差多少。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from syncopate.core.schemas import CaseBundle
from syncopate.core.verifier_engine import ScoreResult, score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.train.rollout_loop import Generation, RolloutConfig, RolloutOutput, run_rollout

# 环境变量开关：verifier 是否放到线程池里跑。
#   "1"（默认）—— 不阻塞事件循环，其它 rollout 继续生成
#   "0"         —— 复现老师那套的阻塞行为，作为对照组
ASYNC_VERIFIER = os.environ.get("SYNCOPATE_ASYNC_VERIFIER", "1") not in {"0", "false", "False"}

# ★★★ 长尾比例的旋钮。设计文档 §35.4：长尾比例是异步 vs 同步实验的**核心自变量**，
# 必须能调。
#
# ⚠️ 在此之前这里从没设过 latency_scale，于是 RL 用的是 registry 默认的 1.0 ——
# `creative.poll_review` 会**真的 sleep 480 秒**。RL 池里 LONG 占 9.4%，
# train_batch_size=2 时约 18% 的 step 会撞上，每撞一次这一步就是 8 分钟起。
# 冒烟阶段跑 10 步，光睡觉就是十几分钟；正式跑 100 步是两个多小时。
#
#   1.0   真实世界（正式做异步对照实验时用）
#   0.01  480 秒 → 4.8 秒，保留长尾的**相对结构**，成本降两个数量级
#   0.0   完全没有长尾（测纯算法效果时用，但这会把第二目标的自变量抹掉）
LATENCY_SCALE = float(os.environ.get("SYNCOPATE_LATENCY_SCALE", "1.0"))

# cap 名单在 import 时固定下来，保证每条 rollout 回给 verl 的 key 集合完全一致——
# verl 用第一条样本的 keys 去 stack 整个 batch，key 集合不齐会在 collate 时炸。
CAP_NAMES: tuple[str, ...] = tuple(build_domain().caps.names())

# ★ 下发侧的记账：每条 rollout **跑完就追加一行**，不管它最后有没有进训练。
#
# 这是「分布漂移」那把尺子的另一半：
#   训练分布  ← trainer.rollout_data_dir 的每步 dump（**实际进入训练**的）
#   下发分布  ← 这个文件（**跑完过**的）
#   两者的差 = 漂移
#
# sync 下有 barrier，两者恒等，天然是对照组；async 下短任务先回、长任务被
# partial_rollout 切断，训练分布会系统性偏向短任务——而长链正是 agentic 的核心。
# 没有这个文件，漂移就只能靠推理，推理不出数值。
DISPATCH_LOG = os.environ.get("SYNCOPATE_DISPATCH_LOG")
_dispatch_lock = asyncio.Lock()


async def record_dispatch(bundle: CaseBundle, output: RolloutOutput, reward: float) -> None:
    if not DISPATCH_LOG:
        return
    line = json.dumps({
        "case_id": bundle.case_id,
        "num_steps": output.metrics["num_steps"],
        "wall_seconds": output.metrics["wall_seconds"],
        "truncated": int(output.metrics["truncated"]),
        "reward": reward,
    }, ensure_ascii=False)
    async with _dispatch_lock:      # 多条 rollout 并发跑完，追加要串行化
        await asyncio.to_thread(_append, line)


def _append(line: str) -> None:
    path = Path(DISPATCH_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_bundle(extra_info: dict[str, Any]) -> CaseBundle:
    """从 parquet 行的 extra_info 把四件套读回来。

    RL 每一步都要重放世界、对照评分标准，所以这三个文件是**运行时依赖**，
    不能提前烤进 parquet（详见 core/schemas.py 的模块 docstring）。
    """
    return CaseBundle.read(Path(extra_info["batch_dir"]), extra_info["case_id"])


def score(bundle: CaseBundle, output: RolloutOutput, domain) -> ScoreResult:
    """纯函数打分。不碰网络、不碰模型，同一条轨迹每次算分完全一样。"""
    return score_trajectory(
        bundle, output.trajectory, output.sandbox,
        policy_scorer=domain.policy_scorer, decision_fn=domain.decision_fn, caps=domain.caps,
    )


def reward_extra_info(result: ScoreResult, output: RolloutOutput) -> dict[str, Any]:
    """回给 verl 的诊断字段。

    必须是 numpy-stackable 的标量或短列表——复杂结构（cap_steps 的嵌套 dict、
    子分明细）放 artifact 文件里，塞进 batch 会在 collate 时炸。
    """
    metrics = output.metrics
    hit = {h.name for h in result.cap_hits}
    return {
        "reward": result.reward,
        "raw_reward": result.raw_reward,
        **{f"subscore/{k}": v for k, v in result.subscores.items()},
        # ★ 逐条 cap 展开，不能只报个数。
        #
        # reward hacking 的指纹是「**reward 涨了但 cap 没降**」——
        # 只有 num_active_caps 一个总数的话，「少踩了两条、多踩了两条」和
        # 「真的改好了」在曲线上一模一样。设计文档 §31.3 要的就是这个分解。
        **{f"cap/{name}": int(name in hit) for name in CAP_NAMES},
        "num_active_caps": len(result.cap_hits),
        "num_steps": metrics["num_steps"],
        "tool_errors": metrics["tool_errors"],
        "parse_errors": metrics["parse_errors"],
        "truncated": int(metrics["truncated"]),
        # ★ 异步对照实验的核心观测量：生成耗时 vs 工具耗时 vs 总墙钟
        "generate_seconds": metrics["generate_seconds"],
        "tool_seconds": metrics["tool_seconds"],
        "wall_seconds": metrics["wall_seconds"],
        # TIS 可信度：占位 logprob 占比越高，重要性采样权重越不可信
        "logprob_coverage": metrics["logprob_coverage"],
    }


def write_artifact(root: Path, bundle: CaseBundle, output: RolloutOutput, result: ScoreResult) -> Path:
    """落盘完整轨迹，供事后复盘。这是**磁盘 IO**，也该放线程池。"""
    directory = root / bundle.case_id / output.trajectory.rollout_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "trajectory": output.trajectory.to_dict(),
        "sandbox": output.sandbox.export(),
        "score": result.to_dict(),
        "token_trace": output.token_trace,
        "metrics": output.metrics,
    }
    path = directory / "rollout.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


async def score_and_persist(
    bundle: CaseBundle, output: RolloutOutput, domain, artifact_root: Path | None
) -> ScoreResult:
    """打分 + 落盘。★ 阻塞与否的开关就在这里。

    我们的 verifier 是纯 CPU 的（没有 LLM judge），比老师那套的 120 秒 HTTP 轻得多，
    但**落盘仍然是磁盘 IO**，长轨迹的 artifact 有几百 KB。在 8 条 rollout 并发时，
    同步 IO 一样会把事件循环卡住。开关保留下来就是为了把这个差值量出来。
    """
    def _work() -> ScoreResult:
        result = score(bundle, output, domain)
        if artifact_root is not None:
            write_artifact(artifact_root, bundle, output, result)
        return result

    if ASYNC_VERIFIER:
        return await asyncio.to_thread(_work)
    return _work()          # 对照组：复现老师那套的阻塞行为


# --------------------------------------------------------------------------
# verl 接线
# --------------------------------------------------------------------------

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
except ImportError:  # pragma: no cover - 没装 verl 时核心循环仍可单测
    AgentLoopBase = object  # type: ignore[assignment, misc]
    register = lambda name: (lambda cls: cls)  # type: ignore[assignment]  # noqa: E731
    AgentLoopOutput = AgentLoopMetrics = None  # type: ignore[assignment]


@register("syncopate_adcampaign")
class SyncopateAgentLoop(AgentLoopBase):  # type: ignore[misc]
    """verl 侧入口。注册名要和 configs/verl_agent_loop.yaml 里写的一致。"""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> "AgentLoopOutput":
        extra_info = kwargs["extra_info"]
        domain = build_domain()
        domain.registry.latency_scale = LATENCY_SCALE
        bundle = load_bundle(extra_info)

        # 把 verl 的 server_manager 包装成核心循环要的接口。
        # request_id 由这里生成（verl 自带的 agent loop 也是 uuid4().hex），
        # 不是从 extra_info 传进来的——它是每次生成请求的标识，不是样本标识。
        async def generate(prompt_ids: list[int], params: dict[str, Any]) -> Generation:
            token_output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=params
            )
            # ★ log_probs 必须带出来，否则 rollout_corr/* 那套 TIS 诊断全是空的
            #   —— 而它正是 docs/syncopate/00-research-question 的观测基础。
            #   需要 actor_rollout_ref.rollout.calculate_log_probs=True 才有值。
            return Generation(token_ids=list(token_output.token_ids),
                              log_probs=list(token_output.log_probs or []) or None)

        config = RolloutConfig(
            max_assistant_turns=int(extra_info.get("max_assistant_turns", 8)),
            max_prompt_length=int(self.config.actor_rollout_ref.rollout.get("max_model_len", 12288)) // 2,
            max_response_length=int(self.config.data.max_response_length),
        )

        output = await run_rollout(
            bundle, registry=domain.registry, tokenizer=self.tokenizer, generate=generate,
            config=config, sampling_params=sampling_params,
            # ⚠️ 必须每条 rollout 一个唯一 id。GRPO 会把同一个 case 复制 n 份，
            # 它们的 extra_info **完全相同**——之前用固定的 "r0"，
            # 结果 4 条 rollout 写到同一个 artifact 路径互相覆盖，只剩最后一条，
            # 而且不会报错。namespace_id 也靠它保证各自的写动作不串台。
            rollout_id=uuid4().hex[:8],
            run_id=os.environ.get("SYNCOPATE_RUN_ID", "verl"),
        )

        artifact_root = extra_info.get("artifact_root")
        result = await score_and_persist(
            bundle, output, domain, Path(artifact_root) if artifact_root else None
        )
        # ★ 记在这里而不是记在返回之后：这条 rollout 到此就算「跑完了」，
        # 至于它会不会被 trainer 采纳，是下游的事——而两者的差正是我们要量的东西。
        await record_dispatch(bundle, output, result.reward)

        return AgentLoopOutput(
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            response_mask=output.response_mask,
            # 有 logprob 才传；全是占位值时传 None，免得给 TIS 喂假数据
            response_logprobs=(output.response_logprobs
                               if output.metrics["logprob_coverage"] > 0.5 else None),
            reward_score=result.reward,
            num_turns=output.num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={"reward_extra_info": reward_extra_info(result, output)},
        )
