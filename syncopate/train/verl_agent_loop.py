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
    return {
        "reward": result.reward,
        "raw_reward": result.raw_reward,
        **{f"subscore/{k}": v for k, v in result.subscores.items()},
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
