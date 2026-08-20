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
import time
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

# ★ 下发侧的记账。⚠️ 2026-08-17 之前这里只记「跑完了」，那是**装错了位置**（见下面 B4 那段）。
#
# 这是「分布漂移」那把尺子的另一半：
#   训练分布  ← trainer.rollout_data_dir 的每步 dump（**实际进入训练**的）
#   下发分布  ← 这个文件的 `event=dispatch` 行（**发出去过**的）
#   完成分布  ← 这个文件的 `event=complete` 行（**跑完过**的）
#   发出−完成 = 漂移的上界；完成−训练到 = 下游丢弃
#
# sync 下有 barrier，两者恒等，天然是对照组；async 下短任务先回、长任务被
# partial_rollout 切断，训练分布会系统性偏向短任务——而长链正是 agentic 的核心。
# 没有这个文件，漂移就只能靠推理，推理不出数值。
DISPATCH_LOG = os.environ.get("SYNCOPATE_DISPATCH_LOG")
_dispatch_lock = asyncio.Lock()


# ★ 判据行。策略版本字段是「靠上游往 extra_fields 里塞、我们再转手」生效的机制 ——
# 正是本项目反复栽的那种「机制在但没接上」。所以每个 worker 进程打**一次**，
# 说清楚到底收没收到：
#     [agent-loop] 策略版本字段 ✓ min=3 max=5     ← fully_async 下应该看到这行
#     [agent-loop] 策略版本字段 ✗ 未上报（keys=[]）← 看到这行就是又断了，别跑长任务
# 只打第一条 rollout 的第一轮，不刷屏。
_version_logged = False


def _log_version_fields_once(fields: dict[str, Any]) -> None:
    global _version_logged
    if _version_logged:
        return
    _version_logged = True
    lo, hi = fields.get("min_global_steps"), fields.get("max_global_steps")
    if lo is None and hi is None:
        print(f"[agent-loop] 策略版本字段 ✗ 未上报（keys={sorted(fields)}）"
              f" —— fully_async 会崩在 detach_utils.py:153", flush=True)
    else:
        print(f"[agent-loop] 策略版本字段 ✓ min={lo} max={hi}", flush=True)


# ★★★ 2026-08-17（队列 B4 / E08-c）：仪器从「跑完了」移到「发出去了」。
#
# 原来只在 `run_rollout` 返回之后记一行 ⇒ 我们量的是「**完成** → 训练」这一段，
# 而漂移发生在「**发出** → 完成」那一段：**被中途杀掉的长轨迹一条都看不见**。
# E08 的「逐桶零差」（7200=7200）因此是**什么都没发生**的零差，不是「发生了被修好」。
# ⇒ 现在每条 rollout 写三类事件，缺口自己会显形：
#
#     dispatch  交给 agent loop 的那一刻（长任务在这里就已经存在了）
#     complete  正常跑完
#     abort     被取消/抛异常（partial_rollout 的 abort→续写走的就是这条）
#
#   发出 − 完成 − 中止 = 还在飞；**发出 − 完成 = 漂移的上界**。
#
# ⚠️ 「记完成」和「记下发」不是详略之别，是**量的东西不同** —— 这是 E08 §6 的教训：
#    仪器装错位置时，读数是干净的零，看起来像结论。
_LAST_SEEN_PARAM_VERSION: dict[str, Any] = {}


_dispatch_logged = False


async def _record_event(event: str, payload: dict[str, Any]) -> None:
    global _dispatch_logged
    if not DISPATCH_LOG:
        return
    if not _dispatch_logged:
        # ★ 判据行（本项目纪律：靠环境变量生效的机制，启动时必须打一行，
        #   没有那行就是没生效 —— 而且这里是 **worker 进程**，driver 打的不算）
        _dispatch_logged = True
        print(f"[agent-loop] 下发记账 ✓ 三类事件（dispatch/complete/abort）→ {DISPATCH_LOG}",
              flush=True)
    line = json.dumps({"event": event, "ts": time.time(), **payload}, ensure_ascii=False)
    async with _dispatch_lock:      # 多条 rollout 并发，追加要串行化
        await asyncio.to_thread(_append, line)


async def record_dispatch_start(bundle: CaseBundle, rollout_id: str) -> None:
    """case 交给 agent loop 的那一刻记一行。**这里还不知道它会不会跑完。**"""
    await _record_event("dispatch", {
        "case_id": bundle.case_id,
        "rollout_id": rollout_id,
        # ⚠️ 尽力而为：进程里最近一次生成回报的版本，不是这条轨迹自己的。
        #    真值要等 complete 那行的 min/max_global_steps。字段名带 _approx 提醒别混用。
        "param_version_approx": _LAST_SEEN_PARAM_VERSION.get("max_global_steps"),
    })


async def record_dispatch(
    bundle: CaseBundle, output: RolloutOutput, reward: float,
    rollout_id: str = "", version_fields: dict[str, Any] | None = None,
) -> None:
    """正常跑完时记一行（原来的那行，补上 rollout_id 与真实策略版本）。"""
    vf = version_fields or {}
    await _record_event("complete", {
        "case_id": bundle.case_id,
        "rollout_id": rollout_id,
        "num_steps": output.metrics["num_steps"],
        "wall_seconds": output.metrics["wall_seconds"],
        "truncated": int(output.metrics["truncated"]),
        "reward": reward,
        "min_global_steps": vf.get("min_global_steps"),
        "max_global_steps": vf.get("max_global_steps"),
    })


async def record_dispatch_abort(bundle: CaseBundle, rollout_id: str, reason: str) -> None:
    """被取消 / 抛异常时记一行。★ 这一类正是原来看不见的那些。"""
    await _record_event("abort", {
        "case_id": bundle.case_id,
        "rollout_id": rollout_id,
        "reason": reason,
    })


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


def failures_fingerprint(bundle: CaseBundle) -> str:
    """这条 case 的失败注入剧本的稳定指纹。

    ★ Q4 判据（`18 §8` / `22 §13`）：失败注入必须**由 case 声明、确定性**——
    GRPO 是组内比较，同组 8 条 rollout 若吃到不同的失败剧本，
    reward 差异就分不清是「策略不同」还是「运气不同」，advantage 被污染。
    机制上剧本存在 bundle 里、匹配无随机源 ⇒ 应当恒同；但「应当」要能被验：
    每条 rollout 把指纹带进 dump，事后按 case 聚合，**集合大小必须是 1**。
    """
    import hashlib

    payload = json.dumps(bundle.env.failures, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def reward_extra_info(result: ScoreResult, output: RolloutOutput,
                      bundle: CaseBundle) -> dict[str, Any]:
    """回给 verl 的诊断字段。

    必须是 numpy-stackable 的标量或短列表——复杂结构（cap_steps 的嵌套 dict、
    子分明细）放 artifact 文件里，塞进 batch 会在 collate 时炸。
    """
    metrics = output.metrics
    hit = {h.name for h in result.cap_hits}
    return {
        # ★ 2026-08-19：case_id 进 dump。此前 rollout dump 里没有它（gts 恒 null），
        #   后果记录在案：defer_watch 拿不到双向拆分、infra 复查 P4 组结构
        #   「不假装测了」直接搁置。这一个字段解卡两件事。
        "case_id": bundle.case_id,
        # ★ Q4 守卫：同一 case 的所有 rollout，这个指纹必须相同（见 failures_fingerprint）
        "failures_fp": failures_fingerprint(bundle),
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

        # ★★★ 策略版本字段：rollout 侧每次生成都在 `TokenOutput.extra_fields` 里带回
        # `min/max_global_steps` —— 这条轨迹的 token 分别是第几个策略版本生成的。
        #
        # **不带上去 fully_async 直接起不来**：`fully_async_policy/detach_utils.py:153`
        # 对它做 `abs(max - min)` 且**不做 None 保护**，漏报就是
        # `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'`。
        # 2026-08-13 的 M7 就崩在这里，而根因不在 verl 上游 —— 是这一层把
        # `extra_fields` 整个丢掉了（只取了 token_ids 和 log_probs）。
        #
        # ★ 但它的价值不止是"让它别崩"：`max - min` 就是**这条轨迹横跨了几个策略版本**，
        # 也就是 staleness 的经验分布。补上它等于把 σ²(k) 的真值测量接上，
        # 可以和单卡离线合成的 ESS/N=0.846 对照（设计文档 D-A6 的核心问题）。
        #
        # 聚合规则照抄 verl 自带的 `agent_loop/tool_agent_loop.py:248-254`：
        # 首轮把整个 extra_fields 收下（拿到 min），后续轮次只抬 max。
        # 其它模式下 rollout 侧不写这两个键，version_fields 保持空 —— 无害。
        version_fields: dict[str, Any] = {}

        # 把 verl 的 server_manager 包装成核心循环要的接口。
        # request_id 由这里生成（verl 自带的 agent loop 也是 uuid4().hex），
        # 不是从 extra_info 传进来的——它是每次生成请求的标识，不是样本标识。
        async def generate(prompt_ids: list[int], params: dict[str, Any]) -> Generation:
            token_output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=params
            )
            extra = getattr(token_output, "extra_fields", None) or {}
            if not version_fields:
                version_fields.update(extra)
                _log_version_fields_once(version_fields)
            elif extra.get("max_global_steps") is not None:
                version_fields["max_global_steps"] = extra["max_global_steps"]
            # 进程级的「最近见过的版本」——给 dispatch 那行做尽力而为的标注（见 B4 注释）
            if extra.get("max_global_steps") is not None:
                _LAST_SEEN_PARAM_VERSION["max_global_steps"] = extra["max_global_steps"]
            # ★ log_probs 必须带出来，否则 rollout_corr/* 那套 TIS 诊断全是空的
            #   —— 而它正是 docs/syncopate/23-research-question 的观测基础。
            #   需要 actor_rollout_ref.rollout.calculate_log_probs=True 才有值。
            return Generation(token_ids=list(token_output.token_ids),
                              log_probs=list(token_output.log_probs or []) or None)

        # ★★★ max_prompt_length 必须用**显式配置的那个**，不能拿 max_model_len 折半。
        #
        # 旧写法是 `max_model_len // 2`——一个凭空的一半。实测后果：
        #     真实 prompt 长度   4170–4210 token
        #     max_model_len=4608 → 上限 2304 → **100% 被左截断，砍掉近 1900 token**
        # 而左截断砍掉的正是 system prompt 的**开头**：工具调用规则、调查规则、
        # 以及最终结论格式里 clarify/reject/defer 那份枚举。
        # 模型看不到「可以反问、可以拒绝」，就只会一路调工具撞到步数上限——
        # 实测 CLAR/REJ 从第 1 步起 reward 恒为 0.000、截断率 92–94%，
        # 而同一个 ckpt 在 eval 上是 0.906 / 0.828。
        #
        # ⇒ 这条把训练和评测拉到了两个不同的输入分布上（设计文档 §33 训推一致性）。
        max_prompt_length = int(self.config.data.max_prompt_length)
        config = RolloutConfig(
            max_assistant_turns=int(extra_info.get("max_assistant_turns", 8)),
            max_prompt_length=max_prompt_length,
            max_response_length=int(self.config.data.max_response_length),
        )

        # ⚠️ 必须每条 rollout 一个唯一 id。GRPO 会把同一个 case 复制 n 份，
        # 它们的 extra_info **完全相同**——之前用固定的 "r0"，
        # 结果 4 条 rollout 写到同一个 artifact 路径互相覆盖，只剩最后一条，
        # 而且不会报错。namespace_id 也靠它保证各自的写动作不串台。
        # ★ 2026-08-17：提到 run_rollout 之前生成 —— dispatch / complete / abort
        #   三类事件要靠它配对，才能算出「发出去了但没回来」的那一批。
        rollout_id = uuid4().hex[:8]

        # ★★ B4：**在这里**记「发出去了」，而不是等它跑完（见上面 record_dispatch_* 的注释）
        await record_dispatch_start(bundle, rollout_id)
        try:
            output = await run_rollout(
                bundle, registry=domain.registry, tokenizer=self.tokenizer, generate=generate,
                config=config, sampling_params=sampling_params,
                rollout_id=rollout_id,
                run_id=os.environ.get("SYNCOPATE_RUN_ID", "verl"),
            )
        except asyncio.CancelledError:
            # partial_rollout 的 abort→续写、或 rollouter 排空在飞请求，都走这里。
            # ⚠️ 记完必须**原样抛出去**：吞掉 CancelledError 会把取消语义弄坏。
            await record_dispatch_abort(bundle, rollout_id, "cancelled")
            raise
        except Exception as exc:                                   # noqa: BLE001
            await record_dispatch_abort(bundle, rollout_id, type(exc).__name__)
            raise

        artifact_root = extra_info.get("artifact_root")
        result = await score_and_persist(
            bundle, output, domain, Path(artifact_root) if artifact_root else None
        )
        # ★ 这条 rollout 到此算「跑完了」；它会不会被 trainer 采纳是下游的事
        # —— 而 dispatch / complete / 训练到 三者两两的差，正是我们要量的东西。
        await record_dispatch(bundle, output, result.reward,
                              rollout_id=rollout_id, version_fields=version_fields)

        return AgentLoopOutput(
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            response_mask=output.response_mask,
            # ★ 策略版本随轨迹一起交回去（见上面 version_fields 的说明）。
            # 有 logprob 才传；全是占位值时传 None，免得给 TIS 喂假数据
            response_logprobs=(output.response_logprobs
                               if output.metrics["logprob_coverage"] > 0.5 else None),
            reward_score=result.reward,
            num_turns=output.num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={"reward_extra_info": reward_extra_info(result, output, bundle),
                          **version_fields},
        )
