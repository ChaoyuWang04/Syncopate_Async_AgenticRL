"""动态分池：按「这条题现在还能不能提供梯度」来决定它被抽中的概率。

★★★ 为什么固定采样是错的

GRPO 的梯度**完全来自组内 reward 的方差**。一条 8 次采样全对、方差为 0 的题，
它消耗的 8 次 rollout 对参数更新的贡献**精确地等于零** —— 不是"贡献小"，是零。

v11 的实测：SFT 之后 198 条 EVAL 里有 66–104 个格子是饱和的（e1/e2）。
按同样的比例，RL 池 738 条里大约有三分之一在空转。而 rollout 是这条链上
最贵的一步（一步 65 秒里，绝大部分花在生成和工具执行上）。

⇒ **采样应该是动态的**：谁还在提供梯度就多抽谁，谁已经稳定做对就少抽。

★ 但「少抽」不等于「剔除」，这是本模块最重要的一条纪律

策略在漂移。这一步饱和的题，几十步之后可能因为策略走偏重新变得有梯度；
永久剔除等于**把回归检测也一起关掉了** —— 而"教了 A 忘了 B"正是这个项目
反复栽跟头的地方（defer 97%→0%、四个模板归零、MISS −0.096）。
所以是**降权到一个地板值**，不是删除。地板保证每条题隔一段时间总会被重新体检。

★ 三个信号，对应用户提的三件事

    组内 reward 方差   →  还有没有梯度            （主信号）
    num_steps         →  是不是"简单短任务"      （次信号，短且总对 = 最该降）
    last_seen_step    →  多久没被体检过了        （staleness / 回归保护）

★ 和异步的关系

同一套 per-case 状态在 `fully_async` 下还要多管一件事：一条样本是第几步的策略
生成的（staleness）。那部分等两卡环境到位再接，这里先把接口留出来
（`Pool.snapshot()` 里的 `last_seen_step` 就是它的锚点）。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 权重地板：饱和的题也保留这么多相对权重。
# 0.05 意味着它大约每 20 轮被体检一次 —— 足够早发现回归，又不至于浪费太多 rollout。
WEIGHT_FLOOR = 0.05
# 方差的指数滑动平均系数。0.3 = 大约看最近 3–4 次的表现，
# 太大则一次噪声就翻案，太小则策略漂移后反应不过来。
EMA_ALPHA = 0.3
# 没被抽过的题给多高的权重（乐观初始化）。
# 必须 > 1：冷启动阶段要保证覆盖，不能让先抽到的几条把权重全占了。
OPTIMISTIC_WEIGHT = 2.0
# 「简单短任务」的判据：步数低于它、且方差为 0，额外降一档。
# 4 步是查一次 + 给结论的量级（HIGH/MISS 那种）。
SHORT_TASK_STEPS = 4.0
# 多久没被抽过就强制回升权重（回归保护）。单位是 step。
STALE_AFTER_STEPS = 20


@dataclass
class CaseState:
    """一条题在池子里的状态。全部可序列化 —— 训练要能断点续跑。"""

    case_id: str
    seen: int = 0                    # 被抽中过几次
    ema_std: float = 0.0             # 组内 reward 标准差的滑动平均 ← 梯度贡献的度量
    ema_reward: float = 0.0
    ema_steps: float = 0.0           # 平均轨迹长度，用来识别"简单短任务"
    last_seen_step: int = -1

    def observe(self, rewards: list[float], steps: list[float], step: int) -> None:
        """吃掉一组（同一条题的 n 次采样）结果。"""
        if not rewards:
            return
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        mean_steps = sum(steps) / len(steps) if steps else 0.0
        if self.seen == 0:
            self.ema_std, self.ema_reward, self.ema_steps = std, mean, mean_steps
        else:
            self.ema_std = (1 - EMA_ALPHA) * self.ema_std + EMA_ALPHA * std
            self.ema_reward = (1 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * mean
            self.ema_steps = (1 - EMA_ALPHA) * self.ema_steps + EMA_ALPHA * mean_steps
        self.seen += 1
        self.last_seen_step = step


def weight_of(state: CaseState, step: int) -> float:
    """这条题现在该有多大的被抽中概率（未归一化）。

    ★ 设计上刻意只用**观测到的量**，不用任何"题目难度"的先验标签。
    难度标签（difficulty L1–L5）是造数据时拍的，而"现在还有没有梯度"是策略的函数 ——
    同一条题在 SFT 刚结束和 RL 跑了 50 步之后，答案可能完全不同。
    """
    if state.seen == 0:
        return OPTIMISTIC_WEIGHT            # 没体检过的先抽，保证覆盖

    # 主信号：组内方差。std 0.25 左右就是很健康的梯度来源了，所以除以 0.25 归一。
    weight = min(1.0, state.ema_std / 0.25)
    floor = WEIGHT_FLOOR

    # 次信号：简单短任务额外降一档。
    # ★ 只在"已经没方差"时才降 —— 短任务本身不是罪，短且总对才是浪费。
    #
    # ⚠️ **地板也要一起降**。第一版只降 weight 不降 floor，结果被
    # `max(WEIGHT_FLOOR, weight)` 夹回同一个 0.05 —— 长的短的出来一模一样，
    # 这个信号结构上不可能影响输出。测试当场抓到。
    # （和今天反复出现的"机制在但没接上"是同一个形状：
    #   cap 监视的工具不在菜单里 / 状态没进分层键 / 规则写在说明第二行。）
    if state.ema_std < 0.01 and state.ema_steps <= SHORT_TASK_STEPS:
        weight *= 0.5
        floor *= 0.5

    # 回归保护：太久没体检就把权重顶回来，不让它永远沉底。
    idle = step - state.last_seen_step
    if idle >= STALE_AFTER_STEPS:
        weight = max(weight, 0.5)

    return max(floor, weight)


class Pool:
    """一池子 case + 它们的动态权重。

    用法（同步 RL）：
        pool = Pool(case_ids)
        pool.ingest(dispatch_log_path, step)      # 吃上一步的结果
        batch = pool.sample(k=4, step=step, seed=step)
    """

    def __init__(self, case_ids: Iterable[str]) -> None:
        self.states: dict[str, CaseState] = {cid: CaseState(cid) for cid in case_ids}
        self._ingested_lines = 0

    # ---- 反馈 ----

    def ingest(self, dispatch_log: Path, step: int) -> int:
        """从 `dispatched.jsonl` 增量读入新结果，按 case 聚合成一组。

        ★ 增量读（记住已消费的行数）而不是每次全量重读：
        这个文件会长到几十万行，每步全量解析会变成训练循环里的隐形开销。
        """
        if not dispatch_log.exists():
            return 0
        groups: dict[str, tuple[list[float], list[float]]] = {}
        with dispatch_log.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < self._ingested_lines:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue            # 并发追加可能写出半行，跳过即可
                cid = row.get("case_id")
                if cid not in self.states:
                    continue
                rewards, steps = groups.setdefault(cid, ([], []))
                rewards.append(float(row.get("reward") or 0.0))
                steps.append(float(row.get("num_steps") or 0.0))
                self._ingested_lines = index + 1
        for cid, (rewards, steps) in groups.items():
            self.states[cid].observe(rewards, steps, step)
        return len(groups)

    # ---- 采样 ----

    def sample(self, k: int, step: int, seed: int) -> list[str]:
        """按权重**无放回**抽 k 条。

        ⚠️ 必须无放回：同一 batch 里出现两条相同的 case，GRPO 会把它们当成
        两个独立的组，而它们的 advantage 是相关的 —— 组内比较的前提被破坏了。

        ⚠️ 必须可复现：seed 由 step 决定，不用全局随机源。
        「RL 里任何跨 rollout 的随机性都是污染」这条纪律在这里同样适用 ——
        断点续跑时必须能重放出同一个 batch。
        """
        ids = sorted(self.states)
        weights = [weight_of(self.states[cid], step) for cid in ids]
        rng = random.Random(seed)
        chosen: list[str] = []
        pool_ids, pool_w = list(ids), list(weights)
        for _ in range(min(k, len(pool_ids))):
            total = sum(pool_w)
            if total <= 0:
                chosen.append(pool_ids.pop(rng.randrange(len(pool_ids))))
                pool_w.pop(0)
                continue
            pick = rng.random() * total
            acc = 0.0
            for i, w in enumerate(pool_w):
                acc += w
                if acc >= pick:
                    chosen.append(pool_ids.pop(i))
                    pool_w.pop(i)
                    break
        return chosen

    # ---- 观测 ----

    def snapshot(self, step: int) -> dict[str, Any]:
        """给 wandb / rl_report 的一行摘要。

        ★ 这几个数就是"动态分池到底有没有在工作"的证据：
        `saturated_share` 应该随训练上升（模型学会的题越来越多），
        而 `effective_pool` 应该随之下降 —— 如果两者都不动，说明池子退化成了均匀采样。
        """
        weights = {cid: weight_of(s, step) for cid, s in self.states.items()}
        total = sum(weights.values()) or 1.0
        # 有效池大小：等价于"实际在被采样的题有多少条"（Kish 有效样本量）
        effective = total ** 2 / sum(w * w for w in weights.values())
        seen = [s for s in self.states.values() if s.seen > 0]
        return {
            "pool/size": len(self.states),
            "pool/effective_size": round(effective, 1),
            "pool/never_sampled": sum(1 for s in self.states.values() if s.seen == 0),
            "pool/saturated_share": round(
                sum(1 for s in seen if s.ema_std < 0.01) / max(1, len(seen)), 4),
            "pool/mean_ema_std": round(
                sum(s.ema_std for s in seen) / max(1, len(seen)), 4),
        }

    # ---- 断点 ----

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ingested_lines": self._ingested_lines,
            "states": {cid: vars(s) for cid, s in sorted(self.states.items())},
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Pool":
        data = json.loads(path.read_text(encoding="utf-8"))
        pool = cls(data["states"].keys())
        pool._ingested_lines = data.get("ingested_lines", 0)
        for cid, raw in data["states"].items():
            pool.states[cid] = CaseState(**raw)
        return pool
