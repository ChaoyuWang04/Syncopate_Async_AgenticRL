"""本地推理评测：让模型**自己生成**，跑完整 rollout 并打分。

★ 为什么必须有这个，SFT 的 val_loss 不够用

SFT 的 loss 是 **teacher forcing** 下算的——每一步都喂正确的前缀，模型只需要预测
下一个 token。而真实 rollout 是**自回归**的：第 3 步的输入是它自己第 2 步的输出，
错误会一路累积放大。

所以 `val_loss=0.001` 和 "能不能跑通一条轨迹" 是两个几乎无关的量。
0.6B 在冒烟测试里出过 42 次格式错误，而它的 teacher-forced loss 很低。
**唯一有意义的 SFT 效果度量，是真的让它自己走一遍。**

引擎是 vLLM（AsyncLLMEngine：prefix caching + CUDA graph + 组内并发）。
2026-08-13 之前走 transformers 逐条 generate —— 每一轮都把整条历史重新
prefill，104×8 的全量评测要一百多分钟；换引擎后进入分钟量级。

⚠️ 引擎决定采样内核：**配对比较必须同引擎**。HF 时代的审计（v8 及更早的
_audit/*.json）不能和新引擎的审计逐 case 配对；新对照组（base/SFT/RL）
全部用新引擎重跑。审计 label 带 [vllm] 后缀就是为了防这种混比。

    python -m syncopate.train.eval_local --model models/Qwen3-0.6B --limit 20
    python -m syncopate.train.eval_local --model models/Qwen3-0.6B \
        --adapter checkpoints/sft/qwen06b_v1 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from syncopate.core.schemas import CaseBundle
from syncopate.core.verifier_engine import score_trajectory
from syncopate.domains.adcampaign import build_domain
from syncopate.pipeline.split import (
    DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, assert_same_data_version, data_version_of,
)
from syncopate.train.rollout_budget import (
    MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, THINK_ON,
    SAMPLING_TOP_K, SAMPLING_TOP_P,
)
from syncopate.train.rollout_loop import RolloutConfig, run_rollout

# 多轮累积的预算：最长的模板（GEO）max_steps=14，实测每步约 140 token
#（模型输出 + 工具返回），留一倍余量。
# 多轮累积余量：response 预算全部用满 + 2048 的安全边（工具返回/结束符等）。
# ⚠️ 写成推导式而不是常量（2026-08-19）：think 模式下 response 预算会变（2048→8192），
#   这里若还是写死 4096，engine 的 max_model_len 就装不下 —— 又是「两处各写各的」。
#   默认（think-off）时 2048+2048=4096，与旧值逐字节相同。
MAX_TURN_ACCUMULATION = MAX_RESPONSE_LENGTH + 2048

ROOT = Path(__file__).resolve().parents[2]


class HFEngine:
    """把 transformers 的 generate 包成核心循环要的接口。

    ⚠️ 已不再是评测入口（评测走 VLLMEngine）。保留是因为 staleness.py
    要用它拿**旧 ckpt** 生成再用新权重重算 logprob —— 那条路需要 HF 前向。"""

    def __init__(self, model, tokenizer, max_new_tokens: int, temperature: float) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = next(model.parameters()).device

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=SAMPLING_TOP_P if self.temperature > 0 else None,
                top_k=SAMPLING_TOP_K if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return out[0][len(prompt_ids):].tolist()


class VLLMEngine:
    """评测的唯一引擎（vLLM AsyncLLMEngine）。接口与 HFEngine 相同。

    为什么值得有第二个后端：HF 路径每一轮都把整条历史重新 prefill——
    多轮之间、同 case 8 次采样之间、同模板 case 之间的公共前缀全部重复计算。
    vLLM 的 prefix caching 让这三层重复都变成缓存命中，只算增量；decode 走
    CUDA graph 而不是 eager。评测独占整卡（没有 actor 抢显存），KV 池给足，
    命中率接近理想值——这本身就是 Ostinato H0 要的对照数据点。

    ⚠️ 换后端 = 换采样内核。**配对比较必须两边同后端**（和「改 system.txt
    基线作废」同一条纪律）；跨后端只能看聚合趋势，不能逐 case 配对。
    采样参数逐项对齐 HF 路径：temperature/top_p 显式，top_k=20 ——
    HF 是从 generation_config.json **隐式**继承的，这里必须显式写出来，
    否则两个后端跑的根本不是同一个采样分布。
    """

    def __init__(self, model_path: str, adapter: str | None,
                 max_new_tokens: int, temperature: float, gpu_util: float) -> None:
        import itertools
        import logging
        import os

        # ★★★ 必须在 import vllm 之前设（2026-08-13 在 4×5090 / 驱动 570.195.03 上实测）
        #
        # vLLM 自带的 FlashAttention-2（`_vllm_fa2_C.varlen_fwd`）在 sm_120 上走 PTX JIT，
        # 而那份 PTX 是用比本机驱动更新的工具链编译的：
        #     CUDA error: the provided PTX was compiled with an unsupported toolchain
        # ⇒ **模型能加载、引擎能起、一开始生成就炸**（最难查的那种时序）。
        #
        # 四个后端的实测：
        #     默认(FA2)      ❌ unsupported toolchain
        #     FLASHINFER     ❌
        #     TRITON_ATTN    ✅   ← 本地 JIT 编译，用的就是本机 CUDA 12.8，不存在错配
        #     FLEX_ATTENTION ✅
        #
        # ⚠️ 用 setdefault：换一台驱动够新的机器时，`export VLLM_ATTENTION_BACKEND=` 之外
        # 什么都不用改；而且**别把它写死** —— FA2 在能用的机器上更快。
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        # vLLM 的日志默认打 stdout，而我们的 stdout 是要被解析的报告 —— 挪去 stderr
        for handler in logging.getLogger("vllm").handlers:
            if hasattr(handler, "stream"):
                handler.stream = sys.stderr

        # eos 按 generation_config 的完整清单（Qwen3 是 [im_end, endoftext]），
        # 与 HF generate 的停机条件一致
        try:
            from transformers import GenerationConfig
            eos = GenerationConfig.from_pretrained(model_path).eos_token_id
            eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        except Exception:
            eos_ids = []

        # ★ async 引擎而不是同步 LLM：同步版每次 generate 都独占引擎，
        # 组内 k 份采样只能排队——continuous batching 整个空转。
        # async 版多条 rollout 同时在引擎里飞，decode 自动拼 batch。
        # ⚠️ 引擎把后台任务绑在第一个事件循环上，所以整个评测必须跑在
        # **同一个 asyncio.run** 里（main 里已按此重构）。
        # E19-c（2026-08-21）：serving 量化的质量配对臂。默认全关 = 行为一字不变；
        # 设了就打判据行（防「机制在但没接上」——env 拼错时这行不出现，臂作废）。
        _quant = os.environ.get("SYNCOPATE_EVAL_QUANT") or None
        _kv_dtype = os.environ.get("SYNCOPATE_EVAL_KV_DTYPE") or "auto"
        if _quant or _kv_dtype != "auto":
            print(f"[eval-quant] quantization={_quant} kv_cache_dtype={_kv_dtype}",
                  file=sys.stderr)
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            quantization=_quant,
            kv_cache_dtype=_kv_dtype,
            gpu_memory_utilization=gpu_util,
            # ★ 这里要的是**多轮累积后的总长**，不是首轮 prompt 的上限。
            #
            # 踩过（2026-08-13）：写成 MAX_PROMPT_LENGTH + 512 = 5632，
            # 结果 GEO 走到第 9 步时对话累积到 5711 token，引擎直接抛
            # `decoder prompt longer than max_model_len` 并杀掉 EngineCore。
            # MAX_PROMPT_LENGTH 管的是"首轮 prompt 太长就左截断"，
            # 和"这一轮对话最终会有多长"是两件事 —— 每步的工具返回都会往上加。
            #
            # 按最坏情况配：首轮 prompt 上限 + 每步约 140 token × 最多 14 步 + 余量。
            max_model_len=MAX_PROMPT_LENGTH + MAX_TURN_ACCUMULATION,
            enable_prefix_caching=True,
            # 周期性打印吞吐 / prefix cache 命中率 / preemption —— H0 的仪表
            disable_log_stats=False,
            enable_lora=bool(adapter),
            max_lora_rank=64,
        ))
        self._request_counter = itertools.count()
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            # 不合并权重，运行时应用 LoRA（W + BA·scale，数学上等价于 merge_and_unload）
            self.lora = LoRARequest("eval_adapter", 1, adapter)
        # ★ 2026-08-18：top_p / top_k 改从 `rollout_budget` 取 —— **和训练同一份**。
        # 此前是 0.95 / 20（对齐的是 eval-HF），而训练是 1.0 / -1 ⇒ 两边采的不是同一个分布。
        self.params = SamplingParams(
            temperature=temperature,
            top_p=SAMPLING_TOP_P if temperature > 0 else 1.0,
            top_k=SAMPLING_TOP_K if temperature > 0 else -1,
            max_tokens=max_new_tokens,
            stop_token_ids=eos_ids,
            detokenize=False,       # 核心循环自己管 token，不需要引擎反解文本
        )

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        request_id = f"eval-{next(self._request_counter)}"
        final = None
        async for out in self.engine.generate(
                {"prompt_token_ids": prompt_ids}, self.params, request_id,
                lora_request=self.lora):
            final = out
        return list(final.outputs[0].token_ids)


def load_model(model_path: str, adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # 合并进权重，推理更快
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def load_frozen_eval(batch_dir: Path, split_dir: Path, limit: int | None) -> list[CaseBundle]:
    """从冻结的 EVAL 桶取 case。

    ★ 这是唯一可信的评测来源。之前用「排序后每 N 条取一条」从 val 里选，
    而那个 val 和 train 同模板、且实测有 6 条内容完全相同的泄漏——
    在它上面得到的所有分数都不可采信。
    """
    ids = json.loads((split_dir / "eval_cases.json").read_text(encoding="utf-8"))["case_ids"]
    if limit:
        ids = ids[:limit]
    return [CaseBundle.read(batch_dir, cid) for cid in ids]


def load_cases(batch_dir: Path, per_class: int, split_every: int) -> list[CaseBundle]:
    """★ 按 signal_class **分层**取样，每类固定条数。

    早期版本是"排序后每 N 条取一条"，结果 20 条里全是 CLAR_ 和 GRAD_
    ——因为 case_id 字母序把它们排在了前面。于是 all_low / long_tail / high_risk
    这些**最该关心的难任务一条都没评到**，而那才是判断"能不能开始 RL"的依据。

    只从 val 切分里挑（和 data build 的 val_every 对齐），保证评的是没训过的。
    """
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(manifest["entries"], key=lambda e: e["case_id"])
    val_entries = [e for i, e in enumerate(entries) if i % split_every == 0]
    # ★ 按 case_id 前缀（= 模板）分层，不是按 signal_class。
    # clarify / reject 的 signal_class 也是 "graded"，按 signal_class 分层会让
    # 字母序靠前的 CLAR_ 吃满配额，真正的 GRAD_（240 条主力）一条都评不到。
    by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in val_entries:
        by_class.setdefault(entry["case_id"].split("_")[0], []).append(entry)
    picked: list[dict[str, Any]] = []
    for _, group in sorted(by_class.items()):
        picked.extend(group[:per_class])
    return [CaseBundle.read(batch_dir, e["case_id"]) for e in picked]


def _report_defer(rows: list[dict]) -> None:
    """★ M1 的验收指标：`defer` 的**双向**准确率。

    只测「该 defer 时 defer 了」会训出一个什么都不敢做的 agent——
    它在业务上和一个乱动手的 agent 一样没用，但单向指标上是满分。
    所以必须同时测「数据已经收敛时，有没有多余的 defer」。

    分母按**采样次数**算而不是 case 数：8 次里错 3 次，按众数统计会显示成全对。
    """
    def deferred(group: list[dict]) -> tuple[int, int]:
        return (sum(b == "defer" for r in group for b in r["behaviors"]),
                sum(len(r["behaviors"]) for r in group))

    hit, total = deferred([r for r in rows if r["expected_behavior"] == "defer"])
    if not total:
        return                      # 这批评测里没有 defer 类 case，指标无意义
    miss, miss_total = deferred([r for r in rows if r["expected_behavior"] != "defer"])
    print("\n★ defer 双向准确率 —— 单向达标没有意义")
    print(f"  该 defer 时 defer 了     {hit}/{total} ({hit / total:.0%})")
    print(f"  不该 defer 却 defer 了   {miss}/{miss_total} ({miss / miss_total:.1%})"
          "   ← 必须接近 0，否则就是训出了一个什么都不敢做的 agent")


# 恢复动作的痕迹。system.wait 是最干净的一个：正常流程里**完全用不到它**。
RECOVERY_MARKERS = ("system.wait",)


def _report_behavior_matrix(rows: list[dict]) -> None:
    """★ 行为五分类的准确率与混淆矩阵 —— M6 六条毕业条件的第 2 条（门槛 90%）。

    为什么要看**矩阵**而不只是一个准确率：
    tool_call 占 EVAL 的 79%，一个"永远答 tool_call"的退化模型能拿到 79% 的
    总准确率 —— 看着不算太差，但它在 clarify/reject/defer/answer 上全错。
    **总准确率会被多数类稀释**，必须逐类看召回。

    ⚠️ 按**采样次数**算，不按 case 算（用 behaviors 而不是众数 behavior）：
    只留众数会把「8 次里错 3 次」压成一个看不见的 0 —— 和 defer 双向指标同一条纪律。
    """
    if not rows:
        return
    pairs = [(r["expected_behavior"], b) for r in rows for b in r.get("behaviors", [])]
    if not pairs:
        return
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    matrix: dict[tuple[str, str], int] = collections.Counter(pairs)
    correct = sum(c for (e, p), c in matrix.items() if e == p)
    total = len(pairs)
    flag = "" if correct / total >= 0.90 else "   ← ★ 低于 M6 门槛 90%"
    print("\n★ 行为分类（五分类，按采样次数算）")
    print(f"  总准确率  {correct}/{total} ({correct/total:.1%}){flag}")
    print(f"  {'期望\\实际':<12}" + "".join(f"{l:>11}" for l in labels) + f"{'召回':>9}")
    for exp in labels:
        row_total = sum(c for (e, _), c in matrix.items() if e == exp)
        if not row_total:
            continue
        cells = "".join(f"{matrix.get((exp, act), 0):>11}" for act in labels)
        recall = matrix.get((exp, exp), 0) / row_total
        warn = "  ⚠" if recall < 0.90 else ""
        print(f"  {exp:<12}{cells}{recall:>8.0%}{warn}")


def _report_diversity(rows: list[dict]) -> None:
    """★ 采样多样性 —— M6 六条毕业条件里最容易被忽略、也最要命的一条。

    设计文档 §30.2：**每 case 8 次采样出现 ≥2 种工具序列的比例 ≥ 70%**。

    为什么它比 reward 重要：GRPO 的梯度**完全来自组内 reward 的方差**，
    而方差来自采样的多样性。SFT 训过头 ⇒ 8 次采样吐出 8 条一模一样的轨迹
    ⇒ 组内 reward 全等 ⇒ advantage 恒为 0 ⇒ **RL 一步都训不动**。
    现象是"RL 没效果"，病根在 SFT 阶段（手册 §20）。

    ⚠️ 它和「零梯度格子占比」不是同一个数，两个都要看：
        零梯度   看的是 **reward** 一不一样（可能走了不同的路，分却一样）
        多样性   看的是 **轨迹** 一不一样（更早期的信号 —— 路都一样了，分必然一样）
    多样性塌了但零梯度还没涨，就是熵正在塌的**前兆**。
    """
    if not rows:
        return
    per_case = []
    for row in rows:
        seqs = {tuple(s) for s in row.get("tool_seqs", [])}
        per_case.append(len(seqs))
    multi = sum(1 for n in per_case if n >= 2)
    ratio = multi / len(per_case)
    flag = "" if ratio >= 0.70 else "   ← ★ 低于 M6 门槛 70%，RL 会训不动"
    print("\n★ 采样多样性 —— GRPO 的梯度来自组内方差，方差来自它")
    print(f"  8 次采样出现 ≥2 种工具序列   {multi}/{len(per_case)} ({ratio:.0%}){flag}")
    print(f"  平均不同序列数               {statistics.mean(per_case):.2f}")
    only_one = [r["case_id"] for r, n in zip(rows, per_case) if n == 1]
    if only_one:
        print(f"  只有一种轨迹的 case          {len(only_one)} 条"
              f"{'，前 8 个: ' + str(only_one[:8]) if only_one else ''}")


def is_write_case(bundle: CaseBundle) -> bool:
    """这条 case 的 gold 里有没有**写动作**。

    判据取自工具注册表的 `kind`，不另立一套分类 —— 一份工具在两个地方
    有两种"是不是写"的说法，迟早会分叉（本项目栽过：沙盒和 runtime 的契约必须同源）。
    """
    from syncopate.core.tool_registry import REGISTRY

    if bundle.gold is None:
        return False
    # GoldPath.actions 是 [{tool, arguments}] 的纯 dict（见 core/schemas.py）
    return any((spec := REGISTRY.get(a.get("tool", ""))) is not None and spec.kind == "write"
               for a in bundle.gold.actions)


def _report_read_write(rows: list[dict]) -> None:
    """★★ 读操作 ⊥ 写操作 —— 设计文档 §21 三条"绝对不能合并"之一。

    原话：**「混在一起，大量读操作会稀释掉写操作的风险」**。
    这条尺子在 M7 验收时是缺的 —— §31.2 的毕业条件里写着
    「E2E 任务成功率按读/写分桶均达标」，而全项目没有任何代码按读/写分过桶，
    于是那一条从头到尾**无法判定**（2026-08-16 审计）。

    为什么这个分桶特别要紧：本项目的 EVAL 里读多写少，
    把两者平均起来，**写动作上的错误会被读操作的高分直接盖掉** ——
    而两者的代价差几个数量级（答得啰嗦 vs 乱花钱）。

    分组依据是 **gold 里有没有写工具**（`ToolSpec.kind == "write"`），
    由数据声明、不是事后按 reward 推断 —— 同 `has_failure` 那条。
    """
    write = [r for r in rows if r.get("is_write")]
    read = [r for r in rows if not r.get("is_write")]
    if not write:
        return

    def line(name: str, group: list[dict]) -> str:
        n = len(group)
        mean = statistics.mean(g["reward"] for g in group)
        # "成功"= 该 case 的 8 次采样里 reward 达到满分档的比例。
        # 用采样次数而不是 case 数：8 次里成功 3 次和成功 8 次不是一回事。
        ok = sum(1 for g in group for v in g["group"] if v >= 0.999)
        tot = sum(len(g["group"]) for g in group)
        caps = sum(len(g["caps"]) for g in group)
        return (f"  {name:<12}{n:>4}{mean:>9.3f}{ok}/{tot:>9}"
                f"{ok/tot:>9.0%}{caps/max(tot,1):>10.2f}")

    print("\n★ 读 / 写分桶 —— §21：混在一起，大量读操作会稀释掉写操作的风险")
    print(f"  {'桶':<12}{'n':>4}{'reward':>9}{'满分次数':>12}{'成功率':>9}{'cap/次':>10}")
    print(line("只读", read))
    print(line("含写动作", write))
    delta = (statistics.mean(g["reward"] for g in read)
             - statistics.mean(g["reward"] for g in write))
    print(f"  读−写 差值 {delta:+.3f}"
          "   ← 正得越多，说明总分越是被读操作撑起来的")


def _rereport(audit_path: Path, batch_dir: Path) -> int:
    """对一份已有的审计 JSON 重出报告（不跑模型）。

    ★ 为什么要有这条路：补一把新尺子之后，历史基线上那个指标是**缺失**的，
    而重跑一遍 EVAL 要几小时 GPU。审计文件里逐 case 的采样结果都在，
    缺的只是分组依据（比如 `is_write` 要从 gold 里读）—— 那从 batch 补上就行。

    ⚠️ 补分组依据时必须用**这份审计当初跑的那个 batch**。
    拿另一个版本的 gold 去给旧审计分组，得到的数看起来正常，但它谁也不是。
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = audit["rows"]
    missing = 0
    for row in rows:
        try:
            bundle = CaseBundle.read(batch_dir, row["case_id"])
        except (FileNotFoundError, OSError):
            missing += 1
            continue
        row["is_write"] = is_write_case(bundle)
        row.setdefault("has_failure", bool(bundle.env.failures))
    print(f"[重出报告] {audit.get('label', audit_path.name)}   {len(rows)} 条 case"
          f"   语料 {batch_dir}")
    if missing:
        print(f"⚠️ 有 {missing} 条 case 在 {batch_dir} 里找不到 —— "
              f"审计和 batch 多半不是同一个版本，下面的分桶数不可信")
    _report_read_write(rows)
    _report_defer(rows)
    _report_recovery(rows)
    _report_behavior_matrix(rows)
    _report_diversity(rows)
    return 0


def _report_recovery(rows: list[dict]) -> None:
    """★ 恢复动作的**双向**准确率 —— 和 defer 双向指标同构。

    只测「该恢复时恢复了」会训出一个**过度恢复**的 agent：
    没出事也去等待、也去重复查证。它在单向指标上满分，
    但在业务上是把每次操作都拖慢几十秒。

    分组依据是 `env.failures` 非空 —— 由 case 声明，不是事后推断。
    分母按**采样次数**算：8 次里多等 3 次，按众数统计会显示成全对。

    ⚠️ 这个指标存在的直接原因：SFT 桶里 F 类占 45%（因为 F 贡献了 48% 的死格），
    我们需要**测**它有没有导致过度恢复，而不是靠调配额提前猜。
    """
    def rate(group: list[dict]) -> tuple[int, int]:
        hit = sum(any(t in RECOVERY_MARKERS for t in seq)
                  for r in group for seq in r.get("tool_seqs", []))
        total = sum(len(r.get("tool_seqs", [])) for r in group)
        return hit, total

    should = [r for r in rows if r.get("has_failure")]
    should_not = [r for r in rows if not r.get("has_failure")]
    hit, total = rate(should)
    if not total:
        return
    over, over_total = rate(should_not)
    print("\n★ 恢复动作双向准确率 —— 和 defer 同理，单向达标没有意义")
    print(f"  有故障时用了恢复动作   {hit}/{total} ({hit / total:.0%})")
    print(f"  无故障却用了恢复动作   {over}/{over_total} ({over / over_total:.1%})"
          "   ← 必须接近 0，否则就是训出了一个见谁都先等三十秒的 agent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地自回归推理评测")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录，不给就是基座")
    # ★ 默认值来自**一份共用常量**（`pipeline/split.py`）——此前这里写死 v2，
    #   而 `data/batches/v2` 在本机根本不存在。⚠️ 这两个参数必须同时动，见下面的断言。
    parser.add_argument("--batch", default=DEFAULT_BATCH_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR,
                        help="用冻结 EVAL 桶（推荐）；设为空字符串则退回旧的 per-class 取样")
    parser.add_argument("--limit", type=int, default=None)
    # ★★ 按 case 分片，配合 `scripts/eval_parallel.sh` 做多卡并行。
    #
    # 评测天生可分：每条 case 的 Sandbox 按 namespace **每次新建**（账本 / 失败计数器 /
    # BUC 积分全在它身上），共享的 `bundle.env` 只读、registry 只持只读工具规格。
    # 这是当年修「rollout_id 固定导致 artifact 互相覆盖」时立下的设计，
    # 现在直接成了分片的通行证 —— **不需要 tensor parallel**（4B 单卡装得下，
    # 而这台机器 P2P 全关，TP 只会让通信变瓶颈）。
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="只跑第 I 片（0-indexed）共 N 片，如 --shard 0/4")
    parser.add_argument("--per-class", type=int, default=4, help="每个 signal_class 取几条")
    parser.add_argument("--split-every", type=int, default=8, help="和 data build 的 val_every 对齐")
    # 单轮生成上限（vLLM SamplingParams.max_tokens）。
    # ★ 2026-08-19 起默认 = MAX_RESPONSE_LENGTH（契约推导，E23「评测跟训练」）：
    #   训练侧单轮可用满剩余预算（至多 2048/think 8192），评测卡低值就是两个分布 ——
    #   SFT 类模型轮短（≤~100 tok）伤不到，但 base/崩塌型的长轮会被拦腰砍
    #   （base 实测截断率 0.203、E27 B 臂第一跑因 256 作废，2026-08-19 两次撞上）。
    #   ⚠️ 旧默认 256 的历史审计仍可比（它们的截断率 ≤0.9%），但**跨代配对要看
    #   审计头部的 max_new_tokens 字段**（本次起记录，e27 双胞胎 label 分不清的教训）。
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="单轮生成上限；不传则按 SYNCOPATE_THINK 取 256/2048")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="测组内方差必须 >0；要看确定性行为才设 0")
    parser.add_argument("--samples-per-case", type=int, default=4,
                        help="★ 模拟 GRPO 的组大小。组内 reward 方差=0 就没有梯度")
    parser.add_argument("--latency-scale", type=float, default=0.0,
                        help="评测时把 480 秒审核压掉；测异步时才设 1.0")
    parser.add_argument("--gpu-util", type=float, default=0.85,
                        help="vLLM 的显存份额。评测独占整卡，可以给大 —— KV 池越大命中率越高")
    parser.add_argument("--out", default=None)
    parser.add_argument("--from-audit", default=None,
                        help="★ 不跑模型，直接对一份已有的审计 JSON 重出报告。"
                             "补了新尺子之后，用它把历史基线的值反算出来 —— "
                             "不必为了一个新指标重跑几小时 GPU")
    args = parser.parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = MAX_RESPONSE_LENGTH   # 预算本身已随 THINK_ON 切换（2048/8192）

    if args.from_audit:
        return _rereport(ROOT / args.from_audit, ROOT / args.batch)

    domain = build_domain()
    domain.registry.latency_scale = args.latency_scale
    model_path = str((ROOT / args.model).resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter = str((ROOT / args.adapter).resolve()) if args.adapter else None
    engine = VLLMEngine(model_path, adapter, args.max_new_tokens, args.temperature, args.gpu_util)
    if args.split_dir:
        # ★ 「两个东西应当相同」型判据：只改一个参数会静默评另一个 case 集（见 split.py 那一节）。
        # ⚠️ 只在用冻结桶时查 —— `--split-dir ""` 是**合法**的退回路径，那时没有版本可比。
        assert_same_data_version(args.batch, args.split_dir)
        bundles = load_frozen_eval(ROOT / args.batch, ROOT / args.split_dir, args.limit)
    else:
        bundles = load_cases(ROOT / args.batch, args.per_class, args.split_every)

    label = (f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (基座)")
             + " [vllm]")
    print(f"[评测] {label}   {len(bundles)} 条 case，temperature={args.temperature}")

    rows = []
    if args.shard:
        index_str, total_str = args.shard.split("/")
        shard_i, shard_n = int(index_str), int(total_str)
        assert 0 <= shard_i < shard_n, f"--shard {args.shard} 越界"
        # ★ 交错取（i, i+n, i+2n…）而不是切块：模板是按 case_id 排序聚在一起的，
        #   切块会让某一片全是 GEO（14 步）、另一片全是 HIGH（1 步），**负载差好几倍**。
        bundles = bundles[shard_i::shard_n]
        print(f"[分片] {shard_i+1}/{shard_n} —— 本片 {len(bundles)} 条", file=sys.stderr)

    started = time.time()

    async def _one_sample(bundle: CaseBundle, k: int) -> dict[str, Any]:
        """一份采样：rollout + 打分。并发安全的依据是 per-rollout 隔离 ——
        Sandbox 按 namespace 每次新建（账本/失败计数器/BUC 积分全在它身上），
        共享的 bundle.env 只读、registry 只持只读工具规格。这是当年修
        「rollout_id 固定导致 artifact 互相覆盖」时立下的设计，现在成了并发的通行证。"""
        output = await run_rollout(
            bundle, registry=domain.registry, tokenizer=tokenizer, generate=engine,
            config=RolloutConfig(max_assistant_turns=bundle.case.max_steps,
                                 max_prompt_length=MAX_PROMPT_LENGTH, max_response_length=MAX_RESPONSE_LENGTH),
            rollout_id=f"eval{k}",
        )
        result = score_trajectory(
            bundle, output.trajectory, output.sandbox,
            policy_scorer=domain.policy_scorer, decision_fn=domain.decision_fn, caps=domain.caps)
        return {
            "reward": result.reward,
            "parse_ok": output.trajectory.parse_ok,
            "parse_errors": output.metrics["parse_errors"],
            "tool_errors": output.metrics["tool_errors"],
            "truncated": output.metrics["truncated"],
            # ★ 截断的**原因**（tokens / observation / turns）——三者修法方向相反，
            #   合并成一个布尔值就等于不知道该拧哪个旋钮（`01 §P1-3`）
            "truncation_reason": output.metrics.get("truncation_reason"),
            "num_steps": output.metrics["num_steps"],
            "caps": [h.name for h in result.cap_hits],
            "behavior": output.trajectory.behavior,
            # 恢复动作的双向指标要用：这一次采样调了哪些工具
            "tools": [a.name for a in output.trajectory.actions],
        }

    async def _eval_case(bundle: CaseBundle) -> list[dict[str, Any]]:
        # ★ 组内 k 份采样并发进引擎，continuous batching 拼 decode
        return list(await asyncio.gather(
            *[_one_sample(bundle, k) for k in range(args.samples_per_case)]))

    async def _eval_all() -> None:
        # ★ 整个评测跑在同一个事件循环里：async 引擎的后台任务绑定首个 loop，
        # 每个 case 一次 asyncio.run 会在第二个 case 上炸。
        # ★ 逐条打进度。一个跑一百多分钟的任务没有进度输出是不合理的 ——
        # 只能靠 nvidia-smi 看它还活着，分不出"正常慢"和"卡在某条 case 上了"。
        # 打到 stderr：stdout 是那份要被解析的报告，别混进去。
        for index, bundle in enumerate(bundles, start=1):
            group = await _eval_case(bundle)
            _append_row(index, bundle, group)

    def _append_row(index: int, bundle: CaseBundle, group: list[dict[str, Any]]) -> None:
        group_rewards = [g["reward"] for g in group]
        rows.append({
            "case_id": bundle.case_id,
            "signal_class": bundle.case.metadata.signal_class,
            "template": bundle.case_id.split("_")[0],
            "reward": statistics.mean(group_rewards),
            "reward_std": statistics.pstdev(group_rewards) if len(group_rewards) > 1 else 0.0,
            "reward_max": max(group_rewards),
            "group": group_rewards,
            "parse_ok": sum(g["parse_ok"] for g in group) / len(group),
            "parse_errors": sum(g["parse_errors"] for g in group),
            "tool_errors": sum(g["tool_errors"] for g in group),
            "truncated": sum(g["truncated"] for g in group) / len(group),
            **{f"trunc_{r}": sum(g.get("truncation_reason") == r for g in group) / len(group)
               for r in ("tokens", "observation", "turns")},
            "num_steps": statistics.mean(g["num_steps"] for g in group),
            "caps": [c for g in group for c in g["caps"]],
            "behavior": collections.Counter(g["behavior"] for g in group).most_common(1)[0][0],
            # ★ 逐次采样的行为要全留下：defer 的双向准确率是按采样次数算的，
            # 只留众数会把「8 次里错 3 次」压成一个看不见的 0
            "behaviors": [g["behavior"] for g in group],
            "expected_behavior": bundle.verifier.expected_behavior,
            "tool_seqs": [g["tools"] for g in group],
            # ★ 这条 case 有没有声明失败剧本 —— 恢复动作双向指标的分组依据
            "has_failure": bool(bundle.env.failures),
            # ★ gold 里有没有写动作 —— 读/写分桶的分组依据（§21，见 _report_read_write）
            "is_write": is_write_case(bundle),
        })
        elapsed = time.time() - started
        eta = elapsed / index * (len(bundles) - index)
        print(f"  [{index:>3}/{len(bundles)}] {bundle.case_id:<12} "
              f"r={rows[-1]['reward']:.3f} 步={rows[-1]['num_steps']:.1f}  "
              f"均分={statistics.mean(r['reward'] for r in rows):.3f}  "
              f"已用 {elapsed/60:.0f}m 剩约 {eta/60:.0f}m",
              file=sys.stderr, flush=True)

    asyncio.run(_eval_all())

    rewards = [r["reward"] for r in rows]
    print(f"\n{'指标':<26}{'值'}")
    print("-" * 46)
    print(f"{'平均 reward':<26}{statistics.mean(rewards):.3f}")
    print(f"{'reward > 0 的比例':<26}{sum(r > 0 for r in rewards)}/{len(rows)}")
    print(f"{'终答解析成功':<26}{sum(r['parse_ok'] for r in rows)}/{len(rows)}")
    print(f"{'格式错误总次数':<26}{sum(r['parse_errors'] for r in rows)}")
    print(f"{'工具报错总次数':<26}{sum(r['tool_errors'] for r in rows)}")
    print(f"{'撞步数上限':<26}{sum(r['truncated'] for r in rows)}/{len(rows)}")
    print(f"{'平均步数':<26}{statistics.mean(r['num_steps'] for r in rows):.1f}")
    print(f"{'耗时':<26}{time.time()-started:.0f}s")

    print("\n★ 按模板 —— 判断能不能开始 RL 看的是 组内std，不是 mean")
    print(f"  {'模板':<14}{'n':>3}{'mean':>8}{'best':>8}{'组内std':>9}{'有梯度':>8}{'截断率':>8}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    for name, group in sorted(by.items()):
        stds = [g["reward_std"] for g in group]
        has_grad = sum(s > 0.01 for s in stds)
        print(f"  {name:<14}{len(group):>3}{statistics.mean(g['reward'] for g in group):>8.3f}"
              f"{max(g['reward_max'] for g in group):>8.3f}{statistics.mean(stds):>9.3f}"
              f"{has_grad}/{len(group):>6}{statistics.mean(g['truncated'] for g in group):>8.0%}")
    all_stds = [r["reward_std"] for r in rows]
    live = [r for r in rows if r["reward_std"] > 0.01]
    sat = [r for r in rows if r["reward_std"] <= 0.01 and r["reward"] > 0.9]
    dead = [r for r in rows if r["reward_std"] <= 0.01 and r["reward"] < 0.15]
    stuck = [r for r in rows if r["reward_std"] <= 0.01 and 0.15 <= r["reward"] <= 0.9]
    print(f"\n★ 零梯度格子的构成 —— **决定 SFT 该往哪调**")
    print(f"  有梯度（σ>0.01）      {len(live):>3}/{len(rows)}   RL 能学的就是这些")
    print(f"  饱和（σ=0, r>0.9）    {len(sat):>3}/{len(rows)}   base 已经会了，**SFT 不该碰**")
    print(f"  全灭（σ=0, r<0.15）   {len(dead):>3}/{len(rows)}   ★ **SFT 冷启动的目标**")
    print(f"  卡死（σ=0, 中间分）   {len(stuck):>3}/{len(rows)}   系统性走偏，查 cap")
    if dead:
        print(f"  全灭清单: {[r['case_id'] for r in dead]}")
    if stuck:
        print(f"  卡死清单: {[(r['case_id'], round(r['reward'],2)) for r in stuck]}")

    _report_defer(rows)
    _report_recovery(rows)
    _report_read_write(rows)
    _report_behavior_matrix(rows)
    _report_diversity(rows)

    caps = collections.Counter(c for r in rows for c in r["caps"])
    print("\ncap 命中:", dict(caps) or "无")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        # ★ 带上**数据版本**：下游只靠 label 里的路径判断，会把一份旧版本的审计
        #   静默当成本次结果（写 select_sft_ckpt 时就撞上过 v3 时代的那份）。
        path.write_text(json.dumps({"label": label,
                                    "data_version": data_version_of(args.split_dir)
                                    if args.split_dir else None,
                                    "batch": args.batch, "split_dir": args.split_dir,
                                    # ★ 生成配置进审计头（e27 两臂 label 一模一样分不清的教训）
                                    "gen": {"max_new_tokens": args.max_new_tokens,
                                            "temperature": args.temperature,
                                            "samples_per_case": args.samples_per_case},
                                    "rows": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"\n明细 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
