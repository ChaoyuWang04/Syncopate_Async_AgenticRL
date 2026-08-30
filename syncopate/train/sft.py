"""最小 LoRA SFT 训练器。

**为什么不用 verl 的 SFT trainer**：它会自己对 messages 做 chat template 渲染，
而整段渲染和我们 RL 侧的增量拼接**天生逐 token 不相等**（Qwen3 只给最后一个
assistant 轮加空 `<think>` 块）。详见 `pipeline/sft_replay.py` 的模块 docstring。

我们的数据已经是**预分词**的 `input_ids` + `loss_mask`，由同一个 rollout 循环
回放 gold 产出——所以训练器只需要：喂 token、按 mask 算 loss。这一百来行比接
verl SFT trainer 更简单，也彻底消除了两阶段分布不一致的风险。

单卡场景下 FSDP 也没有意义（world_size=1 会退化成 NO_SHARD），所以直接用
普通的 PyTorch 训练循环 + peft。

    python -m syncopate.train.sft --model models/Qwen3-0.6B --epochs 3
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH

ROOT = Path(__file__).resolve().parents[2]

# ★★ SFT 的长度上限不是独立参数，是 rollout 契约的推论：
#    一条 SFT 样本 = prompt + 整条 gold 轨迹 ⇒ 上限就是 RL 侧的 max_model_len。
#
# ⚠️ 这里原来是 `--max-length`（默认 4096）+ 静默 `[:max_length]` 切片。
#    v13 实测 92.6%（388/419）的样本超过 4096 —— 实跑靠手传 6656 侥幸躲过，
#    而 `08` 文档示范命令写的 6144 会无声截掉 46 条。和 RL 侧 prompt 截断
#    （defer 崩塌归因翻案那次）是同一个病：**模型看不见被砍掉的部分，且无人计数**。
# ⇒ 修法与守则⑨一致：值只有一份（rollout_budget），超长**硬报错**而不是静默切。
SFT_MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH

# 重定向到文件时 Python 会缓冲 stdout，长训练看不到实时进度。
# 训练脚本的打印量很小，无脑 flush 没有代价。
print = functools.partial(builtins.print, flush=True)  # noqa: A001


class PretokenizedDataset(Dataset):
    """直接吃 parquet 里的 input_ids / loss_mask，不做任何再分词。

    ★ 超过长度预算的样本**硬报错**，不静默截断：
    截掉的是轨迹的**结尾**（最终结论那段），而 loss 恰恰集中在那里 ——
    静默截断 = 教模型「做到一半就停」，且没有任何报警。
    数据超长说明造数据和训练的契约不一致，该修的是数据或契约，不是切样本。
    """

    def __init__(self, path: Path, max_length: int, group_key: str | None = None) -> None:
        import pandas as pd

        frame = pd.read_parquet(path)
        over = [(row["case_id"], len(row["input_ids"]))
                for _, row in frame.iterrows() if len(row["input_ids"]) > max_length]
        if over:
            worst = max(length for _, length in over)
            raise SystemExit(
                f"🔴 {path} 里有 {len(over)}/{len(frame)} 条样本超过长度预算 "
                f"{max_length}（最长 {worst}）。\n"
                f"   静默截断会把轨迹结尾（最终结论）砍掉 —— 拒绝启动。\n"
                f"   预算来源：rollout_budget.py 的 MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH。\n"
                f"   超长样本（前 5 条）：{[cid for cid, _ in over[:5]]}"
            )
        self.rows = [
            {
                "input_ids": list(row["input_ids"]),
                "loss_mask": list(row["loss_mask"]),
                "case_id": row["case_id"],
                "group": str(row[group_key]) if group_key and group_key in frame.columns else "all",
            }
            for _, row in frame.iterrows()
        ]
        # ★ 零监督样本硬报错：单卡路径会静默跳过它（有计数），多卡路径会摊进
        #   一个零梯度——两种语义不同 ⇒ 从数据层面禁止这种样本存在，分叉即消失
        zero_sup = [r["case_id"] for r in self.rows if sum(r["loss_mask"]) == 0]
        if zero_sup:
            raise SystemExit(f"🔴 {path} 有 {len(zero_sup)} 条零监督样本（loss_mask 全 0）："
                             f"{zero_sup[:5]} —— 造数据的 bug，拒绝启动")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """右侧 padding。

    ★ labels 的构造是这里唯一容易错的地方：
      - padding 位置 -> -100（不算 loss）
      - loss_mask=0 的位置 -> -100（prompt 和工具返回，不算 loss）
      - 只有 loss_mask=1 的 token 才是训练目标
    把工具返回也算进 loss，等于教模型复述环境给它的东西。
    """
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        ids, mask = item["input_ids"], item["loss_mask"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append([token if m == 1 else -100 for token, m in zip(ids, mask)] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _free_gpus(threshold_mib: int = 2048) -> list[str]:
    """空闲 GPU 的物理编号（nvidia-smi 口径：显存占用 < threshold 视为空闲）。

    尊重已设的 CUDA_VISIBLE_DEVICES（只在可见集合内数）；查询失败返回 []（回退单卡）。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return []
    free = []
    for line in out.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 2 and int(parts[1]) < threshold_mib:
            free.append(parts[0].strip())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        vis = {x.strip() for x in visible.split(",")}
        free = [i for i in free if i in vis]
    return free


def _unwrap(model) -> tuple[Any, Any]:
    """拿到 (transformer 主干, lm_head)。

    PEFT 在外面包了一层，但 LoRA 是**就地替换模块**（nn.Linear → lora.Linear），
    所以直接调主干仍然走 LoRA 的前向、梯度照样回传到 adapter。
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def token_losses(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """★★ 稀疏投影：只对**被监督的位置**做 lm_head + CE。

    返回 (每个被监督 token 的 loss [N], 它属于 batch 里第几条 [N])。

    ★ 为什么不能直接用 `model(**batch).loss`
    HF 的实现是「先对**所有**位置做 lm_head，再让 CE 用 ignore_index=-100 把不要的扔掉」。
    在我们的负载上这是巨大的浪费——实测 SFT 数据**监督占比只有 3.8%**
    （平均每条 5402 token，只有 204 个 token 参与 loss）：

        朴素   logits [5402 × 151936] fp32 = 3.4 GB，其中 96.2% 算完就扔
        稀疏   logits [ 204 × 151936] fp32 = 0.12 GB

    ⇒ 那张表是全模型最大的瞬态张量，也是 `--batch-size 2` 直接 OOM 的元凶
    （峰值窗口里同时挂着 logits + fp32 副本 + dlogits ≈ 3 份）。

    为什么占比这么低是**我们这个负载特有**的：prompt 3.2k + 每轮工具返回一大堆，
    而模型自己说的话很少。通用 SFT 的监督占比通常有 30–50%，所以上游框架
    默认不做这件事——这是「负载画像决定该优化什么」的又一个例子。

    ⚠️ 数学上和朴素路径**完全等价**（不是近似）：被丢弃位置的 CE 本来就不进 loss，
    它们的 dlogits 恒为 0。有测试逐位对拍 `model(**batch).loss`。
    """
    trunk, lm_head = _unwrap(model)
    hidden = trunk(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]).last_hidden_state
    # 位置 i 的输出预测位置 i+1 的词 —— 所以要错开一位
    shift_hidden = hidden[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    rows, cols = (shift_labels != -100).nonzero(as_tuple=True)
    if rows.numel() == 0:
        # 整个 batch 一个监督 token 都没有：返回空张量，调用方跳过
        return hidden.new_zeros(0), rows
    # ★ 一次 gather 同时跨 batch 展平 —— batch>1 时每条的监督位置各不相同，
    # 用 [N, H] 的扁平表示天然处理这种不齐
    sel_hidden = shift_hidden[rows, cols]                       # [N, 2560]
    sel_labels = shift_labels[rows, cols]                       # [N]
    # A4（E30 §12）：SYNCOPATE_LMHEAD_MXFP8=1 时 lm_head 走 MXFP8 三件套
    # （sm_120 自有 kernel；判据行 [mxfp8-lmhead]；默认关，bf16 逐位不变）
    from syncopate.train import mxfp8_lmhead
    if mxfp8_lmhead.enabled():
        logits = mxfp8_lmhead.mxf8_lm_head(sel_hidden, lm_head.weight).float()
    else:
        logits = lm_head(sel_hidden).float()                    # [N, 151936]
    losses = torch.nn.functional.cross_entropy(logits, sel_labels, reduction="none")
    return losses, rows


def build_model(model_path: str, lora_rank: int, lora_alpha: int, dtype: torch.dtype,
                resume_adapter: str | None = None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if resume_adapter:
        # 续训：LoRA 结构与权重整体从存档重建（不走 get_peft_model 再灌权重的两步路——
        # 一步构造消灭「结构建了、权重没灌上」的静默半成品）。加载是否真生效由
        # main 里的起点 ΔW 断言验证（「adapter 没推送」家族的针对性防御）
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / resume_adapter), is_trainable=True)
        return model, None
    if lora_rank <= 0:
        return model, None

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        # ★ 必须挂全部线性层。只挂 q/v 是老习惯，容量差 2.8 倍
        #   （Qwen3-4B r=32：仅注意力 23.6M vs 全线性层 66.1M）。
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    return model, config


def trainable_summary(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"可训练 {trainable/1e6:.1f}M / 总计 {total/1e6:.1f}M ({trainable/total:.2%})"


@torch.no_grad()
def evaluate(model, dataset, device, pad_id: int, batch_size: int) -> dict[str, Any]:
    """按 token 加权的验证 loss，并**按组分别报**。

    只看总 loss 会掩盖问题：上一轮 clarify/reject 只占 1.4% 的 token，
    它们的 loss 就算完全没降，总 loss 也几乎看不出来。
    分组报出来才知道是不是某一类被牺牲了。
    """
    model.eval()
    by_group: dict[str, list[float]] = {}
    totals = [0.0, 0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: (collate(b, pad_id), [r["group"] for r in b]))
    for batch, groups in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # 走和训练同一条稀疏投影路径（rows 记着每个 loss 属于 batch 里第几条，
        # 按组聚合要用）—— 评测和训练用同一个函数，不留第二条路径
        losses, rows = token_losses(model, batch)
        for i, group in enumerate(groups):
            mine = rows == i
            n = int(mine.sum())
            if n == 0:
                continue
            loss_i = float(losses[mine].sum())
            by_group.setdefault(group, []).append((loss_i, n))
            totals[0] += loss_i
            totals[1] += n
    model.train()
    out: dict[str, Any] = {
        "val_loss": totals[0] / max(1, totals[1]),
        "val_ppl": math.exp(min(20.0, totals[0] / max(1, totals[1]))),
        "val_tokens": totals[1],
        "by_group": {},
    }
    for group, items in sorted(by_group.items()):
        s_loss = sum(x for x, _ in items); s_n = sum(n for _, n in items)
        out["by_group"][group] = {"loss": s_loss / max(1, s_n), "tokens": s_n,
                                  "ppl": math.exp(min(20.0, s_loss / max(1, s_n)))}
    return out



def _save_selection_point(model, out_root: Path, frac: float, step: int) -> Path:
    """存一个**临时选点产物**（bf16，约 126 MB）。

    ★ 为什么用 bf16：实测 ΔW_eff 的保真残差 **0.0024**（抽 8 层，最大 0.0024）
      ⇒ 存储损失可忽略，体积减半（fp32 252 MB → bf16 126 MB）。
      ⚠️ **不要和「合并进基座」那条混了**（`18 §3.3`：残差 0.36/0.87）——
        那条存的是 `W + Δ`，Δ 只占 W 的 0.4%，会掉进 bf16 的舍入格里；
        这里存的是 A/B 本身，量级 O(1)。**两种情况必须分别量，不能类比。**

    ⚠️ 目录用 `sel_` 前缀标成临时的 —— 它们是**选点用的**，不是要归档的。
    """
    import torch

    d = out_root / f"sel_f{frac:g}"
    d.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            saved[name] = param.detach().clone()
            param.data = param.data.to(torch.bfloat16)
    try:
        model.save_pretrained(d)
    finally:
        for name, param in model.named_parameters():
            if name in saved:
                param.data = saved[name]
    mb = sum(f.stat().st_size for f in d.glob("*")) / 1048576
    print(f"[选点] step={step} (f={frac:g}) -> {d}  {mb:.0f} MB")
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syncopate 最小 LoRA SFT")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/sft/v3/train.parquet")
    parser.add_argument("--val-file", default="data/sft/v3/val.parquet")
    parser.add_argument("--out", default="checkpoints/sft/run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)      # LoRA 的 lr 比全参高一到两个量级
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    # ⚠️ 刻意没有 --max-length：长度上限是契约的推论（SFT_MAX_LENGTH，见文件头），
    #    守则⑨「凡是"这个值应该和那边一致"的地方，这里根本不该有这个值」。
    parser.add_argument("--eval-every", type=int, default=1, help="每 N 个 epoch 评一次")
    # ⚠️ 刻意没有采样均衡开关（Chaoyu 2026-08-19 拆除）：要调分布就调**数据本身**
    #    （分桶层已有 behavior 保底饱和），训练器不做二次加权。
    # ★ 默认开 wandb，和 `launch_rl.py` 对齐（那边 --logger 默认就是 console,wandb）。
    # 之前默认是 None，结果 v8 那轮 SFT 整轮没有任何上报 —— 曲线只剩一个人肉 tail 的日志文件。
    # 训练脚本的默认值必须是「跑完就有记录」，要关得显式说。
    parser.add_argument("--wandb-project", default="syncopate")
    parser.add_argument("--no-wandb", action="store_true", help="显式关掉上报（调试/跑测试用）")
    # ★★ 选点定式（Chaoyu 2026-08-19 立、08-29 修订单位，全程见 22 §H）：
    #   **<1 epoch 的打点彻底废除** —— 0.25 遍格式都没学会（parse_err 0.50/条）。
    #   08-29 修订（Chaoyu；v14 实测最优点右移：e1 −0.105 vs e2 +0.058）：打点单位从
    #   「总步数比例」改成 **epoch 位置**——老单位是按 epochs=2 设计的，epochs=3 下会
    #   落到 1.875/2.25/2.625 这种没人要的位置（注释与现实脱节）。默认 1.5,2.5 配合
    #   整 epoch 存档 ⇒ **e1 / e1.5 / e2 / e2.5 / e3 五点谱**，评测选点一遍扫完。
    #   ⚠️ 落在第 1 个 epoch 之前的打点会被**硬过滤**（不是靠默认值挡）。
    parser.add_argument("--save-epochs", default="1.5,2.5",
                        help="在这些 epoch 位置额外存选点 ckpt（整数位置由整 epoch 存档天然覆盖，"
                             "传了也跳过；<1 epoch 硬过滤）。空串关掉。"
                             "★ 临时选点产物，选完就删（跑完会打印清理命令）")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="H0 观测仪：抓 N 个 micro-step 的 torch.profiler 时间线后自动收工"
                             "（0=关）。纯观测不改训练语义，trace 拖进 ui.perfetto.dev 看")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-adapter", default=None,
                        help="从已存的 epoch adapter 续训（例 checkpoints/sft/v14_r2/epoch2）。"
                             "lr 调度按 --resume-epochs 快进到断点位置（总步数语义与连续训练一致）；"
                             "⚠️ AdamW 动量**新起**——epoch 存档只有 adapter 没有优化器状态，"
                             "与连续训练略有差异，验收判定时如实记")
    parser.add_argument("--resume-epochs", type=int, default=0,
                        help="已完成的 epoch 数（配 --resume-adapter，从第 N+1 个 epoch 续到 --epochs）")
    args = parser.parse_args(argv)

    # ---- 默认四卡（Chaoyu 08-29）：CLI 直跑且 ≥4 张空闲卡 ⇒ 自动升格 torchrun×4；
    #      不足 4 张空闲 ⇒ 单卡回退。显式单卡：SYNCOPATE_SFT_SINGLE=1。
    #      只在 CLI 路径（argv is None）触发——程序化调 main([...])（测试/负向认证）不升格。
    if (argv is None and int(os.environ.get("WORLD_SIZE", "1")) == 1
            and not os.environ.get("SYNCOPATE_SFT_SINGLE")
            and torch.cuda.is_available()):
        free = _free_gpus()
        if len(free) >= 4:
            new_argv = list(sys.argv[1:])
            if not any(a == "--grad-accum" or a.startswith("--grad-accum=") for a in new_argv):
                # 单卡配方 2×8=16 ⇒ 四卡等效 2×2×4；不自动降会静默变 64（连 lr 语义一起变）
                new_argv += ["--grad-accum", str(max(1, args.grad_accum // 4))]
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(free[:4]))
            # ★ torchrun 从**当前解释器旁边**找，不假设它在 PATH 上。
            #   ⛔ 2026-08-30：用 `nohup .venv/bin/python -m syncopate.train.sft` 起（没先
            #     `source activate`）⇒ `FileNotFoundError: /bin/torchrun`，**起跑即挂**。
            #     v14.5 的链路脚本恰好先激活了 venv，所以这个隐形前提两个月没暴露
            #     （「干净机器才暴露的缺口」同族：手动装/手动激活过的东西 = 隐形前提）。
            import shutil
            tr = shutil.which("torchrun") or str(Path(sys.executable).parent / "torchrun")
            if not Path(tr).exists():
                raise SystemExit(f"🔴 找不到 torchrun（试过 PATH 与 {tr}）——"
                                 f"用 .venv/bin/python 起就该在 .venv/bin 里找")
            cmd = [tr, "--standalone", "--nproc_per_node=4",
                   "-m", "syncopate.train.sft", *new_argv]
            print(f"[DDP] {len(free)} 张空闲卡 ⇒ 默认四卡升格：{' '.join(cmd)}")
            os.execvpe(tr, cmd, dict(os.environ))
        print(f"[DDP] 空闲卡 {len(free)} 张 < 4 ⇒ 单卡回退（SYNCOPATE_SFT_SINGLE=1 可显式压单卡）")

    # ---- DDP 数据并行（torchrun 启动时自动生效；单进程跑 = 原语义分毫不动）----
    #
    # ★★ 刻意**不用** DistributedDataParallel 包装器，用手动梯度 all_reduce：
    #    token_losses 为了稀疏投影直调主干（trunk/lm_head），绕过包装器的 forward
    #    ⇒ DDP 的 reducer 收不到 prepare_for_backward，梯度会**静默不同步**
    #    （两基石 bug 之一「三个 rank 没同步梯度」的同族死法）。
    #    手动路径 = opd.py 已实战的模式：累积窗口末尾对每个可训练梯度 all_reduce(AVG)，
    #    LoRA 只有 ~66M 参数（fp32 梯度 ~264 MB），四卡 PCIe 上一次 <0.1s，开销可忽略。
    #    正确性由每 epoch 的「rank 间权重一致」硬断言兜底（见 epoch 尾）。
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        if rank != 0:
            # 非 rank0 静默：打印/评测/存盘/wandb 全部只归 rank0
            globals()["print"] = lambda *a, **k: None  # noqa: A001
            args.no_wandb = True

    torch.manual_seed(args.seed)
    if world > 1:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoTokenizer

    model_path = str((ROOT / args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ★ 判据行：长度预算的来源必须在启动日志里可见（没有这行 = 没走契约）
    print(f"[契约] SFT 长度上限 = {SFT_MAX_LENGTH}"
          f"（rollout_budget: prompt {MAX_PROMPT_LENGTH} + response {MAX_RESPONSE_LENGTH}）；"
          f"超长样本会硬报错，不会静默截断")
    # group_key 只用于**按组报 val loss**（某一类被牺牲要看得见），与采样无关
    train_set = PretokenizedDataset(ROOT / args.train_file, SFT_MAX_LENGTH, "behavior")
    val_set = PretokenizedDataset(ROOT / args.val_file, SFT_MAX_LENGTH, "behavior")
    collate_fn = lambda batch: collate(batch, pad_id)  # noqa: E731

    if world > 1:
        from torch.utils.data.distributed import DistributedSampler

        # drop_last=False：DistributedSampler 会补齐到各 rank 等长 ⇒
        # 每个 rank 的 micro-step 数相同 ⇒ 累积窗口边界的 all_reduce 次数天然对齐（无死锁）
        train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank,
                                           shuffle=True, seed=args.seed, drop_last=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, drop_last=False)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=False)

    model, _ = build_model(model_path, args.lora_rank, args.lora_alpha, dtype,
                           resume_adapter=args.resume_adapter)
    model.to(device)
    # ★ LoRA + gradient checkpointing 必须配这两行，否则 backward 直接报
    #   "element 0 of tensors does not require grad"。
    # 原因：LoRA 把 embedding 冻住了，checkpoint 段的输入没有 requires_grad，
    # 重算时接不上计算图。enable_input_require_grads 给输入挂一个 hook 打开梯度；
    # use_reentrant=False 用新版实现，对部分冻结的模型更稳。
    if args.lora_rank > 0:
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if world > 1:
        # 确定性双保险：seed 相同理论上各 rank 初始化一致，但仍从 rank0 广播一次
        # 可训练参数——起点一致是手动梯度同步正确性的前提，不赌初始化的确定性
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
    print(f"[模型] {args.model}  {trainable_summary(model)}")
    eff_batch = args.batch_size * args.grad_accum * world
    print(f"[数据] train={len(train_set)} val={len(val_set)}  "
          f"有效 batch = {args.batch_size}×{args.grad_accum}×{world} = {eff_batch}")
    if world > 1 and eff_batch != 16:
        # 现行配方的有效 batch = 16（2×8 单卡）。torchrun 起多卡时忘了把 --grad-accum
        # 降下来会静默改配方（16→64 连 lr 语义一起变）——必须喊出来
        print(f"⚠️⚠️ 有效 batch {eff_batch} ≠ 现行配方 16 —— 多卡时请配 "
              f"--grad-accum {max(1, 16 // (args.batch_size * world))}（除非有意改配方）")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    start_epoch = 0
    if args.resume_adapter:
        start_epoch = args.resume_epochs
        assert 0 < start_epoch < args.epochs, \
            f"续训要求 0 < --resume-epochs({start_epoch}) < --epochs({args.epochs})"
        # lr 轨迹快进到断点：总步数/调度形状与连续训练完全一致（AdamW 动量除外，见参数说明）
        for _ in range(steps_per_epoch * start_epoch):
            scheduler.step()
        print(f"[续训] {args.resume_adapter}（已完成 {start_epoch} epoch）"
              f"lr 快进到 step {steps_per_epoch * start_epoch}：{scheduler.get_last_lr()[0]:.2e}")

    run = None
    if args.wandb_project and not args.no_wandb:
        import wandb

        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        # 名字带 sft- 前缀：SFT 和 RL 上报到同一个 project，不加前缀在列表里分不出来。
        run = wandb.init(project=args.wandb_project,
                         name=args.wandb_run or f"sft-{Path(args.out).name}",
                         job_type="sft",
                         config={**vars(args), "max_length": SFT_MAX_LENGTH,
                                 "trainable": trainable_summary(model)})
        # ★ 让 W&B 自己把面板分区（train/ val/ health/ perf/ 各一块），
        #   并给关键量定 summary 口径 —— 否则 summary 默认取"最后一个值"，
        #   而我们要看的是**最小 val loss** 和**最大位移**。
        for metric, summary in (("val/loss", "min"), ("val/ppl", "min"),
                                ("train/grad_norm", "max"),
                                ("health/delta_w_ratio", "max"),
                                ("health/skipped_micro_steps", "max"),
                                ("perf/peak_memory_gb", "max")):
            wandb.define_metric(metric, summary=summary)
        print(f"[wandb] {args.wandb_mode}  {run.url if args.wandb_mode == 'online' else run.dir}")

    def log(payload: dict[str, Any], step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    base = evaluate(model, val_set, device, pad_id, args.batch_size)
    base_label = f"断点起点 (e{start_epoch})" if args.resume_adapter else "epoch 0 (未训练)"
    print(f"[eval] {base_label}  val_loss={base['val_loss']:.4f}  "
          f"ppl={base['val_ppl']:.2f}  监督 token={base['val_tokens']}")
    for group, stat in base["by_group"].items():
        print(f"        {group:<10} loss={stat['loss']:.4f}  ppl={stat['ppl']:.2f}")
    log({"val/loss": base["val_loss"], "val/ppl": base["val_ppl"],
         **{f"val/loss_{g}": v["loss"] for g, v in base["by_group"].items()},
         **{f"val/ppl_{g}": v["ppl"] for g, v in base["by_group"].items()}}, step=0)

    history = [{"epoch": 0, **{k: v for k, v in base.items() if k != "by_group"},
                "by_group": base["by_group"]}]
    # ---- H0 观测仪（--profile-steps N 才启用，纯观测不改语义）----
    # 要回答的问题：一个 micro-step 的时间花在 数据加载/前向/反向/优化器 各多少，
    # 以及显存尖峰是不是 logits 物化（K1 fused-CE 的"术前照"）。
    # 前 2 步丢弃 + 2 步预热，然后抓 N 步，trace 落到 out 目录后自动收工。
    prof = None
    prof_left = 0
    if args.profile_steps and device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile, schedule

        def _dump_trace(p) -> None:
            trace_path = ROOT / args.out / "profile_trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            p.export_chrome_trace(str(trace_path))
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))
            print(f"[profile] chrome trace -> {trace_path}")

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=2, warmup=2, active=args.profile_steps),
                       profile_memory=True, on_trace_ready=_dump_trace)
        prof.start()
        prof_left = 4 + args.profile_steps

    # ★★★ ‖ΔW‖/‖W‖ —— 判断「到底训没训动」的唯一直接证据。
    #
    # M7 那次「没测出差异」的根因就是这个数：只动了 **0.0093%**
    # （正常 LoRA 微调 0.5–5%）。**loss 降了、指标好看，而权重几乎没动。**
    #
    # ⚠️⚠️ **口径踩过一次坑，记在这**：第一版图省事算的是
    #     ‖Δθ_trainable‖ / ‖θ_trainable(初始)‖
    # 而 **LoRA 的 B 矩阵初始化为零** ⇒ 分母里只有 A、B 从 0 长起来
    # ⇒ 第一个 epoch 就报 **15.99%**，看着像"训过头"，其实**尺子错了 33 倍**
    #   （正确口径同一份权重量出来是 0.485%）。而且那个数**不能和 M7 的 0.0093% 比**。
    #
    # ⇒ 正确口径：**LoRA 实际叠加到基座上的增量，比基座本身**，且只对被适配的层算：
    #     ΔW_eff = scaling · B @ A        ratio = ‖ΔW_eff‖_F / ‖W_base‖_F
    #   定义与命令行工具在 `syncopate/train/weight_shift.py`（跑完可对任意 adapter 复算）。
    def _delta_w_ratio() -> float:
        num_sq = den_sq = 0.0
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None or not hasattr(module, "base_layer"):
                continue
            for key, a_mod in lora_a.items():
                b_mod = module.lora_B[key]
                scaling = float(module.scaling[key])
                delta = (b_mod.weight.detach().float() @ a_mod.weight.detach().float()) * scaling
                base = module.base_layer.weight.detach().float()
                num_sq += float(torch.linalg.matrix_norm(delta)) ** 2
                den_sq += float(torch.linalg.matrix_norm(base)) ** 2
        return (num_sq ** 0.5) / max(den_sq ** 0.5, 1e-12)

    if args.resume_adapter:
        # ★「adapter 没推送」家族的针对性防御：断点起点的 ΔW 必须非零（真加载了），
        #   且断点 val_loss 应远低于裸基座（上面 [eval] 行人肉可核）。ΔW≈0 = 在空跑基座。
        r0 = _delta_w_ratio()
        print(f"[续训] 起点 ||ΔW||/||W|| = {r0*100:.4f}%（应 ≈ 断点 epoch 存档时的值）")
        assert r0 > 1e-4, ("🔴 续训 adapter 没真正加载（起点 ΔW≈0 = 空跑基座）"
                           "——「adapter 没推送」家族，拒绝启动")

    # 选点：epoch 位置 × steps_per_epoch 换算成绝对步号。⚠️ 用 round 而不是 int ——
    #      截断会"少存一个点"，那是最容易被读成 bug 的那种静默偏移。
    sel_fracs = [float(x) for x in args.save_epochs.split(",") if x.strip()]
    sel_steps = {max(1, round(f * steps_per_epoch)): f
                 for f in sel_fracs if abs(f - round(f)) > 1e-9}   # 整数位=整 epoch 存档已覆盖
    # ★ 选点定式（22 §H）：<1 epoch 的打点**硬过滤**，传了也不存 —— 0.25 遍连格式都
    #   没学会，那些点只会浪费评测预算、诱惑人去选（v13r2 五点验证的实测结论）。
    #   续训时断点之前的位置同样过滤（那些步数已经过去了，存不到）。
    dropped = {s: f for s, f in sel_steps.items()
               if s < steps_per_epoch or s <= steps_per_epoch * start_epoch}
    if dropped:
        sel_steps = {s: f for s, f in sel_steps.items() if s not in dropped}
        print(f"[选点] ⛔ 过滤掉不可达的打点 {sorted(dropped)}"
              f"（<1 epoch 是选点定式 22 §H；续训时断点之前的步数已经过去，存不到）")
    if sel_steps:
        print(f"[选点] 将在 step {sorted(sel_steps)} 额外存点"
              f"（总步数 {total_steps}）—— **临时产物，选完就删**")

    global_step = steps_per_epoch * start_epoch
    skipped = 0                 # ★ 没有监督 token 被跳过的 micro-step 数
    sup_tokens = 0              # ★ 真正参与 loss 的 token 数（不是序列 token 数）
    started = time.time()
    for epoch in range(1 + start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # 不设的话每个 epoch 洗牌顺序相同
        running, seen = 0.0, 0
        epoch_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            token_loss, _ = token_losses(model, batch)
            if token_loss.numel() == 0:
                # ⚠️ **静默跳过是这个项目最贵的失效形状**（SFT 标签 bug 那次
                # val_loss 降到 0.0000 却什么都没学对）。所以跳过要计数、要上报。
                skipped += 1
                continue
            sup_tokens += int(token_loss.numel())
            # mean over 被监督 token —— 和 HF 的 `model(**batch).loss` 同口径
            loss = token_loss.mean() / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            seen += 1
            if micro_step % args.grad_accum == 0 or micro_step == len(train_loader):
                if world > 1:
                    # ★ 手动梯度同步（见文件头 DDP 注释）。grad 为 None 的补零参与
                    #   ——保证所有 rank 的集合通信次数逐参数一致（单 rank 少一次
                    #   all_reduce = 死锁，opd.py 跳步改集体决定的同一课）
                    for p in model.parameters():
                        if p.requires_grad:
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step in sel_steps and rank == 0:
                    _save_selection_point(model, ROOT / args.out, sel_steps[global_step],
                                          global_step)
                elapsed = max(1e-6, time.time() - started)
                log({"train/loss": float(loss) * args.grad_accum,
                     # grad_norm 是最早能看出训练崩没崩的信号：突然飙高 = 有坏样本或 lr 过大
                     "train/grad_norm": float(grad_norm),
                     "train/lr": scheduler.get_last_lr()[0],
                     "train/epoch": epoch,
                     # ★ 下面四条是 2026-08-17 补的「健康度」，见 14-sft-health-metrics.md
                     "health/skipped_micro_steps": skipped,
                     "health/supervised_tokens": sup_tokens,
                     "perf/supervised_tokens_per_sec": sup_tokens / elapsed,
                     "perf/steps_per_sec":
                         (global_step - steps_per_epoch * start_epoch) / elapsed},
                    step=global_step)
            if prof is not None:
                prof.step()
                prof_left -= 1
                if prof_left <= 0:     # 抓够就收工，不给后面的步数留开销
                    prof.stop()
                    prof = None
        if world > 1:
            # epoch 均值跨 rank 汇总——否则打印的只是 rank0 那 1/world 份数据的 loss
            agg = torch.tensor([running, float(seen)], device=device)
            dist.all_reduce(agg)
            running, seen = float(agg[0]), int(agg[1])
        train_loss = running / max(1, seen)
        line = (f"[epoch {epoch}] train_loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"step={global_step}/{total_steps}  用时={time.time()-started:.0f}s")
        if rank == 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            stats = evaluate(model, val_set, device, pad_id, args.batch_size)
            line += f"  val_loss={stats['val_loss']:.4f}  ppl={stats['val_ppl']:.2f}"
            history.append({"epoch": epoch, "train_loss": train_loss, **stats})
            log({"val/loss": stats["val_loss"], "val/ppl": stats["val_ppl"],
                 **{f"val/loss_{g}": v["loss"] for g, v in stats["by_group"].items()},
                 **{f"val/ppl_{g}": v["ppl"] for g, v in stats["by_group"].items()}},
                step=global_step)
            print(line)
            # ★ 分组报出来，才看得到"某一类被牺牲了"
            for group, stat in stats["by_group"].items():
                print(f"          {group:<10} loss={stat['loss']:.4f}  ppl={stat['ppl']:.2f}")
        else:
            print(line)
        # ★★★ 位移：||ΔW||/||W||。**这是判断"到底训没训动"的唯一直接证据。**
        ratio = _delta_w_ratio()
        if world > 1:
            # ★ 每 epoch 硬断言：各 rank 权重必须一致。梯度 all_reduce 的结果全 rank
            #   相同 + 优化器状态起点相同 ⇒ 权重应逐位一致 ⇒ ratio 逐位一致。
            #   发散 = 手动同步失效（[silent-degradation-fsdp-nosync] 那类静默死法），
            #   宁可停机也不许带病训完。
            gathered = [torch.zeros(1, device=device) for _ in range(world)]
            dist.all_gather(gathered, torch.tensor([ratio], device=device))
            vals = [float(g) for g in gathered]
            spread = max(vals) - min(vals)
            assert spread < 1e-7, (
                f"🔴 rank 间权重发散：ΔW ratio spread={spread:.3e}（{vals}）"
                f"—— 梯度同步失效，停机检查")
        print(f"          ||ΔW||/||W|| = {ratio*100:.4f}%  "
              f"（正常 LoRA 0.5%–5%；M7 那次只有 0.0093% ⇒ 白训）")
        log({"health/delta_w_ratio": ratio,
             "health/epoch_seconds": time.time() - epoch_started}, step=global_step)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"          显存峰值 {peak:.1f} GB")
            log({"perf/peak_memory_gb": peak}, step=global_step)

        # ★ 每个 epoch 都存一份 adapter（几十 MB，便宜）。
        #
        # 因为**该选哪个 ckpt 不看 val loss**：手册 §20——SFT 训得越狠，输出熵越低，
        # 接上 GRPO 就探索不动了。我们已经踩过一次：选了 val loss 最低的那个，
        # 结果零梯度格子 63%。要选的是「格式学会了但行为还没定型」的那一版，
        # 而那一版只有在**每个 epoch 都存下来**的前提下才选得到。
        if rank == 0:
            epoch_dir = ROOT / args.out / f"epoch{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)

    out_dir = ROOT / args.out
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)          # LoRA 只存 adapter，几十 MB
        tokenizer.save_pretrained(out_dir)
        (out_dir / "training_history.json").write_text(
            json.dumps({"args": {**vars(args), "max_length": SFT_MAX_LENGTH},
                        "history": history}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[OK] adapter -> {out_dir}")
    sel_dirs = sorted(out_dir.glob("sel_f*"))
    if sel_dirs:
        total_mb = sum(f.stat().st_size for d in sel_dirs for f in d.glob("*")) / 1048576
        print(f"\n[选点] {len(sel_dirs)} 个临时产物，共 {total_mb:.0f} MB")
        print(f"       选完之后删掉未选中的："
              f"  python scripts/select_sft_ckpt.py {args.out} --keep <名字> --prune")
        print("       ⚠️ 别手动删 —— 手动的步骤一定会被忘（本项目第一失效形状）")
    if run is not None:
        run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
