"""E12-c 微基准：decoupled 的 proximal anchor 快照到底多贵，以及只存 LoRA 能省多少。

背景（读码查实，2026-08-14）：
  fully_async 的 `_compute_old_log_prob`（fully_async_trainer.py:465）不只是一次前向：

      local_trigger_step == 1:   save_model_to_cpu(1)                      → 1 趟 D2H
      local_trigger_step >= 2:   save_model_to_cpu(cur)                    → 1 趟 D2H
                                 restore_model_from_cpu(1)                 → 1 趟 H2D
                                 <前向>
                                 restore_model_from_cpu(cur)               → 1 趟 H2D
                                 clear_cpu_model(cur)

  而 `ddp_save_to_cpu`（我们自己的 verl_patches.py:93）遍历的是 **model.named_parameters()**
  —— 对 LoRA 模型来说，这包含**冻结的 4B 基座**，不只是 66M 的 adapter。

★ 预测（跑之前写死）：
  H1  全量快照（~8 GB bf16、数百个张量、非 pinned）单趟 **2–8 s**
  H2  `--sync-every 4` 下平均每步 (3×24GB + 1×8GB)/4 ⇒ 单步搬运 **9–25 s**
      ⇒ 能解释 old_log_prob(76.3s) 与 ref(39.2s) 之间 **37 s** 的缺口的大部分
  H3  只存/恢复 `requires_grad=True` 的参数（LoRA 132 MB）⇒ **省 50× 以上**，单趟 <0.2 s
      依据：基座是**冻结**的，跨版本逐字节相同，**根本不需要存**

  如果我错了会怎样：
  - H2 若远小于 37 s ⇒ 缺口另有来源（回 E01 用 nsys 查），本优化收益下调
  - H3 若不成立（例如 LoRA 参数在 named_parameters 里带着 base 的引用）⇒ 方案要改
"""

from __future__ import annotations

import statistics as st
import time

import torch

MODEL = "models/Qwen3-4B-sft-v11-e1"
LORA_RANK = 32


def timeit(fn, reps: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append(time.perf_counter() - t0)
    return st.median(xs)


def save_all(model):
    """= verl_patches.ddp_save_to_cpu：遍历全部 named_parameters。"""
    return {n: (p.detach().to("cpu", copy=True), None) for n, p in model.named_parameters()}


def save_trainable(model):
    """只存可训练参数（LoRA）。基座冻结、跨版本逐字节相同，不必存。"""
    return {n: (p.detach().to("cpu", copy=True), None)
            for n, p in model.named_parameters() if p.requires_grad}


def load_back(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n][0].to(p.device, non_blocking=True))


def main() -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print("加载 Qwen3-4B + LoRA r32 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2, target_modules="all-linear",
        lora_dropout=0.0, task_type="CAUSAL_LM"))

    allp = list(model.named_parameters())
    trainable = [(n, p) for n, p in allp if p.requires_grad]
    nb_all = sum(p.numel() * p.element_size() for _, p in allp)
    nb_tr = sum(p.numel() * p.element_size() for _, p in trainable)
    print(f"\n全部参数    {len(allp):>5} 个张量   {nb_all/1e9:>7.3f} GB")
    print(f"可训练(LoRA) {len(trainable):>5} 个张量   {nb_tr/1e9:>7.3f} GB   "
          f"占 {nb_tr/nb_all*100:.2f}%\n")

    st_all = save_all(model)
    st_tr = save_trainable(model)

    t_save_all = timeit(lambda: save_all(model))
    t_save_tr = timeit(lambda: save_trainable(model))
    t_load_all = timeit(lambda: load_back(model, st_all))
    t_load_tr = timeit(lambda: load_back(model, st_tr))

    print(f"{'操作':<34}{'全量(s)':>10}{'仅LoRA(s)':>11}{'加速':>9}")
    print(f"{'save_model_to_cpu  (D2H)':<34}{t_save_all:>10.3f}{t_save_tr:>11.3f}{t_save_all/max(t_save_tr,1e-9):>8.1f}×")
    print(f"{'restore_model_from_cpu (H2D)':<34}{t_load_all:>10.3f}{t_load_tr:>11.3f}{t_load_all/max(t_load_tr,1e-9):>8.1f}×")

    # 按 fully_async 的实际调用序列换算每步成本
    heavy = t_save_all + 2 * t_load_all          # local_trigger_step >= 2
    light = t_save_all                            # local_trigger_step == 1
    per_step_all = (3 * heavy + light) / 4        # --sync-every 4
    heavy_tr = t_save_tr + 2 * t_load_tr
    per_step_tr = (3 * heavy_tr + t_save_tr) / 4
    print(f"\n按 --sync-every 4 的实际调用序列换算：")
    print(f"  重步 (3/4 步: save+2×restore)   全量 {heavy:>7.2f} s   仅LoRA {heavy_tr:>6.3f} s")
    print(f"  轻步 (1/4 步: 只 save)          全量 {light:>7.2f} s   仅LoRA {t_save_tr:>6.3f} s")
    print(f"  ★ 平均每步搬运开销              全量 {per_step_all:>7.2f} s   仅LoRA {per_step_tr:>6.3f} s")
    print(f"  ★ 可省                          {per_step_all - per_step_tr:>7.2f} s/步")
    print(f"\n对照：M7 实测 old_log_prob 76.3 s vs ref 39.2 s，缺口 37.1 s；step 合计 296.4 s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
