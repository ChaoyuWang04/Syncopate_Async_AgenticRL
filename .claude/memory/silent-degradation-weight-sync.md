---
name: silent-degradation-weight-sync
description: 异步模式下 LoRA 从没被推给 rollout —— 生成数据的策略两个月没变过；"验了耗时没验内容"
metadata:
  node_type: memory
  type: project
---

**2026-08-18（E22）**：`fully_async` / `one_step_off` 下，每次权重同步推给 vLLM 的都是
**未经修改的冻结基座** —— LoRA adapter **一个字节都没推过去**。
⇒ **rollout 永远用起点策略 π₀ 采样，整条 RL 回路是断开的。我们从没跑过一次正确的异步 RL。**

```
判据   推出去的 q_proj.base_layer.weight ‖W‖=75.377708，与磁盘上起点模型**逐位相同**
       4 跑 × 2 次同步全一致；含 lora_ 的张量 0 个；载荷 8,414 MiB = 完整基座
根因   engine_workers.py:698 disaggregated 分支**只调一次** get_per_tensor_param() 且不传参
       ⇒ base_sync_done=False ⇒ collect_lora_params **显式跳过所有 lora_**
       （colocate 调两次，先基座后 adapter，**是对的**）
止血   --lora-merge（bf16 合并）⇒ ⛔ **已否决**：logprob 偏移中位 1.717e-02
       = **adapter 自身作用的 50%**，是引擎地板的 50×
修法   ✅ **自己把 verl 缺的那段管子接上了**（默认开）：trainer 侧首次送基座、之后送 adapter；
       rollout 侧带上"这是 adapter"的标记 ⇒ TensorLoRARequest + add_lora
       —— **两端能力本来都在，断的只是中间没有传参那一栏**
验证   list_loras() []→[123] · 载荷 8,414→252 MiB · kl 回地板 · param_sync 6.25→0.974 s
       数值：两侧 scaling 都是 2.0 · log_ppl_diff 落在同版本地板 ~3.4e-4
⇒ **异步 RL 第一次真正跑通**（2026-08-18）。整条故事见 docs/infra_exp/STORY-async-lora-weight-sync.md
```

**Why**：这条比 [[silent-degradation-fsdp-nosync]] 影响更大，而且形状更毒 ——
**它制造了一整套自洽的错误解释**：`kl` 单调涨、ESS 随位移掉、陈旧度旋钮不敏感，
全都"看起来像陈旧度"，我们据此追了两个月。
**我们的份**：E12 花一整轮研究「权重同步为什么慢」，**整份建立在"只推 132 MB"这个算出来的前提上**，
而它自己记下的反常「8 GB 与 132 MB 同耗时」正是 bug 在敲门 ⇒ 又一次 [[feedback-measure-dont-infer]]。

**How to apply**：
1. **凡是"把 X 送到 Y"的机制，验过耗时不等于验过内容。** 判据要写成
   「送过去的那一份 == 手上的那一份」，而不是「送成功了 / 没 OOM / 耗时多少」。
2. **"逐次不变"本身就是判据**：正在训练的策略不可能每次同步都逐位相同。
3. **「推了」≠「推对了」**：张量数对、大小对、槽位非空，都只证明"推的是那类东西"。
   ⇒ 端到端判据要用 **logprob 对比**（vLLM 算的 vs trainer 算的，同一批 token）——
   scaling 错 / 模块漏装 / 值损坏，任何一项发生它都会远离地板。
4. **默认值必须是百分之百对的那个来兜底**，而"对"不能依赖第三方的默认值不变。
5. 同一个开关在两条代码路径下语义不同、而默认值只对其中一条正确 ——
   这是 [[project-mechanism-not-wired]] 的又一形态：**机制在、接上了、但接到了另一条路径的语义上**。
