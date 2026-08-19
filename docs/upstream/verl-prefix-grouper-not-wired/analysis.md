# 分析 · `use_prefix_grouper` 的三处断点（中文，证据链）

> 2026-08-19。英文提交正文 → [`submission-EN.md`](submission-EN.md)。
> 实验背景 → [`../../infra_exp/E25-trainer-feed.md §5`](../../infra_exp/E25-trainer-feed.md)。

> 🆕🆕 **2026-08-19 · 上游考古完成，七条 —— ⛔ 本包定位要整个换**：
>
> 1. ⛔ **「从未接上」是错的（保留示错）**：它接上过、还跑过 benchmark ——
>    [#4368](https://github.com/verl-project/verl/pull/4368)（2026-01-05 合入，kevssim）
>    在旧 `dp_actor.py` 里接线，实测 update_actor **1.26–1.70×**（Qwen3-4B，n=4，4/8K）。
>    真相是**重构回归**：[#6067] 把 `dp_actor.py` 删了、engine 新路径没消费这个开关。
> 2. **断点①② 上游已知**：[#7202](https://github.com/verl-project/verl/pull/7202)
>    （2026-07-30，supercharleszhu，+604）原话 *"the option became a **silent no-op**"*，
>    并交了完整修复（重接 engine 边界 + FSDP2 钩子 + **response-only LM-head 投影防 OOM**）。
> 3. **#7202 被维护者关闭未合**：wuxibin89 —— *"PrefixGrouper has some limitation,
>    we're exploring MAGI attention #6689"* → *"Prefer to close this PR"*。
> 4. **#6689（官方指定方向）的现实**：**draft**、+5116 行、两个月 4 条评论、
>    自述 *"old_log_prob seems to **diverge**"*（正确性未闭合）、注入的是
>    **Megatron TE attention 链** —— FSDP 路径覆盖不明。⇒ 短中期内 FSDP 用户没有出路。
> 5. **main 至今仍是 silent no-op**（今日源码核对：`transformer_impl.py:305` 还是那 5 个参数）。
> 6. ★ **断点③（掩码语义）全网空白 —— 我们独有**：verl issues、PrefixGrouper 原仓库
>    （已迁 CASIA-IVA-Lab/PrefixGrouper，2025-06-13 起休眠，仅 2 条 issue）都没人报过。
>    它同时打中 #4368 原设计、#7202 的复活版、以及 #6401/#6689 的 prefix-tree 方向
>    （多轮掩码正是那条 RFC 的主场）。**这是本包对上游的核心增量。**
> 7. ⚠️ **#7202 的 OOM 教训直接约束我们自己的补丁**：原版 helper 把整个 grouped 序列过
>    vocab 级 LM head —— 按我们的形状（~18.9k grouped token/卡 × 151,936 vocab × fp32
>    ≈ 11.5 GB 仅 logits），32 GB 的 5090 上**裸接线必 OOM** ⇒ 必须带 response-only 投影。
>    另：原 #4368 的门槛要求 `not use_remove_padding`，而我们 rmpad=True ⇒ PG 前向要
>    退出 rmpad，净收益需 A/B，不能拿 4.1× 上界当预期。
>
> **人物表**：kevssim（原作者）· supercharleszhu（修复者，天然盟友，另有 #7292 ref-KV-cache wip）·
> wuxibin89（裁决维护者）· arvyanh（#6401 RFC + #6689 MAGI）。
>
> 🆕 **8/19 晚追加三条**（读 #7202 实际 diff，详见 [`SYNC-2026-08-19-fused-kernel-conflict.md`](SYNC-2026-08-19-fused-kernel-conflict.md) 的回复段）：
> 8. ★ **#7202 的复活版仍带掩码 bug**：它对 `prefix_grouper_utils.py` 的 diff 只改 nested→padded
>    与 uid 解包，**`suffix_mask=response_mask` 一字未动** ⇒ 维护者一旦重开它，工具 observation
>    照样被静默删掉。**⇒ 新增行动：直接去 #7202 下评论递给作者本人。**
> 9. **#4368 时代还有一个"静默无效"的平行版本**：它的门槛是
>    `can_use_pg = … and not self.use_fused_kernels` ⇒ 在 PG 真正接着的那半年里，
>    **开着融合算子的用户默默拿不到 PG**（verl 默认 False，但它是推荐的性能项，我们默认 True）。
>    ⇒ 同一个功能，**在两个时代因两个不同原因静默失效**。
> 10. `use_fused_kernels` × PG 的失败形状比"不返回 logits"更毒：`CausalLMOutputForPPO.logits`
>    **存在但为 None** ⇒ 不是 AttributeError，而是 None 流进 `split_output` 后炸在无关的地方。

## 0 · 我们为什么会去看它

E25 证伪了「trainer 没喂饱」（`micro_batch` 拉高是负收益、关梯度检查点显存不够）
⇒ 省时间只剩「**让它少算**」。而我们的负载里最刺眼的一笔重复劳动是：

```
一道题的题面 4196 token，一组采样 8 条
rollout 侧（vLLM 开着 prefix caching）  题面只算 **1 次**
trainer 侧                              题面要算 **8 × 3 = 24 次**（8 条样本 × 三次前向）
⇒ 每条序列 87% 的 token 是共享题面 ⇒ 理论上界 4.1×
```

PrefixGrouper（arXiv 2506.05433）正是干这件事的，而且论文证明它**训练等价**
（前向输出与反向梯度逐位相同）。verl 也确实有这个开关。

## 1 · 三处断点

### 断点① · 注意力的 patch 永不执行

```
verl/models/transformers/monkey_patch.py:324
    if use_prefix_grouper:
        apply_prefix_grouper_patch()      # 包装 ALL_ATTENTION_FUNCTIONS

verl/workers/engine/fsdp/transformer_impl.py:292   ← 全 verl **唯一**的调用点
    apply_monkey_patch(model=..., use_remove_padding=..., ulysses_sp_size=...,
                       use_fused_kernels=..., fused_kernels_backend=...)
    ⇒ 🔴 没有 use_prefix_grouper ⇒ 取默认值 False
```

### 断点② · 打包前向零调用者

```
verl/trainer/ppo/prefix_grouper_utils.py
    build_pg_from_micro_batch()                     模块外引用 0 次
    forward_micro_batch_with_prefix_grouper()       模块外引用 0 次

实际走的路（永远）：
    transformer_impl.py:1253  forward_step()
        → prepare_model_inputs()
        → self.module(**model_inputs, use_cache=False)
        → prepare_model_outputs()          ← 标准 rmpad 路径
```

### 断点③ · `suffix_mask` 拿到的是梯度掩码（今天不咬人，修好①② 之后咬）

包里的语义没有歧义：

```python
prefix_grouper/__init__.py:77    suffix_lens = suffix_mask.sum(dim=1)     # 每条响应多长
prefix_grouper/__init__.py:196   suffix_mask.nonzero(as_tuple=False)      # 哪些位置进入拼接
⇒ suffix_mask = 「这个 token **存不存在**」
```

而 verl 传的是 `response_mask`，它在多轮里是**梯度掩码**——
**verl 自带的 agent loop 就是这个语义**，不是下游用户的自定义行为：

```python
verl/experimental/agent_loop/tool_agent_loop.py:262   response_mask += [1]*n   # 模型生成 → 有梯度
verl/experimental/agent_loop/tool_agent_loop.py:400   response_mask += [0]*n   # 工具返回 → 无梯度
```

⇒ 复现脚本 check C 的实测输出：

```
打包用「存在掩码」 [[1,2,3,4, 10,11,12,13,14,15, 20,21,22,23,24,25]]
打包用 response_mask [[1,2,3,4, 10,11,      14,15, 20,21,      24,25]]
被丢掉的 token       [12,13,22,23]   ← 正是工具 observation
```

⇒ 修法需要**两个掩码**：打包用 `attention_mask[:, prompt_len:]`，损失用 `response_mask`。
而 `pg_forward()` 现在把 `split_output()` 回来的**同一个掩码**当成了两者。

## 2 · ★ 为什么这比「慢」更坏：它会让判据**为错误的理由通过**

一个用户的自然做法是：打开开关 → 量前后墙钟 → 没变化 → **结论「PrefixGrouper 对我没用」**。

这个结论**看起来是干净的实验得出的**，而它是错的。
⇒ 这正是本项目记过的失效形状：
[[blank-thresholds-are-not-passes]]「判据太宽会为错误的理由通过」 +
[[project-mechanism-not-wired]]「机制在但没接上」。

⇒ 所以我们给自己定的验收判据是：
**先验「这条路径真的被走到了」（在 `pg_forward` 里打一行判据），再谈快慢。**
而等价性判据是 **`log_probs` 逐位相同**，不是「快了多少」。

## 3 · 与包①② 的同构性（这是第三例了）

| | 配置意图 | 实际发生 | 信号 |
|---|---|---|---|
| 包① `fsdp_size=1` | 「不要分片」 | 降级成 NO_SHARD，但归约留在大小为 1 的组里 ⇒ 梯度不同步 | 一行 `UserWarning` |
| 包② LoRA adapter sync | 「把新策略推给 rollout」 | 只跑了两段式协议的前半段 ⇒ 永远推冻结基座 | **无** |
| **本包** `use_prefix_grouper` | 「用共享前缀前向」 | patch 不执行 + 前向零调用者 ⇒ 只剩批次划分 | **无** |

★ 三次都是：**两端的能力都在，断的是中间没有传参的那一栏。**
⇒ 一般化：**「配置项被接受」离「功能被执行」之间，至少还隔着一次传参。**
⇒ 工程上的对策：**凡是靠布尔开关生效的功能，都要在生效点打一行判据**
（我们自己的 `SYNCOPATE_*` 探针全部遵守这条）。

## 4 · 我们自己的处置

不等上游。三处补丁走 `syncopate/train/verl_patches.py`（与 E21/E22 同款）：

```
① 让 apply_monkey_patch 收到 use_prefix_grouper
② forward_step 在开关打开时走打包前向
③ 打包用存在掩码、损失用梯度掩码
```

⇒ 验收：`log_probs` 逐位相同（同批数据同权重，开/关两条路）→ 再量吞吐 → 再过任务尺子。
