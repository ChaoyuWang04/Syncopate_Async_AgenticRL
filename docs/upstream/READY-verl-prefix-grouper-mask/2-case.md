# Case · verl `use_prefix_grouper` 的掩码语义 bug（E26 / E25 §5）

```
状态    READY —— **正文 + 分支 + 测试全部备齐（2026-08-21）**，等 Chaoyu 提交
目标    verl-project/verl（bug report + 小 PR：两处掩码修复 + 死开关告警，刻意不含接线）
分支    /workspace/_upstream/verl → fix/prefix-grouper-pack-with-existence-mask @ 03b9a91
        （基于上游 main 4905d0c，DCO 已签）⬜ **未推**
issue/PR 未提交
定位    ⛔ 「从未接上」是错的 —— 真相是 #6067 重构回归；#7202 已交修复被维护者关闭
        （转向 MAGI #6689：draft / old_log_prob diverge / Megatron 向）
        ★ 我们独有的是**休眠代码里的三条静默错误**（C/D/E，全网无人报过；#7202 的复活版仍带着 C 和 D）
验证    复现脚本 5/5 PASS（对 0.8.0 与今日 main 各跑一遍）
        单测 修前 3 failed（红在**行为**：丢 token / prefix_len 7≠4 / 返回梯度掩码）→ 修后 4 passed
        告警单测 修前红（"got []"）→ 修后绿 · pre-commit 14 钩子全过
        ⚠️ 掩码单测在 CI 里会 **skip**（prefix-grouper 不是 verl 依赖，也不在 uv.lock）——
           这句话已明写进 PR 正文，不许含糊过去
风险    同一维护者（wuxibin89）关掉过 #7202，也以 "rare case" 关掉了我们的包①
        ⇒ 本包刻意只提「护栏」不提「接线」；**E（后端支持列表）只进 issue 不进 PR**
```

发现来源 [`E26-prefix-grouper.md`](../../archive/infra_exp/legacy-4x5090/E26-prefix-grouper.md) ·
提交件 [`3-submission.md`](3-submission.md) · 零 GPU 复现 `repro_prefix_grouper_wiring.py`

---

> 2026-08-19。英文提交正文 → [`3-submission.md`](3-submission.md)。
> 实验背景 → [`E25-trainer-feed.md §5`](../../archive/infra_exp/legacy-4x5090/E25-trainer-feed.md)。

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
> 🆕 **8/19 晚追加三条**（读 #7202 实际 diff，详见本文末尾「附 · fused kernel 冲突线」）：
> 8. ★ **#7202 的复活版仍带掩码 bug**：它对 `prefix_grouper_utils.py` 的 diff 只改 nested→padded
>    与 uid 解包，**`suffix_mask=response_mask` 一字未动** ⇒ 维护者一旦重开它，工具 observation
>    照样被静默删掉。**⇒ 新增行动：直接去 #7202 下评论递给作者本人。**
> 9. **#4368 时代还有一个"静默无效"的平行版本**：它的门槛是
>    `can_use_pg = … and not self.use_fused_kernels` ⇒ 在 PG 真正接着的那半年里，
>    **开着融合算子的用户默默拿不到 PG**（verl 默认 False，但它是推荐的性能项，我们默认 True）。
>    ⇒ 同一个功能，**在两个时代因两个不同原因静默失效**。
> 10. `use_fused_kernels` × PG 的失败形状比"不返回 logits"更毒：`CausalLMOutputForPPO.logits`
>    **存在但为 None** ⇒ 不是 AttributeError，而是 None 流进 `split_output` 后炸在无关的地方。

## ★ 提交的是三条，不是一条（2026-08-21 重新定范围）

E26 里我们实际找到六处问题，其中**三条是休眠代码里的静默错误、且全网无人报过**。
此前的提交件只报了 C 一条 —— 那等于把「我们是唯一真正把这条路跑起来过的人」这个最硬的资本丢掉三分之二。

| | 内容 | 判据（复现脚本里的编号） | 去处 |
|---|---|---|---|
| **C** | `suffix_mask` 拿到的是**梯度**掩码 ⇒ 多轮下工具 observation 被从模型输入里删掉 | check C：`[12,13,22,23]` 被丢 | issue + **PR** |
| **D** | 前缀掩码靠 `ne(pad_token_id)` 猜，而那个 id 由 `micro_batch.get("pad_token_id", 0)` 解析 —— **这个键从来进不了训练批**（同仓库 engine 路径用的是 `get_non_tensor_data`）⇒ 任何 pad id≠0 的模型，**prompt 的 padding 会被打包进共享前缀** | check D：`ne(0)` 数出 5 个「真」token，attention_mask 说只有 3 个 | issue + **PR** |
| **E** | wrapper 把 PrefixGrouper 的 **2D padding 掩码**当 HF 的 `attention_mask` 递下去。suffix 子调用 `q_len=R` 而 `k_len=P+R`：sdpa 直接**报错**；把掩码丢掉也救不了 —— torch 的 `is_causal=True` 在 `q_len≠k_len` 时是**左上对齐**（每个回答 token 只看得见前 t+1 个**题面** token）。只有 flash-attn 的 `causal=True` 是右下对齐，而 `_PREFIX_GROUPER_SUPPORTED_ATTENTIONS` 把 `sdpa`/`eager`/`flex_attention` 也列成支持 | check E：`|sdpa(is_causal) − 左上| = 0.000e+00`、`− 右下| = 9.136e-01`；可见 key 数 `[1,2,3]` vs 正确 `[5,6,7]` | **只进 issue** |

⇒ **E 不进 PR** 的理由：它要改 `_create_prefix_grouper_wrapper` 的行为，属于接线范畴 ——
而接线正是 #7202 被关的原因。C/D 是「打包用哪个掩码」，一个函数、十几行，和后端选型无关。

⇒ D 还额外坐实一条：`prefix_grouper` **根本不是 verl 的依赖**（不在 pyproject、不在 uv.lock），
`prefix_grouper_utils.py` 也**零 importer** —— 这条路径是完整孤立的，所以上面三条谁都没撞见过。

---

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


---

## 附 · fused kernel 冲突线（infra 负责人同步 + 三条核实更正）

> 承接你那段同步。**定位反转我接受**（重构回归 / #7202 已撞墙 / 只有掩码那条是我们独有的）。
> 这份只补一件你可能还没有的事，**它不改变掩码那条 issue 的任何内容**。

## 结论先行

你给的 (a)「裸接线会把整个 grouped 序列过 vocab 头 ⇒ 我们的形状 ≈11.5 GB ⇒ 必 OOM」——
**成立，我独立算了：9,428 token × 151,936，bf16 2.67 GB / fp32+梯度 10.67 GB。**

但根因比"logits 太大"更具体，而且是**两个 verl 特性之间的不兼容**：

```
use_fused_kernels=True   把模型的 forward **整个替换**成 dense_common.forward_with_torch_backend
                         ⇒ 它内部走 FusedLinearForPPO（分块投影 + logprob，**从不物化 [T,V]**）
                         ⇒ 它**不返回 logits**
prefix_grouper_utils     pg_forward() 第一行是 `logits = model(...).logits`
                         ⇒ 它**要求** logits，因为它要在 logits 上做 split_output
⇒ 两者互斥。开着 fused kernels 接 prefix grouper，等于把我们**本来已经避开**的账重新引回来。
```

⇒ 也就是说 (a) 不只是"要抄 response-only 投影"，而是 **prefix grouper 目前的实现与
`use_fused_kernels` 结构性冲突** —— 这条在 #4368 / #7202 / MAGI 三条路线上都成立，
因为它们都要在**投影之后**做切分。

## 我们的修法（可能比只做 response-only 更干净，供你判断要不要带给 supercharleszhu）

**把切分点从 logits 之后挪到隐状态之后**，两个机制就不再打架：

```
现状（OOM）   模型 → logits [9,428 × 151,936] → split_output → 取 suffix → logprob
修法          模型 → **hidden** [9,428 × 2,560]（48 MB，小 59×）→ split_output
                   → 只对 suffix 的 5,232 个位置跑 FusedLinearForPPO（分块、不物化）
```

两个收益叠加：**投影的 token 数 9,428 → 5,232（−45%）**，且**融合算子照用**。
实现只需要 `output_hidden_states=True` + 在 suffix 上调 verl 自己的 `FusedLinearForPPO`。

⚠️ **状态：已写进我们的 monkeypatch，但 GPU 被别的实验占着，一行都还没在卡上跑过。**
两个待验前提（都已写成断言，失败会当场炸而不是静默退回）：
① 融合版 forward 认不认 `output_hidden_states=True`；② `prefix_grouper=` 能否顺 `**loss_kwargs` 流到注意力层。
⇒ **验完之前请不要把这段当成已验证的方案转给上游。**

## 对已有提交件的影响

```
掩码那条 issue（我们独有的）   **无影响** —— 它管的是"哪些 token 进模型输入"，
                               在打包这一步；本条管的是"打包之后怎么投影"，两者正交、可叠加
接线那条（已挂起）             无影响
```

⇒ 如果你要在掩码 issue 里提一句，建议只加一行：
「注意本修复位于 packing 阶段，与 projection 阶段的 OOM 问题（#7202 提到的）互相独立。」

## 一条方法论，可能值得一起带上去

三条路线都在**投影之后**切分。而"在投影之后切分"这个选择本身，就把
`use_fused_kernels` 这条路堵死了 —— **没有人报过这一点，因为它表现为 OOM 而不是错误结果**，
而 OOM 会被读成"显存不够，换大卡/减 batch"，不会被读成"接入点选错了"。

---

# ⛔⛔ 回复 · 三条核实结果（2026-08-19 晚，读 #7202 的实际 diff 之后）

> **你的根因诊断成立且更精确了；但两条推论被 #7202 的代码推翻。**
> 纪律：原文一字不删，更正写在下面。

## ✅ 成立（并且我把失败形状钉得更死了一层）

`use_fused_kernels=True` 与 `pg_forward()` 结构性互斥 —— 源码逐处核对：

```
forward_with_torch_backend（dense_common.py）
    hidden_states = forward_base_model(...)[0] → FusedLinearForPPO（分块，不物化 [T,V]）
    return CausalLMOutputForPPO(log_probs=…, entropy=…, hidden_states=…, …)
                                 ↑ **没有 logits=**
pg_forward()（prefix_grouper_utils.py:118）
    logits = model(...).logits          ← 要 logits
```

★ **精确的失败形状**：`CausalLMOutputForPPO` 继承 `CausalLMOutputWithPast`
⇒ `.logits` 字段**存在但为 `None`**（实测 `hasattr=True, value=None`）
⇒ **不是 AttributeError**，而是 `None` 一路流进 `split_output(None, …)`，
炸在一个与根因无关的地方。⇒ 这条比"它不返回 logits"更该写进 issue：
**又一个静默/误导型失败**。

我们的形状复核你的账：prefix 4196 + 8×654 = **9,428 token** × 151,936
= bf16 2.67 GiB / fp32+grad 10.67 GiB —— 与你一致（我先前口算的 11.5 GB 是按 2 组算的，
按"每 micro-batch 一组"应以你的数为准）。

## ⛔ 更正① ·「三条路线都在投影之后切分」—— 对 #7202 **不成立**

#7202 的 `monkey_patch.py` 新增 `apply_prefix_grouper_model_forward_patch`，实际做法：

```python
hidden_states = self.model(*args, **kwargs)[0]                    # ← 基座模型，拿隐状态
_, _, suffix_hidden_raw, suffix_mask_raw = prefix_grouper.split_output(hidden_states, …)   # ← 在隐状态上切
suffix_hidden = suffix_hidden_raw[:, :-1]
flat_hidden = suffix_hidden.reshape(-1, hidden_size)
→ FusedLinearForPPO(hidden_states=…, vocab_weights=self.weight, input_ids=…)  # 只对 suffix 投影
```

⇒ **它恰恰不在投影之后切分。** 三条路线的真实状态是：

```
#4368   ⚠️ 静默避开：can_use_pg = … and not use_fused_kernels ⇒ 开着融合算子的用户**默默拿不到 PG**
        （= 同一个"静默无效"形状的**另一个时代版本**，值得写进 analysis）
#7202   ✅ 已解决：隐状态切分 + response-only 融合投影；并对 use_fused_kernels=True **硬报错**
        （"PrefixGrouper performs its own response-only fused LM-head projection; set use_fused_kernels=False"）
        ⇒ 它的处理方式是 **PG 吃掉融合算子的职责**，而不是共存
MAGI    未核（Megatron/TE 链，另一套投影路径）⇒ 这条留作待查，别写进 issue
```

## ⛔ 更正② ·「我们的修法可能更干净，可以带给 supercharleszhu」—— 他已经做了，且**逐行同构**

你提的「切分点从 logits 之后挪到隐状态之后 + 只对 suffix 跑 `FusedLinearForPPO`」，
与 #7202 上面那段**是同一个设计**。⇒ 这不是可以带过去的新东西。

★ **但这件事的价值没有消失，只是换了个位置**：**独立收敛**本身就是证据 ——
我们在没读他 diff 的情况下推出了同一个设计，说明那是这条路上的**唯一自然解**。
⇒ 该说的话从"我有个更好的方案"变成 **"你的设计我们独立复现了，并且我们要在它上面加一块你缺的"**。

## ★ 真正的新东西（这条才是要带给 supercharleszhu 的）

**#7202 的复活版仍然带着掩码 bug。** 逐行查过它对 `prefix_grouper_utils.py` 的 diff：
只改了 nested→padded 转换和 uid 解包，**`suffix_mask=response_mask` 那行一字未动**。

⇒ 也就是说：**维护者一旦重开 #7202，多轮工具场景下工具 observation 照样会被静默删掉。**
⇒ 新增一条行动（已写进 [`3-submission.md`](3-submission.md) §3）：**去 #7202 下评论**，把掩码这条直接递给作者本人 ——
他是最可能复活这条线的人，而我们补的正好是他缺的那块。

## 🎁 白捡的实现情报（给我们自己的本地补丁，能省一次调试）

#7202 在同一段里留了个注释，是他踩过的坑：

> *"Flatten outside the custom autograd Function. Flattening inside its forward
> runs under no_grad and drops the hidden-state gradient."*

⇒ `reshape(-1, hidden)` 必须在 `FusedLinearForPPO` **外面**做，否则隐状态的梯度会被静默丢掉
（又一个"不报错、只是训歪"）。我们照抄即可，**别自己再踩一遍**。

⇒ 另：你标的两个待验前提里，①（融合 forward 认不认 `output_hidden_states=True`）
在 #7202 的做法下**可以绕开** —— 它直接调 `self.model(...)` 拿最后一层隐状态，
不走 `output_hidden_states=True`（那个会物化**全部 36 层**，≈1.7 GB 白付）。**建议照 #7202 改。**

## 对已有提交件的影响（与你的判断一致）

```
掩码 issue     无影响，且**更强了** —— 现在多一条"#7202 的复活版仍带此 bug"
接线（挂起）    无影响；但本地补丁的设计**改为对齐 #7202**（隐状态切分 + response-only 投影
               + flatten 在外），不再自创
你写的"GPU 上一行没跑过"的免责    **保留**，仍然适用于我们自己的补丁
```
