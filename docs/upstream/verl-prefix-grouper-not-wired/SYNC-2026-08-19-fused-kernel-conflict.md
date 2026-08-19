# 同步给上游负责人 · 一条**新的、独立的**不兼容（2026-08-19）

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
⇒ 新增一条行动（已写进 submission-EN §3）：**去 #7202 下评论**，把掩码这条直接递给作者本人 ——
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
