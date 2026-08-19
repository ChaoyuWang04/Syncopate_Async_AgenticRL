# 提交包 · verl `use_prefix_grouper` 从未接上（+ 接上后会咬人的 mask 语义）

> **状态：材料齐备（掩码主打定位，2026-08-19 重写完成），等 Chaoyu 过目后提交。** 目标 **verl-project/verl**。
> 定位（考古后敲定）：**主打断点③掩码语义**（全网无人报过、打中所有 shared-prefix 路线）+
> 小 PR（掩码修复 + 死开关警告）+ 两条评论递给 MAGI 方向（#6689/#6401）。
> **刻意不提接线 PR** —— #7202 已为此被关（维护者转向 MAGI），接线走我们自己的本地补丁，
> 验证数据回头以评论补进 issue。历史链条与人物表见 analysis.md 顶部七条。
> 与包①（[`../verl-fsdp-size-1/`](../verl-fsdp-size-1/)）、包②（[`../verl-lora-adapter-sync/`](../verl-lora-adapter-sync/)）
> 是**同一个形状的第三例**：**配置项存在、工具函数齐全、文档也写了，唯独中间那根线没接。**

## 一句话

`actor.use_prefix_grouper=True` 在 verl 0.8.0 里**是个空开关**：
注意力的 monkey patch 永不执行（调用点不传这个参数），共享前缀的前向函数**零调用者**。
打开它既不会加速、也不会报错 —— 它唯一还在做的事是让同一个 GRPO 组别被拆到不同卡上。
⇒ 用户会量到「打开没收益」，从而**为错误的理由**放弃一个论文报告 training-equivalent 的优化。

**而且**：一旦把线接上，`prefix_grouper_utils.py` 传的 `suffix_mask` 是**梯度掩码**而不是
**存在掩码** ⇒ 在多轮工具场景下（含 verl 自带的 `tool_agent_loop`）会把
**工具 observation 的 token 从模型输入里静默删掉**。

## 文件清单

| 文件 | 是什么 |
|---|---|
| [`submission-EN.md`](submission-EN.md) | **issue + PR 英文正文**（GitHub 直接粘贴） |
| [`repro_prefix_grouper_wiring.py`](repro_prefix_grouper_wiring.py) | **零 GPU 复现**：三条判据（A 不传参 / B 零调用者 / C 掩码丢 token），全部 PASS |
| [`analysis.md`](analysis.md) | 中文分析与证据链（三处断点 · 为什么它比"慢"更坏 · 与包①② 的同构性 · 顶部十条考古） |
| [`SYNC-2026-08-19-fused-kernel-conflict.md`](SYNC-2026-08-19-fused-kernel-conflict.md) | infra 负责人的 fused-kernel 冲突同步 **+ 三条核实更正**（#7202 已同构解决投影问题；但它仍带掩码 bug） |

## 复现输出（`python repro_prefix_grouper_wiring.py`）

```
verl 0.8.0
[PASS] A. apply_monkey_patch() is never called with use_prefix_grouper
       verl/workers/engine/fsdp/transformer_impl.py:292
       kwargs=['model','use_remove_padding','ulysses_sp_size','use_fused_kernels','fused_kernels_backend']
[PASS] B. forward_micro_batch_with_prefix_grouper() has zero call sites
[PASS] C. passing response_mask silently drops tool-observation tokens
       packed with existence mask : [[1,2,3,4, 10,11,12,13,14,15, 20,21,22,23,24,25]]
       packed with response_mask  : [[1,2,3,4, 10,11,      14,15, 20,21,      24,25]]
       tokens dropped from input  : [12,13,22,23]   <-- the tool observations
```

## 我们自己的处置（不等上游）

三处补丁走 `syncopate/train/verl_patches.py`（与 E21/E22 同款 monkeypatch）：

```
① transformer_impl.py:292      把 use_prefix_grouper 传进 apply_monkey_patch
② forward_step / prepare_model_inputs   走 PrefixGrouper 的打包前向
③ prefix_grouper_utils.py      打包用「存在掩码」，算损失用「梯度掩码」
```
★ **验收判据是「开/关两条路的 log_probs 逐位相同」，不是「快了多少」** ——
因为 A/B 两条断点会让"快了多少"这个判据**为错误的理由通过**（测出 0 收益）。

## 提交前最后一眼

- [x] 零 GPU 复现三条全 PASS（infra 负责人跑过 + 2026-08-19 独立复跑确认）
- [x] main 新鲜度：`prefix_grouper_utils.py` 与 0.8.0 **逐字相同**、调用点仍缺参 —— 掩码 bug 在 main 上原样存在
- [x] submission-EN 按新定位重写（issue 掩码主打 · PR=掩码修复+警告 · 两条评论稿）
- [ ] 查 verl CONTRIBUTING（DCO / pre-commit / 测试目录惯例）
- [ ] （并行，不挡提交）本地接线补丁：**照 #7202 的做法**（`self.model(...)[0]` 拿隐状态 →
      `split_output` 在隐状态上切 → 只对 suffix 跑 `FusedLinearForPPO`；**flatten 必须在
      autograd Function 外**，否则静默丢隐状态梯度）+ rmpad 退出 A/B + logprob 逐位判据
      —— 完成后验证数据以评论补进 #7202 / issue
- [x] fused-kernel 冲突线核实完毕（SYNC 文档回复段）：根因成立、失败形状钉死（logits=None）；
      「更干净的修法」不成立（#7202 已同构）；**新增行动 = 去 #7202 评论**
- [x] 考古完成：#4368（kevssim，原集成）· #6067（切断点）· #7202（supercharleszhu 修复，被关）·
      #6689（arvyanh，MAGI draft）· wuxibin89（裁决人）—— 人物表见 analysis 顶部
