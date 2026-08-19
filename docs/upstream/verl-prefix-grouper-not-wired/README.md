# 提交包 · verl `use_prefix_grouper` 从未接上（+ 接上后会咬人的 mask 语义）

> **状态：⛔ 定位待重写（2026-08-19 考古后）—— 证据全部成立，但「从未接上」这个框架是错的。**
> 真相：#4368 合入过且有 benchmark（1.26–1.70×）→ #6067 引擎重构静默切断 → #7202 已交修复被
> 维护者关闭（转向 MAGI #6689，draft/未闭合/Megatron 向）→ **main 至今 silent no-op**。
> **我们的独有增量 = 断点③掩码语义（全网无人报过）+ 零 GPU 复现。** 详见 analysis.md 顶部七条。
> submission-EN 按旧框架写成，**重写前不要提交**。目标 **verl-project/verl**。
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
| [`analysis.md`](analysis.md) | 中文分析与证据链（三处断点 · 为什么它比"慢"更坏 · 与包①② 的同构性） |

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

- [x] 零 GPU 复现三条全 PASS（`repro_prefix_grouper_wiring.py`）
- [ ] 端到端验证：我们的三处补丁跑通 + logprob 逐位相同（做完回填数字）
- [ ] 查 verl CONTRIBUTING（DCO / pre-commit / 测试目录惯例）
- [x] 考古完成：#4368（kevssim，原集成）· #6067（切断点）· #7202（supercharleszhu 修复，被关）·
      #6689（arvyanh，MAGI draft）· wuxibin89（裁决人）—— 人物表见 analysis 顶部
- [ ] ⛔ submission-EN 按新定位重写（回归而非从未接上；掩码发现为主打；引用 #7202/#6689）
- [ ] 我们自己的补丁计划补上 #7202 的 response-only LM-head 投影（裸接线在 5090 上必 OOM）+
      rmpad 退出的净收益 A/B
