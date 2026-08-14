---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-INFRA-HANDOFF.md
metadata: 
  node_type: memory
  type: project
  originSessionId: d8054c42-ce87-481d-a266-b7806a058358
  modified: 2026-08-14T06:04:18.794Z
---

infra 线与主线训练**分开交接**：主线看 `docs/syncopate/05-handoff.md`，
infra 线看 **`docs/infra_exp/00-INFRA-HANDOFF.md`**（2026-08-13 关机前写，含下一步排序）。

**已定决策（2026-08-13，Chaoyu 批准，别重新讨论）**：
- 框架 **verl 不换**（抛开沉没成本重选仍是它；论证在 E07 §1）
- MoE 线：**GLM-4.7-Flash 30B-A3B + LoRA + GSPO**，三摆法对照
  （FSDP 分片 / QLoRA 4bit 复制 / EP toy→Megatron 探针），先跑探针 P1–P6
- 实验以 E 编号报告组织（`docs/infra_exp/`），**按问题编号不按框架**，
  预测跑之前写死、推翻不删记四段

**2026-08-14 改组**：infra 线按**两条 track** 组织（E 编号仍是身份、永不重排）：
`TRACK-A-hardware-kernel.md`（负载稀疏 × 6.4GB/s 拓扑 ⇒ 该写什么算子）、
`TRACK-B-framework-async.md`（通用 RL 框架假设在 agentic 负载上逐条失效）。
每个实验必须能答「服务哪条兑现物 / 需求由哪个测量指出」，答不上就显式停放（E04/E05/E06 已停）。

**2026-08-14 三条要点**：
1. **fully_async 已解封并在跑**（decoupled + partial_rollout + dynamic_bsz）。
   ⇒ 「把异步跑起来 / 加 staleness 修正」不再是待办。
2. **AReaL（arXiv 2505.24298）的 decoupled PPO / η / 可打断 rollout 我们已经全有**
   ⇒ **staleness 不是我们的创新点，别那么写**。但论文自承**没定量测过长度偏差**、
   没做按长度分层采样（future work）⇒ **这是 Track B 的新重心**。
3. ⛔ **「动态分池在 fully_async 下用不了」是错的**：`fully_async_rollouter.py:464` 调了
   `create_rl_sampler`，真原因是它在 **Ray worker 进程**、driver 侧 patch 够不着。
   修法＝把 sampler patch 装进已在用的 `verl_patches.setup_worker()`。
   ★ 新变种：**不是忘了接，是断定接不上而断定错了。**
   ⇒ 说「verl 不支持 X」前查三层：①配置项在不在 ②代码路径调不调 ③**它在哪个进程里跑**。

**监督密度（agent 负载的结构性特征）**：SFT 3.8–4.9% / **RL 4.17%**（E11，1755 条轨迹）。
prompt 占 88.3%、工具返回 7.6%、助手仅 4.2% ⇒ 切 prompt 省 8.5×、完整按 mask 筛省 24×。
⚠️ 我预测 10–20%，错 2.5–5 倍——**错因是拿 `max_response_length=1536` 这个配置上限当实际值**
（实测 response 中位仅 422）。见 [[feedback-measure-dont-infer]]。

**当前状态关键三条**：
1. one_step_off ✅ 跑通+调优；**fully_async ❌ 崩在 step 0**
   （`fully_async_policy/detach_utils.py:153` None 相减）。
   ⚠️ **不是 verl 上游 bug**（2026-08-14 读码核实）：verl 的
   `FullyAsyncLLMServerManager.generate` 已在 TokenOutput.extra_fields 里给
   min/max_global_steps，是**我们自己的 `syncopate/train/verl_agent_loop.py` 把它扔了**
   ——`generate()` 包装只取 token_ids/log_probs，`AgentLoopOutput.extra_fields` 只填
   reward_extra_info ⇒ verl 用默认键集补成 None（agent_loop.py:991-1004）。
   ⇒ 修法是**在我们的 agent loop 里把字段透传上去**（顺带把 staleness 经验分布的真值测量接上），
   **别 monkeypatch 跳过该 metric**——那正好把要量的东西删了。
   infra handoff §3-⑤/§4-② 的「上游 bug」措辞错，主线 05-handoff §2.1 是对的。
   这是 [[project-mechanism-not-wired]] 的又一例。
2. 权重同步 13.3–24 s/次**未查因**（LoRA 仅 132MB，时间不在传输上）
3. FA2 三点对照待跑（sdpa 静态 84.5s 已有基线；dynamic_bsz 大概率翻正）

相关：[[machine-4x5090-constraints]] [[syncopate-docs-map]] [[user-chaoyu-working-style]]
