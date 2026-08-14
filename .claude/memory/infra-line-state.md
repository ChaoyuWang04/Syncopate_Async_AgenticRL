---
name: infra-line-state
description: infra 线（多卡/异步/MoE）的已定决策与当前状态；入口是 docs/infra_exp/00-INFRA-HANDOFF.md
metadata: 
  node_type: memory
  type: project
  originSessionId: d8054c42-ce87-481d-a266-b7806a058358
  modified: 2026-08-14T13:43:39.113Z
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

**2026-08-14 GPU 时段的主要结果**（详见 `docs/infra_exp/` 的 E11/E12/E13/E08/E02）：
1. ★ **整机占空比只有 31%**：trainer 三卡空闲 54–57%，**rollout 卡空闲 82.5%、平均 47.7 W**。
   ⇒ 比任何算子优化大一个量级（对照：全套自写 kernel 端到端 4.3%）。**Track B 的头号发现。**
2. ★ **权重同步 60 s 的根因不是传输**：编排 8 步合计 0.038 s（0.06%），
   99.94% 在第 5 步；而 8 GB 与 132 MB **同耗时** ⇒ 与数据量无关。
   已排除：建 NCCL 组（一次性 46 s）、buffer 分配（2.6 ms）、`empty_cache`（12.2 ms）、
   **`layered_summon`（A/B 60.13 vs 60.16，无差异）**、「取参数」整环节（两条不同路径同耗时）。
   **只剩 `send_weights`**，探针已加。猜想：`collective.broadcast` 广播的是**整个 2048 MB 定长 buffer**。
3. ✅ **E13 已落地**：`ddp_save_to_cpu` 加一行 `if param.requires_grad`
   （8.309 GB 里只有 3.18% 可训练，冻结基座跨版本逐字节相同）。
   `old_log_prob/ref` 比值 **1.941 → 1.069**（M7 37 样本 vs 3 样本），省 ~8.5 s/步。
   端到端保守报 −5%（样本不足）。3 条测试守着。
4. 🔻 **E11 降级不写 kernel**：密度 4.17% 但 lm_head 只占前向 **4.28%** ⇒ 端到端仅 4.3%，
   而最笨的切片就有 4.0%。★ **「浪费的比例」和「能拿回的收益」隔着一个分母。**
5. ⛔ **`gpu_util 0.75` 不是安全值**：实测第 4 次同步 OOM（vLLM 24.65 + CE 4.71，
   剩 1.99 要 2.00，**差 0.01 GB**）。**解法不是压 gpu_util，是调小
   `--weight-sync-bucket-mb`**（send+recv 各 2048 MB，实际只推 132 MB）。

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
