# E31 · 训推部署 FP8 全盘一致（消费级 sm_120 复刻 Miles 路线）

> 立项：Chaoyu 2026-08-27（**队首，单任务制**）。目标一句话：**rollout、训练前向/反向、
> 部署三方共用同一份 MXFP8 量化契约**——量化项在 IS 比率中逐字节对消，训练的就是部署的那个模型。
> 地基：E30 全套（kernel 三件套 627/2.1×/1.7× · 温度偏置机理 §11 · A4 真训练同带 §13）。
> 参考配方：Miles 端到端 MXFP8 RL（B200，[博客](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/)）·
> Unified FP8（[博客](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)）· DeepSeek-V3 分段累加。

## 0 · 原理卡（为什么这条路成立）

```
IS 修正的总账 = 陈旧度项（fully_async 设计使然）+ kernel 项（引擎差，本底 ~4e-4）
             + 量化项（单侧量化时 = E30 §11 温度偏置，序列级复利爆炸）
统一精度只消第三项 ⇒ TIS 保留（本职=付陈旧度的账），回到只付本职
⚠️ "两侧一样"只在权重字节层面成立；激活因 kernel/批组合而异 ⇒ kernel 项永生，验收对本底比
★ 训练对象从 bf16 模型变为量化模型 π_q 本身 ⇒ 训-推-部署三方一致（不是副作用，是目标）
与 Miles 的三个结构差：①sm120 无 TE/FlashInfer 接线 ⇒ 两侧用我们同一份扩展（位一致近乎免费）
②LoRA 冻结基座 ⇒ 量化一次整训缓存（他们要双向副本）③我们异步 ⇒ 陈旧度是独有第三变量
```

**全程三把尺**（管线常驻）：`rollout_corr/kl` · IS 截断比例（fraction_low+high）· ESS/N；
**终审尺** = 400 步 candidate 任务分配对（±MDE 0.015）+ 奖励曲线与 bf16 基线重叠。

## 1 · 施工步骤与逐步验收（每步都可写成 test）

### 第 0 步 · 契约与地基测试（不碰训练）
| 测试 | 写法 | 通过标准 |
|---|---|---|
| T0.1 量化器位一致 | 5 类张量：随机 / amax=0 / 全同值 / 448 边界±ulp / 2 的整幂 | 训推两侧输出 `torch.equal`(uint8) |
| T0.2 GEMM 确定性 | 同输入 ×100 + 换 stream | 逐位相同 |
| T0.3 本底标定 | bf16 两引擎同轨迹 kl | 记 `kl_floor_bf16`（新机 ~4e-4），全程分母 |

### 第 1 步 · lm_head 两侧同量化（量化项对消的最大头）
做法：vLLM `compute_logits` monkeypatch → 同一扩展+同一量化器（lm_head 无 LoRA，权重字节两侧本同）。
验收①（离线，§9b 方法重打；**08-27 就地改写**，见下方 ⛔）——全部锚定同尺 bf16 对照臂：
  token |Δlp| 中位与均值 ≤ 2×本底 · **逐 token 签名偏置 ≤ 2×本底偏置**（对消的直接读数）·
  序列 |ΣΔ| p95 ≤ 2×本底 p95 · 哨兵=单侧毒臂偏置 ≥5×本底且 16/16 同号（测量必须看得见病）。
验收②：48 步冒烟：kl ≤ 1.5×floor · 截断比例 ≤ 0.10 · ESS/N ≥ 0.85 · 八判据全过。

> ⛔ **原①的「序列 ΣΔ p95 < ln2」判死（08-27 实测推翻）**：bf16 对照臂自己在 1300–1800
> token 长序列上 p95=2.54 ≫ ln2——**引擎漂移本来就超 ln2**（与 kvauto 冒烟 bf16 的
> seq_fraction_low 4–7% 互证）。写阈值时没先给序列级立 bf16 本底 = 守则①原文的学费。
> 阈值改锚对照臂；序列级的最终裁决交给 ②（生产长度混合 + 真 IS 指标）。

### 第 2 步 · trainer old_log_prob/entropy 切 8bit（无梯度，PG 补丁调用点）
统一后序列 IS 应重新安全——本步即"偏置对消"的活体检验。
验收：同上尺 + **序列 IS 模式**48 步无守卫报警。

> **第 1/2 步合并施工（08-27 侦察后定，验收仍分层）**。现场事实：trainer 侧投影接缝只有
> `_pg_forward` 里 `FusedLinearForPPO` 一处（olp / update_actor / entropy 共用）；vLLM 模型
> 跑在 spawn 的 Worker 子进程（smoke 日志 Worker pid 实证），普通 monkeypatch 到不了 ⇒ 走
> `vllm.general_plugins` 入口点（每个 vLLM 进程都加载）。**一个开关 `SYNCOPATE_UNIFIED_FP8`
> 同时切两侧**——分开切就是亲手制造"单侧量化"毒状态（§9b 判死的正是它）。
> 工件：`syncopate/train/unified_fp8.py`（vLLM 插件 + trainer 端 MXFP8LinearForPPO 分块
> autograd，语义逐项对齐 verl 融合算子；entropy_coeff=0 已钉 ⇒ dentropy 非空即炸）·
> pyproject 入口点 · `_pg_forward` 单点分派（默认关 = 逐位走旧路，有单测钉）。
> 对比基线锁定：模型/数据/默认值与 08-27 冒烟 kvauto 臂完全同源（kl_floor 的出处），
> 单变量 = 开关本身。判据行两条：`[unified-fp8] vLLM/trainer lm_head MXFP8 已生效`，
> 冒烟里缺任何一条 = 机制没接上（第八形态），直接判负。

### 第 3 步 · 内层 GEMM 渐进 8bit（前向，两侧同步推）
顺序：前 ~85% 层冻结基座 QKVO/MLP（LoRA 增量保 bf16 = 天然高精度孤岛）；末 15% 层
与非 GEMM 算子保 bf16（Miles 分配）。**每扩一组层验一次**：|Δlp| ≤ 前一步 1.5× ·
kl ≤ 2×floor；炸的层组划进 bf16 孤岛（照登）。

### 第 4 步 · 反向 8bit（三件套 dgrad/wgrad；梯度逐 token 量化；DeepSeek 分段 fp32 累加已在 kernel）
验收：梯度 cos ≥ 0.999（§12 口径）+ 同种子短训 loss 曲线与 bf16 臂重合。

### 第 5 步 · 权重同步契约（E22 推送链的量化版）
LoRA bf16 推送不变；量化基座两侧字节一致由 1/3 步保证。
测试：推送后 rollout 侧抽层 `‖W_q‖` 与 trainer 侧逐字节同。

### 终审
400 步 candidate 全默认 + 任务分配对 ±MDE + 奖励曲线重叠；负结果同权入档。

## 2 · 状态

🟡 **第 0/1/2 步 ✅（2026-08-27 一日三步）**；下一步 = 第 3 步内层 GEMM 渐进 8bit。

**第 0 步**：三契约测试进 suite（`tests/train/test_e31_fp8_contract.py`，8 项）——
T0.1 量化器五类张量位一致（bf16/fp32 同值·非连续·换 stream 三接缝 + 全零/整幂/448±ulp
语义）· T0.2 GEMM ×100+换 stream 逐位同、对 dequant 参考仅 bf16 舍入（rel 1.7e-3）·
T0.3 **kl_floor_bf16 = 4.27e-4** 固化（`logs/e31/kl_floor_bf16.json`，标定守卫实测拒
fp8 臂 5.3e-3）。

**第 1/2 步（合并施工，验收分层全过）**：`syncopate/train/unified_fp8.py` 一个开关
`SYNCOPATE_UNIFIED_FP8` 同切两侧——vLLM 经 `vllm.general_plugins` 入口点进 spawn 的
Worker 进程（判据行实测在 EngineCore/Worker 打出）；trainer 经 `_pg_forward` 单点分派
（olp/update_actor/entropy 共用，MXFP8 分块 autograd，backward dgrad 走同 kernel）。
- 验收①离线四臂（`logs/e31/step1_offline.json`，16 条真轨迹 1299–1813 token）：
  **签名偏置 +3.76e-3（单侧毒臂，16/16 全正）→ +4.21e-4（统一，= 本底 3.39e-4 的 1.24×）**
  ——温度偏置 9× 消减到引擎本底；token |Δlp| 中位 1.23×本底 · 序列 |ΣΔ| p95 1.74×本底
  （残余=无偏舍入噪声的 √N 游走，非复利偏置）。
- 验收②48 步冒烟（`logs/e31/step1_smoke.json`，全默认=序列 IS）：kl {3.28,5.55,5.19}e-4
  ≤1.5×floor 且在 bf16 臂同带 · 序列截断 0.081（bf16 臂 0.072）· ESS 0.868（bf16 0.876）·
  八判据 8/8 · **步速 9.12–9.20 s/gstep 与 bf16 臂持平** · 守卫零报警（=第 2 步序列 IS
  活体检验过）。单元层 10 项（默认关逐位走旧路·分块逐位不变·反向对拍解析公式逐位同·
  零 dentropy 放行回归——entropy_coeff=0 时 verl 仍连图传零梯度，首跑在此炸过一次）。
- 单变量对照：模型/数据/默认值与 08-27 kvauto 冒烟完全同源，唯一变量=开关。全套
  pytest 712/0 skip（A1/A2 工件测试常驻）。
