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

### 第 3 步 · 内层 GEMM 渐进 8bit（前向，两侧同步推；**08-27 施工细化，分 3a/3b 两级**）
顺序：前 30/36 层（83%）冻结基座 QKVO/MLP；LoRA 增量、attention/RoPE/norm、embedding、
末 6 层保 bf16（Miles 分配）。**每扩一组层验一次**，炸的层组划进 bf16 孤岛（照登）。

**3a（eager 版，本级交付）**——绕开两个硬阻塞拿正确性：
- 事实依据：`enforce_eager=True` ⇒ vLLM 强制 O0（inductor 关）+ cudagraph NONE ⇒
  python 补丁安全（vllm/config/vllm.py:599/706 实读）；速度税由冒烟记账，3b 再赎。
- **位一致基石**：行块量化对行拼接不变 ⇒ vLLM 合并权重（qkv[6144,2560]·gate_up[19456,2560]）
  与 trainer 分开权重的量化字节逐位同；GEMM 输出元素只依赖自己行列 ⇒ 合并/分开输出
  逐位同（U5 测试钉）。全部维度过 128 约束（4096/1024/2560/9728/6144/19456 ✓）。
- 开关：`SYNCOPATE_UNIFIED_FP8_LAYERS=N`（前 N 层；默认 0=现状只有 lm_head）；
  **N>0 必须同时 `SYNCOPATE_UNIFIED_FP8=1`**，否则启动即炸（半接线=毒状态）。
- trainer 侧：`_pg_forward` 首调时按名遍历 swap `base_layer.forward`（PEFT 外裸 Linear
  同支持）；autograd 只做 dgrad（dx=dY_q·Wᵀ_q，**不存激活**，wgrad 冻结拒绝）；
  swap 数断言 =N×7，差一个都不许跑。
- vLLM 侧：插件加患 `UnquantizedLinearMethod.apply`，按 `layer.prefix` 解析层号选层；
  bias 非空即炸。
- 显存账（预算，跑中验证）：u8 缓存≈bf16 一半 ⇒ vLLM +~2.6GB（画像阶段自动入 KV 预算）·
  trainer 前向+转置 +~5.2GB ⇒ 峰值 ~21/32 GB。
- **渐进组与验收（跑前写死）**：G0'=仅 lm_head（eager 重锚）→ G1=8 层 → G2=16 → G3=24 →
  G4=30。每组离线四臂（§9b 同尺，全 eager）：unified token |Δlp| mean ≤ **1.5× 上一组** ·
  签名偏置 ≤ 2×本底 · 序列 p95 ≤ 2×本底（本底=同 eager 的 bf16 臂）。
- 终验收：48 步冒烟（N=30 + `--enforce-eager True`）：**kl ≤ 2×floor（8.54e-4）** ·
  截断 ≤0.10 · ESS ≥0.85 · 八判据；速度税记账：**s/gstep ≤ 12.0（1.3×）为绿**，
  超了黄牌入档、3b 提级。

**3b（后续提速）**：`mxf8_gemm` 注册 torch custom op（fake impl）+ 量化路径过 inductor ⇒
恢复 compile/cudagraph，赎回 eager 税；正确性判据=与 3a eager 输出逐位同。

> ⛔⛔ **第 3 步定界（08-27 实测判负，负结果同权入档；工件 `logs/e31/step3_offline.json`）**
> 逐组数据（分母=同 eager bf16 臂 mean 1.37e-2 · bias −4.4e-4 · p95 2.71）：
> `N=0 ✅(1.63e-2/−5.2e-4/4.39) · N=8 🔴(2.12e-2/−1.26e-3/6.59) · N=16 🔴(3.57e-2/−2.63e-3/10.9)
> · N=24 🔴 · N=30 🔴(4.96e-2/−3.56e-3/12.6)` —— **偏置随层数近线性 ~−1.2e-4/层**，
> 8 层即破 2×本底门，30 层序列 p95=12.6 ⇒ 序列 IS 必死，冒烟无需再跑。
> 三个替代解释全排除：①接线错位——trainer N×7 硬断言 + vLLM 命中审计恰 N×4、层号 0..N-1
> 对齐；②权重字节——U5 证合并≡分开逐位同；③跑间噪声——同配置 vLLM 重跑**逐位相同**。
> **根因是结构性的**：§0 原理卡"激活因 kernel 而异 ⇒ kernel 项永生"的极端化——两引擎
> hidden 本有微差（异构 attention），**每层激活量化放大它并向下一层传递**。Miles 内层
> 可行靠两侧同 TE kernel（hidden 同构）；消费卡 vLLM↔FSDP 异构引擎无此前提。
> ⇒ **裁决：内层全部划 bf16 孤岛；统一 FP8 的可行域 = lm_head**（对消成立·零速度税）。
> 代码留库停放（`SYNCOPATE_UNIFIED_FP8_LAYERS` 默认 0，U5/U6 测试常驻）。
> **复活条件**：① 切 token 级 IS（N=30 的 token mean 4.96e-2 ⇒ 逐 token 扰动 ~1.05，
> §9b 判读表 token-IS 安全档）；② 两侧引擎同构化（前提大改，当前不成立）。

### 第 4 步 · 反向 8bit（✅ 08-27 随存活范围收口）
存活范围=lm_head：dgrad 走同 kernel **已在生产接线**（第 1/2 步 `_MXF8LinearForPPOFn`，
U3 逐位钉）+ 梯度 cos 0.99928（E30 §12 真数据）+ A4 完整 SFT 同带 + 本线 48/400 步
真 RL 训练全程带着它学出 +0.109 —— 验收各面已实测覆盖。wgrad：lm_head 冻结 ⇒ 不适用
（非冻结即炸的断言常驻）。内层反向随第 3 步定界一并停放。

### 第 5 步 · 权重同步契约（✅ 08-27 收口）
LoRA bf16 推送不变（E22 链 + [sync-payload] 探针常驻）；存活范围下"量化基座两侧字节
一致"归结为 lm_head（tie 到 embedding·冻结·启动推一次基座）——契约测试 T5
（`test_t5_weight_contract_disk_is_truth`）：磁盘 safetensors 直读与 HF 加载两条路径
逐位同 ⇒ 两侧量化缓存必然同；运行期由 ‖W‖ 探针 + kl 地板判据兜底（字节漂 = kl 起飞）。

### 终审
400 步 candidate 全默认 + 任务分配对 ±MDE + 奖励曲线重叠；负结果同权入档。

> ⚠️ 口径评注（08-27）：「与 bf16 臂任务分差 ±MDE(0.015)」在**单种子跨跑**下不可判——
> 家族内种子带宽实测 0.085（+0.101 vs +0.186），比 MDE 大 5×。可达的终审口径 =
> 落家族带 + 曲线同形 + 三把尺全程（已由 08-27 长跑满足）；要收紧到 ±MDE 只有
> 多种子配对或同种子同位选点重跑（save-freq 调密），是否加跑归 Chaoyu 裁定。

## 2 · 状态

🟢 **全六步闭环（2026-08-27 单日）**：0/1/2 ✅ · 3 ⛔定界（内层判负入档，可行域=lm_head）·
4/5 ✅ 随存活范围收口 · 终审=可达口径已满足（±MDE 收紧与否待 Chaoyu 裁定，见 §1 终审评注）。
**定案一句话：消费级异构引擎上，训推统一 FP8 的可行域 = lm_head 层——量化项对消实测
成立（偏置 9× 消减至本底）、400 步真训练三把尺健康、质量入带、零速度税；内层因激活
量化逐层放大引擎间 hidden 微差而判负（结构性，复活条件=token 级 IS 或同构引擎）。**

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

**✅ 400 步中期长跑过闸（第 1/2 步的剂量检验，非终审；门槛跑前写死、当日兑现 08-27）**
起点 `models/Qwen3-4B-sft-v13r2-e1`（与 hf_assets 底座 shard sha256 同源核验）· 全默认 +
开关 · purpose=candidate · 工件 `logs/e31/e31s12_cand_run.json` + `_audit/e31s12_cand_unified.json`：
① 跑完 ✓：400 步全完 · abort=0 · clip 恒 0 · 判据行全在 · 步速 **8.5–9.1 s/gstep**（快于冒烟）；
② 三把尺 ✓：**kl 中位 6.25e-4 ≤ 6.40e-4（贴线 98%，如实标注；比 bf16 长跑中位 4e-4 抬
  ~1.5×，无趋势、尖峰瞬时 max 3.4e-3 后回落）** · 截断 max 0.096 · **ESS 中位 0.902 /
  min 0.863**（bf16 带 0.92/0.816——同带且更稳）；
③ 质量 ✓：配对 **+0.109（t=+7.4）**，落 bf16 家族带 [+0.101, +0.186] 下沿——评的是
  400 终点（过训回落段，峰 0.951@gstep 223 与 bf16 峰~200 同形）；bf16 的 +0.186 是
  RL-100 峰前选点，位置不同不做选美。⚠️ 终点 defer −72% = 已知 400 步过训形态
  （bf16 家族同样，RL-100 因此才是晋级点）；单种子无法排除 FP8 加重过训，终审时
  按 bf16 选点法同位对比。
⇒ **第 1/2 步定案：lm_head 统一 FP8 在 400 步真训练剂量下三把尺健康、质量入带、零速度税。**
**第 3 步侦察先记两个硬阻塞**（本日发现，动工前必须解）：vLLM 内层 Linear 在 CUDA graph
捕获区内（python 级补丁会被烤进图或撕破捕获，与 lm_head 的 eager 位置本质不同）；
逐层 python 量化 ~144 调用/token 会拖垮 decode ⇒ 需要静态缓冲 custom op 或融合量化。
