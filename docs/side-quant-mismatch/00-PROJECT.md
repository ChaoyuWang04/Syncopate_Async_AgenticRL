# Side Project · 训推量化失配的剂量学（quant-mismatch）

```
性质    独立 side project —— 与 infra 主线互不 gate（Chaoyu 2026-08-27 立项裁定）
归属    ⬜ 待认领（非 infra/主线两条线的同事）；本文档 = 完整交接件，读完即可开工
前身    infra E15「训推一致性尺子」（立项未动工）——其使命整体并入本项目，E15 编号封存
仓库约定 沿用两线共用仓库的文档纪律（00-START「文档更新秩序」）；产物落本目录
```

## 1 · 一句话

rollout 引擎与训练后端即使共享同一份权重，logprob 也对不上（mismatch）；rollout 一旦
量化（fp8 KV / W8A8 / MX 格式），gap 里多出一个**量化项**。本项目把这个量化项
**单独隔离、定剂量、看它对 RL 训练健康度与最终任务分的响应曲线**，并占住文献里
四个没人占的角（§3）。

## 2 · 文献地图（2026-08-27 查证，都读过摘要/正文）

| 工作 | 一句话 | 对我们的意义 |
|---|---|---|
| [Yao & Liu, On the Rollout-Training Mismatch](https://opt-ml.org/papers/2025/paper116.pdf)（2025.08 奠基） | vLLM↔FSDP 同权重 token 概率可差到 1 vs 0；INT8/FP8 放大；提出 TIS | verl `rollout_correction`（我们管线常驻）就是这个谱系 |
| [Flash-RL](https://github.com/yaof20/Flash-RL) | FP8/INT8 rollout 插件+TIS | INT8 残留 ~4% 差距 |
| [AIS](https://arxiv.org/html/2605.13907v1) | FP8 rollout + 三诊断量（ESS 比/logprob 差/方差放大）自适应 IS | **明确没做剂量扫描、没做 KV/MX**；诊断量与我们 rollout_corr 家族同构 |
| [QaRL](https://arxiv.org/html/2604.07853) | W4A16/W8A8/FP8 + 训练侧低精度 GEMM 对齐 + TBPO | **明确声明没实现 fp8 KV**；量化项与 kernel 项没定量分离 |
| [Jet-RL](https://arxiv.org/html/2601.14243v1) · [FP8-RL](https://arxiv.org/pdf/2601.18150) · [NVIDIA e2e FP8](https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/) | 反路线：训推统一 FP8 消灭 gap | NVIDIA 三件套打包对齐 bf16，但没拆项 |
| [QUADS](https://arxiv.org/html/2607.15810v1) · HiFloat4 · QeRL · [QuRL](https://arxiv.org/html/2602.13953) | FP4 家族 rollout | QeRL 系发现量化噪声早期=探索红利、后期=毒（非平稳双刃剑） |
| [Defeating the Mismatch via FP16](https://arxiv.org/pdf/2510.26788) | 换 fp16（尾数多 5 位）直接缩 gap | 系统侧路线代表 |

⇒ **"量化 rollout+IS 修正"整体已是热战场；本项目只做下面四个没人占的角。**

## 3 · 四个空角（= 本项目的全部范围）

1. **KV cache 量化作为单变量隔离**：QaRL/AIS 都没做。已有第一个数据点（§4）。
2. **量化项 vs kernel 项的定量分解**：文献只说"量化会放大"，没人给过减法。
3. **剂量-响应曲线**：现有全是"bf16 vs 单档量化"两点式；AIS 自认没扫强度。
4. **多轮 agentic 负载 + 误差随轮累积**：文献清一色单轮数学题。我们的负载多轮
   工具调用、增量拼 token 不重渲染 ⇒ 早轮的量化 KV 躺在上下文里参与后续每轮，
   **误差按轮复利**——「mismatch vs 轮数」曲线从 rollout dumps 就能画，没人画过。
   （远期第五角：MX 块缩放格式的误差是块内相关的，对 IS 权重分布的形状效应无人测；
   infra 线有现成 sm120 MXFP8 kernel 可借，见 infra E30。）

## 4 · 已有的一手数据点（infra 线 2026-08-27 冒烟 A/B，可直接引用）

```
4×5090 · Qwen3-4B+LoRA · fully_async 48 步 · 单变量 = vLLM --kv-cache-dtype
  kl(bf16 KV) ≈ 3.6–4.8e-4      ← 纯 kernel 项（本底）
  kl(fp8 KV)  ≈ 4.8–5.5e-3      ← 本底+量化项 ⇒ 量化项≈4.4e-3 = 本底的 ~11×
  IS 截断比例 0.07→0.47（破 H3 红线 0.40）· IS 均值 0.97→0.65-0.72（有偏）
  ESS/N 0.92→0.6 · 步速反慢 4.6%（训练 rollout KV 池仅用 16.7%，容量杠杆无着力点）
原始日志 `logs/smoke_newbox_0827*.log` · 历史决策背景
`docs/archive/infra_exp/legacy-4x5090/02-DECISIONS.md` 的“fp8 KV cache”行
```

## 5 · 实验设计（建议形状，认领人可改）

**剂量阶梯**（每臂 = 48 步冒烟 ~15min + 固定 EVAL ~11min，工具全现成）：
bf16 → fp8 KV → fp8 W8A8 → 叠加；每臂 × {序列 IS 开/关}；
读数 = {rollout_corr/kl · ESS · IS 截断比例 · IS 均值} + 任务分。
**两张独家图**：①量化项/kernel 项分解柱状图；②kl 随对话轮数的累积曲线
（从 dispatched.jsonl / rollout_dumps 逐轮算 logprob 差）。
微观-宏观对账：infra E19 有「FP8 误差 = vLLM↔FSDP 数值地板的 316×」的探针读数。

## 6 · 现成资产清单（认领人开箱即用）

```
度量仪器   verl rollout_correction 指标族（管线常驻）：rollout_corr/kl · ESS · fraction_low/high
开关       launch_rl --kv-cache-dtype {auto,fp8} · vLLM W8A8（E19-c 跑过）· --rollout-is {sequence,token}
跑法       infra 00-START §6 冒烟模板 + 06-rl-run-protocol 判据清单
微观探针   scripts/infra/probe_fp8_logprob_error.py（E19）
远期借用   syncopate/train/tilelang_mxfp8.py（sm120 MXFP8 kernel，E30）
```

## 7 · 判据与产出

- 判据：每臂对拍口径一致（同 seed 同数据同步数）；结论必须以「量化项剂量→健康度指标→任务分」三层对齐呈现；负结果照登；
- 产出定位：workshop 论文 / 技术报告级；四个空角占住任意两个即值得写。
