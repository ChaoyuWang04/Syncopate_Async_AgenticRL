# DRAFT · sm_120（消费级 Blackwell）MXFP8 生态数据点回帖包

> 状态：DRAFT。一份材料多处回帖——E30/E31 攒下的消费卡 MXFP8 第一手数据，
> 生态里每家都缺这一块（2026-08-27 搜索核实）。**不是一个 PR，是一组带证据的回帖/评论。**

## 去处与各自要说的话

| 去处 | 现状（已核实） | 我们的数据点 |
|---|---|---|
| TE [#2304](https://github.com/NVIDIA/TransformerEngine/issues/2304) | 官方报错原文 "MXFP8 not supported on 12.0+ architectures yet" | sm_120 可行且有包络数：`mma.sync.m16n8k32.kind::mxf8f6f4` 路径 627 TFLOPS = 61.2% 发射峰（E30 §10）；**传统 FP8 mma 在 sm_120 只有半速，必须走新 kind**（E30/E16 §7 头号发现）；缩放 lane 映射已逆向（probe_mxf8_scale_mapping.cu） |
| triton [#7550](https://github.com/triton-lang/triton/issues/7550) | `tl.dot_scaled` 在 sm_120 = bf16 仿真（docstring 自供） | 实测 MXFP8 反而比 bf16 慢 38%（119 vs 192.5 TFLOPS，E16 §6）；原生路径参考实现在我们仓库 |
| CUTLASS [#2867](https://github.com/NVIDIA/cutlass/issues/2867) | sm120 blockscaled 缺口 | 同上包络数 + 消费卡寄存器堆物理约束分析（4 warp×255 reg×64×64 = 可行域顶点，E30 §10b） |
| DeepGEMM [#236](https://github.com/deepseek-ai/DeepGEMM/issues/236) · SGLang [#9233](https://github.com/sgl-project/sglang/issues/9233) · vLLM [#51884](https://github.com/vllm-project/vllm/issues/51884) | sm120 FP8 块缩放全线缺位/报错 | 参考实现与布局文档同上；能救 sm120 port 的人少走弯路 |
| RL 系统社区（Miles/verl 讨论区，形式待定） | Miles B200 端到端 MXFP8 RL 已发；消费卡无人做 | **正结果**：lm_head 两侧统一在异构引擎下对消成立（偏置 9× 消减至本底·400 步三把尺健康·+0.109 入带·零速度税）；**负结果**：内层双侧量化被异构引擎 hidden 微差逐层放大判死（~−1.2e-4/层线性，8 层破门；三替代解释排除；E31 §1 定界框）——这正是"为什么消费卡只能做到 lm_head、B200 能做全模型"的机理级解释，也间接解释 NVIDIA e2e FP8 选 token-TIS 的原因 |

## 已知不提的（PARKED 理由留档）

- vLLM prompt_logprobs 显存尖峰不入预算 ⇒ OOM：**上游已知且有 tracking**
  （[#5067](https://github.com/vllm-project/vllm/issues/5067) ·
  [#5550](https://github.com/vllm-project/vllm/issues/5550) ·
  [#5907](https://github.com/vllm-project/vllm/issues/5907)）——不重复报；
  我们的 V1/0.12/sm120 复现参数（util 0.6 + 批 8192 必死；0.55→0.72 + 批 2048 过）
  最多补条评论。本地解法已固化在 scripts/e31_step1_offline.py。

## 材料清单（全在仓库）

E30（kernel/包络/机理全套）· E31（统一 FP8 六步含定界负结果）·
scripts/mxf8_gemm_limit_tma.cu（T1 kernel）· probe_mxf8_scale_mapping.cu ·
logs/e31/step3_offline.json（定界工件）· tests/train/test_e31_*.py（54 项常驻）

## 提交前待办（Claude 考据）

- [ ] 各 issue 的最新回复态（搜索时点 2026-08-27）
- [ ] RL 社区数据点用什么载体（issue 评论 / discussion / 短文）——归 Chaoyu 定调
