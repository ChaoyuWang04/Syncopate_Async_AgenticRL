---
name: infra-line-state
description: G0 至 G7 候选队列与旧 B 系列证据边界
metadata:
  node_type: memory
  type: project
  modified: 2026-09-07
---

# Infra 当前记忆

当前主线是“真实训练→速度画像→按瓶颈安排优化→原入口A/B”。2026-09-08用户锁定Modal最多8卡B200、六个RL框架（verl为主）、FSDP2/Megatron、vLLM/SGLang/TensorRT-LLM以及一个MoE和一个dense。详细范围、数据形状与仅供参考的G地图见[Infra SYSTEM](../../docs/infra_exp/02-SYSTEM.md)，当前顺序只认[TASKS](../../docs/infra_exp/01-TASKS.md)，不从此记忆复制排期。

首次画像不预设bug、不要求未解缺口或业务准确率；保留身份/工作量/非零有效更新/权重交接和运行健康。形成改动后按影响补正确性回归和真实A/B。B11停止主动下钻；B12/B13只保留调查，复活依赖真实画像。已有B编号与结果保留，不能把搁置写成全部通过。数据只做公开输入的必要转换，不恢复业务造数；两个Lab仍独立。

历史证据导航（未在本次文档改写中重跑）：

- B01：环境认证通过。
- B02：分段修复后的机械全链，`pipeline_ok=true`、`all_passed=false`，不是性能 baseline。
- B03：35B c/d 有更新、同步、原始 token/概率记录；tiny FSDP2 梯度/AdamW/恢复通过。35B 跨引擎概率、GDN/MoE 解释与三次同源码性能重复未完成。
- B04：已查 Transformers/PEFT TP 实现，tiny TP 完整正确性和正式性能尚未验。
- B06：代码、preflight、规格与代码质量复审通过；目标 CPU 零跳过与双 B200 两次真实更新尚未验。
- B13：初始双 B200 通信/GEMM 读数；可靠拓扑/利用率/能耗/跨机器复验未完成，不是训练 baseline。

旧结果与未验边界仍保留各 `_audit/infra/Bxx/REPORT.md`，不改写原始源码身份。新实验必要设置、逐次结果与结论留简洁 REPORT；详细背景/问题/方案移交 upstream，精确复现和检索细目留小型 JSON 索引，关闭不搬迁。两个 Lab 默认独立；资源与进程归属按 [Compute](../../docs/syncopate/05-COMPUTE.md) 核验。
