# docs/archive — 历史资料

> **这里没有当前答案，也没有待办。**
> 主线现状只看 `docs/syncopate/00-START.md`，infra 现状只看
> `docs/infra_exp/00-START.md`；待办分别只看两边的 `01-TASKS.md`。

归档的作用是保留旧方案、旧机器结果、施工过程和调查笔记。归档文档可能包含
已经失效的数字、路径、判断或待办，不能直接拿来指导现在的运行。

## 目录地图

| 路径 | 保存什么 |
|---|---|
| `syncopate/pre-consolidation-v16/` | 主线整合前的 v16 施工、决策、机器和状态快照 |
| `syncopate/legacy-notes/` | 更早的主线调查、数据与轨迹分析、训练记录、复盘和旧部署说明 |
| `infra_exp/legacy-4x5090/` | 旧 4×5090 时期的 E01～E33、设计文档、简历材料和已退役的跨线文件 |
| `infra_exp/legacy-notes/` | 旧框架调研、verl 学习笔记、调度/训推一致性/去 padding/权重同步调查 |
| `infra_exp/b-series/` | B200/B300 时代已经结束并验收过的完整实验报告；目前还没有正式报告 |

`docs/archive/` 根目录只保留这份地图，不再平铺历史文档。

## 本次分流

原先散落在 archive 根部和旧学习笔记目录的资料按责任线归档：

- 主线：项目概览、状态审计、数据分布、轨迹、verifier/reward、sandbox、
  信用分配、RunPod 旧部署、第一次 RL、M7-b、RL 学不动和 2026-08-18 复盘。
- infra：框架选型、为什么用 verl、资源调度、训推不一致、remove padding 和
  E12 权重同步。

原文件名全部保留，方便追溯旧引用；目录位置表示它现在归哪条线管理。

## 归档规则

1. 当前事实提炼回 `docs/syncopate/02～07` 或 `docs/infra_exp/02～05`。
2. 未完成事项只进入对应线路的 `01-TASKS.md`，不能留在归档中继续追踪。
3. B 系列施工报告与原始证据先放 `_audit/infra/Bxx/`；实验结束后，完整报告
   才移入 `infra_exp/b-series/`。
4. 归档后原则上冻结内容；只修坏链接、归档提示或会造成误读的明显标记。
5. 引用旧数字前，先看
   [`syncopate/pre-consolidation-v16/21-invalidated-numbers.md`](syncopate/pre-consolidation-v16/21-invalidated-numbers.md)。
