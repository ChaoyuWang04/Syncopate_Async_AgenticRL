---
name: syncopate-project-framing
description: 在模拟场景中把项目工程做好，用证据理解 SFT/RL/OPD，不以业务上线或投稿为目标
metadata:
  node_type: memory
  type: project
  modified: 2026-09-05
---

# 当前项目定位

用户最新裁定：场景是虚构的，模型不会真正投产。以跑通并做好项目为主，理解 SFT、RL、OPD、训练基础设施、推理和 kernel 的实际工作方式。

- 现有数据冻结，不再造数或语义清洗。只保留读取、身份、三桶隔离、预算、token/mask 和数值检查。
- 模型质量 WARN 用于观察学习，不再阻塞 infra 起跑；错误身份、NaN、坏产物、零真实更新仍停止。
- 先修真实工程问题、取得可靠 baseline，再并行做有意义的对比，采用项组合后完成全链学习运行。
- SFT/RL/OPD 的效果用相同冻结题目观察：变好、变差或没动都如实记录。不能仅凭 loss、权重位移或步数声称能力改善。
- 不按求职目标倒推任务，不强求 PR 或论文。可能通用的工程发现按 `docs/upstream/README.md` 留 DRAFT；后续考据、提交或论文由用户决定。
- 只使用 Modal CPU/B200。每批卡数、时限、费用和证据先登记，独立实验尽量并行。
- 当前任务只看双 TASKS，资源只看主线 Compute，实验计划与读数只看 B REPORT。

不要把旧的“真实业务第一、质量收口后才准做 infra”继续当成当前指令。
