---
name: modal-migration-state
description: Modal 主资源及保留 B200 实现和历史验证边界
metadata:
  node_type: memory
  type: project
  modified: 2026-09-07
---

# Modal 当前记忆

资源唯一说明是 [Compute](../../docs/syncopate/05-COMPUTE.md)，操作入口说明是 [Modal README](../../modal_app/README.md)，任务只认 [Infra TASKS](../../docs/infra_exp/01-TASKS.md)。

- Modal为主要资源，本轮上限8卡B200，按需分配且所有角色计入；其他卡型不在本轮执行范围。旧入口主要单/双B200，新8卡真实训练入口须按TASKS核验，不把预算许可当实现完成。
- 保留资源为 `syncopate-home` Volume、每容器独立 checkout；新实验登记专属写入目录、卡数、时限和费用。不同项目不默认共享资源。
- 旧栈版本、B200 运行身份与数值以 Compute 和 B REPORT 为准；每个新实验重查最新官方来源并固定实际版本。
- v16 业务数据保留，数据构建/教师扩写/完整学习运行停止。旧合并模型、部分 checkpoint 和私有缓存已按用户授权清理；产物实际可用性见 [存储清理记录](../../docs/infra_exp/storage/20260907-cleanup.md)，不能仅凭旧 REPORT 推断仍存在。
- B01/B02 已有环境和机械链证据；B03/B04/B06/B13 的已验/未验边界见 [infra-line-state](infra-line-state.md) 及原 REPORT，不能推成新基线。
- Mac 做控制和可用 CPU 检查；本地/5090 偶尔有其他项目负载。清理任何进程前核对项目、run-id、PID/Modal app 和资源归属，不按卡型或名称批量杀进程。
- 先真实入口画像，再按瓶颈做定位；新场景准备按TASKS，不把旧G0.3当预设施工。2026-09-08已只读盘点模型/数据，见Compute资源记录；分片存在不等于加载或完整哈希通过。
