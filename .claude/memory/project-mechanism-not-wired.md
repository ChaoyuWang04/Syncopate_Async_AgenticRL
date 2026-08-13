---
name: project-mechanism-not-wired
description: 这个项目最反复出现的失效形状是「机制建好了但没接上」，测试抓不到，只能靠日志判据和硬失败
metadata: 
  node_type: memory
  type: project
  originSessionId: 254d8707-7512-4e9b-bd89-6e1eeec39011
  modified: 2026-08-13T17:23:59.437Z
---

**「我建了一个机制，然后假设它会自动生效。」** 这是 Syncopate 反复栽的同一个形状，
到 2026-08-13 已经累计十次以上（cap 监视的工具不在菜单里、稀疏格子被取模削成 0 条、
verl 把日志级别硬编码成 WARN 导致统计看不到、动态分池的 monkeypatch 对另一个 trainer
无效、staleness 修正在 bypass 模式下不产出 ESS 指标……）。

**Why**：**测试抓不到这类问题** —— 单元测试验的是"机制本身对不对"，
不是"机制有没有被接上"。只有实跑才暴露。

**How to apply**：
1. **判据永远是日志，不是代码。** 每个靠 monkeypatch / 环境变量 / 配置生效的机制，
   都要在启动时打一行；`[pool] 动态分池启用` / `[rl] 模式=...` / `[verl-patch] ...`
   **没有这行就是没生效**。review 代码没用，去看那行日志。
2. **硬失败，不自动降级。** `--mode` 在卡数不够时直接报错而不是退回单卡 ——
   静默降级的表现是「跑起来了、但测的根本不是你以为的东西」。
3. **需要精确控制的量，别靠取模碰运气。**
4. 上游也会犯：verl 的 `train_batch_size` 在 fully_async 里强制为 0（而不是忽略），
   就是**逼你发现"你以为在控制的东西其实没接上"** —— 这是好设计，值得抄。

相关：[[feedback-measure-dont-infer]] [[machine-4x5090-constraints]]
