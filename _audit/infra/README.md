# Infra 施工记录与原始证据

> 本目录保存实验施工记录和机器证据，不决定当前状态。当前状态看
> `docs/infra_exp/00-START.md`，当前队列看 `docs/infra_exp/01-TASKS.md`。

## 旧的平铺文件

目录根部现有的 `a*`、`b*`、`e*` JSON 和 `nsys/` 文件来自 4×5090、旧模型或旧软件栈。

- 计时文件仍能说明当时那次运行花了多久。
- 一部分机制和精度解释后来被 E21/E22 的静默正确性问题推翻。
- 它们不能作为 Modal B200 的 before，也不能和新的 B 系列直接计算加速比。
- 完整背景和 E 报告在 `docs/archive/infra_exp/legacy-4x5090/`。

这些文件原样保留，不改名、不覆盖。

## 新的 B 系列

新实验按下面的目录形状落盘：

```text
_audit/infra/B01/
  REPORT.md          预注册、过程记录、结论和证据索引
  manifest.json       环境、代码、模型、数据、拓扑和用户授权身份
  baseline/           before 臂
  candidate/          after 臂
  summary.json        预注册判据与最终读数
```

多臂实验可增加子目录，但不能让两个写者共享同一路径。`REPORT.md` 不维护
第二份任务队列；每份 `summary.json` 必须能追到原始日志、trace、checkpoint 或
评测文件，空字段和跳过不能算通过。

实验编号、报告路径和验收规则见 `docs/infra_exp/06-EXPERIMENTS.md`。
