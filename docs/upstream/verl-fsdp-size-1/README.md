# 提交包 · verl `fsdp_size=1` 静默不同步梯度（E21）

> **状态：材料齐备，等 Chaoyu 点头后提交。** 目标仓库 **verl-project/verl**（已迁库，别用 volcengine 旧名）。
> 顺序：先 issue（拿编号）→ PR 填 `Fixes #<n>` → （可选）去 pytorch#154888 评论补证据 + 链接。

## 一句话

`fsdp_size=1`（多卡、不分片——小模型/LoRA 的自然写法）会让 verl 造出 `(N,1)` 网格并选中
`HYBRID_SHARD`；FSDP1 把它钳成 `NO_SHARD` 却把梯度归约留在 size-1 的分片组上 ⇒
**N 个 rank 各训各的，静默**。PyTorch 侧已确认是 bug 且不修（#154888 not_planned）⇒ verl 是唯一能兜住的地方。

## 文件清单

| 文件 | 是什么 |
|---|---|
| [`submission-EN.md`](submission-EN.md) | **issue + PR 英文正文**（GitHub 直接粘贴）+ 提交注意事项 |
| [`verl-get-sharding-strategy.patch`](verl-get-sharding-strategy.patch) | 可直接 `git apply` 的修法（基于 0.8.0，与 main 逐字同源） |
| [`repro_degenerate_mesh.py`](repro_degenerate_mesh.py) | 独立复现脚本（纯 PyTorch，2–3 卡，issue 里内联的那份） |
| [`test_degenerate_mesh_grad_sync.py`](test_degenerate_mesh_grad_sync.py) | PR 要带的测试（2 卡即触发；修前红、修后绿） |
| [`analysis.md`](analysis.md) | 中文分析与证据链（触发条件 / 代码路径 / 后果 / 修法论证） |
| [`pytorch-background.md`](pytorch-background.md) | PyTorch 侧背景（#154888 时间线 / 四行源码定位 / FSDP2 对照） |

## 证据（仓库内产物，提交时引用数字即可，不用上传）

```
_audit/infra/e21_grad_sync_matrix.json            七变体确定性矩阵（+ _fixmode 版）
_audit/infra/e21_verl_gdiff_rank_fingerprint.json 确切 diff 在 verl 源码树真跑 4 步：
                                                  三 rank 504/504 张量逐位相同 + 优化器一致
logs/e21_grad_sync_matrix{,_fixmode}_20260819.log 矩阵控制台全文
logs/e21_verl_gdiff_20260819.log                  verl 源码树验证跑全文
scripts/repro_fsdp_hybrid_nosync.py               七变体矩阵脚本（本包 repro 是它的极简版）
完整实验记录                                        ../../infra_exp/E21-ddp-not-syncing.md
```

## 提交前最后一眼

- [x] 查 verl CONTRIBUTING ✅ 全部落实：标题格式受 CI 检查、PR 模板六节填全、Apache 许可头、
      测试放 `tests/special_distributed/` 并注册进 `run_all.sh`（`model.yml` 的 paths 已覆盖 ⇒
      CI 自动触发，无需改 workflow）、ruff + ruff-format 干净、commit 带 DCO 签名
- [x] issue 正文扫过，无内部路径/实验名
- [x] PR 分支已基于 verl-project/verl main 建好并本地验证 —— `/workspace/_upstream/verl`
      分支 `fix/fsdp-size-1-degenerate-mesh-grad-sync`；测试实弹验证**修前红**
      （[0.313, 1.254]）**修后绿**（逐位相同）
- [x] ★ **声明审计（2026-08-19）**：逐条核对"写下的每个数是不是真跑出来的"，**抓到两处并已修**：
      ① issue 内联脚本从没跑过，且其模型/损失/种子与表里数字的来源脚本不同 ⇒ 已重写、实跑、
         用**它自己的输出**替换表格（顺带把 FSDP2 对照并进同一个脚本）
      ② "ckpt 格式修复前后一致"当时只探了修复后两档 ⇒ 已补做修复前那档，实测一致
- [ ] 提交后：CI 结果与 review 回复
- [ ] 操作手册见 [`READY-TO-PASTE/0-STEPS.md`](READY-TO-PASTE/0-STEPS.md)
