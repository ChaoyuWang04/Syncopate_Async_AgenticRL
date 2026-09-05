---
name: two-foundational-bugs-2026-08-18
description: "2026-08-18 查出两个基石级 bug（梯度没跨 rank 同步 · 权重从没推给 rollout），把 08-14 至 08-18 所有 RL 实测污染了；作废登记在主线历史归档 21"
metadata:
  node_type: memory
  type: project
  modified: 2026-08-18T15:00:00.000Z
---

**2026-08-18 一天里查出两个基石级 bug，它们静默跑完了整整两轮训练（M7 + M7-b）。**

```
E21  三个 trainer rank 的梯度没有 all-reduce ⇒ 每次更新只用 1/3 的数据，各训各的 LoRA
     根因：verl `fsdp_size=1` 造出 (N,1) 退化网格 ⇒ PyTorch 把 HYBRID_SHARD 降级成
           NO_SHARD，却把梯度归约留在了**大小为 1 的组**上，只打一行 UserWarning
E22  trainer 的权重**从没推给 rollout engine** ⇒ π_rollout 全程钉在 π₀
     根因：`engine_workers.py:698` 在 disaggregated 分支只调一次 `get_per_tensor_param()`
           ⇒ `base_sync_done=False` ⇒ `collect_lora_params` 显式跳过所有 lora_ 张量
           （colocate 那条路调两次，是对的）
```

## ⛔ 后果：所有 RL 的"结论"作废

**「lr 被夹在两堵墙之间」「ESS 崩塌」「异步的代价」这一整套叙事，量的是这两个 bug。**
那堵"异步的 ESS 墙"不是异步的代价 —— 陈旧度不是 1–2 个版本，是"从头到现在"。

⇒ **作废登记的历史来源：`docs/archive/syncopate/pre-consolidation-v16/21-invalidated-numbers.md`**
（七个根因 B1–B7 · 22 条机器可读的作废清单 · **§3 仍然有效的部分**）。
⇒ 判据强制：`check_pipeline_invariants --only quarantine` ——
含作废数字的文档顶部必须挂 ⛔ 横幅。**作废的数字不删**（删了就没人知道当初错在哪）。

⚠️ **过度作废的代价和引用错数字一样大。** 仍然有效的：**SFT 线全部**（单进程单卡，
碰不到这两个 bug）· 数据侧全部 · 静态代码事实 · 产物比对 · infra 的硬件/通信测量。
★ 最难防的是**评测审计**：测量有效，**解读无效** —— 它评的是"用 π₀ 生成的数据、
按 1/3 batch 训出来的 rank_0 权重"。

## ★★★ 为什么它们能静默跑完两轮：判据缺了一整族

**此前所有停止条件问的都是「模型学得怎么样」（ESS / 熵 / grad_norm / cap），
没有一条在问「训练系统还活着吗」。**
打个比方：仪表盘上全是车速油耗转速，没有一个灯告诉你方向盘和车轮之间的连杆断了。

⇒ 已补上 A 族（历史见 `docs/archive/syncopate/pre-consolidation-v16/06-rl-run-protocol.md §2.A`；现行规则见 `docs/syncopate/04-TRAINING.md`），**全是「两个东西应当相同」型**：

```
A1 每次权重同步后 `rollout_corr/kl` 必须回落到**首步那个数量级**（数值失配地板）
   ★ 主线独立佐证 E22 就是靠它：实测从第 3 个 param_version 起再没回落，末尾 36×
   ⚠️ 同步在工作时它应当是**锯齿**；平滑单调上升 = 没生效
A2 第 1 步各 rank 的 `lora_B` 梯度逐位相同（B 零初始化 ⇒ 起点必然一致）
A3 训练/评测契约五元组相等 (max_prompt, max_response, max_turns, top_p, top_k)
A4 **绝对有效条数** N × ESS/N ≥ 24（不是比例 —— 0.3 隐含大 batch 假设）
```

⚠️ 而**守卫此前不在仓库里**（跑时临时写的 shell，跑完就没，且只盯"模型学得怎么样"）
⇒ 这是它们能漏过去的另一半原因。现在是 `scripts/tools/rl_guard.sh`。

## ⇒ 三条可迁移的纪律

1. **判据要写在「两个东西应当相同 / 某集合应当完整」的地方**，不要写在
   「这个数应该在某范围里」——前者非黑即白、不需要阈值、不会因基线漂移失效。
   ★ 今天六个问题里，抓到它们的全是前者；而 `ESS<0.3` 正是后者的典型。
2. **凡是"我假设 X 成立"的地方，写成断言。** E21 就是被一句顺手写的
   「DDP 下各 rank 的 LoRA 应该相同」炸出来的 —— 而当时另外两个读 ckpt 的脚本**没有这句**。
   ⇒ 保护性逻辑必须**提成一份函数，所有路径共用**（`syncopate/train/ckpt_guards.py`）。
3. **删证据之前先把它变成数**：`syncopate/train/ckpt_fingerprint.py` 把 27 GB 的
   跨 rank 证据提成 KB 级 json，检查器还能继续读它 ⇒
   「为了省空间把证据删了」这条路被堵上。

相关：[[project-mechanism-not-wired]] [[blank-thresholds-are-not-passes]]
[[rl-step-size-is-lr-times-steps]] [[infra-line-state]] [[syncopate-docs-map]]
