# Infra 线交接（独立于主线训练）

> 更新于 **2026-08-13 晚（关机前）**。给下一个上下文窗口。
> **分工**：主线训练（数据/SFT/RL 里程碑）看 `../syncopate/05-handoff.md`；
> **本文档只管 infra 线**——多卡并行、异步 RL、通信、kernel、框架/模型选型。
> 按 Chaoyu 的约定：**短，只保证下一个窗口能接上**；细节全部指向对应文档。

---

## 0 · 三十秒读懂

infra 线的目标：**把 4×5090 压榨到极限**，亲手感受每种并行的开与不开，
最终在同框架下训完 dense（收尾中）和一个 MoE（决策已定，探针未跑）。
实验以 **E 编号报告**组织（`README.md` 有索引、模板、编号规则），
**框架是每份报告里的受控变量，不是目录结构**——因为我们可能换后端甚至换框架。

## 1 · 已定决策（别再重新讨论，论证都在指向的文档里）

| 决策 | 结论 | 详见 |
|---|---|---|
| 框架 | **verl 不换**（抛开沉没成本重选仍是它：sm_120 纯 PyTorch 路径 + 三档异步 + agentic rollout 三条全过） | E07 §1 |
| MoE | **GLM-4.7-Flash（30B-A3B，MIT）+ LoRA + GSPO**；三摆法对照：FSDP 分片 / QLoRA 4bit 复制 / EP（toy→Megatron 探针） | E07 全文 |
| 训练侧并行 | **DDP 必选**（`--fsdp-size 1`）。实测 3 卡 FULL_SHARD 1182 s/步 vs 单卡 198 s——**多给卡慢 6 倍**；DDP 3.00× 线性 | 记忆 + 08 文档 |
| attention | **`flash_attention_2` 默认**（2026-08-13 晚起）。真轮子已装（见 §2），sdpa 只做对照/排障 | launch_rl 注释 |
| dynamic_bsz | 默认关（sdpa 时代实测 2.19× 倒退，根因查实）。**FA2 装上后要重测，大概率翻正** | §3-③ |
| NCCL | 多卡自动 `NCCL_CUMEM_ENABLE=0`（`NCCL_P2P_DISABLE` 无效，根因是 P2P 缺失×Ray 单卡可见性） | 分布式文档 §2.0.1 |
| 分卡 rollout 显存 | `gpu_util 0.75`，不能 0.85——权重同步 bucket（2048MB×双缓冲）住在 rollout 卡上 | 分布式文档 §7.3 |
| slime/Miles | 不当底座当教材；Megatron 消费卡有先例（老师 4×4090），sm_120 待探针 P4 | E07 §1.2 |

## 2 · ★ flash-attn：垫片已退役，真轮子已装（2026-08-13 晚）

```
flash_attn 2.8.3+cu128torch2.9-cp312（mjun0812/flash-attention-prebuild-wheels v0.9.0）
轮子存档:  /workspace/wheels/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
验证过:    cuobjdump 确认 sm_120 kernel 在内（80/90/100/120 各 72 个 cubin）
           GPU 冒烟: varlen 对拍逐序列 SDPA 差 7e-3（bf16 量级）; 跨序列隔离精确 = 0
换机器:    直接 uv pip install 这个轮子; scripts/install_flash_attn_shim.py 已打 ⛔ 头
```

**为什么重要**（读码+CPU 实验查实，详见 launch_rl 的 dynamic_bsz 注释和主线 handoff §7.3 后记）：
垫片时代被迫走 sdpa，verl rmpad 的 `attention_mask=None`+打包 position_ids 约定
是为 varlen 设计的 ⇒ transformers 4.57 恒物化 `[1,1,L,L]` mask（**连单序列都物化**，
`find_packed_sequence_indices` 永不返回 None）+ SDPA 有显式 mask 丢 fused 后端
+ 打包时跨序列白算 3.2×。⇒ **不打包的旧基线也在慢路径上，FA2 是双份收益。**

## 3 · 今天的实测数字（所有后续实验的分母）

| 量 | 值 | 出处 |
|---|---|---|
| 卡间 all-reduce 上限 | **6.4 GB/s**（`CUMEM=0`；socket 退化只有 2.1） | 分布式 §2.0.1 |
| ① DDP vs FSDP | 3 卡 FULL_SHARD 1182 s / 单卡 198 s / DDP **3.00× 线性** | rl_ddp.log |
| ② one_step_off | ✅ 跑通并调优（3+1，rl_best.log 是当前最优配置的完整跑） | logs/ |
| ③ dynamic_bsz（sdpa） | update_actor 84.5 s → 184.9 s（**2.19×倒退**，根因见 §2） | rl_1t_dyn.log |
| ④ 权重同步耗时 | one_step_off `update_weights` **13.3 s（占步 27%）**；fully_async `param_sync` **24.0 s**。LoRA 只 132MB，**时间不在传输上，未查因** | logs/ |
| ⑤ fully_async | ❌ **上游 bug 崩在 step 0**：`detach_utils.py:153` `param_version_diff` 对 None 做减法（TypeError）。M7 主线想用 fully_async 跑 150 步被它挡住 | m7_fullyasync.log |

## 4 · 下一步（按优先级，新窗口从 ① 开始）

1. **FA2 三点对照**（~10 分钟，GPU 空闲即可跑）：
   `①sdpa+静态 84.5s（已有）→ ②FA2+静态 → ③FA2+dynamic 16384`。
   预期 ③<②<①；写成 E 报告（编号往后取），dynamic_bsz 若翻正改回默认开。
2. **fully_async 崩溃修复/绕过**：读 `verl/experimental/fully_async_policy/detach_utils.py:153`
   前后，查 `param_version_end` 为何是 None（疑与 partial_rollout 或 agent loop 不上报
   param_version 有关）。修不动就 monkeypatch 跳过该 metric——**M7 主线等着它**；
   或 fallback：M7 用 one_step_off 跑（已调优、稳定）。
3. **colocate 同机基线**（30 分钟）：所有加速比现在没有同机分母。
4. **E00 补齐**：4 卡 NCCL 曲线 / 主机内存带宽 / **满载降频**（2.3kW，会污染一切对照）/ PCIe 负载下代数。
5. **E01**：nsys 拆一步 + **纯训练 microbench**（后面 E02–E07 的跑台；④ 的查因也挂在这）。
6. **E07 探针 P1–P6**（GLM-4.7-Flash 下载、vLLM 加载、4bit mismatch ESS、TE 编译后台挂、AReaL-lite、bnb）。

## 5 · 新窗口阅读顺序

```
本文档 → README.md（E 索引/模板/纪律）→ E07（MoE 决策全文）
→ ../syncopate/08-machine-and-environment.md（环境怎么跑起来）
→ ../distributed-training-design-v0.1.md（多卡实验设计 + 四堵墙）
→ ../ostinato-project-design-v0.2.md（单卡时代结论，§4.0 被推翻的因果链）
→ ../llm-rl-framework.md（框架全景调查，选型的背景知识）
```

⚠️ 关机重启后：`/workspace` 是网络盘会活着（venv/模型/轮子/文档全在）。
Claude 记忆的实体就在项目内 **`.claude/memory/`**（已进 git；`/root/.claude/projects/
-workspace-Syncopate-Async-AgenticRL/memory` 是指向它的软链接）——重启后若软链接没了，
重建它即可：`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`。
Ray 集群不会自启，直接跑 launch_rl 即可（它自己 ray.init）。
