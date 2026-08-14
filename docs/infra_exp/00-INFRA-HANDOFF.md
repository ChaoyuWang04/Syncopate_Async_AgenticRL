# Infra 线交接（独立于主线训练）

> 更新于 **2026-08-14**（目标收窄为两条 track，见 §1）。给下一个上下文窗口。
> **分工**：主线训练（数据/SFT/RL 里程碑）看 `../syncopate/05-handoff.md`；
> **本文档只管 infra 线**——多卡并行、异步 RL、通信、kernel、框架/模型选型。
> 按 Chaoyu 的约定：**短，只保证下一个窗口能接上**；细节全部指向对应文档。

---

## 0 · 三十秒读懂

infra 线的目标（**2026-08-14 收窄**）：不再是笼统的「把 4×5090 压榨到极限」，
而是**做出两个有真实需求支撑、可验证的项目**。判据变成一句话：

> **先有被测量出来的需求，才有优化目标。** 答不上「服务哪条 track 的哪条兑现物」的实验，
> 一律停放，不先做后想。

实验仍以 **E 编号报告**组织（编号是身份，**永不重排**），track 是叠加在上面的索引视图。

## 1 · ★ 两条 track（新窗口从这里进）

```
TRACK-A-hardware-kernel.md   负载形状 × 硬件拓扑 决定该写什么算子
    两把筛子：① agent 负载的监督密度（SFT 实测 3.8%）② 无 P2P、6.4 GB/s 的拓扑约束
    ⇒ 量化的动机是**省通信不是省显存**；kernel 的动机是**负载的稀疏结构**

TRACK-B-framework-async.md   agentic RL 训练系统的框架级改造
    通用 RL 框架的默认假设在 agentic 负载上逐条失效：barrier 的代价（长尾 1.37–2.75×）、
    权重同步「便宜」（实测 36%/步且不是传输）、训练到的=下发的（异步下不成立）、
    staleness 有人记（我们没接上）
```

⚠️ **三个实验已被显式停放**（E04 TP/PP、E05 SP、E06 更大模型）——
理由在 `README.md` §2.1。**不是忘了做，是没有需求指向它们。**

## 2 · 已定决策（别再重新讨论，论证都在指向的文档里）

| 决策 | 结论 | 详见 |
|---|---|---|
| 框架 | **verl 不换**（抛开沉没成本重选仍是它） | E07 §1 |
| MoE | **GLM-4.7-Flash（30B-A3B，MIT）+ LoRA + GSPO**；三摆法：FSDP 分片 / QLoRA 4bit 复制 / EP | E07 全文 |
| 训练侧并行 | **DDP 必选**（`--fsdp-size 1`）。3 卡 FULL_SHARD 1182 s/步 vs 单卡 198 s——**多给卡慢 6 倍**；DDP 3.00× 线性 | 记忆 + 08 文档 |
| attention | **`flash_attention_2` 默认**。真轮子 2.8.3 已装，`/workspace/wheels/`，sm_120 kernel 验证过 | launch_rl 注释 |
| **dynamic_bsz** | 🆕 **建议翻回默认 True**：FA2 下 ÷1.37 提升（sdpa 下才是 ×2.18 倒退）。**符号由 attention 决定** | README §6 |
| NCCL | 多卡自动 `NCCL_CUMEM_ENABLE=0`（`NCCL_P2P_DISABLE` 无效） | 分布式文档 §2.0.1 |
| 分卡 rollout 显存 | `gpu_util 0.75`，不能 0.85——权重同步 bucket（2048MB×双缓冲）住在 rollout 卡上 | 分布式文档 §7.3 |

## 3 · 当前的分母（所有后续实验引用这里，不要各自重测）

完整常量表在 `README.md` §6。三条最要紧的：

| 量 | 值 |
|---|---|
| 卡间 all-reduce 上限 | **6.4 GB/s**（无 P2P，经主机中转；NVLink 的 1/140） |
| 端到端 3 卡 one_step_off | 49.5 → **32.6 s/步**（FA2 + dynamic_bsz，2026-08-14） |
| **权重同步 update_weights** | **11.1–13.6 s 恒定，占步 36%**，与 attention/打包/卡数**全无关** ⇒ 不是计算侧；LoRA 仅 132 MB ⇒ **也不在传输上**。根因未查 → **E12** |

### 3.1 ⚠️ 一条要更正的归因：fully_async 崩溃**不是 verl 上游 bug**

本文档旧版写「上游 bug，修不动就 monkeypatch 跳过该 metric」。**2026-08-14 读码核实，这是错的**：

```
verl  FullyAsyncLLMServerManager.generate  已把 min/max_global_steps 放进 TokenOutput.extra_fields
      （fully_async_rollouter.py:148）
我们  verl_agent_loop.py:208 的 generate() 包装只取 token_ids / log_probs —— 整个 extra_fields 被丢掉
我们  verl_agent_loop.py:266 的 AgentLoopOutput.extra_fields 只填 reward_extra_info
verl  agent_loop.py:991-1004 用默认键集 .get(key) 补成 None
崩    detach_utils.py:153 对 None 相减 → TypeError（logs/m7_fullyasync.log:1005）
```

⇒ **修法是在我们的 agent loop 里透传该字段，不是 monkeypatch 跳过。**
那个 metric 正是 Track B 要量的 staleness 经验分布——**跳过它等于把研究目标删了**。
主线 `../syncopate/05-handoff.md` §2.1 的说法是对的，本文档旧版是错的。
（这是「机制建好了但没接上」的第 N 次，见记忆 `project-mechanism-not-wired`。）

## 3.2 🆕 2026-08-14 的三条状态变更

1. **fully_async 已解封并在跑**（M7：3 trainer + 1 rollout，decoupled + partial_rollout
   + dynamic_bsz 16384）。⇒ §3.1 的 bug 已修；**「把异步跑起来」不再是待办**。
2. **读了 AReaL 论文**（arXiv 2505.24298）：decoupled PPO / 最大陈旧度 η / 可打断 rollout
   **我们已经全有了**（分别对应 `bypass_mode=False` / `staleness_threshold` / `partial_rollout`）
   ⇒ **staleness 那块不是我们的创新点，不能那么写**。
   ★ 但论文**自己承认没有定量测过长度偏差、也没做按长度分层采样**，列为 future work
   ⇒ **这是 Track B 的新重心**，详见 `TRACK-B` §0.6。
   🆕 **AReaL 2.0**（2607.01120，cs.DC，2026-07）也读了：**算法层一处未改**，转做在线服务化
   （ATDP 轨迹协议 / 数据治理 / Evolution Control Plane），**全文无任何评测数字**。
   ⇒ 我们的三条缝隙（长度漂移 / 分池×异步补偿 / 权重同步根因）**它一条都没碰** ——
   最有资格填的团队连着两篇跳过，缝隙被再次确认。
   ⚠️ 但它的框架性主张与 Syncopate 三条前提高度重合 ⇒ **「框架判断对」不再是差异化，数字才是。**
3. **⛔「动态分池在 fully_async 下用不了」是错的** ——
   `fully_async_rollouter.py:464` 确实调 `create_rl_sampler`，且 import 写在函数体内
   （对 monkeypatch 友好）。真原因是**作用域**：它跑在 Ray worker 进程，driver 侧 patch 够不着。
   修法是把 sampler patch 装进已在使用的 `verl_patches.setup_worker()`。**M7 跑完再改。**
   详见 `TRACK-B` §0.7 与 `main_ppo_pool.py` 的模块 docstring。

## 4 · 下一步（新窗口从 ① 开始；GPU 被主线占用，前四条都不需要 GPU）

1. **E08-a · 分布漂移分析**（🟢，Track B）——M7 跑完即有产物：
   `dispatched.jsonl`（发出去的）vs `rollout_dumps/*.jsonl`（训练到的），
   **比长度分布/轮数分布/截断率，不是比均值**。★ AReaL 明确留下的 future work，我们工具现成。
2. **E11-b · 切片对照组**（🟢 写 / 🔴 测，Track A）——按 mask 挑行 → 喂 verl 现成 fused kernel。
   **E11 密度已测：4.17%**，切 prompt 省 8.5×、完整筛省 24× ⇒ **先试最笨的，别直接写 kernel**。
3. **E12-a · update_weights 查因读码**（🟢，Track B）。
   嫌疑：checkpoint engine 的 bucket（2048MB × 双缓冲）是否每次 alloc/free 4 GB？
   是否遍历全部 4B 参数而非 132 MB 的 LoRA？握手/序列化是否串行？
   🆕 M7 首次 `param_sync = 104.24 s`——**等第 2/3 次同步的数**才能把「推基座」和「推 LoRA」分开。
4. **E02/E03/E08 补成报告**（🟢）——数字都在 `logs/`，索引里标着 🟡 但目录里只有 E07/E11 两个文件。
5. **E01 nsys**（🐟）——**下次主线启动训练时带上 nsys，一鱼两吃**。它是所有优先级的裁判。
6. **接上动态分池**（M7 跑完后）——见 §3.2-③。改完**必须有日志证明它在 worker 进程生效**。
7. 攒一个**独占 GPU 窗口**批量跑 🔴：E08 colocate 同机基线（30 min）+ E11-d 三方跑分
   + E14 CUDA graph 盲盒。**别零散抢卡**——每次都要重付「等 GPU + 热机 + 降频不可比」的成本。

## 5 · 新窗口阅读顺序

```
本文档 → TRACK-A / TRACK-B（看你接哪条线）→ README.md（E 索引/模板/纪律）
→ E07（MoE 决策全文）
→ ../syncopate/08-machine-and-environment.md（环境怎么跑起来）
→ ../distributed-training-design-v0.1.md（多卡实验设计 + 四堵墙）
→ ../ostinato-project-design-v0.2.md（单卡时代结论，§4.0 被推翻的因果链 ★ 必读）
→ ../llm-rl-framework.md（框架全景调查，选型的背景知识）
```

⚠️ 关机重启后：`/workspace` 是网络盘会活着（venv/模型/轮子/文档全在）。
Claude 记忆的实体在项目内 **`.claude/memory/`**（已进 git）；软链接没了就重建：
`ln -s /workspace/Syncopate_Async_AgenticRL/.claude/memory /root/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory`。
Ray 集群不会自启，直接跑 launch_rl 即可（它自己 ray.init）。
