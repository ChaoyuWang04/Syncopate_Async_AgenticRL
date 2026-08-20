# E14 · 执行层优化工具箱（消泡 / 编译 / CUDA graph / 算子融合）

> 状态：🟡 第一批（归因）完成   最后更新：2026-08-20

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | B-④ 执行层工具箱；JD：字节 C「编译技术」· DeepSeek G |
| **需求从哪来** | E01 §4.6：计算时段内 GPU 仍有 22–25% 空档，成因未归属（A18）；Chaoyu 08-20 立意「用真实负载学会什么时候该用什么工具」 |
| **问题** | 空档是什么构成的？compile / CUDA graph / 算子融合各自的适用条件是什么？ |
| **答案（第一批）** | 空档**不是发令开销**：trainer 的空全在**大洞**（>10ms 档占 8.7/9.0s，最大单洞 2.65s），且 **85% 落在 update_actor 内部**——洞内 CPU 在做同步乒乓（单洞 `cudaStreamSynchronize`×144 / 1.76s）。rollout 侧相反：**微空隙**（10–100µs 档 2.0s，7.9 万 kernel）= 发令开销真实存在。⚠️ **首采窗口错位待复核**（§7-窗口）：三采（e14_anatomy2，真稳态中段窗）复核后此卡片再定稿 |
| **信心** | 低-中：**首采窗落在跑的尾巴**（末 1–2 步+最终存档+善后生成+拆机），洞归因可能被最终存档的 D2H 同步污染；2+1 冒烟拓扑。三采复核中 |
| **推翻了什么** | 无（E01「卡是满的」已在 §4.6 自我修正过，本次是给那 22–25% 定性） |
| **下一步** | 第二批：① 定位 144 个 sync 的调用点（torch profiler 带栈/py-spy）② vLLM 关 enforce_eager 探针 ③ compile 微基准 |

## 1 · 问题与预测

**预测（跑前写死）**：空档主要来自 kernel 发令间隙（CPU 发号慢）⇒ CUDA graph 应有效。
**实际**：trainer 侧完全相反——微空隙仅 0.17s，空档是**少数几个多秒级大洞**；
CUDA graph 对 trainer **预判死刑**（它治千刀万剐，我们的病是几个大窟窿）。
rollout 侧预测成立（微空隙 2.0s）。⇒ **同一台机器、同一份负载，两侧病因相反**——
「工具×位置」而不是「工具×机器」。

## 2 · 环境指纹

```
2026-08-20 · fully_async 2+1（GPU1-3；GPU0=主线端点）· PG开+KL关（生产默认）· steps 32
nsys --delay 540 --duration 240 --trace=cuda,nvtx,osrt · NVTX 阶段标注（--nvtx，driver 5 处替换）
窗口实际 33.4 s，覆盖稳态 step 4–8 · trace logs/nsys/e14_pg_anatomy.nsys-rep（166 MB）
数据 _audit/infra/e14_pg_anatomy.json · 尺子 scripts/analyze_nsys_step.py + 本报告 §4 的洞归因查询
```

## 3 · 方法

analyze_nsys_step.py（阶段归属 + gap_analysis）→ 补两个 sqlite 查询：
① 每个 >10ms 大洞与 NVTX 阶段区间求交归属；② 最大洞窗口内的 CUDA runtime / OSRT API 聚合。

## 4 · 数据

```
按进程忙占比（窗内）  trainer-A 45.4% · trainer-B 71.9% · vLLM 42.1%
trainer-A 空档分档     <10µs 0.037 · 10–100µs 0.128 · 0.1–1ms 0.05 · 1–10ms 0.029 · **>10ms 8.712 s**
trainer 大洞归属       17 个大洞 8.71s：update_actor 内 **7.39s** · old_log_prob 内 1.30s · 阶段间 0.02s
最大洞（2.65s）内 CPU  cudaStreamSynchronize ×144 = **1.755s** · 发令类合计 <0.1s
vLLM 空档分档          10–100µs **2.01s**（79,421 kernel）· >10ms 0.71s
NVTX 阶段（kernel 时间）old_log_prob 44.4% · update_actor 34.1% · gen 13.5%
  ⚠️ 与墙钟口径（update_actor 54.3% > olp 18.8%）反差 ⇒ update_actor 的墙钟里有大块非 GPU 时间——与大洞归属互证
两 rank 不对称          trainer-A nccl 占 kernel 13% vs B 3%（A 在归约上等 B）
kernel 构成             GEMM 全部仍是 cutlass_80（FP8 任务的钩子再确认）· elementwise 27–31%
```

## 5 · 工具判决（第一批后的边界表雏形）

| 工具 | 判决 | 依据 |
|---|---|---|
| CUDA graph（trainer） | ⛔ 预判死刑 | 微空隙合计 <0.2s，病不在发令 |
| CUDA graph（vLLM/生成端） | 🟢 升级为有据探针 | 微空隙 2.0s + 现状 `enforce_eager=True`（graph 从没开过） |
| torch.compile | 🟡 待微基准 | elementwise 占 kernel 27–31% 是真靶子；但它治不了 sync 乒乓大洞 |
| **新靶子：sync 乒乓** | 🔴 第二批首项 | 单洞 ×144 次 streamSync；先找调用点再谈工具（可能是 .item()/D2H/我们某补丁） |

## 6 · ⛔ 推翻了什么

> **原猜想**：空档 ⇒ 发令开销 ⇒ CUDA graph。**实测**：trainer 病因是 CPU 同步乒乓的大洞。
> **教训**：「空档」不是一种病——先按长度分档、再按阶段归属、最后看洞内 CPU 在干嘛，三步之后工具才可选。

## 7 · 踩的坑

- ★★ **窗口错位两连**（同一天两次，方向相反）：
  ① delay 300 ⇒ 窗整个落在**启动段**（nsys 把启动拖慢 ~2 min）；
  ② delay 540 ⇒ 窗落在**尾巴**（步进只有 ~200s：save-freq 999 下每步 25s，
    比按"每步都存档"的冒烟外推快一倍多）⇒ 二采里「trainer→vLLM 交替」是**收尾伪影**
    （末步 + 最终存档 + rollouter 善后生成 + 70s 拆机），撤回该图读；
    洞归因也可能混入最终存档的 D2H 同步 ⇒ 三采（delay 460/duration 120，钉中段）复核。
  ⇒ 教训：**采集窗的对齐必须事后用日志时间戳验证**（步进起止 vs 窗起止四个点都要对上），
    "窗口在稳态"不能靠估算宣称——这本身就是覆盖数问题（E26 坑 5 / 本周第三次同形）。
- sqlite 导出 4.6 GB、超 2 min（放后台）；分析完即删（守则⑧）。
- osrt 的 futex/epoll 合计秒数是**多线程求和**，会远超墙钟——只能看构成不能当时长读。

## 8 · 下一步（对上 01 §1-1 工具箱的第二批）

- ① sync 乒乓定位：torch profiler(with_stack) 或 py-spy 对准 update_actor 的一个 micro-batch —— 洞的调用点
- ② vLLM enforce_eager 考古 + 关闭探针（30 min；先弄清当初为何设 True，launch_rl:305 注释块只讲了 max_num_seqs）
- ③ compile 微基准（先 update_actor 前向段），带三判据（rank 逐位/logprob 和/四常驻）
- ④ 生产拓扑（3+1）复采一次做成立范围校验（等 GPU0 空）
