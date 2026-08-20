# E14 · 执行层优化工具箱（消泡 / 编译 / CUDA graph / 算子融合）

> 状态：🟡 第一批（归因）完成   最后更新：2026-08-20

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | B-④ 执行层工具箱；JD：字节 C「编译技术」· DeepSeek G |
| **需求从哪来** | E01 §4.6：计算时段内 GPU 仍有 22–25% 空档，成因未归属（A18）；Chaoyu 08-20 立意「用真实负载学会什么时候该用什么工具」 |
| **问题** | 空档是什么构成的？compile / CUDA graph / 算子融合各自的适用条件是什么？ |
| **答案（第一批定稿）** | **rollout/vLLM 侧（四采完整数据 ✅）**：84.2s 里忙仅 29.1s（35%）；**10–100µs 微间隙 ×796,689 个、合计 32.4s——比计算本身（29.1s）还多**；另有 >10ms 等待 17.7s（等下一批）。⇒ eager 模式的发令/调度开销是生成端头号损失，enforce_eager/CUDA graph 探针升格为**最大单点机会**。**trainer 侧**：大洞非微隙 + 洞内 sync 乒乓（首采观察），但 nsys 对 Ray trainer 进程存在**不可修复的事件截断**（三采/四采两次实锤，flush-interval 无效）⇒ 按预立判据放弃 nsys 路线，trainer 侧交给 torch profiler（第二批①，本来就要用它带栈定位 sync 调用点） |
| **信心** | vLLM 侧高（完整性判据 ✅ + 三/四采互证 33%/35%）；trainer 侧低（仪器截断，结论待 torch profiler 复核） |
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

### 4.0 四采定稿（3+1 生产拓扑 · `--trace=cuda,nvtx --cuda-flush-interval 100` · 完整性判据执行）

```
vLLM::Worker（完整 ✅，1,351,931 kernels）  跨度 84.2s · 忙 29.1s（35%）
  微间隙 10–100µs   ×796,689   合计 32.40s   ← 头号损失，超过计算本身
  <10µs             ×474,145   合计  0.46s
  1–10ms            ×  1,263   合计  3.71s
  >10ms             ×     24   合计 17.67s   ← 等下一批（供给/流水线）
trainer 三 rank：两个 20s 处截断、一个 67s 处截断（NVTX 显示训练仍在步进）
  ⇒ flush-interval 未修复 Ray trainer 进程的事件丢失 ⇒ 按预立判据弃 nsys，转 torch profiler
```

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

## 4.5 第二批 · Test A：vLLM CUDA graph A/B（2026-08-20，同尺子 24 gstep，唯一变量 `--enforce-eager`）

```
                     eager(对照)      graph(处理)        Δ
s/gstep                12.84            11.89          −7.4%
gen（trainer 等题）     4.10 (28.3%)     2.75 (23.8%)   −33%
update_actor / olp     6.82 / 2.31      6.79 / 2.37    持平（受控性 ✓）
graph 捕获判据          —               Capturing 行 ×31+，FULL_AND_PIECEWISE
显存                    —               无 OOM（担心未兑现）
```
⇒ E14 批1 的微间隙预测兑现：生成端提速 → 队列回填快 → gen 等待 −33%。
⚠️ 单种子短跑；要进默认值需任务配对闸 + 正式跑复验（enforce_eager 的历史动机
已确认是 colocate sleep/wake 时代遗产，分家后不适用）。

## 4.6 第二批 · Test B：闸门放宽（2026-08-20，64 gstep，`--sync-every 16 --staleness-threshold 0.5`，eager）

```
                     对照(sync4/0.1)    闸门臂(sync16/0.5)     Δ
s/gstep                12.84              9.77               **−24%**
gen（trainer 等题）     4.10 (28.3%)       0.06 (~0.6%)       **接力赛消灭**
update_actor / olp     6.82 / 2.31        6.73 / 2.34        持平（受控 ✓）
陈旧度剂量              0（从未出现）       stale 计数 1152/3072 条（37.5%）——**首次真实出现**
ESS（会动的仪表）        0.92 中位（cand）   0.80–0.92（中位 ~0.84）——降但远高于 0.5 停机线
probs_diff mean         0.9–1.3e-3         1.2–2.4e-3（~1.5×，绝对值仍 1e-3 量级）
partial_ratio           0                  0（轨迹仍不跨版本——同步少了反而更完整）
```
⇒ 「假异步」的三道闸门里拧开两道，**流水线重叠兑现**：trainer 等题从 28% 掉到 <1%，
端到端 −24%。新构成：三次前向占 ~93% ⇒ 靶子彻底回到训练侧（FP8/乒乓/compile）。

**学习质量闸判决（同日，两臂 ×64 步 + 冻结评测配对）**：
```
SFT 基线 0.711 → 控制臂(sync4) 0.874（+0.163, t=15.1）→ 闸门臂(sync16/0.5) 0.844（+0.133, t=12.2）
臂对臂配对：−0.030（t=−4.3，MDE 0.014）⇒ 同步数下学习质量**显著退化**（显著变差 55 题，
集中 CONF/FAIL/FRESH/RAG；有梯度格子 125→159 反而更多——探索开了但没学扎实）
```
**★★ 行为维度的真实伤情比均值重得多（常驻行为判据抓获）**：
```
该 defer 率   92% → 64%（−28pt，门槛 10%）🔴 拒绝能力系统性退化
模板级重灾区  FRESH −0.210 · CLAR −0.078 · CONF −0.065 · SCALE −0.051 · ATTR −0.041
模板级受益    BUD +0.053 · MISS +0.033 · REJ +0.031
cap 混合      risk_blocked_write −16 · unauthorized_write −14（变好）· weak_attribution +6（变坏）
```
⇒ 陈旧度的伤集中在**判断/拒绝类行为**（时效判断 FRESH、该不该做 defer）——正是这条业务线的
核心能力。与 defer 崩塌家族历案不同源（本次两臂预算全同，唯一变量=陈旧度），机制待究。

⇒ **⛔ 按现值不晋级默认（双重理由：均值 −0.030 + defer 行为红线）**。
R2 剂量曲线开张（两点已落），开放问题：
① 中间剂量（sync 8 / staleness 0.25）找质量无损点；
② **等时对比**——每墙钟秒学习量粗算 +7%，但 defer 崩塌未必是"多跑几步"能补的，等时臂见分晓。

## 4.7 第二批 · 乒乓元凶名单（torch-prof + 栈对齐，单次 update_actor，rank 0）

```
总账：cudaStreamSynchronize ×912 = 2.17 s CPU 罚站 · .item() ×1917 · 拷贝 ×9403
按【真实同步数】排（判据 = 该栈同时出现 cudaStreamSynchronize 行）：
×584   prefix_grouper/utils:4 batch_repeat_cat → repeat_interleave(tensor)
       ← PG 依赖库：repeat_interleave 传张量 repeats 必须同步取输出尺寸
       （解法 = output_size=suffix.shape[0]，cat_dim=1 下零同步可得）→ **修理②已落**
×100   verl padding no_padding_2_padding / F.pad（GPU 标量当尺寸）→ 候补④（上游，~0.24s）
×96    syncopate/train/verl_patches.py:1636 _to_jagged 逐行 .item() → **修理③已落**
×132   杂项未归属
⛔ **Adam 无罪释放**（差点为不存在的问题动刀，E11 分母教训同款）：
   ×1008 的 adam _get_value .item() 是 **CPU 张量**读取（torch 原生 AdamW 无
   fused/capturable 时 step 住 CPU），不产生 GPU 同步——账本上它没有配对的
   Synchronize 行，1008 次合计仅 4.4ms。**按 _local_scalar_dense 计数排凶手会冤枉它，
   必须按"同栈 Synchronize"配对判罪。**
```
修理开关：`SYNCOPATE_FIX_PG_RI` ②·`SYNCOPATE_FIX_JAGGED` ③（默认开，=0 对照）；
A/B 四臂（off/③/②/②③）各带 torch-prof 账本对拍 + 全修后 64 步测速——夜跑 phase2。

## 4.8 R2 剂量扫描定稿（2026-08-20 夜，5 臂 ×64/80 步 + 冻结评测；参照 ctrl64=0.874 / defer 92%）

| 臂 | 均值 Δ vs ctrl | defer | vs SFT | s/gstep |
|---|---|---|---|---|
| s8/0.25 | −0.012（无差异） | −33pt 🔴 | +0.151 | — |
| s8/0.1 | −0.030（t=3.8） | −58pt 🔴 | +0.133 | — |
| **s16/0.1** | −0.012（无差异） | **−3pt 🟡 门槛内** | +0.151 | **9.57（−25%）** |
| s16/0.5（64 步） | −0.030（t=4.3） | −28pt 🔴 | +0.133 | 9.77 |
| **isotime80**（s16/0.5×80 步等时） | **+0.008（无差异）** | **✅ 无退化** | **+0.171（全场最高）** | 9.77 |

三条定论：
1. **杀接力赛的是同步频率，不是陈旧度阈值**（s16/0.1 拿到全部吞吐收益 9.57 且陈旧样本仅 7%）；
2. **defer 对剂量完全不单调**（−33/−58/−3/−28）⇒ 64 步单种子下 defer 是刀锋态、
   跑间方差主导——**任何臂对臂的 defer 因果结论都无效**，晋级判断只认多次一致的读数；
3. **等时论证成立**：即便最激进档（0.5），墙钟对齐后质量全额补回（+0.008 无差异、defer 干净、
   vs SFT +0.171 全场最高）——「快 24% × 多跑 25% 步 = 又快又好」。
⇒ 晋级候选 = **s16/0.1**（同步数质量已净）；多种子复核后切默认。

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
- ★★ **nsys 的 per-process 事件截断**（三采实锤）：三采窗口对齐完美，但 rank B/C 的 kernel
  流在窗口第 12 s 戛然而止、之后 93 s 零记录——而训练健康跑完 48 步（DDP 下两 rank 真闲
  会死锁归约）⇒ **是采集断了不是 GPU 闲了**；二采「rank 不对称（nccl 13% vs 3%）」同签名，
  该观察连同其洞分析一并降级存疑。修法 = `--cuda-flush-interval 100` + 砍掉 osrt（事件量
  最大的源）。⇒ **trace 自己要过判据：每个进程的末 kernel 时间 ≈ 窗末，不过不许读**——
  今天第三个"仪器先于结论翻车"案例（窗口×2 + 截断×1），全部由"结果物理上不可能"暴露。
- sqlite 导出 4.6 GB、超 2 min（放后台）；分析完即删（守则⑧）。
- osrt 的 futex/epoll 合计秒数是**多线程求和**，会远超墙钟——只能看构成不能当时长读。

## 8 · 下一步（对上 01 §1-1 工具箱的第二批）

- ① sync 乒乓定位：torch profiler(with_stack) 或 py-spy 对准 update_actor 的一个 micro-batch —— 洞的调用点
- ② vLLM enforce_eager 考古 + 关闭探针（30 min；先弄清当初为何设 True，launch_rl:305 注释块只讲了 max_num_seqs）
- ③ compile 微基准（先 update_actor 前向段），带三判据（rank 逐位/logprob 和/四常驻）
- ④ 生产拓扑（3+1）复采一次做成立范围校验（等 GPU0 空）
