# E32 · B-4 收尾包：四卡分布式 serving 压测与优化

> 状态：✅ **单日收官（2026-08-28，S0–S5 全档）**   最后更新：2026-08-28
>
> 立项：Chaoyu 2026-08-28（B-4 展开；单任务制唯一在办）。四项裁定已定：
> ① 简历空格三格 → **四格**（新增多卡拓扑/投机一格）；② after 主口径 = **goodput@SLO**，
> 饱和 tok/s 作辅；③ PD 探针**接受账面判定收口**，不追加实跑投入；④ trace 数据集用
> 真实 rollout prompt 落盘（只进 `_audit/`，不出库）。
> 前提变更（Chaoyu 08-28）：训练线已闭环、**训练与 serving 分时共用整机（colocate 不并发）**
> ⇒ serving 可用全部 4 卡——B-4 从"GPU0 单卡作业"升级为**整机作业**。

## 0 · 结论卡片（收工时填）

| | |
|---|---|
| **Track / 兑现物** | A（推理家族简历 §3 四个〔B-4〕格）+ 主线 `11 §5` after 回填 |
| **需求从哪来** | E19-c 实测：48 并发下 bf16 KV 池装不下一半请求，TTFT P99 23.6s 全是排队 ⇒ 瓶颈是**容量**不是速度；旧 before 只用 1/4 的机器 |
| **问题** | 4×5090（无 P2P）上业务负载（prefill 重 88%·prefix 命中 97%·输出短）的最优 serving 结构是什么；批调度参数、PD 分离、投机解码各值多少 |
| **答案** | **4×独立引擎 + 前缀亲和 router**（内建 DP×LoRA 上游判死 ⇒ 自研补位；重负载扩展 **3.66–3.86×**、突发 TTFT P90 −40%）；goodput@SLO=**64 并发**且膝点在**编排层**（单卡/四卡逐级等值证明，引擎从不是业务约束）；批调度默认已优 + mnbt16384（冷 prefill +5.3%）；**PD no-go**（98% 命中=无物可卸，搬运账 21.5ms 反而便宜）；**ngram 投机全场景采纳**（单流 TPOT 2.3×·48 并发 +41%·接受率 64%·50/50 逐字无损）——后两项已进生产默认并冒烟实跑 |
| **信心** | 高：全部门槛预注册、0.02–2.8% 噪声地板先立、关键数双跑复现（mnbt 逐毫秒/E19-c 逐位级）、PD 与路由各带对照臂 |
| **推翻了什么** | §6：P4 低估 ngram（预测 −10~30% 实测 −57%）；PD 字面 go 被 chunked 两态对照拆穿（负载分摊≠prefill 干扰）；goodput 首跑两处假通过（cost-cap 秒失败×"到过终态"判据）；P2 的 trace 轨扩展 2.84× 是仪器上限非引擎 |
| **下一步** | 无（全线收官）；遗留归属：编排层膝点归主线、ngram 参数面（K/lookup 窗）用当前配方即够 |

## 1 · 问题与预测（跑之前写死，不许事后改）

**P1（S1 批调度）**：调参收益 **< 10%**——chunked prefill 默认已开、负载本就 prefill 重，
默认值大概率已在甜点附近；若有收益，最可能来自 `max-num-seqs` 上调配合 fp8 KV 大池。
若实际 >10%：说明默认调度对"短输出×高命中"负载有系统性错配，要定位到具体参数。

**P2（S2 拓扑）**：4×DP 饱和吞吐 **3.4–3.9× 单卡**（引擎间零通信，损耗来自 host CPU/
tokenization 共享与 2+2 跨 socket）；**真实 trace 下亲和路由显著优于 round-robin**
（cache 命中回到单卡水平，TTFT 中位差可见），**random 流量下两者无差**（cache 无用武之地
——这一对照本身就是"为什么要亲和路由"的证据）。若扩展比 <3.2×：先查 NUMA 绑定，
再查 API server 单点，如实记账。

**P3（S3 PD）**：**no-go**。chunked prefill 已吃掉大部分干扰（三态测量会显示带 chunked
的干扰差很小），而 KV 搬运账固定要付 ~300 MB/请求（fp8 72 KB/token × 4200）；
97% 命中意味着可省的 prefill 计算只有 ~3%。成立范围预测：命中率低的负载、
CoT 长输出时代、或有 P2P/NVLink 的机器。若实际 go：说明干扰远比预想大，
要回头查 chunked prefill 是否真在生效（判据行没出现=机制没生效）。

**P4（S4 投机）**：ngram 接受率中等（输出模板化利好、但短输出限制窗口）；
单流 TPOT **−10~−30%**；48 并发**持平或略负**（算力已被 batch 填满，draft 是纯开销）
⇒ 预计结论是**分场景**：单流在线可开、批式饱和不开。若单流也无收益：
说明接受率被工具调用里的自由文本段拖垮，记负结果与复活条件（CoT 时代重测）。

**P5（总账）**：goodput（守住 §19 全表的最大并发）旧 before 在 8–16 之间（8 并发 1.04×、
16 并发 2.43×）；after（4 卡 + fp8 KV + 调参）预计 **≥ 4×**，可能到 6–8×
（fp8 KV 单独 +50% 已实测）。

## 2 · 环境指纹

```
日期          2026-08-28 起
机器          4×RTX 5090 32GB / sm_120 / 驱动 595.71.05 / CUDA 12.8 轮子
torch/vllm    torch 2.9.0+cu128 / vllm 0.12.0
模型          Qwen3-4B（models/Qwen3-4B-sft-v13r2-e1 = SFT merged 真身）
              + LoRA candidate r32（checkpoints/grpo/cand_v13r2_e1/adapter_global_step_25）
              36 层 · 8 KV 头 · head_dim 128 ⇒ KV 73,728 el/token（fp8 ≈72 KB · bf16 ≈144 KB）
并行配置      serving DP=1..4（独立引擎，零卡间通信）；⛔ TP 不用（02 §1：无 P2P 净亏损）
KV 口径       --kv-cache-dtype fp8（serving 侧默认，Chaoyu 08-27 拆两侧裁定；质量 −0.009 已定价）
基线出处      旧 before = 08-20 旧机器 + bf16 KV（11 §5 表；原始日志未随搬家保留）
              ⇒ ★ 新机 fp8-KV 单卡基线必须在 S0 重记，否则收益与换机/换 KV 混账
原始日志      logs/b4/（本实验统一落这里，每臂一份机读 results json）
```

## 3 · 拓扑论证（三本账摘要；细节见 08-28 分析对话，数字出处在括号）

```
拓扑账   无 P2P ⇒ TP 每层 2 次 all-reduce 走主机内存（E04 实测 rollout TP=2 净负 20%，
         02 §1 已裁"模型内并行净亏损"）。⚠️ 新机通信重画像未跑，但见负载账——TP 没有靶子。
负载账   单流 TPOT 6.6ms = 25ms 门槛的 3.8× 余量（11 §5）⇒ 延迟不缺；
         缺的是容量（E19-c：48 并发 TTFT P99 23.6s 全是排队）⇒ 正解 = DP 复制，KV 池 ×4，
         卡间零通信把无 P2P 短板完全绕开。
PD 账    每请求 KV ~300 MB（fp8）走主机内存两跳固定要付；97% 前缀命中 ⇒ 可省的 prefill
         计算仅 ~3%——收益趋零、成本不减 ⇒ 倾向 no-go，S3 探针给数。
DP 自付  同一 4k system prompt 被 4 副本各缓存一份 ⇒ 前缀池稀释
         ⇒ 路由器做前缀亲和（一致性哈希定副本），S2 用 A/B 量化这笔账。
投机     MTP 需模型自带多 token 头（Qwen3-4B 无）、EAGLE 需训头（收尾期不开训练任务）
         ⇒ 可行探针 = ngram（prompt lookup，vllm 0.12 --speculative-config，零额外模型）。
```

**目标结构**：4 个独立单卡引擎（:8101–8104，冻结的 S1 最优配置）+ 前缀亲和路由器
**监听 :8100**——decider URL / 主线 chatbox / `start_vllm.sh` 调用方全部无感。

## 4 · 施工步骤（S0–S5，每步预注册判据；⛔ 判据行没出现 = 机制没生效）

### S0 · 立尺子 ✅（08-28 上午，数据 logs/b4/base_fp8kv_{s0,r2,r3}/arm.json）

```
trace 数据集   512 条真实 episode（28,416 候选分层采样）：prompt p50 4151 tok · 输出 p50 552 ·
              全局公共前缀 1905 tok（4409 字符）· KV 292 MB/请求（fp8）——PD 账输入齐
新机单卡基线   （fp8 KV·生产旗子）random 轨 1406.6 tok/s·TTFT 中位 5.10s/P99 10.1s
              —— 与 E19-c 旧机 fp8kv 臂（1409/5.0/10.0）逐位级吻合 ⇒ serving 侧新旧机同水位，
              E19-c 五臂表在新机直接可引；trace 轨 2403 tok/s·TTFT p50 48ms·TPOT 12.5ms·
              prefix cache 命中 98.0%（quries 2.15M=512×4193 账目自洽）
噪声地板 ×3   trace tok/s 离散 0.02%（ignore_eos 定长红利）· random tok/s 0.6% · TTFT 2.8%
              ⇒ 判定门槛（2×地板）= 0.05% / 1.2% / 5.6%
★ 白捡结论    真实流量 TTFT p50 48ms = 98% 命中让 prefill 近乎免费 ⇒ PD 无可卸载物（S3 预演）
```

仪器四件（用法见各脚本头注）：`b4_make_trace.py`（trace 落 `_audit/`，不出库）·
`b4_bench.sh`（单臂双轨：random 对齐 E19-c / trace 才是 cache 尺子）· `b4_replay.py` ·
`b4_stack.sh`（全栈起停）。⚠️ S0.3 的全栈 goodput 基线挪到 S2.4 与 after 同 session 测
（同尺原则）；loadtest 全量 `--skip model_down`（会杀 vLLM 不自动重启）。

### S1 · 单卡批调度调参 ✅（08-28 上午；P1 命中——真实流量下默认已优）

```
五臂 OFAT 全表（trace 轨 tok/s·对基线 2403）：mnbt2048 2406 · mnbt16384 2402 · mns256 2401 ·
util085 2403 —— 全在地板内；mns64 2341（−2.6% 真实劣化，排除）
★ 唯一胜手 = --max-num-batched-tokens 16384：random（冷 prefill）轨 +5.3%（1483/1482 双跑
  互证，差 0.1%）· TTFT 中位 −12.4%（4466.05/4466.41ms 逐毫秒复现）· trace 轨零代价
⇒ 〔批调度参数〕定案：真实 98% 命中流量对批调度不敏感（chunked prefill 默认粒度已优）；
  fleet 统一配置 = 生产旗子 + mnbt 16384（买的是冷 prefill/缓存失效场景的健壮性）
```

（OFAT 网格与判据见 `b4_sweep.sh` 头注；胜出配置已冻结为 S2 fleet 统一配置。）

### S2 · 4×DP + 路由 ✅（08-28 上午；胜出拓扑 = 4 独立引擎 + 亲和 router）

```
⛔ 内建 DP 判不可用：vllm 0.12 上游自供 "LoRA in DP mode is not supported yet."
   （NotImplementedError 启动即死）⇒ 生产必须带 candidate LoRA ⇒ 自研 router 是必需品不是优化
扩展曲线（random 轨=预注册扩展尺子；trace 轨另注）：
   1 卡 1488 → 2 卡 2838（1.91×）→ 4 卡 aff 5422（3.66×）/ rr 5727（3.86×）tok/s
   ⇒ ≥3.2× 预注册门槛双双通过，无需 NUMA 复查；零卡间通信架构兑现
   trace 轨 2404 → 4332（1.80×）→ 6821/7055（2.84×/2.94×）——短板在重放客户端单进程
   +512 条尾部排空（有限批 drain），已注记不作扩展判据；全臂 512 条零失败
affinity vs rr（4 卡）：cache 命中 95.9-97.6% vs 95.1-95.5%（+1~1.5pt）·
   TTFT p90 1.61s vs 2.70s（**−40% 尾延迟**）· 吞吐 −3.3%；本 trace 的 case 重复率被
   采样稀释（338 case/512 条），生产 8-rollout×多轮下亲和差距只会放大 ⇒ 生产选 affinity
路由器自证：rr 分发 178/179/178/178 完美均衡；aff 分发 176/204/155/178 · 零 case 分裂
```

> ⛔⛔ **本节"膝点与引擎无关（单卡/四卡等值）"的推断于 08-28 下午被 E33 翻案**：
> after 臂的亲和路由对 decider 短 prompt（~3-4k 字节 < 哈希窗起点 4409）越界成空切片
> ⇒ **1713/1713 个请求全塌到 0 号引擎——"四卡臂"实际是单卡臂**，等值是两个单卡的等值。
> 分发计数这行判据当时没被读（只在 trace bench 看过）＝判据行没生效。修正后的真相
> （E33：router v2 自适应窗 + 扩池 + 多进程）：真四卡分流下 C=96 I01 P95 6159→**3030ms
> 全绿**、吞吐 735→1606 runs/min——**引擎分流对业务延迟有效**，"编排层是唯一瓶颈"的
> 强版结论不成立（弱版仍真：C≤64 时单卡也够）。goodput@SLO=64 的 before 数字不受污染
> （before 臂本就是单引擎直连）。全案见 E33 §6。

**S2.4 goodput@SLO ✅（全栈同尺双阶梯 8→384，worker 并发 512 先不设编排瓶颈）**：

```
goodput@SLO = 64 并发（混合四意图，全部合法终态；C=96 破线——I01 P95 5.75s/5.31s 超 5s 预算）
★ 膝点与引擎无关：单卡与 4 卡在每一级几乎等值（C=64: 651 vs 683 runs/min·C=96 同破）
  ⇒ 当前业务 episode（输出中位 80-198 tok·2-4 次串行 LLM 调用）的膝点绑定在编排层
  （worker/API/PG），引擎从来不是约束——"训推分离是显存逼的"式结论在 serving 侧的镜像：
  4 卡买的是引擎层容量（重生成负载 3.66-3.86×·TTFT 尾 −40%·KV 余量），
  是 CoT/长输出时代的头寸；业务层膝点要抬得动编排层，越出本包范围
可见的 after 增益：16 并发劣化 1.37×→1.03× · I01@C=96 5747→5305ms（同破但更缓）
loadtest 服务面子集（lat/comp/conc）before/after 双跑 21/22 同表全绿——唯一红项
  = 08-20 已登记的"判据自己的病"（I01 P50/P95 比值行，待 §19 改判），无新失败
```

### S3 · PD go/no-go 探针 ✅（08-28 午；判定 = **no-go**，机理级）

```
实测三件（logs/b4/pd_*.json）：
① 干扰两态   decode 流 TPOT 7.79ms → 风暴下 13.48/16.54(p50/p99)；⛔ 但 chunked on/off
   两态逐位同（13.34/16.15）⇒ 减速机理是**负载分摊不是 prefill 计算干扰**——98% 命中下
   风暴的 prefill 吞吐 84.4 万 tok/s（几乎全为缓存命中），**没有可卸载的 prefill 计算**
② 搬运账     pinned D2H 26.7 / H2D 26.3 GiB/s ⇒ 292 MB KV/请求 = 21.5 ms——搬运其实便宜
③ 判定       no-go：PD 不是死于搬运，是死于「无物可卸」+ 被已落地的 4×DP 支配
   （DP 同样稀释负载干扰、不砍 decode 容量、零改造）。字面 go 条件曾被负载效应满足
   （4819ms > 21.5ms），两态对照臂把假机理拆穿——预注册时留的 chunked-off 档救了结论
成立范围（翻案条件）：prefill 计算重新变重的时代——缓存冷启动/失效风暴、CoT 长思考、
   单副本装不下的模型或 KV；届时 21.5ms 的搬运账证明无 P2P 通路并不贵
```

### S4 · ngram 投机解码 ✅（08-28 午；**全场景采纳**——预测 P4 被向好推翻）

```
兼容一次通过（ngram × LoRA × fp8 KV × mnbt16384 同开）；A/B（logs/b4/s4/）：
  单流   TPOT 6.76 → 2.94 ms（−57%，2.30×）· tok/s 146 → 324
  并发8  915 → 2012 tok/s（2.20×）        · TPOT 8.57 → 3.90
  并发48 3173 → 4484 tok/s（**+41%**）    · TPOT 14.0 → 10.1
  draft 接受率 63.9%（419,924/656,981 token，vllm 自报计数器）
  无损性 50/50 greedy 逐字一致 ✅
⇒ 预注册门槛（单流 −15% 且 48 并发 ≥−3% 且逐字全过）大幅超越 ⇒ 采纳进生产默认。
★ P4 预测错在低估：预计单流 −10~30%/高并发持平，实测 −57%/+41%——agentic 输出从题面
  抄参数（campaign id/日期/JSON 字段名）正是 prompt-lookup 的甜点区，接受率 64% 兑现；
  高并发仍赚是因为 decode 步数减半省的是调度轮次，不只是算力
配方：{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}
```

### S5 · 收尾手续 ✅（08-28 下午全部办结）

```
① NARRATIVE 四格全填 ② after 回填 11 §5 ③ 生产落地：start_vllm.sh 默认+mnbt16384+ngram
（冒烟实跑：健康✓·sft-base/candidate 双模型出合法 JSON·spec 指标活）；四卡高吞吐模式
= b4_serve_4x.sh（router 顶 :8100 零改动）④ 01 队列清空·MAINLINE 换完成通报行·00-START
收官改写 ⑤ 结论卡片/§6/§7 齐；编排链三份入库 scripts/b4_chain_{s0s1,goodput,s3s4}.sh
```

## 5 · 评估框架总表（口径一次定死）

| 指标 | 尺子 | 角色 |
|---|---|---|
| **goodput@SLO**（守住 §19 全表的最大并发） | b4_goodput.py（全栈） | **简历主数字**（Chaoyu 认可口径） |
| 饱和吞吐 tok/s · TTFT/TPOT P50/P99 | b4_bench 双轨 | 辅数字 · 臂间排序 |
| 扩展线性度 1→2→4 | b4_bench(random 轨) | DP 结构成色 |
| 各副本 prefix cache 命中率 | /metrics 抓取（指标实名以首跑为准） | 路由 A/B 裁判 |
| PD：残余干扰 vs 搬运账 | b4_pd_probe.py | go/no-go |
| 投机：接受率 · TPOT · 并发吞吐 | b4_bench + S4.2 | 采纳/负结果 |
| 质量 | **不重评**：DP/路由/PD 不改数值；fp8 KV 已定价 −0.009（E19 §8）；投机走无损性逐字实证 | 精度闸 |

**口径纪律**：① 一切差异先过 2× 噪声地板（S0.4）；② cache/路由结论只认 trace 轨，
容量/扩展结论 random 轨与 E19-c 可比；③ I01 若 P95 破线，先查是不是"编排收短输出"
未做（11 §5 已知账，归主线，不算 serving 退化）；④ 并发臂 TTFT 是饱和态排队数字，
只作臂间排序不作 SLO 承诺（E19-c 口径注记沿用）。

## 6 · 风险与已知坑（跑前自查）

```
① gpu_gate 先过再起（守则⑦）；先起跑后挂守卫（rl_guard 心跳会喂死静默期——08-27 两次）
② runtime_loadtest 的 model_down 阶段会杀 vLLM 不自动重启 ⇒ 默认 --skip，全量只跑一次且错峰
③ worker 口径：loadtest 用 org_acme 专职 worker；org_demo 常驻 worker 别掺和（09 的抢单前科）
④ 杀 vLLM 要连 EngineCore 子进程（30G 显存不放的前科）；b4_bench 拆服务后等显存 <2G 再下一臂
⑤ 兼容矩阵三处未验（S2.0 / S4.0 各 10 min 冒烟先行）：dp×LoRA×fp8KV · ngram×LoRA · router 转发 SSE/流式
⑥ vLLM prompt_logprobs fp32 尖峰不在其显存预算内（E31 前科）——压测臂不开 logprobs
⑦ 4 引擎共享 host：CPU tokenization 争抢 ⇒ 扩展比不达标先 numactl 绑 NUMA 再归因
⑧ 主线 chatbox 在人手上：压测期间端点会反复起停 ⇒ MAINLINE 行已登记错峰；router 上线后保 :8100 无感
```

## 7 · 踩的坑（施工中随手记，症状→根因→修法）

```
① 生产端点在新机上从来起不来（S0 首跑当场撞上）：start_vllm.sh 随 git 搬了家，但它引用的
   candidate adapter（checkpoints/grpo/cand_v13r2_e1/）没搬——ckpt 不进 git，搬家清单
   只搬了 bases/。vLLM 启动即死 "No adapter found"。修法：HF 资产库 cand_v13r2_e1/step_25
   拉回原路径，判据=safetensors 可开+504 键（与 E29 逐位校验记录吻合）+r32/α64。
   ★ 教训：搬家验收跑了训练冒烟没跑 serving 起点——"脚本在 ≠ 依赖在"，
   00-START §6 搬家清单⑤已补注。
② 门禁等待循环自己往 logs/ 写心跳 ⇒ 静默期永久续住（08-27 rl_guard 同款坑在自家脚本重演）；
   修法=等待期只写 stderr，b4 产物前缀照 e14/e19 先例登记进 gpu_gate 排除表。
③ bash 全角括号 `）` 在 $(...) 里不闭合，静默吞掉后面整个 heredoc，报错位置在 11 行外
   （报错位置≈误导，守则②又一证）。
④ goodput 首跑双假：(a) org_acme 日预算 10M micros 在 ~300 run 处刷爆 ⇒ 其后所有 run
   秒失败 `daily_cost_cap`；(b) 我的判据把"到过终态"当成功——run.failed 也是终态，
   秒失败把 P95 拉成 400ms、runs/min 623→12161 的**物理不可能**才让它显形。
   修法：worker 加 --daily-cost-cap-micros 透传（默认不变，压测 org 抬 1000×）+
   判据收紧为合法业务终态 {succeeded, waiting_for_user} + 终态构成入产物。
   ★ 又是"空门槛为错误的理由通过"：判据必须能对自己失败。
⑤ pg_isready 不在 PATH ⇒ 活着的 PG 被判死（检查器指向不存在的工具）；改 TCP 探活。
⑥ httpx AsyncClient 默认连接池 100：C≥100 的阶梯会在客户端排队污染延迟（b4_goodput
   与 b4_replay 都要显式给 limits）。
```

## 8 · 下一步 / 衍生问题

〔收工填；已知候补：训练线若复活，DP 副本 + trace 压测框架可直接迁给 rollout 扩容〕
