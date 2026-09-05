# Infra · 01 · TASKS — 队列（唯一来源）

> **只记「接下来做什么、谁在打、到哪一步」。办完就删行**，结论写进 E 报告，
> 决策原因写进 [`02-DECISIONS.md`](02-DECISIONS.md)。其它文档不许再维护第二份队列。
>
> **排序原则**（沿革见 02 §2）：**① 影响正确性的 > 影响速度的；② 看端到端收益，
> 不看组件收益；③ 能把某个 before 变成 after 的排前面；④ 短探测优先。**

---

## 1 · 队首 · B200 新栈探索队列（Chaoyu 2026-09-03 立；排序=收益×重要性×优先级；**先量基线再逐条改**）

> 环境事实：2×B200（NVLink 5，871 GB/s）· vLLM 0.28 · verl 0.9（V1 统一 trainer/FSDP2/Megatron-Bridge）· torch 2.13 cu13 ·
> FA4 · FlashInfer TRT-LLM 核 · 学生 Qwen3.6-35B-A3B（MoE）· 教师 Qwen3.8-27B。读数与旧栈/5090 **一律不混比**。
> 纪律：每条先在默认配置上量 before（同尺子、多种子/多次取中位数），再单变量改，after 与 before 同表落 E 报告。
> 判据先注册（主线守则⑬）；不是最新稳定版的组件必须写原因（守则⑯）。

| 优先 | # | 任务（单变量） | 判据 / 读数 | 归谁 | 成本 |
|---|---|---|---|---|---|
| **P0 基线与正确性** | B0-1 | **v16 全管线冒烟基线**：FSDP2 单卡 LoRA 训 35B-A3B，bf16，V1 sync colocate；数据→SFT+eval→RL+eval→OPD+eval 全通 | 每段退出码 0 · loss/grad 有限 · DDP 各 rank 权重逐位同 · 位移≈lr×步数 · 权重真推到 rollout（校验和） | 主线 | 1–2 天 |
| | B0-2 | **训推一致性尺子重立**：rollout logprob vs trainer 重算逐 token 差；`rollout_correction` 开/关；**MoE 路由不一致率**（同 token 两侧选的专家不同的比例） | 差值分布 + 噪声地板；路由不一致率数字（R3 的 before） | infra | 0.5 天 |
| | B0-3 | wandb 全链上报判据：SFT/RL/OPD 每步 loss、grad_norm、cap、行为读数都在同一 project；Modal secret 注入 | 每段训练结束 wandb run 里指标条数 == 步数；无 offline 回退 | 主线 | 0.5 天 |
| | B0-4 | Modal 稳定性探针进代码：拓扑指纹、显存归零、缓存落 Volume、抢占后从 ckpt 续跑一次（主动杀容器） | 续跑后 step/loss 接上、无重复样本 | 主线 | 0.5 天 |
| **P1 训练侧高收益** | B1-1 | **Megatron-Bridge + EP=2 训 35B-A3B** vs FSDP2 基线 | 步速 tok/s、峰值显存、loss 曲线重叠（同种子） | infra | 2 天 |
| | B1-2 | **MXFP8 训推统一（Miles 07-29 recipe 正版）**：TE MXFP8 fwd/wgrad/dgrad + vLLM FP8 rollout；两侧量化器逐位对拍 | 两侧量化差=0 · reward 曲线与 bf16 重叠（±MDE）· 步速 | infra | 3 天 |
| | B1-3 | verl 0.9 三模式对照：sync / colocate_async / separate_async（1+1 分离） | 占空比、每步墙钟、陈旧度分布、任务分配对 | infra | 1 天 |
| | B1-4 | **路由回放 R3**（verl `router_replay_patch`）：训练侧复用 rollout 路由 | 路由不一致率→0 · reward 方差、IS 比率尾部 | infra | 1 天 |
| | B1-5 | MTP：k=1/2/3/4 接受率按任务分桶（工具调用 vs 人话 vs 思考）；MTP 头作训练辅助损失开/关 | 接受率表 · 单流/并发 TPOT · 任务分 | 主线+infra | 1 天 |
| | B1-6 | FA4 进训练侧（verl attention backend 切 cute）vs FA2 | 步速、loss 逐位/噪声地板内 | infra | 0.5 天 |
| **P2 推理/serving** | B2-1 | 单卡 vs DP=2 vs TP=2 vs EP=2 吞吐曲线（allgather_rs 基线）→ **DeepEP** 两后端 → **EPLB** | 每档 goodput/TPOT/TTFT 曲线；EPLB 开关 Δ | infra | 2 天 |
| | B2-2 | vLLM 原生 DP+LoRA 替代自研前缀亲和路由（E32 翻案）；PD 分离在 NVLink 上重判（MAINLINE ⑦） | 同 trace 下 goodput、cache 命中；PD go/no-go 三测量 | infra | 1.5 天 |
| | B2-3 | NVFP4 W4A4 / FP8 KV 在 Blackwell 上的 serving 与 rollout 曲线（Miles NVFP4） | 五臂曲线 + 任务分配对 | infra | 1 天 |
| | B2-4 | 执行层：CUDA graph 覆盖率、GDN decode 小核微间隙 profile（E14 方法重跑）、torch.compile 收益 | nsys 间隙总时长、graph 命中率 | infra | 1 天 |
| **P3 核/通信** | B3-1 | NVLink 下 all_gather 对齐悬崖是否仍在（E18 重跑）；all-reduce 后端 NCCL_SYMM_MEM/QUICK_REDUCE/CUSTOM 对比 | 各消息大小 busbw 表 | infra | 0.5 天 |
| | B3-2 | tcgen05/TMEM 块缩放 MMA 峰值（MXFP8/NVFP4）vs sm_120 的 543/627；DeepGEMM 对照 | TFLOPS 表 + 占峰比 | infra | 1 天 |
| | B3-3 | MoE 训练微优化消融：grouped GEMM · 选择性重计算 · 序列打包 · LoRA 挂专家 vs 只挂注意力 | 每项单变量 Δ 步速/显存/任务分 | infra | 1.5 天 |
| | B3-4 | **B300 重跑全链收坑**（sm_103a 只编 sm_100a 的核）；每个坑一个上游 issue/PR | 全链绿 + PR 列表 | 双方 | 1 天 |

> ✅ 旧队列（B-5 / B-4 / E31）全部收官，摘要保留在下方备查；上面的表是唯一现役队列。

| # | 任务 | 归谁 | 成本 | 填哪些简历空格 |
|---|---|---|---|---|
| — | 常驻行为判据进 `compare`（归主线，不 gate 收尾） | 主线 | 小 | — |

> ✅ **B-5 业务调度层提升 2026-08-28 单日收官**（[E33](E33-orchestration-goodput.md) 全档）：
> **goodput@SLO 64→192（3×，三遍复核）**——分账插桩定位四瓶颈，五刀全落（扩池×多进程
> 连落·PG 触发器门铃·router v2 零解析·SLO 优先级）；杀 1/4 worker 零丢单 24/24 ·
> loadtest 22/22 · pytest 722 净值（红线零妥协）；C=256 判死于引擎（llm 91%）=调度层
> 份额 <10%，膝点移交引擎。七个翻案全档 E33 §6——含 **E32"膝点在编排层"强版结论被
> 自家分发计数判据翻案**（after 臂路由塌缩单引擎）。生产落地：09 §0 栈形态 ·
> start_vllm.sh +priority · PG 300 连接/2GB。

> ✅ **B-4 四卡分布式 serving 压测 2026-08-28 单日收官**（[E32](E32-serving-loadtest-4gpu.md) 全档，
> NARRATIVE 四格全填）：拓扑=4 引擎+亲和 router（DP×LoRA 上游判死自研补位·重负载扩展
> 3.66–3.86×·TTFT P90 −40%）· goodput@SLO=64 并发且膝点在编排层（四卡等值证明）·
> 批调度默认已优+mnbt16384 · PD no-go（无物可卸，chunked 两态对照拆穿字面 go）·
> **ngram 投机全场景采纳**（单流 2.3×·48 并发 +41%·64% 接受率·50/50 逐字无损）——
> 后两项已进 `start_vllm.sh` 生产默认并冒烟实跑；四卡高吞吐模式=`b4_serve_4x.sh`。
> 早前已收：占空比 31→**73.4%** · 乒乓⑤⑥⑦验证关账。

> ⛔ **CoT 训练支持：infra 侧退出（Chaoyu 08-28 裁定，随之撤出简历）**——主线数据线节奏自定；
> 挂在它身上的三个复活条件（陈旧度剂量 · 同步暂停 · 投机解码重估）**长期停放**。

> ✅ **E31 训推统一 FP8 2026-08-27 单日全六步闭环**（原队首，Chaoyu 08-27 立/同日完）：
> 可行域=lm_head（偏置 9× 对消至本底·400 步长跑三把尺健康·配对 +0.109 入带·零速度税）；
> 内层判负定界（异构引擎 hidden 微差被激活量化逐层放大，~−1.2e-4/层；复活=token 级 IS
> 或同构引擎）⇒ 全档在 [E31](E31-unified-fp8.md)；开着的裁定：终审 ±MDE 加跑与否 ·
> 生态数据点是否外发（DRAFT-sm120-mxfp8-ecosystem-datapoints）。
> ✅ A3/TileLang 2026-08-27 一日收官（[E30](E30-tilelang-nvfp4-gemm.md) 十三节）：sm120 首个 MXFP8 GEMM
> 543(tilelang)/627(裸 CUDA=消费卡包络) · 反向 dgrad/wgrad 2.1×/1.7× · 温度偏置机理+c* 补偿 ·
> **A4 同日收官**：8bit lm_head 前向+反向完整 SFT 与 bf16 同带（§13）· 上游两 DRAFT 包移交。
> ✅ ckpt IO（E29）2026-08-20 已完：save 7.91→0.83 s（9.5×）· ckpt 18→1.5 GB · 续跑合成加载
> 与 adapter 提取链路全验 ⇒ 结论在 [`E29`](E29-ckpt-lora-only.md)，改动登记 02 §3。
> ✅ E14/R2 收官三件 2026-08-21 已完：s16/0.1 三种子复核过 + graph 单变量精度闸过（**两默认值
> 已切库**：`--sync-every 16`、`--enforce-eager False`）· compile 微基准判死（生产形状零收益
> + 重编译税）⇒ 结论与边界表定稿在 [`E14 §4.10/§5`](E14-bubble.md)。
> ✅ 原 #5 量化推理 A/B 2026-08-21 已完（Chaoyu 点单，E19-c 四臂+两步归因）：**fp8 KV=容量杠杆**
> （吞吐 +50%·质量 −0.009 在 MDE 界）· fp8 W8A8 单流反慢 37% · **FP4 权重对 4B 判死**
> （W4A4/W4A16/无 adapter 三读数互证 −0.68/−0.67/−0.49）⇒ 全表在 [`E19 §8`](E19-fp8-in-training.md)；
> 是否把 fp8 KV 设为 rollout/serving 默认 → **Chaoyu 裁定**（质量代价在 MDE 界上）。


## 2 · 接下来（正确性收尾）

| 任务 | 归谁 | 说明 |
|---|---|---|
| **P8 占位 logprob 归因**（~0.1% 样本 coverage<1.0，污染 IS 权重） | 待认领（建议 infra） | 找到来源、判可否消除 |
| **Q4 失败注入确定性 / Q5 Q6 token 序列与 mask 同构** | 双方 | 每条写成断言/探针，不要「读代码确认」 |
| **E23 落地：评测侧采样对齐**（top_p=1.0 / top_k=-1） | 主线 | 决策已定，还没落地 |

## 3 · 速度线（E26 后靶子已换，按剩余大小排）

> ⛔ **速度线随 08-28 单任务制裁定整体停放**（收尾不做；理由与复活条件留档）：
> E17-C ref 抽样（前提"KL 多种子"已撤销，实质过时）· B11 配比（原要随 CoT 重设计，随停）·
> A4 RL 侧切片（~3.6%）· A18 整机级空泡（trainer 侧已被 E14 §4.11 清零结案）。
> 仍搭车的两件：**乒乓修理⑤⑥⑦ 收益验证 + 整机占空比 after**（并入 B-4 包的冒烟/压测采样，
> 见 §1）；R6 NCCL 流量实测=有跑就顺手，无跑不追。

## 4 · 独立线（不 gate 也不被 gate）

| 任务 | 状态 |
|---|---|
| **MoE 线** → ⛔ **Chaoyu 08-28 裁定停做**（并撤出简历成果栏；E07 的账与陷阱留作面试故事）。Qwen3-30B-A3B 57GB 仍在盘上，复活按 E07 §4.5 重启 | ⛔ 停做 |
| ~~trainer 侧 FP8 融合栈~~ → **已并入队首 E31 第 3/4 步**（08-27；自有 kernel 替代 torchao/TE 路线） | 📦 并入 |
| A17 · 对齐补丁端到端（上次钩子没挂白跑） | 🔴 可重跑 |
| 上游包（`docs/upstream/`）：16 字节对齐 · HYBRID_SHARD · verl 两条 ✅ 已移交 upstream 同事（08-20）；🆕 **E31 路新增两 DRAFT（08-27）**：verl entropy_coeff=0 连图（一行修 PR 候选）· sm120 MXFP8 生态数据点回帖包（TE#2304/triton#7550/CUTLASS#2867/DeepGEMM#236 + RL 社区正负双结果）；vLLM prompt_logprobs OOM 判 PARKED（上游已有 tracking #5907） | 本窗口只做证据支援；新 DRAFT 待考据成稿 → Chaoyu 点头 |
| 🔵 lr 1e-4 @5120 上限基线（已降级，脚本 `scripts/infra/run_e20h_lr1e4_5120.sh` 备好） | 想测随时跑，不挡人 |
| 🆕 **训推量化失配剂量学**（Chaoyu 08-27 立项）：独立 side project，**归其他同事**，两线只供数据不参与——交接件 [`docs/side-quant-mismatch/00-PROJECT.md`](../side-quant-mismatch/00-PROJECT.md) | ⬜ 待认领 |

## 5 · 管线验证状态（引用前查这张，别重新论证）

| 环节 | 状态 |
|---|---|
| 数据生成 / SFT / merge / rollout 生成 / eval 合并 / ckpt→adapter | ✅ |
| RL trainer 梯度同步（E21 修 + 归约口径 1.000000） | ✅ |
| RL 权重同步 trainer→vLLM（E22 修 + 数值验证） | ✅ |
| PG 打包前向（E26：等价 + 归约逐位同 + 四常驻判据） | ✅（B5 由 candidate 兜底兑现：400 步 +0.186，已切默认） |
| `logprob_coverage` 全样本=1.0（旧 P8） | 🔴 **已降级**：~0.1% 样本 <1.0（最低 0.9932） |
| 组结构 P4（按 fit step 口径） | ✅ 复查已做（cand_v13r2_e1 有 4/400 步重复，归因=fully_async 完成序重排、非采样器；Chaoyu 08-20 裁定当前不重要不动管线，细节见 git e1043ee） |

## 6 · 维护约定

- 每条做完**就地删行**；新条目必须有「归谁」。
- 交界的事**先写 /MAINLINE-INFRA.md** 再动。
- 目标变了**整张表重排**，不是往上堆新条目。
