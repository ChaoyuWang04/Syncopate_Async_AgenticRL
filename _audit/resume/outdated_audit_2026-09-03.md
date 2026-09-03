# 旧栈→新栈 简历过时项审计（2026-09-03，只读扫描，逐行带出处）

> 用法：这是 NARRATIVE-AND-RESUME（infra 版与主线版）重写的**施工单**。每行 = 文件:行号 | 旧栈绑定表述 | 为何在 B200+新栈上过时 | 类别。
> 处置三档：**删**（消费卡背景句）· **降格为历史**（verl 0.8/vLLM 0.12 时代的修复，必须带上游 PR 编号）· **重量/新立**（B200 上重跑或新做）。
> 新栈事实源：docs/syncopate/08 §Modal（B200 实测）· docs/infra_exp/01-TASKS §1（B200 探索队列）。

## 十行总结
1. 最大类=硬件限制：约半数发现源于"P2P 全关 + 无 NVLink + 消费卡显存/带宽"这一前提；2×B200+NVLink5 移除该前提，牵连 DDP 选型、E18 对齐悬崖、TP/PD 判断、四卡 serving 拓扑一整条决策链。
2. 第二类=旧 verl/FSDP1 补丁：E21 退化网格、E22 LoRA 同步、E26 PrefixGrouper、E29 LoRA-only ckpt 全对着 verl 0.8 内部写；MAINLINE-INFRA 已自认"新栈上要重对" ⇒ 简历里降级为历史工作。
3. 第三类=旧 vLLM 缺功能：DP×LoRA 不支持（E32）、compressed-tensors 拒收 NVFP4（E19）、Triton 滞后（PRIMER）——vLLM 0.28+FlashInfer 已原生解决，继续讲"自研补位"显得不了解现状。
4. 00-START 守则⑯已自曝根因（为 5090 迁就旧版本），迁移已启动（08 §Modal、MAINLINE B200 探针）。
5. "仍然有效"的例外：flash-attn 坏轮子判据脚本、判据纪律、IS 数学、异步剂量、调度层分账、量化偏置机理。
6. PRIMER-precision-sm120.md 整份框架过时（"消费卡 vs 数据中心卡"），需整体重写。
7. 两份 NARRATIVE 的开场白"4×消费级 GPU 无 NVLink/P2P"是修改优先级最高的单句。
8. 08 §Modal 已有新旧对照实测（NVLink 34×、FA4 4×、MTP、EP=2），是重写的现成数字源。

## Top 10 最该重写
1. infra NARRATIVE:96-98 "4×消费级 GPU 无 NVLink/P2P（1/35 带宽）"背景句
2. infra NARRATIVE:145-148 NCCL 16 字节对齐悬崖——3-rank/PCIe 专属
3. infra NARRATIVE:233-243 "4 独立引擎+自研亲和路由"因 vLLM 0.12 不支持 DP×LoRA
4. infra NARRATIVE:169-181 "sm_120 首个可用 MXFP8 GEMM/消费卡包络 61.2%"
5. infra 02-DECISIONS:18 "FSDP 后端留在 FSDP1"及守护补丁
6. E32/E33 四卡 serving 拓扑数字（3.66–3.86×、goodput 64→192）——拓扑换了不可比
7. infra NARRATIVE:250-254 PD 分离 no-go——文档自称"有 NVLink 会翻"，现在翻了
8. 主线 NARRATIVE:291-303 投递版项目三"消费卡上利用 Blackwell 原生 FP8"
9. E31 训推统一 FP8"可行域=lm_head"——B200 两侧同 TE 核前提改变
10. PRIMER-precision-sm120.md 全篇

## 逐文件清单

### docs/infra_exp/00-START.md
| 位置 | 旧栈绑定表述 | 为何过时 | 类别 |
|---|---|---|---|
| :44-46,92-96 | E31 lm_head 可行域·占空比 73.4%·E30 sm120 MXFP8 61.2% | 异构 vLLM/FSDP1+sm120 缺块缩放核下的结果 | 硬件限制 |
| :140-141 | 防"uv pip 毁 torch 2.9→2.13"事故 | 新栈本就 torch 2.13 | 旧版本 |
| :144-146 | flash-attn 换轮先跑反向判据 | 方法论仍适用 | 仍然有效 |
| :210-236 | 换机器重建清单/重画像全建立在 4×5090 | 文档自注"换非 5090 全部作废" | 硬件限制 |

### docs/infra_exp/01-TASKS.md（旧队列部分）
| :36-40 | E31 可行域=lm_head | 两侧引擎一致性前提改变 | 旧补丁 |
| :49-52 | fp8 KV 容量杠杆·FP4 对 4B 判死 | sm120 半速+生态缺位的产物 | 硬件限制 |
| :78 | verl 0.8 上游包/sm120 MXFP8 回帖包 | verl 0.9/sm100 后多半不适用 | 旧版本 |

### docs/infra_exp/README.md
| :91 | E00 满载降频/PCIe 代数 | NVLink 拓扑 | 硬件限制 |
| :93-94 | DDP 必选·NCCL_CUMEM_ENABLE=0·LL128 | P2P 关闭+消费 NCCL 专属 | 硬件限制 |
| :99 | E08 "4 卡只换 1.59×" | 2 卡 NVLink 结构变 | 旧版本 |
| :105,109 | E16 sm120 FP8 首枪·E17 三次前向占步 | 基线换代 | 硬件限制 |
| :108 | E18 3 卡 ZeRO-3 塌陷 | 触发条件（PCIe+非 2 幂 rank）消失 | 硬件限制 |
| :277-296 | 全局常量 16.3/17.9 GB/s·16B 悬崖·SHM 传输 | NVLink 546–871 GB/s，P2P 可用 | 硬件限制 |
| :304-306 | flash-attn cu13torch2.9 轮子+cuda-runtime 补丁 | 新栈 torch 2.13 另一套 | 旧版本 |
| :329-353 | 权重同步 55.8s→8.4GB→0.97s | verl 0.8 LoRA 同步缺陷 | 旧补丁 |

### E 报告
| E01:41,91-96 | GEMM 全是 cutlass_80 | B200 原生块缩放核 | 旧 vLLM |
| E02:9-13,36-37 | DDP 完胜 FULL_SHARD（6.4 GB/s 无 P2P） | NVLink 下 FSDP2 代价大降 | 硬件限制 |
| E03 | NCCL_CUMEM_ENABLE=0/LL128 | 触发条件消失 | 硬件限制 |
| E07:9,36-37,88 | 4×5090 无 P2P 训 MoE 最佳组合 | B200 原生 EP+MXFP8 | 旧 vLLM |
| E08:320 | fsdp_size=1 退化网格 | FSDP2 无此 bug | 旧补丁 |
| E11:43 | verl 0.8 只做统计 | 需 0.9 重测 | 旧版本 |
| E13:39 | 单张 5090/torch2.9 环境指纹 | 过时 | 旧版本 |
| E14:62,201 | cutlass_80 | 同 E01 | 旧 vLLM |
| E16 全篇 | sm120 FP8 半速·tl.dot_scaled 退化 | B200 无此限制（FA4 4.0× 已测） | 硬件限制 |
| E18 全篇 | 3-rank all_gather 16B 对齐+FSDP1 _get_shard | FSDP2+2 卡双重消失 | 旧补丁 |
| E19:215-270 | 5090 FP8 主场·vLLM 0.12 拒收 NVFP4 | 0.28 原生支持 | 旧 vLLM |
| E21 全篇 | HYBRID_SHARD 退化 NO_SHARD | V1 trainer 默认 FSDP2 | 旧补丁 |
| E22 全篇 | LoRA 从未推给 vLLM | 0.9 权重同步需重验 | 旧补丁 |
| E25:143 | PrefixGrouper 在 0.8 没接上 | V1 结构不同，重核集成点 | 旧补丁 |
| E26:73,159 | FSDP1 hook 绕过=归约竞态 | FSDP2 机制不同 | 旧补丁 |
| E29:10,29,32 | 0.8 无 LoRA-only ckpt | 0.9 是否已有需核实 | 旧补丁 |
| E30 全篇 | 消费卡 100KB smem 墙·627 TFLOPS 包络 | 价值锚点=消费卡极限 | 硬件限制 |
| E32 多处 | 无 P2P 最优 serving·TP 净负·PD no-go·DP×LoRA 不支持·ngram 采纳 | 全部需在 NVLink+0.28+MTP 下重判 | 硬件/旧 vLLM |
| E33:65,69,108,136 | 4 卡亲和 fleet 生产口径 | 2 卡拓扑重压测 | 硬件限制 |

### docs/infra_exp/NARRATIVE-AND-RESUME.md（简历本体）
| :96-98 | 4×消费级 GPU 无 NVLink/P2P（1/35） | 定位句作废 | 硬件限制（头句） |
| :145-148 | 16 字节对齐悬崖 | 3-rank 专属 | 硬件限制 |
| :154-164 | cutlass_80·FP8 定界 316×·inline PTX 点亮 sm120 | B200 原生路径 | 硬件/旧 vLLM |
| :169-181 | sm120 首个 MXFP8 GEMM·627 TFLOPS 包络 | 消费卡稀缺性消失 | 硬件限制（核心卖点） |
| :182-193 | 8bit lm_head SFT·训推统一 FP8 消费卡首次 | 同上 | 硬件限制 |
| :196-197 | PCIe 画像·坏轮子判据 | 画像作废/判据仍有效 | 混合 |
| :233-243 | 4 独立引擎+自研亲和路由（DP×LoRA not supported） | 0.28 原生 | 旧 vLLM（重点） |
| :240-243 | ngram 全场景采纳，MTP/EAGLE 因无头排除 | Qwen3.6 自带 MTP，已实测 | 旧版本 |
| :250-254 | PD no-go，成立范围=无 NVLink | 现在有 NVLink | 硬件限制（自我预警） |

### docs/infra_exp/TRACKS.md · 02-DECISIONS.md · PRIMER-precision-sm120.md
| TRACKS:14,39-51,64-67 | 立论=消费级拓扑把通信变硬约束；兑现物锚定 5090 容量 | 立论基础动摇 | 硬件限制 |
| 02:13 | DDP 必选 | FSDP2 代价大降 | 硬件限制 |
| 02:14 | flash_attention_2 必须官方 cu13torch2.9 | 另一套轮子 | 旧版本 |
| 02:17 | LoRA 同步必须 adapter 推送 | 0.9 需重验 | 旧 vLLM |
| 02:18 | FSDP 后端留 FSDP1 | 整条作废 | 旧补丁（重点） |
| 02:20 | NCCL 协议按并行策略分设 | 触发条件消失 | 硬件限制 |
| 02:28 | fp8 KV 两侧拆 | B200 两侧一致性不同 | 硬件限制 |
| 02:33-34 | 生产默认四卡舰队·DP×LoRA 不支持 | 重设计 | 硬件/旧 vLLM |
| 02:77 | "E18/E03/E16 ✅完全不受影响" | 指同机型内；易误读为跨硬件 | 措辞易误导 |
| PRIMER:39-79 | 消费卡 vs 数据中心卡整套框架·速率阶梯·安全表 | 框架级失效 | 硬件限制 |

### docs/syncopate 侧
| 00-START:372 | 只能用 DDP（P2P 全关） | 不再适用 | 硬件限制 |
| 08:10-33 | 4×5090 画像·16.3/17.9 GB/s | 被 §Modal 取代 | 硬件限制 |
| 08:218-244 | flash-attn 轮子版本链 | 方法论仍适用 | 混合 |
| 08:650-667 | peer access not supported 故事 | 触发条件消失 | 硬件限制 |
| 08:695-697 | verl 0.8 ckpt 存全量 | 0.9 重核 | 旧补丁 |
| 11:378-399 | vLLM 0.12·1×5090 TPOT/TTFT 基线·§19 24/25 | 重测后才能再用 | 旧版本 |
| 主线 NARRATIVE:66 | 4×5090 全流程自建 | 更新背景句 | 硬件限制 |
| 主线 NARRATIVE:291-303 | 投递版项目三 消费卡 Blackwell FP8 | 卖点转移 | 硬件限制 |
| 主线 NARRATIVE 全篇 | 33×/31→73% 等数字引自 E 系列 | 需新栈重量 | 旧版本 |
| MAINLINE-INFRA:21-22 | 补丁重对·B200 探针七项 | 官方自认的迁移计划 | ★ |
