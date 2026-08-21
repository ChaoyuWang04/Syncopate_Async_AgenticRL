# Infra · 02 · DECISIONS — 已定决策 · 作废登记 · 已落地改动

> **回查「当时为什么这么定 / 这个数字还能不能引用」才读它，不必通读。**
> 新决策**就地改行**（理由变了要写为什么），不追加重复行。

---

## 1 · 已定决策（别再重新讨论）

| 决策 | 结论 | 详见 |
|---|---|---|
| 框架 | **verl 不换** | E07 §1 |
| 训练侧并行 | **DDP 必选**（`--fsdp-size 1`）。三档稳态实测：DDP 7.97 s / ZeRO-2 慢 3.42× / ZeRO-3 慢 6.02×（3 卡）；TP=2 rollout 净负 20% ⇒ **这台 P2P 全关的机器上模型内并行是净亏损** | E02 · E04 · E18 |
| attention | `flash_attention_2`，**必须官方 cu13torch2.9 轮子**（社区 cu128 反向是坏的，RL 静默空转）。换轮子先跑 `check_flash_attn_backward.py` | 00 §5-⑧ |
| `micro_batch` / `dynamic_bsz` / GC | **mb=1 · dynamic_bsz=False · GC 开着**（喂饱 GPU 的单位是 token；mb 拉高负收益、关 GC 就 OOM）。**PG 开时例外：mb=8**（一组一批；mb16 慢 5.7%） | E25 · E26 §6.6.1 |
| **PrefixGrouper** | ✅ **已切库默认开**（Chaoyu 2026-08-20；launch_rl `setdefault SYNCOPATE_PREFIX_GROUPER=1`，mb 联动默认 8，`=0` 可关做对照）。证据：E26 端到端 2.31× + cand_v13r2_e1 400 步全绿、候选 +0.186 兜底兑现 | E26 · /MAINLINE-INFRA |
| **LoRA 权重同步** | **必须走 adapter 推送**（`SYNCOPATE_LORA_ADAPTER_SYNC` 默认开）。⛔ 禁 `--lora-merge`（bf16 合并毁掉 adapter 一半作用，启动即拦） | E22 §6.1/§6.4 |
| **FSDP 后端** | **留在 FSDP1**（Chaoyu 08-18；上游不修退化网格）⇒ `SYNCOPATE_FSDP_DDP_FIX` 必须一直默认开；且**前向必须走根模块**（绕过 = 归约竞态） | STORY §8.1 · E26 §6.3 |
| **MoE 选型** | `Qwen3-30B-A3B-Instruct-2507`（GLM-4.7-Flash 当前栈不支持）；LoRA **绝不 all-linear**（98.7% Linear 在专家里 ⇒ 26×），用「注意力+router」30.1M | E07 §4.5 |
| NCCL 协议 | 按并行策略分设：DDP 用默认；分片路径开 `NCCL_PROTO=LL128`（**不能全局开**：all_reduce −30%）。launch_rl 已自动 | E18 §9–10 |
| 满载降频 | 多卡对照**不需要扣这一项**（满载单卡 −2.0%） | E00 |
| E11 稀疏 logprob | 🔻 降级不写 kernel（端到端 4.3%，切片就有 4.0%）。★「浪费的比例」和「能拿回的收益」隔着一个分母 | E11 §6 |
| 采样口径 | 训练侧不截尾（`top_p=1.0/top_k=-1` 已钉死），**评测对齐训练**（落地挂 01 §2） | E23 §3 |
| 配对比较基线 | 一律 `_audit/v13_sft_e1_merged.json`（旧 v13_sft_e1 会系统性低估 RL +0.025） | E24 |
| **KL / ref** | ✅ **已切库默认关**（Chaoyu 2026-08-20；`--use-kl-loss` 默认 False）。E17 两臂：省 15.4%、任务分无差异；cand 400 步 KL-off 长跑兑现。★ 判据③ `rollout_corr/kl` **不随 ref 消失**（rollout-IS 诊断；cand 全程 98 点中位 4e-4 在地板，第二次实证）。`fabricated_safety_line_cap` 保持常驻观察；⛔ 连带「ref 走 FP8」失效 | E17 §9 · /MAINLINE-INFRA |
| **lr 与步数**（Chaoyu 08-19） | 「学不动」主因是**步数太少**（≤1 epoch）不是 lr 低 ⇒ 解法是加步数/固定 epoch；**上线候选不用 lr 1e-4**；400 步是下限、停由判据定 | 01 §1-4 |
| 常驻判据四条 | lora-probe 非空 · 第 2 次同步 lora_>0 · kl 回落 3.4e-4（⚠️ fp8 KV 默认后地板会略抬，首个正式跑实测重标） · `clip_ratio=0.0000` | 00 §6 |
| **fp8 KV cache** | ✅ **已切默认**（Chaoyu 2026-08-21；launch_rl `--kv-cache-dtype fp8` + serving 端点同步）。E19 §8 五臂+归因：KV 池 ×2 ⇒ 并发 +50%（容量杠杆）；质量 −0.009 恰在 MDE 界、defer 门槛内、归因=代价全在 KV 侧。⛔ FP4 权重同案判死（4B 崩塌，三读数互证）；trainer 侧 FP8 融合栈降独立线（01 §4） | E19 §8 |
| launch_rl 默认值 | bucket 512 · `--rollout-is sequence`（08-19 改回，序列级 ESS 才会动；cand 实测中位 0.92/最低 0.816，健康）· `ulysses_sp=1` · 数据文件跟 `DATA_VERSION` 走 · **PG 开（mb 联动 8）· KL 关**（08-20）· **sync-every 16 + CUDA graph 开**（08-21，三种子复核 + 单变量精度闸，E14 §4.10） | launch_rl help |
| **thinking 开关** | `SYNCOPATE_THINK=1` **只许评测**（launch_rl 拦训练）；开关在契约模块（模板 kwarg + 预算 8192 一起切）；**裸基座 eval 臂单轮上限必须给 2048**（256 的砍断与真实弱分不开）。零训练拨开关不上生产（净 −0.057）；红利路径=带思考的 SFT 数据 | **E27** |
| **永久基线** | eval 配对的基座参照 = `_audit/e27_base_off.json`（base think-off @2048/轮，修复后管线产）；SFT/RL 配对基线仍是 `v13_sft_*_merged` 族（E24） | E27 §5 |
| **常驻观察** | `fabricated_safety_line_cap`：两处独立反向信号汇合（E17 KL 臂 +2 · E27 SFT vs 基座 +18）⇒ 每次 compare 必看 | E17 §9 · E27 §5 |

## 2 · 排序原则沿革

```
08-17  短探测优先（30 min go/no-go 挡 2–3 天）→ 看对核心数字的贡献
08-18  Chaoyu：影响正确性的 > 影响速度的；第二看端到端收益不看组件收益
08-19  Chaoyu：lr 1e-4 类「上限基线」不占队首；能把 before 变 after 的排前面
```

## 3 · 已落地的改动（都在 `syncopate/train/`，出问题先查这张表）

| 改动 | 效果 | 守护 |
|---|---|---|
| 🔴 `_patch_lora_adapter_sync`（默认开） | E22 修法①：首推基座、之后推 adapter。载荷 8414→252 MiB、param_sync→0.97 s | 常驻判据①②③ + `--lora-merge` 互斥 |
| 🔴 `_patch_fsdp_degenerate_mesh`（默认开） | E21 修复：退化网格→NO_SHARD+默认组，三 rank 梯度逐位同 | 常驻断言 + `repro_fsdp_hybrid_nosync.py` |
| 🔴 `_patch_prefix_grouper`（默认关） | E26：打包前向走根 FSDP + hook 捕获 hidden + 0×根输出锚 | `repro_pg_dtype.py` + 判据A/组构成 |
| `ddp_save_to_cpu` 加 `if requires_grad` | E13：old_log_prob/ref 比值 1.94→1.07（≈8.5 s/步） | 3 条测试 |
| launch_rl 启动守卫 | mini_batch×rollout_n 整除检查（列可用值 [3,6,9,12,15]）· lora-merge 拦截 · 数据默认值跟 DATA_VERSION | 都在最早能判的地方炸 |
| 🆕 乒乓修理②③（默认开，E14 §4.7/4.9） | PG `repeat_interleave` 补 output_size（×584→0）· `_to_jagged` 批量 tolist（×96→2）；数值零算术不变；A/B 阶梯 912→236 逐级命中 | SYNCOPATE_FIX_PG_RI / FIX_JAGGED（=0 对照） |
| 🆕 `--enforce-eager` 旗子（默认 True 沿现状，E14） | False=开 vLLM CUDA graph：gen −33%（sync4 档）；晋级默认欠精度闸 | 捕获判据行 ×31+ |
| 🆕 torch-prof 探针（默认关） | SYNCOPATE_TORCH_PROF=N 抓 rank0 第 N 次 update_actor：chrome trace + 同步类算子账本（判罪按同栈 Synchronize 配对） | E12 §262 元数据搬运 |
| 🆕 `_patch_lora_only_ckpt`（默认开，E29） | LoRA 下 ckpt 只存可训练部分：save 7.91→0.83 s（9.5×）、ckpt 18→1.5 GB；load 端合成加载；全参自动回退全量 | 5 条单测 + [ckpt-lora] 判据行 + 续跑实跑 |
| 探针族（verl_patches） | SYNCOPATE_SYNC_PAYLOAD / OPT_STEP_PROBE / DDP_PROBE / SYNC_TIMING | 绑不上就报红，不打结论 |
| NVTX 阶段标注（`--nvtx`） | verl `marked_timer` 有名无实，补齐后 nsys 才能按阶段归属 | — |

## 4 · ⛔ 作废登记（引用旧数字前必查）

> 判据一句话：**算了多少、搬了多少字节 → 不受影响；算得对不对、学到没有 → 作废。**

| 结论 | 状态 |
|---|---|
| E20 全部数字（ESS/chi2/位移；量的是「当前策略 vs π₀」） | 🔴 作废，连比值都不能用 |
| B10 陈旧度阈值 / B19 精度侧（旋钮当时没接到任何东西） | 🔴 作废 ⇒ 01 §2 R2 |
| B3 三模式**学习类**对比 | 🔴 作废；吞吐对比保留 |
| E12「99.9% 不是传输」及全部权重同步耗时（55.8 s / 18.8%…） | 🔴 前提错（以为推 132 MB 实推 8.4 GB）。现值 param_sync ~0.97 s / 0.8%。报告已归档 `docs/archive/E12-weight-sync.md` |
| E00/E02「DDP 每步同步 260 MB」 | 🔴 算出来的，从没量过 ⇒ 01 §3 R6 |
| E26 早期「2.12×」 | 🔴 单行 timing 的覆盖数没实测 ⇒ 正式数 2.31×（E26 §6.6） |
| `_audit/v13_base.json`（基座@256/轮） | 🔴 **已删**（Chaoyu 08-19）：256 的砍断与真实弱分不开（截断 40.2%/parse_errors 909）。替身 = `e27_base_off.json`（@2048） |
| E25「mb=4 OOM」外推到变长真实批 | 🟡 未迁移（mb=8 实测 29–31 GB 贴顶跑完），但余量不能当生产配置 |
| 「lr 被夹在两堵墙之间」叙事 · M7/M7-b 全部评测 | 🔴 双重作废（E21+E22 坏基线） |
| B11 拓扑放置 1.6% | 🟡 数字仍真，理由要按 E26 后的新构成重写 |
| E18/A14/A16 对齐 · E03 NCCL · E16/E19 FP8 数值 · B20 · E01/A5 kernel · E13 | ✅ 完全不受影响 |

## 5 · 已结案 / 停放（别再捡起来；复活条件写明的除外）

```
✅ E21 梯度不同步 · E22 LoRA 没推 · E23 采样口径 · E24 基线选择 · E25 喂饱证伪
✅ E26 数学/微基准/集成/吞吐 A/B · E17 KL 两臂 —— B5 与 KL-off 长跑均由
   cand_v13r2_e1 兜底兑现（400 步 +0.186，ESS 中位 0.92），PG/KL 已切默认（§1）
✅ E18+A14+A16 16 字节对齐 · E03 NCCL 旋钮 · E01+A5 一步时间 · A7 降频 · A6 三档稳态
⛔ E04 rollout TP=2 净负 · A9 预量化 · B1 权重同步优化（param_sync 只占 0.8%）
⛔ E11 手写 kernel · mb/GC/mb16 三条路（E25/E26 证伪）
🔄 token vs seq 多种子撤销（Chaoyu 08-20）：cand 实测 seq IS 的 ESS 98 点中位 0.92/
   最低 0.816、无衰减 —— 健康且有读数，多种子没有要回答的问题
✅ R2 剂量曲线已测（08-20 夜由闸门实验兑现，早于 CoT 路径）：5 臂+等时臂定案在 E14 §4.8
✅ 「同步不打断 rollout」被更好的解法取代（E14）：降低同步频率（sync16）直接把暂停次数 ÷4，
   gen 28%→0.6%——双缓冲/加深队列不必做了
```
