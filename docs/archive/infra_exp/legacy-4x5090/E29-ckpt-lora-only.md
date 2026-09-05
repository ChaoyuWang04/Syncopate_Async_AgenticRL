# E29 · LoRA 训练的 checkpoint 只存可训练部分

> 状态：✅ 完成（A/B + 续跑 + 下游链路三关全过）   最后更新：2026-08-20

## 0 · 结论卡片

| | |
|---|---|
| **Track / 兑现物** | A · 框架级改造（批组织/IO 假设失效）；JD：字节 C「存储和IO」· DeepSeek G「CPU/IO」 |
| **需求从哪来** | cand_v13r2_e1：每次存档落盘 27 GB（3 rank × 8.5 GB + optim），97% 是与基座逐字节相同的冻结权重；verl 0.8.0 无「只存可训练部分」能力（checkpoint_contents 只能整类开关） |
| **问题** | LoRA RL 的 ckpt 能不能只存 adapter，且断点续跑/下游链路不断？ |
| **答案** | 能。同分母冒烟 A/B：单次 save **7.91 → 0.83 s（9.5×）**、ckpt **18 GB → 1.5 GB（12×）**（2 rank 配置）；模型分片 8.5 GB → 252 MB/rank（504/903 键）；adapter 提取链路原样工作；**续跑 ✅**（合成加载判据行双 rank 命中、optimizer 状态接续、step 2→4 继续训练并再次过滤存档） |
| **信心** | 高：A/B 同配置同尺子；5 条单测 + 判据行 + 续跑实跑 + 下游链路验证 |
| **推翻了什么** | §6：本任务自己的动机数字「save 占步 19.5%」——解析器稀疏键假象，实为 **0.6%** |
| **下一步** | 续跑验证 → 默认保持开 → 素材交 upstream 同事（第 5 包候选） |

## 1 · 问题与预测

**问题**：verl FSDP checkpoint manager 对 LoRA 训练无条件全量落盘；能否 save 端过滤 + load 端合成，
不破坏断点续跑与下游（adapter 提取/晋级评测）？

**预测（跑前写死）**：save 时间与字节量近线性 ⇒ 过滤后 save 应降到 ~1/30（字节比 252MB/8.5GB）；
实际只到 9.5×——**固定开销（barrier/D2H 启动/optim 529MB 未动）留下了地板**，方向符合预期、
幅度低于线性外推。若续跑失败，退路 = 只做异步写盘。

## 2 · 环境指纹

```
2026-08-20 · 4×RTX 5090 / verl 0.8.0 / torch 2.9 cu128 / Qwen3-4B-sft-v13r2-e1 + LoRA r32
冒烟配置：fully_async 2+1（CUDA_VISIBLE_DEVICES=1,2,3，GPU0 让给主线 B-4 端点）
        steps 8 · save-freq 1 · PG 开(mb8) · KL 关 —— 除开关外 off/on 全同
开关：SYNCOPATE_CKPT_LORA_ONLY（默认开；全参训练无 lora 键时自动回退全量，不靠开关兜底）
日志：logs/e29_ckpt_{off,on,resume}.log · ckpt：checkpoints/grpo/e29_ckpt_{off,on}
```

## 3 · 方法

补丁（`verl_patches._patch_lora_only_ckpt`，装在 trainer WorkerDict 进程）：
- **save**：实例遮蔽 `self.model.state_dict`，按 `lora_` 键过滤（504/903 键）；无 lora 键打 ⚠️ 行回退全量
- **load**：检测 lora-only ckpt（全部键含 `lora_`）⇒ 取当前（基座已由模型初始化就位）state_dict、
  `merge_lora_into` 更新 lora 键、整载——**不赌 FSDP 分片 load 的 strict=False 语义**；
  ckpt 里有当前模型没有的键 ⇒ 硬失败（结构变了不猜）
- optimizer 分片不动（本来就只含可训练参数的状态，529 MB）
- 验收：5 条单测（tests/train/test_ckpt_lora_only.py）+ 冒烟 A/B + 续跑 + adapter 提取链路

## 4 · 数据

```
                          off（全量）      on（lora-only）      比
单次 save_checkpoint          7.91 s            0.83 s         9.5×
ckpt 大小（2 rank）           18 GB             1.5 GB          12×
  模型分片/rank               8.5 GB            252 MB（504/903 键）
  optim/rank                  505 MB            529 MB（不动，含少量元数据差）
adapter 提取                  ✅                ✅ 504 张量 · 跨 rank 逐位同 · PEFT 目录正常
判据行                        无（预期）        [ckpt-lora] 过滤行 2 步 × 2 rank 全出现
生产外推（3 rank + save-freq 25，cand 口径）：单跑写盘 108 GB → ~9 GB；
  吞吐收益 ~0.5%（save 本占跑 0.6%，见 §6）；滚动瘦身/prune 链路可退役（存的直接就是终态）
续跑（--resume auto，合成加载路径，共两轮实跑）：
  ✅ 第 1 轮（无数值探针，logs/e29_ckpt_resume.log）：合成加载×双 rank · optimizer 接续 ·
     global_step 2→4 训完 · 再次过滤存档
  ✅ 第 2 轮（带数值探针，logs/e29v_resume2.log）：**逐位校验 504/504 × 双 rank** ·
     RNG/lr_scheduler 一并接续 · 训完 16 步
  ⚠️ 期间一次失败（e29v_resume）与本补丁无关，见 §7
```

## 5 · 结论

LoRA-only 存档成立：**价值主体是字节与链路简化**（写盘 12×、瘦身退役、秒级 save 解锁高频存档
——陈旧度研究要的"相隔 k 步的 policy 对"从此不心疼磁盘），吞吐收益如实记 ~0.5%。

## 6 · ⛔ 推翻了什么：本任务自己的动机数字

> **原 claim**（2026-08-20 上午，进过 01 队列行与简历草稿）：「save_checkpoint 占步 **19.5%**，
> 优化后端到端快 ~18%」。
> **实测**：cand 日志里 save_checkpoint 只出现在 **3/100** 条 timing 行（每次 ~8.8 s）；
> 旧解析器把「出现行的中位 / 每行步长」当占比——**没算出现率** ⇒ 0.6% 被报成 19.5%。
> **推翻后**：解析器已修（share=Σ键/Σstep，稀疏键强制 ⚠️ 行）；四处引用就地更正。
> **教训**：E26 坑 5 变体（「单行 timing 的覆盖数没实测」）——**稀疏事件的份额必须按总量算**；
> 解析器输出也是判据，判据要能对自己失败。

## 7 · 踩的坑

- vLLM 在正常收尾时打 "Engine core died unexpectedly" + Traceback——**拆机噪声不是训练失败**，
  判据要看 trainer 步数走没走完。
- ★ **首次权重同步的偶发引擎死亡再现一例**（e29v_resume，2026-08-20 09:34）：数值校验通过后、
  首次同步广播 8.4 GB 基座的窗口里，EngineCore 无声死亡（无 traceback、KV 余量 15 GB、
  主机内存 434 GB 空闲、cgroup 上限 483 GB——OOM 假设两次被证伪），两个同步对端卡死在
  `update_weights` 的 NCCL 集合里 17 min。**同命令复跑通过**（当日同路径 5/6 存活）⇒
  归档为已知偶发（P4 家族：rollout 卡首次同步余量贴边），与 E29 存档路径正交。
  留观：若频率上来，先量首次同步瞬时显存峰值再动手。
- 排查时差点误杀主线 GPU0 的常驻 vLLM 端点（`VLLM::EngineCore` 30 GB）——**清残留必须先按
  gpu_uuid 归卡再按 PID 杀**，共存机器上 pkill 式清理是事故预备役。
- 冒烟 save 7.9 s vs 生产 8.8 s（27 GB vs 18 GB）不成比例 ⇒ save 里有固定开销分量，
  外推别用线性。

## 8 · 下一步

- 续跑验证落数后：状态改 ✅；素材（补丁 diff + 本报告数字）交 upstream 同事作第 5 包候选
- 异步写盘：**不做**（save 已 0.83 s，占跑 <0.1%，没有分母了）——对上 01 §1-1 收尾
