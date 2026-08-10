# 00 · 课程项目现状调查（2026-07-27）

> 调查对象：`reference/industrial_posttrain_training_release/`（老师提供的工业级 post-training 训练包，版权：深圳途明智启科技，不得传播/商用）
> 调查目的：完全理解代码结构，为改造成自己的项目（Syncopate：sync colocate vs fully-async 对比）做准备。

---

## 1. 定位与结构

### 1.1 目录树（关键部分）

```text
industrial_posttrain_training_release/
├── agent/                     # Agent runtime、provider、verifier（老师自研）
│   ├── runtime.py             #   standalone agent loop + parse_tool_calls（Qwen3 <tool_call> 文本协议）
│   ├── verifier.py            #   1196 行规则+LLM judge 混合打分器（reward 核心）
│   ├── trajectory.py          #   轨迹数据结构
│   ├── rollout_store.py       #   rollout artifact 落盘
│   ├── prompts/               #   system/step_user/tool_error_feedback/verifier_llm 模板
│   └── providers/             #   api/vllm/local_hf provider + factory（judge 接入点）
├── envs/                      # 售后业务 sandbox（老师自研）
│   ├── toolfactory.py         #   工具注册工厂
│   ├── sandbox_state.py       #   从 env_snapshot 构造带 namespace 的沙盒状态
│   └── toollist/              #   35 个售后工具（OMS/TMS/退款/发票/工单/风控…）
├── train/                     # ★ verl 接线层（最重要的自定义部分）
│   ├── verl_agent_loop_adapter.py   # AgentLoopBase 子类，420 行
│   ├── verl_reward_adapter.py       # verifier reward + artifact 落盘
│   ├── grpo_builder.py / sft_builder.py  # parquet 数据构造
├── verl/
│   ├── upstream/              # ★ 固定随包的 verl 源码快照（无 .git）
│   └── adapters/              # 另一份 adapter（agent_loop/dataset/reward，与 train/ 下的关系待查）
├── scripts/                   # 一键脚本：setup_env / build_data / run_sft / run_agenticrl / serve_vllm
├── configs/                   # verl_agent_loop.yaml（AgentLoop 注册）+ 项目自有 yaml
├── schemas/                   # case/env/reward/route/trajectory/verifier 的 pydantic schema
├── routing/                   # 数据路由/难度分池（sampling_policy、promotion）
└── data/                      # 583MB：batches 源文件 + sft/rl parquet
```

### 1.2 是 fork 还是外挂？

**"pip 装自带快照 + 外挂 adapter"模式**，三层结构：

1. `verl/upstream/` 是一份**完整 verl 源码快照**（非 fork、无 .git、无法 diff 上游），通过 `pip install -e ./verl/upstream[vllm]` 安装；
2. 老师的业务代码（agent/envs/train/schemas/routing）是独立 Python 包（`industrial-posttrain-project`），与 verl 平行安装；
3. 两者只通过 **verl 的官方扩展点**对接：`AgentLoopBase` 子类 + `@register("industrial_posttrain_agent")` + `agent_loop_config_path` 配置文件。**没有发现对 verl 源码本体的魔改痕迹**（待逐一确认，见 §6）。

### 1.3 verl 版本

- `verl/upstream/verl/version/version` = **`0.8.0.dev`**（不是课件说的 v0.6）
- torch 依赖 pin 为 `torch==2.9.1`，vLLM 约束 `>=0.8.5,<=0.12.0`
- 关键佐证：`fully_async_policy` 已从早期版本的 `recipe/` 迁移到 `verl/experimental/fully_async_policy/`，这是 0.8 时代的布局

---

## 2. 入口与配置

### 2.1 入口清单（两级启动结构）

| 一键 shell | Python 启动器 | 最终入口 |
|---|---|---|
| `scripts/run_sft_stage.sh` | `scripts/train_sft.py` | `torch.distributed.run -m verl.trainer.sft_trainer`（engine=fsdp） |
| `scripts/run_agenticrl_stage.sh` | `scripts/train_grpo_verl.py` | `python -m verl.trainer.main_ppo` + Hydra overrides |
| `scripts/build_training_data.sh` | `scripts/build_sft.py` / `build_grpo.py` | parquet 构造 |
| `scripts/serve_vllm.sh` | — | 独立 vLLM 服务（standalone 评测用） |

设计模式：shell 只管环境变量/日志/后处理；Python 启动器集中维护 Hydra overrides；额外 override 可从命令行透传（`bash run_agenticrl_stage.sh actor_rollout_ref.actor.optim.lr=5e-7`）。`DRY_RUN=1` 可打印完整命令不执行——**读配置的最佳工具**。

### 2.2 GRPO 关键配置（train_grpo_verl.py 摘录）

```yaml
algorithm:  adv_estimator=grpo, use_kl_in_reward=False
actor:      lr=1e-6, use_kl_loss=True (coef=0.001), entropy_coeff=0,
            FSDP + param_offload + optimizer_offload
rollout:    name=vllm, mode=async, TP=1, gpu_memory_utilization=0.2,
            enforce_eager=True, free_cache_engine=True, n=8 (ROLLOUT_N)
multi_turn: enable=True, max_assistant_turns=8, max_user_turns=8, max_parallel_calls=1
agent:      default_agent_loop=industrial_posttrain_agent
            agent_loop_config_path=configs/verl_agent_loop.yaml
critic:     enable=False        # GRPO 无 critic
reward_model: enable=False      # reward 由 AgentLoop 内部产生，不走 RM
data:       max_prompt_length=12288, max_response_length=4096, return_raw_chat=True
trainer:    NNODES=1, N_GPUS_PER_NODE=64（默认单机 64 卡！）
模型:        Qwen3-8B（models/original_model/Qwen3-8B）
```

另有一个默认开启的进阶特性：**rollout correction（训推不一致处理）**——`calculate_log_probs=True` 保存 vLLM rollout logprob，训练侧做 sequence-level truncated importance sampling（`rollout_is=sequence, threshold=2.0`）。这是 2025-2026 年"训推分离偏差"的工业级处理，值得单独精读。

---

## 3. 老师的自定义部分（重点）

**verl 本体零修改（初步判断），全部自定义集中在扩展点上：**

### 3.1 AgentLoop adapter（`train/verl_agent_loop_adapter.py`，核心中的核心）

`IndustrialPosttrainAgentLoop(AgentLoopBase)`，一条 rollout 的完整生命周期：

1. 从 parquet row 的 `extra_info` 读 case/env_snapshot/verifier_spec 三个 JSON 路径，构造带唯一 namespace 的 `SandboxState`（同 case 的 8 条 rollout 互不污染）；
2. 循环：`server_manager.generate()` 让 verl/vLLM 生成 token → 解析 `<tool_call>`（**文本协议**，不是 vLLM native tool calling）→ 执行 ToolFactory 工具 → observation 以 user 消息追加回上下文；
3. **loss masking**：模型生成 token `response_mask=1`，tool observation / parse_error feedback token `response_mask=0`——agentic RL 最经典的正确性关键点，实现在 adapter 内部逐段维护；
4. parse_error 不执行工具，渲染 `tool_error_feedback.txt` 反馈给模型重试（错误计入 metrics）；
5. 结束后调 `verl_reward_adapter.score_and_persist_rollout()` 打分并落盘 `token_trace`（每段 token 标注 prompt/model/tool/feedback 来源——审计对齐问题的利器）。

### 3.2 Reward：verifier（`agent/verifier.py`，1196 行）

**规则 + LLM judge 混合打分**，不是单一 scalar：

- 子分数：write_score（写操作对不对）、info_score、policy_score、evidence_score、efficiency_score、communication_score；
- caps 机制：`calculate_caps` 对危险行为封顶（重复写、customer_harm、未 dry-run 就退款等）；
- LLM judge 通过 `VERIFIER_PROVIDER`（qwen/deepseek/none）接入，`none` 时有 heuristic 兜底判断——本地无 API key 也能跑通接线。

### 3.3 工具沙盒（`envs/`）

35 个跨境电商售后工具（订单/物流/退款/发票/工单/风控/订阅…），读写分离（写工具是目录、内含审批流），全部作用于内存 SandboxState——**无外部服务依赖，rollout 可完全离线**。

### 3.4 数据（`data/`，583MB）

RL parquet（2273 train / 304 eval）每行是 **prompt-only** 结构：

| 字段 | 内容 |
|---|---|
| `prompt` | `[system, user]` 两条消息（中文售后 Agent 指令 + 工单上下文） |
| `extra_info` | case_path / env_snapshot_path / gold_path / verifier_spec_path + difficulty(L4)/intent 等元数据 |
| `reward_model.style` | `industrial_posttrain_verifier_with_deepseek_judge`（实际 reward 在 AgentLoop 内算） |

多轮轨迹**不在数据里**，是训练时在线 rollout 出来的。SFT parquet（121/14 行）则是完整 11 条 messages 的多轮对话 + tools 字段（离线专家轨迹）。

数据分池：`batches/{sft,rl,eval,stage5_full}`，routing/ 模块负责按难度/意图路由，eval 池 held-out 绝不进 train。

---

## 4. 环境可运行性（本机 5090）

| 项 | 状态 |
|---|---|
| GPU | RTX 5090 32GB，Driver 595.71.05，CUDA 13.2，sm_120 |
| 老师包的 `.venv-verl` | ❌ 未创建（`setup_env.sh` 未跑过） |
| 系统 python | 3.13（对 vLLM 偏新，不建议直接用） |
| **可复用**：conda `verl-omni` | ✅ python 3.12 + torch 2.11 + **vllm 0.22.0** + verl 0.9.0.dev（editable，指向 `~/code/upstream/verl`）+ ray 2.55 + flashinfer |

**判断：单卡 32GB 能跑的最小规模**（未实测，纯推算）：

- **可行路径**：Qwen2.5-0.5B/1.5B 或 Qwen3-0.6B/1.7B full-param GRPO + colocate vLLM rollout。1.5B 级 fp32 AdamW 状态约 18GB，叠 `gpu_memory_utilization=0.2`（约 6.5GB）+ 激活，32GB 够；老师脚本已含 param/optimizer offload 兜底。8B 模型 full-param 训练单卡不可行（仅推理可行）。
- **必改参数**：`N_GPUS_PER_NODE=64→1`、`MODEL=<小模型>`、`MAX_PROMPT_LENGTH=12288` 视 KV 预算酌减；调试参数老师已备好（`TOTAL_STEPS=1 TRAIN_MAX_SAMPLES=1 ROLLOUT_N=1 LOGGER=console`）。
- **风险点**：① 老师包 pin `torch==2.9.1` + vLLM ≤0.12，在 sm_120 上的 wheel 可用性需实测——**建议优先试 verl-omni 环境（vllm 0.22 对 Blackwell 支持更好）直接跑老师的启动器**，verl 0.8→0.9 的 AgentLoop 接口兼容性是主要变数；② reward 需要 `QWEN_API_KEY`（通义 judge），本地接线检查可先 `VERIFIER_PROVIDER=none`；③ 中文长 prompt（12K）+ 8 轮工具循环，rollout 每步偏慢属正常。

---

## 5. 待调查清单（2026-07-27 Task 02 更新）

### ✅ 已消除

1. ~~**`sitecustomize.py`**~~ → **无害**。全文只有 `sys.dont_write_bytecode = True`，防止陈旧 .pyc 覆盖新源码（老师踩过 `false_promise_cap` 旧字节码导致 holdout 假红的坑）。**没有任何 monkey-patch、没有改 sys.path**。
2. ~~**双份 adapter**~~ → `verl/adapters/` 下的两个文件是**纯说明性存根**，正文写着"真正可运行的 adapter 在 `train.verl_agent_loop_adapter`"，原因是包名 `verl` 会和上游 verl 冲突（shadow）。**活的只有 `train/` 那份**，`configs/verl_agent_loop.yaml:11` 指向它。
3. ~~**verl 快照是否零魔改**~~ → **高置信度未改逻辑**（详见 Task 02 报告 §A3）。唯一本地化痕迹是 `experimental/agent_loop/agent_loop.py` 里 **108 行中文教学批注**（开头自述"仅注释，不改任何逻辑"），程序化校验确认全部为注释行，0 行是代码。无 `.orig/.rej/.patch` 残留。
4. ~~**rollout correction（TIS）细节**~~ → 已完成精读，独立成篇：[[02-train-inference-mismatch]]。
5. ~~**reward 回传路径**~~ → 已追通全链路（Task 02 §C）：`verifier.score_trajectory` → `AgentLoopOutput.reward_score` → `agent_loop.py:1063-1069` 转成 token 级 `rm_scores`（只挂末位 token）→ `reward.py:164 extract_reward` → `ray_trainer.py:1592 token_level_scores` → `:1624 compute_advantage` → `core_algos.py:304` 对序列求和还原标量。

### ⬜ 仍待调查

6. **routing/ 模块**：sampling_policy、promotion 的难度分池逻辑如何驱动 stage5 数据构成？（"课程学习"式数据策略）
7. **verifier 的 reward shaping 全貌**：subscores 加权、caps 封顶、judge 失败的 fallback 路径，值得画一张打分流程图。
8. **`rollout_is=sequence` 在 8 轮 multi-turn 下是否合适**：`masked_sum` 会把全轨迹所有模型 token 的 log_ratio 相加，序列越长越容易撞上截断上界。见 [[02-train-inference-mismatch]] §6。

---

## 6. 与我们项目（Syncopate）的对接点

- 老师包的 `rollout.mode=async` 只是 **AgentLoop 协程级 async**（colocate 架构内的异步生成），不是我们要研究的 fully_async_policy 分离架构——两者区别正是 Task 2 的核心。
- 好消息：`verl/upstream/verl/experimental/fully_async_policy/` 完整在包里（trainer/rollouter/main + config + shell 示例），Phase 2 读码可直接用这份快照。
- 老师的 adapter 模式（AgentLoopBase 子类 + 文本工具协议 + verifier reward）就是我们 Phase 0/1 自建任务时应模仿的骨架；`token_trace` 落盘设计值得抄。
