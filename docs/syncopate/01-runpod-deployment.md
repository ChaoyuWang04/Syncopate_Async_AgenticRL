# Syncopate · 01 — RunPod 部署清单（H100 80GB × 1）

> 🔴 **状态：备用路径，从未启用。** 项目实际一直跑在 **4×RTX 5090 / sm_120** 上
> （2026-08-17 换过一次机器，仍是 4×5090）—— sm_120 的 wheel 问题已经逐条解掉，
> **不需要退到 H100**。这份留作「单卡节点从零搭起」的参考清单，
> **当前环境的唯一真相是 `08-machine-and-environment.md`**，别照这份配环境。
>
> 目标：在一台干净的 H100 80GB 单卡节点上，从零到能跑通老师训练包的 **minimal 接线验证**（Task 02 §D2 配置）。
> 场景：本地 5090（sm_120）环境不可用时的备用路径。H100 是 sm_90，wheel 生态成熟，**不存在 sm_120 的兼容风险**。
> 预计耗时：40–60 分钟（大头是 pip 下载 ~8GB 和模型下载 1.5GB）。

---

## 0. 开机前：选配置

| 项 | 选择 | 理由 |
|---|---|---|
| GPU | **H100 80GB PCIe/SXM × 1** | sm_90，torch/vLLM 官方 wheel 直接可用 |
| 模板 | **RunPod PyTorch 2.x（CUDA 12.4+）** 或 `runpod/base:cuda12.4` | 只需要驱动和 CUDA runtime，torch 我们自己装进 venv |
| 磁盘 | **Container ≥ 40GB，Volume ≥ 60GB** | pip 缓存 + venv ≈ 15GB，模型 1.5GB，数据 200MB，checkpoint 预留 |
| Python | 需要 **3.10–3.12**（verl `requires-python>=3.10`） | 3.13 太新，vLLM 0.12 无 wheel |

> ⚠️ 别选 CUDA 11.x 的模板——vLLM 0.12 / torch 2.9 需要 CUDA 12.x runtime。

---

## 1. 传文件（本地 → RunPod）

### 1.1 打包（在本地执行）

最小传输集 = **代码 6.8MB + 数据 170MB**（完整 `data/` 是 583MB，其中 `batches/{sft,rl,eval}` 三个子目录对 RL 训练**不需要**）：

```bash
cd /home/samwang/code/projects/verl-async-agentic-rl/reference

tar czf /tmp/syncopate-pack.tgz \
  --exclude='.venv-verl' \
  --exclude='__pycache__' \
  --exclude='checkpoints/*' \
  --exclude='runs/*' \
  --exclude='data/sft' \
  --exclude='data/batches/sft' \
  --exclude='data/batches/rl' \
  --exclude='data/batches/eval' \
  industrial_posttrain_training_release

ls -lh /tmp/syncopate-pack.tgz     # 预期 ~120MB（压缩后）
```

**为什么保留 `data/batches/stage5_full/`（141MB）**：RL parquet 每行的 `extra_info` 指向四个文件——`cases/`、`env_snapshots/`、`gold_paths/`、`verifier_specs/`，AgentLoop 运行时会逐条读。少了它 rollout 第一条就报 FileNotFoundError。

**为什么保留 `data/rl/`（29MB）**：train/eval parquet 本体。

### 1.2 上传

```bash
# 方式 A：RunPod 提供 SSH（推荐）
scp -P <PORT> /tmp/syncopate-pack.tgz root@<POD_IP>:/workspace/

# 方式 B：走 runpodctl（无需 SSH 配置）
runpodctl send /tmp/syncopate-pack.tgz
# 然后在 pod 内执行它打印出的 `runpodctl receive <code>`
```

### 1.3 解压

```bash
cd /workspace && tar xzf syncopate-pack.tgz && cd industrial_posttrain_training_release && ls
# 应看到 agent/ envs/ train/ scripts/ configs/ verl/ data/
```

---

## 2. 装环境

```bash
cd /workspace/industrial_posttrain_training_release

# 确认 python 版本在 3.10–3.12
python3 --version

PYTHON_BIN=python3 bash scripts/setup_env.sh 2>&1 | tee /workspace/setup.log
```

脚本做四件事（`scripts/setup_env.sh`）：建 `.venv-verl` → 装 `.[monitoring]`（项目自身）→ 装 `verl/upstream[vllm]`（**随包 verl 快照**）→ 装 pandas/pyarrow/transformers/accelerate/sentencepiece。

**验证**：

```bash
.venv-verl/bin/python -c "
import torch, vllm, verl
print('torch   ', torch.__version__, 'cuda', torch.version.cuda)
print('vllm    ', vllm.__version__)
print('verl    ', verl.__version__ if hasattr(verl,'__version__') else open('verl/upstream/verl/version/version').read().strip())
print('gpu     ', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))   # H100 应为 (9, 0)
"
```

预期：torch 2.9.0（vLLM 0.12.0 钉死）、vllm 0.12.0、verl 0.8.0.dev、capability (9,0)。

> **注意**：`verl/upstream/setup.py` 里的 `torch==2.9.1` 属于 **SGLANG_REQUIRES**，走 `[vllm]` extra 时不生效，torch 版本由 vLLM 决定。

---

## 3. 配 `.env`

```bash
cp .env.example .env
sed -i 's/^VERIFIER_PROVIDER=.*/VERIFIER_PROVIDER=none/' .env
sed -i 's/^WANDB_MODE=.*/WANDB_MODE=offline/' .env
```

**接线验证阶段用 `VERIFIER_PROVIDER=none`**：verifier 走启发式兜底（`agent/verifier.py:471`），reward 数值无意义但链路完整，不需要通义 API key。真实训练时再填 `QWEN_API_KEY`。

`WANDB_MODE=offline` 避免卡在登录。

---

## 4. 下模型

```bash
.venv-verl/bin/python -m pip install huggingface_hub
.venv-verl/bin/python -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-0.6B \
  --local-dir models/original_model/Qwen3-0.6B
```

**为什么是 Qwen3-0.6B 而不是 Qwen2.5-0.5B**：老师的 `parse_tool_calls` 解析 **Qwen3 风格的 `<tool_call>` 文本块**，SFT 数据里还有 `enable_thinking` 字段。用 Qwen2.5 会引入模板变量。

> 若 HF 访问慢：`export HF_ENDPOINT=https://hf-mirror.com` 后重试。

---

## 5. Dry-run（先确认 Hydra override 拼对了，不启训练）

```bash
DRY_RUN=1 LOG_DIR=/workspace/logs \
MODEL=models/original_model/Qwen3-0.6B \
N_GPUS_PER_NODE=1 \
bash scripts/run_agenticrl_stage.sh
```

检查打印出的命令里：`actor_rollout_ref.model.path` 指向 0.6B、`trainer.n_gpus_per_node=1`、`agent_loop_config_path` 指向 `configs/verl_agent_loop.yaml`。

---

## 6. 正式跑 minimal 接线验证

```bash
cd /workspace/industrial_posttrain_training_release

MODEL=models/original_model/Qwen3-0.6B \
N_GPUS_PER_NODE=1 NNODES=1 \
TOTAL_STEPS=1 TRAIN_MAX_SAMPLES=2 VAL_MAX_SAMPLES=1 \
ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=2048 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.5 \
AGENT_WORKERS=1 \
LOGGER=console TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=false MERGE_HF=0 \
LOG_DIR=/workspace/logs \
SAVE_PATH=/workspace/checkpoints/smoke \
bash scripts/run_agenticrl_stage.sh \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
```

**参数说明（与本地 5090 版的差异）**：

| 参数 | 5090 (32GB) | H100 (80GB) | 原因 |
|---|---|---|---|
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.35 | **0.5** | 显存宽裕，给 vLLM 更多 KV cache |
| 其余 | 同 | 同 | — |

**`MAX_PROMPT_LENGTH=8192` 不能再降**：prompt 正文 ~1500 token，但 `apply_chat_template` 会注入 31 个工具的 schema（13681 字符 ≈ 4000–5500 token），合计 6000–7000。而 `_pad_token_ids` 用 `tokenizer.pad(padding="max_length")` **不截断**，超长不报错、直接在 batch 拼接时炸 shape mismatch。

---

## 7. 验收标准（接线通了的标志）

按出现顺序检查：

1. ✅ vLLM engine 起来（日志有 `Loading model weights` / KV cache 分配）
2. ✅ **AgentLoop 注册成功**——无 `industrial_posttrain_agent not found`
3. ✅ rollout artifact 落盘：`ls data/rollouts_verl/<run_name>/` 有 `scores.jsonl`
4. ✅ `critic/rewards/mean` 有数值（哪怕是 0）
5. ✅ 走完 1 个 step 无异常退出

**不要求**：reward 上涨、格式正确率高。0.6B 大概率产不出合法 `<tool_call>`，`parse_error` 100% 是**正常的**——只要 parse_error 反馈循环走通、artifact 落盘、reward 进 advantage，就算通过。**别在这里调模型。**

---

## 8. 接线通过后：立刻验证 TIS 指标链路

这是 [[00-research-question]] 的前置条件，趁环境热着做：

```bash
# 在上面的命令后追加：
  algorithm.rollout_correction.rollout_is=sequence \
  algorithm.rollout_correction.bypass_mode=false \
  actor_rollout_ref.rollout.calculate_log_probs=True
```

然后确认日志里有 `rollout_corr/` 前缀的指标，**至少要有这四个**：`kl`、`k3_kl`、`rollout_is_eff_sample_size`、`chi2_seq`。

有了它们，[[00-research-question]] §4.2 的 σ² 反解自检就能做了（`σ² ≈ 2·k3_kl − kl²`），**零额外成本**。

---

## 9. 常见故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: data/batches/stage5_full/cases/*.json` | 打包时误排除了 stage5_full | 补传该目录（141MB） |
| `ValueError: Invalid rollout_is` | 配了 `geometric` | 0.8.0.dev 只支持 `token`/`sequence` |
| batch 拼接 shape mismatch | `MAX_PROMPT_LENGTH` 太小 | 升到 8192+ |
| `industrial_posttrain_agent` 未注册 | `PYTHONPATH` 没含项目根 | 一键脚本已 export，确认是从项目根启动 |
| 卡在 wandb 登录 | `.env` 没改 | `WANDB_MODE=offline` |
| CUDA OOM | vLLM 和训练峰值叠加 | 降 `ROLLOUT_GPU_MEMORY_UTILIZATION` 到 0.3 |
| pip 解析极慢 | 在 backtrack | 正常，torch/vllm 依赖树很深，耐心等 |

---

## 10. 附：本地 5090 路线其实可能还活着

2026-07-28 本机跑 `setup_env.sh` 失败了，**但失败原因是带宽不是兼容性**：

- 依赖解析**零冲突**通过：torch 2.9.0 + vllm 0.12.0 + triton 3.5.0 + **nvidia-\*-cu12 12.8**；
- 卡在下载 4.9GB 大件，实测 150–550 kB/s（`pip.conf` 没配镜像，直连 PyPI）；
- **装到的是 CUDA 12.8 wheel，而 cu128 支持 sm_120** —— 5090 到底能不能跑，**至今没有证据说不能**。

**所以在开 RunPod 之前，值得先花 10 分钟做一次带镜像的本地重试**：

```bash
cd /home/samwang/code/projects/verl-async-agentic-rl/reference/industrial_posttrain_training_release
PYTHON_BIN=python3.12 \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/setup_env.sh
# 半成品 .venv-verl 和 PIP_CACHE_DIR 都在，可续传
```

装完直接验证 sm_120：

```bash
.venv-verl/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability(0))          # 5090 应为 (12, 0)
print((torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum())  # 真跑一次 kernel
"
```

若这里报 `no kernel image is available for execution on the device`，**那才是真正的 sm_120 判决**，此时再转 RunPod。

---

## 11. 省钱提示

- **接线验证跑完立刻 stop pod**，别让它空转。RunPod H100 约 $2–3/hr。
- 装好环境后 **做一个 volume snapshot / 保存 template**，下次开机跳过第 2 步（省 30 分钟 × 每次）。
- Phase 1 正式训练前，先在本 pod 上把所有配置改动验证完，避免多卡上试错。
