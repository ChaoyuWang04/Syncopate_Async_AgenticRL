# Syncopate · 08 — 机器、环境与运行配置

> 2026-08-13 在 4×RTX 5090（RunPod）上从零搭起来并逐条实跑验证过。
> **这份是"怎么把它跑起来"，方法论和研究结论在 `05-handoff.md`。**

---

## 1 · 硬件画像（实测）

> 🔴 **2026-08-17 换了一台机器**（仍是 4×5090）。下面是**当前这台**；旧机器的画像留在末尾对照。

```
🆕 当前机器（2026-08-17）
4 × RTX 5090  32607 MiB / 575W / sm_120   驱动 595.58.03  CUDA 12.8  torch 2.9.0+cu128
CPU   2× EPYC 9V74（80 核/路，320 线程，2 NUMA）   RAM 503 GB
拓扑  🔴 **2+2 跨 socket**：GPU0/1@node0、GPU2/3@node1，组内 NODE、跨组 SYS（走 UPI）
PCIe  **Gen5 x16**（32 GT/s；空闲时报 Gen1 属正常）
盘    /workspace = 本地 XFS 300G（**持久卷**，workspace_is_volume=true，权限位生效）
      / 只有 **16G overlay** ⇒ 见 §2.0，缓存必须重定向

〔旧机器，2026-08-13〕
驱动 570.195.03 · CPU 2× EPYC 7543（120 核）· RAM 944 GB
拓扑 GPU0–3 两两 PHB、**全部 NUMA 0，四卡完全对称**
盘   /workspace 是网络盘（mfs），有 df 看不见的卷配额
```

🔴🔴 **`/workspace` 有卷配额，而 `df` 看不到它 —— 2026-08-14 因此丢了一个 ckpt。**

```
df -h /workspace     →  733 T 可用     ← 这是底层网络文件系统，不是给你的额度
真实额度              →  当时 100 G（已扩到 200 G）
```

**超限是静默的**：`cp` 产出 0 字节文件不报错（`cat` 才会喊 `Disk quota exceeded`）；
M7 收尾那个 27 GB 的 ckpt 被写到一半掐断 —— 三个 rank 分片大小各不相同、
zip 中央目录缺失、`torch.load` 报 `failed finding central directory`，
**而训练日志里一个字都没有**，进程还是 exit 0。

⇒ **判断空间只能用写入探针**（真写几百 MB 再删），别信 `df`：

```bash
python - <<'PY'
import os; p="/workspace/.quota_probe"; CH=os.urandom(1<<20); n=0
try:
    with open(p,"wb") as f:
        for _ in range(5000): f.write(CH); n+=1
    print(f"写入 {n} MiB 未触上限")
except OSError as e: print(f"写到 {n} MiB 失败 → {e}")
finally:
    os.path.exists(p) and os.remove(p)
PY
```

⇒ **容量账**：fully_async 一个 ckpt **27 GB**（3 个 rank 的全量 state_dict，
其中 97% 是和基座逐字节相同的冻结权重）。200 G 也只放得下 7 个。
跑完用 `scripts/prune_rl_ckpts.py` 瘦身（只留 LoRA，约 250 MB）。

🔴🔴 **〔旧机器〕`/workspace` 不支持权限位 —— M9 的 PostgreSQL 因此放不进去**

> ⚠️ **2026-08-17 起这一整段只对旧机器成立。** 这台是本地 XFS，`chmod 700` 实测生效 ⇒
> `PGDATA=/workspace/pgdata/16/syncopate`（已写进 `/workspace/.env`），**数据库不再是
> 「重启即丢的派生产物」**。下面保留原文是因为「推翻的预期不删」这条约定。


```
chmod 700 /workspace/xxx   →  仍是 777    （mfs 是 FUSE，不认权限位，实测）
capsh --print              →  !cap_sys_admin   ⇒ 不能 mount
                              ⇒ 「在 /workspace 上放个 ext4 镜像再 loop 挂载」这条路也堵死
PostgreSQL                 →  PGDATA 必须 0700 或 0750，否则**拒绝启动**（硬编码检查）
```

⇒ **PGDATA 在物理上放不进 `/workspace`。** 处理办法是把数据库降级成**派生产物**：

| 东西 | 放哪 | 重启后 |
|---|---|---|
| PG 二进制/库 | `/workspace/tools/postgres/root` | ✅ 在 |
| deb 离线包（5 个，19 M） | `/workspace/tools/postgres/debs` | ✅ 在，断网也能重装 |
| **schema / 迁移** | **仓库** `syncopate/runtime/schema.sql` | ✅ 在（真相来源） |
| PGDATA | 本地盘 `/var/lib/postgresql/16/syncopate` | ❌ 丢（可重建） |

```bash
bash scripts/pg_bootstrap.sh          # 幂等：initdb → 起服务 → 建库 → 应用 schema
bash scripts/pg_bootstrap.sh --reset  # 推倒重来
```

⚠️ 真要长期保存业务数据，得换**支持权限位的卷**或托管 PG —— 那是部署问题，不是开发期问题。

⚠️ 二进制丢了（容器换了镜像）时的重装，不需要联网：
```bash
mkdir -p /workspace/tools/postgres/root
for d in /workspace/tools/postgres/debs/*.deb; do dpkg -x "$d" /workspace/tools/postgres/root/; done
```

🔴 **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起就被驱动关掉了
PCIe P2P，5090 同样 —— **这是所有 4×5090 机器的常态，不是这台坏了**。
⇒ 卡间通信一律经主机内存中转。

**🆕 当前机器实测**（`scripts/probe_allreduce_bw.py`，口径同 README §6：
all-reduce busbw / `NCCL_CUMEM_ENABLE=0` / 256MB）：

```
组内 (0,1)/(2,3)  28.8 GB/s      跨 socket (0,2)/(1,3)  22.2      四卡（DDP 走这条）25.6
⇒ 跨 socket 只掉 22%；NUMA 绑定救不回来（22.23→22.34，噪声内）＝ UPI 跳的物理代价
⇒ 换算负载：DDP 梯度 260MB 从旧机 40.4 ms → 10.2 ms，跨 socket 净代价 1.2 ms/步 ＝ 一步的 0.004%
⇒ 传输通道实测 SHM/direct/direct（P2P 仍全 0）
```

〔旧机器〕2 卡同 NUMA **6.44 GB/s** —— 🔴 **这个数是 README §6 的全局常量、
E02/E07/E11/E12 都拿它当分母，现在作废了，必须在这台重测。**

⬜ 还没测：主机内存带宽、**满载功耗与降频**（4×575W=2.3kW，会污染所有对照实验，队列 A7）。

---

## 2 · 从零搭环境（新机器按顺序做）

```bash
set -a; . /workspace/.env; set +a                  # ★ 先加载持久化重定向，见 2.0
uv sync --extra train --extra dev --extra runtime  # ← 三个 extra 都要，见 2.1
hf download Qwen/Qwen3-4B   --local-dir models/Qwen3-4B     # 7.6G
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B   # 1.4G，31 个测试要它
python scripts/ingest_external.py                  # ★ 派生数据，不跑的话 27 个测试红
bash scripts/pg_bootstrap.sh                       # ★ 不起 PG 的话 45 条 runtime 测试 skip
# reference/ 870M 只能手动 scp —— 版权所有（深圳途明智启科技），永不进 git
python -m pytest -q                                # 应为 365 passed, 0 skipped
```

⚠️ **2026-08-17 换机器时发现，上面这个序列以前是错的** —— 照旧版做完是
「27 failed / 45 skipped」，四个缺口全都**只在干净机器上暴露**：

| 缺口 | 症状 | 已修 |
|---|---|---|
| `--extra runtime` 不存在（M9 依赖从没进依赖表） | `asyncpg`/`fastapi` 缺失，4 个测试模块**收集阶段**就 ImportError | pyproject 加 `runtime` extra |
| `ingest_external.py` 不在本节（它在 §3） | `data/external/ingested.json` 是 gitignore 的派生产物，缺了 27 红，签名像"数据过期"极易误判 | 提到本节 |
| `pg_bootstrap.sh` 不在本节 | 45 条 runtime 测试静默 skip —— 而它们自己的 docstring 写着**「不要把跳过当通过」** | 提到本节 |
| `install_flash_attn_shim.py` | 垫片**已退役**（满足 import 不满足契约），真轮子现在由 `[tool.uv.sources]` 装 | 从序列里删掉 |

### 2.0 ★★ 持久化重定向（`/workspace/.env`）

这台机器 `/` 是 **16 GB 的 overlay**（`/root` `/tmp` `/var` `/home` 都在上面，recycle 即丢），
`/workspace` 才是 **300 GB 持久卷**（`vast-capabilities` 的 `workspace_is_volume=true`）。
镜像的 `bootstrap.sh` 只管了 uv/npm/HF 的缓存，**训练栈这几个默认全落在 overlay 上**：

```
RAY_TMPDIR            ★ Ray 对象溢写，verl 四卡能吃掉几十 GB，16 G 直接爆
TRITON_CACHE_DIR      Triton JIT
TORCH_EXTENSIONS_DIR  融合 kernel 编译
VLLM_CACHE_ROOT / CUDA_CACHE_PATH / XDG_CACHE_HOME / TMPDIR / PIP_CACHE_DIR
PGDATA                见下
LD_LIBRARY_PATH       PG 的 libpq（`dpkg -x` 不进 ldconfig 路径）
```

⇒ 全部写在 `/workspace/.env` 里，**每个 shell 先 `set -a; . /workspace/.env; set +a`**。
实测：整个环境重建 + 跑通管线，overlay 用量从 48 M 没涨过。

🟢 **PGDATA 的限制在这台机器上解除了**：这台 `/workspace` 是 **XFS**（不是旧机器的 mfs），
实测 `chmod 700` 生效 ⇒ `PGDATA=/workspace/pgdata/16/syncopate`，
**数据库不再是"重启即丢的派生产物"**。下面 §1 里那段 mfs 的论证是**旧机器的**，别照搬。

### 2.1 依赖的三个坑（都已修进 pyproject / uv.toml）

| 坑 | 表现 | 修法 |
|---|---|---|
| **numpy 冲突** | `uv sync` 直接判 unsatisfiable（verl 钉 `<2.0`，我们要 `>=2.2`） | `[tool.uv] override-dependencies` |
| **numpy 版本** | 装成 2.5 ⇒ **vLLM 起引擎那一刻**才炸（numba 卡 ≤2.2） | 钉 `numpy>=2.2,<2.3` |
| **tensordict** | verl 依赖表写 `>=0.8.0`，**代码里 `assert >=0.10`** ⇒ 分卡异步必踩 | 钉 `tensordict==0.10.0` |
| **openpyxl 不在依赖表** | 数据再生链第一步就断 | 已补进 `[project] dependencies` |

⚠️ **uv 有一条警告是错的，别照着"修"**：它说「同目录有 uv.toml ⇒ 整段 `[tool.uv]`
被忽略」。实测 override 写 `[tool.uv]` 里**能装出来**，挪到 uv.toml 反而 unsatisfiable。

⚠️ **`uv.lock` 在 .gitignore 里** ⇒ 每台机器解析结果可能不同。想要真正可复现，
该考虑把它放进版本管理。

### 2.2 ★★★ flash-attn：判据必须包含**反向**

verl 0.8 在 CUDA 路径上**无条件** `from flash_attn.bert_padding import ...`，
所以这个包必须在。~~垫片~~**已退役**（满足 import 不满足契约）。

**2026-08-17 踩到更深的一层，务必读完再换轮子：**

社区那个 `cu128torch2.9cxx11abiFALSE` 轮子（版本号完全匹配、我验过前向三项全过）——
**反向是坏的**：

```
flash_attn_func 反向         dq/dk/dv 全 nan
flash_attn_varlen_func 反向  有限但**恒为 0**（参考值 136/178/1450）  ← verl rmpad 走这条
```

后果：RL 每步 `grad_norm=nan` ⇒ verl 打 `WARN: grad_norm is not finite` 并
`optimizer.zero_grad()` **跳过更新** ⇒ **训练完全空转，模型一次没动过**。
⚠️ **返回 0 那条比 nan 更毒**：没有 nan、没有报错，训练"正常跑完"什么都没学到。

⇒ **教训：「import 成功 ≠ 契约满足」的下一层是「前向对 ≠ 反向对」。**

**现在用的（可直接照抄）**：官方 `cu13torch2.9` 轮子 —— 它 `cxx11abiTRUE`（和 torch 一致）、
含 sm_120 cubin，唯一缺 `libcudart.so.13`，由 PyPI 的 `nvidia-cuda-runtime<=13.2` 补上
（驱动 595 的 `driver_max_cuda=13.2` 支持 CUDA 13；torch 自己仍是 cu128，**两个运行时共存**）。
两者都已写进 `pyproject.toml`，`uv sync` 会自动装；`LD_LIBRARY_PATH` 在 `/workspace/.env`。

```bash
python scripts/check_flash_attn_backward.py     # ★ 换轮子/换机器后必跑，退出码 0 才算可用
```

实测：反向与 fp32 参考对到 4–5 位有效数字；真实 RL 里 `grad_norm` 0.0147/0.0224
（M7 整跑是 0.011–0.06，同区间）；`update_actor` **16.78 s vs sdpa 26.01 s ⇒ 1.55×**。
⇒ 判据不过就显式传 `--attn-implementation sdpa`（正确但慢约 1.55×）。

⚠️ 官方 `cu12torch2.8` 那个别用：`ImportError: undefined symbol _ZNK3c106SymInt6sym_neERKS0_`
（torch 2.8 编的接不上 torch 2.9，而 torch 2.9 由 vllm 0.12.0 钉死）。

### 2.3 Claude 的项目记忆已经进仓库了

2026-08-13 起放在 **`.claude/memory/`**（原来在 `~/.claude/projects/<slug>/memory/`，
不进 git、换机器带不走 —— 那是交接文档记过的一条搬家缺口）。
原位置留了软链指回仓库，读写照常。

⚠️ **新机器 clone 完要把软链重建一次**，做法见 `.claude/memory/README.md`。

### 2.4 git 推送凭据

旧机器靠 VSCode 的 GitHub 登录，换机器就没了。**新服务器上建议一劳永逸配 SSH key 或
装 gh CLI。**

---

## 3 · 数据再生链（`data/batches/`、`data/sft/`、`data/rl/` 都在 .gitignore 里）

进版本管理的只有**生成脚本**和**源 xlsx**。数据不是 clone 下来的，是跑出来的：

```bash
python scripts/make_test_external_data.py
python scripts/ingest_external.py
python -m syncopate cases generate --spec configs/buckets/v11.yaml --out data/batches/v11
python scripts/set_tool_menus.py --batch data/batches/v11 --sft-audit _audit/v8_sft_epoch1.json
python -m syncopate data split --batch data/batches/v11 --out data/splits/v11 \
       --dead-from _audit/v11_base.json
python -m syncopate data build --pool sft --batch data/batches/v11 \
       --out data/sft/v11 --split-dir data/splits/v11 --val-every 6 --model models/Qwen3-4B
python -m syncopate data build --pool rl  --batch data/batches/v11 \
       --out data/rl/v11  --split-dir data/splits/v11 --model models/Qwen3-4B
```

产出：**1370 条 · 17 模板 · EVAL 198 / SFT 434 / RL 738（train 590 + val 148）**

**🆕 v12（当前版本）的链条 —— 必须先建 v11，因为 `--freeze-from` 指向它**：

```bash
python -m syncopate cases generate --spec configs/buckets/v11.yaml --out data/batches/v11
python scripts/set_tool_menus.py --batch data/batches/v11 --sft-audit _audit/v8_sft_epoch1.json
python -m syncopate cases generate --spec configs/buckets/v12.yaml --out data/batches/v12
python scripts/set_tool_menus.py --batch data/batches/v12 --sft-audit _audit/v8_sft_epoch1.json \
       --freeze-from data/batches/v11                      # ★ 少了它会动 1030 条存量 case
python -m syncopate data split --batch data/batches/v12 --out data/splits/v12   # 不传 --dead-from ⇒ difficulty_proxy
python -m syncopate data build --pool sft --batch data/batches/v12 --out data/sft/v12 \
       --split-dir data/splits/v12 --val-every 6 --model models/Qwen3-4B
python -m syncopate data build --pool rl  --batch data/batches/v12 --out data/rl/v12 \
       --split-dir data/splits/v12 --model models/Qwen3-4B
```

产出：**1550 条 · 19 模板 · EVAL 254 / SFT 477 / RL 819（train 655 + val 164）**
✅ **2026-08-17 在一台干净机器上实测逐字节可复现**：重建出来的 `data/splits/v12`
四个文件与 git 里的 **SHA-256 完全一致**。

✅ **2026-08-13 实测逐字节可复现**：重跑出来的三桶切分和 git 里的 SHA-256 完全一致。

⚠️ **两个会静默跑错的默认值**：
- 子命令是 **`cases generate`**，不是 `data generate`；
- **`set_tool_menus.py --batch` 默认 `data/batches/v8`**，不显式传就改错批次
  （和 `launch_rl --train-file` 默认指向 v3 同一个形状）。
- `set_tool_menus` 必须在 generate 之后、split 之前 —— 它改 `tool_menu`，而 split
  的内容级去重按最终 prompt 算。

⚠️ `_audit/v8_sft_epoch1.json` 已进版本管理（`set_tool_menus` 要读它当干扰项），别删。

---

## 4 · 训练入口与调好的默认值

**入口只有两个，不许再写第三个**（临时脚本会静默地和主路径长出差异，这个项目栽过两次）：

```
SFT  →  python -m syncopate.train.sft
RL   →  python -m syncopate.train.launch_rl
```

### 4.1 SFT

```bash
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v11/train.parquet --val-file data/sft/v11/val.parquet \
  --out checkpoints/sft/v11 --epochs 2 --batch-size 1 --grad-accum 4 \
  --lr 1e-4 --warmup-ratio 0.1 --lora-rank 32 --max-length 6144
```

实测：单卡 **412 s/epoch**，显存峰值 12.0 GB（稀疏投影已上线，只对被监督的位置做
lm_head+CE —— 本项目监督占比只有 3.8–4.9%）。

⚠️ **别改 `--batch-size`**：实测 bs=1 反而最快，而且改了要同步改 `--grad-accum`。
⚠️ v11 最长序列 5806 token < 6144，不会截断。**换数据版本要重新量。**

### 4.2 RL（★ 2026-08-13 调优后，默认值已是最优的那套）

```bash
python -m syncopate.train.launch_rl --mode one_step_off \
  --model <merge 后的 SFT 模型> \
  --train-file data/rl/v11/train.parquet --val-file data/rl/v11/val.parquet \
  --save-path checkpoints/grpo/<name> --experiment <name> --lora-rank 32 \
  --steps 150 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 \
  --micro-batch-size 1 --max-num-seqs 64 --object-store-gb 2 \
  --max-prompt-length 3584 --max-response-length 1536 --save-freq 25 --latency-scale 0.01
```

★ **RL 起点必须是 merge 后的模型**（`train/merge_adapter.py`）：`launch_rl` 没有加载
adapter 的入口，而且 verl 用 LoRA 时 reference = 关掉 adapter = **基座**，
合并之后 reference 才等于 SFT。

**已调优的默认值（全部有实测支撑）**：

| 参数 | 值 | 依据 |
|---|---|---|
| `--fsdp-size` | **1**（DDP，不切分） | 3 卡 FULL_SHARD 1182 s/步 vs 1 卡不切分 198 s ⇒ **慢 6 倍**。改 DDP 后 **3.00× 完美扩展** |
| `--dynamic-bsz` | **False** | 实测慢 **2.2×**（垫片下打包让注意力退化成 O(总长²)）。装真 flash-attn 后要重测 |
| `--trainer-gpus / --rollout-gpus` | 3 / 1 | gen 只占 12%，rollout 不是瓶颈 |
| `--rollout-gpu-util` | 0.75 | 0.85 会让第一次权重同步 OOM（bucket 也在 rollout 卡上） |
| `--attention-backend` | TRITON_ATTN | vLLM 自带 FA2 的 sm_120 PTX 比驱动新，编不了 |
| `--bypass-mode` | **False**（decoupled） | **只有这个模式产出 ESS 等 `rollout_corr/*` 指标**，停止条件 P6 依赖它 |
| `NCCL_CUMEM_ENABLE=0` | 多卡时自动设 | 见 §5 |

**调优后实测（one_step_off，3 卡 DDP + 1 卡 rollout，稳态）**：

```
step 49.5 / 50.7 s   ← 重复性很好
  update_actor    30.0 s   61%
  update_weights  13.3 s   27%   ← ★ 下一个目标，见下
  ref              6.0 s   12%
  gen              4.1 s    8%   ← 异步把它藏掉了（step1 是 89.6 s）
```

⚠️ **`update_weights` 13.3 s 是未解之谜**：LoRA 只有 132 MB，按 6.4 GB/s 该是毫秒级。
时间不在传输上。它占 27% 且随训练卡数上升，是异步方案的上限所在。

⚠️ **rollout 卡的显存是反直觉点**：KV 池 110,496 token 听着大，实测**峰值只用 16.7%、
零 preemption**，排队是 `max_num_seqs` 卡的不是显存卡的。**给 rollout 卡加显存换不来速度。**

### 4.3 `--mode` 的三个值不是一个开关，是三套 trainer

```
colocate      verl.trainer.main_ppo                      rollout 和 train 同卡（单卡时代唯一选择）
one_step_off  experimental/one_step_off_policy           分卡、落后一步   ✅ 验证充分
fully_async   experimental/fully_async_policy            分卡、两个独立池 ✅ 2026-08-14 打通
```

⚠️ **`fully_async` 的数据流模型不同**：rollouter 按 `gen_batch_size=1` 连续产样本，
trainer 攒够 `ppo_mini_batch_size × require_batches` 就训一步。
⇒ `data.train_batch_size` **必须是 0**（有硬断言），`rollout.total_rollout_steps` 才是收工点。

⚠️⚠️ **两个键在两种模式下语义不同，别照搬**：

| 键 | one_step_off | fully_async |
|---|---|---|
| `train_batch_size` | 一步取几条 | **强制 0**（硬断言） |
| `--save-freq` | 每 N 个 **global step** | 每 N 个 **param_version**（= N×`sync-every` 个 step） |

⇒ `--save-freq 25` + `--sync-every 4` ⇒ **每 100 步才存一次**，150 步只出 2 个 ckpt
（中途 1 + 收尾强制 1，`fully_async_trainer.py:410`）。
⇒ ckpt 目录按 **param_version** 命名：跑 150 步看到的是 `global_step_37`，不是 `global_step_150`。

#### ✅ `fully_async` 打通（2026-08-14）—— 翻过的四堵墙

| 墙 | 性质 | 修法 / 判据 |
|---|---|---|
| `detach_utils.py:153` `None - None` | **我们的** AgentLoop 把 `TokenOutput.extra_fields` 整个丢了 | `verl_agent_loop.py` 照 `tool_agent_loop.py:248-254` 聚合（首轮收 min、后续抬 max）；判据 `[agent-loop] 策略版本字段 ✓` |
| `save_model_to_cpu` 断言要 DTensor | 上游假定 trainer 侧在分片，而本机**必须** DDP | `verl_patches` P2，**只在无 DTensor 时接管**；4 条数值往返测试 |
| 补丁打在 driver、断言在 worker | 作用域 | `ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=syncopate.train.verl_patches.setup_worker` |
| 第 8 步权重同步 OOM（差 0.09 GB） | bucket 住在 rollout 卡上 | `--weight-sync-bucket-mb 1024 --rollout-gpu-util 0.70` |

★ 那个策略版本字段**正好就是异步研究要量的东西**：`max-min` = 轨迹横跨几个版本。
实测 `stale_trajectory_processed 576/7200 = 8.0%`、`partial_ratio 0.0`。

⚠️ **动态分池在 fully_async 下没接上，但不是模式限制**：
`fully_async_rollouter.py:464` **确实调**了 `create_rl_sampler`（import 还写在函数体内，
最适合 monkeypatch）；没生效是因为 rollouter 跑在**另一个 worker 进程**里，
而 `setup_worker()` 目前只装 fsdp 那个补丁。⇒ 把 sampler patch 也装进去即可。
（旧注释「它不调 create_rl_sampler」**是错的**，2026-08-14 读码更正。）

#### 合并 fully_async 的 ckpt 要先补 `huggingface/`

`fully_async` 存盘不写 `actor/huggingface/`，而 `verl.model_merger` 要它。
从起点模型拷 config/tokenizer 九件套（**逐字节校验大小**，配额超限时 `cp` 会静默产出 0 字节），
再 `python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <out>`。
产出里同时有完整模型和 `lora_adapter/`。

---

## 5 · 多卡的三个环境变量（都已在 `launch_rl` 里自动处理）

```
NCCL_CUMEM_ENABLE=0        多卡时必须
VLLM_ATTENTION_BACKEND     默认 TRITON_ATTN
RAY_object_store_memory    --object-store-gb 控制
```

**NCCL 那条的完整故事**（值得读，因为推理链错过一次）：

```
症状   transport/shm.cc:590 NCCL WARN Cuda failure 217 'peer access is not supported'
       ← FSDP 初始化第一次参数广播

原猜想 P2P 全关 ⇒ 设 NCCL_P2P_DISABLE=1        ❌ **实测完全无效**

实测   ① 裸 torchrun（每进程看得见 4 张卡）：默认就通
       ② 模拟 Ray（每进程只看得见 1 张卡）：
            默认 ❌ / P2P_DISABLE=1 ❌ / SHM_DISABLE=1 ✅ / CUMEM_ENABLE=0 ✅

真根因 **P2P 缺失 × Ray 只给每个 worker 开放一张卡** 两个条件叠加 ——
       进程看不见对端设备，SHM 传输要用的 CUDA IPC 建不起来。
       `NCCL_P2P_DISABLE` 只关 P2P 传输，管不到 SHM 里的 IPC。

选哪个 按带宽定：SHM_DISABLE 2.09 GB/s  vs  CUMEM_ENABLE=0 **6.44 GB/s** ⇒ 选后者
```

---

## 6 · 给 verl 打的补丁（`syncopate/train/verl_patches.py`）

不 fork、不改 site-packages。目前一条：

**`OneStepOffRayTrainer` 漏调 `_init_dump_executor()`**（`RayPPOTrainer` 和两个
`FullyAsync*` 都调了，只有它漏）。触发条件是设了 `trainer.rollout_data_dir` ——
上游大概没开着 dump 跑过异步。

⚠️ **不能用"关掉 dump"绕过**：那份 dump 是分布漂移的一半（dump = 训练到的，
`dispatched.jsonl` = 下发过的），关掉等于为了跑通把要测的东西关了。

★ 补丁做成子类 + 幂等，上游哪天修了会自动不生效。
**判据永远是日志**：`[verl-patch] ...` / `[pool] 动态分池启用` / `[rl] 模式=...`
**这三行没出现就是没接上。**

---

## 7 · 其它运行期须知

- **评测引擎是 vLLM**：198 条 × 8 采样约 11 分钟（旧 HF 引擎要 84 分钟）。
  ⚠️ 旧 HF 引擎的审计**不能和新审计逐 case 配对**（引擎决定采样内核），
  要配对比较必须同引擎重跑。
- **wandb 默认开**，要关得显式 `--no-wandb`。⚠️ 本机**没有 wandb 登录**，
  当前用 `--wandb-mode offline`，之后 `wandb sync <dir>` 可补传。
- **磁盘**：verl 0.8 的 FSDP checkpoint manager 存**全量** state_dict，
  LoRA 下一个 ckpt 8.5 GB，其中 8.22 GB 和基座逐字节相同。
  `scripts/prune_rl_ckpts.py` 只留 LoRA 键（默认跳过最后一个，保留断点续跑）。
- **`pkill -f <模式>` 会自匹配**（执行 pkill 的 shell 自己也含该模式）—— 犯过三次。
- **最可信的不是 commit message，是 `outputs/<日期>/<时刻>/.hydra/overrides.yaml`**
  —— 那是那次实际跑的全部配置。查"上次到底怎么跑通的"就看它。
