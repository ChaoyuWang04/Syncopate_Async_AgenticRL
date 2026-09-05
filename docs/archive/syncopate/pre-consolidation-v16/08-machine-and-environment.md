# Syncopate · 08 — 机器、环境与运行配置

> 📦 **历史机器与环境快照，不代表当前启动方法。** 现行说明见
> [docs/syncopate/05-COMPUTE.md](../../../syncopate/05-COMPUTE.md)。

> 2026-08-13 在 4×RTX 5090（RunPod）上从零搭起来并逐条实跑验证过。
> **这份是"怎么把它跑起来"，方法论和研究结论在 `00-START.md`。**

---

## 1 · 硬件画像（实测）

> ⚠️ 2026-08-27 换机（vast.ai）后全量重画像，旧机（2×EPYC 9V74 2+2 跨 socket）数字去 git log。

```
4 × RTX 5090  32607 MiB / 575W / sm_120   驱动 595.71.05  CUDA 12.8  torch 2.9.0+cu128
CPU   1× EPYC 9B14（96 核，192 线程，**4 NUMA/NPS4**）   RAM 566 GB
拓扑  单 socket：GPU0/1 同桥（PHB）@node3 · GPU2@node2 · GPU3@node0，跨 NUMA 走 SYS
PCIe  **Gen5 x16**（32 GT/s；空闲时报 Gen1 属正常）
盘    /workspace = 本地 300G（**持久卷**，workspace_is_volume=true，权限位生效）
      / 只有 **16G overlay** ⇒ 见 §2.0，缓存必须重定向
```

🔴 **P2P 全关**（`can_device_access_peer` 4×4 全 0）。GeForce 从 4090 起就被驱动关掉了
PCIe P2P，5090 同样 —— **这是所有 4×5090 机器的常态，不是这台坏了。**
⇒ 卡间通信一律经主机内存中转（NCCL 实测走 `SHM/direct/direct`）。

**卡间 all-reduce bus bandwidth**（`scripts/infra/probe_allreduce_bw.py`，
`NCCL_CUMEM_ENABLE=0` / 256MB，2026-08-27 实测，与 `infra_exp/README.md` §6 同口径）：

```
同桥对 (0,1) 16.3 GB/s    跨 NUMA 对 14.4–15.1    四卡（DDP 走这条）17.9
⇒ 比上一台（四卡 25.6）**低 30%**：同 socket 无 UPI 跳，但四卡挤一份内存子系统，净输
⇒ 对间差距只剩 ~8% ⇒「摆哪张卡」在这台机器上基本不重要；NUMA 绑定仍无效
⇒ 换算负载：DDP 梯度 260MB ≈ 15 ms/步 = 一步(9.7s)的 0.15%，仍可忽略
```

✅ 满载降频已测（08-27）：4×575W 稳态频率 −0.5%、单卡 TFLOPS −0.9%（221→219）、69°C
⇒ 多卡对照的非通信损失可忽略。⬜ 还没测：主机内存带宽。

## 1.1 · PostgreSQL 的落盘（训练机 root 路数；本机用户态与 K 线中间件见 §1.2）

`/workspace` 是 XFS，`chmod 700` 生效 ⇒ **PGDATA 直接放持久卷**：

```
PGDATA=/workspace/pgdata/16/syncopate      # 已写进 /workspace/.env
PG 二进制/库   /workspace/tools/postgres/root
deb 离线包     /workspace/tools/postgres/debs   ← 断网也能重装
schema/迁移    仓库 syncopate/runtime/migrations/versions/（Alembic 迁移链 = 唯一真相，K2 2026-09-02；
               schema.snapshot.txt 是生成的只读快照，pg_bootstrap 末尾 --check 判漂移）
```

```bash
bash scripts/serving/pg_bootstrap.sh          # 幂等：建用户 → initdb → 起服务 → 建库 → 应用 schema
bash scripts/serving/pg_bootstrap.sh --reset  # 推倒重来
```

⚠️ 二进制丢了（容器换了镜像）时的重装，不需要联网：
```bash
mkdir -p /workspace/tools/postgres/root
for d in /workspace/tools/postgres/debs/*.deb; do dpkg -x "$d" /workspace/tools/postgres/root/; done
```
⚠️ `dpkg -x` **不跑 maintainer 脚本** ⇒ 不会建 `postgres` 用户、`libpq.so.5` 也不进
ldconfig 路径。`pg_bootstrap.sh` 已经把这两件都处理了（`LD_LIBRARY_PATH` 也在 `/workspace/.env`）。

⚠️ **容量账**：fully_async 一个 ckpt **27 GB**（3 个 rank 的全量 state_dict，
其中 97% 是和基座逐字节相同的冻结权重）。跑完用 `syncopate/train/prune_ckpts.py` 瘦身（只留 LoRA，约 250 MB）。
⚠️⚠️ **`--save-freq 999` 挡不住收尾那次保存** —— 计时/探针类短跑跑完就删
`checkpoints/grpo/<exp>/global_step_*`（`dispatched.jsonl` 和 `rollout_dumps` 要留）。

---

## 1.2 · Serving 生产栈的落盘与重建（K 线，2026-09-02；搬家时照这节）

K 线（27）给 serving 加了三样中间件：**Redis 8**（Celery broker）、**Celery 5.6**（worker 投递）、
**Alembic**（schema 唯一真相）。PostgreSQL 不换。纪律与 §1.1 相同：**PG 里的 outbox/agent_runs
是事实来源，Redis 是派生产物**（丢了重投即可，`sweeper` 与 `requeue_outbox` 兜底）。

### 1.2.1 两种机器的落盘位置

| | 训练机（root，/workspace） | 本机 5090 工作站（**无 sudo**，conda 用户态） |
|---|---|---|
| PG 二进制 | `/workspace/tools/postgres/root/...`（deb 解包） | conda env `syncopate-infra`（conda-forge `postgresql=16`，实测 16.15） |
| PGDATA | `/workspace/pgdata/16/syncopate` | `$HOME/.local/share/syncopate/pgdata/16`（socket 在 `/tmp`） |
| Redis 二进制 | 未装（训练机上 K 线未跑过，见下"搬家步骤"） | 同一 conda env（`redis-server`，实测 8.10） |
| Redis 数据/配置 | — | `$HOME/.local/share/syncopate/redis/`（redis.conf · AOF · RDB · log · pid） |
| 判别 | `pg_bootstrap.sh` 按 `id -u` 自动分支（`PG_USER_MODE`） | 同左；`PG_HOME/PG_SHARE/PG_LIB/REDIS_HOME` 都可用环境变量覆盖 |

```bash
# 本机用户态装法（一次性；训练机用 deb 解包那套，§1.1）
conda create -n syncopate-infra -c conda-forge postgresql=16 redis-server
export PG_HOME=$HOME/Downloads/ENTER/envs/syncopate-infra PG_SHARE=$PG_HOME/share PG_LIB=$PG_HOME/lib
bash scripts/serving/pg_bootstrap.sh          # 幂等：initdb → 起 → 建库 → alembic upgrade head → 快照核对 ✅
bash scripts/serving/redis_bootstrap.sh       # 幂等：写 redis.conf → 起 → 判据行 [redis-config] appendonly=yes … ✅
```

### 1.2.2 Python 依赖

`runtime` extra 已含 `alembic sqlalchemy psycopg[binary] celery[redis] redis`（pyproject，uv.lock 已锁）。
训练机照 §2 `uv sync --frozen --all-extras`；**本机**（不装 train extra，避免拆 torch/vllm）：

```bash
uv sync --inexact --extra runtime --extra dev     # ⛔ 不带 --inexact 会把手装的 torch/vllm 卸掉（守则⑧同族）
```

### 1.2.3 环境变量（K 线新增的都在这；默认值就是本机开发值）

| 变量 | 默认 | 读者 | 说明 |
|---|---|---|---|
| `SYNCOPATE_PG_DSN` | `postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate` | db/alembic/脚本 | Alembic 也读它（`alembic.ini` 的 `syncopate.dsn` 优先） |
| `SYNCOPATE_REDIS_URL` | `redis://:syncopate-dev@127.0.0.1:6379/0` | celery_app | ⛔ 生产密码从 secret 注入，**永不进日志/URL 打印**；db0 broker · db1 限流 · db2 信号量 · db3 缓存 |
| `REDIS_PASS` / `REDIS_PORT` / `REDIS_DIR` | `syncopate-dev` / 6379 / `~/.local/share/syncopate/redis` | redis_bootstrap.sh | 与上一行必须一致 |
| `SYNCOPATE_WORKER_ORG_ID` | — | celery worker | ★ 常驻 worker 必设 `org_demo`（08-20 抢走测试租户 run 那课） |
| `SYNCOPATE_API_DB_POOL` / `_WORKER_DB_POOL` / `_DISPATCHER_DB_POOL` / `_SWEEPER_DB_POOL` | 10 / 4（celery 子进程；轮询 worker 10）/ 4 / 3 | 各进程 | prefork N 进程 × 池 ≤ PG `max_connections`（28 P-07） |
| `SYNCOPATE_VISIBILITY_TIMEOUT_S` | max(2×lease, 900) | celery_app | 必须 > 最长任务，否则 Redis 伪 ack 重投（28 C-02） |
| `SYNCOPATE_LEASE_TTL_S` | 60 | worker/sweeper | 心跳每 TTL/3 续一次（判据行 `[lease-heartbeat]`） |
| `SYNCOPATE_SWEEP_INTERVAL_S` / `_RECONCILE_EVERY` / `_STUCK_QUEUED_S` / `_WAITING_TOO_LONG_S` / `_OUTBOX_ARCHIVE_DAYS` | 10s / 每 30 轮 / 300s / 21600s / 30 天 | sweeper | 五类扫描的节拍与阈值（K8） |
| `SYNCOPATE_ORG_DAILY_TOKENS` / `_ORG_DAILY_COST_MICROS` | 2,000,000 / 10,000,000（=$10） | api/budget | 未建 `org_budgets` 行时的默认两档（K9） |
| `SYNCOPATE_RELEASE_HALTED=1` | 未设 | release/action_gate | 总开关：写动作全停（发布五能力①） |
| `SYNCOPATE_DISABLED_TOOLS=a,b` | 未设 | tool_governance | 禁单个工具，读工具照常（发布五能力②） |
| `SYNCOPATE_DECIDER_URL` / `_MODEL` / `_TOKENIZER` / `_TIMEOUT` / `SYNCOPATE_CONTRACT` | 老变量 | decider | K 线未改；共用件归 26 线 |

起服务的完整命令序列在 **09 §0**（redis_bootstrap → dispatcher → sweeper → celery worker → uvicorn）。

### 1.2.4 恢复数据库（三种情形）

| 情形 | 做法 | 已验 |
|---|---|---|
| 库结构丢了（换机/重装） | `bash scripts/serving/pg_bootstrap.sh` 从迁移链 0001→0008 重建，末尾 `schema_snapshot.py --check` 必须 ✅ | `scripts/serving/dr_drill.sh` 09-02 RTO 1.5s |
| 迁移链改了 | `alembic upgrade head` → `python -m syncopate.runtime.schema_snapshot --write` → 提交快照；`--check` 红 = 有人手改库，**找人不改快照** | 30 runbook 05 |
| 业务数据要保（灰测放真人后） | dev 期数据库是派生产物**无备份**（30 D3 挂账）；到时 `pg_dump -Fc` 周期 + 演练恢复 | ⬜ |

Redis 不需要恢复：AOF 在 `REDIS_DIR`，丢了就 `redis_bootstrap.sh --reset`，
在途消息由 sweeper 的 stuck_queued 分支 + `requeue_outbox` 从 PG 重投（30 runbook 01/02）。

### 1.2.5 K 线新增的仓库内产物（哪些进 git、哪些是派生）

| 路径 | 性质 |
|---|---|
| `syncopate/runtime/migrations/versions/0001…0008` + `schema.snapshot.txt` | **进 git**，唯一真相与只读快照 |
| `scripts/serving/pg_bootstrap.sh` `redis_bootstrap.sh` `schema_snapshot.py` `dr_drill.sh` `runbook_queries.py` `slo_readout.py` | 进 git |
| `_audit/serving_k11/`（SLO 基线、演练日志） | 进 git（证据本体） |
| `data/feedback_exports/`（K10 回流导出，考卷 v4 题形） | 派生产物，26 线吸入时再定去向 |
| `frontend/dist` | 派生产物，`npm ci && npm run build`（本机无 node，三处改名与 👍👎 未 build） |

### 1.2.6 搬家步骤（在 §2 之外多做的三步）

```bash
bash scripts/serving/redis_bootstrap.sh                                    # §2 之后：起 Redis
python -m pytest tests/runtime -q                                  # 372 passed · 10 skipped（skip 的是无 node/无端点那几条）
bash scripts/serving/dr_drill.sh                                           # 干净目录重建演练，末行 RTO；新暴露前提=0 才算搬完
```

⚠️ 训练机上 K 线**从未跑过**：Redis 要按 §1.1 的 deb 解包路数装（`REDIS_HOME` 指过去），
Celery prefork 与四卡舰队的连接数账（28 P-07/S-05）要重算，压测 goodput 重测（27 §14 挂账）。

---

## 2 · 从零搭环境（新机器按顺序做）

```bash
set -a; . /workspace/.env; set +a                  # ★ 先加载持久化重定向，见 2.0
uv sync --frozen --all-extras                      # ← 锁文件是救命绳；裸 uv sync 会卸掉 extras（00-START 守则⑧）
hf download Qwen/Qwen3-4B   --local-dir models/Qwen3-4B     # 7.6G
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B   # 1.4G，31 个测试要它
python scripts/infra/check_flash_attn_backward.py        # ★ 退出码 0 才算环境可用（反向恒 0 比 nan 更毒）
python -m syncopate.domains.adcampaign.ingest_external                  # ★ 派生数据，不跑的话 27 个测试红
bash scripts/serving/pg_bootstrap.sh                       # ★ 不起 PG 的话 45 条 runtime 测试 skip；末尾 alembic 链 + 快照核对
bash scripts/serving/redis_bootstrap.sh                    # ★ K3 起：Celery 集成测试要真 Redis（§1.2）
python -m syncopate.runtime.ingest_corpus                    # ★ RAG 语料入 PG，不跑的话 3 条 retrieval 测试红
# reference/ 870M 只能手动 scp —— 版权所有（深圳途明智启科技），永不进 git
python -m pytest -q                                # 应为全 passed, 0 skipped（2026-08-27 实测 694 passed·1 xfailed）
```

⚠️ **五个最容易漏的**（漏了的表现都不像缺步骤，很容易误判成别的问题）：

| 步骤 | 漏了会怎样 |
|---|---|
| `--all-extras`（或三个 `--extra` 全给） | `asyncpg`/`fastapi` 缺失 ⇒ tests/runtime 4 个模块**收集阶段**就 ImportError |
| `ingest_external.py` | `data/external/ingested.json` 是 gitignore 的派生产物 ⇒ **27 个测试红**，签名像"数据过期" |
| `pg_bootstrap.sh` | 45 条 runtime 测试静默 **skip** —— 它们的 docstring 自己写着**「不要把跳过当通过」** |
| `ingest_corpus.py` | 政策语料在 PG 不在文件 ⇒ **3 条 retrieval 测试红**（StopIteration/KeyError，签名像检索逻辑坏了）——2026-08-27 干净机器重建暴露 |
| `set -a; . /workspace/.env` | 缓存全落 16 G overlay，Ray 一溢写就爆 |

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

🟢 `PGDATA=/workspace/pgdata/16/syncopate` 也在这里 —— `/workspace` 是 XFS、权限位生效，
PostgreSQL 的 0700 要求满足得了，**数据库不是"重启即丢的派生产物"**（见 §1.1）。

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
所以这个包必须在。⚠️ **纯 Python 的 shim 不行** —— 它满足 import 但不满足契约。

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
两者都已写进 `pyproject.toml`，`uv sync` 会自动装；`LD_LIBRARY_PATH` 在 `/workspace/.env`
（⚠️ 2026-08-27 实测：`libcudart.so.13` 装在 `.venv/…/site-packages/nvidia/cu13/lib`，
不是望文生义的 `nvidia/cuda_runtime/lib` —— 路径写错的症状是 import flash_attn 直接 ImportError）。

```bash
python scripts/infra/check_flash_attn_backward.py     # ★ 换轮子/换机器后必跑，退出码 0 才算可用
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

⚠️ **凭据不进容器镜像，容器一重建就没。** 因此 SSH key 放在持久卷上：

```
/workspace/tools/ssh/id_ed25519      私钥（~/.ssh/id_ed25519 是软链）
/workspace/tools/ssh/config          指定 IdentityFile，~/.ssh/config 是软链
```

⇒ 公钥加到 GitHub 后，remote 用 `git@github.com:...` 即可，不再需要 token。
⚠️ git 的 `user.name` / `user.email` 也会丢，`git config` 在仓库级设一次。

---

## 3 · 数据再生链（`data/batches/`、`data/sft/`、`data/rl/` 都在 .gitignore 里）

进版本管理的只有**生成脚本**和**源 xlsx**。数据不是 clone 下来的，是跑出来的：

```bash
python -m syncopate.domains.adcampaign.generate_test_external_data
python -m syncopate.domains.adcampaign.ingest_external
python -m syncopate cases generate --spec configs/buckets/v11.yaml --out data/batches/v11
python -m syncopate.pipeline.tool_menus --batch data/batches/v11 --sft-audit _audit/v8_sft_epoch1.json
python -m syncopate data split --batch data/batches/v11 --out data/splits/v11 \
       --dead-from _audit/v11_base.json
python -m syncopate data build --pool sft --batch data/batches/v11 \
       --out data/sft/v11 --split-dir data/splits/v11 --val-every 6 --model models/Qwen3-4B
python -m syncopate data build --pool rl  --batch data/batches/v11 \
       --out data/rl/v11  --split-dir data/splits/v11 --model models/Qwen3-4B
```

产出：**1370 条 · 17 模板 · EVAL 198 / SFT 434 / RL 738（train 590 + val 148）**

**v12 的链条 —— 必须先建 v11，因为 `--freeze-from` 指向它**：

```bash
python -m syncopate cases generate --spec configs/buckets/v11.yaml --out data/batches/v11
python -m syncopate.pipeline.tool_menus --batch data/batches/v11 --sft-audit _audit/v8_sft_epoch1.json
python -m syncopate cases generate --spec configs/buckets/v12.yaml --out data/batches/v12
python -m syncopate.pipeline.tool_menus --batch data/batches/v12 --sft-audit _audit/v8_sft_epoch1.json \
       --freeze-from data/batches/v11                      # ★ 少了它会动 1030 条存量 case
python -m syncopate data split --batch data/batches/v12 --out data/splits/v12   # 不传 --dead-from ⇒ difficulty_proxy
python -m syncopate data build --pool sft --batch data/batches/v12 --out data/sft/v12 \
       --split-dir data/splits/v12 --val-every 6 --model models/Qwen3-4B
python -m syncopate data build --pool rl  --batch data/batches/v12 --out data/rl/v12 \
       --split-dir data/splits/v12 --model models/Qwen3-4B
```

产出：**1550 条 · 19 模板 · EVAL 254 / SFT 477 / RL 819（train 655 + val 164）**

**🆕 v13（当前版本）—— 同样要先建 v12（`--freeze-from` 指向它）**：

```bash
python -m syncopate cases generate --spec configs/buckets/v13.yaml --out data/batches/v13
python -m syncopate.pipeline.tool_menus --batch data/batches/v13 --sft-audit _audit/v8_sft_epoch1.json \
       --freeze-from data/batches/v12
python -m syncopate data split --batch data/batches/v13 --out data/splits/v13
python -m syncopate data build --pool sft --batch data/batches/v13 --out data/sft/v13 \
       --split-dir data/splits/v13 --val-every 6 --model models/Qwen3-4B
python -m syncopate data build --pool rl  --batch data/batches/v13 --out data/rl/v13 \
       --split-dir data/splits/v13 --model models/Qwen3-4B
python -m syncopate.pipeline.data_gates --batch data/batches/v13     # ★ 大版本重建前必跑
```

产出：**1670 条 · 21 模板 · 106 格 · 30 工具 · 35 cap**
⚠️ **三桶大小一律以 `data/splits/v13/*_cases.json` 为准**（实测 EVAL **343**）——
文档里出现过 278 那个数，是错的。
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

### 4.0 ★★ 固定管线的两条纪律（2026-08-19 立，**判据已进仓库**）

> `python -m syncopate.pipeline.invariants --only contract`

**① 入口只有这两个 —— 不许直接调训练框架。**
`launch_rl` 上挂着契约默认值、起手断言（`--model` 目录带 `lora_adapter/` 就报错）、
以及守卫的挂载点。**绕过去，这些全都不生效。**

**② 契约参数一律不许在脚本里传** —— 长度预算与采样参数的唯一来源是
`syncopate/train/rollout_budget.py`，`launch_rl` / `eval_local` 的默认值从那里取，
**不传就是对的**。

```
--max-prompt-length   --max-response-length
--temperature         --top-p                --top-k
```

★★ **为什么判据是"一律不许传"，而不是"传的值要对"** —— 这是这条纪律的全部要点：

2026-08-18 的实况是，15 个启动脚本**全都**走 `launch_rl`（纪律①一直守着），
但**每个脚本各自抄了一份参数**，于是抄着抄着漂成了两套：

```
11 个   5120/2048   ✅ 当时是对的
 4 个   3584/1536   🔴 停在旧值 —— 而 2026-08-19 02:55 还真有一跑用了它
```

⇒ **那 11 个当时是对的，而它们才是下一次漂移的来源。**
  旧判据只比"值对不对"，所以一路绿灯 —— 问题不是值错，是**存在一份副本**。
⇒ 所以判据写在「**只该有一份**」上，不写在「这一份的值对不对」上（守则①）。

⚠️ **逃生口**：确实要做契约参数的 A/B，在**同行或上一行**写

```bash
# CONTRACT-OVERRIDE: <为什么这次必须覆盖>
```

声明过的不算违反，但会被打出来**留痕**。
留这个口子是刻意的：没有它，合法的对照实验会天天报红，
而**假警报会训练人忽略这条判据 —— 那比没有判据更糟**（守则③）。
⚠️ 标记只在同行或上一行生效 —— 否则在文件顶上写一句就把整个文件豁免了。

### 4.0.1 已清理掉的（2026-08-19）

```
scripts/run_v10_eval.sh   ⚰️ 删除 —— v10 时代，cd 到 /home/samwang/…（本机不存在）
15 个启动脚本             剥掉显式传的契约参数（其余一字未动，bash -n 全过）
budget 组的第③条         并进新的 contract 组（避免"同一件事两份实现"）
budget 组的第②条         🟡→🔴：那句「这是修复**之前**的跑，属预期」是**无条件**的，
                          它刚放过了一跑修复**之后**的跑，且以后每次都会放过
```

### 4.1 SFT

```bash
python -m syncopate.train.sft --model models/Qwen3-4B \
  --train-file data/sft/v11/train.parquet --val-file data/sft/v11/val.parquet \
  --out checkpoints/sft/v11 --epochs 2 --batch-size 1 --grad-accum 4 \
  --lr 1e-4 --warmup-ratio 0.1 --lora-rank 32
```

实测：单卡 **412 s/epoch**，显存峰值 12.0 GB（稀疏投影已上线，只对被监督的位置做
lm_head+CE —— 本项目监督占比只有 3.8–4.9%）。

⚠️ **别改 `--batch-size`**：实测 bs=1 反而最快，而且改了要同步改 `--grad-accum`。
⚠️ 2026-08-19 起**没有 `--max-length` 这个参数了**：长度上限从 `rollout_budget.py`
推（prompt 5120 + response 2048 = 7168），超长样本**硬报错**而不是静默截断。
此前是 `--max-length` 默认 4096 + 静默切片 —— v13 数据 92.6% 超 4096，
实跑靠手传 6656 侥幸躲过，而本文旧版示范命令写的 6144 会无声截掉 46 条。
判据：`check_pipeline_invariants --only contract`（源码 + 数据两头都查）。

### 4.0.1 🔴 nsys 不要包住 RL 长跑（2026-08-17 用一次报废的跑换来的）

infra 想搭车做 A5/E01 的一步拆解，我把整跑包进了
`nsys profile --delay 900 --duration 180`。**两件事同时出错**：

```
① 中间文件 30 分钟涨到 74 GB，且以 426 MB/分 继续涨
   （--duration 180 没有限制住 —— 它 trace 的是 Ray 拉起的一整片子进程）
   算总账：剩余 85 步还要吃 72 GB + checkpoint 108 GB = 180 GB > 剩余 134 GB
   ⇒ **必然撑爆，正是 M7 丢最终 ckpt 的那个形状**

② `nsys stop --session=…` **会把目标进程一起杀掉**
   （日志：`The target application terminated`）——
   我本以为它只停采集。RL 死在第 25 步，checkpoint 一个没存下，
   而那份 74 GB 的 trace 产出是**一个 0 字节的 .nsys-rep**。
```

⇒ **纪律**：
- **nsys 只包短跑**（≤10 步的专门 profile 跑），**绝不包正式训练**。
- 想在长跑里截窗口，用 `torch.profiler`（`--profile-steps`，SFT 侧已经有）——
  它按步数抓、写在我们自己的目录、跑完自动收工。
- ⚠️ **磁盘守卫要和停止条件一起挂**：`logs/rl_guard.log` 那个守卫现在同时看
  ESS / 熵 / **剩余空间 < 40 GB 就停机**。

★ 顺带一条已确认的观察：nsys 的中间文件落在 `$TMPDIR/nvidia`，
而 `/workspace/.env` 把 `TMPDIR` 指到了 `/workspace/tmp` ——
**幸亏指对了**，否则 74 GB 会写进只有 16 G 的 overlay，几分钟就整机挂掉。

---

### 4.1.1 ★ SFT 要不要多卡？—— 调查结论：能，但现在不做

**现状**：`syncopate/train/sft.py` 是**手写的单进程单卡循环** ——
没有 `torch.distributed` / DDP / accelerate，也不走 HF `Trainer` 或 verl。
2 个 epoch **19 分钟**，显存峰值 **12.3 / 32 GB**。

#### 该用什么并行

```
可训练参数 66.1M / 4088.5M (1.62%)   LoRA
显存峰值   12.3 GB / 32 GB           单卡绰绰有余
```

⇒ **只该用数据并行（DDP）**，不需要任何模型内并行。而且本机 **PCIe P2P 全关**，
FSDP/TP 的通信要经主机内存中转 —— RL 侧已经验过是净亏。
LoRA 梯度只有 66M 参数 ≈ 260 MB，一步 all-reduce 约 10 ms，可以忽略。

#### ★★ 不要自己写 DDP —— verl 已经有 SFT trainer，而且三样都齐

```
.venv/…/verl/trainer/sft_trainer.py       单机（torchrun）
.venv/…/verl/trainer/sft_trainer_ray.py   Ray 版
```

| 我们要的 | verl 有没有 |
|---|---|
| 数据并行 | ✅ `engine.fsdp_size`（**设 1 = DDP**，-1 才是全量分片） |
| 序列并行 | ✅ `ulysses_sequence_parallel_size`（我们最长 6372 token，用得上） |
| LoRA | ✅ `model.lora_rank` / `lora_alpha` / `target_modules` |
| 动态 batch | ✅ `use_dynamic_bsz` + `max_token_len_per_gpu`（按 token 配平） |
| **插自己的数据集** | ✅ **`data.custom_cls: {path, name}`** ← 关键 |

#### ★★★ 一个重要的旁证：verl 自己承认了我们当初绕开它的那个问题

`sft_replay.py` 记着不用 `MultiTurnSFTDataset` 的理由：Qwen3 的 chat template
**只给最后一个 assistant 轮加空 `<think>`**，所以「整段渲染」和「增量拼接」
天生逐 token 不相等。

而 verl 自己在 `config/sft_trainer_engine.yaml` 的注释里写着同一件事：

> `MultiTurnSFTDataset` apply_chat_template to each turn separately and concat
> `input_ids` … **which may not equal to** apply_chat_template to whole messages
> at once. For example, **Qwen Thinking series models add `<think></think>` tags
> to last turn** … Set to True to **ignore** input_ids mismatch…

⇒ 它的"解法"是加个 `ignore_input_ids_mismatch` 开关去**忽略**不一致；
我们的解法是**只保留一条代码路径**（SFT 数据由 `run_rollout` 回放 gold 产出，
和 RL 序列同构是构造保证的）。**我们那个判断是对的，现在有上游背书。**

#### 真要做的时候怎么接

**数据构造这条路径一个字不改**，只把训练循环换成 verl 的：

```yaml
data.custom_cls.path: syncopate/train/verl_sft_dataset.py   # 30 行，读我们预分词的 parquet
data.custom_cls.name: SyncopatePretokenizedSFT
engine.fsdp_size: 1          # DDP，不分片
model.lora_rank: 32
```

#### 为什么现在不做（三条）

1. **收益 19 分钟**，而接线要验的不少：collator 字段契约、LoRA adapter 保存格式要和
   `merge_adapter` / `weight_shift` 对得上。
2. **★ loss 归一化口径会变。** 我们现在是 **每条 case 等权**（bs=1，每个 micro-batch
   一条序列，取该序列的 token 均值）；verl 的 `use_dynamic_bsz` 是**按 token 配平**
   ⇒ 长序列权重更高。监督 token 变异系数 0.37，差别不至于翻天，
   **但足以让 ckpt 选型的数（决策位熵 / 有梯度格子）和现有的不可比** —— 而那是选 e1 的依据。
3. **引入新的依赖路径**：现在 SFT 完全不依赖 verl。而我们刚在 RL 侧被 verl 咬过三次
   （`save_model_to_cpu` 断言、日志级别硬编码成 WARN、`create_rl_sampler` 写死）。

#### ★ 触发条件写死，别靠感觉决定

```
SFT 数据 > 2000 条（现在 419）     单卡一遍超过 1 小时，并行才有意义
或 序列 > 8000 token              单卡显存开始紧（现在 12.3/32 GB）
或 要做全参微调而非 LoRA          那时单卡真装不下，必须 FSDP
```

⚠️ 到那时如果自己写 DDP，记住两条：**`grad_accum` 要同步除以卡数**
（否则有效 batch 翻 4 倍、步数砍 4 倍，位移直接掉进"白训"区间）；
**别改 loss 的归一化口径**（`DistributedSampler(drop_last=True)` 保证各卡条数相同时，
DDP 的 rank 平均恰好保持"每条 case 等权"）。

---

### 4.2 RL（★ 2026-08-13 调优后，默认值已是最优的那套）

```bash
python -m syncopate.train.launch_rl --mode one_step_off \
  --model <merge 后的 SFT 模型> \
  --train-file data/rl/v11/train.parquet --val-file data/rl/v11/val.parquet \
  --save-path checkpoints/grpo/<name> --experiment <name> --lora-rank 32 \
  --steps 150 --train-batch-size 6 --rollout-n 8 --ppo-mini-batch-size 6 \
  --micro-batch-size 1 --max-num-seqs 64 --object-store-gb 2 \
  --max-prompt-length 5120 --max-response-length 2048 --save-freq 25 --latency-scale 0.01
```

★ **RL 起点必须是 merge 后的模型**（`train/merge_adapter.py`）：`launch_rl` 没有加载
adapter 的入口，而且 verl 用 LoRA 时 reference = 关掉 adapter = **基座**，
合并之后 reference 才等于 SFT。

**已调优的默认值（全部有实测支撑）**：

| 参数 | 值 | 依据 |
|---|---|---|
| `--fsdp-size` | **1**（DDP，不切分） | LoRA 下每步只同步 260 MB（≈10 ms）；分片要每层 all-gather 全部权重。⚠️ FULL_SHARD/ZeRO-2 稳态对照由队列 A6 补 |
| `--dynamic-bsz` | **False** | ⚠️ **符号由 attention 决定**，当前机器 + FA2 下**未测**。sdpa 下打包会让注意力退化成 O(总长²) |
| `--trainer-gpus / --rollout-gpus` | 3 / 1 | gen 只占 12%，rollout 不是瓶颈 |
| `--max-prompt-length` / `--max-response-length` | **5120 / 2048** | 🆕 2026-08-18：**训练与评测共用一份**（`syncopate/train/rollout_budget.py`）。此前训练 3584/1536、评测硬编码 5120/2048，**两边跑在不同的输入分布上** —— 见 `22 §P0-1`。取宽的那一档：放宽只会让原本被截断的轨迹跑完，不会新增截断 |
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

⚠️ **`update_weights` 13.3 s 是未解之谜**：LoRA 只有 132 MB，按卡间带宽该是毫秒级。
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

✅ **动态分池在 fully_async 下已接上（2026-08-17）** ——
判据行由 `FullyAsyncRollouter` 那个进程打出：`[pool] 动态分池启用：659 条 case`。
修法：把 `install_sampler_patch()` 也装进 `verl_patches.setup_worker()`。
⛔ 下面这段是**修之前**的诊断，留作记录：
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

选哪个 按带宽定：SHM_DISABLE 明显更慢 vs CUMEM_ENABLE=0 ⇒ 选后者（当前口径见 infra_exp/README §6）
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
  `syncopate/train/prune_ckpts.py` 只留 LoRA 键（默认跳过最后一个，保留断点续跑）。
- **`pkill -f <模式>` 会自匹配**（执行 pkill 的 shell 自己也含该模式）—— 犯过三次。
- **最可信的不是 commit message，是 `outputs/<日期>/<时刻>/.hydra/overrides.yaml`**
  —— 那是那次实际跑的全部配置。查"上次到底怎么跑通的"就看它。

## §Modal · 算力搬家实测记录（★ 叙事/学到的/进度/怎么起在 `31-modal-and-new-stack.md`，本节只留读数）

```
为什么是 PRO 6000   与 4×5090 同芯片 GB202 / sm_120：flash-attn 特制轮子、TRITON 注意力后端、E30 的 MXFP8 GEMM、
                   四条 DDP 补丁**全部原样能跑**；96 GB×2 = 192 GB ⇒ 装得下 9B 学生 + 27B 教师 + 18k 上下文 rollout；
                   ~$3.03/卡时，比 H100 便宜 1/4。B200 只做 infra 探针（MAINLINE-INFRA 已交接）。
Modal 事实         GPU 函数默认可抢占、**不可关闭**（RL 靠 ckpt 续跑；考场每遍落盘、可重入）· 单次调用 ≤24h（--detach）·
                   Volumes 持久（1 TiB 免费）· 主机驱动 580.95 / CUDA 13.0，12.x 镜像兼容 · 基础镜像无 nvcc，用 nvidia/cuda devel
                   · 容器互联为 experimental ⇒ 考场链（PG+Redis+API+worker+vLLM）**单容器全套**起
形状              训练：先 colocate 同步 DDP=2（最干净基线），再训推分离 1+1 异步对照（26 §W4 之后的 R6 选形）
                   serving 实测（27/30 的挂账 T5/D2）：2 卡 ⇒ 2 引擎 + 亲和 router；dev mode 四模型 = 每卡两个 vLLM 进程
                   （4B 各 ~8 GB + KV）；⚠️ 与 E32/E33 的四卡读数**不可比**，按新拓扑重立 SLO 基线（30 §5）
★09-03 实测       探针 `modal_app/probe.py`（用法/判据表见 `modal_app/README.md`；读数 `_audit/modal_probe/summary_*.json`）：
                   镜像（nvidia/cuda:12.8.1-devel + python3.12 + `uv sync --frozen --all-extras --no-install-project`）一次建成 ✅ ·
                   Volume `syncopate-home` 跨容器读写一致 ✅ · 代码经 git clone 到 /vol/repo、HEAD 与 origin 一致、invariants 违反集合 ⊆ 本机 ✅ ·
                   单卡 PRO 6000 Server Edition 97887 MiB · 驱动 580.95 · sm_120 · **flash-attn 反向判据退出码 0**（cu13torch2.9 轮子原样可用）✅ ·
                   Qwen3-4B/0.6B 权重字节数 == HF 声明、bf16 贪心两次逐 token 相同 ✅ · 数据链 0–4 步在 Modal 上全部跑通 ✅
                   ⚠️ 切分 SHA 与 git 冻结版不同 —— **不是环境差异**：HEAD 代码本机也生成不出旧切分（裁定⑨ 后 4 对 case 题面同形被去重），
                   待 Chaoyu 裁 DATA_VERSION 升版（26 §6③）。⚠️ 容器里 `lscpu` 型号与 `pci.bus_id` 被隐藏，拓扑指纹只剩 NUMA/区域/云（GCP asia-south1）。
                   ⚠️ 双卡 NCCL 各变体（`--steps nccl`）尚未跑；08 §5 记着 4×5090 上 NCCL_P2P_DISABLE=1 无效、真解是 NCCL_CUMEM_ENABLE=0，两条都在探针里。
★09-03 新栈实测     裁定⑪（26 §6）换法三：`modal_app/stack/`（vLLM 0.28.0 · torch 2.13.0+cu130 · verl 0.9.0 · transformers 5.10.4 · FLA 0.5.2 ·
                   flash-attn 2.8.3 社区轮子 mjun0812 cu130torch2.13）在 PRO 6000 上**六步全绿**（`modal_app/stack_probe.py`，读数 `_audit/stack_probe/`）：
                   · flash-attn 社区轮子反向六项与 fp32 参考对到 4 位有效数字（社区轮子的闸过了）
                   · FLA GDN：chunk（训练核）前向相对误差 0.6%、五路梯度相对误差 ≤0.67%、有限非零；fused_recurrent（解码核）前向 0.46%，
                     **FLA 明说不实现它的反向**；T=4096·H=8 前向+反向 1.91 ms
                   · vLLM 0.28 起 Qwen3.5-9B（18432 ctx）：选 FLASH_ATTN 后端；GDN prefill 走 Triton/FLA、**decode 走 CUDA 核（不是回退路径）**；
                     识别 Qwen3_5MTP、draft 与目标共享 embedding/lm_head；**MTP 开 ⇒ 每 token 12.25→6.68 ms（1.83× 更快）**
                     ⇒ 上面待验风险⒝「27B 开 MTP 反慢 3.6×」的先验在 0.28 + 9B 上**不成立**，那条是 0.2x 时代 27B 的读数，别再引用
                   · 坑：venv/bin 必须在 PATH（FlashInfer 采样核启动时用子进程 JIT 调 ninja，找不到 ⇒ EngineCore 起不来）；
                     FLA naive 参考的参数顺序是 (q,k,v,beta,g) 而 chunk 是 (q,k,v,g,beta)，位置传参会把衰减传成 β ⇒ 参考发散成 NaN
                   · verl 0.9 结构：V1 trainer 在 `verl.trainer.ppo.v1`；`verl.trainer.distillation`（fsdp/megatron 各一套 loss）；
                     actor 后端 dp/megatron/mindspeed/torchtitan/veomni；engine automodel/fsdp/megatron
                   ⚠️ Chaoyu 09-03 晚：PRO 6000（sm_120）**物理上没有 TMEM/tcgen05，跑不了 FA4**，学习项目要最新栈 ⇒ 转 **B200（sm_100）**；
                   PRO 6000 上的这组读数留作对照，不再投入。
★09-03 晚 B200 实测   裁定⑫：一切在 **B200（sm_100，183359 MiB，驱动 580.95）** 上配。新栈镜像 + FA4 独立 venv（flash-attn-4 4.0.0b29 与 vllm 0.28 的
                   apache-tvm-ffi 钉冲突 ⇒ /env/.venv-fa4）。`stack_probe.py` 读数（`_audit/stack_probe/summary_2026-09-03_19*.json`）：
                   · versions ✅（守则⑯机器判据首次抓到 2 条未写原因：flash-attn 2.8.3 vs post1 · cutlass-dsl 4.6.2 被 vllm 钉，已登记）
                   · 权重：Qwen3.6-35B-A3B 67.0 GiB · Qwen3.8-27B 51.7 GiB · 3.5-4B · 3.5-0.8B 字节数 == HF ✅
                   · flash-attn 2.8.3 社区轮子在 sm_100 反向六项对 fp32 ✅；FLA chunk 核梯度误差 ≤0.67% ✅（单次计时 0.75–2.93 ms 抖动大，别拿单次当读数）
                   · **FA4（CuTe DSL，`flash_attn.cute.interface`）**：前向 vs sdpa 误差 0.2% · 反向有限非零 · S=8192 前向 0.212 ms vs FA2 0.85 ms = **4.0×，≈1297 TFLOPS**
                   · **NVLink**：每卡 18 链路×53.125 GB/s；双卡 all_reduce busbw 66/495/546 GB/s（16MB/256MB/1GB）· all_gather algbw 241/792/871 GB/s
                     ⇒ 对照 4×5090 PCIe 的 25.6 GB/s（08 §1）= **~34×**；P2P 可用（can_device_access_peer=True），NCCL 默认参数直接通、无挂死
                   · 坑：容器里 `nvidia-smi topo -m` 报 Failed（拓扑矩阵被隐藏，只有 nvlink -s 可读）；torch mp.spawn 脚本必须有 `__main__` 守卫；
                     判据里的 capability 期望值随卡变（sm_120→sm_100）。
                   · **vLLM 0.28 + Qwen3.6-35B-A3B（AOT 镜像：flashinfer-jit-cache/cubin 同版本）**：单卡起服务 268 s（含 torch.compile 缓存命中）；
                     单流每 token **MTP 关 4.35 ms / 开 3.83 ms（0.88×，MTP 快 12%）**；两卡 **EP=2**（`--data-parallel-size 2 --enable-expert-parallel`）
                     起服务 186 s、单流 4.99 ms/token（单流比单卡慢是预期：EP 换的是吞吐不是延迟）。核选择（学习项）：
                     attention=FLASHINFER（decode_backend=trtllm-gen, arch=sm100）· GDN prefill=FlashInfer 核、decode=CUDA 核 ·
                     MoE=FlashInfer TRTLLM（TrtLlmBf16ExpertsMonolithic；cooperative launch 140/148 SM 给 MoE）· EP 的 all2all=AgRsAll2AllManager
                     （默认 allgather_reducescatter；DeepEP 后端未装）· all-reduce 候选 NCCL_SYMM_MEM/QUICK_REDUCE/FLASHINFER/CUSTOM/SYMM_MEM/PYNCCL ·
                     MTP 架构名 Qwen3_5MoeMTP。收尾后显存 4 MiB（杀全家+等归零判据生效）。
                     · 首启 JIT 路线（无 AOT 包）实测：单卡 15 min 起不来、双卡 gcc 段错误 ⇒ AOT 包是必需品不是优化（守则⑰）。
试点四步          ① 镜像：nvidia/cuda:12.8 devel + uv sync --all-extras + check_flash_attn_backward + check_pipeline_invariants ✅（上行）
                   ② PRO6000:2 跑 SFT 五点谱 + 冻结 EVAL，与训练机 v15_r3 读数对一遍（可比性判据）
                   ③ 考场链单容器化，跑一遍 v4 考卷（同时做 W1④ 校准）
                   ④ RL 100 步 + 主动杀容器测抢占续跑
待验风险          ⒜ 双 PRO 6000 有 NCCL P2P 在首个集合通信挂死的公开案例（NVIDIA 论坛 09-2026），解法 NCCL_P2P_DISABLE=1
                   ⇒ 与 4×5090 一样按"无 P2P"起 DDP，不影响正确性，只影响速度；上机第一件事跑 E21 那套对齐带宽探针
                   ⒝ Qwen3.5/3.8 的 Gated DeltaNet 层：vLLM 的融合 GDN 解码核曾按 capability 100 门控、sm_120 静默退回 Triton/FLA
                   参考路径（main 已修为 ≥80）；**27B 开 MTP 时因 v/k 头比 48:16≠8 融合核不可用、MTP 净慢 3.6×**（HF 讨论区实测）
                   ⇒ 学生换 Qwen3.5-9B 前先在 PRO 6000 上量：FLA 前向/反向对拍 + 开/关 MTP 的 TPOT，判据跑前注册
```
