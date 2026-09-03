# modal_app · 把家搬到 Modal（现为 B200；全貌见 `docs/syncopate/31-modal-and-new-stack.md`）

> ★ 09-03 裁定⑪（26 §6）：**全新栈**。`stack/` = 新依赖表（vLLM 0.28 / torch 2.13 cu13 / verl 0.9 / transformers 5.10 / FLA 0.5.2），
> `stack_probe.py` = 新栈探针（镜像 → verl 结构 → Qwen3.5 全家权重 → 单卡 flash-attn 反向 + FLA/GDN 对拍 → vLLM MTP 关/开）。
> flash-attn 用 mjun0812 预编译轮子（cu130torch2.13，锁在 `stack/uv.lock`），社区轮子的闸 = 卡上反向判据。
> 下面 `probe.py` 那套是**旧栈**（vLLM 0.12）的搬家探针，只作历史读数，Volume 上旧模型/旧数据已清。
>
> ```bash
> modal run --detach modal_app/stack_probe.py --steps image,verl # ① 镜像 + 导入判据 + verl 0.9 结构
> modal run --detach modal_app/stack_probe.py --steps models     # ② Qwen3.5-9B/27B/4B/0.8B（~80 GiB）
> modal run --detach modal_app/stack_probe.py --steps gpu,vllm   # ③④ 单卡判据 + 学习读数
> ```
> 判据在 `stack_probe.py` 顶部注释里先注册；读数落 `_audit/stack_probe/`（本机）与 Volume `/vol/_audit/stack_probe/`。

> 为什么叫 `modal_app` 不叫 `modal`：仓库根目录下的 `modal/` 会遮住 `import modal` 本身。

## 第一次（本机，一次性）

```bash
uv tool install modal                          # 隔离装 CLI，不碰项目 .venv（已装 1.5.5）
modal token set --token-id ak-… --token-secret as-…   # 写到 ~/.modal.toml；无头机器也可 export MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
```

token 在 modal.com → Settings → API Tokens 生成，由 Chaoyu 的账号出（Starter 档 $0/月含 $30 额度，GPU 并发上限 10）。

## 探针（`probe.py`）

```bash
modal run modal_app/probe.py                   # P1–P6：镜像 · git · Volume · GPU · 权重 · 数据重建
modal run modal_app/probe.py --steps gpu       # 单跑一步
modal run --detach modal_app/probe.py --steps rebuild,pytest   # 长步骤断网不中止
```

退出码 0 全绿 · 1 有判红 · 2 有没跑成的。明细在 `_audit/modal_probe/summary_*.json`（本机）与 Volume `/vol/_audit/modal_probe/`。

| 步 | 回答的问题 | 判据（「两边相同」型） |
|---|---|---|
| image | 依赖表三件套能在 Modal 上 `uv sync --frozen --all-extras` 装出来 | torch/vllm/verl/flash_attn import 退出码 0 |
| git | 代码能经 GitHub 到 Volume | `/vol/repo` HEAD == `git ls-remote origin main`；`check_pipeline_invariants` 在 Modal 上违反的判据集合 ⊆ 本机集合（本机读数由入口现算） |
| volume | 网络盘跨容器可见 | 容器 A 写、容器 B reload 后读到的 == 写入的 |
| gpu（单卡） | PRO 6000 是 sm_120；flash-attn 反向对 | capability == (12,0)；`check_flash_attn_backward` 退出码 0 |
| nccl（双卡，按需） | 双卡 NCCL 通 | 四种 NCCL 环境变量组合各自「通/挂死/红」（120 s 超时抓挂死） |
| model | 权重能从 HF 落 Volume 并在卡上跑 | safetensors 总字节 == HF 仓库声明；贪心生成两次逐 token 相同 |
| rebuild | 数据能在 Modal 上从 git 里的外部数据重新生成 | 切分三份 SHA-256 == git 里 `data/splits/v13`（同 `run_pipeline_shadow_rebuild.sh` 0–4 步） |
| pytest（按需） | 全量回归 | 本机基线 908 passed；无 PG/Redis 的 skip 不算通过 |

## 家的形状

```
镜像      nvidia/cuda:12.8.1-devel + python 3.12 + uv + /env/.venv（只装依赖，不装项目）
Volume    syncopate-home → /vol
            /vol/repo      git clone，每次函数起跑先 reset --hard origin/main（幂等）
            /vol/models    HF 权重（Qwen3-4B · Qwen3-0.6B）
            /vol/data      在 Modal 上重新生成的批次/切分
            /vol/_audit    每步判据 json
运行       PYTHONPATH=/vol/repo · cwd=/vol/repo · /env/.venv/bin/python · SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1
```

## 运维坑（09-03 实测）

- `modal app stop <id>` 在非交互终端会**静默不执行**（提示要 `--yes`），停错过一次让 B200 多跑了 40 分钟 ⇒ 一律 `modal app stop <id> --yes`，停完 `modal app list` 核对。
- `modal volume delete` 同理要 `--yes`。
- 本机拉 GitHub 大轮子 `uv lock` 会超时 ⇒ 用 `modal_app/lock_on_modal.py` 在容器里解锁并拉回 uv.lock。
- FlashInfer 默认 JIT：sm_100 首启编 TRTLLM MoE/GEMM 核十几分钟、并发编译 gcc 段错误 ⇒ 装同版本 `flashinfer-jit-cache` + `flashinfer-cubin`（GitHub release 资产，不在 PyPI），并把 `FLASHINFER_WORKSPACE_BASE` / `VLLM_CACHE_ROOT` 指到 Volume。
- ⛔ Modal 对象（Secret/Volume/Image）不许按环境变量条件定义，本机与容器求值不同 ⇒ hydrate 失败；可选的 secret 先建占位再无条件引用。
- `modal volume cp` 要求目标父目录已存在，否则 "No such file or directory"；容器内 `shutil.move` 更省事。
- vLLM 收尾必须杀全家（APIServer/EngineCore/Worker）并等显存归零，否则下一个实例启动报 free memory 不足。

## Modal 事实（官方文档 09-03 查证）

- Volume 是持久网络盘，多容器共享；写后要 `commit()`，别的容器要 `reload()` 才看得到；v1 上限 50 万文件（建议 <5 万）。
- 存储 $0.09/GiB·月，**每月前 1 TiB 免费**；GPU 按秒计费，PRO 6000 $3.03/卡时，CPU $0.047/核时，内存 $0.008/GiB·时。
- GPU 函数默认可抢占且**不能设成不可抢占**；抢占后 Modal 用同一输入重跑 ⇒ 每步必须幂等。单次调用 ≤ 24 h。
- `modal run` 是临时 App，客户端断开即停；`--detach` 后不随客户端停，`modal app stop` 才停。
- 宿主驱动 580.95 / CUDA 13.0，`12.*` 与 `13.*` 镜像都兼容；nvcc 不预装，用 `nvidia/cuda:*-devel` 镜像自带。
- `modal shell --gpu RTX-PRO-6000 --volume syncopate-home --image …` 可交互进容器排障。
