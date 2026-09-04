# Syncopate · 31 · Modal + B200 新栈：学到的一切、进度与坑（2026-09-03 → 09-04）

> **这份是 Modal/新栈这条线的唯一入口**（Chaoyu 09-04 立）。读数的原始记录仍在 `08 §Modal`（实测表）与 `_audit/stack_probe/`，
> 施工步骤与判据在 `26 §W4′`，探针代码在 `modal_app/`。本文只放：为什么这么做 · 现场长什么样 · 学到了什么 · 进度到哪 · 下一步怎么起。
> 维护纪律同守则⑪：就地改写不追加；数字只从探针 json 抄，不凭记忆。

---

## 0 · 一句话现场（09-04 14:45 收口）

**新栈全管线的"机器部分"全部冒烟通过（SFT 机制 / 考场链路 / RL verl 0.9 首次跑通 / OPD 机制）；管线收成唯一入口 `scripts/v16_pipeline.sh`（17 stage，默认值直接跑）；
三桶隔离做成硬机制（守则⑱）；题库扩量到 2030 道新情景；唯一还没落地的产物 = v16 SFT 训练集：run27 正在云上跑最后几道闸（出厂体检那几项可能红、且属于"要 Chaoyu 裁定"的类型）。**
真数据的 SFT/评测/RL/OPD 一步都还没跑（都等训练集）。

---

## 1 · 为什么是 Modal、为什么是 B200、为什么全换栈（三条裁定的因果链）

```
09-03 上午   裁定：训练搬 Modal RTX PRO 6000×2（sm_120，与 5090 同指令集零移植）
09-03 下午   Chaoyu：辅导老师看简历说「技术栈都是一年前的」——根因是长期为 5090 迁就旧版本（vLLM 0.12/torch 2.9/verl 0.8）
             ⇒ 裁定⑪：整套栈换最新（vLLM 0.28 / torch 2.13 cu13 / verl 0.9 / transformers 5.10 / FA4 / FLA），学生教师换 Qwen3.5 家族
09-03 晚     实测：PRO 6000 物理上没有 TMEM/tcgen05 ⇒ FA4 跑不了、MXFP8 块缩放核走回退
             ⇒ 裁定⑫：PRO 6000 也不用，一切在 B200（sm_100）上配；B300 待全链通后重跑收坑（sm_103a 库在追）
09-03 晚     裁定⑩：数据口径全部 v16 从零重来（HEAD 代码已生成不出冻结的 v13 切分）；裁定⑬：教师只要装得下就用大的（Qwen3.8-27B 兼两角色）
09-04        Chaoyu：万亿旗舰不做；首要目的是**用新栈学新东西**，模型本身 0 价值
```

Modal 和 RunPod 的本质差别：RunPod 租的是一台固定机器，拓扑是常量；Modal 租的是"N 张同机的卡"，机器/区域/云每次可能不同，函数可抢占且不可关。
由此派生的全部纪律在 `00 §5 守则⑰`（一切网络重活进容器、一个写者、拓扑指纹、并发 A/B、对象不许条件定义）。

---

## 2 · 现场长什么样（家的形状）

```
账号        Modal profile spaemtuerl（Starter：$30/月额度，GPU 并发 10）· token 在 ~/.modal.toml
镜像        nvidia/cuda:13.0.2-devel-ubuntu24.04 + python 3.12 + uv + `uv sync --frozen --all-extras`（依赖表 modal_app/stack/）
            + /env/.venv-fa4（FA4 单独 venv：与 vllm 的 apache-tvm-ffi 钉冲突）+ postgresql-16 + redis-server
Volume      syncopate-home（免费 1 TiB 内）
              /vol/repo.git        git bare 镜像（flock），每容器 clone 到 /tmp/repo（一个写者）
              /vol/data/{batches,sft,rl}   gitignored 数据（data/splits 在 git 里）
              /vol/models          Qwen3.6-35B-A3B(67G) · Qwen3.8-27B(52G) · Qwen3.5-0.8B · 对照留存 Qwen3.5-9B/27B/4B
              /vol/flashinfer_cache · /vol/vllm_cache   编译/自动调优缓存（编一次永久用）
              /vol/_audit/stack_probe/<step>.json · /vol/_audit/v16/{build.log,cache/,gallery.md…}
GPU 标签    B200（sm_100，183359 MiB，驱动 580.95）· B200:2（NVLink 5，每卡 18 链路×53 GB/s）
secret      wandb-secret（Chaoyu 建，键 WANDB_API_KEY；判据 --steps wandb ✅）
运行        PYTHONPATH=/tmp/repo · SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 · /env/.venv/bin/python
```

**探针 = `modal_app/stack_probe.py`**（一个文件，每步一个判据，读数落 Volume + 本机 `_audit/stack_probe/summary_*.json`）：

| 步 | 判据 | B200 读数（09-03/04） |
|---|---|---|
| image | 主栈全 import；torch cuda 13；transformers 有 qwen3_5 | ✅ |
| versions（守则⑯） | 装的 == PyPI 最新，否则白名单里有原因 | ✅（6 条被 vllm/verl 钉，有登记） |
| models | safetensors 字节 == HF 声明 | ✅ 四模型 |
| gpu | sm_100；flash-attn 社区轮子反向六项对 fp32；FLA chunk 核梯度误差 ≤0.67% | ✅ |
| fa4 | CuTe DSL 前向 vs sdpa ≤5e-2；反向有限非零 | ✅ **4.0× FA2，≈1297 TFLOPS** |
| nccl | 双卡 sm_100；带宽 | ✅ all_reduce 546 · all_gather 871 GB/s（4×5090 PCIe 的 34×） |
| vllm | 35B-A3B 单卡起；MTP 关/开 | ✅ 4.35 / 3.83 ms/token（MTP 快 12%） |
| vllm_ep | 两卡 EP=2 起 | ✅ 4.99 ms/token（单流慢是预期；all2all=AgRsAll2All） |
| pytest | 仓库全量（PG/Redis 容器内起） | ✅ 退出码 0（951 passed / 23 skipped） |
| rebuild_v16 | Modal 生成 vs 本机生成切分 SHA | ✅ 三份逐一相同 |
| wandb | key 注入 + 写 3 读 3 | ✅ |
| build_v16 | 26 §W4 七步 | 🔴 run16：前五段 ✅，CoT 成行 0（归因待查，见 §4 与 26 §W4′） |
| sft_smoke / exam_v4 | 已写好未跑 | ⬜ |

---

## 3 · 学到的东西（按"以后还会用到"排序）

**硬件与栈**
- sm_120（5090/PRO 6000）和 sm_100（B200）不是一回事：前者是 HMMA 老路线、**物理上没有 TMEM**，FA4/MXFP8 融合核跑不了；数据中心 Blackwell 才是所有新库的第一目标。
- `sm_100a` 编的核只跑 10.0，`sm_100f` 家族版跑 10.0/10.3；B300 的坑集中在"谁还只编了 100a"（FlashInfer 已带 Sm103a）。
- FA4 = CuTe DSL 写的 Python 核，`flash_attn.cute.interface`；与 vllm 0.28 因 apache-tvm-ffi 钉冲突 ⇒ 分 venv。
- FLA 的 GDN 两个核分工：`chunk` 训练（有反向），`fused_recurrent` 解码（**不实现反向**）；naive 参考的参数顺序是 (q,k,v,beta,g) 与 chunk 相反 ⇒ 一律关键字传参。
- MTP 是模型预训练时就带的 1 层小 transformer（config `mtp_num_hidden_layers`），挂在最后一层与 lm_head 之间；没训过 MTP 头的模型不能用；接受率随任务可猜性变。vLLM 0.28 认 `Qwen3_5MoeMTP`。
- Qwen3.5/3.6 MoE：40 层×256 专家 top-8 + 共享专家，30 层线性注意力 + 10 层全注意力，词表 248k；`AutoModelForCausalLM` 解析到 `Qwen3_5MoeForCausalLM`（纯文本）。
- **Qwen3.5 chat 模板的工具调用是 XML 线格式**（`<function=…><parameter=…>`），不是 Qwen3 的 JSON；`enable_thinking=True` 才开 `<think>\n`；只渲一条 tool 消息会 raise "No user query found"。
- transformers 5：`apply_chat_template(tokenize=True)` 返回 BatchEncoding ⇒ `len()` 量成 2、`+ list` 报错。
- FlashInfer 默认 JIT：sm_100 首启编 TRTLLM MoE/GEMM 核十几分钟、并发编译 gcc 段错误 ⇒ 必须装同版本 `flashinfer-jit-cache` + `flashinfer-cubin`（GitHub release 资产，不在 PyPI）。
- verl 0.9 未声明但 import 的依赖：`transferqueue`（V1 trainer）· `cupy-cuda13x`（NCCL ckpt engine）· `prefix-grouper`；上游已吸收我们的三件（PrefixGrouper、LoRA-only ckpt、rollout_correction）。
- Megatron 是 NVIDIA 的；Megatron-Bridge = HF↔Megatron 权重双向翻译 + 配方库，verl 0.9 默认；Miles（LMSYS）= Megatron+SGLang，MXFP8/NVFP4 训推统一 recipe 在 8×B200 上 reward 曲线与 bf16 重叠。

**Modal 运维**（细则在 `00 §5 ⑰` 坑表与 `modal_app/README.md`）
- `modal app stop` 不带 `--yes` 静默不执行（白烧 40 min）；Modal 对象不许按环境变量条件定义；`--detach` 的 `modal run` 仍阻塞到结束；本机 `uv lock` 拉大轮子超时 ⇒ 在容器里解锁（`modal_app/lock_on_modal.py`）。
- 一个写者：并发 run 对同一 Volume 路径做 git 会撞 index.lock ⇒ bare 镜像 + 每容器 /tmp checkout。
- 收尾杀全家等显存归零；起服务日志落 Volume；子进程结果走文件不走 stdout 尾行；PATH 必须含 venv/bin（FlashInfer JIT 找 ninja）。
- A/B 各臂并发是 Modal 独有红利，纪律五条（各臂各目录、拓扑指纹、本机统一汇总）。

**数据与契约**
- HEAD 代码生成不出冻结的 v13 切分：裁定⑨（account_id 不进题面）后 4 对题面同形被去重 ⇒ 数据是代码的纯函数，判据改为"两地独立生成逐字节相同"。
- 换线格式后，凡是 grep/regex 找 `"name":` 的地方都是雷（教师动作解析、行为探针、数据审计、决策位熵锚点）——全库扫了一遍。
- 压舱桶不能再来自旧 parquet：取当前切分 sft 题、每 6 留 1。多轮行源题必须是终答型。

---

## 4 · 进度与四次建库失败的归因

| 阶段 | 状态 |
|---|---|
| 环境（九步探针）| ✅ |
| S0 对齐地图 | ✅ 639/10/318 → 951 passed 全绿 |
| S1 对齐（v16 口径、模型路径、XML 线格式、补丁分诊、PG/Redis） | ✅（26 §W4′ 逐条） |
| S2 v16 题库确定性 | ✅ 09-04 13:30 扩量后重定：本机与 Modal 各自生成 2030 条 / 8121 文件，三份切分 SHA 逐一相同（Modal 128 s）；SHA 在 _audit/v16/local_gen_sha.json |
| S3 v16 训练集建库 | 🟡 run17–27（09-04）逐道闸显形并修（见 26 §W4′ 各 run 记录）；**run27 在跑**，若红下一轮用 `--build-gates report` 一次看全貌 |
| S4 SFT 冒烟（p_sft_smoke） | ✅ 机制 mech_dry（DRY 占位数据）：可训 42.3M · loss 1.93→0.29 · ‖ΔW‖/‖W‖ 0.63% · 峰值 74 GB · 30 步 346 s · adapter 落盘；真数据版等 S3 |
| S5 考场 v4 单容器（p_exam_v4） | ✅ 链路 plumb（底座、40 题）：PG/Redis/播种/端点（起 500 s）/40 题 137 s/judge/triage 全 rc 0；分数不看 |
| S6 RL 冒烟（verl 0.9 V1） | ✅ 09-04 01:54 **首次跑通**：B200×2 · 2 步 580 s · 动态分池在 TaskRunnerV1 进程生效 · loss/grad 有限 · reward 0.34/0.39 · 权重同步 · LoRA-only ckpt（233 MB/rank）· wandb syncopate-b200/rl_v16_smoke |
| S7 OPD 冒烟 | 🟡 机制通（vocab ✓ · 42.3M · 真蒸馏步 KL 有限 · adapter 落盘）；底座无 adapter 几乎全跳步 ⇒ 语义冒烟等 S4 真 SFT adapter |

```
（run17–27 的逐轮记录在 26 §W4′：CoT 语言闸 → 419 旧行数 → OOV 跑量 → 考场泄漏 → think 双开头/压舱兜底 → DIA 译名 → 三桶隔离 367 行越桶 → 旧切分行缓存 → L2 供给 144<280）
run13  hydrate 即死：secret 按环境变量条件定义，本机/容器求值不同
run14  L2/L1 断言 FRESH_0125 无人话终答：源题没按收场类型过滤（defer 无人话终答），旧缓存作废后显形
run15  CoT 段崩：候选从上一版 parquet 选（v16 没有上一版）；同时发现教师动作解析/行为探针按 JSON 找 "name":（XML 下永远 0）
run16  前五段全过；CoT 蒸馏采样成功（≥14 条「收」）但成行 0 ⇒ 下限闸红
09-04  归因（26 §W4′「S3 run16 成行 0 的归因」）：① 选择步 Σsurplus≥0 在"每行 1/10 步有思考"下无解 ⇒ 0 行是算术不是丢行
       ② 27B 采样 892 步只中 12（1%）；那些"1 步思考"几乎全是 v15_materials 里 8B 旧料的静默复用 ③ 过滤链每个丢弃原因静默
       ⇒ 裁定⑭（v16 不混任何旧物料）+ 丢弃计数 + teacher_diag 画像（预注册判读）
```

---

## 5 · 怎么起（命令都在 `modal_app/README.md`，这里只给顺序）

```
★ 固定管线   bash scripts/v16_pipeline.sh [--dry-run] [--profile smoke|candidate] <stage|all>   # 09-04 起唯一入口（本机 == 云上）
             stages: cases menus split gates rl-data teacher sft-data sft-train sft-eval sft-select merge exam rl-train rl-adapter rl-eval opd-train opd-eval
环境健康   modal run modal_app/stack_probe.py --steps versions,gpu,nccl        # 换镜像/换卡后
建库       modal run --detach modal_app/stack_probe.py --steps build_v16       # 读数 /vol/_audit/v16/
冒烟       modal run --detach modal_app/stack_probe.py --steps sft_smoke --max-steps 30
考场       modal run --detach modal_app/stack_probe.py --steps exam_v4 --exam-adapter /vol/checkpoints/sft/… --exam-passes 1
容器里跑脚本 modal run modal_app/stack_probe.py --steps exec --exec-file x.py
停         modal app list → modal app stop <id> --yes → 再 list 核对
诊断       modal run --detach modal_app/stack_probe.py --steps teacher_diag                     # 27B 原始思考画像（S3-diag）
并行冒烟   --steps sft_smoke --sft-arm mech_dry · --steps exam_v4 --exam-arm plumb --exam-limit 40 · --steps rl_cfg（CPU）→ rl_smoke · --steps opd_smoke
推代码     本机 HTTPS push 会静默挂死（09-04 实测 15 min 无进展）⇒ `git push git@github.com:ChaoyuWang04/Syncopate_Async_AgenticRL.git main`（SSH 已配）
```

成本：B200 $6.25/卡时；今天全部实验（含反复）估计 <$40。
