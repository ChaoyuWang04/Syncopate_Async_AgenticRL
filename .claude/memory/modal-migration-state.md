---
name: modal-migration-state
description: ★09-04 15:05 收口：新栈全管线机器部分全冒烟通过；唯一入口 scripts/v16_pipeline.sh；守则⑱ 三桶隔离硬机制；题库扩量 2030；v16 SFT 训练集仍未落地（run27 在跑最后几道闸，红了用 --build-gates report 看全貌）；三件待 Chaoyu 裁定；唯一入口 docs/syncopate/31
metadata:
  type: project
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**（§0 一句话现场 · §4 进度 · §5 怎么起）；施工与判据 `26 §W4′`（每个 run 倒在哪、修了什么、闸表、固定管线 stage 表）；守则 `00 §5`（⑯⑰⑱ 本轮新增/扩写）；探针 `modal_app/stack_probe.py`（只调 runbook）。

**已落地（09-04）**：裁定⑭ v16 不混旧物料 · ⑮ CoT 语言不限+上限 12288/12288/24576+THINK 2048 · 守则⑱ 三桶隔离三层硬机制（源头/登记/出口唯一写盘 + 复核器）· 题库扩量新情景（拒绝 14 种、RELN/FRCP/BCUT；2030 道；两地 SHA 同）· **固定管线 runbook `scripts/v16_pipeline.sh`**（17 stage · --dry-run · smoke/candidate 档；所有入口默认值从 DATA_VERSION/model_paths 派生；test_pipeline_defaults 守着）· 本机可验：DRY 走完全部结构闸；`check_supply_vs_floors.py` 供给对数量闸；离线全量建库 `sft-data-offline`；闸观察模式 `U_BUILD_GATES=report`（一次看全部红项）· 行缓存绑定切分 SHA + 构造器版本 tag（自动作废）· 云盘过时产物已清（冒烟 ckpt、v15 归档缓存）。

**新栈冒烟读数**：SFT 机制 42.3M/74 GB/30 步 346 s · 考场链路全通（35B 端点冷启 500 s）· RL verl 0.9 V1 sync 2 步 580 s（step2 37 s，LoRA-only ckpt 233 MB/rank）· OPD 机制通（vocab 相同）· 老师 CoT：英文思考、命中 67%、截断 0.1%。

**卡点**：v16 SFT 训练集。run27（09-04 15:01）走到最后一道闸出厂体检，剩 3 项全是「六族行终答借压舱人话 / CLAF 跑题回复走错生成器 / WIN 模板重复」⇒ 修法在 26 §W4′ run27（派生行要有自己的教师人话）。产物先写 staging、体检全绿才搬正式目录（已改）。下一轮上云用 `--build-gates report`。**未验**：rl-adapter（verl 0.9 model_merger 对 FSDP2+LoRA-only）· sft-eval/select/merge/rl-eval/opd 真数据链。

**待 Chaoyu 裁定**：L1 底题复用作历史 · 切分格 SFT=0 加保底否 · 六桶份额带宽回填。

**坑（本轮）**：HTTPS push 挂死 ⇒ SSH 推；pkill -f 会杀自己；`modal volume rm` 不吃 --yes；判据红先问"量的是不是那件事"（本轮 6 次红是判据量错对象）；"记得删缓存"不是机制（run25 实案）；DRY 不走数量闸 ⇒ 供给要单独算。

**Why：** Chaoyu 09-04：一切 v16 · 硬机制保证数据可靠 · 固定脚本默认值直接跑 · 放开闸看全貌。
**How to apply：** 接手先读 31 → 26 §W4′ → `modal app list`；本机 `bash scripts/v16_pipeline.sh --dry-run all` 看每段命令；改闸先本机 DRY/供给核对/离线建库，再上云。
