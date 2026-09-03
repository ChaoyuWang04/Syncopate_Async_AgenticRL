---
name: modal-migration-state
description: ★09-04 深夜：家在 Modal B200×2 新栈；裁定⑭ v16 不混任何旧物料；S3 成行 0 已归因（算术+27B 命中 1%+8B 旧料暗道）；teacher_diag / sft mech_dry / exam plumb 三臂并行在跑；S6/S7 已写未跑；唯一入口 docs/syncopate/31
metadata:
  type: project
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**（为什么/现场/学到的/进度/怎么起）；施工与判据 `26 §W4′`（含「S3 run16 成行 0 的归因」「S3-diag」「并行冒烟」三节）；探针 `modal_app/stack_probe.py`。

**裁定链（Chaoyu 09-03/04）**：⑩ 全部口径 v16 从零重来 · ⑪ 全新栈、目的是学新东西 · ⑫ 一切在 B200 · ⑬ 教师只要装得下就用大的（Qwen3.8-27B 兼两角色）·
**⑭（09-04）v16 不许混进任何旧版本产物**：4B/8B 物料（reply/think/定义/闲聊）全由 27B 重生成，v13 triage 不再读，缓存名全带版本 v16_*，Volume 上 run14–16 的 v15_* 缓存搬 pre_v16_run16/ 留档。

**S3 成行 0 的真相（09-04，前任 -7b 确认）**：选择步 Σsurplus≥0 在"每行 1/10 步有思考"下必然无解 ⇒ 0 行是算术不是丢行；上游 27B 采样 892 步只中 12（1%）；那 64 行的"1 步思考"几乎全是 v15_materials.json 里 8B 旧思考的静默复用。过滤链（900 token 内要 </think> · cjk≥0.5 · 首动作==gold）每个丢弃原因静默 ⇒ 已加计数；`teacher_diag` 步量 27B 原始思考画像，判读预注册（closed_within_900 <50% ⇒ 上限是主拦截；cjk_below_0.5 >50% ⇒ 语言闸）。**诊断结果出来前不改任何阈值。**

**进度（09-04 深夜）**：环境九步 ✅ · S0/S1/S2 ✅ · S3 归因定、诊断在跑 · S4 机制冒烟 mech_dry（DRY 占位数据，非候选）在跑 · S5 链路冒烟 plumb（底座、40 题）在跑 · S6 `launch_rl_v1`（verl 0.9 V1 薄壳；动态分池补丁改挂 `trainer.ppo.utils`+`v1/trainer_base`）已写，`rl_cfg`（CPU 键名判据）先跑 · S7 opd.py v16 化已写（学生/教师 vocab 逐项相同已核，chat_template 不同）。读数落 `/vol/_audit/v16/{teacher_think_diag.*, sft_mech_dry/, exam_plumb/, rl/, opd/}`，本机 `modal volume get`。

**坑（新增两条）**：本机 HTTPS `git push` 静默挂死（15 min 无进展，凭据无关）⇒ 一律 `git push git@github.com:ChaoyuWang04/Syncopate_Async_AgenticRL.git main`；`pkill -f "git push"` 会连自己的 shell 一起杀。其余见 31 §3 与 00 §5 ⑰。

**Why：** Chaoyu 09-04 原话「不允许任何之前版本的产物混进我们这一版……一切都是 v16」；并要求能并行的冒烟多起机器、最终整条管线跑通。
**How to apply：** 接手先读 31 → 26 §W4′ → `modal app list`；判据红了先怀疑解析器/路径/判据量错对象（本轮 stale 判据就是量错对象）再怀疑模型；每步先注册判据再跑。
