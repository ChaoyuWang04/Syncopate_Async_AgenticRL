---
name: modal-migration-state
description: ★09-04 03:10 收口：新栈全管线机制冒烟全通（SFT/考场/RL verl0.9 V1/OPD）；S3 建库卡在一个待裁定（27B 教师英文思考撞中文闸，推荐=中文引子+改尺子）；裁定⑭ v16 不混旧物料已落地；唯一入口 docs/syncopate/31
metadata:
  type: project
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**；施工与判据 `26 §W4′`（「S3 run16 成行 0 的归因」「S3-diag」「并行冒烟」+ S4′–S7 读数）；探针 `modal_app/stack_probe.py`（步：teacher_diag · sft_smoke · exam_v4 · rl_cfg · rl_smoke · opd_smoke）。

**裁定链（Chaoyu 09-03/04）**：⑩ v16 从零重来 · ⑪ 全新栈学新东西 · ⑫ 一切在 B200 · ⑬ 教师用大的（Qwen3.8-27B 兼两角色）· **⑭ v16 不许混进任何旧版本产物**（4B/8B 物料全由 27B 重生成、v13 triage 不读、缓存名 v16_*、旧缓存搬 pre_v16_run16/）。

**★ 唯一待 Chaoyu 裁定（S3 建库的闸）**：27B 教师**全程英文思考**（cjk p50=0.0，中文闸拦下 100%；900 上限 93% 够用；英文思考里 68% 选中 gold 动作，质量好）。B 臂「<think> 后加中文引子」：cjk p50 0.48、动作命中 72%、现行链通过 35%；尺子改成「中文字÷(中文字+拉丁字母)」（不被工具名稀释）后通过 52%。三选一：① 撤中文闸收英文思考 ② 中文引子（+改尺子，**我推荐**）③ 换教师。定了之后：改阈值=重新注册 → `--steps build_v16` → 画廊逐条看。

**新栈全管线冒烟读数（09-04，全在 26 §W4′）**：S4′ SFT 机制（DRY 占位数据）✅ 可训 42.3M · ΔW 0.63% · 峰值 74 GB · 30 步 346 s · S5′ 考场链路 ✅（端点起 500 s 待查）· S6 **RL verl 0.9 V1 sync 首次跑通** ✅ B200×2 · 2 步 580 s（step2 37 s）· 动态分池在 TaskRunnerV1 进程生效 · LoRA-only ckpt 233 MB/rank · S7 OPD 机制 ✅（vocab 相同、真蒸馏步 KL 有限、adapter 落盘）但底座几乎全跳步 ⇒ 语义冒烟等真 SFT adapter。**真数据版 S4→S5→S6→S7 全部等 S3 建库过。**

**verl 0.9 要点**：`trainer.use_v1` + `trainer.v1.trainer_mode`（sync/colocate_async/separate_async）；`create_rl_sampler` 在 `trainer/ppo/utils.py`，V1 trainer_base 导入时绑名 ⇒ 补丁要挂定义处+消费者（已改）；`save_lora_only` 要 `+` 追加；入口 `launch_rl_v1.py`（旧 launch_rl.py 留给旧栈）。

**坑（本轮新增）**：本机 HTTPS `git push` 静默挂死 ⇒ 一律 SSH 推；`pkill -f "git push"` 会杀自己的 shell；并行臂同分钟收尾会盖掉本机 summary（已改带秒+步名）；判据红了先问「量的是不是那件事」——本轮四次红全是判据量错对象（stale 缓存名 / grad_norm 只上 wandb / 峰值字样 / KL 正则）。

**Why：** Chaoyu 09-04「一切都是 v16」+「能并行的多起机器，最终整条管线跑通」。
**How to apply：** 接手先读 31 → 26 §W4′ → `modal app list`；读数在 `/vol/_audit/v16/` 用 `modal volume get` 拉；每步先注册判据再跑。
