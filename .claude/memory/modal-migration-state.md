---
name: modal-migration-state
description: ★09-04 晚收口：run28 全绿 v16 SFT 训练集落地（1222 行）+ 真数据 SFT 冒烟通过（30 步，单卡 208 tok/s）；管线逐段审查修完（旧名脚本全改 v16_*）；下一步 = SFT batch 三臂标定改两卡 → 探针加通用 stage 步接 smoke 链 → candidate；00/01 文档被另一会话重写中；入口 docs/syncopate/31
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ed1b9c6-3ae1-544f-b9a5-207777e46c52
  modified: 2026-09-04T09:55:42.946Z
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**（§0 现场 · §4 进度 · §5 怎么起）；施工与判据 `26 §W4′`（run 逐轮记录、闸表、stage 表、**09-05 审查与修法块、run28 块、S4 真数据冒烟块**）；探针 `modal_app/stack_probe.py`（只调 runbook）。
⚠️ `00-START.md` / `01-TASKS.md` 在 09-04 17:42 被**另一个会话**大幅重写（424→107 行、237→70 行，工作区未提交），不是本会话改的；接手先 `git status docs` 看是否已入库，别把别人的重写当成自己的提交。

**已落地（09-04/05，commit 4483337 · 538fe18 · c7b40fc + 收口提交）**
- 管线逐段审查（Chaoyu：runbook 固定"调谁"≠被调的是新版，见 [[pipeline-callees-must-be-current-too]]）：v16 路径 14 个旧名脚本 git mv 成 `v16_*`；OPD 渲染进 `syncopate/train/opd_render.py`；supply 崩/gates 漏泄漏闸/menus v8 审计/考场链三处必死/探针空判据/缓存标签拆分/分词器唯一定义 `model_paths.build_tokenizer_path`/entropy 2048/L2 val 旧数/select 无版本审计/行为探针全库 全修；判据 `test_runbook_references_no_old_version_scripts` + supply 真跑测试。
- run27 三红修法：as_multiturn 不再借压舱句（硬机制）+ gen_variant_reply · CLAF 跑题回放取观测 + gen_fact_reply · WIN 教师现写 + 素材五池轮转。
- **run28（report 模式）零红**：1222 行 / 18 桶；同形 0 · 越桶 0 · 预设答案无 · 六族句式全唯一；产物 `/vol/data/sft/v16`。份额 l2 4.1% / l1 2.3% 仍带外（只报）。人工抽看六族行终答 OK；个别终答夹机器词（"executed"）登记为下一轮闸候选。
- **真数据 SFT 冒烟 ✅**：30 步 317 s · val loss 0.31 · ΔW 0.54% · 42.3M · ~208 sup-tok/s；adapter `/vol/checkpoints/sft/v16_smoke`（非候选）。探针峰值显存行没抓到（正则）。

**发现（等排期）**
- SFT 现为**单卡**：探针 sft 步申请 GPU_ONE，sft.py 自动 torchrun 阈值 ≥4 张卡 ⇒ B200:2 永远单卡；bs 2×accum 8 显存用不到一半 ⇒ 三臂标定（单卡 2×8 / 单卡 8×2 / 两卡 2×4，有效 batch 16、lr 不动）后改默认与探针 GPU_PAIR。
- RL/OPD 默认也是保守值（RL train batch 2 / mini 2 / micro 1 / rollout util 0.45 / max_num_seqs 32 / TP 1；OPD 学生单卡 batch 8），未在 B200 上标定。FP8（lm_head MXFP8、E31 插件）与 Prefix Grouper 都默认关；RL 默认只活两个自研补丁（FSDP CPU 拷贝、动态分池）。
- 探针缺六段：sft-eval / sft-select / merge / rl-adapter / rl-eval / opd-eval 没有对应步 ⇒ 要加通用 stage 步才能把 smoke 链接到底。

**待 Chaoyu 裁定**：L1 底题复用作历史 · 切分格 SFT=0 保底 · 六桶份额带宽回填 · 三查门槛表按 v16 首考读数重登记 · "机器词进人话"闸。**未验**：rl-adapter（verl 0.9 model_merger 对 FSDP2+LoRA-only）· sft-eval/select/merge/rl-eval/opd 真数据链。

**坑（本轮新增）**：`--expected-sha` 取 local_gen_sha.json 的 `sha256` 字段；`pkill -f` 再次杀到自己（括号法 `pgrep -f "[s]tack_probe"`）；建库进度中途看不到（tee 进容器本地，整步结束才进 Volume）；未跟踪 AGENTS.md 是 Codex 指令文件。

**Why：** Chaoyu 09-05：v16 全新版本，管线不许有临时调用；机制错改机制、多样性不足加数据；先静静跑完冒烟再动 batch。
**How to apply：** 接手先读 31 → 26 §W4′ 三个新块 → `modal app list`；改建库先本机 `pytest tests/pipeline` + `U_BUILD_DRY=6` + `v16_pipeline.sh supply`，再上云 report 模式。
