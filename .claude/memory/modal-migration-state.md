---
name: modal-migration-state
description: ★09-05 现场：run27 三项红的修法已落地 + 整条管线逐段审查修完（旧名脚本全改 v16_*、v8 审计入口去掉、gates 泄漏闸补上、考场链 candidate 档三处必死修了）；run28 观察模式在云上跑（日志 /tmp/v16/run_build28.log）；唯一入口 docs/syncopate/31；清单在 26 §W4′「09-05 管线审查与修法」
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ed1b9c6-3ae1-544f-b9a5-207777e46c52
  modified: 2026-09-04T08:30:13.697Z
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**（§0 现场 · §4 进度 · §5 怎么起）；施工与判据 `26 §W4′`（每个 run 倒在哪、闸表、stage 表、**09-05 审查与修法块**）；守则 `00 §5`；探针 `modal_app/stack_probe.py`（只调 runbook）。

**09-05 做了什么（commit 4483337，已推 GitHub）**
- Chaoyu 问「runbook 固定了调谁，但被调的还是 v14.5/v15 名字的脚本，旧东西还在起作用吗」⇒ 逐段审查 17 段 + supply（两只只读探查 + 手工复核），见 [[pipeline-callees-must-be-current-too]]。
- 改名（git mv，旧文件留作 legacy）：v16 路径上 14 个旧名脚本 → `v16_build_sft / v16_multiturn / v16_cot_prompt / v16_data_audit / v16_prompt_budget_gate / v16_data_gallery / v16_behavior_think_probe / v16_budget_table / v16_exam_chain.sh / v16_exam_run / v16_exam_judge(_core) / v16_gate_triage / v16_exam_certify`；OPD 渲染/分段进 `syncopate/train/opd_render.py`。判据 `tests/pipeline/test_pipeline_defaults.py::test_runbook_references_no_old_version_scripts`。
- 机制缺口全修：supply 崩（下限提成 v16_build_sft 常量两边 import）· gates 没传 --split-dir（泄漏闸曾被静默跳过）· menus 去掉 v8 审计（Chaoyu 裁定；菜单变 1880 条、切分 SHA 不变）· 考场链 mkdir/多遍判卷/三查只读本次产物（rc 2 报不拦）· 探针 stale 判据空绿 · 缓存标签拆 l2l1/fam/cot · 分词器唯一定义 `model_paths.build_tokenizer_path` · entropy 写死 2048 · L2 val 切片旧数 · select 接受无版本审计 · 行为探针只探 SFT 桶 · 考卷清单 EXAM_FILES 唯一。
- run27 三红修法：派生行终答硬机制（as_multiturn 不再借压舱句，正式报错）+ gen_variant_reply · CLAF 跑题先回放取观测 + gen_fact_reply · WIN 教师现写 + 五类素材扩池按序轮转。
- 本机验证：定向 26 测试绿；全量 pytest 987 过，剩 14 败 1 错全是本机没有 FP8 扩展 / verl 0.9 / prefix_grouper（容器才有）；DRY 演练 exit 0（六族 53 行）；menus/split/gates/supply 全绿。

**正在跑**：run28 = `--steps rebuild_v16,build_v16 --expected-sha <三份 SHA> --build-gates report`。读数：`modal volume get syncopate-home _audit/stack_probe/build_v16.json` 与 `_audit/v16/build.log`；report 产物也拷进 `/vol/_audit/v16/report/`。

**待 Chaoyu 裁定**：L1 底题复用作历史 · 切分格 SFT=0 加保底否 · 六桶份额带宽回填（U_BUILD_BANDS_STRICT）· 三查门槛表按 v16 首考读数重登记。**未验**：rl-adapter（verl 0.9 model_merger 对 FSDP2+LoRA-only）· 真数据 SFT/评测/RL/OPD 链。

**坑（本轮新增）**：`--expected-sha` 要从 local_gen_sha.json 的 `sha256` 字段取（不是 `split` 那组条数）；`pkill -f "<含自己命令行的串>"` 又杀了自己一次（用 `pgrep -f "[s]tack_probe"` 括号法）；未跟踪的 AGENTS.md 是 Codex 指令文件，不是我们的。

**Why：** Chaoyu 09-05：v16 是全新版本（多轮构造、CoT、模型、硬件全换），管线上不许再有任何临时性调用；机制错误改机制、多样性不足加数据。
**How to apply：** 接手先读 31 → 26 §W4′ 09-05 块 → `modal app list`；改建库先本机 `pytest tests/pipeline` + `U_BUILD_DRY=6` + `v16_pipeline.sh supply`，再上云 report 模式。
