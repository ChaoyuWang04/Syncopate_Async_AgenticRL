---
name: u-route-unified-training
description: 主线唯一队首=U 路（OPD+多轮+CoT 三合一，施工图 docs/syncopate/24）；08-29 现场=P2 验收链在跑，接手人从这里继续
metadata: 
  node_type: memory
  type: project
  originSessionId: 57f8d3ca-0848-5dc0-847c-d137e8824067
  modified: 2026-08-29T02:44:33.091Z
---

U 路（2026-08-28 Chaoyu 融合裁定）= 主线唯一队首：OPD 闲聊 + 多轮承接 + CoT 合成一条 v14 管线，「数据合一、监督分家」（NL 段←OPD 教师=裸底座 · 工具段←SFT gold+RL · think 段←8B 蒸馏冷启+RL）。施工图与全部数字在 `docs/syncopate/24`，进度行在 `01-TASKS §U`。

**08-29 02:40 现场**：P0 ✅ P1 ✅（任务分 +0.022 显著、说人话 1.37 超底座；两红旗 L2=36 / acted_on_bad_data+18 已升格为 P2 门槛⑤⑦）；P2 数据 752 行 ✅ + SFT v14_r1 ✅，**验收链在跑**：`scripts/u_p2_accept.sh`（e1/e2 各 4 卡评 → compare vs `_audit/v13_sft_v13r2_e1_merged.json` → 选优 → merge_adapter 合并 → 服务化跑 talk/context 考场 → 判七门槛）。之后 P3 多轮 RL（fully_async，会话级 GRPO+分段 KL 参照）→ P4 抛光 → 终验含 Chaoyu 真人 10 段会话。

**接手人易踩的坑（都付过学费）**：
- eval_parallel 必须显式 `MODEL=`（SFT ckpt 贴 `models/Qwen3-4B` 裸基座；RL adapter 才贴 SFT 合并基座）。
- pkill -f 全面禁用（三次自杀）；杀进程用 pidfile / `nvidia-smi --query-compute-apps=pid` 精确 PID。
- 考场 worker 必须带 `--daily-cost-cap-micros 10000000000`（org_demo 默认 cap 会污染考试，P0 吃过 cand 全灭）。
- OPD 训练器（syncopate/train/opd.py）的 mask 判据是三版校准出来的：只对 `"reply"` 在文本里但没被 mask 的样本报错（仪器坏）；全零 batch 走集合跳步（先 all_reduce mask 数再决定，防 DDP 死锁）。
- 分段器用 [[project-mechanism-not-wired]] 里那类静默死法验过：token 级 BPE 对不齐会让 mask 永远为空，必须用 segment_text（offset_mapping）+ reply 值白名单。
- 无人值守常设纪律：[[blank-thresholds-are-not-passes]]（判据要能对自己失败）、每步过门槛才进下一步、随步更新 24/01、按路径 commit+push。

Chaoyu 常设指令（08-28）：无人值守跑到 v14 OPD 训练完毕；ckpt 只存 adapter；4 卡吃满；每重大节点跑 eval 看任务能力+梯度信息；OPD 指标进 wandb（还欠：编辑 sft/rl 两个 view + 加 OPD view）；本地 MoE 模型与 >1GB torchprof 文件已删。
