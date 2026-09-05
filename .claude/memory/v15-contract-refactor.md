---
name: v15-contract-refactor
description: v15 契约重构与 W0–W5 的历史施工记录；当前状态只看 docs/syncopate/00-START.md 与 01-TASKS.md
metadata:
  node_type: memory
  type: project
  originSessionId: f5bed7da-2659-5ec6-bac9-73049c9ac8d6
  modified: 2026-08-31T08:48:53.627Z
---

> **历史记忆，不是当前队列。** 现行协议看 `docs/syncopate/02-SYSTEM.md` 与
> `docs/syncopate/06-RUNTIME.md`；现行进度只看 `docs/syncopate/01-TASKS.md`。

**v15 = 2026-08-29 Chaoyu 立项的业务大版本**：行为标签从自研终答 JSON 壳搬进 function calling 强通道（session.defer/clarify/reject/report），终答纯自然语言。历史施工图 = `docs/archive/syncopate/pre-consolidation-v16/25-v15-contract.md`；R0 结案（方向裁定确立，无回头闸）· R1/R2 达标 · R3 判分负向认证达标+桥测取消 · R4 ①③④达标。

**★08-31 当时现场：R5 停下，全线转入维修（历史施工图 = `docs/archive/syncopate/pre-consolidation-v16/26-repair-rulers-and-data.md`）**：
- R5 真读数只有一条（L2 53 vs 70）；其余七行全是尺子的病（不可测/不可达/挂错阶段/没跑/噪声内）——「测量系统没建好就开考了」。
- 根因族=守则⑮（训练样例与线上 7+1 处不同形，26 §2.1 全代码确认；含新抓的 L2 switch 分支 context/gold 指向不同对象 bug）。
- CoT 排查结论：**是数据不是 mask**——4049 个 think 块 90 个非空（2.2%）且空块全在监督段；20 行是「行重 2500tok×19% 预算」数学顶死的；24 §2 的「监督按段分家」代码里没实现。8B 蒸馏有效（教师拿全套工具 schema+gold 前缀，拒绝采样只收自己选中 gold 动作的思考），但不覆盖行为/信令步（reject 类 2/5）。
- 修理顺序 = W0 门槛三查（可测/可达/阶段归属）→ W1 考卷 v4（REJ 32+defer/clarify/HARD 档+思考率尺子）→ W2 管线同形（build_messages 加 prior 造真消息对）→ W3 CoT 重设计（think 做轻+触发显性化）→ W4/W5 重建重训。W0–W3 本机 0 GPU。
- ★08-31 Chaoyu 五裁已下（26 §6）：字段清单=训练也不给 · 菜单=训练改全量34（超预算则精简工具描述）· 时间=训练改纯日期 · CoT 带宽上沿=30%（天花板34%不顶满）· 思考率=SFT 只记录、≥50% 硬闸挂 R6。唯一待批=W0 修订门槛表。**W0 可立即开工。**
- Chaoyu 已裁：不留 r1–r3 失败 ckpt；产物在训练机（`v15_r3/sel_f2.5`，本机没有）。

保留资产：S1–S3 库、OOV held-out、盲评口径、OPD prompts、pool/守卫、四卡 DDP、dev mode 栈。[[u-route-unified-training]] 的 P3/P4 并入 R6/R5；[[train-data-must-match-production-shape]] 是本轮总根因。

**★09-02 接手核对（⚠️ 未经前任确认：ListAgents 只有 oasis-dit-37 在线，回信"不是"；前任会话已下线）**
本机实况（samwang-X870I，单张 5090，**不是** 4×5090 训练机）：无 /workspace/.env；**09-02 起 K 线常驻了本机 PG 5432 + Redis 6379**（DSN postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate，别 --reset；.venv 有 asyncpg）、
data/sft/v15 只有 manifest.json（无 parquet）、checkpoints 无 v15；本机有 v15_cot_rows.json(114) /
v15_l2l1_rows.json / _audit/v15_r2/gates.json / logs/u_route/run_v15r3c_r1..4 + judged。
本机查实的四个文档外要点：① 考场原始记录 run_*.jsonl 的 turns **不含 events**（只有 behavior/reply/tools…），
思考率的"1 条非空"来自训练机 PG 的 run_events ⇒ W1④ 校准在本机做不了，除非改从 jsonl 里取 thinking；
② decider.build_messages **已有** prior 参数 + SYNCOPATE_PRIOR_INLINE（叫停实验遗留），训练侧 rollout_loop.build_messages 没有 ⇒ W2① 是对齐这份形状；
③ contract.py:117 **已有** defer 人话正则（W1② 那句"实施时验证"答案=有）；
④ 26 引用的行号 08-31 至今仍准（u_make_exams_v3:44 range(8) · u_exam_judge:145 fails[:20] · v16_build_sft:1139 cot (0.05,0.20) · decider:113 注释仍写 30）。
00 §5 ⑬ 尚无"三查附则"（W0 要加）。
历史上与 K 线约定：W2⑤ 若精简 tool_registry 工具描述，需要先跨线核对。当前不再写 MAINLINE-INFRA；使用 Codex 独立任务消息工具，并把未完成事项登记到唯一负责方 TASKS。decider.build_messages 的 prior/PRIOR_INLINE 渲染归 W2，K 线不碰。

**★09-02 W0 完成（Chaoyu 已放行施工；修订门槛表待批）**：产物 `syncopate/evaluation/gate_triage.py`（三查机器出表，
`--legacy` 负向认证报 7 缺口）+ `tests/train/test_v16_gate_triage.py` + 25/26/00/01 就地改写。W0 查出 08-31 漏的四条：
盲评绝对线→≥1.46−MDE(0.18)·N1 正则无装置·方差闸 8pp 永远过不了→SE 口径·cap 恶化无阈值→泊松 2√n。
R7③ 1pp/2pp 撤销改 −MDE。待裁一条：R7④ 差值 MDE≈14pp 粗。**下一步 W1**（含 ⑦ N1 正则/cap 全表/思考率落 jsonl，⑧ --strict 零缺口）。
Chaoyu 09-02 追加要求：数据多样性 + 训练/生产同形要做成**可反复跑的脚本 + 量化指标**（W2⑥/W4 出厂体检承接）。
**★09-02 晚 W1 完成**：考卷 v4 361 题（六族第一波：REJ32/DEF/CLA/HARD/DEF-F/REJ-F/CLA-F/L2-x/WIN + 报告项 META/PRG/COR/TIME）、
judge_v4 13 判类负向认证、脚本化历史（直接插 agent_runs 终态行，与线上同一读取路径）、`--strict` 零缺口。
runtime Ⓐ/Ⓑ 已落地（clarify→waiting_for_user、reject 轮进历史）；APR-F 欠账。**下一步 W2**（rollout_loop.build_messages 加 prior、
L2/L1 生成器重写、字段清单/时间/菜单三裁定、同形断言测试、新科目训练行）。K 线（a8）在同机改 db/api/worker/action_gate/gateway，我避开。
**★09-02 深夜 W2 基本完成**：历史消息对共用渲染（core/prior_turns）、build_messages 加 prior、纯日期、chat 行改同一渲染路径（原走 probe 私渲染 = 四处不同形）、
六族第一波训练行（syncopate/pipeline/multiturn.py）、DRY 演练 146 行零不同形；34 条工具描述修剪（6534→5468 tok，硬事实核对不丢）。
**待 Chaoyu 定数：MAX_PROMPT_LENGTH 5760→9216 / 服务 max_model_len 14336→18432**（全量菜单最长 prompt 7167）。本机已有 batches/v13（影子重建）与 4B/0.6B tokenizer。
K 线通报：v15 契约下 runtime 导入触发治理表断言（session.* 未登记）。下一步 W3（CoT 做轻/触发显性化/行为类 think 探针）。
**★09-02 收尾 W3**：think 做轻闸（350 tok/350 字/2 段）、触发显性化 explicit_hard_prompt（探针族内 65.5%→88.5%）、预算表（30% 可装 ≈70 行 CoT）、
行为类 think 探针脚本待训练机。**阻塞项只有一个：MAX_PROMPT_LENGTH/max_model_len 定数**；定了就写 rollout_budget + 启动脚本，然后 W4（训练机）：
重建（含 ballast_replies 补 6 个源 case、v15_r2_migrate 四项全等、份额带宽回填、出厂同形体检）→ W5 五点谱 + 考卷 v4 四遍（~1.8h）+ 思考率尺子校准（第一遍对照 PG）。
**★09-02 夜 Chaoyu 三裁（已落地，全量 908 passed）**：① 空 think 块**不监督**（初衷=只采难题的高质量思考，不是教"输出空思考"；sft_replay._mask_empty_think，空块留位置对齐、mask 置 0；简单集 ≤10% 降报告项）② **不缩短 CoT**（W3① 撤回；比例问题由 mask 解决；30% 带宽可装 ≈42 行）③ 上限 **9216 / 18432**（不爆显存就抬到线上真实形状；start_vllm.sh、exam_chain、decider 默认同步改）。阻塞项清零 ⇒ 下一步 W4 训练机（26 §W4 清单 7 步）。
**★09-02 末：全链路设定一览在 26 §4.5**（每个数一份来源+消费者表）；菜单策略 contract.effective_tool_menu（v15 一律全量，含压舱/RL）；训练数据画廊 syncopate/pipeline/data_gallery.py（⟦⟧ 标监督 token）本机产物 _audit/v15_w2/gallery_dry.md。本机可做的全部完成，下一步 W4 训练机。
**★09-02 画廊复核（Chaoyu 逐条看）抓到 4 条并修**：闲聊行空块有梯度 · 压舱行列字段清单 · context 塞 campaign 清单（裁定⑥：只带 account_id）· WIN 行窗口没裁（6 轮窗口下沉到 render_prior_messages）+ 历史"好的。"占位。教训：**数据要逐条渲染给人看**，画廊元信息行就是判据。
**★09-02 裁定⑨ 运行态注入（已落地）**：模型只装知识与策略，不装运行态身份。account_id 不进 prompt/工具 schema/gold，沙盒 registry.execute 与线上 ActionGate 按租户注入并**覆盖模型值**；contract.RUNTIME_INJECTED_PARAMS 一处定义。Chaoyu 原话："这一版就是最终版，改好为止"——不许把问题拖到 v16。
**历史 09-03 算力裁定已被后续迁移覆盖**：当时曾计划用 RTX PRO 6000×2、只让 B200 做探针；当前实际主场已经是 Modal 2×B200，模型与栈只看 `docs/syncopate/05-COMPUTE.md`，infra 新队列只看 `docs/infra_exp/01-TASKS.md`。
**★09-03 接手核对（已经前任 -22 确认）**：本机 Modal 零基础（无 SDK/token/镜像脚本，从零写 `modal/image.py`，可与 infra 会话 -77 共用基础层；账号找 Chaoyu）。
**教师与学生都没裁定**：教师候选 Qwen3.5-27B/3.8-27B，先在 sm_120 量 GDN/MTP，量不过沿用 8B@8211；W3③ ≥70% 探针对真正采 CoT 的那个教师跑。学生**试点②可比性必须用 Qwen3-4B**（与 v15_r3 同模型），换 8B/9B 是之后另开一臂、要 Chaoyu 拍板。
对照读数本机够用：`_audit/v15_r5/r3_*.json` 五点谱 + `logs/u_route/run_v15r3c_r1..4_context_v3.jsonl`+judged；v4 L1–L4 按层可比、REJ 分列。Modal 上缺 data/sft/v13 parquet 与 v13 批次（上传或按 run_pipeline_shadow_rebuild 0–4 步重建，切分 SHA 对齐 data/splits/v13；migrate 先 `--scope all`）。
带宽探针 = `scripts/infra/probe_alignment_cliff.py` + `analyze_allgather_alignment.py`（run_batch3_gpu.sh 有示例）；⚠️ 08 §5 记着 4×5090 上 NCCL_P2P_DISABLE=1 实测无效、真解是 NCCL_CUMEM_ENABLE=0，两条都试。
文档外八坑：建库产物 grep "[DRY" 必须 0 · 两份旧缓存改名（W4③）· seed_demo_data --check 7 条 · v16_exam_run.seed_prior 直接 INSERT 别改走 create_run · 考场链 logs/v15_r5/ 目录名写死自己 mkdir · DATA_VERSION 仍 v13 别改 split.py · 起任何东西前 SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 · Modal 先跑 pytest 对 908 基线。`_audit/v15_w2/*_dry*.parquet` 可删。
