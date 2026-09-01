---
name: v15-contract-refactor
description: ★唯一队首=v15 契约重构；08-31 起停在 R5 复盘 → 维修施工图 docs/syncopate/26（先修尺子再修数据后重训）
metadata:
  node_type: memory
  type: project
  originSessionId: f5bed7da-2659-5ec6-bac9-73049c9ac8d6
  modified: 2026-08-31T08:48:53.627Z
---

**v15 = 2026-08-29 Chaoyu 立项的业务大版本**：行为标签从自研终答 JSON 壳搬进 function calling 强通道（session.defer/clarify/reject/report），终答纯自然语言。施工图 = `docs/syncopate/25`；R0 结案（方向裁定确立，无回头闸）· R1/R2 达标 · R3 判分负向认证达标+桥测取消 · R4 ①③④达标。

**★08-31 现场：R5 停下，全线转入维修（施工图 = `docs/syncopate/26`）**：
- R5 真读数只有一条（L2 53 vs 70）；其余七行全是尺子的病（不可测/不可达/挂错阶段/没跑/噪声内）——「测量系统没建好就开考了」。
- 根因族=守则⑮（训练样例与线上 7+1 处不同形，26 §2.1 全代码确认；含新抓的 L2 switch 分支 context/gold 指向不同对象 bug）。
- CoT 排查结论：**是数据不是 mask**——4049 个 think 块 90 个非空（2.2%）且空块全在监督段；20 行是「行重 2500tok×19% 预算」数学顶死的；24 §2 的「监督按段分家」代码里没实现。8B 蒸馏有效（教师拿全套工具 schema+gold 前缀，拒绝采样只收自己选中 gold 动作的思考），但不覆盖行为/信令步（reject 类 2/5）。
- 修理顺序 = W0 门槛三查（可测/可达/阶段归属）→ W1 考卷 v4（REJ 32+defer/clarify/HARD 档+思考率尺子）→ W2 管线同形（build_messages 加 prior 造真消息对）→ W3 CoT 重设计（think 做轻+触发显性化）→ W4/W5 重建重训。W0–W3 本机 0 GPU。
- ★08-31 Chaoyu 五裁已下（26 §6）：字段清单=训练也不给 · 菜单=训练改全量34（超预算则精简工具描述）· 时间=训练改纯日期 · CoT 带宽上沿=30%（天花板34%不顶满）· 思考率=SFT 只记录、≥50% 硬闸挂 R6。唯一待批=W0 修订门槛表。**W0 可立即开工。**
- Chaoyu 已裁：不留 r1–r3 失败 ckpt；产物在训练机（`v15_r3/sel_f2.5`，本机没有）。

保留资产：S1–S3 库、OOV held-out、盲评口径、OPD prompts、pool/守卫、四卡 DDP、dev mode 栈。[[u-route-unified-training]] 的 P3/P4 并入 R6/R5；[[train-data-must-match-production-shape]] 是本轮总根因。
