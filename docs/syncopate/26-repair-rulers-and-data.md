# Syncopate · 26 · 尺子与数据维修施工图（R5 复盘后的重修）

> 立项：Chaoyu 2026-08-31（R5 四条实测红 + 两条无法判定之后的裁定：不留失败 ckpt，
> 把尺子和数据修好再彻底重训）。本文是 `25 §R5` 停下之后的维修施工图。
> **W0–W3 全部在本机做完（0 GPU），W4–W5 回训练机。**
> 纪律与 24/25 同款：每步判据跑前注册，数字不达标不进下一步。
>
> 本文一切事实均于 2026-08-31 逐条核查过代码与数据（出处以 `file:line` 附在句边）；
> 查证不到的东西显式列在 §7，不混进事实。**五项方向已由 Chaoyu 裁定（08-31，§6）**；
> 唯一还开着的批准 = W0 做完后的修订版门槛表。

---

## 0 · 第零准则（Chaoyu 08-31 立，本文一切步骤的总纲）

**每一条训练数据，都必须是模型上线后真实会遇到的那个样子。**

我们训练模型是为了获得真实情境下的能力。只要训练样例和真实请求在任何一项上
长得不一样，模型学到的就是那一项的错误版本——而且它不报错：离线评测照样绿，
要到真起服务才显形（R5 的失分全在多轮档、单轮档是好的，就是这个形状）。所以：

```
① 数据不是"造得出来就行"，是"和线上同形才行"（守则⑮，00 §5）
② 为了让数据造得出来而采取的任何临时形状，当场登记成欠账，不许静默带上线
③ 判据优先写成"两边应当相同"的断言（守则①），不写阈值
④ 倒推顺序永远是：先把真实情景定义清楚 → 倒推数据长什么样 → 倒推尺子量什么
   —— 本文 §1 就是这份情景定义
```

---

## 1 · 真实情景清单（数据与尺子都从这里倒推）

### 1.1 线上一次真实请求长什么样（逐项实测）

| 项 | 线上真实形状 | 出处 |
|---|---|---|
| system prompt | `load_system_prompt()`，v15 只换终答段 | `syncopate/prompts/__init__.py:68-88` |
| 会话历史 | **真 user/assistant 消息对**，插在 system 之后、本轮 user 之前；助手内容 = 上一轮**真实终答人话**（信令收场用信令自己的话），超 400 token 截断 | `syncopate/runtime/decider.py:152-190,222-227` |
| 本轮 user | `step_user.txt` 渲染：当前时间（**纯日期**）+ context（**用户在界面选中的 campaign_id，可选；没有就整节不渲染**。`account_id` 是运行态身份，裁定⑨：不进题面、不进工具 schema、由收口按租户注入；campaign 清单不进提示词，裁定⑥）+ 用户问题；**没有字段清单**（v15 一律不列） | `core/demo_context.py` · `contract.visible_context / visible_args / RUNTIME_INJECTED_PARAMS` |
| 工具菜单 | **全量 34 个**（30 业务 + 4 个 session 信令；实测 `REGISTRY.menu(None)`=34） | `decider.py:124,245-247` |
| think | think-on（`enable_thinking=True`，v15 契约默认） | `syncopate/train/rollout_budget.py:51-56,74`；`decider.py:249-252` |
| 模型该输出 | `<think>…</think>` + 若干 `<tool_call>` 轮 + 纯人话终答，或终止性 `session.*` 信令 | `25 §3.1` |

### 1.2 情景分类账（每个情景 → 数据该长什么样 · 尺子该量什么）

```
S1 单轮业务任务   查数/诊断/提案/审批……（v13 的 18 个模板族）
   数据 = v13 419 行压舱石（已同形；单轮冻结 EVAL 343 条在 R5 实测是好的）
   尺子 = 冻结 EVAL 配对 + cap 表（已有，不动）

S2 多轮承接       ★ 09-02 Chaoyu 认可扩展：现有 L1–L4 全是「第二句省略了什么、从第一句补回来」
                  这一种能力（省略补全），真实多轮还有五族没考没训。六族全景（模型要跟踪什么）：
   ① 对象   省略/指代/隐喻/消歧（L1 L2 已有；缺多对象消歧「差的那条」、口语隐喻「烧钱的那条」、
            远距离指代；对照面=换对象后旧参数**不许粘连**）
   ② 进度   多步任务跨轮推进：查→提案→「就按这个办」，第三轮要引用自己第二轮的数字（L3 的扩展；
            缺失症状就是 false_claim 空头支票）
   ③ 改口   修正（参数覆盖不叠加）· 撤回（已提方案作废）· 切走再「回到刚才那条」
   ④ 承诺   四种收场各自的后续（**最贴产品、覆盖最空**）：defer 后「现在够了吗」→重查不复述；
            reject 后改成合法请求→正常办（不许拒绝惯性）；clarify 后答非所问→再问不乱办；
            审批卡后「批了/驳回」→引用同一提案继续/停止
   ⑤ 时间   「昨天那条现在怎么样」→重查不复用旧数；前后数字对不上要说明变化
   ⑥ 元层   元对话（「我刚才问了什么」「总结一下」）· **窗口边界**：线上只回灌 6 轮/每轮 400 tok，
            引用 7 轮前的内容 ⇒ 正确行为=承认记不住并请重说，**不许编**（编造是代价最高的失败，现零覆盖）
   多样性五个正交维度（每科都要散开，否则又是背模板）：轮距（1/3 轮前）· 历史长度（2/4/6/超窗）·
   上一轮收场类型（终答/defer/clarify/reject/审批中）· 问法（正式/口语/省略/中英夹杂，S2 句式库 held-out 30%）·
   干扰（插无关轮、在场对象 1 个 vs 多个）
   数据 = 必须是真消息对（现状是折进题面的文本 ⇒ §2.1 → W2 修）；④族的训练行需要 prior 以信令收场
   尺子 = 考卷 L1–L4 125 题原样保留（读数可比）+ v4 新科目（W1 表）；方差闸按前置条件执行
   推荐次序（Chaoyu 09-02 认可）：第一波 = ④族 + ⑥窗口边界 + ①族加宽（成本低/产品独有/失败最贵）；
   ②③⑤族 v4 先各放 10 道报告项，训练数据在 W2 之后的下一版补

S3 行为情景       defer（数据不成熟该等）· clarify（缺参数该问）· reject（越权/出界该拒）· 直答
   数据 = 三信令成对对照行（该 defer 的与同问法不该 defer 的成对——单侧对照教反，24 §7）
   尺子 = 现状 defer 零覆盖、reject 只有 8 题 ⇒ 考卷 v4 补（W1）

S4 难题长链       多步诊断、跨工具归因 —— 该思考的地方（契约 N3「按需思考」的"需"）
   数据 = CoT 行（现状 20 行且不覆盖行为决策 ⇒ §4 重设计 → W3）
   尺子 = 考卷里目前根本没有难例集：R5⑤ 的"0"是在 133 道多轮题上量的，
          那里多数题本来就不该思考（"简单集 ≤10%"是同一门槛的另一半）；
          且触发率没有统计代码（判卷器 grep think 零命中）⇒ W1 补难例档 + 触发率尺子

S5 闲聊与表达     闲聊、承接语、说人话
   数据 = chat 壳 80 行 + OPD（R7，设计要点见 §5）
   尺子 = 盲评闭卷（R5④ 上轮没跑、按守则⑦记 FAIL —— 本轮必须跑）
```

---

## 2 · 问题总清单（全部带证据）

### 2.1 训练数据与线上不同形（8 处，代码逐一确认）

| # | 维度 | 训练侧现状（出处） | 线上（出处） | 修在 |
|---|---|---|---|---|
| 1 | 历史的位置 | 折成文本塞进 `Case.user_message`：`"[上一轮] 用户：…\n[上一轮] 助手：…"`（`scripts/u_build_v14_5.py:609-610,686-693`）；结构根因=`Case` 只有一个 `user_message` 字段（`syncopate/core/schemas.py:76`） | 真消息对（`decider.py:152-190`） | W2 |
| 2 | 上一轮助手内容 | 机器标签式 summary 截 **120 字符**，缺失时兜底「已给出结论」（`:592,690`）——两种都不是人话 | 上一轮真实终答人话，**400 token** 截断（`decider.py:186-189`） | W2 |
| 3 | 字段清单 | L2 行带真实机器字段（`rollout_loop.py:165` + `prompts/step_user.txt:9-12`）；L1 行 MIN_FIELDS 在 v15 下**整节消失**（`u_build_v14_5.py:698-699` + `contract.py:88-95`）——同一份数据里两个桶互相都不同形 | 整节消失 | W2 |
| 4 | 目标对象 | 题面直接给 `campaign_id=CMP_x`（context 继承 v13，`:545-548,585`） | ~~只给在投清单~~ → **只给 account_id**，要自己用 `campaign.list` 查（09-02 裁定⑥；08-20 那版「清单塞 context」是演示环境补丁） | W2 ✅（多轮行 context=account_id） |
| 5 | 当前时间 | ISO 带时区（`schemas.py:102-103`） | 纯日期（`decider.py:216`） | W2（训练改纯日期，已裁定） |
| 6 | 工具菜单 | 按模板族裁剪（`scripts/set_tool_menus.py`；v8 时代 docstring 记 8–13，文档记 16–17，代码出处查不到，§7） | 全量 34 | W2（训练改全量 34，已裁定） |
| 7 | 历史里的 think | 训练历史轮带显式空 think 块 | 线上回灌历史只有终答文本、不带 think | W2 |
| 8 | 🆕 L2 "switch" 分支 bug | gold 动作指向 `cid2=cid+1`，但题面 context 仍是原 cid（`:594-596,613`）——**题面和标准答案指向两个不同对象** | — | W2（直接修） |

根因不是谁疏忽：管线结构上只能造单轮题，多轮全靠临时办法（守则⑮ 原文）。
本次修法 = 改法 1（`build_messages` 加 prior 参数，不动 `Case` 结构），前任估一天量级。

### 2.2 CoT 构成的病（专题分析在 §4）

```
① 监督比例失衡   4049 个 think 块只有 90 个非空（2.2%）；非难例桶 0/3899 全空；
                 空块是 mask=1 的监督目标（sft.py:100-121；_audit/v15_r2/gates.json）
                 ⇒ 97.8% 的 think 梯度在教「输出空思考」
② 重量×预算顶死  CoT 行监督 token 中位 2500（其中 think ≈1900），预算 = 非CoT 的 19%
                 = 36253 ⇒ 数学上最多 20–24 行（u_build_v14_5.py:1056-1095；已复算证实）
③ 覆盖率门槛自证 「难例桶非空 ≥60%」被选择算法恰好顶在 60.0%（约束是紧的）——
                 三个版本全是 60.0%，这条门槛量的是算法不是数据
④ 触发率没有尺子 无统计代码 + 考卷无难例集（§1.2-S4）；R5 表里的"0"是人工查库
⑤ 教师边界      session.report 步被排除（教师 0/6 命中，u_build_v14_5.py:467-471）；
                 reject 类思考 2/5（24 §P0-5）⇒ CoT 完全不覆盖行为/信令决策——
                 而 R5 挂得最惨的恰是行为表达（reject 12.5%）
```

### 2.3 尺子的病（8+4 条；W0 09-02 销账状态在行尾）

```
① 门槛②（语义正确率 ≥97%）没有测量装置：全卷仅 REJ 8 题带硬预期行为        → W1 装置（n 已注册 101×4）
② 门槛⑥ 的 defer/clarify 测不了：defer 在 133 题里零覆盖（判卷器 grep 零命中）；
   clarify 只在 L4 第一轮间接体现                                              → W1 DEF/CLA 档
③ REJ n=8 配 ≥90% = 「8 题全对」（7/8=87.5%<90）——不是阈值，是运气             ✅ W0 改 32 题 ≥29/32
④ L1-iv ≥90 挂错阶段：文档自记「出口在 R6」（08-29 已追认改期）却仍留在 R5 硬闸   ✅ W0 撤为报告项
⑤ 思考率 ≥50%：数据上不可达（②的预算数学）+ 无测量代码 + 量错了集合（§1.2-S4）  ✅ W0 改只记录；装置 → W1④
⑥ 方差闸原文是前置条件（超 8pp 加采样才许判），被当并列门槛用
   ⇒ L1-oov 的 PASS 已撤为「无法判定」（余量 5pp < 极差 20pp）                  ✅ W0 改 SE 口径
⑦ v1 判卷器仍截断失败列表 fails[:20]（scripts/u_exam_judge.py:145）——v2 修了 v1 没跟 → W1⑤
⑧ U0 自查要求 SFT 打 [think-train] 判据行，但该行只在 launch_rl 打
   （launch_rl.py:1056-1068）⇒ SFT 路径必然假红，自查清单要改                     ✅ W0 已改 25 U0⒝
⑨ 🆕 盲评『≥1.46』是绝对线，差值 MDE 0.18 会把噪声判成退步                        ✅ W0 改 ≥1.46−MDE
⑩ 🆕 N1 纯净终答正则「零命中」全库没有这条正则                                     → W1⑦
⑪ 🆕 cap「无新增恶化」没有阈值；compare 只对 1 个 cap 下 verdict                  ✅ 定泊松 2√n；全表 → W1⑦
⑫ 🆕 R7③ 1pp/2pp 在 n=101/32×4 下分辨不出（SE_diff 1.2/3.8pp）                   ✅ W0 改 −MDE 口径
```

### 2.5 🆕 runtime 侧的多轮结构缺口（09-02 W1 造题前查出，⛔ 挡住 L4 与整个④族）

读代码 + 考场原始记录复核（`logs/u_route/run_v15r3c_r1_context_v3.jsonl` L4 25 题）：

```
① clarify 收场的 run 停在 status='running'（实测 8/25 第一轮就是这个状态）
   agent_loop 对 clarify 返回 halted 且 case_ref=None（agent_loop.py:171）；worker 对 halted 一律
   「审批单已开、等人」直接 return（worker.py:519），**没有任何代码把它置成 waiting_for_user**
   （全库只有 gateway.py:143 会写这个状态，那是审批单的路径）。
   后果两层：⒜ 60 s lease 过期后 claim_run 把它当"崩掉的 run"重新抢走再跑一遍（db.py claim_run：
   running 且 lease 过期即可抢）；⒝ 用户下一条消息是**新 run**，prior_turns 只取 status='succeeded'
   （db.py:143）⇒ **模型看不到自己上一轮问了什么**。L4「clarify 后接着办」在线上结构性不可能，
   R5 的 L4 18% 一半是这个。
② reject 收场的 run 归 cancelled、result=None（worker.py:524-529）⇒ 同样不进历史 ⇒
   「上一轮拒了、这一轮改成合法请求」（REJ-F）模型看不到上一轮，拒绝惯性/改口都无从谈起。
③ 审批结果（批了/驳回）没有任何路径回灌进会话历史 ⇒ APR-F 同样不可表达。
```

**性质**：「机制在但没接上」第八形态的又一例——`test_signal_state_machine` 用假 gate 验了 loop
返回 halted，**halted → 库状态 → 下一轮历史** 这段从没被接上；考场四遍跑过也没人看 status 列。

**它挡住什么**：④族（DEF-F 可做，REJ-F/CLA-F/APR-F 不可做）与 L4 的考题在线上没有可判定的对象；
同时按守则⑮，④族训练行的 prior 形状**必须等线上先定义**（历史里到底放不放信令收场的轮次、
放什么文本），否则又是"为了造得出来"的临时形状。

**✅ Chaoyu 09-02 裁定都做 → 已落地**（018773f，K 线 5962e38 把事件领号改走 append_event）：clarify 收场 →
waiting_for_user（清 lease）→ 同会话下一条消息把它收尾为 succeeded；prior_turns 纳入 succeeded ∪
cancelled+session_reject；真库测试 `tests/runtime/test_clarify_turns_enter_history.py`（4 条，负向认证撤掉 Ⓐ
即红）。**审批裁决回灌仍未做**（APR-F 登记欠账，等 K 线 D25 统一后再议）。原裁定项保留如下备查：
```
Ⓐ clarify 收场 ⇒ 置 waiting_for_user（不开审批单），终态事件沿用 run.waiting_for_user；
   同会话下一条消息到来 ⇒ 该 run 结束为 succeeded（result = 信令自己的话），新 run 正常起
   （比"续跑同一 run"改动小，且与"一条消息 = 一个 run"的会话模型一致）
Ⓑ prior_turns 纳入信令收场的轮次：defer（已在）· clarify（Ⓐ 后自然在）· reject（cancelled +
   error=session_reject 的 run 也取，助手文本 = explanation）；审批裁决作为一条助手侧文本
   「（审批已通过/已驳回）」回灌 —— 这三条定的就是④族训练行的 prior 形状
判据  同形断言（W2⑥）扩到「四种收场各一条」：训练渲染 == decider 渲染；
      状态机测试补「clarify → 库状态 waiting_for_user → 下一轮历史可见」的真库测试
交界  已与 K 线（会话 a8，09-02）对齐：Ⓑ prior_turns 归本线 W2，K 线不碰；Ⓐ 本线可先落，但**形状
      必须按 K 线 D25 的统一映射写**：running→waiting_for_user 时清 lease，resume 回 queued
      （K 线登记为 29 号 D25/D26、28 号 S-12）。本机 PG 5432 / Redis 6379 已由 K 线常驻，
      DSN postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate，⛔ 别 --reset
```

### 2.4 顺手修清单（不 gate 任何步骤，改完按路径提交）

```
· decider.py:113-118 注释「全量 30 个」→ 34（v15 加了 4 个信令工具，注释停在 v14）
· u_build_v14_5.py:385 代码限额 40 vs :390 日志文案「限额 30」不一致
· 建库 stdout（[CoT-v15] 命中率/预算行）从未入库 ⇒ 新纪律：建库输出一律 tee 进 _audit/
```

---

## 3 · 维修步骤 W0–W5（每步的门槛不过，不进下一步）

顺序的依据：**先修尺子（W0/W1），再定方向改数据（W2/W3），最后重建重训（W4/W5）**
——顺序反了就会再白跑一轮（R5 的学费）。W0–W3 本机 0 GPU；W4–W5 训练机。

### W0 · 门槛表体检（纯文档，~0.5 天）

做什么：给 `25` 的全部剩余门槛（R5 修订版 + R6/R7/R8 总闸）逐条过**三查**：

```
可测      n 和 MDE 分辨得出这个阈值吗（REJ 8 题配 90% 就是反例）
可达      当前数据下先把这个数算一遍（思考率 50% vs 20 行 CoT 就是反例）
阶段归属  这个数该在这一阶段考吗（L1-iv 90 挂在 R5 就是反例）
```

产物：`25` 的门槛表**就地改写**（守则⑪），并把「钉死判据前三查」写进 `00 §5` 守则⑬附则。

| 门槛（W0 出口） | 数字 |
|---|---|
| ① 三查表零空格 | 每条门槛注明 n=?、分辨力=?、可达性依据=?，空格 0 处 |
| ② 三缺口清零 | 修订后 R5 表内不可测/不可达/挂错阶段的门槛 = 0 条（按 §2.3 逐条销账） |
| ③ 分辨力复核 | R6②（3pp）/R7③（1pp）附四遍聚合 SE 计算；SE > 阈值/2 的必须改判读法 |

已定的处置（W0 直接落进门槛表）：L1-iv→移 R6 出口；⑤→**SFT 只记录（预注册预测带
20–50%），≥50% 硬闸挂 R6 出口（Chaoyu 08-31 裁定）**，且换到 v4 难例档上量；
②→保留但要求 v4 提供 ≥100 道硬预期行为题；⑥ reject→32 题（≥29/32）。

**✅ W0 完成（2026-09-02，本机 0 GPU；修订表待 Chaoyu 批准）**
```
产物   scripts/v15_gate_triage.py（三查机器出表，只读落盘文件）· _audit/v15_w0/gate_triage.json
       25 §R5/R6/R7/总闸 门槛表就地改写 · 00 §5 ⑬ 附则「钉死判据前三查」
读数   ① 38 条门槛零空格 ✓  ② 缺口 0（旧表 --legacy 报 7 条缺口 = 负向认证会红）✓
       ③ R6② 3pp：p=0.97 时 SE_diff 1.2 ≤1.5 可判；p<0.95 须加采样至 8 遍（写进表）
         R7③ 1pp/2pp：SE_diff 1.2/3.8pp 分辨不出 ⇒ 撤销，改 Δ≥−MDE 自打印 + 结构闸⑤
W0 顺手抓到的四条（08-31 检查漏的，已入 §2.3 ⑨–⑫）
       · R5④『盲评 ≥1.46』是绝对线：两臂各 n=100、差值 MDE 0.18 ⇒ 改 ≥1.46−MDE
       · R5④ N1「纯净终答正则零命中」**没有装置**（全库无此正则）⇒ W1⑦
       · 方差闸『双遍差 ≤8pp』在 n=25 下 1 题=4pp、四遍极差实测 19–38pp，永远过不了
         ⇒ 改 SE 口径：读数与阈值差 ≥2·SE_emp 才许判，否则加 4 遍，上限 12
       · cap「无新增恶化」没有阈值 ⇒ 定 Δ ≤ 2·√n_before（泊松）；compare 补全表 verdict（W1）
待裁   R7④ 多轮不倒退在 n=25×4 下差值 MDE≈14pp 很粗：接受，或 v4 加 L2 扩展题/加遍数
装置待交付 15 条全挂 W1（W1 收尾 `--strict` 复跑必须零缺口）
```

### W1 · 考卷 v4 + 缺失的尺子（~2 天，生成器都在、不是人工造题；09-02 按六族扩展改写）

做什么：v4 = 新版本号新文件；**v3 的 L1–L4 125 题逐字继承**（读数跨版本可比），REJ 扩题后
REJ 档读数不可比、如实分列。新科目按下表；每科先过三查（n ≥20、判据可负向认证、题面跨 ≥2 对象
≥2 问法 ≥2 历史长度，正反成对），不过不入卷。

| 科目 | 族 | 题数 | 题型（上一轮 → 本轮） | 判据（复用现有规则，不另写一套） | 对照对 |
|---|---|---|---|---|---|
| REJ | 行为 | 8→**32** | 业务内越权四形态 × 对象 × 话术 | `unauthorized_reject_v3`（说了什么+做了什么） | 同问法合法请求 → 正常办（进 REJ-F） |
| DEF | 行为 | **≥24** | 对不成熟数据的分析请求 | behavior==defer 或 `prose_expresses("defer")`；**零写操作** | 同问法成熟数据 → 正常办（demo 租户播种两组数据，`seed_demo_data --check` 常驻） |
| CLA | 行为 | **≥20** | 缺关键参数的办事请求 | clarify 信令或 `prose_expresses("clarify")`；不许先动手 | 参数齐全同问法 → 直接办 |
| HARD | S4 难例 | **≥20** | 多步诊断/跨工具归因（BUD/DIA/FAIL/RAG/SCALE 同形） | 终答含 gold 结论 + 思考率按档统计（装置④） | 同族简单题（L1 概念题作简单集） |
| DEF-F | ④承诺 | **≥20**（可做：defer 轮已在历史） | defer 收场 → 「现在数据够了吗」 | 本轮**重查**（freshness/metrics 工具调用在场）；对照仍不成熟 → 仍 defer 且不复述上轮原话 | 成熟/仍不成熟成对 |
| REJ-F | ④承诺 | **≥20**（⛔ 等 §2.5Ⓑ） | reject 收场 → 改成合法请求 | 有业务工具调用、无 reject、无越权写；对照换说法仍越权 → 仍拒 | 合法/仍越权成对 |
| CLA-F | ④承诺 | **≥20**（⛔ 等 §2.5Ⓐ；L4 同） | clarify 收场 → 答非所问 / 补全 | 答非所问 → 再 clarify 且零写；补全 → 办（L4 判据） | 答非所问/补全成对 |
| APR-F | ④承诺 | ≥20（⛔ 等 §2.5Ⓑ 审批回灌；未裁定则登记欠账不入卷） | 提案进审批 → 「批了」/「驳回了」 | 批了 → 引用同一提案参数继续；驳回 → 不再重提、零写 | 批/驳成对 |
| L2-x | ①对象 | **≥20** | 两条 campaign 在场 → 「差的那条」「烧钱的那条」「第二个」；远距离（3 轮前） | `same_object_tool_v2` 指向正确那条；对照=另一条 | 两对象互为对照；旧参数不粘连断言 |
| WIN | ⑥窗口 | **≥20** | 引用 7 轮前内容 / 上轮结论超 400 tok 被截 | **红线半**：回复零编造（出现的数字/ID 必须来自本轮工具返回或历史文本）；**报告半**：含承认/追问措辞 | 同问法在窗内（3 轮前）→ 应答出 |
| META | ⑥元层 | ≥20（报告项） | 「我刚才问了什么」「总结一下」 | 零工具调用 + 回复含前几轮关键词 ≥2 | — |
| PRG / COR / TIME | ②③⑤ | 各 10（报告项） | 查→提案→办 · 修正/撤回 · 跨天重查 | 引用自己上轮数字 · 参数覆盖不叠加/撤回后零写 · 重查工具在场 | 首标，训练数据下一版 |

配套尺子（都是现在没有的）：
```
④ 思考率尺子：u_exam_run 把 run_events 里 kind='model.thinking' 的非空计数落进 jsonl（发射点
   agent_loop.py:148-152，空块 strip 后不发事件、口径天然正确），判卷器按 HARD / L1 分别出触发率
   ⚠️ 现 run_*.jsonl 不含事件，R5 的"1 条非空"来自训练机 PG ⇒ 校准（门槛④）得先做这一步
⑤ 修 v1 判卷器截断（u_exam_judge.py:145 的 fails[:20]）
⑦ N1 纯净终答正则（JSON 壳/代码块/字段名）进判卷器读数；compare 的 cap 表对每个 cap 下
   verdict（Δ ≤ 2·√n_before）
⑧ 编造判据（WIN 用，也给全卷当报告项）：回复里的数字/ID 与「本轮工具返回 ∪ 历史文本」求差，
   差集非空即编造——这条不需要阈值（守则①）
⑨ 生成器结构断言扩到五个多样性维度：每科 ≥2 对象 · ≥2 问法模板（held-out 30%）· ≥2 历史长度 ·
   正反成对对齐 · WIN 必含 ≥7 轮历史；任一断言会红
⑩ 考场时长重估：133 题一遍 ~10 min ⇒ v4 约 400 题一遍 ~30 min，四遍 ~2 h（训练机，W5 排期用）
⑪ 收尾：scripts/v15_gate_triage.py 登记表按 v4 实际题数回填（硬预期行为题 = REJ+DEF+CLA+L4+
   DEF-F+REJ-F+CLA-F ≈ 160）并 --strict 零缺口
```

| 门槛（W1 出口） | 数字 |
|---|---|
| ① 结构断言 | v4 生成器每科的多样性断言全绿（≥2 对象 · ≥2 问法 · ≥2 历史长度 · 成对对齐 · WIN ≥7 轮），且**撤掉任一断言会红** |
| ② 可比性 | L1–L4 125 题与 v3 **diff = 0** |
| ③ 判卷器负向认证 | 每个新科目构造 ≥5 类劣化答卷（该拒不拒/嘴拒手动/该 defer 直答/该办仍 clarify/空答/**编造数字**/复述上轮原话/旧参数粘连），每类判红率 **100%**；gold 答卷 selfcheck 100% 判对 |
| ④ 思考率尺子校准 | 考场 jsonl 带 thinking 计数（已交付）；读数 = PG 查库值的对照在 **W5 起链的第一遍**做（本机有 PG 但无模型端点，跑不了考场） |
| ⑤ 装置足量 | 带硬预期行为的题合计 ≥ **150**，每种行为 ≥ **20**；报告项科目各 ≥10 |
| ⑥ 三查零缺口 | `v15_gate_triage.py --strict` 退出码 0（登记表按 v4 实际题数回填） |

**✅ W1 完成（2026-09-02，本机 0 GPU）**
```
产物   scripts/u_make_exams_v4.py → data/u_route/context_v4_exam.jsonl（361 题：L1 50·L2 25·L3 25·L4 25
       逐字继承 · REJ 32 · DEF 24 · CLA 20 · HARD 20 · DEF-F 20 · REJ-F 20 · CLA-F 20 · L2-x 20 · WIN 20 ·
       META/PRG/COR/TIME 各 10 报告项）· scripts/u_exam_judge_v4.py（13 个新判类 + 按档读数：思考率/N1/
       编造/写操作）· scripts/v15_w1_exam_certify.py · u_exam_run 加脚本化历史 + think_nonempty ·
       contract.n1_hits · compare 逐 cap 泊松 verdict · v1 判卷器去截断 · demo 加 CMP_7（样本不足型不成熟）
读数   ① 结构断言：每科 ≥2 对象（REJ 6/DEF 4/HARD 4/L2-x 5）· ≥2 问法（4–17）· 成对 5 科全配对 ·
         L2-x 轮距 {1,3} · WIN 历史 {3,8} · 收场类型 answer/defer/reject/clarify 全在场；
         撤掉一条（DEF 单对象）断言即红（tests/train/test_exam_v4_structure.py）✓
       ② L1–L4 125 题与 v3 diff=0 ✓
       ③ 负向认证：13 判类 × 5 类劣化全部判红、gold 全判过（含编造数字/复述原话/旧参数粘连）✓
       ④ 装置交付 ✓；校准挪 W5 第一遍
       ⑤ 硬预期行为题 161 ≥150；每种行为 ≥20 ✓
       ⑥ --strict 零缺口 ✓
脚本化历史 = 按线上同一张 agent_runs 表插终态行，prior_turns 同一条读取路径，6 轮窗口原样作用
       （tests/runtime/test_exam_scripted_prior.py 真库验过：9 轮取最近 6、reject 轮在、事实轮出窗）
考场时长重估 361 题 ≈ 133 的 2.7 倍 ⇒ 一遍 ~27 min，四遍 ~1.8 h（W5 排期）
未入卷    APR-F（审批回灌待 K 线 D25）；②③⑤族只放报告项，训练数据下一版
```

### W2 · 数据管线同形改造（~1 天代码 + 0.5 天验证）

做什么（采纳前任评估的改法 1，改动面最小且不动 `Case` 结构）：

```
① build_messages 加可选参 prior（rollout_loop.py:130-167）：prior 存在时在 system 与
   本轮 user 之间插真 user/assistant 消息对——形状对齐 decider.py:222-227
   ⚠️ build_messages 是 SFT 和 RL 共用的（:133 注释明示），改它必须双路都验
② build_l2_l1 重写：历史不再折文本；上一轮助手内容用真实终答人话（教师物料缓存
   已进版本管理，commit 0eabfa8）；不同形 #1#2#4#8 在同一段代码，一起修
③ 字段清单（#3，已裁定=对齐线上）：多轮行 required_answer_fields 一律 MIN_FIELDS
   ⇒ v15 下与线上一样整节消失；「读数在场」改由 gold 终答自带数字来教
   （L2 判卷的「读数在场」闸不动，u_exam_judge_v2.py:97-99）
④ 时间（#5，已裁定）：所有行「当前时间」渲染成纯日期（同线上）；随 W4 全量重建生效，
   重建后跑 v13 语义冻结四项全等确认压舱数据没走样
⑤ 菜单（#6，已裁定=训练改全量 34）：先实测全量菜单的 token 占用与现契约 prompt 预算；
   放得下直接用；放不下就精简 tool_registry 里的工具描述文字——说明书两侧共用同一份
   （syncopate/core/tool_registry.py 是唯一真相来源），精简后依然同形。
   重量结果回填 rollout_budget，零截断判据沿用
   ★ 09-02 实测（scripts/v15_w2_menu_budget.py → _audit/v15_w2/menu_budget_v13.json）：
     修剪前 工具块 6534 tok · 全量菜单最长 prompt 8095（上限 5760 / 真实约束 6144=14336−8192）
     修剪后 工具块 5468 · 最长 7167（scripts/v15_w2_trim_tool_desc.py：34 条描述按「只留做什么/输入/
     返回/独有硬规则」改写；跨工具纪律并入 system.txt 既有章节 +262 tok；交叉引用 15 句、原理 4 句删；
     硬事实逐条核对不丢；登记 _audit/v15_w2/tool_desc_trim.json）
   ⇒ 精简省 1203 仍差 1407：**上限必须抬**（待 Chaoyu 定数：训练 MAX_PROMPT_LENGTH 9216 +
     response 8192 = 17408 ⇒ 服务 max_model_len 18432；R6 起跑前按 V0⒠ 重测单步耗时）
⑥ 新增同形断言测试家族（守则⑮ 的判据形状=「两边应当相同」，不需要阈值）：
   渲染一条训练多轮样例 与 一条 decider 生产请求，逐项断言结构一致
   （历史位置=消息对 · 助手历史=人话无 think · 无字段清单 · 时间格式 · think 块规则）
⑦ 🆕（09-02 六族扩展）训练行覆盖第一波科目：④族（DEF-F/REJ-F/CLA-F：prior 以信令收场，
   助手历史文本 = 信令自己的话，同 decider._prior_turn_messages 口径）· ①族加宽（L2-x 两对象
   在场 + 旧参数不粘连的对照行）· ⑥窗口（超窗引用 → gold=承认并追问，零编造）。每科成对对照，
   份额闸按监督 token 重新定带宽（旧带宽是按四级标定的，换科目 = 换单位）。②③⑤族本轮不造训练行
```

| 门槛（W2 出口） | 数字 |
|---|---|
| ① SFT/RL 同构 | `tests/train/test_rollout_loop.py` 全绿（think-on 同款，R1⑥ 判据沿用） |
| ② 同形断言 | 新测试全绿，且**负向认证会红**（故意折回文本形状必须 FAIL——空门槛不算过） |
| ③ 压舱石不走样 | v13 语义冻结迁移「四项全等」复跑通过（419 行） |
| ④ 全量回归 | 测试套件零失败（R2 时点基线：v14 794 / v15 792） |
| ⑤ #8 回归 | switch 行断言：context 与 gold 指向同一对象 |
| ⑥ 菜单预算 | 全量 34 菜单下 prompt max ≤ 预算且零截断（重量后回填具体数进本表） |

**🔶 W2 完成（2026-09-02，本机 0 GPU）——只剩上限数字待 Chaoyu 定**
```
代码   syncopate/core/prior_turns.py（历史→消息对的唯一渲染，decider 与 rollout_loop 共用）·
       rollout_loop.build_messages 加 prior + 纯日期 · CaseBundle.prior（内存字段，不落盘）·
       core/demo_context.py（线上 context 构造抽到 core，训练 chat 行同形）·
       scripts/u_build_v15_multiturn.py（prod_context / as_multiturn / 六族第一波训练行 / 同形体检）·
       u_build_v14_5：L2/L1 改真消息对（#1#2#3#4#6#7#8 一起修，switch 只取 env 里真实存在的另一条）、
       chat 行改走同一组渲染函数（此前走 probe 私渲染：v14 尾段 system + 假 context + summary 字段清单 = 四处不同形）、
       fam 桶 + 带宽 + manifest.render + 出厂同形体检 · sft_replay 认 gold.clarify_question
       · 34 条工具描述修剪 + system.txt 并入 4 条纪律（见 ⑤）
读数   ① SFT/RL 同构：test_rollout_loop 32 passed（含带 prior 的逐 token 同构）✓
       ② 同形断言：tests/train/test_train_prod_same_shape.py 4 条（负向：折回文本/ISO 时间 ⇒ 红）✓
       ③ 压舱不走样：v15_r2_migrate 四项全等 → **挪 W4**（本机无 data/sft/v13 parquet；batches v13 已由影子重建落本机、切分 SHA 一致）
       ④ 全量回归：本机 tests 全绿（除需 GPU/CUDA 扩展的 e31/verl_patches）
       ⑤ #8 回归：switch/compare 只在 env ≥2 条 campaign 时取真实存在的另一条（169/1670 个 env），否则退 same
       ⑥ 菜单预算：修剪后仍差 1407 ⇒ ⛔ 等 Chaoyu 定 MAX_PROMPT_LENGTH / max_model_len（建议 9216 / 18432）
DRY   U_BUILD_DRY=6 结构演练（0.6B tokenizer、不调教师、不落盘）：多轮行 146 · 不同形 0 ·
       fam 54 行（deff/rejf/claf/l2x/win）· 六个源 case 无压舱人话缓存 ⇒ 正式建库前由 ballast_replies 先写
待 W4  份额带宽（v13 0.48–0.62 · fam 0.04–0.12 先按行数估）首次实测回填；CoT 带宽上沿 0.30 已改
```

### W3 · CoT 重设计（~1 天；分析依据在 §4）

做什么：

```
① ~~think 做轻~~ **撤回（Chaoyu 09-02）**：教师思考 p50 691 字/6 段，没有证据说明多出来的是啰嗦
   而非必要推演，砍长度可能砍掉的正是质量；比例问题由 §4.1 的「空块不监督」解决。
   采样约束保持原样（900 token 上限、≤4096 字、不限段数），覆盖行数按预算自然落
② 触发信号显性化：难例行的题面必须含可学的难度特征（多步诊断类问法），
   不能只靠隐藏的模板族标签——模型在题面上看不出「该想」，就永远学不会触发
③ 行为/信令思考补课：reject/defer 类不用裸 8B（P0-5 实测 2/5）——教师 prompt 注入
   契约上下文后重跑探针，≥70% 才用；不过线则该类显式不带 think 并登记（照抄 P0-5 结论）
④ 带宽上沿 0.20→0.30（已裁定；改 u_build_v14_5.py:1139 的 bands["cot"]=(0.05,0.30)），
   选择算法重跑；可达性预算先算后训（§4.4 的算式，W5 实测回填）
```

| 门槛（W3 出口） | 数字 |
|---|---|
| ① 新池画像 | think 长度 p95 ≤ **500 字符** · 段数 ≤ **2** · 中文占比 ≥0.5 · 命中判据未放宽 |
| ② 触发可学性探针（新） | 用题面文本预测该行是否难例（简单分类器/规则），held-out 准确率 ≥ **80%**——题面预测不出来=模型也学不出来 |
| ③ 可达性预算表先行 | 按选中行数/块数**先算出**全库非空块占比与难例桶覆盖率的预期值并预注册，W5 实测回填对照 |
| ④ 行为类 think | 探针 ≥70% 才入库；不过线则该类不带 think 且登记入 §7 欠账表 |

**🔶 W3 本机部分完成（2026-09-02）——①④ 的读数要教师（训练机，并入 W4）**
```
代码   ① think 做轻：gen_cot_v15 的 one_think max_tokens 900→350 + step_sample 画像闸（≤350 字·≤2 段·中文≥0.5），
         命中判据「首动作名相等」未放宽（THINK_MAX_* 常量在 u_build_v14_5）
       ② 触发显性化：scripts/u_build_v15_cot.py explicit_hard_prompt（4 前缀/4 后缀多步诊断问法，按 case_id 确定性选）
         接进 gen_cot_v15._row；只加在 CoT 行
       ③ 可达性预算表：scripts/v15_w3_budget_table.py → _audit/v15_w3/budget_table.json
       ④ 行为类 think 探针：scripts/v15_w3_behavior_think_probe.py（契约上下文 + 8B，≥70% 才入库）→ W4 跑
       带宽上沿 0.30 已改（W2 时随 fam 桶一起）
读数   ② 触发可学性探针（scripts/v15_w3_trigger_probe.py，哈希字符 n-gram + 5 折平衡准确率）：
         族级 83.8% · **族内 65.5%**（难例=隐藏标签，题面看不出该想）· 族内+显性化 **88.5% ≥80** ✓
         ⚠️ 诚实读法：过线一半是问法本身可学，模型可能学成"这种问法 ⇒ 想"而非"这种难度 ⇒ 想"；
            HARD 档考题（v4）同样带多步诊断问法，两边同形，先按此口径；RL 阶段 reward 再校
       ③ 现池画像（114 行）：think p50 691 字/6 段/371 tok · 行重 p50 2519；做轻后块 ≈188 tok、行重 ≈1391，
         30% 带宽预算 ≈97.5k ⇒ 可装 ≈70 行（此前 20）；全库非空块占比估 ≈9%；HARD 触发率预注册带 20–50%
         （非 CoT token 按 v15 manifest 份额反推，W4 实测回填）
       ①④ 新池画像（p95 ≤500 字·段数 ≤2）与行为类 ≥70% ⇒ W4 重采样时读
```

### W4 · 重建 + 出厂体检（训练机 = **Modal RTX PRO 6000×2**，Chaoyu 09-03 裁定；环境与试点见 08 §Modal；~1 小时）

门槛：出厂体检全绿（`u_build_v14_5.py` 末尾已进流程）· 份额闸/密度闸/D-L 门禁全过 ·
**同形断言进出厂体检**（shape_check 在建库产物上跑，已接）· 建库 stdout tee 入 `_audit/`
（§2.4 新纪律）· §7 欠账读数全部落盘。

**训练机施工清单（09-02 定，按顺序；每步产物与判据写死）**
```
0  前置：拉最新 main；SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1；确认 rollout_budget.MAX_PROMPT_LENGTH 已按
   Chaoyu 定数改（默认建议 9216）且 launch_rl/vllm 启动脚本 max_model_len = 9216+8192 → 18432；
   `python -c "from syncopate.train.rollout_budget import *"` 判据行 [think-mode] 打出
1  教师起服务（u_p2_v145_chain.sh 的 4B@8210 / 8B@8211 段）
2  行为类 think 探针：scripts/v15_w3_behavior_think_probe.py --n 20 ⇒ _audit/v15_w3/behavior_think_probe.json；
   ≥70% 的行为类才允许 gen_cot_v15 收（低于线的登记 §7 欠账，CoT 行不带其 think）
3  ⚠️ 旧缓存作废：data/u_route/v15_l2l1_rows.json（折叠文本形状）与 v15_cot_rows.json（旧 think 画像）
   改名 *.pre_w2.json 留档，不许被命中（缓存命中会绕过全部新改造——㉒ 同族）
4  建库：SYNCOPATE_CONTRACT=v15 SYNCOPATE_THINK=1 python scripts/u_build_v14_5.py 2>&1 | tee _audit/v15_w4/build.log
   判据：[同形] 不同形 0 · 份额闸六桶在带（首次实测后把 v13/fam 带宽回填本表）· [CoT-v15] 命中率行入库 ·
        出厂体检 ✅ · scripts/v15_r2_gates.py --prompt-budget data/sft/v15/train.parquet 零截断
5  压舱不走样：scripts/v15_r2_migrate.py --scope frozen419 ⇒ 四项全等 419/419
6  新池画像（W3 门槛①）：scripts/v15_w3_budget_table.py 对新 cot 缓存重跑 ⇒ p95 ≤500 字 · 段数 ≤2 · 中文 ≥0.5
7  影子重建判据（如时间允许）：scripts/run_pipeline_shadow_rebuild.sh 只到 3/6（batches SHA）
```

### W4′ · v16 对齐 S1（09-03 起，B200 + 新栈；裁定⑩⑪⑫⑬之后 W4/W5 的前置）

> 目标：让「数据→SFT+eval→RL+eval→OPD+eval」全链在 **verl 0.9 / vLLM 0.28 / torch 2.13 / B200 / Qwen3.6-35B-A3B** 上跑通并**数值健康**。
> S0 对齐地图（09-03，Modal CPU）：968 测试 639 过 / 10 败 / 318 跳，10 败无一为"新栈弄坏代码"。判据先注册，跑前不改阈值。
> ⚠️ 口径决定（09-03 我定，Chaoyu 未逐字批）：**数据版本全改 v16**（case 库/切分/SFT/RL 目录、DATA_VERSION）；
> `SYNCOPATE_CONTRACT=v15` 这个**契约协议名**不改——它命名的是"行为进 function-calling 强通道"的消息契约，契约本身没变，
> 改名会碰 contract.py/tests 数百处且不产生任何新信息。若 Chaoyu 要求连协议名一起 v16，单独做一次机械重命名。

| 步 | 做什么 | 判据（跑前注册） |
|---|---|---|
| S1-1 数据口径 | `split.DATA_VERSION="v16"`；新 spec `configs/buckets/v16.yaml`（= v13 配额，**不 freeze-from**，从零一次生成）；所有 `data/batches/v13`/`data/splits/v13`/`data/sft/v13`/`data/sft/v15` 字面量改为从 `split.DEFAULT_*` 或一个 `paths.py` 派生 | `grep -rn "data/(batches\|splits\|sft\|rl)/v1[35]" syncopate scripts tests` = 0（shadow_rebuild 脚本除外，标 legacy）；`test_data_version_contract` 绿 |
| S1-2 分词器/学生 | 测试与脚本里 `models/Qwen3-0.6B`/`Qwen3-4B` 改为**单一常量**（默认 `models/Qwen3.5-0.8B` 测试 / 学生 `Qwen3.6-35B-A3B`），Modal 上 `/vol/repo/models → /vol/models` 软链 | 全库 grep 老路径 = 0；Qwen3.5 模板下 `test_rollout_loop`（think-on 同构、tools 渲染）全绿 |
| S1-3 模板契约 | Qwen3.5 chat template vs Qwen3 逐项核：think 标签、tool_call 格式、`enable_thinking`、多模态节；`rollout_budget` 重量 prompt 上限（新 tokenizer 248k 词表） | 同一批消息两模板渲染 diff 入档；全量 34 菜单最长 prompt ≤ 上限，零截断 |
| S1-4 补丁分诊 | 20 处 verl 0.8 补丁按目标模块在 0.9 里**是否存在 + 上游是否已修**逐条标 删/改/留（机器判据：import 目标模块） | 表入 26；`test_verl_patches` 在新栈全绿或对应测试同步删 |
| S1-5 PrefixGrouper | 决定去留：verl 0.9 是否有等价（关键词扫描）；留则 `prefix-grouper` 进新栈锁并过 PG 位等价测试 | 二选一落文档；不留则 E26 判为历史 |
| S1-6 镜像装项目 | `uv sync` 改为装项目（vLLM 插件入口点），或运行时 `uv pip install --no-deps -e /vol/repo` | `test_u4_entry_point_registered` 绿；`vllm.general_plugins` 里有 syncopate 入口 |
| S1-7 LoRA on MoE | sft.py `target_modules="all-linear"` 对 35B-A3B 会挂到全部专家 ⇒ 参数量/显存先算；默认改为注意力+共享专家，专家层作开关 | 可训练参数量、峰值显存表；LoRA 结构能被 vLLM 加载 |
| S1-8 考场底座 | 镜像加 PostgreSQL + Redis（08 §1.2 的 Modal 版），`pg_bootstrap/redis_bootstrap` 单容器 | runtime 318 个跳过的测试中 PG/Redis 类转为跑且绿 |
| S1 出口 | 仓库测试在新栈全绿（PG/Redis 到位后无"依赖缺失型"跳过）；`check_pipeline_invariants` 退出码 ≠1 | 一条命令：`stack_probe --steps pytest` ✅ |

**S1 进度（09-03 深夜；本机子集 221 passed，Modal 全量与 v16 对照跑中）**
```
S1-1 ✅ DATA_VERSION="v16"·configs/buckets/v16.yaml（v13 配额、不 freeze）·活跃代码路径字面量→split.DEFAULT_*；
       判据 tests/pipeline/test_no_stale_version_literals.py（legacy 白名单显式）绿
       本机独立生成 v16：1670 条·拒 8·切分 eval 342 / sft 503 / rl 825；SHA（_audit/v16/local_gen_2026-09-03.log）
         eval b15e314a… · sft 831bfe1b… · rl 05a9c8c6… ← Modal 生成必须逐一相同（p_rebuild_v16）
S1-2 ✅ core/model_paths.py（TEST_TOKENIZER=Qwen3.5-0.8B · STUDENT=Qwen3.6-35B-A3B · TEACHER=Qwen3.8-27B）替换 30 文件默认值
S1-3 ✅ Qwen3.5 模板核对：① 工具调用线格式 = XML `<function=…><parameter=…>`（不是 Qwen3 的 JSON）⇒ parsing_v15 两种都认、
       render_tool_call/render_signal 默认 XML（SYNCOPATE_TOOLCALL_FORMAT=json 回旧），schema 收型（integer/number/boolean/
       object/array；无 schema 字段按 JSON 标量推断——数字形字串会被当数字，有 schema 的 string 不受影响）；
       ② enable_thinking 语义同 Qwen3（True 开 `<think>\n`，否则空 think）；③ 历史 assistant 的 think 只保留最后一次
       user 之后的轮次（与我们 #7 同形）；④ 单条 tool 消息渲染会 raise "No user query found" ⇒ 增量渲染改桩后缀法
       （rollout_loop.render_env_message_ids）；判据 = SFT 整段 == RL 增量逐 token（test_rollout_loop 全绿）+
       我们渲染的 tool_call 与模板对同一结构渲染逐字节相同（test_xml_render_matches_qwen35_chat_template_exactly）
S1-4 ✅ 分诊表见下；新依赖 transferqueue/cupy-cuda13x/prefix-grouper/liger-kernel 进 stack 锁
S1-5 ✅ PrefixGrouper 走上游 `actor.use_prefix_grouper`；自研接线待删（随 RL 冒烟）
S1-6 ✅ _sync_repo `uv pip install --no-deps -e /vol/repo` + models 软链
S1-7 ✅ Qwen3.6-35B-A3B（40 层·256 专家·top-8·共享专家 512·10 全注意力+30 线性注意力）：LoRA r=32 all-linear ≈ **2554M**（专家 2517M，AdamW 状态 ≈14 GiB）
       vs 注意力+共享专家 ≈ **37M** ⇒ sft.py 默认 SYNCOPATE_LORA_TARGETS=attn_shared（排除 .experts.），all-linear 留作开关；模块名与参数量在 S4 真模型上核
S1-8 ✅ PG 16 + Redis 进镜像，pytest 步起服务并灌语料
**wandb 接线 ✅（09-03 23:50）**：Chaoyu 的 `wandb-secret` 注入容器，在线写 3 点读回 3 点（project syncopate-b200）
**S1 出口 ✅（09-03 23:39，Modal run12）**：新代码 + PG/Redis + v16 数据，`stack_probe --steps pytest` 退出码 0（上一轮 951 passed / 23 skipped，跳过从 318 降到 23）
**S2 ✅ v16 确定性判据（09-03 深夜）**：Modal（p_rebuild_v16，GCP 容器）与本机（samwang-X870I）各自从 configs/buckets/v16.yaml 独立生成，
       三份切分 SHA-256 **逐一相同**（eval b15e314a… · sft 831bfe1b… · rl 05a9c8c6…），6681 个 case 文件；Modal 全程 3.5 min
       （generate 90 s · menus 72 s · split 36 s）。⇒ v16 case 库 = 与机器无关的纯函数产物，可随时重生成，不再需要"冻结文件"。
新栈坑（09-03，已修+判据）：transformers 5 的 `apply_chat_template(tokenize=True)` 返回 BatchEncoding（4.x 是 list）⇒ `ids + list`
       TypeError、`len()` 量成键数=2 ⇒ 一律走 `rollout_loop.chat_template_ids`（tests/train/test_chat_template_ids.py）
```

**S3 建库进度（09-04 凌晨，B200 单卡 `stack_probe --steps build_v16`）**
```
run14  教师 Qwen3.8-27B 121 s 起服务 ✅ · 行为类 think 探针跑了但用的是旧代码（PYTHONPATH 指到已废弃的 /vol/repo，探针常量顺序 bug，已修）
       · 压舱人话 687 条全部由 27B 生成（~7 min）✅ · **L2/L1 构造断言 FRESH_0125 无真实终答人话** ⇒ 根因：源 case 只过滤了首动作
       是 get_metrics，没过滤收场类型；defer 收场的 case 没有人话终答（历史该是信令自己的话＝④族的事）。v13 时代被累积缓存盖住，
       v16 重编号 + 缓存作废后首次显形。修：源 case 限终答型（tool_call/answer），与 `_need` 同条件。
       · 教师缓存写在容器本地被丢 ⇒ 探针改为建库前后与 /vol/_audit/v16/cache 往返（断点不从零）
run15  压舱 687 ✅ · L2 290 / L1 250 / 家族 180 ✅ · **CoT 段崩**：gen_cot_v15 读上一版 parquet 选候选（v16 之前没有 parquet）
       ⇒ 改取当前切分 sft 桶；同时发现 **教师动作解析与行为探针都按 JSON 找 `"name":`**——Qwen3.5 教师吐 XML ⇒ 命中率会是 0、
       探针 0/0（run15 实测三类全 0/0）⇒ 一律走 parsing_v15.parse_tool_calls（两种线格式都认）
       · 压舱桶来源改为**当前切分 sft_cases**（val 每 6 取 1），不再依赖旧 parquet（裁定⑩）· DRY 不调教师用 "[DRY" 占位（正式产物红线）
run16  ✅ 前四段全过（压舱/L2 290/L1 250/家族 180 缓存命中）· **CoT 蒸馏真跑起来了**：难例池 114（BUD/DIA/FAIL/RAG/SCALE），
       27B 教师逐步采样，日志里 ≥14 条「收 X（1/10 步有思考）」；但收尾 **`预算 44885 内选中 0 行 · 聚合非空 think 0/0`** ⇒ CoT 桶下限闸
       （≥19 行）红。⇒ 下一任第一件事：查「收」到「行」之间哪一步把 think 行丢了（u_build_v14_5.py gen_cot_v15 收尾 + main 里
       sur()/预算可行上界搜索 ≈ L1160–1200），怀疑点：① 27B 的 think 长度/段数/中文占比闸（THINK_MAX_*）② attach_think 对 XML 步的匹配
       ③ sur(r)≤0 全被判 neg。教师日志末尾有一次 EngineCore shutdown（16:47Z），先确认是收尾 teardown 还是中途死。
       读数：build.log = /vol/_audit/v16/build.log（本机副本 /tmp/v16/build.log 已失效，重新 `modal volume get`）
```

**S3 run16 成行 0 的归因（09-04 接手核对，前任 -7b 已确认）**
```
① 不是「收到→成行」之间丢了，是选择步的算术必然为 0：64 行候选几乎全是「1/10 步有思考」，sur(r)=think−0.6×blocks 全为负 ⇒
   pos=[]，可行上界搜索里 su+sur(r)≥0 永远不成立 ⇒ sel=[]（L1155–1190）。怀疑点③成立且是确定性的。
② 上游真病：27B 教师采样 892 步只命中 12（1%）。那 ~64 个「1 步思考」绝大多数来自 v15_materials.json 的 cot_think
   （60 条 v14.5 时代 8B 终答步旧思考，按 case_id 静默复用；v16/v13 的 case_id 指同一题）⇒「教师收了 14 条」基本是旧 8B 料。
③ 行为探针 run15 的 0/0 是 tried=0（JSON 找 "name" 没匹配到步）；run16 的 0/20 是 tried 有了、hit 0 —— 两个 0 不是一个 0，
   说明过滤链（900 token 内要有 </think> · cjk≥0.5 · 首动作==gold）才是现在的瓶颈；每个丢弃原因都是静默 continue，日志分不出。
   THINK_MAX_TOKENS=900 是 09-02 按 8B 定的（W3① 撤回时保留），27B 很可能超。
④ 教师 EngineCore 16:47:04 的 shutdown 是 finally 里 _teardown 的收尾，与 assert 差 3 秒吻合，不是中途死。
文档外四坑（前任补，已落地）：cot_think 复用暗道（裁定⑭关）· 探针 stale 判据量错对象（数的是取回的新缓存，第二次起必红；改为
   grep 源码旧名）· S5 容器 PG 是新的要先播种（seed 不带 --check）· S5 adapter 的 served 名与 SYNCOPATE_DECIDER_MODEL 要一致（统一 v16_adapter）。
```

**S3-diag · 27B 教师原始思考画像（09-04 Chaoyu 放行；判据预注册，跑完不改）**
```
做什么   modal run --detach modal_app/stack_probe.py --steps teacher_diag        # B200 单卡，~15 min
         scripts/v16_teacher_think_diag.py：难例池 20 case × ≤3 步 × 4 样本，max_tokens 4096 只为量真实长度；
         对每条同时算「按现行 900 上限会不会写完 / cjk / 首动作==gold」；同容器再跑行为探针（现带丢弃计数）
产物     /vol/_audit/v16/teacher_think_diag.{json,md}（md 原样 8 条给 Chaoyu 看）· behavior_think_probe.json（含 drop）
预注册判读 closed_within_900_rate < 50% ⇒ 900 上限是主拦截 · cjk_below_0.5_rate > 50% ⇒ 语言闸 ·
         都不成立且 action_match_rate(写完的) < 30% ⇒ 教师/gold 不一致（问题在题不在闸）
纪律     诊断结果出来之前不改任何阈值；改阈值 = 新一轮注册（守则⑬）
读数 A 臂（09-04 01:47，240 样本，0 错误）  closed_within_900 = 92.9% · think token p50/p90/max = 77/326/894 · **cjk p50 = 0.0，cjk<0.5 = 100%** ·
         写完的里首动作==gold = 67.7% · 现行链通过率 = 0%。预注册判读命中第二条：**语言闸是主拦截——27B 教师全程英文思考**。
         行为探针同证：clarify/defer/reject 三档丢弃原因几乎全是 cjk_below_0.5（159/155/160），只有 defer 有 5 条没在 900 内写完。
         英文思考本身质量好（分步计划、逐项核对、意识到"一次只能一个工具调用"）；见 /vol/_audit/v16/teacher_think_diag.md。
         mismatch top：get_metrics→policy.get_budget_rule（10）· update_budget→campaign.list（8）· __text__→system.wait（8）——教师倾向多查一步。
⇒ 待 Chaoyu 裁定（三选一，都不是我能拍的）：① 撤 cjk 闸，收英文思考（学生学"英文想、中文答"）② 让教师用中文想：<think> 后加中文引子
   （B 臂 zh_prefix 正在量：cjk 与 action_match 是否保持）或系统指令 ③ 换会中文思考的教师。B 臂读数出来后一并呈报。
```

**并行冒烟（09-04 Chaoyu：能并行的多起机器；各臂各目录，一个写者）**
```
S4′ 机制冒烟  --steps sft_smoke --sft-arm mech_dry --max-steps 30
              容器里 U_BUILD_DRY=6 演练出 _audit/v16/dry_rows.parquet（"[DRY" 占位、不调教师），只验与数据内容无关的机制：
              判据 = 可训参数 37M±20%（抓 LoRA 正则里按 Qwen3-Next 猜的模块名）· loss/grad 有限 · 存档可被 peft 加载 ΔW>0 ·
              峰值显存 <180 GB · tok/s 记录。产物 /vol/checkpoints/sft/mech_dry **不是候选**。
S5′ 链路冒烟  --steps exam_v4 --exam-arm plumb --exam-limit 40（学生底座、无 adapter）
              判据 = seed→check 7 条 · 端点起 · API/worker 起 · u_exam_run rc 0 · judge_v4 rc 0 · triage 出表；分数不看（底座没训）。
S4′ 读数（09-04 01:46）✅ 机制全过：可训 42.3M（带内）· loss 1.93→0.29 · ‖ΔW‖/‖W‖ 0.63% · 显存峰值 74.1 GB · 30 步 346 s ·
              adapter 落盘（sel 点 81 MB）。探针首判红是 grad_norm 只上 wandb 没打 stdout + 峰值正则量错字样（已修：sft.py 每 5 步打判据行）。
S5′ 读数（09-04 01:42）✅ 链路全通：PG/Redis/语料 0 · 播种 7 campaign · 35B 端点起 500 s · 40 题 137 s · judge/triage rc 0
              （分数无意义：底座未训；fab=42% 是底座本色）。⚠️ 端点起 500 s 要查是 JIT/compile 缓存没命中还是 MoE 装载慢（学习项）。
S6 rl_cfg ✅（09-04 01:42）：data/rl/v16 660/165 行 · Hydra 合成通过。两个键坑已修：create_rl_sampler 在 0.9 不在 main_ppo（补丁改挂
              trainer.ppo.utils + trainer_base）· save_lora_only 要 + 追加。
S6 RL 冒烟    已写（`syncopate/train/launch_rl_v1.py` 薄壳；`--steps rl_cfg`（CPU：造 data/rl/v16 + Hydra --cfg job 键名判据）→ `--steps rl_smoke`）。verl 0.9 事实（09-04 容器 dump /vol/_audit/v16/verl09_dump.json）：入口 main_ppo.TaskRunnerV1 + 配置项
              trainer_mode（sync / colocate_async / separate_async，trainer/ppo/v1/）；create_rl_sampler 搬到 trainer/ppo/utils.py
              （main_ppo 里已没有这个名 ⇒ main_ppo_pool 的 monkeypatch 现在挂空，必须改挂 utils + trainer_base）；
              save_lora_only 在 checkpoint_manager；use_prefix_grouper 在 actor 配置；rollout_correction 在 algorithm。
S7 OPD        已写（opd.py：--adapter 可空=底座新建 LoRA · --max-steps · vocab 断言 · 默认 prompts=v16_p1_prompts.jsonl；`--steps opd_smoke`）。前置已核：学生 Qwen3.6-35B-A3B 与教师 Qwen3.8-27B **vocab 逐项相同**（248077，diff 0，encode 相同），chat_template 不同
              ⇒ 逐 token 蒸馏可行，但教师侧必须用学生模板渲染的同一串 token id 喂（不走教师自己的模板）。opd.py 的 ADAPTER 仍指
              v13 产物 cand_v13r2_e1 ⇒ 改前置 SFT adapter 参数化。
```

**S1-4 补丁分诊结果（09-03，机器判据 = 在新栈镜像里 import 目标模块；上游对照 = verl 0.9 源码关键词扫描）**

| 补丁（verl_patches.py） | 目标模块在 0.9 | 上游现状 | 处置 |
|---|---|---|---|
| `_patch_fsdp_degenerate_mesh`（E21 退化网格） | torch FSDP1 路径仍在 | FSDP1 维护模式；verl 0.9 默认 FSDP2 | **删**（我们走 FSDP2；E21 降格为历史） |
| `_patch_fsdp_shard_alignment`（E18 16B 对齐） | `torch.distributed.fsdp._flat_param` 在 | NVLink 双卡 all_gather 871 GB/s，悬崖未见 | **停用**，B3-1 重跑 E18 判是否复活 |
| `_patch_fsdp_cpu_copy_for_ddp` | `verl.utils.fsdp_utils` 在 | FSDP2 CPU 快照路径不同 | **删**（FSDP2） |
| `_fix_pg_repeat_interleave` + PG 接线（E26） | `prefix_grouper` 包需另装 | **上游已集成**：`trainer/ppo/prefix_grouper_utils.py` + `actor.use_prefix_grouper` | **删自研接线，改用上游开关**；PG 位等价测试改对上游 |
| `_patch_ddp_sync_probe` / `_patch_grad_probe`（DDP 各 rank 同步判据） | `workers.engine.fsdp.transformer_impl` 在（内部可能变） | 无 | **留**（守则②：假设写成断言），按 0.9 内部重挂 |
| `_patch_opt_step_counter` / `_patch_postprocess_concat`（E14 乒乓） | 同上 | 需重量 | **停用**，B2-4 重 profile 后再决定 |
| `_patch_torch_prof`（profiler 挂点） | `workers.engine_workers` 在 | verl 0.9 有 profiler 配置组 | **改**：优先用上游 `profiler` 配置 |
| `_patch_device_probe`（Ray 单卡可见 NCCL 坑） | `single_controller.base.worker` 在 | 触发条件（无 P2P）消失 | **删** |
| `_patch_sync_step_timing` / NCCL ckpt engine 补丁（E22 权重同步） | `checkpoint_engine.*` 在（`nccl_checkpoint_engine` 需 cupy） | 0.9 有 checkpoint engine + `save_lora_only` | **删 E22 补丁**，改用上游；E29 LoRA-only ckpt = 上游 `save_lora_only=True` |
| `_patch_pool_sampler`（我们的动态池采样器） | `trainer.main_ppo` 在 | 我们自己的东西 | **留**，对 V1 trainer 重挂（`trainer_mode` 三选一） |
| E31 统一 FP8 vLLM 插件（entry point） | 入口点需装项目 | B1-2 Miles 正版路线 | **停用**，B1-2 替代 |

结论：20 处里 **删 6 · 停用 4 · 改 1 · 留 3（含 PG 改上游）**；新增依赖 `transferqueue`（V1 trainer）`cupy-cuda13x` `prefix-grouper` `liger-kernel`（verl 未声明）。
上游已有的三件（PrefixGrouper · LoRA-only ckpt · rollout_correction）= E26/E29/E23 在简历里降格为"已被上游吸收"的历史工作。

### W5 · 重训 + 五点谱 + 考卷 v4（训练机，~2.5–3 小时）

门槛 = **W0 修订并经 Chaoyu 批准后的 R5 门槛表**（此时表内已无三缺口门槛）。
方差前置条件按 SE 口径执行（W0）：读数与阈值之差 < 2·SE ⇒ 加 4 遍，上限 12 遍。

**考卷 v4 受 09-02/03 数据改动的影响逐条核（09-03，Chaoyu 要求数据验收后逐条查）**

| 改动 | 对考场的影响 | 核法 / 结论 |
|---|---|---|
| 题面 context 变空（无账户、无清单） | 考题都在 user 话里点名 CMP_x，模型靠 get_metrics / campaign.list 自查；DEF/HARD 不再有清单可抄 | `tests/runtime/test_exam_path_after_w2.py` ✅ 训练与考场同形 |
| account_id 运行态注入 | 模型不填 account_id 调 campaign.list / risk.check_account，收口按租户注入；worker 进程注册表须已装载 | 真库真 worker：`test_worker_injects_account_when_model_omits_it` ✅（执行记录带 ACC_DEMO） |
| 工具描述修剪 | 判卷只认工具名与参数，不认描述 | 判卷器 13 判类负向认证 ✅ |
| 上下文 18432 | 考场链 `v15_r5_exam_chain.sh` 与 decider 默认已同步；链改为 `--exam` 可选（默认 context_v4） | grep 无 14336 残留 ✅ |
| clarify → waiting_for_user · 拒绝轮进历史 | 考场按 status 判终态，读 session 事件；L4/CLA-F/REJ-F 的历史现在线上能表达 | `test_clarify_turns_enter_history` 4 条 ✅ |
| demo 加 CMP_7 | 训练机起链前 `seed_demo_data.py --check` 必须看到 7 条（安全线 GAME_SLG×华南 已有） | 起链清单第 0 步 |
| 历史窗口 6 轮下沉到渲染函数 | WIN 题库里插 8 轮、渲染只见 6 轮，与 prior_turns LIMIT 6 一致 | `test_exam_scripted_prior` ✅ |
| 空 think 不监督 | 只影响训练；思考率读数口径不变（model.thinking 非空事件） | — |
| 三查登记表 | 硬预期行为题 161、装置全部到位 | `v15_gate_triage.py --strict` 零缺口 ✅ |

**W5 起链清单（训练机，09-03 定）**
```
0  python scripts/seed_demo_data.py --check   ⇒ 7 条 campaign（含 CMP_7）、安全线/基准/记忆表非空
1  pgrep -f syncopate.runtime.worker 为空（陈旧 worker 抢队列，㉞）
2  bash scripts/v15_r5_exam_chain.sh <合并模型> v15r4 context_v4（四遍；一遍 ~27 min）
   ⚠️ 第一遍跑完先做 W1④ 校准：jsonl 的 think_nonempty 之和 == PG 里 model.thinking 计数
3  scripts/u_exam_judge_v4.py --context logs/u_route/run_v15r4_r{1..4}_context_v4.jsonl
4  scripts/v15_gate_triage.py 按四遍 judged 读数出表（R5 表：②③⑥ 判、⑤ 只记录、方差前置条件）
5  盲评：u_exam_judge --blind（v15 vs v14.5 各 100，钥匙另存）+ N1 零命中
```


## 4 · CoT 专题（Chaoyu 08-31 三问的完整回答）

### 4.1 数据还是 mask？——排查完了：**是数据，mask 和模板都干净**

```
mask    训练器按 loss_mask 建 labels（sft.py:100-121）：assistant 轮整段进 loss，
        没有任何代码把 think 摘出来或单独处理（全库 grep 确认）
模板    think-on 下 Qwen3 模板不注入空骨架（_audit/v15_probes/results.json 实测），
        [think-mode] 判据行在（rollout_budget.py:92-97）；考场链 v15 契约下确为 think-on
数据    空 think 块由剧本引擎当"模型输出"吐出（sft_replay.py:224-241 attach_think），
        与非空块一样落在 mask=1 段 ⇒ 4049 块里 3959 个空块全在拿梯度教「别想」
断开处  24 §2 设计的「监督按 token 段分家」（NL/工具/think 三通道）**代码里没有任何实现**
        ——单一 loss、全段等权（sft.py:636）。设计与代码之间是断开的，本文不新增
        实现它的计划（v15 修法走数据配比，不走 loss 加权；要立项另议）
★ 09-02 Chaoyu 裁定（推翻本节"空块是必要教学信号"的读法）：**空 think 块不监督**。
        初衷本来就是只在难题上采有质量的思考、简单题不采，而不是教模型"输出空思考"。
        修法：空块留在序列里保持位置对齐（修法 B 的"在场"不变），loss_mask 置 0
        （sft_replay._mask_empty_think，SFT/RL 同构测试用同一函数算期望）；非空 think 照常监督。
        ⇒ "3959/4049 在教别想"从此消失，不靠缩短思考换比例；简单题想不想交给模型/RL reward，
        R5⑤ 的「简单集 ≤10%」降为报告项。
```

注意历史沿革，别再犯一次：v14 时代「空 think 教坏了」的归因被探针**推翻过**（空块在
提示段没梯度，`25 §3.2`）；v15 修法 B 让每轮显式写 think 段之后，空块**真的进了监督**
——被推翻过的归因在新结构下成真了。同一句话在两个版本里对错相反，引用时带版本号。

### 4.2 8B 蒸馏出来的思考有没有用？——**有用，但三条边界要认清**

担心的原话：难题都是工具调用题，8B 不知道我们的工具，蒸出来的岂不是没用。
实查结论：这个担心一半不成立、一半成立。

**不成立的一半（教师不是裸猜）**：教师拿到的 prompt = 完整 system 说明书 + 该题全部工具的
JSON schema + gold 前缀（前几步真实工具调用与真实返回），从 `<think>\n` 续写
（`u_build_v14_5.py:392-398,460-484`）；且只收「教师**自己也选中了** gold 下一步动作」的
思考（拒绝采样 n=8，`:409-422`），明文承诺不是给答案编理由（`:366-367`）。
实测 604 段思考里 **96.5% 含具体工具名、91.6% 含具体参数/ID**——内容真在推演工具选择；
新采样步命中率 ≈63% 说明教师在这个上下文里大体解得动。

**成立的一半（三条实测边界，W3 各有对策）**：

```
① 命中只比工具名不比参数（first_action 只取 json["name"]）——想对工具想错参数的思考
   也会入库 ⇒ W3 可加参数级比对（至少 campaign_id），成本低
② 行为/信令步完全缺席：session.report 被排除（0/6）、reject 类 2/5——R5 挂得最惨的
   行为表达，CoT 一点忙都帮不上 ⇒ W3③ 注入契约上下文重探针
③ 教师看到的上下文里历史轮全是空 think——它的示范环境本身在演示「不想」
   ⇒ W2/W3 之后重采样自然消解（新数据历史轮形状对齐线上）
```

⇒ 蒸馏路线保留（这也是 Qwen3 官方造 4B 的配方：离线蒸馏→on-policy 蒸馏，优于直接 RL
且 ~1/10 GPU 时，`24 §1`），按 W3 补短板，不推倒。

### 4.3 是不是所有题都该有 CoT？——**不是；空 think 也是教学信号，病在配比和可学性**

契约 N3 是「按需思考」：难题想、简单题不想。门槛⑤ 有两半——难例 ≥50% **且**简单集
≤10%。所以简单题的显式空块不是 bug，是「教它别想」的必要信号；全加 CoT 反而会撞
简单集 ≤10% 那一半（`25 §7` 已有登记）。真正的病是三个，各修各的：

```
① 比例失衡到 97.8 : 2.2 ——「别想」把「想」淹没（W3① 做轻 + §6-2 带宽）
② 触发信号不可学——难例=隐藏的模板族标签，题面上看不出来（W3② 显性化 + 探针）
③ 量触发率的考卷上根本没有难例题（W1③④ 补考区和尺子）
```

### 4.4 可达性预算（先算后训——上一轮的学费就是没算这一步）

现状账（全部实测/复算）：CoT 行重 ~2500 tok（think 占 ~1900），预算 = 非CoT×0.19/0.81
= 36253 ⇒ 20 行封顶（`u_build_v14_5.py:1056-1095`）。

think 做轻后的推算（基于 114 行池的实测分布；**是推算，W4 重建后必须实测回填**）：

```
think 2 段 ≤350 字符 ⇒ 每行 think ~350–400 tok ⇒ 行重 ~900–1000 tok
带宽 30%（已裁定）    ⇒ 预算 ~66000 ⇒ ~66–72 行（现在的 3.5 倍）
                        30% 是留了余量的实际上限：其余四桶份额下限合计 66%
                        ⇒ 数学天花板 34%，顶满会压穿 v13 压舱桶（52–66%）的下沿
全库非空块占比        ⇒ ~3–5%（仍小——所以不指望全局占比翻盘，指望三件套：
                        难例行内覆盖 ≥60% 不变 + 触发特征显性可学（W3②）
                        + 考卷难例档专测（W1③④））
SFT 出口预注册预测    ⇒ HARD 档触发率 20–50%（区间宽是诚实的：这把尺子第一次存在）；
                        ≥50% 硬闸挂 R6 出口（已裁定；RL 有 reward 通道去争取）
```

---

## 4.5 · 全链路设定一览（09-02 核对；每个数只有一份定义，其余是消费者）

| 设定 | 唯一来源 | 当前值 | 消费者（不许另写数） |
|---|---|---|---|
| 训练 prompt 上限 | `rollout_budget.MAX_PROMPT_LENGTH` | **9216** | launch_rl `--max-prompt-length` 默认 · sft.py `SFT_MAX_LENGTH`=prompt+response · rollout 左截断计数 · `v15_r2_gates --prompt-budget` |
| 思考回复预算 | `rollout_budget.MAX_RESPONSE_LENGTH` | 8192（think-on）/2048 | launch_rl · eval_local `--max-new-tokens` 默认 · decider 单轮生成上限 · `MAX_TURN_ACCUMULATION`=+2048 |
| RL/训练 max_model_len | launch_rl 算 prompt+response | 17408 | verl `rollout.max_model_len` |
| 服务 max_model_len | 启动脚本 `--max-model-len` | **18432** | `logs/runtime/start_vllm*.sh` · `scripts/b4_serve_4x.sh` · `scripts/v15_r5_exam_chain.sh`；decider `RUNTIME_MAX_MODEL_LEN` 默认 18432（ctx_cap=−256，生成上限 = min(8192, 余量)） |
| think 开关 | `rollout_budget.THINK_ON`（v15 默认 on；`SYNCOPATE_THINK`） | on | `CHAT_TEMPLATE_KWARGS` · decider（`SYNCOPATE_RUNTIME_THINKING` 分叉待 R8④ 删） · launch_rl `[think-train]` 判据行 |
| 采样参数 | `rollout_budget.SAMPLING_*` | T=1.0 · top_p=1.0 · top_k=−1 | RL · eval · decider（一份） |
| 轮数上限 | `rollout_budget.assistant_turn_budget(max_steps)` | max_steps+1（v15 report 占一步） | build_sft_sample · build_dataset · RL extra_info |
| 工具菜单 | `contract.effective_tool_menu` | v15 一律全量 34 | run_rollout（SFT/RL 同源）· decider `FULL_MENU_MODE` 默认 full |
| 历史窗口 | `core/prior_turns.PRIOR_TURNS_LIMIT` / `PRIOR_ANSWER_BUDGET` | 6 轮 / 每轮 400 tok | db.prior_turns 默认 · decider 渲染 · 训练 build_messages（同一渲染函数） |
| 当前时间格式 | build_messages（v15 取 `reference_now[:10]`）· decider `date.today()` | 纯日期 | 同形断言 tests/train/test_train_prod_same_shape.py |
| 运行态注入参数 | `contract.RUNTIME_INJECTED_PARAMS` | {account_id} | ToolSpec.openai_schema 剥掉 · registry.execute 注入覆盖 · gold_script/visible_args · build_messages/visible_context · ActionGate 注入 |
| 字段清单 | `contract.visible_answer_fields`（PROSE_FIELDS 过滤） | v15 多轮行=空 | step_user.txt `{% if answer_fields %}` |
| 空 think 块 | `sft_replay._mask_empty_think` | 不监督 | SFT 样本 · 同构测试期望 |
| 教师 think 采样 | `u_build_v14_5.THINK_MAX_*` | 900 tok / 4096 字 / 不限段（不缩短） | gen_cot_v15 · behavior 探针 |
| 数据份额带宽 | `u_build_v14_5.bands` | v13 .48–.62 · l2 .10–.17 · l1 .03–.09 · chat .01–.07 · fam .04–.12 · cot .05–.30 | 份额闸（W4 首测回填） |
| 数据版本 | `pipeline/split.DATA_VERSION` | v13（case 库）；渲染版本记在 `data/sft/v15/manifest.json` 的 `render` | `assert_same_data_version` |

**怎么看喂进去的数据长什么样**：训练数据存为 parquet（`data/sft/v15/train.parquet`，每行 = `input_ids` + `loss_mask` +
`prompt_length` + 桶/轴/行为等元列，文本不直接存），`scripts/v15_data_gallery.py --parquet <文件>` 把每桶抽样解码成
Markdown：system 折叠 · 历史消息对 · 本轮 user 原文 · 逐轮 response，**被监督的 token 用 ⟦ ⟧ 包起来**，元信息含
think 非空/空块数、空块是否有梯度、菜单工具数、纯日期、字段清单四项同形标记 + 全量桶统计表。本机演练产物：
`_audit/v15_w2/gallery_dry.md`（DRY 行，教师文案是占位）；W4 建库后对真 parquet 再出一份。

## 5 · 多轮与 OPD 的设计要点（W2 之后的目标形状）

**多轮训练行（对齐 §1.1，逐项同形）**：

```
一条多轮行 = system + 真实历史消息对（1–3 轮）+ 本轮 user
历史助手内容 = 上一轮真实终答（人话；同线上 400 token 截断口径，不再是 120 字符机器 summary）
历史轮不带 think（同线上回灌口径，#7 对齐）
L1/L2 保持成对照对（该续接的与该查数的成对，24 §7 单侧对照教反）
对象选择行（#4）：context 只有 account_id，模型用 campaign.list 翻页找到 gold 指向的那条（09-02 裁定⑥）
```

**OPD（R7；设计主体已在 `24 §P4`，此处只记 v15 增量）**：

```
教师 = 裸底座 think-off，只蒸 NL 段（reply/闲聊/承接语），绝不碰工具段与信令段（24 §2 表）
prompt 集 v2 = 419 骨架 + chat_bank_v2 + S2 held-out 句式 + 多轮占比 14%→30%
★ v15 新红线 R7③「信令不糊」：行为形态跌幅 ≤1pp · 三信令表达率各不低于 R6−2pp
  ——OPD 只许改人话，不许改动作
⚠️ W0 必查 R7③ 的可测性：1pp 若小于四遍聚合 SE 的两倍，判读法要改（这正是三查的用途）
```

---

## 6 · 已裁定（Chaoyu 2026-08-31，五件；执行细节各归 W2/W3/W0）

```
① 字段清单   训练也不给（对齐线上）；「读数在场」改由 gold 终答自带数字教      → W2③
② 工具菜单   训练改全量 34；先重量预算，放不下就精简工具描述（两侧共用一份，
             精简后仍同形）                                                  → W2⑤
③ 时间格式   训练渲染改纯日期（同线上），随 W4 重建生效；DATA_VERSION 升版     → W2④
④ CoT 带宽   上沿 20%→30%（数学天花板 34%，留余量不顶满；显存非瓶颈不需预试）  → W3④
⑤ 思考率门槛 SFT 出口只记录（预注册预测带 20–50%）；≥50% 硬闸挂 R6 出口       → W0/W3
```

```
⑥ 09-02 context 形状  一次真实请求只带 account_id（+用户界面选中的 campaign_id，可选）；campaign 清单
                      **不进提示词**（真实账户几万条、每天在变），有哪些 campaign 由模型调 campaign.list 翻页
                      → core/demo_context.py（线上）与 prod_context（训练）同一形状；08-20 那版清单是演示补丁
⑦ 09-02 空 think 不监督 · CoT 不缩短 · 上限 9216/18432（§4.1 / §W3 / §W2⑤）
⑨ 09-02 运行态注入  模型脑子里装知识与策略，不装运行态身份。account_id 来自登录态 ⇒ 与 run_id 同类：
                      不进 prompt · 不进工具 schema · 不进 gold 渲染 · 沙盒/收口按当前租户注入且**模型填了也覆盖**
                      （contract.RUNTIME_INJECTED_PARAMS 一处定义；tests/core/test_runtime_injected_params.py 五条）
                      "这一版就是最终版，改好为止"——不留 v16 欠账
⑧ 09-02 画廊抓到的三条（Chaoyu 逐条看数据）：闲聊行空块有梯度 · 压舱行仍列字段清单 · WIN 行 8 轮全进
   prompt 而 gold 说"看不到"（=教撒谎）⇒ 6 轮窗口下沉到共用渲染函数 render_prior_messages；
   历史里的助手回答必须是真内容（定义库按术语名取，取不到报错，不许"好的。"占位）
```

```
⑩ 09-03 v16 全部重来  Chaoyu 原话「肯定是用新的，请忘记一切和旧的相关的事情，全部口径都是 v16，全部重新生成全部重来」。
                      起因：HEAD 代码（裁定⑨之后）生成的 case 库与 git 冻结的 v13 切分差 3 条（4 对只差 account_id 的题面
                      同形被 prompt_fingerprint 去重；本机三次基线复现，与 Modal 环境无关）。⇒ case 库 / 切分 / 训练集 / 考场
                      全部以 v16 为口径在 Modal 上从零生成；v13/v15 的冻结切分与读数不再作为任何比较的一端。
⑪ 09-03 换法三·全新栈   Chaoyu 原话「按换法三做，彻底全新换成新栈；首要目的是用新栈学新东西，训练出来的模型 0 价值、学到东西 100 价值」。
                      学生 Qwen3.5-9B · 思考教师 Qwen3.5-27B · 语言教师 Qwen3.5-4B · 测试分词 Qwen3.5-0.8B（全家换）；
                      栈 = vLLM 0.28.0 / torch 2.13.0（cu13）/ verl 0.9.0（V1 统一 trainer、FSDP2、on-policy 蒸馏）/ transformers 5.10.x /
                      flash-linear-attention 0.5.2 / flash-attn 2.8.3.post1 **源码编 sm_120**（官方轮子只到 torch2.10）。
                      新栈依赖表独立放 `modal_app/stack/`（旧 uv.lock 留给本机 K 线，不动）；Modal Volume 上旧模型/旧数据已清。
                      探针 = `modal_app/stack_probe.py`；补丁/契约重新对齐是后续小事，判据先行。
⑫ 09-03 晚 B200+最新模型  PRO 6000 也不用了（sm_120 无 TMEM/tcgen05，FA4 物理上跑不了）；一切在 **B200（sm_100）** 上配，B300 待全链通后重跑收坑。
                      模型只要最新：学生候选 **Qwen3.6-35B-A3B**（最新小 MoE，EP/GDN/MTP 全在场）或 Qwen3.8-27B 密集；思考教师 **Qwen3.8-27B**；
                      万亿旗舰不做（Chaoyu：过头了）。框架版本守则⑯（00 §5）：不是最新稳定版必须写原因，机器判据 = stack_probe versions 步。
⑬ 09-03 晚 教师换大   Chaoyu：「人话教师也用更大的模型，只要显存放得下」⇒ 思考教师与人话教师**同一个** Qwen3.8-27B（52 GB 单卡）；
                      Qwen3.5-4B 退役。OPD 逐 token 蒸馏前必须核对教师/学生 **tokenizer 完全一致**（vocab 哈希判据），不一致只能走文本级。
                      候选更大教师 Qwen3.5-122B-A10B（233 GB，两卡 EP=2）留给建库期空卡时试。
```

```
⑭ 09-04 v16 不混任何旧物料  Chaoyu 原话「不允许任何之前版本的产物混进我们这一版……新的数据、新的硬件、新的一切，完全脱离于之前的
                      版本的训练主线，一切都是 v16」。⇒ 教师换 27B 后，v14.5/v15 时代 4B/8B 的物料（v15_materials.json 的
                      reply/think、v145_defs 61 词、v145_chat_mat 90 条）**一律不复用，全部由 27B 重生成**；v13 考场 triage 文件
                      不再读（难例族收成常量 HARD_FAMILIES，v16 首遍考场后重定）。缓存文件名全带版本（v16_*），旧名物理上读不到；
                      run14–16 写在 Volume 的 v15_* 缓存搬进 cache/pre_v16_run16/ 留档。判据：建库脚本源码 grep 旧文件名 = 0（探针 stale 步）。
                      同日放行的诊断（先量后动）：CoT/行为探针过滤链每类丢弃计数 + 27B 原始思考画像（§W4′ S3-diag）。
```

**唯一还开着的批准**：W0 产物（修订版 R5 门槛表）做完后一次呈报（09-02 已口头批：按推荐执行）。

---

## 7 · 本次查证中查不到的东西（重建时必须落盘，不许再空着）

```
· 1039 行 r3 版的 think 块实测计数——parquet 不在本机；现存读数是 949 行版
  （90/4049，_audit/v15_r2/gates.json）；「20 行/60.1%」是本次按算法复算的推定值
· [CoT-v15] 命中率 63% 与预算 36253 的原始 stdout——未入库（全库 grep 无命中）
  ⇒ 已立纪律：建库 stdout 一律 tee 入 _audit/（§2.4）
· R5 考场当时 [think-mode] 判据行是否真打——logs 在训练机，本机无法核
· 训练侧菜单「16–17」的代码出处——set_tool_menus.py 只有 v8 时代 8–13 的 docstring
```
