# E27 · thinking 三臂探针：思考对这族任务值几分

> 状态：✅ **三臂已跑完（2026-08-19 晚）**——thinking 净效果 **−0.057**（t=−4.9），
> 但 REJ/FRESH 变好、**有梯度格子 170→233**；A 臂 `_audit/e27_base_off.json` 即日起为**永久基线**
> 尺子 `scripts/run_e27_think_probe.sh` · 开关判据 `scripts/check_think_mode.py`
> 开关 `SYNCOPATE_THINK=1`（**默认关 = 现行为逐字节不变**；训练路径 launch_rl 启动即拦）

## 0 · 结论卡片

|  |  |
|---|---|
| **问题** | 我们的 rollout 短（中位 4 动作 × ~95 字符 JSON）是**模板显式关掉 thinking**（`enable_thinking=False`，继承自老师包）+ SFT 参考答案短 + 任务大多饱和三层叠出来的。打开思考通道，这族任务能换到什么？ |
| **三臂** | A `think-off 裸基座`（★ 兼任修复后管线的**永久基线**）· B `think-on 裸基座`（探索）· C `think-off 新 SFT`（最新系统+版本） |
| **结果** | A 0.356 · B 0.298 · C **0.703**。**A vs B（thinking 净效果）−0.057（t=−4.9）**：43 好/196 平/104 差；好的集中在 **REJ/FRESH**，差的集中在 FAIL/ATTR/CHAT/CONF/POL（`acted_when_should_not` 0→14 —— thinking 会把自己说服到动手）。**零梯度：有梯度 170→233、卡死 109→60** ⇒ thinking 不涨均分但把 RL 探索空间打开一半。A vs C +0.347（t=17.8） |

## 1 · 问题与预测（跑之前写死，不许事后改）

```
P1  B vs A：defer/REJ/审慎类行为 ↑（thinking 最该帮的是"该不该做"的判断，
    不是"怎么调工具"）；总分方向不确定 —— 敢押的是行为读数不是均值
P2  B vs A：12 个卡死格子里至少 1 个首次得分；96 个饱和格子基本不动
    （饱和 = 不需要思考就满分，长 CoT 帮不了满分题）
P3  B 臂耗时 3–5×（decode 长且带宽受限）；单轮思考 0.5–1.5k token
P4  B 臂 truncation=tokens 比例 ≤10%（预算 8192）；超了 = 预算不够，结论无效、加大重跑
P5  C vs A：显著为正（SFT 本来就该赢裸基座；这一臂同时是新管线的健康检查）
```

## 2 · 环境指纹

```
模型     models/Qwen3-4B（思考能力原生；SFT/RL 一直用 enable_thinking=False 关着）
评测     scripts/eval_parallel.sh → syncopate.train.eval_local（全管线同一份脚本）
契约     rollout_budget.py：off = 5120/2048（现基线）；on = 5120/8192
单轮上限 --max-new-tokens：SFT 臂 256（生产默认，实测 0 token 截断）；
         **裸基座两臂都给 2048** —— 256 会把未 SFT 模型的长输出砍断，截断与真实弱
         分不开（v13_base@256 实测截断 40.2%/parse_errors 909，已删并在此登记）
         采样三臂同一份（1.0/1.0/-1）——单变量纪律；⚠️ Qwen3 官方给 thinking 模式
         推荐 0.6/0.95，温度 1.0 下思考可能啰嗦/复读，第一轮先不动，异常再开第二轮
增量拼接 rollout 循环不重渲染 ⇒ 历史轮 <think> 留在上下文并计入 response 预算
         （这也是预算给 8192 的原因，账在 rollout_budget.py 注释里）
```

## 3 · 设计取舍（为什么是这个形状）

- **开关放在契约模块**（`rollout_budget.THINK_ON`）：开 thinking 同时动模板 kwarg 和
  response 预算两个契约量，两处都从这里取 —— 不留第二份。
- **默认必须 off**：全管线同一份脚本；off 态与改造前**逐字节相同**
  （CPU 判据①：空思考骨架 `<think>\n\n</think>` 仍在生成提示里）。
- **训练路径硬拦**：SFT 是 think-off 模板练的，开 think 会让「增量拼接 vs 整段渲染」
  逐 token 不相等（rollout_loop.py:50，有测试守着）⇒ think-on 训练的前置是**带思考的
  SFT 数据**，不是拨开关。
- **B 臂用裸基座而不是 SFT ckpt**：SFT adapter 在 think-off 分布上练过，
  很可能已把"跳过思考"练进权重 —— 挂着它测 thinking 等于测一个被压制过的通道。

## 4 · CPU 判据（已过，2026-08-19）

```
① off：骨架出现 ✅ · 预算 2048/4096 ✅（= 改造前逐字节）
② on ：骨架消失 ✅ · 预算 8192/10240 ✅ · [think-mode] 判据行打印 ✅
④ launch_rl 拦截 SYNCOPATE_THINK=1 ✅（rc=1，信息含"评测探针"）
tests/train 130 条全过（含守「增量拼接=整段渲染」的那条）
```

## 5 · 结论（2026-08-19 晚，审计 `_audit/e27_{base_off,base_on}.json` + 主线 `v13_sft_v13r2_e1_merged.json`）

```
A vs B  −0.057（t=−4.9）  thinking 在这族任务上净减分；但 REJ/FRESH ↑（+0.75 级）、
        FAIL/ATTR/CHAT/CONF/POL ↓，acted_when_should_not 0→14 —— 想多了会越界
零梯度  有梯度 170→233 · 卡死 109→60 ⇒ ★ thinking 的真实价值：解锁探索空间（对 RL 而非 eval）
A vs C  +0.347（t=17.8）  任务 SFT 完胜零训练的 thinking；B vs C +0.404
B 细节  token 截断 0%（8192 足够）· parse_ok 100% · 步数中位 3.5 · 墙钟 ~1.4×（非 3–5×）
A 澄清  A@2048 撞轮率 35.3% ≈ 被删的 @256 版 38.8% ⇒ 裸基座烧轮数是真实弱，不是 256 砍的
        （909 个 parse_errors 里只有 ~116 归 256）
A 尾巴  A 有 5 条（1.5%）tokens 截断（GEO_0004 + POL×4）：think-off 轨迹预算仍是契约值
        2048，而 B 是 8192 —— 预算不对称是开关设计的一部分，量级（1.5%）不动 −0.057 的结论
```

**决策含义**：① 零训练拨开关不值得上生产；② 要吃 thinking 红利，路径是**带思考的 SFT 数据**
（B 反超 C 的 7 题集中在 CHAT —— 判断类是它的主场）；③ `_audit/e27_base_off.json` 为永久基线；
④ `fabricated_safety_line_cap`：SFT 比裸基座 +18（6→24），与 E17 KL 臂的反向信号**汇合** ⇒ 升常驻观察。

## 6 · 预测对账（跑前写死的 P1–P5）

```
P1 半对   REJ/FRESH 变好方向命中；总分为负（当时明确说"敢押行为不押均值"）
P2 ✓✓    「≥1 个卡死格子首次得分」→ 实际解锁 ~49 个（109→60）
P3 ✗     预测慢 3–5×，实测 ~1.4× —— vLLM 批内并行把 decode 增量吸收了
P4 ✓     token 截断 0%（预算 8192 甚至偏大）
P5 ✓     SFT +0.347（t=17.8）
```
