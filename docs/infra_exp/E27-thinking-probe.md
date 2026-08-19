# E27 · thinking 三臂探针：思考对这族任务值几分

> 状态：⬜ **脚本与开关已备好（CPU 判据全过），等主线新 SFT adapter 落地开跑**
> 尺子 `scripts/run_e27_think_probe.sh` · 开关判据 `scripts/check_think_mode.py`
> 开关 `SYNCOPATE_THINK=1`（**默认关 = 现行为逐字节不变**；训练路径 launch_rl 启动即拦）

## 0 · 结论卡片

|  |  |
|---|---|
| **问题** | 我们的 rollout 短（中位 4 动作 × ~95 字符 JSON）是**模板显式关掉 thinking**（`enable_thinking=False`，继承自老师包）+ SFT 参考答案短 + 任务大多饱和三层叠出来的。打开思考通道，这族任务能换到什么？ |
| **三臂** | A `think-off 裸基座`（★ 兼任修复后管线的**永久基线**）· B `think-on 裸基座`（探索）· C `think-off 新 SFT`（最新系统+版本） |
| **结果** | 〔待跑〕 |

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

## 5 · 结论

〔待跑 —— gate：主线新 SFT adapter〕

## 6 · ⛔ 推翻了什么

〔待跑〕
