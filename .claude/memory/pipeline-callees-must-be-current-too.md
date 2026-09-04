---
name: pipeline-callees-must-be-current-too
description: "★runbook 固定了\"调谁\"不等于被调的脚本是当前版本的；09-05 逐段审查一天查出 9 类旧东西还在起作用（v8 审计喂 v16 菜单、泄漏闸静默跳过、供给脚本崩、三查读旧判卷文件）"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ed1b9c6-3ae1-544f-b9a5-207777e46c52
  modified: 2026-09-04T08:30:43.907Z
---

**把管线收成一个固定入口只解决了"谁调谁"；被调的脚本读什么、默认值指向哪、名字是哪一代的，一样要逐段核。**

Chaoyu 2026-09-05 原话（大意）：「我们不是已经把管线固定在脚本里了吗？为什么还在用 u_build_v14_5 这种名字看着就是旧版本残存的脚本？
v16 是全新版本，多轮构造、CoT、模型、硬件全换了。请确认默认跑的脚本都是最新的，v16 之前的代码和产物不再影响管线任何一部分；还要用的就改名到 v16。」

逐段审查（17 段 + supply）查出的形状，全部不报错：
```
喂进来的旧物料   menus 段并入 _audit/v8_sft_epoch1.json（08-12 v8 模型的评测审计）⇒ 每条 v16 题的菜单里有一个退役模型的行为
被跳过的闸       gates 段没传 --split-dir ⇒ 三桶泄漏闸静默跳过，脚本打印"跳过不是通过"，stage 标题却写着"三桶互斥"
崩掉的检查       supply 脚本抠源码正则；14:41 把 assert 收成 gate() 后就崩了；runbook --dry-run 只打印不执行 ⇒ 没人发现
读旧产物当新     三查脚本不接参数，默认读 v15-R3 的 judged 文件，本机旧文件还在 ⇒ 把旧读数当本次的报出来
旧名字的依赖     画廊/预算表拿 models/Qwen3-4B 存不存在决定分词器；v16 建库 import u_build_v14 的 GLOSSARY；OPD 从探针脚本 import 渲染函数
旧数字           entropy 写死 2048（其余段 12288）；L2 val 切片按 280 切（168 行时 val 为空）；OPD 渲染写死 reference_now=2026-08-20
空判据           探针"无旧物料"正则要求 open( 前缀，匹不到 ternary 里的 v145_* ⇒ 一直绿
```

**Why：** runbook 消灭的是"临场敲参数"的随机性；脚本内部的默认值、冻结输入、import 链是另一层随机性，且都是"能跑、不报错、量的是另一件事"型（[[project-mechanism-not-wired]] 第七形态、[[thresholds-calibrated-in-old-units]]）。旧名字本身就是线索：名字里带旧版本号的文件，十有八九内部还有旧版本的东西。

**How to apply：**
1. 换大版本时，对 runbook 的每一段做三问：底层脚本叫什么名（带旧版本号 ⇒ 改名并清内部）· 读了哪些不从 DATA_VERSION/model_paths 派生的文件 · 哪些数字的注释说"按 vN 标定"。
2. 判据要能在本机测试里红：runbook dry-run 输出不含旧版本名脚本（`test_runbook_references_no_old_version_scripts`）；关键脚本代码行 grep 旧物料名；供给脚本真跑一遍 rc 0。
3. dry-run 绿 ≠ 脚本能跑：只打印不执行的段，要另有一条真执行的测试。
4. 机制错误改机制（默认分支、参数、目录顺序），多样性不足加数据（扩池、教师现写），不要用放宽闸来"修"。

相关：[[modal-migration-state]] · [[registered-is-not-implemented]] · [[blank-thresholds-are-not-passes]]
