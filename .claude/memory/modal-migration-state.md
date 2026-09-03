---
name: modal-migration-state
description: ★09-04 收口：家在 Modal B200×2 + 全新栈（vLLM 0.28/verl 0.9/torch 2.13/FA4），九步探针全绿；v16 口径、Qwen3.6-35B-A3B 学生/Qwen3.8-27B 教师；S3 建库卡 CoT 成行 0；唯一入口 docs/syncopate/31
metadata:
  type: project
---

**入口文档 = `docs/syncopate/31-modal-and-new-stack.md`**（为什么/现场/学到的/进度/怎么起）；施工与判据 `26 §W4′`；读数 `08 §Modal` + `_audit/stack_probe/`；探针 `modal_app/stack_probe.py`。

**裁定链（Chaoyu 09-03/04）**：⑩ 全部口径 v16 从零重来（HEAD 生成不出冻结 v13 切分）· ⑪ 全新栈、目的是学新东西（模型 0 价值）· ⑫ PRO 6000 也不用（无 TMEM 跑不了 FA4），一切在 B200；B300 全链通后重跑收坑；万亿旗舰不做 · ⑬ 教师只要装得下就用大的 ⇒ Qwen3.8-27B 兼两角色。守则⑯（版本必须最新否则写原因，机器判据 versions 步）、⑰（一切网络重活进容器；坑表；并发 A/B）。

**进度（09-04 00:30）**：环境九步 ✅（FA4 4.0×/1297 TFLOPS · NVLink 871 GB/s=34× · MTP 快 12% · EP=2 起得来）· S0/S1 ✅（Modal 全量 951 passed）· S2 ✅（v16 两地 SHA 同）· **S3 建库 run16：前五段 ✅，CoT 蒸馏采样成功但成行 0 ⇒ 下限闸红（下一任第一件事，线索在 26 §W4′ S3 表）**· S4 sft_smoke / S5 exam_v4 已写未跑 · S6 RL / S7 OPD 未写。

**最贵的坑（细节在 31 §3 与 00 §5 ⑰）**：`modal app stop` 要 `--yes`；Modal 对象不许按环境变量条件定义；一个写者（bare 镜像+每容器 checkout）；FlashInfer 必装 AOT 包；transformers 5 的 apply_chat_template 返回 BatchEncoding；**Qwen3.5 工具调用是 XML 线格式**——凡 grep `"name":` 的地方都是雷；FLA naive 参数顺序反；secret 名是 `wandb-secret`。

**Why：** 09-03 一天从"搬到 PRO 6000"变成"换栈换卡换模型换数据版本"，是简历被指"一年前的技术栈"触发的；所有旧栈读数（5090/PCIe/vLLM 0.12）不再与新栈混比。
**How to apply：** 接手先读 31 → 26 §W4′ → `modal app list` 看有没有在跑的；改代码后 `git push` 才会到容器；每步先注册判据再跑；红了先怀疑解析器/路径再怀疑模型。
