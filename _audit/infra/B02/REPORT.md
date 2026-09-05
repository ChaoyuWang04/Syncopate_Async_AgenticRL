# B02 · 2×B200 真实 v16 全链 smoke

## 结论卡

- **链路结论**：通过。真实 v16 产物已按 `SFT → merge → Exam → RL → RL adapter → RL eval → OPD → OPD eval` 连续传递，SFT/RL/OPD 都有真实更新和可加载产物。
- **质量结论**：仅诊断，尚未通过 candidate 门槛。Exam、RL 长回复和 OPD 评测都有明确质量欠账。
- **性能结论**：无结论。B02 是边跑边修后拼成的完整证据链，不是一份同一源码、同一机器、重复计时的干净 A/B baseline。
- **是否改变默认**：不改变。仍默认 `smoke/observe`、SFT 单卡、RL 官方均匀采样；不启用动态分池、PrefixGrouper 或实验低精度。

## 1. 身份和边界

- run id：`b02_20260905a`
- profile / gate：`smoke / observe`
- 机器：Modal 2×NVIDIA B200，compute capability 10.0，driver 580.95.05。
- 学生：`Qwen3.6-35B-A3B`；教师：`Qwen3.8-27B`。
- SFT 合并模型：`models/Qwen3.6-35B-A3B-sft-v16_smoke_b02_20260905a`。
- RL adapter：`models/adapters/rl_v16_smoke_b02_20260905a/lora_adapter`。
- OPD final：`checkpoints/opd/v16_smoke_b02_20260905a/final`。

这不是一次不可变源码的整跑。施工中依次修复了合并映射、FSDP2 LoRA 导出、Exam 行为归类、OPD 真实更新计数，以及 Qwen think-on 解析/评测统计；关键后段使用 overlay `97baef783306ef07`，最终评测统计使用 `77d26c45d8b0b334`，补跑 SFT eval 使用 `47727ac279a87224`。因此本报告可以验“最终机制已接通”，不能拿总墙钟或跨段耗时写性能结论。

## 2. 逐段结果

| 段 | 结果 | 关键证据 | 边界 |
|---|---|---|---|
| SFT train | 健康通过 | 30 次更新；329s；val loss `0.5670 → 0.3083`；`ΔW=0.5426%`；峰值 80.6GB；末端约 200 supervised tok/s | 只有一次短跑，不是稳态性能样本 |
| SFT eval | 程序通过、质量诊断 | 修复后 8×2：平均 reward 0.374，行为 16/16，轨迹多样 7/8，思考统计非零 | 19 次工具错误；不是晋级门槛 |
| merge | 健康通过 | 官方 checkpoint 映射 310/310；310 个非零 `ΔW`；全局加权残差 0.284；幅度比 1.03 | 只证明 SFT 增量真实进入合并模型 |
| Exam | 链路通过、质量 WARN | 40/40 原始与判卷记录齐全，模型身份正确，40 条都有非空思考 | 6/40 判卷失败；1/40 终答仍有机器语法 |
| RL train | 健康通过、质量 WARN | 2 次真实更新；loss/grad/reward 有限且非零；权重同步约 6.33s/6.27s；step-2 checkpoint 完整 | 每步 2/8 rollout 撞到 12,288 token，clip ratio 25% |
| RL adapter | 健康通过 | FSDP2 两个 shard 导出 350 组 A/B、700 张量，350 个 B 全非零，90,008,768 bytes | 与本轮 step-2 actor 绑定 |
| RL eval | 程序通过、质量诊断 | 修复后 8×2：平均 reward 0.258；8/8 非零；行为 12/16；多样性 7/8；无评测截断 | 4 次工具错误；行为仅 75% |
| OPD train | 健康通过 | 第一次尝试无自然语言 token 合法跳过；第二次得到 84 个有效 token，完成 1 次真实 optimizer update；KL/token 0.3345；零掩码和跨 rank 判据通过 | 只是一更新 smoke，不代表训练收益 |
| OPD eval | 程序通过、质量诊断 | 修复后 8×2：平均 reward 0.271；7/8 非零；行为 12/16；无评测截断 | 9 次工具错误；多样性 5/8，低于旧诊断线 70% |

最终 manifest 是 `pipeline_ok=true`、`all_passed=false`。前者表示所有程序和产物能接着走；后者表示至少有质量 WARN，不能叫“全绿”或“candidate 通过”。

## 3. 本轮修掉的真实错误

1. Qwen3.5/3.6 think-on 的开标签由 chat template 写在 prompt 中，completion 通常从思考正文开始再输出 `</think>`。旧解析器只认显式 `<think>...</think>`，把思考泄漏进可见回复，并把思考统计成 0。Runtime、Exam、RL、OPD 和 eval 已改用同一共享解析语义。
2. 合法的审批暂停会产生 proposal 并等待用户，Exam 旧门禁却把它当成“没有行为”。现在只把这种合法暂停归为 `tool_call`，真正失败仍保持未分类。
3. verl FSDP2 的通用 merger 会留下不完整全量模型目录；现在按实际 shard 重建并只导出严格校验过的 LoRA adapter。
4. OPD 旧 smoke 可以 0 次真实更新仍退出 0；现在真实更新数、attempt/skip、有效 token、KL、final 和 completion marker 必须对账。
5. SFT/RL 健康错误与模型质量告警已经拆开：健康错误任何模式都停；smoke 的质量告警报告后继续，candidate strict 阻止晋级。

## 4. 为什么还不是性能 baseline

- 代码 overlay 在各段之间变化，失败重试和冷启动混在一起。
- RL 只有一个冷 step 和一个较热 step：step 1 为 251.11s / 188.68 tok/s，step 2 为 98.45s / 421.27 tok/s；一个较热样本不能给中位数或噪声带。
- SFT 的 200 supervised tok/s 是短跑末端累计读数，不是多次预热后的稳定窗口。
- 没有可靠的全程 GPU busy、功耗、通信占比、总费用；多次 Modal 分配的 region/主机不同，`nvidia-smi topo -m` 还没有成功落出矩阵。
- vLLM 日志显示 FlashInfer MoE shape 不在 tuning bucket、推理期仍有 Triton JIT、raw prompt API 已弃用。这些会污染冷启动和延迟。

所以 B02 的性能结论是“无结论”，下一轮必须用固定源码和尺子重跑，不能把这里的最好数字挑出来当简历 before。

## 5. 下一步必须处理

1. 先解决 RL 的未闭合/重复思考导致的 25% 长回复截断；保持 12,288 上限不变，先看实际输入和失败轨迹，不靠缩短预算掩盖问题。
2. 处理 Exam 的 6 个失败和 1 个机器壳终答；补齐 candidate 的正式门槛。
3. 解释 OPD 的 9 次工具错误和 5/8 多样性；一更新结果不能用于宣称 OPD 提升。
4. 用固定源码做 B03 重复性/噪声地板和训推/adapter 身份验证，再开始 B04 的 SFT 1 卡对 2 卡实验。
5. 把 FlashInfer tuning bucket、Triton JIT 预热、vLLM Renderer API 和完整拓扑/利用率/费用记录纳入后续 infra 实验。

## 6. 证据索引

- 最终账本：[final_47727ac2_manifest.json](rechecks/final_47727ac2_manifest.json)
- 修复后 SFT eval：[final_47727ac2_sft_eval.json](rechecks/final_47727ac2_sft_eval.json)
- Exam 门禁：[exam_run_gate.json](rechecks/exam_97baef78/exam/exam_run_gate.json)
- RL 日志与当时门禁：[rl_97baef78/run/](rechecks/rl_97baef78/run/)
- RL 原始轨迹：[rollout_dumps/](rechecks/rl_97baef78/rollout_dumps/)
- RL adapter 导出：[rl_adapter_export_manifest.json](rechecks/final_97baef78/rl_adapter_export_manifest.json)
- OPD 门禁与完成标记：[final_97baef78/](rechecks/final_97baef78/)
- 修复后 RL eval：[final_77d26c45_eval_rl.json](rechecks/final_77d26c45_eval_rl.json)
- 修复后 OPD eval：[final_77d26c45_eval_opd.json](rechecks/final_77d26c45_eval_opd.json)
- 云端逐次汇总：[stack_probe/](../../stack_probe/)

注意：RL 当时生成的旧门禁 JSON 还没有“截断属于质量 WARN”的字段；本报告依据同一日志中的两次 `response_length/clip_ratio=0.25`，按修复后的现行门禁语义记录为 WARN。保留旧 JSON 是为了不篡改原始证据。
