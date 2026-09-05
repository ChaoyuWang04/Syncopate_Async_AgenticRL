# B01 · 上云前管线认证

## 结论卡

- **状态**：通过。
- **一句话**：本机能验证的代码、runbook 和负向门禁没有遗留失败；本机受限环境跑不了的 5 项已转到 Modal CPU 容器，并在完整训练依赖、PostgreSQL 和 Redis 下 5/5 通过。
- **边界**：这只证明上云前检查和测试接线可信，不证明 B200 训练质量或性能。
- **是否改变默认**：固定入口默认保持 `smoke/observe`；`candidate/strict` 仍必须显式选择。

## 验了什么

1. `scripts/v16_pipeline.sh` 的默认 profile、逐 run 目录、阶段清单和账本语义。
2. SFT → merge → RL → RL adapter → OPD 的上游产物身份守卫。
3. SFT、Exam、RL、OPD 健康闸的 `PASS / WARN / BLOCK_NEXT / FATAL` 分层。
4. 错模型、旧目录回退、短 candidate、坏 checkpoint、RL 无权重同步和 OPD 零真实更新等负对照。
5. Qwen think-on 的“开标签在 prompt、闭标签在 completion”真实线格式。

## 测试结果

- 本机目标测试：`183 passed, 1 skipped`。
- 本机广覆盖回归：`815 passed, 275 skipped, 2 deselected, 1 xfailed`；无失败。
- 本机没有硬补环境：一个 PostgreSQL 测试被沙箱禁止 socket、三个 `asyncio.to_thread` 测试受本机线程环境影响、一个等价测试缺训练 extra 中的 `prefix-grouper`。
- 上述 5 项随后在 Modal CPU 目标镜像中定向执行：`5 passed, 1 warning in 22.85s`。
- Shell 语法、改动模块 `py_compile` 与 `git diff --check` 均通过。

云端定向测试证据：
[summary_2026-09-05_051523_pytest.json](../../stack_probe/summary_2026-09-05_051523_pytest.json)。

## 对后续的约束

- 本机不具备目标 CUDA、B200、vLLM/verl 或服务环境时，不在本机重配；把最小定向验证放到 Modal。
- smoke 的质量缺口可以记 WARN 后继续收集后段证据；身份错误、NaN、坏产物和零真实更新仍必须停止。
- B01 不是性能基线。B02 只提供机械链路读数；可重复的性能、显存、通信和成本基线从 B03 的固定源码目标机器证据开始。
