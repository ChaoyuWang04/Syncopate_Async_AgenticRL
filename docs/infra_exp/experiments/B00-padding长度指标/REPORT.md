# infra-probes/B00 · padding 宽度与长度指标

状态：已结论 / 已关项；任务 Q01，分组 G1.1；负责人：当前实验任务。

## 问题与调查结论

同一 response token/mask 的有效长度不变，只改 padding 宽度，verl 的 response_length/clip_ratio 是否变化？必须区分张量边界占比、生成预算耗尽和真实 finish_reason。查当前 release/main、指标定义和真实调用方，不从旧报告认定现行 bug。

## 设置与判据

CPU、无模型、无 GPU。固定长度 [2,4]，padding 宽度 4/12，生成预算 12；再覆盖真实 length 结束与 stop 正好占满宽度的对照。先核当前源码，再用真实 compute_data_metrics 验张量边界行为；真实结束原因独立记账，不能从张量反推出不存在的字段。保留原 response token 与有效 mask。

局部判据：实际最大长度两臂均为4；若源码使用张量宽度则 clip_ratio 应为0.5/0，若使用真实预算则两臂为0；不同实现按其明确契约判定，不事后改阈值。需验证当前官方调用方是否固定按预算补齐，不能只凭手造动态 padding 判上游有 bug。

本机相关测试 → Modal CPU（2 CPU、8 GiB、每次15分钟以内、最多2批；本项预算上限 $2）。源码/镜像/入口指纹写 runs 索引。无需新建数据、下载模型、修改训练算法；目标依赖缺失或源码身份不符就停止本批。

## 测试结果

本机：本地真实结束原因消费测试 16 passed；独立证据目录复用/越界保护 1 passed。云端最终结果见下节；背景见 [背景检索](research.json)。

当前稳定 v0.9.0（483b8a0）与 main（2f07745）的 metric_utils blob 完全相同（c9c8c1e）；指标使用张量宽度，官方 agent_loop 按 rollout_config.response_length 补齐。PR #5794 关闭但未合并，#6675 与 #5969 已合并，分别处理空序列与样本级 padding，并未改变本指标语义。只做 CPU 确认，不据此宣称默认路径有 bug。

入口：`PYTHONPATH=. modal run --detach modal_app/infra_probes.py --run-id <new-id>`。执行真实安装包函数并断言源码 blob；四组对照覆盖宽度4/12、预算12耗尽、stop恰好满宽度；目标上复跑上述17项，无 skip 才通过。原始数据和源码快照按各run独立落 `syncopate-home:/_audit/infra-probes/B00/<run-id>/`。

首批 `b00-cpu-01` 在容器导入阶段失败（入口未加入镜像源码路径），未执行指标测试。已停止该 app 并修复路径；第二批 `b00-cpu-02` 沿用相同判据，独立目录。见 [失败索引](runs/b00-cpu-01.json)。

## 结论与下一步

第二批真实安装包验证通过：固定有效长度[2,4]，padding4/12得到 clip_ratio=0.5/0；真实 length 与 stop 恰好满宽度都给0.5，指标不能区分结束原因。目标 CPU 回归 **17 passed、0 skipped**，函数运行19.82秒；结果已从 Volume 回读并核对SHA。见 [成功索引](runs/b00-cpu-02.json)。

结论：这是张量边界指标；本次未证明官方固定预算调用方存在 padding 误报，本地消费逻辑已正确分开 finish_reason。Q01从活动队列移除，不新增重复上游提案。只验证函数与本地消费，不扩展到所有 V1/TQ、多轮真实 rollout；未测性能和GPU。

保留约2.7MB源码快照、源码/日志/JSON，无 checkpoint 或需清理大文件。两次CPU批次在登记范围内；实际账单未取得，不能把函数耗时当最终计费。首批导入失败已按精确app停止。
