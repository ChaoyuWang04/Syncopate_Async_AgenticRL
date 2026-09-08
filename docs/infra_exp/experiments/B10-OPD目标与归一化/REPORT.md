# infra-probes/B10 · OPD目标与归一化

状态：已结论，空mask缺口已留DRAFT（未修、未发送）；任务：Q06；更新：2026-09-08。

## 问题与调查结论

不同长度样本拆成不同microbatch后，直接蒸馏的损失/梯度应保持相同全局目标。[#7200](https://github.com/verl-project/verl/issues/7200) 已由 [#7225](https://github.com/verl-project/verl/pull/7225)、SHA `594c51bc04a14d9084218b488c20d6d9ec7a897e` 修复；v0.9.0与main均已填入全局分母，本次确认性复验通过，不重复提该问题。官方forward_kl_topk是教师到学生的截断forward KL，不等同旧业务全词表学生到教师reverse KL。

进一步复现了独立缺口：正式拒绝采样可以产生全零response_mask，真实OPD dispatcher随后在 `student_mass.min()` 抛异常；全局分母为正也不能保护空局部microbatch。详细根因、已有修法边界与建议见 [DRAFT](../../../upstream/DRAFT-infra-probes-B10-opd-empty-mask/1-finding.md)。查询release SHA `483b8a009ba3a97563edee3a19887e4862b8094a`，main SHA `c8687a02e5b38172d6645cf120b71220f28cce88`；完整检索在 [research.json](research.json)。

## 设置与预注册判据

正常import verl FSDP distillation函数、agg_loss与真实dispatcher，独立float64数学参考。固定5条×4 token×7词表、长度[4,1,3,2,1]且mask有内洞；microbatch为5、2+2+1、1×5；K=V与K=3分开验证。FP32 loss/梯度最大绝对误差≤2e-6；故意局部分母必须偏离>1e-5；forward/reverse可区分；同分布零KL、正全局分母的零mask零梯度、被mask位置梯度逐位零、非零目标梯度及weight-decay-only负例均检查。

CPU02在CPU01封印后预注册：正式rejection helper输入有效mask、old=-1/rollout=-3、token_k2阈值0.1应全拒绝，old=rollout应全保留；其空mask进入真实dispatcher，global分母0/4两态均应捕获RuntimeError与真实命中行，全部数学回归重跑。不通过预期反例就不能宣称缺陷；不以返回零loss冒充optimizer跳步。

仅Modal 4 CPU/16 GiB、GPU 0；函数20分钟/内部命令总17.5分钟，实际错误持续20分钟仍未解决则暂停。入口 `python -m syncopate.infra.opd_contract_probe --cpu --out DIR`，结果 `opd-cpu.json`，缺依赖不skip。Volume `syncopate-home` 下独立 `/vol/_audit/infra-probes/B10/<run-id>/<attempt>/`；精确镜像、锁、源码与产物SHA见run索引。实际torch 2.13.0+cu130、verl 0.9.0；无模型下载。

## 测试结果

| run-id / UTC | 结果 | 证据索引 |
|---|---|---|
| batch5-opd-cpu-01 / 2026-09-08 | 数学/梯度/三种dispatcher切分通过；最大loss误差9.62e-8，最大梯度误差8.30e-9。primitive全局零分母非有限，故追加真实路径反例 | [CPU01](runs/batch5-opd-cpu-01.json) |
| batch5-opd-cpu-02 / 2026-09-08 03:18:10Z | 2 tests通过、零skip；最大loss误差2.30e-8、梯度8.30e-9；错误分母偏差2.63575，mask梯度逐位零。正式helper全拒绝/全保留对照通过；global_tokens=0与4均在verl `losses.py:356` 的student_mass.min抛RuntimeError，未进入backward。探针执行35.39秒（不含镜像构建与调度）、退出0，JSON SHA回读核验 | [CPU02](runs/batch5-opd-cpu-02.json) |

## 结论与收尾

本项局部数学与空监督存在性调查关闭：非空mask目标归一化正确，空mask路径缺口真实复现。测试通过表示复现判据满足，不表示空监督缺陷修好。未修改生产OPD；未验证真实Ray生成、FSDP2跨rank、Megatron schedule或全局optimizer/scheduler跳步，也没有性能/稳定性验收声明。

DRAFT材料已备，尚未发送或正式提交；修复与真实分布式回归由后续明确任务承接。无checkpoint产生；保留Volume中的小型源码、封印源码包、JSON和异常栈供复现。GPU费用为零；CPU账单未读取，不把35.39秒换算成已结算费用。
