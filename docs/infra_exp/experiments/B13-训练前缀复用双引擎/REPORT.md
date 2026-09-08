# infra-probes/B13 · 训练前缀复用双引擎

状态：Q09暂停排期，首轮源码支持调查保留；真实前缀计算瓶颈触发才重开；未运行CPU/GPU验证，未建立性能结论。2026-09-08。

## 问题

GRPO同prompt多条回答是否能复用训练前缀计算且保留正确梯度？默认分别查FSDP2和Megatron，不把某后端缺实现写成另一个后端已通过，也不为凑覆盖强行移植。旧方案曾以公开普通attention dense为候选，新范围固定Volume的一MoE一dense，不能为旧方案静默增加第三个主模型；不默认支持GDN或LoRA。完整设置/预算须在实际验证前注册。

## 已查现状

verl v0.9.0与main均核对。FSDP2当前engine没有PrefixGrouper实际调用，虽有config/辅助函数；原PR4368已合入，补接线PR7202关闭但未合入。本地syncopate/train/verl_patches.py历史补丁确实包装FSDPEngineWithLMHead.forward_step，不是旧dp_actor；但未在当前栈认证，不能继承旧收益。

Megatron当前verl engine未接prefix-tree/Magi；RFC6401仍开放。评论提供独立AXRL的实现，源码有前后向测试，但不是verl stock支持，也未证明本批B200/LoRA/逐参数梯度等价。不能声称没有现成方案。源码SHA、查询状态和链接见[research.json](research.json)。

## 触发重开后的定位参考（不是当前任务）

FSDP2先以stock ON/OFF判据确认是否命中，再决定是否单独认证最小历史接线；不加载所有旧patch制造隐含变量。Megatron先完成已存在替代方案的兼容性审查，不未经登记更换整套框架。只有受支持且正确的臂才测速。

前置判据覆盖相同输入字节、分组/还原、兄弟回答隔离、因果边界、变长后缀、前缀KV梯度（不detach）、真实root调用与逐参数梯度；至少两次非零更新且缓存不跨更新失效。无共享前缀为负对照，分组成本与显存进入性能读数。full与LoRA分开登记。

## 当前结论

两引擎是互补的默认调查范围；当前都不能因为存在一个开关就直接宣称前缀复用生效。没有云端产物、checkpoint或上游提交。
