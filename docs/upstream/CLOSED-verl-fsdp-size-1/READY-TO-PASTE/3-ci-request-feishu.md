# 飞书群 · CI 触发申请留言（包①）

> ⚠️ Slack 的 `ci-request` 频道**进不去**：它限定邮箱域名（anyscale.com / bytedance.com / together.ai）。
> ⇒ 走飞书群：https://applink.larkoffice.com/client/chat/chatter/add_by_link?link_token=772jd4f1-cd91-441e-a820-498c6614126a
> 格式照抄群里既有的 CI 申请（一段说明 + 链接 + 一句请求）。

## 正式版（推荐，直接复制）

大家好，我提交了一个 FSDP1 梯度同步相关的修复：多卡下设置 `fsdp_size=1`（本意是"不分片、纯数据并行"）会构造出 `(world_size, 1)` 的退化 device mesh 并选中 `HYBRID_SHARD`，FSDP1 随后把它钳成 `NO_SHARD`，但梯度归约仍留在那个只有 1 个 rank 的分片组上——结果是各 rank 的梯度从不同步，而且不报错、loss 会降、指标全正常。现在改为在分片维退化时显式选 `NO_SHARD`，让 FSDP 在复制维（`mesh_dim=0`）上归约；同时补了一个 2 卡回归测试（`tests/special_distributed/`，已注册进 `run_all.sh`，修复前会红）。

Issue： https://github.com/verl-project/verl/issues/7493
PR： https://github.com/verl-project/verl/pull/7494

麻烦帮忙触发一下 CI，谢谢！

---

## 备用：更短的一版（如果群里习惯短消息）

大家好，提交了一个 FSDP1 的修复：多卡 `fsdp_size=1` 会造出退化的 `(N,1)` mesh，FSDP1 钳成 `NO_SHARD` 后仍在 size-1 的分片组上归约梯度 ⇒ **各 rank 梯度静默不同步**（不报错、指标正常）。改为退化时显式选 `NO_SHARD`，并补了 2 卡回归测试。
PR： https://github.com/verl-project/verl/pull/7494 （Issue #7493）
麻烦帮忙触发一下 CI，谢谢！

---

## 备用：英文版（若群里以英文沟通）

Hi all, I submitted a fix for silent gradient desync in FSDP1: with `fsdp_size=1` on multiple GPUs, verl builds a degenerate `(world_size, 1)` device mesh and selects `HYBRID_SHARD`; FSDP1 clamps that to `NO_SHARD` but keeps reducing gradients over the size-1 shard group, so ranks never synchronize — with no error and normal-looking metrics. The fix selects `NO_SHARD` explicitly for a degenerate shard dim so FSDP reduces over the replicate dim (`mesh_dim=0`), plus a 2-rank regression test under `tests/special_distributed/` (registered in `run_all.sh`, fails before the fix).

Issue: https://github.com/verl-project/verl/issues/7493
PR: https://github.com/verl-project/verl/pull/7494

Could someone help trigger CI? Thanks!

---

## 发完之后

1. 回 PR 页面把最后一条 checklist 勾上：
   `- [ ] Once your PR is ready for CI, send a message in the ci-request channel.`
2. ⚠️ **先签 CLA 再发**（或同时）：https://cla-assistant.io/verl-project/verl?pullRequest=7494
   —— CLA 未签时维护者一般不会 review，CI 也可能不给触发。
3. 相关 CI：本 PR 的测试需要**多卡**，落在 `model.yml`（它跑 `tests/special_distributed/run_all.sh`），
   触发路径 `verl/**/*.py` 与 `tests/special_distributed/run_all.sh` 我们都改到了，会自动匹配。
