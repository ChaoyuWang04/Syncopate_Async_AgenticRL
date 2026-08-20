# 包① 提交操作手册（照着做即可）

> 我这边已完成：分支、修复、测试（实弹验证修前红/修后绿）、ruff、DCO 签名提交、正文成稿。
> 你需要做的只有 **推分支 + 网页上贴两次正文 + 签 CLA + 发一条 Slack**。

分支位置：`/workspace/_upstream/verl`，分支名 `fix/fsdp-size-1-degenerate-mesh-grad-sync`
备用补丁：`/workspace/_upstream/0001-fsdp-fix-fsdp_size-1-silently-disables-gradient-sync.patch`

---

## 第 1 步 ✅ 已完成（2026-08-20）

issue = **https://github.com/verl-project/verl/issues/7493**
（标题/五字段/代码块/实测数字均已核对无误，`bug` 标签自动生效）

<details><summary>原步骤（留档）</summary>

## 第 1 步 · 先提 issue（拿编号，2 分钟）

1. 打开 https://github.com/verl-project/verl/issues/new/choose → 选 **Bug report**
2. ⚠️ Bug report 是**结构化表单**（System Info / Information / Tasks / Reproduction /
   Expected behavior），**不能整篇粘贴** ⇒ 用 **[`1-issue-BUGREPORT-FORM.md`](1-issue-BUGREPORT-FORM.md)**，
   它已按字段拆好，逐个复制即可（System Info 已是今天在上游 main `2eaaa8f` 实跑
   `scripts/diagnose.py` 的真实输出）
   - 标题用：`[BUG][FSDP1] fsdp_size=1 on multiple GPUs silently disables gradient synchronization`
   - `1-issue.md` 保留作**整篇版**（万一改用 Blank issue 就贴它）
3. 提交后**记下编号**（形如 `#7501`），下一步要用

</details>

## 第 2 步 · Fork 仓库（30 秒）

打开 https://github.com/verl-project/verl → 右上角 **Fork** → 创建到你自己账号下
（得到 `https://github.com/<你的用户名>/verl`）

## 第 3 步 · 推分支（在这台机器上跑，需要你的凭证）

```bash
cd /workspace/_upstream/verl
git remote add fork https://github.com/<你的用户名>/verl.git
git push fork fix/fsdp-size-1-degenerate-mesh-grad-sync
```

⚠️ 会要求账号密码 —— **GitHub 已不接受密码**，用 Personal Access Token：
https://github.com/settings/tokens → Generate new token (classic) → 勾 `repo` → 生成后
把那串 token 当密码粘进去（用户名填你的 GitHub 用户名）。

> 也可以在你自己电脑上做：把上面那个 `.patch` 文件拷走，在你 fork 的克隆里
> `git am 0001-*.patch` 然后 push，效果完全一样。

## 第 4 步 · 开 PR（3 分钟）

1. 推完之后 GitHub 会在 fork 页面顶部显示 **Compare & pull request**，点它
   （或直接开 https://github.com/verl-project/verl/compare ）
2. base 选 `verl-project/verl` 的 `main`，compare 选你刚推的分支
3. 标题与正文：复制 `2-pr.md`（**issue 编号 #7493 已填好，不用再改**）
4. 提交

## 第 5 步 · 签 CLA（1 分钟，PR 开完才会出现）

PR 页面会自动出现一条 **CLA assistant** 的评论 → 点里面的链接 → 按提示签署。
（提交里我已经带了 `Signed-off-by`，DCO 那一项会自动过。）

## 第 6 步 · 申请 CI（他们的规矩，不做 CI 不会跑）

去 verl Slack 的 `ci-request` 频道发一条消息，内容例如：

```
Hi, requesting CI for PR #<PR编号> ([fsdp] fix: fsdp_size=1 silently disables
gradient synchronization). It adds a 2-rank regression test under
tests/special_distributed/ (registered in run_all.sh). Thanks!
```

Slack 邀请链接：https://join.slack.com/t/verl-project/shared_invite/zt-3855yhg8g-CTkqXu~hKojPCmo7k_yXTQ
（进不去可用飞书群：https://applink.larkoffice.com/client/chat/chatter/add_by_link?link_token=772jd4f1-cd91-441e-a820-498c6614126a ）

## 第 7 步 · 可选：给 PyTorch 那条补一刀（5 分钟）

去 https://github.com/pytorch/pytorch/issues/154888 评论：说明这个 bug 在 torch main 上
仍未修（我们逐字核对过三处源码）、贴我们的真实训练后果 + verl 侧 issue 链接，请求 reopen。
正文可用 `../pytorch-background.md` 里的材料；不指望被重开，但留个交叉引用。

---

## 提交后可能被问到的三个问题（我准备了答案）

**Q: 为什么不直接让 `create_device_mesh` 别造二维 mesh？**
A: 那要同时改两个函数的契约，且 `mesh_dim_names` 会从 `("ddp","fsdp")` 变成 `("fsdp",)`，
撞 `model_merger` 的断言、影响旧 ckpt 恢复。改动面数倍，收益相同。

**Q: `NO_SHARD` 不是被 PyTorch 标记废弃了吗？**
A: 是（`FutureWarning` 指向 DDP）。但在 FSDP1 内部，它是这个拓扑唯一正确的策略；
真正的长期解是迁 FSDP2（同样的 `(N,1)` mesh 在 `fully_shard` 下已经是对的，我们实测过）。

**Q: 有没有可能只是我们的配置写错了？**
A: FSDP2 在**同一个 mesh** 上是正确的，纯 DDP 对照组也正确 —— 只有 FSDP1 的降级路径错。
另外 PyTorch 维护者自己确认过是 bug（#154888）。
