# Claude 的项目记忆

这个目录是 Claude Code 的**项目记忆**（每个文件一条事实，`MEMORY.md` 是索引）。

## 为什么放在仓库里

默认位置是 `~/.claude/projects/<项目路径>/memory/`，**不在仓库里也不进 git** ——
换机器 / 重建 pod / 重新 clone 都带不走。这是交接文档记过的一条搬家缺口。

⇒ 2026-08-13 迁到这里，原位置留了一个软链指回来：

```
~/.claude/projects/-workspace-Syncopate-Async-AgenticRL/memory
   →  <repo>/.claude/memory
```

**读写都照常**（Claude 那边完全无感），但内容跟着仓库走。

## 换机器时要做的

新机器上 clone 完仓库之后，把软链重建一次：

```bash
REPO=$(pwd)                      # 仓库根目录
SLUG=-workspace-Syncopate-Async-AgenticRL   # 按新的绝对路径改：/ 和 _ 都变成 -
TARGET=~/.claude/projects/$SLUG/memory
mkdir -p "$(dirname "$TARGET")" && rm -rf "$TARGET"
ln -s "$REPO/.claude/memory" "$TARGET"
```

⚠️ **`SLUG` 是仓库绝对路径按规则转出来的**，换了路径就会变 ——
不确定就先跑一次 Claude，看它在 `~/.claude/projects/` 下建了哪个目录，再把那个目录换成软链。
