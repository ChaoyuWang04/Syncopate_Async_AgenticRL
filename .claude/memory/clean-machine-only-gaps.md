---
name: clean-machine-only-gaps
description: 「一条命令重建」的缺口只有换机器才暴露——手动装的东西不进依赖表，就在环境里活成隐形前提；2026-08-17 一次重建挖出五个
metadata:
  node_type: memory
  type: feedback
  modified: 2026-08-17T07:20:00.000Z
---

**2026-08-17 在一台干净机器上重建环境，文档说的「一条命令重建」全是假的** ——
照 `08 §2` 做完是「27 failed / 45 skipped」。五个缺口，**没有一个能靠 review 发现**，
因为它们在旧机器上都被"手动装过一次"这件事盖住了：

| 缺口 | 为什么只在干净机器暴露 |
|---|---|
| M9 runtime 依赖（`asyncpg`/`fastapi`/`httpx`/`uvicorn`）从没进依赖表 | 旧机器手动 `pip install` 过 ⇒ 4 个测试模块**收集阶段**就 ImportError |
| **flash-attn 不在依赖表** ⇒ `uv sync` 把它**静默卸载** | 日志只有一行 `- flash-attn`；而 verl 0.8 是**无条件** `from flash_attn.bert_padding import` ⇒ 下次 RL 启动才炸 |
| `data/external/ingested.json` 是 gitignore 的派生产物，但不在搭环境步骤里（在另一节） | 缺了 27 个测试红，失败签名长得像**数据过期**，极易误判成别的问题 |
| `pg_bootstrap.sh` 不建 `postgres` 用户 | `dpkg -x` 只解包、**不跑 maintainer 脚本**；旧机器那个用户是 apt 建的 |
| `pg_bootstrap.sh` 不设 `LD_LIBRARY_PATH`（libpq）、`LOGFILE` 写死在 PGDATA 搬走后不存在的目录、日志父目录没 chown | 旧机器 libpq5 是 apt 装进系统的；三个连环，逐个撞 |

## 提炼

**① 手动装的东西 = 环境里的隐形前提。** 它让「一条命令重建」变成一句没人验证过的话。
⇒ **凡是手动装过的，当场补进依赖表 / 脚本**，否则那次手动就成了下一台机器的地雷。

**② `uv sync` 会移除不在依赖表里的包 —— 这是"机制没接上"的一个新形态**：
不是忘了接，是**接上了又被工具拆掉**，而且拆得静默。
⇒ 判据：装完关键包之后**再 `uv sync` 一次**，看它还在不在（幂等性测试）。

**③ 「跳过」不是「通过」。** 45 条 runtime 测试在没 PG 时 skip，
而它们自己的 docstring 就写着「⚠️ 没有就跳过 —— 但**不要把跳过当通过**」。
⇒ 验收口径必须是 **0 skipped**，不是 "all passed"。同 [[blank-thresholds-are-not-passes]]。

**④ 版本标签不可信，判据是实测 —— 而且是双向的。**
flash-attn 那个轮子标着 `cxx11abiFALSE` 而我们的 torch 是 `True`，**按标签该挂，实测能用**
（import 通过 + 标准前向 / varlen 打包无跨序列泄漏 / bert_padding 往返三项数值全对）；
反过来官方 `cu12torch2.8` 那个标签看着更近，实测 `undefined symbol`。
⇒ 「import 成功不等于契约满足」的反面同样成立：**标签不匹配也不等于不能用。**

**⑤ 干净机器重建本身就是一次审计。** 它是唯一能把"隐形前提"逼出来的手段 ——
比任何 review 都有效，因为查的不是代码对不对，是**环境里有没有没写下来的东西**。

★ 顺带验证到的好消息：**数据再生链是真可复现的** ——
v11 → v12（`--freeze-from data/batches/v11`）重跑一遍，
`data/splits/v12` 的四个文件和 git 里的 **SHA-256 逐字节一致**。
⇒ [[incremental-rebuild-freeze]] 那条纪律的构造保证是成立的，不只是当时对。

相关：[[project-mechanism-not-wired]] [[machine-4x5090-constraints]]
[[blank-thresholds-are-not-passes]] [[incremental-rebuild-freeze]] [[feedback-measure-dont-infer]]

---

## ★★★ 2026-08-17 追加：判据只验了**前向**，反向坏了整整空转

换机器后装的 flash-attn 社区轮子（`cu128torch2.9cxx11abiFALSE`）：

```
前向  标准 / varlen 无跨序列泄漏 / bert_padding 往返   三项全过 ✅
反向  flash_attn_func      dq/dk/dv 全 nan
      flash_attn_varlen_   有限但**恒为 0**（参考 136/178/1450）   ← verl rmpad 走这条
```

后果：每步 `grad_norm=nan` ⇒ verl `optimizer.zero_grad()` **跳过更新** ⇒ **RL 完全空转**。

⇒ **「import 成功 ≠ 契约满足」的下一层是「前向对 ≠ 反向对」。**
⇒ ⚠️ **返回 0 比 nan 更毒**：nan 至少让上游报警；恒 0 没有报错、指标好看、什么都没学到。
   ⇒ 判据里**必须显式拦「梯度恒为 0」**，`scripts/check_flash_attn_backward.py` 就是这么写的。

★ 定位过程本身也是教训（三个假判据）：
① `/proc/<pid>/environ` 是 exec 时快照，读不到运行时的 `os.environ` 改动；
② `torch.cuda.is_initialized()` 为 False **不代表** CUDA 驱动没被摸过（枚举已固化）；
③ 先试了 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`，症状消失但根因没动，
   还撞出 `NCCL Duplicate GPU` —— **绕过症状的修法会把你带离根因**。
真正定位靠的是 **colocate 正常 × 分卡异常** 这个 A/B。

★ 好消息：官方 `cu13torch2.9` 轮子是好的，缺的 `libcudart.so.13` 用 PyPI
`nvidia-cuda-runtime<=13.2` 补上即可（驱动 595 支持 CUDA 13.2），**不必本地编译**。
