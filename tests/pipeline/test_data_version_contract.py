"""`--batch` 与 `--split-dir` 是**必须同时动的一对**，只动一个必须硬失败。

★ 起因（2026-08-18 晚，为 SFT 选点流程做上机前检查时查出来的）

`entropy.py` 默认 `data/batches/v3` + `data/splits/v3`，
`eval_local.py` 默认 `data/batches/v2` + `data/splits/v2`
—— 而 `data/batches/v2` 与 `data/batches/v3` 在本机**根本不存在**。

两个失效方向，严重程度差很远：

    ① 两个都用默认        `FileNotFoundError`，**是响的**，只浪费一次上机
    ② 🔴 只传 `--batch data/batches/v13`、忘了 `--split-dir`
       ⇒ 拿 v3 的 eval 桶 id 去 v13 的 batch 里读，**24/24 全部成功，不报错**
         [实测] v3 eval 64 条 vs v13 343 条、交集仅 49 ⇒ 量的是另一个 case 集；
         且那 24 条里 **4 条落在 v13 的 sft/rl 桶**（模型训过的题）⇒ 熵被记忆压低。

⇒ 而决策位熵**正是决定 RL 起点的那把尺子** ——
  这就是记过多次的第七形态：**默认值指向了另一件事，且不报错**。

判据形状：「两个东西应当相同」，非黑即白、不需要阈值（守则①）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from syncopate.pipeline.split import (
    DATA_VERSION, DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR,
    assert_same_data_version, data_version_of,
)

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_are_the_same_version():
    """两个默认值必须同版本 —— 它们此前一个 v3 一个 v2，各写各的。"""
    assert data_version_of(DEFAULT_BATCH_DIR) == DATA_VERSION
    assert data_version_of(DEFAULT_SPLIT_DIR) == DATA_VERSION


def test_defaults_actually_exist_on_disk():
    """★ 默认值必须指向**真实存在**的目录。

    此前 `data/batches/v2` 和 `data/batches/v3` 都不存在 ——
    一个指向不存在路径的默认值，等于一个保证会浪费一次上机的陷阱。
    """
    assert (ROOT / DEFAULT_BATCH_DIR).is_dir(), f"{DEFAULT_BATCH_DIR} 不存在"
    assert (ROOT / DEFAULT_SPLIT_DIR).is_dir(), f"{DEFAULT_SPLIT_DIR} 不存在"


def test_matching_versions_pass_and_return_the_version():
    assert assert_same_data_version("data/batches/v13", "data/splits/v13") == "v13"


def test_the_exact_silent_failure_combo_now_hard_fails():
    """★ 钉住那个**真实发生过**的组合：只改了 `--batch`，`--split-dir` 还是旧默认。"""
    with pytest.raises(ValueError, match="数据版本不一致"):
        assert_same_data_version("data/batches/v13", "data/splits/v3")


def test_mismatch_fails_in_both_directions():
    """反向也要失败 —— 判据不该只挡一个方向。"""
    with pytest.raises(ValueError):
        assert_same_data_version("data/batches/v3", "data/splits/v13")


def test_trailing_slash_does_not_change_the_verdict():
    """路径写法不该改变判据（尾斜杠是最常见的手滑）。"""
    assert assert_same_data_version("data/batches/v13/", "data/splits/v13") == "v13"


def test_entrypoint_defaults_come_from_the_shared_constant():
    """★ 两个入口的默认值必须**取自共用常量**，不许各自写死。

    ⚠️ 这条是源码判据，因为「将来有人又写死一个版本」这件事**还没发生**，
       没有行为可测 —— 只有扫源码看得见（同 `truncation` 组那条）。
    ⚠️ 它挡的是 v14 重建时的复发：换版本只该改 `split.py` 那一行。
    """
    import re
    for rel in ("syncopate/train/entropy.py", "syncopate/train/eval_local.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        # 找 --batch / --split-dir 的 add_argument 行，其 default 不许是字面路径
        for flag in ("--batch", "--split-dir"):
            for line in src.splitlines():
                if f'"{flag}"' in line and "add_argument" in line:
                    assert not re.search(r'default\s*=\s*["\']data/', line), (
                        f"{rel} 的 {flag} 又写死了字面路径：{line.strip()}\n"
                        f"⇒ 应该用 pipeline/split.py 的 DEFAULT_BATCH_DIR / DEFAULT_SPLIT_DIR")


def test_v3_eval_bucket_really_differs_from_current(  # noqa: D103
):
    """把「为什么这条判据值得存在」钉成实测，而不是留在注释里。

    ⚠️ 这条依赖仓库里真的有 v3 与 v13 的 split —— 都在（v3 是历史遗留）。
       ★ 如果哪天 v3 被清掉了，这条会红 ⇒ 那时把它删掉即可，
         但**别把它改成 skip**（「跳过不是通过」）。
    """
    from syncopate.pipeline.split import load_bucket
    v3 = load_bucket(ROOT / "data/splits/v3", "eval")
    cur = load_bucket(ROOT / f"data/splits/{DATA_VERSION}", "eval")
    assert set(v3) != set(cur), "两个版本的 eval 桶若相同，这条判据就没有存在意义"
    # 关键在于：v3 的 id 在当前版本里**大部分仍然存在** ⇒ 所以读得进去、不报错
    assert len(set(v3) & set(cur)) > 0, "有交集才会静默成功，这正是危险所在"
