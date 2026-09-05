"""`--rollout-is` 的默认值：**sequence**，并且这个决定带着一个明确的适用范围。

★ 这个默认值被改过两次，方向相反 —— 所以这里钉的不只是值，还有**理由**。

    2026-08-18  sequence → token
                依据：`chi2_seq 64.19 vs chi2_token 0.065`（差 989×）
                ⇒ ⛔ 那批数字在作废清单里（`21 §2.1`，B1+B2 污染）：
                  它们量的是「trainer 权重从没推给 rollout」那个**无界 bug**。
                  策略错位无限增长，序列级当然指数崩塌 ——
                  **那不是序列级的性质，是那个 bug 的性质。**

    2026-08-19  token → sequence（现在）
                依据①  干净基线 120 步（≈ 一个 epoch 的 88%）ESS **无衰减**：
                       前半 0.8768 / 后半 0.8734 / 斜率 +0.00016，全程 [0.78, 0.94]
                依据②  行为维度序列级明显更好：该 defer 97%（=起点）vs token 83%；
                       REJ 类 −0.031 vs **−0.188**；而任务总分**完全打平**
                依据③  ★ 可观测性的不对称：序列级的 ESS **会动**，
                       token 级恒 ≈0.999 ⇒ 那是一个**永远不会响的警报器**

⚠️⚠️ **适用范围（这条比结论本身更要紧）**

`[实测]` `seqis_long120` 的 `partial_ratio` **30 个点全是 0.0**
⇒ **没有任何一条轨迹跨越过权重版本边界** —— trainer 一步远比 rollout 慢，
  rollout 每次都早早做完在等 ⇒ **我们从来没真正跑出过 fully_async 的陈旧度条件。**
⇒ π_rollout ≈ π_train ⇒ IS 修正近乎恒等
  ⇒ 「任务总分打平」**不是"两者一样好"，是"这个条件下 IS 几乎没参与"**。
⇒ ★ **ESS 的作用没被观测出来 ≠ ESS 没有作用。** 它只是还没面对它该检测的条件。
  陈旧度真起来之后（长尾工具延迟 / rollout 更快 / sync_every 更大），这个默认值要重审。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "syncopate" / "train" / "launch_rl.py").read_text(encoding="utf-8")


def _default_of(flag: str) -> str:
    m = re.search(rf'add_argument\(\s*"{re.escape(flag)}"\s*,\s*default="([^"]+)"', SRC)
    assert m, f"找不到 {flag} 的默认值"
    return m.group(1)


def test_default_is_sequence():
    assert _default_of("--rollout-is") == "sequence"


def test_token_is_still_available_as_an_escape_hatch():
    """★ 不能把 token 删掉 —— ESS 真跌破 0.3 时它是逃生口（`06 §2.B`）。

    判据写在「两个选项都在」上，而不是「默认值对不对」上：
    删掉逃生口这件事，光看默认值是看不见的。
    """
    m = re.search(r'"--rollout-is".*?choices=\[([^\]]+)\]', SRC, re.S)
    assert m and "token" in m.group(1) and "sequence" in m.group(1)


def test_the_reversal_and_its_scope_are_recorded_at_the_decision_site():
    """★★ 这个默认值被相反地改过两次 ⇒ **理由必须钉在改它的那一行旁边**。

    ⚠️ 判据不是"注释要长"，是这三样必须在同一处能读到：
        ① 上一次的依据为什么作废（否则下一个人会拿它再改回去）
        ② 这一次的依据是什么
        ③ **适用范围** —— 陈旧度条件从没跑出来过，所以结论有边界

    ③ 最容易掉：「我们没观测到 X 有用」被读成「X 没用」是本项目反复踩的形状
    （`blank-thresholds-are-not-passes`：空门槛不等于通过）。
    """
    i = SRC.index('"--rollout-is"')
    block = SRC[max(0, i - 4000):i]
    assert "21 §2.1" in block or "作废" in block, "①：没写清上一次的依据为什么作废"
    assert "seqis_long120" in block, "②：没写清这一次的实测出处"
    assert "partial_ratio" in block, "③：没写清适用范围（陈旧度条件从没跑出来过）"
    assert "≠" in block, "③：没有把「没观测到作用」和「没有作用」区分开"


def test_guard_advice_stays_consistent_with_the_default():
    """★ 守卫里那句「ESS 跌破 0.3 就换 token」必须和默认值**保持一致**。

    默认是 sequence 时，token 是**逃生口**；
    若哪天默认翻回 token，那句话就变成「换成你已经在用的那个」—— 一句无意义的建议，
    而人会照着它去做。⇒ 这是一条真实的耦合：两处必须同时改。

    ⚠️⚠️ 这条判据是**第二版**。第一版写的是「没有脚本可以显式传 --rollout-is」，
      当场误伤 **24 处**：其中 `rl_guard.sh` 那处只是 `say` 的提示文本、不是传参，
      另外 23 处是 infra 的 A/B 实验脚本（`e20_seqis` vs `e20_tokenis` 等）——
      **扫这个参数正是它们的用途。**
      ⇒ 守则③：判据太宽会制造假警报，而假警报会训练人忽略这条判据。
        黑名单型判据落地前必须先确认它只命中你想要的那处 —— 我这次是跑了才发现。
    """
    guard = (ROOT / "scripts" / "tools" / "rl_guard.sh").read_text(encoding="utf-8")
    default = _default_of("--rollout-is")
    other = "token" if default == "sequence" else "sequence"
    hint = [l for l in guard.splitlines() if "0.3" in l and "rollout-is" in l]
    assert hint, "守卫里应当有那句 ESS 逃生口的提示"
    line = hint[0]
    assert other in line, f"默认是 {default}，逃生口该指向 {other}，而守卫写的是：{line.strip()}"
    assert "逃生口" in line, "措辞要说明它是**逃生口**，不是默认动作"
