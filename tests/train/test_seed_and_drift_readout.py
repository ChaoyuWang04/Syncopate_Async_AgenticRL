"""固定种子 + 位移读数 —— 两条都是 2026-08-19 用一次实验换来的。

★ 起因：E17 的 KL 两臂只差 `--use-kl-loss` 一个变量，
  但 `launch_rl` **没有 seed 参数** ⇒ 它们其实是**两次独立的随机跑**。

  于是逐模板出现 4 个"显著退化"、1 个"显著提升"（|t|>2），
  而 KL 惩罚项只占损失的 **0.0019%**（kl_loss_coef 0.001 × kl_loss 0.0155
  ÷ reward 信号 0.835）—— 它**没有能力**造成那种量级的差异。
  ⇒ 那组差异量的是**跑间方差**，不是 KL 的效应。

★★ 副产品，留着当尺子：**同配置两次跑的模板级差异可以到 ±0.14。**
   低于这个幅度的模板级"差异"不该当信号看。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "syncopate" / "train" / "launch_rl.py").read_text(encoding="utf-8")

# 同配置两次跑的模板级噪声上界（E17 两臂实测：CONF +0.139 是最大的一个）
TEMPLATE_NOISE_BAND = 0.14


def test_seed_has_a_fixed_default():
    """★ 默认必须**固定** —— 两次跑之间不该多一个自由变量。

    ⚠️ 同「wandb 默认开、要关得显式」那条：
      默认随机 = 每次跑都要有人**记得**传种子，而手动步骤一定会被忘。
    """
    m = re.search(r'add_argument\(\s*"--seed",\s*type=int,\s*default=(\d+)', SRC)
    assert m, "launch_rl 没有 --seed 参数"
    assert int(m.group(1)) > 0


def test_the_seed_is_actually_passed_down():
    """★★ 加了参数却不传给 verl，就是**"机制在但没接上"** —— 本项目第一失效形状。

    判据钉在"传下去"这件事上，而不是"参数存在"。
    """
    assert "data.seed={args.seed}" in SRC, "seed 没有传给 verl 的 data 配置"


def test_the_noise_band_is_recorded_where_it_will_be_read():
    """★ 那个 ±0.14 必须写在**改种子的那一行旁边**，不能只活在对话里。

    它是判读逐模板差异的尺子 —— 尺子和被量的东西分开放，就没人会去查。
    """
    i = SRC.index('"--seed"')
    block = SRC[max(0, i - 2000):i]
    assert "0.0019%" in block, "没写清 KL 项到底占多少 ⇒ 下一个人会重新怀疑"
    assert "±0.14" in block, "没记下模板级噪声带"


def test_drift_readout_does_not_need_the_ref_forward():
    """★★ 关掉 KL 会失去 `actor/kl_loss` —— 而那是漂移的唯一仪表。

    替代品必须**不依赖 ref 前向**（否则等于把省下的 15.4% 又还回去）。
    `weight_shift` 只读 base 与 adapter，满足这一条。
    """
    src = (ROOT / "syncopate" / "train" / "weight_shift.py").read_text(encoding="utf-8")
    for forbidden in ("ref_policy", "ref_log_prob", "kl_loss"):
        assert forbidden not in src, f"位移读数依赖了 {forbidden} ⇒ 又要跑 ref 前向"
