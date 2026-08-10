"""给本项目的 venv 安装一个 `flash_attn.bert_padding` 垫片。

## 为什么需要

verl 0.8.0 的训练主路径已经改成 padding-free：`_compute_old_log_prob` 里
**无条件**调用 `left_right_2_no_padding()`，它一路走到

    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

也就是说 flash-attn 在 verl 0.8.0 里是**硬依赖**，`use_remove_padding=False` 关不掉它。
（`attention_utils.py` 里只给 NPU 留了后路，CUDA 没有 fallback。）

## 为什么垫片是正确解法，不是绕过

被 import 的这四个函数**全是纯 PyTorch 的 gather/scatter 工具**，
和 flash-attention 的 CUDA kernel 一点关系都没有：

    unpad_input      按 attention_mask 把 padded 张量压成变长
    pad_input        反向操作
    index_first_axis 按索引取第一维
    rearrange        einops 的维度重排

真正的注意力计算我们已经通过 `attn_implementation=sdpa` 走 PyTorch 原生实现了。
transformers 自己就带了这三个函数的等价实现（`modeling_flash_attention_utils._unpad_input`
等），签名和返回值完全一致（实测 5 个返回值，verl 用 `*_` 吸收多余的）。

所以这不是"绕过依赖"，是"把一个被错误耦合进 flash-attn 包的通用工具接回来"。
在 sm_120 上从源码编译 flash-attn 要一两个小时，为四个纯 Python 函数不值当。

## 什么时候该卸掉

装了真的 flash-attn 之后（想用它的 kernel 加速长序列时）。脚本会检测并拒绝覆盖真包。

    python scripts/install_flash_attn_shim.py            # 安装
    python scripts/install_flash_attn_shim.py --uninstall # 卸载
"""

from __future__ import annotations

import argparse
import shutil
import site
import sys
from pathlib import Path

MARKER = ".syncopate_shim"

INIT_SOURCE = '''"""Syncopate 垫片，不是真的 flash-attn。

见 scripts/install_flash_attn_shim.py 的说明。只提供 bert_padding 里那几个
纯 PyTorch 工具函数，没有任何 CUDA kernel。
"""

__version__ = "0.0.0+syncopate-shim"
__is_syncopate_shim__ = True
'''

BERT_PADDING_SOURCE = '''"""flash_attn.bert_padding 的纯 PyTorch 等价实现。

转发到 transformers 自带的同名实现（签名与返回值一致），rearrange 直接用 einops。
"""

from einops import rearrange  # noqa: F401
from transformers.modeling_flash_attention_utils import (
    _index_first_axis as index_first_axis,
    _pad_input as pad_input,
    _unpad_input as unpad_input,
)

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
'''


def site_packages() -> Path:
    for path in site.getsitepackages():
        candidate = Path(path)
        if candidate.name == "site-packages" and candidate.exists():
            return candidate
    raise RuntimeError("找不到 site-packages")


def install(target: Path) -> int:
    package = target / "flash_attn"
    if package.exists() and not (package / MARKER).exists():
        print(f"[跳过] {package} 已存在且不是垫片——看起来装了真的 flash-attn，不覆盖。")
        return 1
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(INIT_SOURCE, encoding="utf-8")
    (package / "bert_padding.py").write_text(BERT_PADDING_SOURCE, encoding="utf-8")
    (package / MARKER).write_text("installed by scripts/install_flash_attn_shim.py\n", encoding="utf-8")
    print(f"[OK] 垫片已安装 -> {package}")
    return 0


def uninstall(target: Path) -> int:
    package = target / "flash_attn"
    if not package.exists():
        print("[跳过] 没有 flash_attn 目录")
        return 0
    if not (package / MARKER).exists():
        print(f"[拒绝] {package} 不是垫片（可能是真的 flash-attn），不删。")
        return 1
    shutil.rmtree(package)
    print(f"[OK] 垫片已卸载 <- {package}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    target = site_packages()
    return uninstall(target) if args.uninstall else install(target)


if __name__ == "__main__":
    sys.exit(main())
