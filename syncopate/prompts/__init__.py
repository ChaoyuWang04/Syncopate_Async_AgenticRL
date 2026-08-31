"""Prompt 模板加载与渲染。

抄老师包的两个做法（`agent/prompts/templates.py`）：

1. **模板是 .txt 文件，不是 Python 字符串常量**。改 prompt 不用改代码，
   diff 也看得清。
2. **prompt_hash**。system prompt + 工具菜单一起哈希，落进 artifact。
   SFT 和 RL 两个阶段的 hash 必须一致，否则就是训练/推理不同分布。

   ⚠️ 老师包里这个 hash **算了、落盘了，但全仓库没有任何一处做跨阶段比对**
   （sft-truth-report T10）。我们提供 `assert_prompt_consistency` 真的去比。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

PROMPT_VERSION = "syncopate_prompt_v1"
PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """读模板原文。显式 utf-8——模板含中文，依赖平台默认编码会让 hash 漂移。"""
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def render_prompt(name: str, context: dict[str, Any]) -> str:
    """Jinja 只做变量替换和循环，不承载业务逻辑。"""
    return Template(load_prompt(name), trim_blocks=False, lstrip_blocks=False).render(**context)


def stable_hash(value: Any) -> str:
    """稳定哈希：非字符串先 json.dumps(sort_keys=True)，保证键顺序无关。"""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_hash(system_text: str, tool_menu: list[dict[str, Any]]) -> str:
    """system prompt + 工具菜单的联合指纹。SFT 和 RL 必须相等。"""
    return stable_hash([stable_hash(system_text), stable_hash(tool_menu)])[:16]


def assert_prompt_consistency(sft_hash: str, rl_hash: str) -> None:
    """真的去比，不只是算完落盘就完事。

    两阶段 prompt 不一致是最难查的一类 bug——训练时模型看到 A 分布，
    RL 时看到 B 分布，指标会莫名其妙地差，但没有任何报错。
    """
    if sft_hash != rl_hash:
        raise ValueError(
            f"prompt 指纹不一致：SFT={sft_hash} RL={rl_hash}。"
            "两阶段的 system prompt 或工具菜单被改过其中一处。"
        )


__all__ = [
    "PROMPT_VERSION", "load_prompt", "render_prompt",
    "stable_hash", "prompt_hash", "assert_prompt_consistency",
]


def load_system_prompt() -> str:
    """系统提示。**v15 换掉「最终结论格式」那一段**，v14 逐字节不变。

    ⛔ 2026-08-30 考场炸出来的：v15 把契约从「自造 JSON 壳」换成了
      「纯自然语言 + session.* 信令」，数据侧、解析侧、判分侧全改了，
      **只有说明书没改** —— system.txt 还在教模型输出 `{"behavior": ..., "answer": {...}}`
      和"不要输出隐藏推理过程"（而 v15 是 think-on）。教学面和契约互相矛盾，
      模型只能靠监督信号硬掰。文档 25 从头到尾没有一处提到 system prompt ——
      「机制在但没接上」的第 N 次，这次接的是**指令面**。

    ⚠️ 训练侧（rollout_loop）与生产侧（decider）必须用同一个函数取 ——
      两边各拼一次就是两份说明书。
    """
    from syncopate.core.contract import IS_V15

    text = load_prompt("system.txt")
    if not IS_V15:
        return text
    return text[:text.index("## 最终结论格式")] + load_prompt("final_answer_v15.txt")
