"""v15 gold 回放**不许出现工具错误** —— 两个真 bug 逼出来的结构性守卫（2026-08-30）。

⛔ 抓到的两个（都属「登记了 ≠ 接上了」，且都是**测试全绿时**发生的）：
  ① `menu()` 让信令族豁免了 case 菜单裁剪（模型看得见），但 rollout_loop 的**执行白名单**
     还在拿 `case.tool_menu` 卡 ⇒ session.report 换回 `tool_not_available`，
     每条 v15 gold 轨迹的终答前都插了一条**假报错**（= 教模型"报数一定失败"）。
  ② session handler 写成 `(ctx, **kwargs)`，而全项目约定是 `(args, ctx)` ⇒ 注册过、
     schema 也对，但**一次都没被真的执行过**，直到 gold 回放才炸 TypeError。

⇒ 判据写在「两个应当相同的东西」上：**gold 是我们自己写的正确答案，它跑出来
  就不该有任何工具报错**。这条判据不需要阈值、不会因基线漂移失效（守则①）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CODE = r"""
import asyncio, json
from pathlib import Path
from transformers import AutoTokenizer
from syncopate.domains.adcampaign import build_domain
from syncopate.pipeline.sft_replay import _ScriptedEngine, gold_script
from syncopate.pipeline.split import load_bundles
from syncopate.train.rollout_loop import RolloutConfig, run_rollout
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR

tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
reg = build_domain().registry
reg.latency_scale = 0.0
by = {}
for b in load_bundles(Path(DEFAULT_BATCH_DIR)).values():
    if b.gold:
        by.setdefault(b.verifier.expected_behavior, []).append(b)
bad = []
for beh in sorted(by):
    for b in by[beh][:3]:            # 五种行为各 3 条，够覆盖每条 v15 分支
        o = asyncio.run(run_rollout(
            b, registry=reg, tokenizer=tok, generate=_ScriptedEngine(tok, gold_script(b)),
            config=RolloutConfig(max_assistant_turns=14), rollout_id="t", run_id="t"))
        txt = tok.decode(o.prompt_ids + o.response_ids)
        for marker in ('"error"', "tool_not_available", "TypeError", "unknown_tool"):
            if marker in txt:
                bad.append([b.case_id, beh, marker])
                break
print("@@JSON@@" + json.dumps(bad))
"""


@pytest.mark.parametrize("contract", ["v14", "v15"])
def test_gold_replay_has_no_tool_errors(contract: str) -> None:
    # 契约在 import 期决定 ⇒ 必须起子进程（core/contract.py 是唯一真相源的代价）
    env = dict(os.environ, SYNCOPATE_CONTRACT=contract)
    p = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env)
    assert "@@JSON@@" in p.stdout, f"子进程失败:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    bad = json.loads(p.stdout.split("@@JSON@@", 1)[1].strip())
    assert bad == [], f"{contract} 的 gold 回放里出现工具报错（gold 是我们自己的正确答案）：{bad}"
