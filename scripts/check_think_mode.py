#!/usr/bin/env python
"""E27 · think 开关的 CPU 判据（不吃卡，秒级）。

四条判据（每条都真的比，比不了就报错退出）：
  ① 默认（不设 SYNCOPATE_THINK）：渲染出的生成提示**含**空思考骨架
     `<think>\\n\\n</think>` —— 与改造前行为逐字节一致的充分标志
  ② SYNCOPATE_THINK=1：骨架**不出现**（模板放行思考）
  ③ 预算随开关切换：off 2048 / on 8192；eval 的 MAX_TURN_ACCUMULATION 同步
  ④ launch_rl 在 SYNCOPATE_THINK=1 时启动即拦（退出码非 0 且信息含"评测探针"）

用法：.venv/bin/python scripts/check_think_mode.py
（会自我 subprocess 一次来测另一态；对照计数 = 两态都必须打印「渲染长度」）
"""
import os
import subprocess
import sys

MODEL = "models/Qwen3-4B"
SCAFFOLD = "<think>\n\n</think>"


def render_and_report() -> int:
    from transformers import AutoTokenizer

    from syncopate.train import rollout_budget as rb
    from syncopate.train.rollout_loop import CHAT_TEMPLATE_KWARGS
    from syncopate.train.eval_local import MAX_TURN_ACCUMULATION

    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "system", "content": "你是业务 agent。"},
            {"role": "user", "content": "查一下 A 部门本月预算。"}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   **CHAT_TEMPLATE_KWARGS)
    on = rb.THINK_ON
    has_scaffold = SCAFFOLD in text
    print(f"[probe] THINK_ON={on} enable_thinking={CHAT_TEMPLATE_KWARGS['enable_thinking']} "
          f"渲染长度={len(text)}（对照计数，必须两态都出现）")
    print(f"[probe] 骨架出现={has_scaffold} · MAX_RESPONSE_LENGTH={rb.MAX_RESPONSE_LENGTH} "
          f"· MAX_TURN_ACCUMULATION={MAX_TURN_ACCUMULATION}")

    if on:
        ok = (not has_scaffold) and rb.MAX_RESPONSE_LENGTH == 8192 \
             and MAX_TURN_ACCUMULATION == 8192 + 2048
        print(f"[probe] 判据②③(on): {'✅' if ok else '🔴'}")
    else:
        ok = has_scaffold and rb.MAX_RESPONSE_LENGTH == 2048 \
             and MAX_TURN_ACCUMULATION == 4096
        print(f"[probe] 判据①③(off): {'✅' if ok else '🔴'}")
    return 0 if ok else 1


def main() -> int:
    if os.environ.get("_THINK_PROBE_CHILD") == "1":
        return render_and_report()

    # 态一：默认 off（当前进程，未设环境变量）
    assert os.environ.get("SYNCOPATE_THINK") is None, "别在设了 SYNCOPATE_THINK 的 shell 里跑本脚本"
    rc_off = render_and_report()

    # 态二：on（子进程，环境变量隔离——模块级常量在 import 时定死，必须换进程）
    env = dict(os.environ, SYNCOPATE_THINK="1", _THINK_PROBE_CHILD="1")
    rc_on = subprocess.run([sys.executable, __file__], env=env).returncode

    # 判据④：训练路径拦截
    env_t = dict(os.environ, SYNCOPATE_THINK="1")
    r = subprocess.run([sys.executable, "-m", "syncopate.train.launch_rl", "--dry-run"],
                       env=env_t, capture_output=True, text=True)
    blocked = r.returncode != 0 and "评测探针" in (r.stderr + r.stdout)
    print(f"[probe] 判据④(launch_rl 拦截): {'✅' if blocked else '🔴'} rc={r.returncode}")

    ok = rc_off == 0 and rc_on == 0 and blocked
    print(f"[probe] 终态: {'✅ 全过' if ok else '🔴 有失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
