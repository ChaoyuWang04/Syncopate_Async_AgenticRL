"""PEFT adapter 目录判据（固定管线 rl-adapter/sft 出口）：adapter_config.json + adapter_model.safetensors 在、张量非零、r/alpha 可读。
    python scripts/check_lora_adapter.py <adapter_dir>
退出码 0 = 可被 peft/eval_local 加载；非 0 = 别往下走。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    d = Path(sys.argv[1])
    cfg = d / "adapter_config.json"
    w = d / "adapter_model.safetensors"
    if not cfg.exists() or not w.exists():
        print(f"🔴 {d} 不是 PEFT adapter 目录（缺 {'adapter_config.json' if not cfg.exists() else 'adapter_model.safetensors'}）")
        return 1
    c = json.loads(cfg.read_text())
    from safetensors import safe_open
    n, nz, keys = 0, 0, []
    with safe_open(str(w), framework="pt") as fh:
        for k in fh.keys():
            t = fh.get_tensor(k); n += 1; nz += int(t.abs().sum().item() > 0); keys.append(k)
    print(f"[adapter] {d}: r={c.get('r')} alpha={c.get('lora_alpha')} 张量 {n} · 非零 {nz} · 例 {keys[:2]}")
    if n == 0 or nz == 0:
        print("🔴 adapter 张量为空/全零"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
