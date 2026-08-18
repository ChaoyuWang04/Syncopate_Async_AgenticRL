#!/usr/bin/env python
"""磁盘盘点：把产物按「能不能删」分类，并解释理由。

    python scripts/disk_report.py            # 只报告，不动任何文件
    python scripts/disk_report.py --plan     # 额外打印可执行的删除命令（仍然不执行）

★ 为什么要有这个脚本（2026-08-18）

本项目丢过一次最终 ckpt（M7，27 GB 写盘撞上配额被静默截断，训练日志里一个字都没有）。
纪律是「判断空间要用写入探针，不能信 df」——但更前面一步是：
**先知道 300 GB 里哪些是死的。** 手工 du 一次要十分钟，而且下次还得再来一遍。

⚠️ 本脚本**永远不删东西**。分类里凡是「先提取再删」的，提取动作要单独做。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = 1024 ** 3


def size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out else 0


def fmt(n: int) -> str:
    return f"{n / GB:7.2f} GB"


def collect() -> dict[str, list[tuple[str, int, str]]]:
    """返回 {类别: [(路径, 字节, 理由)]}"""
    out: dict[str, list[tuple[str, int, str]]] = {"safe": [], "extract": [], "keep": [], "decide": []}

    # ① 瘦身残留：同目录已有 model_lora_only.pt，却还留着全量分片
    #    根因：prune_rl_ckpts 旧版 `next(glob)` 只删一个文件（2026-08-18 已修）
    for actor in sorted((ROOT / "checkpoints/grpo").glob("*/global_step_*/actor")):
        shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
        if (actor / "model_lora_only.pt").exists() and shards:
            for s in shards:
                out["safe"].append((
                    str(s.relative_to(ROOT)), s.stat().st_size,
                    "同目录已有 model_lora_only.pt ⇒ 瘦身残留（旧版只删一个分片）",
                ))

    # ② 「合并模型」里与其基座逐位相同的主权重（增量只在 lora_adapter/ 里）
    for d in sorted((ROOT / "models").glob("*")):
        if d.is_dir() and (d / "lora_adapter").exists():
            main = sum(f.stat().st_size for f in d.glob("*.safetensors"))
            if main:
                out["safe"].append((
                    f"{d.relative_to(ROOT)}/*.safetensors", main,
                    "实测与其基座逐位相同（verl 的 LoRA merger 不折增量）⇒ 纯重复；"
                    "lora_adapter/ 必须保留",
                ))

    # ③ 未瘦身的全量分片：先提取小产物，再回收
    for actor in sorted((ROOT / "checkpoints/grpo").glob("*/global_step_*/actor")):
        shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
        if shards and not (actor / "model_lora_only.pt").exists():
            run = actor.parent.parent.name
            out["extract"].append((
                str(actor.relative_to(ROOT)) + "/model_world_size_*.pt",
                sum(s.stat().st_size for s in shards),
                f"[{run}] 先 rl_ckpt_to_adapter 提成 ~127 MB 的 adapter（**每个 rank 各提一份**，"
                "E21 之后三份不一样），再删",
            ))

    # ④ SFT 选点产物：**用完就该删**，而"记得删"是手动步骤 ⇒ 这里显形
    sel = sorted((ROOT / "checkpoints/sft").glob("*/sel_f*"))
    if sel:
        out["extract"].append((
            "checkpoints/sft/*/sel_f*", sum(size(d) for d in sel),
            f"{len(sel)} 个 SFT 选点临时产物 ⇒ "
            "`python scripts/select_sft_ckpt.py <out> --keep <名字> --prune`",
        ))

    # ⑤ 优化器状态：只有「续跑同一次训练」才用得上
    opt = sorted((ROOT / "checkpoints/grpo").glob("*/global_step_*/actor/optim_*.pt"))
    if opt:
        out["decide"].append((
            "checkpoints/grpo/*/global_step_*/actor/optim_*.pt",
            sum(o.stat().st_size for o in opt),
            "断点续跑用。E21 修好后这些跑都要重来 ⇒ 除非要留 Adam 状态当证据，否则可删",
        ))

    # ⑥ 主线用不到的大件（infra 线的资产，要它们的主人决定）
    for rel, why in [
        ("models/Qwen3-30B-A3B-nf4",
         "A9 的预量化产物。E07 §9 已证「预量化救不了碎片」，§9.0 又推翻了 A9 的整个前提"
         "（expandable_segments 在分卡下本来就能用），A19 取代 A9 ⇒ **高度可疑**，"
         "且一小时可重造"),
        ("models/Qwen3-30B-A3B-Instruct-2507",
         "MoE 线（A2 三摆法）的模型，仍在队列但被 A19 gate 住 ⇒ 若 MoE 线延后，这是最大的一块"),
        ("/workspace/tmp/fa2", "flash-attn 轮子解包残留（选型已结案：官方 cu13torch2.9）"),
        ("/workspace/tmp/fa3", "同上"),
        ("/workspace/tmp/fa_probe", "同上"),
        ("/workspace/tmp/a5_backup.nsys-rep", "A5 已结案（E01 §4.5），nsys 报告可留可删"),
        ("/workspace/tmp/ray", "Ray 的临时目录；当前无 ray 进程 ⇒ 可清，但会丢正在跑的作业"),
    ]:
        pp = Path(rel) if rel.startswith("/") else ROOT / rel
        out["decide"].append((rel, size(pp), why))

    # ⑦ 必须留的
    for rel, why in [
        ("models/Qwen3-4B", "基座"),
        ("models/Qwen3-0.6B", "31 个测试要它"),
        ("models/Qwen3-4B-sft-v13-e1", "RL 起点 + 评测基座"),
        ("checkpoints/sft/v13", "epoch1/2 的 adapter —— 基线审计与 ckpt 选型要它"),
        ("_audit", "所有历史审计（在 git 里，但目录也别动）"),
        ("outputs", "每次跑的 .hydra/overrides.yaml —— 「上次到底怎么跑通的」唯一可信来源"),
    ]:
        out["keep"].append((rel, size(ROOT / rel), why))
    dumps = sum(size(p) for p in (ROOT / "checkpoints/grpo").glob("*/rollout_dumps"))
    dumps += sum(size(p) for p in (ROOT / "checkpoints/grpo").glob("*/dispatched.jsonl"))
    out["keep"].append(("checkpoints/grpo/*/{rollout_dumps,dispatched.jsonl}", dumps,
                        "漂移记账与 GRPO 组结构的原始数据 —— 探针跑完删 ckpt 但**这两样要留**"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="磁盘盘点（只读，不删）")
    ap.add_argument("--plan", action="store_true", help="打印删除命令（仍不执行）")
    args = ap.parse_args(argv)

    st = subprocess.run(["df", "-h", str(ROOT)], capture_output=True, text=True).stdout.splitlines()[-1]
    print(f"磁盘：{st}\n")

    groups = collect()
    titles = {
        "safe": "🟢 可以直接删（零信息损失，理由逐条给出）",
        "extract": "🟡 先提取小产物、再删（信息保留，体积回收）",
        "decide": "🟠 要人决定",
        "keep": "🔴 不能删",
    }
    for key in ("safe", "extract", "decide", "keep"):
        items = groups[key]
        if not items:
            continue
        total = sum(s for _, s, _ in items)
        print(f"══ {titles[key]}　小计 {fmt(total)} ══")
        agg: dict[str, list[int]] = {}
        for path, sz, why in items:
            agg.setdefault(why, [0, 0])
            agg[why][0] += sz
            agg[why][1] += 1
        for why, (sz, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"  {fmt(sz)}  ×{n:<3} {why}")
        print()

    if args.plan:
        print("══ 可执行命令（复制前请自己再看一眼）══")
        for path, sz, _ in groups["safe"]:
            print(f"  rm -f {path}    # {fmt(sz)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
