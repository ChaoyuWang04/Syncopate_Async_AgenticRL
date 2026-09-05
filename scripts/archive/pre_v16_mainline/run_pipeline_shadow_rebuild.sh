#!/usr/bin/env bash
# 管线正确性验证：把整条数据链（外部数据 → v11 → v12 → v13 → 切分 → 双 parquet）
# **从头重跑到影子目录**，与现役数据逐字节/逐内容对比。
#
#   bash scripts/run_pipeline_shadow_rebuild.sh          # 全链，约几十分钟 CPU
#
# ★ 判据全是「两个东西应当相同」：
#   外部数据      git diff 为空（ingested.json 等在版本管理里）
#   batches v13   与 data/batches/v13 逐字节 diff -r 为空
#   splits v13    四个 json 与 data/splits/v13 SHA-256 一致（08 §3 曾在干净机器上验过 v12）
#   sft/rl parquet 与现役 parquet 逐行逐 token 相同（parquet 字节级可因元数据差，比内容）
#   门禁          D 族 + L 族在影子批上全过
#
# ⚠️ 影子目录 data/shadow_rebuild/ —— **现役数据一个字节不碰**，验证完可整目录删。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SH=data/shadow_rebuild
mkdir -p "$SH"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "═══ 0/6 外部数据（写在原位，git diff 即判据）═══"
$PY scripts/make_test_external_data.py
$PY scripts/ingest_external.py
# ⚠️ xlsx 的字节判据曾抓到 openpyxl 内嵌时间戳（已在 write_excel 钉死为 2026-01-01）。
#   git 里的旧版带的是当年的时间戳 ⇒ xlsx 与 git 比**单元格内容**，其余文件仍逐字节。
if git diff --quiet -- data/external ':(exclude)*.xlsx'; then
  say "✅ 外部数据（xlsx 之外）与 git 逐字节一致"
else
  say "🔴 外部数据（xlsx 之外）有 diff："; git diff --stat -- data/external ':(exclude)*.xlsx'; exit 1
fi
$PY - <<'PYEOF'
import io, subprocess, sys
from openpyxl import load_workbook
for f in ['data/external/safety_lines/2026-W30.xlsx', 'data/external/safety_lines/2026-W32.xlsx']:
    old = subprocess.run(['git', 'show', f'HEAD:{f}'], capture_output=True).stdout
    a = [[c.value for c in r] for ws in load_workbook(io.BytesIO(old)) for r in ws.iter_rows()]
    b = [[c.value for c in r] for ws in load_workbook(f) for r in ws.iter_rows()]
    if a != b:
        print(f'🔴 {f} 单元格内容与 git 不同'); sys.exit(1)
print('✅ xlsx 单元格内容与 git 逐一致（字节差异只来自旧版内嵌时间戳）')
PYEOF

say "═══ 1/6 生成 v11（freeze 祖链的根）═══"
$PY -m syncopate cases generate --spec configs/buckets/v11.yaml --out $SH/batches/v11
$PY scripts/set_tool_menus.py --batch $SH/batches/v11 --sft-audit _audit/v8_sft_epoch1.json

say "═══ 2/6 生成 v12（freeze-from 影子 v11）═══"
$PY -m syncopate cases generate --spec configs/buckets/v12.yaml --out $SH/batches/v12
$PY scripts/set_tool_menus.py --batch $SH/batches/v12 --sft-audit _audit/v8_sft_epoch1.json \
    --freeze-from $SH/batches/v11

say "═══ 3/6 生成 v13（freeze-from 影子 v12）═══"
$PY -m syncopate cases generate --spec configs/buckets/v13.yaml --out $SH/batches/v13
$PY scripts/set_tool_menus.py --batch $SH/batches/v13 --sft-audit _audit/v8_sft_epoch1.json \
    --freeze-from $SH/batches/v12

say "── 对比 batches v13：影子 vs 现役（逐字节）──"
if diff -rq $SH/batches/v13 data/batches/v13 > /tmp/shadow_batch_diff.txt 2>&1; then
  say "✅ batches v13 逐字节一致（$(find data/batches/v13 -type f | wc -l) 个文件）"
else
  say "🔴 batches v13 有差异（前 10 行）："; head -10 /tmp/shadow_batch_diff.txt; exit 1
fi

say "═══ 4/6 切分 v13 ═══"
$PY -m syncopate data split --batch $SH/batches/v13 --out $SH/splits/v13
for f in eval_cases sft_cases rl_cases; do
  a=$(sha256sum $SH/splits/v13/$f.json | cut -d' ' -f1)
  b=$(sha256sum data/splits/v13/$f.json | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then say "✅ splits/$f.json SHA-256 一致"
  else say "🔴 splits/$f.json 不一致"; exit 1; fi
done
# split_report.json 含自引用的 batch_dir（影子目录固有差异）⇒ 剔除该键后逐键比
$PY - <<'PYEOF'
import json, sys
def strip(p):
    d = json.load(open(p)); d.get("args", d).pop("batch_dir", None)
    for v in d.values():
        if isinstance(v, dict): v.pop("batch_dir", None)
    return d
a = strip("data/shadow_rebuild/splits/v13/split_report.json")
b = strip("data/splits/v13/split_report.json")
if a == b: print("✅ splits/split_report.json 一致（仅自引用 batch_dir 剔除）")
else: print("🔴 splits/split_report.json 有实质差异"); sys.exit(1)
PYEOF

say "═══ 5/6 构建双 parquet ═══"
$PY -m syncopate data build --pool sft --batch $SH/batches/v13 --out $SH/sft/v13 \
    --split-dir $SH/splits/v13 --val-every 6 --model models/Qwen3-4B
$PY -m syncopate data build --pool rl --batch $SH/batches/v13 --out $SH/rl/v13 \
    --split-dir $SH/splits/v13 --model models/Qwen3-4B

say "── 对比 parquet：影子 vs 现役（逐行逐 token；parquet 元数据可不同，比内容）──"
$PY - <<'EOF'
import sys
import pandas as pd

def canon(df, kind):
    if kind == "sft":
        df = df.sort_values("case_id").reset_index(drop=True)
        return [(r.case_id, list(r.input_ids), list(r.loss_mask)) for r in df.itertuples()]
    # rl：case_id 在 extra_info 里（无顶层列）；路径字段指向各自目录，剔除后比
    out = []
    for r in df.itertuples():
        extra = {k: v for k, v in dict(r.extra_info).items()
                 if k not in ("batch_dir", "artifact_root")}
        out.append((extra["case_id"], str(list(r.prompt)), str(sorted(extra.items()))))
    return sorted(out)

ok = True
for kind, cur, sh in [("sft", "data/sft/v13", "data/shadow_rebuild/sft/v13"),
                      ("rl", "data/rl/v13", "data/shadow_rebuild/rl/v13")]:
    for name in ("train", "val"):
        a = canon(pd.read_parquet(f"{cur}/{name}.parquet"), kind)
        b = canon(pd.read_parquet(f"{sh}/{name}.parquet"), kind)
        if a == b:
            print(f"✅ {kind}/{name}: {len(a)} 行内容逐一致")
        else:
            bad = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
            print(f"🔴 {kind}/{name}: {bad} 行不同（现役 {len(a)} vs 影子 {len(b)}）")
            ok = False
sys.exit(0 if ok else 1)
EOF

say "═══ 6/6 门禁（在影子批上跑 D 族 + L 族）═══"
$PY scripts/check_data_gates.py --batch $SH/batches/v13 --split-dir $SH/splits/v13

say "🎉 全链复现验证通过：外部数据 · 三代 batches · 切分 · 双 parquet · 门禁"
say "   影子目录 $(du -sh $SH | cut -f1)，确认后可删：rm -rf $SH"
