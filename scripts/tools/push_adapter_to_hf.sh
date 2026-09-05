#!/usr/bin/env bash
# 把一个 adapter 目录推到 HF 资产库（SamWang0405/Syncopate-AgenticRL）。
#   用法: bash scripts/tools/push_adapter_to_hf.sh <adapter目录> <仓库内子路径> ["提交说明"]
#   例:   bash scripts/tools/push_adapter_to_hf.sh models/adapters/e14x_graphgate adapters/e14x_graphgate
# 幂等：同内容重推 = 无 diff 不产生提交。SSH 走 ~/.ssh/id_ed25519_hf（Host hf.co 已配）。
# ⚠️ 收官/晋级流程的默认动作（2026-08-21 Chaoyu 定）：单仓库制——底座在 bases/、
#    SFT 出处链在 sft/、RL adapter 在 adapters/ 与 cand_*/；本脚本管 adapter 类推送。
set -euo pipefail
SRC="${1:?用法: push_adapter_to_hf.sh <adapter目录> <仓库内子路径> [说明]}"
DST="${2:?}"
MSG="${3:-push $DST}"
HF_CLONE=/workspace/hf_push
[ -f "$SRC/adapter_model.safetensors" ] || { echo "❌ $SRC 不是 PEFT adapter 目录"; exit 1; }
if [ ! -d "$HF_CLONE/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:SamWang0405/Syncopate-AgenticRL "$HF_CLONE"
fi
cd "$HF_CLONE"
git pull --rebase -q
mkdir -p "$(dirname "$DST")"
rm -rf "$DST"; cp -r "$OLDPWD/$SRC" "$DST" 2>/dev/null || cp -r "$SRC" "$DST"
git add -A
git diff --cached --quiet && { echo "无变化，跳过"; exit 0; }
git -c user.name="Chaoyu Wang" -c user.email="spaemtuerl@gmail.com" commit -q -m "$MSG"
git push origin main
# 判据行：远端 HEAD 必须等于本地（推真的到了，不是缓存假象）
[ "$(git rev-parse HEAD)" = "$(git ls-remote origin HEAD | cut -f1)" ] \
  && echo "✅ 已上 HF：$DST（远端 HEAD 校验一致）" \
  || { echo "🔴 远端 HEAD 不一致，推送存疑"; exit 1; }
