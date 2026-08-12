set -x
cd /home/samwang/code/projects/verl-async-agentic-rl
V=.venv/bin/python
bash scripts/wait_for_gpu.sh 24000 900 || exit 1
for E in epoch1 epoch2; do
  $V -m syncopate.train.eval_local --model models/Qwen3-4B --adapter checkpoints/sft/v10/$E \
     --batch data/batches/v10 --split-dir data/splits/v10 --samples-per-case 8 \
     --out _audit/v10_sft_${E}.json > _audit/v10_sft_${E}.log 2>&1
  echo "[DONE] eval $E rc=$?"
done
for E in epoch1 epoch2; do
  $V -m syncopate.train.entropy --model models/Qwen3-4B --adapter checkpoints/sft/v10/$E \
     --batch data/batches/v10 --split-dir data/splits/v10 \
     --out _audit/v10_entropy_${E}.json > _audit/v10_entropy_${E}.log 2>&1
  echo "[DONE] entropy $E rc=$?"
done
$V -m syncopate.train.compare _audit/v10_base.json _audit/v10_sft_epoch1.json > _audit/v10_cmp_e1.log 2>&1
$V -m syncopate.train.compare _audit/v10_base.json _audit/v10_sft_epoch2.json > _audit/v10_cmp_e2.log 2>&1
echo "[ALL DONE]"
