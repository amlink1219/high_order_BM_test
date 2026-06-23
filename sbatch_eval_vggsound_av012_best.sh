#!/usr/bin/env bash
#SBATCH --job-name=vgg-av012-eval
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_av012_eval_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_av012_eval_%j.err

set -euo pipefail

ROOT=/home/Hongjie_Zeng/high_order_BM
PY=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python
RUN_DIR="$ROOT/runs_vggsound_full_AV012_twoport_videolstm4096_audioaf031_lstm4096_h8_g115_lc5_e320"

cd "$ROOT"
mkdir -p logs

echo "Start time: $(date)"
echo "Workdir: $PWD"
echo "Python: $PY"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Eval-only: AV012 best.pt full Gibbs test"

"$PY" eval_vggsound_twoport_checkpoint.py \
  --ckpt "$RUN_DIR/best.pt" \
  --config_json "$RUN_DIR/config.json" \
  --out_json "$RUN_DIR/full_eval_best_evalonly_3000.json" \
  --experiment_id AV012_eval_best_3000 \
  --eval_batch_size 16 \
  --eval_steps 3000 \
  --eval_burn_in 500 \
  --eval_thin 2 \
  --label_init random_onehot \
  --label_update binary \
  --num_workers 0 \
  --device auto

echo "End time: $(date)"
