#!/usr/bin/env bash
#SBATCH --job-name=vggmini20-bm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vggsound_mini20_bm_%j.out
#SBATCH --error=logs/vggsound_mini20_bm_%j.err

set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONUNBUFFERED=1

echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON_BIN}" -m py_compile \
  train_vggsound_mini20_bm.py \
  run_vggsound_mini20_experiments.py

"${PYTHON_BIN}" -u run_vggsound_mini20_experiments.py \
  --root . \
  --python_bin "${PYTHON_BIN}" \
  --feature_npz ./data_vggsound_mini/features/vggsound_mini20_features_2048.npz \
  --total_pbits 4096 \
  --input_dim 2048 \
  --num_classes 20 \
  --label_copies 5 \
  --epochs 180 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --weight_clip 1.2 \
  --grad_clip 5.0 \
  --gamma_h 1.15 \
  --gamma_l 1.15 \
  --label_inhibit 0.3 \
  --label_update binary \
  --label_init random_onehot \
  --neg_init random_onehot \
  --eval_every 5 \
  --quick_eval_steps 600 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --seed 123 \
  --num_workers 0 \
  --device auto \
  --binarize none

echo "End time: $(date)"
