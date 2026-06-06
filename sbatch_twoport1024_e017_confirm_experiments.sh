#!/usr/bin/env bash
#SBATCH --job-name=twoport1024-e017-confirm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/twoport1024_e017_confirm_%j.out
#SBATCH --error=logs/twoport1024_e017_confirm_%j.err

set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM
mkdir -p logs

export PYTHONUNBUFFERED=1
PYTHON=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python

"${PYTHON}" -u run_twoport1024_e017_confirm_experiments.py \
  --root . \
  --data_dir . \
  --teacher_dir ./runs_twoport1024_teacher_latefusion_lam05 \
  --epochs 100 \
  --early_stop_patience 10 \
  --quick_eval_steps 800 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --batch_size 50 \
  --eval_batch_size 128 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --label_inhibit 0.3 \
  --label_init random_onehot \
  --num_workers 2
