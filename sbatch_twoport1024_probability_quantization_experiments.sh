#!/usr/bin/env bash
#SBATCH --job-name=twoport1024-probquant
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/twoport1024_probquant_%j.out
#SBATCH --error=logs/twoport1024_probquant_%j.err

set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM
mkdir -p logs

export PYTHONUNBUFFERED=1
PYTHON=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python

"${PYTHON}" -u run_twoport1024_probability_quantization_experiments.py \
  --root . \
  --data_dir . \
  --teacher_dir ./runs_twoport1024_teacher_latefusion_lam05 \
  --eval_steps 3000 \
  --eval_burn_in 500 \
  --eval_thin 2 \
  --eval_batch_size 128 \
  --num_workers 2 \
  --label_init random_onehot \
  --eval_seed 20260608
