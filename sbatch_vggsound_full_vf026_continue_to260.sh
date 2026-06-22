#!/usr/bin/env bash
#SBATCH --job-name=vgg-vf026-c260
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_vf026_continue_to260_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_vf026_continue_to260_%j.err

set -euo pipefail

ROOT=/home/Hongjie_Zeng/high_order_BM
PY=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python

cd "$ROOT"
mkdir -p logs

echo "Start time: $(date)"
echo "Workdir: $PWD"
echo "Python: $PY"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"$PY" run_vggsound_full_vf026_continue_to260.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --out_dir "$ROOT/runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260" \
  --feature_npz "$ROOT/data_vggsound_full/features/vggsound_full_video_lstm8192_resnet50_f16_h1024_p1024_seed123.npz" \
  --target_epochs 260 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --eval_batch_size 64 \
  --num_workers 0 \
  --device auto

echo "End time: $(date)"
