#!/usr/bin/env bash
#SBATCH --job-name=vgg-p2vid
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/vggsound_backbone_video_%j.out
#SBATCH --error=logs/vggsound_backbone_video_%j.err

set -euo pipefail

ROOT=/home/Hongjie_Zeng/high_order_BM
PYTHON_BIN=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python

cd "$ROOT"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "Start time: $(date)"
echo "Workdir: $PWD"
echo "Python: $PYTHON_BIN"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "P2V001: EfficientNet-B3 video backbone -> LSTM4096 -> standard BM"

"$PYTHON_BIN" run_vggsound_backbone_upgrade_video.py \
  --root "$ROOT" \
  --python_bin "$PYTHON_BIN" \
  --backbone efficientnet_b3 \
  --num_frames 16 \
  --frame_size 300 \
  --resize_mode direct \
  --num_shards 2 \
  --frame_batch_size 6 \
  --lstm_proj_dim 768 \
  --lstm_hidden 768 \
  --lstm_epochs 80 \
  --lstm_batch_size 192 \
  --lstm_eval_batch_size 384 \
  --bm_epochs 360 \
  --bm_eval_batch_size 64

echo "End time: $(date)"
