#!/usr/bin/env bash
#SBATCH --job-name=vgg-p1v2
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/vggsound_phase1_video_continuation_%j.out
#SBATCH --error=logs/vggsound_phase1_video_continuation_%j.err

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
echo "Phase 1 video continuation: P1V002 + P1V003"

"$PYTHON_BIN" run_vggsound_phase1_video_continuation.py \
  --root "$ROOT" \
  --python_bin "$PYTHON_BIN" \
  --num_shards 2 \
  --video_teacher_batch_size 256 \
  --video_teacher_eval_batch_size 384 \
  --video_bm_eval_batch_size 64 \
  --p1v003_lstm_epochs 80 \
  --p1v003_lstm_batch_size 256 \
  --p1v003_lstm_eval_batch_size 384 \
  --p1v003_bm_epochs 360

echo "End time: $(date)"
