#!/usr/bin/env bash
#SBATCH --job-name=vgg-rgbm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/vggsound_rgb_motion_fused_video4096_%j.out
#SBATCH --error=logs/vggsound_rgb_motion_fused_video4096_%j.err

set -euo pipefail

ROOT=${ROOT:-/home/Hongjie_Zeng/high_order_BM}
PYTHON_BIN=${PYTHON_BIN:-/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python}

cd "$ROOT"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

echo "Start time: $(date)"
echo "Workdir: $PWD"
echo "Python: $PYTHON_BIN"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Branch: RGB+motion fused encoder -> video4096 -> standard BM"

"$PYTHON_BIN" run_vggsound_rgb_motion_fused_video4096.py \
  --root "$ROOT" \
  --python_bin "$PYTHON_BIN" \
  --extract_shards 2 \
  --num_frames 16 \
  --frame_size 224 \
  --encoder_epochs 60 \
  --bm_epochs 360

echo "End time: $(date)"
