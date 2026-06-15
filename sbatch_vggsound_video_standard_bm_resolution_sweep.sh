#!/usr/bin/env bash
#SBATCH --job-name=vggvideo-bm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vggsound_video_standard_bm_resolution_%j.out
#SBATCH --error=logs/vggsound_video_standard_bm_resolution_%j.err

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
  make_vggsound_mini_features.py \
  train_vggsound_mini20_bm.py \
  run_vggsound_video_standard_bm_resolution_sweep.py

"${PYTHON_BIN}" -u run_vggsound_video_standard_bm_resolution_sweep.py \
  --root . \
  --python_bin "${PYTHON_BIN}" \
  --reuse_audio_npz ./data_vggsound_mini/features/vggsound_mini20_features_2048.npz \
  --hidden_factor 1.0 \
  --epochs 180 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --eval_every 5 \
  --quick_eval_steps 600 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --label_init random_onehot \
  --seed 123 \
  --num_workers 0 \
  --device auto \
  --binarize none \
  --video_fps 4 \
  --decode_timeout 180

echo "End time: $(date)"
