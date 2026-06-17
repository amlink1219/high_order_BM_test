#!/bin/bash
#SBATCH --job-name=vgg-a4-av2p
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio4x_aligned_twoport_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio4x_aligned_twoport_%j.err

set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM
mkdir -p logs

PYTHON_BIN="/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python"
export TORCH_HOME="/home/Hongjie_Zeng/.cache/torch"

echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
nvidia-smi || true

"${PYTHON_BIN}" -u run_vggsound_audio4x_aligned_twoport.py \
  --root . \
  --python_bin "${PYTHON_BIN}" \
  --epochs 220 \
  --eval_every 5 \
  --quick_eval_steps 600 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --label_init random_onehot \
  --num_workers 0 \
  --device auto \
  --frame_size 224 \
  --video_fps 4 \
  --audio_cnn_batch_size 48 \
  --audio_cnn_eval_batch_size 128 \
  --audio_cnn_lr 0.001 \
  --audio_cnn_weight_decay 0.0001 \
  --audio_cnn_dropout 0.2 \
  --audio_cnn_eval_every 5 \
  --label_inhibit 0.3 \
  --label_condition both \
  --label_update binary \
  --neg_init random_onehot

echo "End time: $(date)"
