#!/bin/bash
#SBATCH --job-name=vgg-audio-bm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_audio_stft_bm_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_audio_stft_bm_%j.err

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
echo "This job reserves 2 GPUs. Audio feature extraction is mostly CPU/ffmpeg; BM training uses CUDA if available."
nvidia-smi || true

"${PYTHON_BIN}" -u run_vggsound_full_audio_stft_bm.py \
  --root . \
  --dataset_root /home/Hongjie_Zeng/datasets/VGGSound_full \
  --python_bin "${PYTHON_BIN}" \
  --max_classes 0 \
  --min_train 50 \
  --min_test 10 \
  --parallel_feature_shards 2 \
  --epochs 60 \
  --eval_every 5 \
  --quick_eval_steps 400 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --batch_size 128 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --label_init random_onehot \
  --num_workers 0 \
  --device auto \
  --sample_rate 16000 \
  --decode_duration 10.0 \
  --crop_duration 5.0 \
  --nperseg 512 \
  --noverlap 353 \
  --decode_timeout 120

echo "End time: $(date)"
