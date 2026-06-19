#!/bin/bash
#SBATCH --job-name=vgg-audio-cnn
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_audio_cnn_bm_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_audio_cnn_bm_%j.err

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
echo "This job trains supervised audio CNN encoders on STFT128x96, exports embeddings, then trains audio-only standard BM baselines."
nvidia-smi || true

"${PYTHON_BIN}" -u run_vggsound_full_audio_cnn_bm.py \
  --root . \
  --source_audio_npz ./data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz \
  --python_bin "${PYTHON_BIN}" \
  --n_freq 128 \
  --n_time 96 \
  --cnn_width 32 \
  --teacher_batch_size 256 \
  --teacher_eval_batch_size 256 \
  --teacher_lr 0.001 \
  --teacher_weight_decay 0.0001 \
  --teacher_dropout 0.25 \
  --teacher_eval_every 5 \
  --teacher_num_workers 0 \
  --eval_every 5 \
  --quick_eval_steps 500 \
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
  --device auto

echo "End time: $(date)"
