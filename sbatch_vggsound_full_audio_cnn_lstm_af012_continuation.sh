#!/usr/bin/env bash
#SBATCH --job-name=vgg-af012-cont
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_cnn_lstm_af012_cont_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_cnn_lstm_af012_cont_%j.err

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

"$PY" run_vggsound_full_audio_cnn_lstm_af012_continuation.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --feature_npz "$ROOT/data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz" \
  --batch_size 96 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --eval_every 5 \
  --quick_eval_steps 500 \
  --quick_eval_burn_in 100 \
  --quick_eval_thin 2 \
  --full_eval_steps 3000 \
  --full_eval_burn_in 500 \
  --full_eval_thin 2 \
  --num_workers 0 \
  --device auto

echo "End time: $(date)"
