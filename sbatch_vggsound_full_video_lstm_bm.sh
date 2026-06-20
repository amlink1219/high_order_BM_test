#!/usr/bin/env bash
#SBATCH --job-name=vgg-vlstm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_video_lstm_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_video_lstm_%j.err

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

"$PY" run_vggsound_full_video_lstm_bm.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --csv /home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv \
  --clips_root /home/Hongjie_Zeng/datasets/VGGSound_full/clips \
  --num_shards 1 \
  --shard_devices auto \
  --num_frames 16 \
  --video_fps 4 \
  --frame_size 224 \
  --decode_timeout 120 \
  --teacher_batch_size 512 \
  --teacher_eval_batch_size 512 \
  --teacher_num_workers 4 \
  --proj_dim 512 \
  --lstm_hidden 512 \
  --lstm_layers 1 \
  --teacher_lr 0.001 \
  --teacher_weight_decay 0.0001 \
  --teacher_dropout 0.25 \
  --teacher_eval_every 5 \
  --eval_batch_size 64 \
  --cd_k 3 \
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
