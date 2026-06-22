#!/usr/bin/env bash
#SBATCH --job-name=vgg-apaper
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=110G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_paper_resnet50_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_paper_resnet50_%j.err

set -euo pipefail

ROOT=/home/Hongjie_Zeng/high_order_BM
PY=/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python

cd "$ROOT"
mkdir -p logs

export OMP_NUM_THREADS=8

echo "Start time: $(date)"
echo "Workdir: $PWD"
echo "Python: $PY"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"$PY" run_vggsound_full_audio_paper_resnet50_bm.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --csv /home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv \
  --clips_root /home/Hongjie_Zeng/datasets/VGGSound_full/clips \
  --sample_rate 16000 \
  --duration 10.0 \
  --nperseg 512 \
  --noverlap 353 \
  --stft_normalization log_per_clip_zscore \
  --stft_dtype float16 \
  --stft_workers 28 \
  --stft_worker_chunksize 8 \
  --teacher_epochs 100 \
  --teacher_batch_size 128 \
  --teacher_eval_batch_size 64 \
  --teacher_export_batch_size 64 \
  --teacher_num_workers 16 \
  --teacher_export_num_workers 8 \
  --teacher_lr 0.001 \
  --teacher_weight_decay 0.0001 \
  --teacher_dropout 0.2 \
  --teacher_eval_every 2 \
  --train_crop_frames 500 \
  --sequence_num_chunks 4 \
  --sequence_chunk_frames 500 \
  --lstm_epochs 120 \
  --lstm_batch_size 512 \
  --lstm_eval_batch_size 512 \
  --lstm_num_workers 8 \
  --lstm_proj_dim 1024 \
  --lstm_hidden 1024 \
  --lstm_layers 1 \
  --lstm_lr 0.001 \
  --lstm_weight_decay 0.0001 \
  --lstm_dropout 0.25 \
  --lstm_eval_every 5 \
  --label_copies 5 \
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
