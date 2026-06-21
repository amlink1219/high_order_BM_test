#!/usr/bin/env bash
#SBATCH --job-name=vgg-aresseq
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_resnet50_sequence_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_resnet50_sequence_%j.err

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

"$PY" run_vggsound_full_audio_resnet50_sequence_bm.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --source_audio_npz "$ROOT/data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz" \
  --n_freq 128 \
  --n_time 96 \
  --base_epochs 60 \
  --base_batch_size 512 \
  --base_eval_batch_size 512 \
  --base_num_workers 16 \
  --base_lr 0.0003 \
  --base_weight_decay 0.0005 \
  --base_dropout 0.2 \
  --base_eval_every 5 \
  --num_chunks 8 \
  --chunk_frames 32 \
  --sequence_batch_size 256 \
  --sequence_num_workers 16 \
  --lstm_epochs 80 \
  --lstm_batch_size 512 \
  --lstm_eval_batch_size 512 \
  --lstm_num_workers 16 \
  --lstm_proj_dim 1024 \
  --lstm_hidden 1024 \
  --lstm_layers 1 \
  --lstm_lr 0.001 \
  --lstm_weight_decay 0.0001 \
  --lstm_dropout 0.25 \
  --lstm_eval_every 5 \
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
