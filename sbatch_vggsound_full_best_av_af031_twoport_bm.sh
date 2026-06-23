#!/usr/bin/env bash
#SBATCH --job-name=vgg-avaf31
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_best_av_af031_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_best_av_af031_%j.err

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
echo "This branch supersedes the older AV006-AV010 AF029 mean/std audio branch."
echo "Video: VLF002/VF024-compatible 4096-d video LSTM feature."
echo "Audio: AF031 paper-STFT ResNet50 LSTM4096 feature."

"$PY" run_vggsound_full_best_av_af031_twoport_bm.py \
  --root "$ROOT" \
  --python_bin "$PY" \
  --video_npz "$ROOT/data_vggsound_full/features/vggsound_full_video_lstm4096_resnet50_f16_seed123.npz" \
  --audio_npz "$ROOT/data_vggsound_full/features/vggsound_full_audio_paperresnet50_lstm4096_chunks4_w500_h1024_seed123.npz" \
  --label_copies 5 \
  --eval_batch_size 32 \
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
