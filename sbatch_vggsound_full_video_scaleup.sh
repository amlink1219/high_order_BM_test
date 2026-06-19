#!/bin/bash
#SBATCH --job-name=vgg-video-scale
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_video_scaleup_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_full_video_scaleup_%j.err

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
echo "This job reuses existing f16 ResNet50 video features; no video feature extraction is run."
echo "It runs VF009-VF013: h8 resume to 180/240, h10 e160, h12 e160, h16 e120."
nvidia-smi || true

"${PYTHON_BIN}" -u run_vggsound_full_video_scaleup.py \
  --root . \
  --feature_npz ./data_vggsound_full/features/vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz \
  --python_bin "${PYTHON_BIN}" \
  --eval_every 5 \
  --quick_eval_steps 400 \
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
