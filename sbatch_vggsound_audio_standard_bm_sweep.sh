#!/bin/bash
#SBATCH --job-name=vggaudio-bm
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_bm_%j.out
#SBATCH --error=/home/Hongjie_Zeng/high_order_BM/logs/vggsound_audio_bm_%j.err

set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM
mkdir -p logs

PYTHON_BIN="/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python"

echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
nvidia-smi || true

"${PYTHON_BIN}" -u run_vggsound_audio_standard_bm_sweep.py \
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
  --batch_size 24 \
  --eval_batch_size 64 \
  --cd_k 3 \
  --lr 0.0002 \
  --momentum 0.6 \
  --weight_decay 0.0 \
  --label_init random_onehot \
  --seed 123 \
  --num_workers 0 \
  --device auto

echo "End time: $(date)"
