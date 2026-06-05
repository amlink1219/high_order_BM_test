#!/usr/bin/env bash
#SBATCH --job-name=twoport1024-next
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=twoport1024_next_%j.out
#SBATCH --error=twoport1024_next_%j.err

set -euo pipefail

# Run from the directory where sbatch was submitted. Submit this script from the
# project root that contains train_twoport_1024_optimization_wsd.py.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Uncomment and edit these lines if your cluster needs an environment setup.
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate pbit

export PYTHONUNBUFFERED=1

IMAGE_CKPT="${IMAGE_CKPT:-./runs_rbm_wsd_lc5_p1000_20x20_mnist20crop_e100/best.pt}"
AUDIO_CKPT="${AUDIO_CKPT:-./runs_audioonly_mlp_raw507_zsig/best.pt}"
DATA_DIR="${DATA_DIR:-.}"

python -u run_twoport1024_server_next_experiments.py \
  --root . \
  --data_dir "${DATA_DIR}" \
  --image_ckpt "${IMAGE_CKPT}" \
  --audio_ckpt "${AUDIO_CKPT}" \
  --teacher_dir ./runs_twoport1024_teacher_latefusion_lam05 \
  --teacher_lambda_audio 0.5 \
  --teacher_eval_steps 3000 \
  --teacher_eval_burn_in 500 \
  --teacher_eval_thin 2 \
  --distill_teacher_temperature 1.0 \
  --distill_epochs 40 \
  --distill_start_epoch 1 \
  --early_stop_patience 6 \
  --batch_size 50 \
  --eval_batch_size 128 \
  --num_workers 2
