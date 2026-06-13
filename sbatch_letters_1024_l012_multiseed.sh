#!/bin/bash
#SBATCH --job-name=emnist1024-ms
#SBATCH --partition=gpu5090
#SBATCH --nodelist=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/emnist1024_l012_multiseed_%j.out
#SBATCH --error=logs/emnist1024_l012_multiseed_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/Hongjie_Zeng/high_order_BM}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [ -x "/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python" ]; then
  export PYTHON_BIN="/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python"
else
  export PYTHON_BIN="$(command -v python)"
fi

echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
nvidia-smi || true

"${PYTHON_BIN}" -m py_compile \
  train_1000pbit_20x20_wsd_mnist20.py \
  train_twoport_4096_letters_isolet.py \
  eval_twoport_4096_letters_isolet.py \
  run_letters_1024_l012_multiseed_server.py

"${PYTHON_BIN}" run_letters_1024_l012_multiseed_server.py

echo "End time: $(date)"
