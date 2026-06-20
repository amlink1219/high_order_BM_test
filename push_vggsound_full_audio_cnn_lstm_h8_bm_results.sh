#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  train_vggsound_mini20_bm.py \
  run_vggsound_full_audio_cnn_lstm_h8_bm.py \
  sbatch_vggsound_full_audio_cnn_lstm_h8_bm.sh \
  push_vggsound_full_audio_cnn_lstm_h8_bm_results.sh \
  README_vggsound_full_audio_cnn_lstm_h8_bm.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_AF020_standard_audio_cnnlstm4096_h8_lc5_e300 \
  runs_vggsound_full_AF021_standard_audio_cnnlstm4096_h8_lc5_e500_resume_af020
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF020_*_stdout.log runs_vggsound_full_AF020_*_stderr.log \
  runs_vggsound_full_AF021_*_stdout.log runs_vggsound_full_AF021_*_stderr.log \
  logs/vggsound_audio_cnn_lstm_h8_*.out \
  logs/vggsound_audio_cnn_lstm_h8_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged audio h8 files to commit."
else
  git commit -m "Add audio CNN-LSTM h8 BM results"
fi

git push -u origin main
