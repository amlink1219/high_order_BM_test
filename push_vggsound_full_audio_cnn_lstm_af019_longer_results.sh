#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  train_vggsound_mini20_bm.py \
  run_vggsound_full_audio_cnn_lstm_af019_longer.py \
  sbatch_vggsound_full_audio_cnn_lstm_af019_longer.sh \
  push_vggsound_full_audio_cnn_lstm_af019_longer_results.sh \
  README_vggsound_full_audio_cnn_lstm_af019_longer.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_AF022_standard_audio_cnnlstm4096_h6_lc5_e900_resume_af019 \
  runs_vggsound_full_AF023_standard_audio_cnnlstm4096_h6_lc5_e1000_resume_af022
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF022_*_stdout.log runs_vggsound_full_AF022_*_stderr.log \
  runs_vggsound_full_AF023_*_stdout.log runs_vggsound_full_AF023_*_stderr.log \
  logs/vggsound_audio_cnn_lstm_af019_long_*.out \
  logs/vggsound_audio_cnn_lstm_af019_long_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF019 longer files to commit."
else
  git commit -m "Add AF019 longer audio CNN-LSTM BM results"
fi

git push -u origin main
