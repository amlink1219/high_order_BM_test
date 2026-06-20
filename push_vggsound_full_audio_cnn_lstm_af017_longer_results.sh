#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  train_vggsound_mini20_bm.py \
  run_vggsound_full_audio_cnn_lstm_af017_longer.py \
  sbatch_vggsound_full_audio_cnn_lstm_af017_longer.sh \
  push_vggsound_full_audio_cnn_lstm_af017_longer_results.sh \
  README_vggsound_full_audio_cnn_lstm_af017_longer.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_AF018_standard_audio_cnnlstm4096_h6_lc5_e500_resume_af017 \
  runs_vggsound_full_AF019_standard_audio_cnnlstm4096_h6_lc5_e700_resume_af018
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF018_*_stdout.log runs_vggsound_full_AF018_*_stderr.log \
  runs_vggsound_full_AF019_*_stdout.log runs_vggsound_full_AF019_*_stderr.log \
  logs/vggsound_audio_cnn_lstm_af017_long_*.out \
  logs/vggsound_audio_cnn_lstm_af017_long_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF017 longer files to commit."
else
  git commit -m "Add AF017 longer audio CNN-LSTM BM results"
fi

git push -u origin main
