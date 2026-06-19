#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  train_vggsound_mini20_bm.py \
  run_vggsound_full_audio_cnn_lstm_af012_continuation.py \
  sbatch_vggsound_full_audio_cnn_lstm_af012_continuation.sh \
  push_vggsound_full_audio_cnn_lstm_af012_continuation_results.sh \
  README_vggsound_full_audio_cnn_lstm_af012_continuation.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_AF016_standard_audio_cnnlstm4096_h6_lc5_e260_resume_af012 \
  runs_vggsound_full_AF017_standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF016_*_stdout.log runs_vggsound_full_AF016_*_stderr.log \
  runs_vggsound_full_AF017_*_stdout.log runs_vggsound_full_AF017_*_stderr.log \
  logs/vggsound_audio_cnn_lstm_af012_cont_*.out \
  logs/vggsound_audio_cnn_lstm_af012_cont_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF012 continuation files to commit."
else
  git commit -m "Add AF012 longer audio CNN-LSTM BM continuation"
fi

git push -u origin main
