#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add audio CNN-LSTM BM scripts and compact result files =="
git add \
  make_vggsound_full_audio_cnn_lstm_encoder_features.py \
  run_vggsound_full_audio_cnn_lstm_bm.py \
  sbatch_vggsound_full_audio_cnn_lstm_bm.sh \
  push_vggsound_full_audio_cnn_lstm_bm_results.sh \
  README_vggsound_full_audio_cnn_lstm_bm.md

if [ -f vggsound_full_audio_cnn_lstm_bm_log.md ]; then
  git add vggsound_full_audio_cnn_lstm_bm_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_audio_cnn_lstm_bm_*.out' -o -name 'vggsound_full_audio_cnn_lstm_bm_*.err' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type f \( \
    -name 'runs_vggsound_full_ACL001_*_stdout.log' -o \
    -name 'runs_vggsound_full_ACL001_*_stderr.log' -o \
    -name 'runs_vggsound_full_ACL002_*_stdout.log' -o \
    -name 'runs_vggsound_full_ACL002_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF009_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF009_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF010_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF010_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF011_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF011_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF012_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF012_*_stderr.log' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find ./data_vggsound_full/features -maxdepth 1 -type f \( \
    -name 'vggsound_full_audio_cnnlstm2048_*_summary.json' -o \
    -name 'vggsound_full_audio_cnnlstm2048_*_history.json' -o \
    -name 'vggsound_full_audio_cnnlstm4096_*_summary.json' -o \
    -name 'vggsound_full_audio_cnnlstm4096_*_history.json' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_full_AF009_*' -o \
    -name 'runs_vggsound_full_AF010_*' -o \
    -name 'runs_vggsound_full_AF011_*' -o \
    -name 'runs_vggsound_full_AF012_*' \
  \) -print0 \
  | while IFS= read -r -d '' d; do
      find "$d" -maxdepth 1 -type f \( \
        -name 'config.json' -o \
        -name 'history.json' -o \
        -name 'summary.json' -o \
        -name 'full_eval_best_3000.json' \
      \) -print0 | while IFS= read -r -d '' f; do git add "$f"; done
    done

echo "== Git status after add =="
git status --short

if git diff --cached --quiet; then
  echo "No staged changes. Nothing to commit."
else
  git commit -m "Add VGGSound full audio CNN-LSTM BM results"
fi

git push -u origin main
echo "== Done =="
