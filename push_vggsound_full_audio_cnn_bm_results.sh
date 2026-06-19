#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add audio CNN BM scripts and compact result files =="
git add \
  make_vggsound_full_audio_cnn_encoder_features.py \
  run_vggsound_full_audio_cnn_bm.py \
  sbatch_vggsound_full_audio_cnn_bm.sh \
  push_vggsound_full_audio_cnn_bm_results.sh \
  README_vggsound_full_audio_cnn_bm.md

if [ -f vggsound_full_audio_cnn_bm_log.md ]; then
  git add vggsound_full_audio_cnn_bm_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_audio_cnn_bm_*.out' -o -name 'vggsound_full_audio_cnn_bm_*.err' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type f \( \
    -name 'runs_vggsound_full_ACF001_*_stdout.log' -o \
    -name 'runs_vggsound_full_ACF001_*_stderr.log' -o \
    -name 'runs_vggsound_full_ACF002_*_stdout.log' -o \
    -name 'runs_vggsound_full_ACF002_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF005_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF005_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF006_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF006_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF007_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF007_*_stderr.log' -o \
    -name 'runs_vggsound_full_AF008_*_stdout.log' -o \
    -name 'runs_vggsound_full_AF008_*_stderr.log' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find ./data_vggsound_full/features -maxdepth 1 -type f \( \
    -name 'vggsound_full_audio_cnn2048_*_summary.json' -o \
    -name 'vggsound_full_audio_cnn2048_*_history.json' -o \
    -name 'vggsound_full_audio_cnn4096_*_summary.json' -o \
    -name 'vggsound_full_audio_cnn4096_*_history.json' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_full_AF005_*' -o \
    -name 'runs_vggsound_full_AF006_*' -o \
    -name 'runs_vggsound_full_AF007_*' -o \
    -name 'runs_vggsound_full_AF008_*' \
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
  git commit -m "Add VGGSound full audio CNN BM results"
fi

git push -u origin main
echo "== Done =="
