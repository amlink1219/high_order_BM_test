#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add scripts and compact result files =="
git add \
  make_vggsound_audio_cnn_encoder_features.py \
  make_vggsound_audio_only_features.py \
  make_vggsound_video_encoder_features.py \
  run_vggsound_video4x5x_audio_cnn_bm.py \
  sbatch_vggsound_video4x5x_audio_cnn_bm.sh \
  README_vggsound_video4x5x_audio_cnn_bm.md \
  push_vggsound_video4x5x_audio_cnn_bm_results.sh \
  vggsound_video4x5x_audio_cnn_bm_log.md

find logs -maxdepth 1 -type f \( -name 'vggsound_video4x5x_audio_cnn_bm_*.out' -o -name 'vggsound_video4x5x_audio_cnn_bm_*.err' \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find data_vggsound_mini/features -maxdepth 1 -type f \( \
    -name 'vggsound_mini20_audio_cnn_*_summary.json' -o \
    -name 'vggsound_mini20_audio_cnn_*_history.json' -o \
    -name 'vggsound_mini20_audio_m96_t64_per_mel_zscore_sigmoid_summary.json' -o \
    -name 'vggsound_mini20_audio_m96_t64_per_mel_zscore_sigmoid_manifest.csv' \
  \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_mini20_V036_*' -o \
    -name 'runs_vggsound_mini20_V037_*' -o \
    -name 'runs_vggsound_mini20_V038_*' -o \
    -name 'runs_vggsound_mini20_V039_*' -o \
    -name 'runs_vggsound_mini20_V040_*' -o \
    -name 'runs_vggsound_mini20_V041_*' -o \
    -name 'runs_vggsound_mini20_V042_*' -o \
    -name 'runs_vggsound_mini20_V043_*' \
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
  git commit -m "Add VGGSound video 4x/5x and audio CNN BM results"
fi

git push -u origin main
echo "== Done =="
