#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add scripts and compact result files =="
git add \
  make_vggsound_aligned_av_features.py \
  make_vggsound_audio_cnn_encoder_features.py \
  make_vggsound_audio_only_features.py \
  make_vggsound_video_encoder_features.py \
  run_vggsound_audio4x_aligned_twoport.py \
  sbatch_vggsound_audio4x_aligned_twoport.sh \
  README_vggsound_audio4x_aligned_twoport.md \
  push_vggsound_audio4x_aligned_twoport_results.sh \
  vggsound_audio4x_aligned_twoport_log.md

find logs -maxdepth 1 -type f \( -name 'vggsound_audio4x_aligned_twoport_*.out' -o -name 'vggsound_audio4x_aligned_twoport_*.err' \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find data_vggsound_mini/features -maxdepth 1 -type f \( \
    -name 'vggsound_mini20_audio_cnn_e1024_per_dim_minmax_seed123_summary.json' -o \
    -name 'vggsound_mini20_audio_cnn_e1024_per_dim_minmax_seed123_history.json' -o \
    -name 'vggsound_mini20_audio_cnn_e4096_per_dim_minmax_seed123_summary.json' -o \
    -name 'vggsound_mini20_audio_cnn_e4096_per_dim_minmax_seed123_history.json' -o \
    -name 'vggsound_mini20_aligned_video4096_audio4096_summary.json' \
  \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_mini20_V044_*' -o \
    -name 'runs_vggsound_mini20_V045_*' -o \
    -name 'runs_vggsound_mini20_V046_*' -o \
    -name 'runs_vggsound_mini20_V047_*' -o \
    -name 'runs_vggsound_mini20_V048_*' \
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
  git commit -m "Add VGGSound audio4x aligned two-port BM results"
fi

git push -u origin main
echo "== Done =="
