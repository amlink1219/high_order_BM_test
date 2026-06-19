#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add audio STFT BM scripts and compact result files =="
git add \
  make_vggsound_full_audio_stft4096_features.py \
  merge_vggsound_full_audio_stft_shards.py \
  run_vggsound_full_audio_stft_bm.py \
  sbatch_vggsound_full_audio_stft_bm.sh \
  push_vggsound_full_audio_stft_bm_results.sh \
  README_vggsound_full_audio_stft_bm.md

if [ -f vggsound_full_audio_stft_bm_log.md ]; then
  git add vggsound_full_audio_stft_bm_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_audio_stft_bm_*.out' -o -name 'vggsound_full_audio_stft_bm_*.err' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find data_vggsound_full/features -maxdepth 1 -type f \( \
    -name 'vggsound_full_audio_stft*_summary.json' -o \
    -name 'vggsound_full_audio_stft*_manifest.csv' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type f \( \
    -name 'runs_vggsound_full_audio_stft*_stdout.log' -o \
    -name 'runs_vggsound_full_audio_stft*_stderr.log' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d -name 'runs_vggsound_full_AF*' -print0 \
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
  git commit -m "Add VGGSound full audio STFT BM results"
fi

git push -u origin main
echo "== Done =="
