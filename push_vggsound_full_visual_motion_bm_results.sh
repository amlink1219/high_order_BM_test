#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add scripts and compact result files =="
git add \
  make_vggsound_full_visual_motion_features.py \
  merge_vggsound_full_visual_motion_shards.py \
  run_vggsound_full_visual_motion_bm.py \
  sbatch_vggsound_full_visual_motion_bm.sh \
  push_vggsound_full_visual_motion_bm_results.sh \
  README_vggsound_full_visual_motion_bm.md

if [ -f vggsound_full_visual_motion_bm_log.md ]; then
  git add vggsound_full_visual_motion_bm_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_visual_motion_bm_*.out' -o -name 'vggsound_full_visual_motion_bm_*.err' \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find data_vggsound_full/features -maxdepth 1 -type f \( \
    -name 'vggsound_full_visual_motion_*_summary.json' -o \
    -name 'vggsound_full_visual_motion_*_manifest.csv' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d -name 'runs_vggsound_full_VF*' -print0 \
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
  git commit -m "Add VGGSound full visual motion BM results"
fi

git push -u origin main
echo "== Done =="
