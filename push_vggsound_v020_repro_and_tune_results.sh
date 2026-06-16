#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add scripts and compact result files =="
git add \
  make_vggsound_video_encoder_features.py \
  run_vggsound_v020_repro_and_tune.py \
  sbatch_vggsound_v020_repro_and_tune.sh \
  README_vggsound_v020_repro_and_tune.md \
  push_vggsound_v020_repro_and_tune_results.sh \
  vggsound_v020_repro_and_tune_log.md

find logs -maxdepth 1 -type f \( -name 'vggsound_v020_repro_tune_*.out' -o -name 'vggsound_v020_repro_tune_*.err' \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find data_vggsound_mini/features -maxdepth 1 -type f \( \
    -name 'vggsound_mini20_videoenc_resnet50_mean_std_*_summary.json' -o \
    -name 'vggsound_mini20_videoenc_resnet50_mean_max_*_summary.json' -o \
    -name 'vggsound_mini20_videoenc_resnet50_mean_std_*_manifest.csv' -o \
    -name 'vggsound_mini20_videoenc_resnet50_mean_max_*_manifest.csv' \
  \) -print0 \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d -name 'runs_vggsound_mini20_V0[2-3][0-9]_*' -print0 \
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
  git commit -m "Add VGGSound V020 reproduction and tuning results"
fi

git push -u origin main
echo "== Done =="
