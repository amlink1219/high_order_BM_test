#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add video scale-up scripts and compact result files =="
git add \
  run_vggsound_full_video_scaleup.py \
  sbatch_vggsound_full_video_scaleup.sh \
  push_vggsound_full_video_scaleup_results.sh \
  README_vggsound_full_video_scaleup.md

if [ -f vggsound_full_video_scaleup_log.md ]; then
  git add vggsound_full_video_scaleup_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_video_scaleup_*.out' -o -name 'vggsound_full_video_scaleup_*.err' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type f \( \
    -name 'runs_vggsound_full_VF009_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF009_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF010_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF010_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF011_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF011_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF012_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF012_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF013_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF013_*_stderr.log' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_full_VF009_*' -o \
    -name 'runs_vggsound_full_VF010_*' -o \
    -name 'runs_vggsound_full_VF011_*' -o \
    -name 'runs_vggsound_full_VF012_*' -o \
    -name 'runs_vggsound_full_VF013_*' \
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
  git commit -m "Add VGGSound full video scale-up BM results"
fi

git push -u origin main
echo "== Done =="
