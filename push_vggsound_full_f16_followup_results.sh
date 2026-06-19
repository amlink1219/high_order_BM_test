#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add f16 follow-up scripts and compact result files =="
git add \
  run_vggsound_full_f16_followup.py \
  sbatch_vggsound_full_f16_followup.sh \
  push_vggsound_full_f16_followup_results.sh \
  README_vggsound_full_f16_followup.md

if [ -f vggsound_full_f16_followup_log.md ]; then
  git add vggsound_full_f16_followup_log.md
fi

find logs -maxdepth 1 -type f \( -name 'vggsound_full_f16_followup_*.out' -o -name 'vggsound_full_f16_followup_*.err' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type f \( \
    -name 'runs_vggsound_full_VF005_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF005_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF006_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF006_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF007_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF007_*_stderr.log' -o \
    -name 'runs_vggsound_full_VF008_*_stdout.log' -o \
    -name 'runs_vggsound_full_VF008_*_stderr.log' \
  \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do git add "$f"; done

find . -maxdepth 1 -type d \( \
    -name 'runs_vggsound_full_VF005_*' -o \
    -name 'runs_vggsound_full_VF006_*' -o \
    -name 'runs_vggsound_full_VF007_*' -o \
    -name 'runs_vggsound_full_VF008_*' \
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
  git commit -m "Add VGGSound full f16 follow-up BM results"
fi

git push -u origin main
echo "== Done =="
