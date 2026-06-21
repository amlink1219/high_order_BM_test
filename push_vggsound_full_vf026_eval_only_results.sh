#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  run_vggsound_full_vf026_eval_only.py \
  sbatch_vggsound_full_vf026_eval_only.sh \
  push_vggsound_full_vf026_eval_only_results.sh \
  README_vggsound_full_vf026_eval_only.md || true

git add \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/config.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/history.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/summary.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/full_eval_best_3000.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/vf026_eval_only_precheck.json || true

git add \
  runs_vggsound_full_VF026_eval_only_stdout.log \
  runs_vggsound_full_VF026_eval_only_stderr.log \
  logs/vggsound_vf026_eval_only_*.out \
  logs/vggsound_vf026_eval_only_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged VF026 eval-only files to commit."
else
  git commit -m "Add VF026 eval-only result"
fi

git push -u origin main
