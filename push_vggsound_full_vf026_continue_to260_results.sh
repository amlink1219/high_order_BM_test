#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  run_vggsound_full_vf026_continue_to260.py \
  sbatch_vggsound_full_vf026_continue_to260.sh \
  push_vggsound_full_vf026_continue_to260_results.sh \
  README_vggsound_full_vf026_continue_to260.md || true

git add \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/config.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/history.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/summary.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/full_eval_best_3000.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/vf026_continue_to260_precheck.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/summary_eval_only_epoch149.json \
  runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260/full_eval_best_3000_eval_only_epoch149.json || true

git add \
  runs_vggsound_full_VF026_continue_to260_stdout.log \
  runs_vggsound_full_VF026_continue_to260_stderr.log \
  logs/vggsound_vf026_continue_to260_*.out \
  logs/vggsound_vf026_continue_to260_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged VF026 continuation files to commit."
else
  git commit -m "Add VF026 continuation to epoch 260"
fi

git push -u origin main
