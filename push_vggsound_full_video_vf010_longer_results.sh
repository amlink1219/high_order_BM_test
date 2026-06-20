#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  train_vggsound_mini20_bm.py \
  run_vggsound_full_video_vf010_longer.py \
  sbatch_vggsound_full_video_vf010_longer.sh \
  push_vggsound_full_video_vf010_longer_results.sh \
  README_vggsound_full_video_vf010_longer.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_VF014_video_resnet50_meanstd_f16_h8_lc5_resume360 \
  runs_vggsound_full_VF015_video_resnet50_meanstd_f16_h8_lc5_resume480
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_VF014_*_stdout.log runs_vggsound_full_VF014_*_stderr.log \
  runs_vggsound_full_VF015_*_stdout.log runs_vggsound_full_VF015_*_stderr.log \
  logs/vggsound_video_vf010_long_*.out \
  logs/vggsound_video_vf010_long_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged VF010 longer files to commit."
else
  git commit -m "Add VF010 longer video BM results"
fi

git push -u origin main
