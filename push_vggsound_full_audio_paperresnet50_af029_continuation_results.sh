#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  run_vggsound_full_audio_paperresnet50_af029_continuation.py \
  sbatch_vggsound_full_audio_paperresnet50_af029_continuation.sh \
  push_vggsound_full_audio_paperresnet50_af029_continuation_results.sh \
  README_vggsound_full_audio_paperresnet50_af029_continuation.md \
  vggsound_full_experiment_status.md || true

for d in \
  runs_vggsound_full_AF032_standard_audio_paperresnet50_meanstd4096_h6_lc5_e650_resume_af029 \
  runs_vggsound_full_AF033_standard_audio_paperresnet50_meanstd4096_h6_lc5_e850_resume_af032
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF032_*_stdout.log runs_vggsound_full_AF032_*_stderr.log \
  runs_vggsound_full_AF033_*_stdout.log runs_vggsound_full_AF033_*_stderr.log || true

git add \
  logs/vggsound_audio_paperresnet_af029_cont_*.out \
  logs/vggsound_audio_paperresnet_af029_cont_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF029 continuation files to commit."
else
  git commit -m "Add AF029 paper-ResNet audio continuation results"
fi

git push -u origin main
