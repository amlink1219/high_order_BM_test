#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  run_vggsound_full_best_av_af031_twoport_bm.py \
  sbatch_vggsound_full_best_av_af031_twoport_bm.sh \
  push_vggsound_full_best_av_af031_twoport_bm_results.sh \
  README_vggsound_full_best_av_af031_twoport_bm.md \
  make_vggsound_full_aligned_av_features.py \
  vggsound_full_experiment_status.md \
  vggsound_full_best_av_af031_twoport_bm_log.md || true

git add \
  data_vggsound_full/features/vggsound_full_aligned_videolstm4096_audioaf031_lstm4096_seed123_summary.json || true

for d in \
  runs_vggsound_full_AV011_standard_avg_videolstm4096_audioaf031_lstm4096_h8_lc5_e320 \
  runs_vggsound_full_AV012_twoport_videolstm4096_audioaf031_lstm4096_h8_g115_lc5_e320 \
  runs_vggsound_full_AV013_twoport_videolstm4096_audioaf031_lstm4096_h8_g050_lc5_e320 \
  runs_vggsound_full_AV014_twoport_videolstm4096_audioaf031_lstm4096_h8_g000_lc5_e320 \
  runs_vggsound_full_AV015_twoport_videolstm4096_audioaf031_lstm4096_h6_g115_lc5_e320
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AV011_*_stdout.log runs_vggsound_full_AV011_*_stderr.log \
  runs_vggsound_full_AV012_*_stdout.log runs_vggsound_full_AV012_*_stderr.log \
  runs_vggsound_full_AV013_*_stdout.log runs_vggsound_full_AV013_*_stderr.log \
  runs_vggsound_full_AV014_*_stdout.log runs_vggsound_full_AV014_*_stderr.log \
  runs_vggsound_full_AV015_*_stdout.log runs_vggsound_full_AV015_*_stderr.log || true

git add \
  logs/vggsound_best_av_af031_*.out \
  logs/vggsound_best_av_af031_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF031 audio-video fusion files to commit."
else
  git commit -m "Add AF031 audio-video fusion results"
fi

git push -u origin main
