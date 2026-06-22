#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  make_vggsound_full_aligned_av_features.py \
  run_vggsound_full_best_av_paper_audio_twoport_bm.py \
  sbatch_vggsound_full_best_av_paper_audio_twoport_bm.sh \
  push_vggsound_full_best_av_paper_audio_twoport_bm_results.sh \
  README_vggsound_full_best_av_paper_audio_twoport_bm.md \
  vggsound_full_experiment_status.md || true

git add \
  data_vggsound_full/features/vggsound_full_aligned_videolstm4096_audiopaperresnet4096_seed123_summary.json || true

for d in \
  runs_vggsound_full_AV006_standard_avg_videolstm4096_audiopaperresnet4096_h8_lc5_e320 \
  runs_vggsound_full_AV007_twoport_videolstm4096_audiopaperresnet4096_h8_g115_lc5_e320 \
  runs_vggsound_full_AV008_twoport_videolstm4096_audiopaperresnet4096_h8_g050_lc5_e320 \
  runs_vggsound_full_AV009_twoport_videolstm4096_audiopaperresnet4096_h8_g000_lc5_e320 \
  runs_vggsound_full_AV010_twoport_videolstm4096_audiopaperresnet4096_h6_g115_lc5_e320
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AV006_*_stdout.log runs_vggsound_full_AV006_*_stderr.log \
  runs_vggsound_full_AV007_*_stdout.log runs_vggsound_full_AV007_*_stderr.log \
  runs_vggsound_full_AV008_*_stdout.log runs_vggsound_full_AV008_*_stderr.log \
  runs_vggsound_full_AV009_*_stdout.log runs_vggsound_full_AV009_*_stderr.log \
  runs_vggsound_full_AV010_*_stdout.log runs_vggsound_full_AV010_*_stderr.log || true

git add \
  logs/vggsound_best_av_paper_audio_*.out \
  logs/vggsound_best_av_paper_audio_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged paper-audio AV two-port files to commit."
else
  git commit -m "Add VGGSound paper-audio AV two-port BM results"
fi

git push -u origin main
