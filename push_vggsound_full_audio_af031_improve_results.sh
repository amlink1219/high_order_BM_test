#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  make_vggsound_full_audio_bm_feature_variants.py \
  run_vggsound_full_audio_af031_improve.py \
  sbatch_vggsound_full_audio_af031_continue_capacity.sh \
  sbatch_vggsound_full_audio_af031_encoding_variants.sh \
  push_vggsound_full_audio_af031_improve_results.sh \
  README_vggsound_full_audio_af031_improve.md \
  vggsound_full_experiment_status.md || true

git add \
  data_vggsound_full/features/vggsound_full_audio_paperresnet50_seqconcat8192_chunks4_w500_seed123_summary.json \
  data_vggsound_full/features/vggsound_full_audio_paperresnet50_global2048_lstm4096_concat6144_seed123_summary.json || true

for d in \
  runs_vggsound_full_AF034_standard_audio_paperresnet50_lstm4096_h6_lc5_e650_resume_af031 \
  runs_vggsound_full_AF035_standard_audio_paperresnet50_lstm4096_h6_lc5_e850_resume_af034 \
  runs_vggsound_full_AF036_standard_audio_paperresnet50_lstm4096_h8_lc5_e500 \
  runs_vggsound_full_AF037_standard_audio_paperresnet50_lstm4096_h10_lc5_e500 \
  runs_vggsound_full_AF038_standard_audio_paperresnet50_seqconcat8192_h4_lc5_e500 \
  runs_vggsound_full_AF039_standard_audio_paperresnet50_seqconcat8192_h6_lc5_e500 \
  runs_vggsound_full_AF040_standard_audio_paperresnet50_global2048_lstm4096_concat6144_h6_lc5_e500
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF034_*_stdout.log runs_vggsound_full_AF034_*_stderr.log \
  runs_vggsound_full_AF035_*_stdout.log runs_vggsound_full_AF035_*_stderr.log \
  runs_vggsound_full_AF036_*_stdout.log runs_vggsound_full_AF036_*_stderr.log \
  runs_vggsound_full_AF037_*_stdout.log runs_vggsound_full_AF037_*_stderr.log \
  runs_vggsound_full_AF038_*_stdout.log runs_vggsound_full_AF038_*_stderr.log \
  runs_vggsound_full_AF039_*_stdout.log runs_vggsound_full_AF039_*_stderr.log \
  runs_vggsound_full_AF040_*_stdout.log runs_vggsound_full_AF040_*_stderr.log || true

git add \
  logs/vggsound_audio_af031_capacity_*.out \
  logs/vggsound_audio_af031_capacity_*.err \
  logs/vggsound_audio_af031_encoding_*.out \
  logs/vggsound_audio_af031_encoding_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AF031 improvement files to commit."
else
  git commit -m "Add AF031 audio BM improvement results"
fi

git push -u origin main
