#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  make_vggsound_full_rgb_motion_sequence_features.py \
  merge_vggsound_full_rgb_motion_sequence_shards.py \
  make_vggsound_full_rgb_motion_fused_encoder_features.py \
  run_vggsound_rgb_motion_fused_video4096.py \
  sbatch_vggsound_rgb_motion_fused_video4096.sh \
  push_vggsound_rgb_motion_fused_video4096_results.sh \
  README_vggsound_rgb_motion_fused_video4096.md \
  vggsound_full_experiment_status.md \
  logs/vggsound_rgb_motion_fused_video4096_*.out \
  logs/vggsound_rgb_motion_fused_video4096_*.err \
  logs/P3V001_*.out \
  logs/P3V001_*.err \
  data_vggsound_full/rgb_motion_fused/*summary.json \
  data_vggsound_full/rgb_motion_fused/*history.json \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360/config.json \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360/history.json \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360/summary.json \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360/full_eval_best_3000.json \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360_stdout.log \
  runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360_stderr.log \
  2>/dev/null || true

git status --short
git commit -m "Add RGB motion fused video4096 results" || true
git push -u origin main
