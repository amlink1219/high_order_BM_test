#!/bin/bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

echo "== Git status before add =="
git status --short

echo "== Add VGGSound video encoder BM scripts and compact result files =="
git add \
  make_vggsound_video_encoder_features.py \
  run_vggsound_video_encoder_bm_sweep.py \
  sbatch_vggsound_video_encoder_bm_sweep.sh \
  README_vggsound_video_encoder_bm_sweep.md \
  push_vggsound_video_encoder_bm_results.sh \
  vggsound_video_encoder_bm_sweep_log.md \
  logs/vggsound_video_encoder_bm_*.out \
  logs/vggsound_video_encoder_bm_*.err \
  runs_vggsound_mini20_V017_standard_video_resnet18_mean_h2/config.json \
  runs_vggsound_mini20_V017_standard_video_resnet18_mean_h2/history.json \
  runs_vggsound_mini20_V017_standard_video_resnet18_mean_h2/summary.json \
  runs_vggsound_mini20_V017_standard_video_resnet18_mean_h2/full_eval_best_3000.json \
  runs_vggsound_mini20_V018_standard_video_resnet18_meanstd_h2/config.json \
  runs_vggsound_mini20_V018_standard_video_resnet18_meanstd_h2/history.json \
  runs_vggsound_mini20_V018_standard_video_resnet18_meanstd_h2/summary.json \
  runs_vggsound_mini20_V018_standard_video_resnet18_meanstd_h2/full_eval_best_3000.json \
  runs_vggsound_mini20_V019_standard_video_resnet50_mean_h2/config.json \
  runs_vggsound_mini20_V019_standard_video_resnet50_mean_h2/history.json \
  runs_vggsound_mini20_V019_standard_video_resnet50_mean_h2/summary.json \
  runs_vggsound_mini20_V019_standard_video_resnet50_mean_h2/full_eval_best_3000.json \
  runs_vggsound_mini20_V020_standard_video_resnet50_meanstd_h2/config.json \
  runs_vggsound_mini20_V020_standard_video_resnet50_meanstd_h2/history.json \
  runs_vggsound_mini20_V020_standard_video_resnet50_meanstd_h2/summary.json \
  runs_vggsound_mini20_V020_standard_video_resnet50_meanstd_h2/full_eval_best_3000.json \
  runs_vggsound_mini20_V021_standard_video_resnet18_meanstd_threshold_h2/config.json \
  runs_vggsound_mini20_V021_standard_video_resnet18_meanstd_threshold_h2/history.json \
  runs_vggsound_mini20_V021_standard_video_resnet18_meanstd_threshold_h2/summary.json \
  runs_vggsound_mini20_V021_standard_video_resnet18_meanstd_threshold_h2/full_eval_best_3000.json \
  runs_vggsound_mini20_V022_standard_video_resnet50_mean_threshold_h2/config.json \
  runs_vggsound_mini20_V022_standard_video_resnet50_mean_threshold_h2/history.json \
  runs_vggsound_mini20_V022_standard_video_resnet50_mean_threshold_h2/summary.json \
  runs_vggsound_mini20_V022_standard_video_resnet50_mean_threshold_h2/full_eval_best_3000.json

echo "== Git status after add =="
git status --short

if git diff --cached --quiet; then
  echo "No staged changes. Nothing to commit."
else
  git commit -m "Add VGGSound video encoder BM sweep results"
fi

git push -u origin main
echo "== Done =="
