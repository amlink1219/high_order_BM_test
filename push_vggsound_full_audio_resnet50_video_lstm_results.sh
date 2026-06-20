#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  make_vggsound_full_audio_resnet50_encoder_features.py \
  make_vggsound_full_video_resnet_sequence_features.py \
  merge_vggsound_full_video_sequence_shards.py \
  make_vggsound_full_video_lstm_encoder_features.py \
  run_vggsound_full_audio_resnet50_bm.py \
  run_vggsound_full_video_lstm_bm.py \
  sbatch_vggsound_full_audio_resnet50_bm.sh \
  sbatch_vggsound_full_video_lstm_bm.sh \
  push_vggsound_full_audio_resnet50_video_lstm_results.sh \
  README_vggsound_full_audio_resnet50_video_lstm.md \
  vggsound_full_audio_resnet50_bm_log.md \
  vggsound_full_video_lstm_bm_log.md || true

git add \
  data_vggsound_full/features/*audio_resnet50*_summary.json \
  data_vggsound_full/features/*audio_resnet50*_history.json \
  data_vggsound_full/features/*video_resnet50_seq*_summary.json \
  data_vggsound_full/features/*video_lstm*_summary.json \
  data_vggsound_full/features/*video_lstm*_history.json || true

git add \
  runs_vggsound_full_ARF001_*_feature_stdout.log \
  runs_vggsound_full_ARF001_*_feature_stderr.log \
  runs_vggsound_full_VLF001_*_feature_stdout.log \
  runs_vggsound_full_VLF001_*_feature_stderr.log \
  runs_vggsound_full_VLF002_*_feature_stdout.log \
  runs_vggsound_full_VLF002_*_feature_stderr.log \
  runs_vggsound_full_video_seq_*_stdout.log \
  runs_vggsound_full_video_seq_*_stderr.log || true

for d in \
  runs_vggsound_full_AF013_standard_audio_resnet50_2048_h4_lc5_e220 \
  runs_vggsound_full_AF014_standard_audio_resnet50_2048_h6_lc5_e220 \
  runs_vggsound_full_AF015_standard_audio_resnet50_2048_h8_lc5_e220 \
  runs_vggsound_full_VF020_standard_video_lstm2048_h6_lc5_e220 \
  runs_vggsound_full_VF021_standard_video_lstm4096_h6_lc5_e220 \
  runs_vggsound_full_VF022_standard_video_lstm4096_h8_lc5_e220
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF013_*_stdout.log runs_vggsound_full_AF013_*_stderr.log \
  runs_vggsound_full_AF014_*_stdout.log runs_vggsound_full_AF014_*_stderr.log \
  runs_vggsound_full_AF015_*_stdout.log runs_vggsound_full_AF015_*_stderr.log \
  runs_vggsound_full_VF020_*_stdout.log runs_vggsound_full_VF020_*_stderr.log \
  runs_vggsound_full_VF021_*_stdout.log runs_vggsound_full_VF021_*_stderr.log \
  runs_vggsound_full_VF022_*_stdout.log runs_vggsound_full_VF022_*_stderr.log || true

git status --short

if git diff --cached --quiet; then
  echo "No staged result files to commit."
else
  git commit -m "Add VGGSound audio ResNet50 and video LSTM BM experiments"
fi

git push -u origin main
