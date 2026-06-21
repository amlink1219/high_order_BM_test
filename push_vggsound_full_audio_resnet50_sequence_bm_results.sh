#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/high_order_BM

git add \
  make_vggsound_full_audio_resnet50_encoder_features.py \
  make_vggsound_full_audio_resnet50_sequence_features.py \
  make_vggsound_full_audio_resnet50_lstm_encoder_features.py \
  run_vggsound_full_audio_resnet50_sequence_bm.py \
  sbatch_vggsound_full_audio_resnet50_sequence_bm.sh \
  push_vggsound_full_audio_resnet50_sequence_bm_results.sh \
  README_vggsound_full_audio_resnet50_sequence_bm.md \
  vggsound_full_audio_resnet50_sequence_bm_log.md \
  vggsound_full_experiment_status.md || true

git add \
  data_vggsound_full/features/*audio_resnet50_stft128x96*_summary.json \
  data_vggsound_full/features/*audio_resnet50_stft128x96*_history.json \
  data_vggsound_full/features/*audio_resnet50_seq_chunks*_summary.json \
  data_vggsound_full/features/*audio_resnet50_seqmeanstd*_summary.json \
  data_vggsound_full/features/*audio_resnet50_lstm4096*_summary.json \
  data_vggsound_full/features/*audio_resnet50_lstm4096*_history.json || true

git add \
  runs_vggsound_full_ARF001_resnet50_base_stdout.log \
  runs_vggsound_full_ARF001_resnet50_base_stderr.log \
  runs_vggsound_full_ARF002_audio_resnet50_sequence_stdout.log \
  runs_vggsound_full_ARF002_audio_resnet50_sequence_stderr.log \
  runs_vggsound_full_ARF003_audio_resnet50_lstm_feature_stdout.log \
  runs_vggsound_full_ARF003_audio_resnet50_lstm_feature_stderr.log || true

for d in \
  runs_vggsound_full_AF024_standard_audio_resnet50seq_meanstd4096_h6_lc5_e320 \
  runs_vggsound_full_AF025_standard_audio_resnet50seq_meanstd4096_h8_lc5_e320 \
  runs_vggsound_full_AF026_standard_audio_resnet50seq_lstm4096_h6_lc5_e320 \
  runs_vggsound_full_AF027_standard_audio_resnet50seq_lstm4096_h8_lc5_e320
do
  if [ -d "$d" ]; then
    git add "$d/config.json" "$d/history.json" "$d/summary.json" || true
    git add "$d"/full_eval*.json || true
  fi
done

git add \
  runs_vggsound_full_AF024_*_stdout.log runs_vggsound_full_AF024_*_stderr.log \
  runs_vggsound_full_AF025_*_stdout.log runs_vggsound_full_AF025_*_stderr.log \
  runs_vggsound_full_AF026_*_stdout.log runs_vggsound_full_AF026_*_stderr.log \
  runs_vggsound_full_AF027_*_stdout.log runs_vggsound_full_AF027_*_stderr.log || true

git add \
  logs/vggsound_audio_resnet50_sequence_*.out \
  logs/vggsound_audio_resnet50_sequence_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged audio ResNet50 sequence files to commit."
else
  git commit -m "Add VGGSound audio ResNet50 sequence BM results"
fi

git push -u origin main
