# VGGSound RGB + Motion Fused Video4096

Goal: test whether video-side performance improves when RGB appearance and frame-difference motion are fused inside the supervised video encoder, instead of treating motion as a separate BM modality.

Reference video-only controls:

```text
VF026 video-only BM full = 42.84%
P1V002 32-frame/320 input full = 42.29%
P2V001 EfficientNet-B3 video full = 40.45%
```

This branch is only promoted to audio-video two-port if its video-only BM clearly beats VF026.

## Pipeline

```text
mp4
-> sample 16 RGB frames at 224x224
-> RGB stream: frame[t]
-> motion stream: abs(frame[t+1] - frame[t])
-> ImageNet ResNet50 for RGB and motion frames
-> two-branch fused BiLSTM encoder
-> video4096 embedding
-> standard BM video-only screening
```

The output feature file is BM-compatible:

```text
video_train/test: 4096-d fused RGB+motion embedding
audio_train/test: dummy 1-d zero
motion_train/test: dummy 1-d zero
```

## Run On Old Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_rgb_motion_fused_video4096_code_20260627.zip

chmod +x sbatch_vggsound_rgb_motion_fused_video4096.sh
chmod +x push_vggsound_rgb_motion_fused_video4096_results.sh
mkdir -p logs

python -m py_compile \
  make_vggsound_full_rgb_motion_sequence_features.py \
  merge_vggsound_full_rgb_motion_sequence_shards.py \
  make_vggsound_full_rgb_motion_fused_encoder_features.py \
  run_vggsound_rgb_motion_fused_video4096.py \
  train_vggsound_mini20_bm.py

sbatch sbatch_vggsound_rgb_motion_fused_video4096.sh
```

Default resources:

```text
2 GPU, 24 CPU, 120G, 4 days
```

The feature extraction step launches 2 shard processes, each on one GPU. The encoder uses `DataParallel`. The BM stage uses the existing single-GPU BM trainer.

## Check Progress

```bash
cd /home/Hongjie_Zeng/high_order_BM
squeue
tail -n 80 logs/vggsound_rgb_motion_fused_video4096_*.out
tail -n 80 logs/P3V001_fused_encoder.out
tail -n 80 runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360_stdout.log
```

## Upload Results

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_rgb_motion_fused_video4096_results.sh
```

The upload helper intentionally excludes `.npz`, `best.pt`, and `last.pt`.
