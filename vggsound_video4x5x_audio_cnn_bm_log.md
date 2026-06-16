# VGGSound-mini20 Video 4x/5x And Audio CNN BM Sweep

Updated: 2026-06-16 22:01:47

Purpose: extend the best video encoder BM with larger hidden layers, and improve audio-only BM using supervised audio CNN embeddings.

Best full eval in this batch: 57.54%

## Video BM

| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| V036_video_resnet50_meanstd_hidden4 | 4096 | 16384 | 20580 | 210 | 55.08% | 56.00% |
| V037_video_resnet50_meanstd_hidden5 | 4096 | 20480 | 24676 | 205 | 57.23% | 56.92% |
| V038_video_resnet50_meanstd_hidden4_lc10 | 4096 | 16384 | 20680 | 205 | 57.23% | 57.54% |

## Audio CNN Teacher Features

| feature | embedding dim | best epoch | teacher test acc |
|---|---:|---:|---:|
| A001 | 512 | 95 | 49.13% |
| A002 | 1024 | 75 | 50.00% |

## Audio BM

| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| V039_audio_cnn512_hidden2 | 512 | 1024 | 1636 | 220 | 27.91% | 29.07% |
| V040_audio_cnn512_hidden3 | 512 | 1536 | 2148 | 220 | 30.81% | 31.10% |
| V041_audio_cnn1024_hidden2 | 1024 | 2048 | 3172 | 210 | 40.41% | 40.70% |
| V042_audio_cnn1024_hidden3 | 1024 | 3072 | 4196 | 220 | 43.60% | 43.60% |
| V043_audio_cnn1024_threshold_hidden2 | 1024 | 2048 | 3172 | 215 | 43.02% | 42.73% |
