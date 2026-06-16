# VGGSound-mini20 Standard BM Video Encoder Sweep

Updated: 2026-06-16 18:00:52

Purpose: test whether pretrained video encoder features carry useful visual signal for a two-layer standard BM.

Random chance for 20 classes is 5%. This is video-only; no audio or two-port fusion is used.

| experiment | video feature dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| V017_standard_video_resnet18_mean_h2 | 512 | 1024 | 1636 | 135 | 12.00% | 9.54% |
| V018_standard_video_resnet18_meanstd_h2 | 1024 | 2048 | 3172 | 190 | 11.69% | 11.69% |
| V019_standard_video_resnet50_mean_h2 | 2048 | 4096 | 6244 | 200 | 31.38% | 31.08% |
| V020_standard_video_resnet50_meanstd_h2 | 4096 | 8192 | 12388 | 205 | 47.69% | 47.38% |
| V021_standard_video_resnet18_meanstd_threshold_h2 | 1024 | 2048 | 3172 | 215 | 20.92% | 20.62% |
| V022_standard_video_resnet50_mean_threshold_h2 | 2048 | 4096 | 6244 | 100 | 31.08% | 31.08% |
