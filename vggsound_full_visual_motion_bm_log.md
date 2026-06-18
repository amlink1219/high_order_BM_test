# VGGSound Full Visual/Motion Standard BM

Updated: 2026-06-18 22:03:59

Purpose: pure visual standard BM on the available full VGGSound clips, comparing static video appearance with frame-difference motion features.

Motion definition: adjacent sampled RGB-frame absolute differences are encoded by the same ImageNet-pretrained ResNet50 and pooled with mean+std.

Best full eval in this batch: 20.08%

| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VF001_video_resnet50_meanstd_f8_h4_lc5 | 309 | 4096 | 1545 | 16384 | 22025 | 60 | 19.80% | 19.86% |
| VF002_motion_diffresnet50_meanstd_f8_h4_lc5 | 309 | 4096 | 1545 | 16384 | 22025 | 60 | 14.80% | 14.82% |
| VF003_video_resnet50_meanstd_f16_h4_lc5 | 309 | 4096 | 1545 | 16384 | 22025 | 60 | 20.04% | 20.08% |
| VF004_motion_diffresnet50_meanstd_f16_h4_lc5 | 309 | 4096 | 1545 | 16384 | 22025 | 60 | 14.91% | 15.04% |
