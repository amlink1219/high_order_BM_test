# VGGSound Full f16 Video BM Follow-up

Updated: 2026-06-19 06:23:46

Purpose: continue the best f16 video BM and test whether larger hidden layers improve the standard BM baseline.

Feature: existing `vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz`, video appearance branch only.

Baseline reference: VF003 full eval = 20.08% at epoch 60, hidden 4x.

Best full eval in this batch: 32.91%

| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VF005_video_resnet50_meanstd_f16_h4_lc5_resume120 | 309 | 4096 | 1545 | 16384 | 22025 | 120 | 25.49% | 25.47% |
| VF006_video_resnet50_meanstd_f16_h4_lc5_resume180 | 309 | 4096 | 1545 | 16384 | 22025 | 180 | 28.96% | 28.92% |
| VF007_video_resnet50_meanstd_f16_h6_lc5_e120 | 309 | 4096 | 1545 | 24576 | 30217 | 120 | 28.92% | 29.02% |
| VF008_video_resnet50_meanstd_f16_h8_lc5_e120 | 309 | 4096 | 1545 | 32768 | 38409 | 120 | 32.91% | 32.91% |
