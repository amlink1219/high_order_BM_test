# VGGSound Full Video BM Scale-up

Updated: 2026-06-19 19:45:15

Purpose: increase the video-side standard BM training horizon and hidden-layer scale after VF008 reached 32.91% at epoch 120.

Feature: existing `vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz`, video appearance branch only.

Reference results:

- VF003 h4 e60 full eval = 20.08%
- VF006 h4 e180 full eval = 28.92%
- VF008 h8 e120 full eval = 32.91%

Best full eval in this batch: 37.66%

| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VF009_video_resnet50_meanstd_f16_h8_lc5_resume180 | 309 | 4096 | 1545 | 32768 | 38409 | 175 | 35.68% | 35.64% |
| VF010_video_resnet50_meanstd_f16_h8_lc5_resume240 | 309 | 4096 | 1545 | 32768 | 38409 | 240 | 37.36% | 37.66% |
| VF011_video_resnet50_meanstd_f16_h10_lc5_e160 | 309 | 4096 | 1545 | 40960 | 46601 | 160 | 37.01% | 37.16% |
