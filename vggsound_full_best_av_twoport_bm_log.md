# VGGSound Full Best Audio-Video Two-Port BM

Updated: 2026-06-23 07:55:52

Aligned feature: `/home/Hongjie_Zeng/high_order_BM/data_vggsound_full/features/vggsound_full_aligned_videolstm4096_audiocnnlstm4096_seed123.npz`

Current reference before this branch:

- video-only BM VF022 full = 40.68%
- audio-only BM AF023 full = 31.21%

## Results

| experiment | model | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---|---:|---:|---:|---:|---:|---:|
| AV001 | standard | 4096 | 32768 | 38409 | 320 | 37.26% | 37.29% |
| AV002 | twoport | 4096 | 32768 | 38409 | 270 | 41.06% | 40.63% |
| AV003 | twoport | 4096 | 32768 | 38409 | 320 | 37.04% | 37.13% |
| AV004 | twoport | 4096 | 32768 | 38409 | 320 | 36.84% | 36.93% |
| AV005 | twoport | 4096 | 24576 | 30217 | 285 | 40.48% | 40.32% |
