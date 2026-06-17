# VGGSound-mini20 Audio 4x, Aligned Audio, And Two-Port BM

Updated: 2026-06-16 23:10:38

Purpose: run audio hidden 4x, align audio/video features to 4096 dims, then compare same-pbit standard BM and two-port BM.

Same-pbit reference: V038 video-only standard BM used total_pbits=20680 and full best=57.54%.

Best full eval in this batch: 56.31%

## Audio CNN Features

| feature | embedding dim | teacher best epoch | teacher test acc |
|---|---:|---:|---:|
| A002 | 1024 | 75 | 50.00% |
| A_e4096 | 4096 | 115 | 48.26% |

## BM Results

| experiment | model | input/image dim | hidden dim | label dim | total pbits | best epoch | quick best | full best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V044_audio_cnn1024_hidden4 | standard | 1024 | 4096 | 100 | 5220 | 210 | 43.90% | 43.90% |
| V045_audio_cnn4096_aligned_hidden4_lc10 | standard | 4096 | 16384 | 200 | 20680 | 130 | 53.23% | 53.23% |
| V046_twoport_aligned_gamma115_lc10 | twoport | 4096 | 16384 | 200 | 20680 | 205 | 57.23% | 56.31% |
| V047_twoport_aligned_gamma0_lc10 | twoport | 4096 | 16384 | 200 | 20680 | 205 | 55.69% | 54.77% |
| V048_twoport_aligned_gamma05_lc10 | twoport | 4096 | 16384 | 200 | 20680 | 220 | 56.00% | 55.69% |
