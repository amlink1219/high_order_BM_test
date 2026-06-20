# VGGSound Full Video LSTM BM

Updated: 2026-06-20 15:24:01

Purpose: preserve frame order with per-frame ResNet50 sequence features, then train a BiLSTM temporal encoder before BM.

References:

- VF010 video ResNet50 mean/std h8 e240 full eval = 37.66%

Best BM full eval in this batch: 40.68%

## Video LSTM Teacher

| feature | embedding dim | best epoch | teacher top1 |
|---|---:|---:|---:|
| VLF001 | 2048 | 5 | 42.17% |
| VLF002 | 4096 | 5 | 42.34% |

## Video BM On LSTM Embeddings

| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|
| VF020_standard_video_lstm2048_h6_lc5_e220 | 2048 | 1545 | 12288 | 15881 | 220 | 33.79% | 33.79% |
| VF021_standard_video_lstm4096_h6_lc5_e220 | 4096 | 1545 | 24576 | 30217 | 220 | 38.92% | 38.99% |
| VF022_standard_video_lstm4096_h8_lc5_e220 | 4096 | 1545 | 32768 | 38409 | 220 | 40.67% | 40.68% |
