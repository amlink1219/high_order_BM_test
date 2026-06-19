# VGGSound Full Audio CNN BM

Updated: 2026-06-19 19:39:39

Purpose: replace direct STFT-to-BM audio input with supervised CNN embeddings learned from STFT spectrograms.

Source STFT feature: `vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz`.

Reference direct-STFT BM:

- AF003 STFT128x96 h3 full eval = 4.05%
- AF004 STFT128x96 h4 full eval = 4.47%

Best BM full eval in this batch: 20.75%

## Audio CNN Teacher

| feature | embedding dim | best epoch | teacher top1 |
|---|---:|---:|---:|
| ACF001 | 2048 | 50 | 30.92% |
| ACF002 | 4096 | 5 | 30.58% |

## Audio BM On CNN Embeddings

| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|
| AF005_standard_audio_cnn2048_h4_lc5_e180 | 2048 | 1545 | 8192 | 11785 | 170 | 12.36% | 12.03% |
| AF006_standard_audio_cnn2048_h6_lc5_e180 | 2048 | 1545 | 12288 | 15881 | 170 | 12.98% | 12.45% |
| AF007_standard_audio_cnn4096_h4_lc5_e180 | 4096 | 1545 | 16384 | 22025 | 175 | 19.41% | 19.37% |
| AF008_standard_audio_cnn4096_h6_lc5_e180 | 4096 | 1545 | 24576 | 30217 | 180 | 20.73% | 20.75% |
