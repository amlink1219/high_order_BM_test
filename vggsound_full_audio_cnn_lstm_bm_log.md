# VGGSound Full Audio CNN-LSTM BM

Updated: 2026-06-20 00:33:52

Purpose: test a CNN-BiLSTM audio encoder before BM. The CNN extracts local time-frequency patterns, while the BiLSTM models temporal evolution over STFT frames.

Source STFT feature: `vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz`.

Reference direct-STFT BM:

- AF003 STFT128x96 h3 full eval = 4.05%
- AF004 STFT128x96 h4 full eval = 4.47%

Reference CNN-only audio embedding BM:

- AF005-AF008, if available, should be compared directly to AF009-AF012.

Best BM full eval in this batch: 22.78%

## Audio CNN-LSTM Teacher

| feature | embedding dim | lstm hidden | best epoch | teacher top1 |
|---|---:|---:|---:|---:|
| ACL001 | 2048 | 512 | 35 | 35.94% |
| ACL002 | 4096 | 512 | 50 | 35.36% |

## Audio BM On CNN-LSTM Embeddings

| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|
| AF009_standard_audio_cnnlstm2048_h4_lc5_e180 | 2048 | 1545 | 8192 | 11785 | 180 | 19.54% | 19.35% |
| AF010_standard_audio_cnnlstm2048_h6_lc5_e180 | 2048 | 1545 | 12288 | 15881 | 180 | 19.65% | 19.07% |
| AF011_standard_audio_cnnlstm4096_h4_lc5_e180 | 4096 | 1545 | 16384 | 22025 | 180 | 20.86% | 20.75% |
| AF012_standard_audio_cnnlstm4096_h6_lc5_e180 | 4096 | 1545 | 24576 | 30217 | 180 | 22.67% | 22.78% |
