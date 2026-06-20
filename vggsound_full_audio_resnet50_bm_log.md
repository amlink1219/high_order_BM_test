# VGGSound Full Audio ResNet50 BM

Updated: 2026-06-20 01:20:51

Purpose: replace the small audio CNN teacher with a ResNet50 spectrogram encoder before BM.

Source STFT feature: `vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz`.

References:

- AF004 direct STFT BM full eval = 4.47%
- AF008 small-CNN4096 BM full eval = 20.75%

Best BM full eval in this batch: 22.32%

## Audio ResNet50 Teacher

| feature | embedding dim | best epoch | teacher top1 |
|---|---:|---:|---:|
| ARF001 | 2048 | 25 | 32.84% |

## Audio BM On ResNet50 Embeddings

| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|
| AF013_standard_audio_resnet50_2048_h4_lc5_e220 | 2048 | 1545 | 8192 | 11785 | 220 | 22.58% | 22.32% |
| AF014_standard_audio_resnet50_2048_h6_lc5_e220 | 2048 | 1545 | 12288 | 15881 | 210 | 22.21% | 20.98% |
| AF015_standard_audio_resnet50_2048_h8_lc5_e220 | 2048 | 1545 | 16384 | 19977 | 215 | 21.80% | 19.10% |
