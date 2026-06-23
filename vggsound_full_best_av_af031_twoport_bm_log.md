# VGGSound Full AF031 Audio-Video Two-Port BM

Updated: 2026-06-23 01:31:59

Aligned feature: `/home/Hongjie_Zeng/high_order_BM/data_vggsound_full/features/vggsound_full_aligned_videolstm4096_audioaf031_lstm4096_seed123.npz`

Reference before this branch:

- audio-only BM: AF031 paper-STFT ResNet50 LSTM4096 h6 full=44.31%
- video-only BM, same 4096-d visible feature: VF024 / VLF002 video LSTM4096 h8 full=42.74%
- video-only BM, overall best: VF026 / VLF003 video LSTM8192 h6 full=42.84% but not same-dim with AF031

This branch intentionally uses the 4096-d VLF002/VF024-compatible video feature, because AF031 is 4096-d and the current two-port trainer requires equal port dimensions.

## Results

| experiment | model | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---|---:|---:|---:|---:|---:|---:|
| AV011 | standard | 4096 | 32768 | 38409 | 320 | 43.60% | 43.69% |
