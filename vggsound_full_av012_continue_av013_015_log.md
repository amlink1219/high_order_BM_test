# VGGSound Full AV012 Continuation And AV013-AV015 Ablations

Updated: 2026-06-24 07:33:45

Aligned feature: `/home/Hongjie_Zeng/high_order_BM/data_vggsound_full/features/vggsound_full_aligned_videolstm4096_audioaf031_lstm4096_seed123.npz`

Reference:

- video-only same dim: VF024 / VLF002 video LSTM4096 h8 full=42.74%
- video-only overall: VF026 video LSTM8192 h6 full=42.84%
- audio-only best: AF036 paper-STFT ResNet50 LSTM4096 h8 full=44.98%
- interrupted two-port baseline: AV012 epoch180 full Gibbs acc=55.09%

AV016 resumes the interrupted AV012 run from `last.pt` but writes into a new directory.
The copied AV012 `best.pt` is kept as the baseline best checkpoint unless later epochs improve the quick Gibbs metric.

## Results

| experiment | best epoch | quick best | full best | hidden dim | total pbits |
|---|---:|---:|---:|---:|---:|
| AV016 | 310 | 57.83% | 57.86% | 32768 | 38409 |
