# VGGSound Full Audio Paper-STFT ResNet50 BM

Updated: 2026-06-22 02:25:13

Purpose: remove the earlier 128x96 bottleneck before the audio teacher. ResNet50 sees paper-style STFT, while BM receives only learned embeddings.

## Teacher

- ResNet50 best top1: 51.74%
- ResNet50 best epoch: 78
- Input: 257 x 1004, train random 257x500 crops, eval/export full 10s.
- LSTM feature teacher top1: 49.18%

## BM Results

| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| AF028 | 2048 | 16384 | 19977 | 450 | 27.02% | 20.11% |
| AF029 | 4096 | 24576 | 30217 | 445 | 39.72% | 38.78% |
| AF030 | 4096 | 32768 | 38409 | 420 | 41.01% | 36.65% |
| AF031 | 4096 | 24576 | 30217 | 430 | 44.44% | 44.31% |
