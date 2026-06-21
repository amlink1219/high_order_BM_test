# VGGSound Full Audio ResNet50 Sequence BM

Updated: 2026-06-21 17:14:27

Purpose: use a ResNet50 spectrogram teacher, then compare temporal mean/std pooling and LSTM sequence embeddings before BM.

## Features

| feature | kind | dim | best epoch | teacher top1 |
|---|---|---:|---:|---:|
| ARF001 | Embeddings are supervised audio ResNet50 features from STFT inputs | 2048 | 25 | 32.84% |
| ARF002_meanstd | Mean/std pooled audio ResNet50 chunk-sequence feature for BM input_mode=audio | 4096 |  |  |
| ARF003 | Embeddings are supervised audio BiLSTM features from audio ResNet50 temporal chunk sequence features | 4096 | 80 | 32.42% |

## BM Results

| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| AF024_standard_audio_resnet50seq_meanstd4096_h6_lc5_e320 | 4096 | 24576 | 30217 | 320 | 26.65% | 26.58% |
| AF025_standard_audio_resnet50seq_meanstd4096_h8_lc5_e320 | 4096 | 32768 | 38409 | 305 | 27.74% | 26.14% |
| AF026_standard_audio_resnet50seq_lstm4096_h6_lc5_e320 | 4096 | 24576 | 30217 | 320 | 29.06% | 28.97% |
| AF027_standard_audio_resnet50seq_lstm4096_h8_lc5_e320 | 4096 | 32768 | 38409 | 320 | 30.77% | 30.76% |
