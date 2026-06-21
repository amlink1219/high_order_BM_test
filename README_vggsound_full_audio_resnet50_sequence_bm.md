# VGGSound Full Audio ResNet50 Sequence BM

This package tests a stronger audio feature pipeline before standard BM classification.

## Motivation

The current best audio-only BM is:

```text
AF023 = STFT -> small CNN-LSTM4096 embedding -> BM h6
full eval = 31.21%
```

This branch tests whether a ResNet50 spectrogram teacher plus temporal aggregation can provide a stronger audio input.

## Pipeline

### 1. Base ResNet50 Spectrogram Teacher

Input:

```text
STFT feature: [128 frequency bins x 96 time bins]
```

Model:

```text
1-channel ResNet50 spectrogram classifier
ImageNet pretrained conv1 adapted from RGB to 1 channel
```

The runner reuses the existing teacher checkpoint if present:

```text
data_vggsound_full/features/vggsound_full_audio_resnet50_stft128x96_per_dim_zscore_sigmoid_seed123_teacher_best.pt
```

If it is missing, it trains the base ResNet50 teacher first.

### 2. Temporal Chunk Sequence

The 128x96 spectrogram is split into 8 overlapping temporal chunks:

```text
num_chunks = 8
chunk_frames = 32
each chunk = [128 x 32]
ResNet50 backbone -> 2048-d feature per chunk
sequence shape = [8, 2048]
```

### 3. Two Audio Feature Variants

Mean/std variant:

```text
[8,2048] -> mean[2048] + std[2048] = 4096 visible inputs
```

LSTM variant:

```text
[8,2048] -> BiLSTM teacher -> 4096 visible inputs
```

### 4. BM Experiments

| ID | feature | visible dim | hidden | total p-bit | epochs |
|---|---|---:|---:|---:|---:|
| AF024 | ResNet50 chunk mean/std | 4096 | 24576 | 30217 | 320 |
| AF025 | ResNet50 chunk mean/std | 4096 | 32768 | 38409 | 320 |
| AF026 | ResNet50 chunk LSTM | 4096 | 24576 | 30217 | 320 |
| AF027 | ResNet50 chunk LSTM | 4096 | 32768 | 38409 | 320 |

These should be compared to:

```text
AF023 audio CNN-LSTM h6 e1000 full = 31.21%
VF022 video LSTM h8 e220 full = 40.68%
```

## Server Commands

Unpack and submit:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_resnet50_sequence_bm_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_resnet50_sequence_bm.sh
chmod +x push_vggsound_full_audio_resnet50_sequence_bm_results.sh
sbatch sbatch_vggsound_full_audio_resnet50_sequence_bm.sh
```

Upload after completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_audio_resnet50_sequence_bm_results.sh
```

## Resource Request

The sbatch file requests:

```text
4 GPUs
32 CPU cores
240G memory
48 hours
```

The ResNet50 teacher, sequence extraction, and LSTM teacher use DataParallel over the allocated GPUs. The BM stage currently uses one GPU per training process.

## Useful Variants

Run only mean/std branch:

```bash
/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python run_vggsound_full_audio_resnet50_sequence_bm.py \
  --root /home/Hongjie_Zeng/high_order_BM \
  --python_bin /home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python \
  --only_meanstd
```

Run only LSTM branch:

```bash
/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python run_vggsound_full_audio_resnet50_sequence_bm.py \
  --root /home/Hongjie_Zeng/high_order_BM \
  --python_bin /home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python \
  --only_lstm
```

Large files intentionally stay on the server and are not pushed:

```text
*.npz feature arrays
*.pt checkpoints
```
