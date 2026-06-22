# VGGSound Full AF029 Paper-ResNet Audio Continuation

This package continues the current best uploaded audio-only BM:

```text
AF029 = paper-STFT ResNet50 mean/std4096, h6, epoch450
full eval = 38.78%
```

It tests whether this strong audio feature still benefits from longer BM training, similar to the earlier CNN-LSTM audio branch.

Experiments:

| ID | setup |
|---|---|
| AF032 | continue AF029 from epoch450 to epoch650 |
| AF033 | continue AF032 from epoch650 to epoch850 |

Geometry:

```text
input_dim = 4096
label_dim = 309 * 5 = 1545
hidden_dim = 24576
total_pbits = 30217
```

## Submit

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_paperresnet50_af029_continuation_code_20260622.zip
chmod +x sbatch_vggsound_full_audio_paperresnet50_af029_continuation.sh push_vggsound_full_audio_paperresnet50_af029_continuation_results.sh
sbatch sbatch_vggsound_full_audio_paperresnet50_af029_continuation.sh
```

The sbatch requests 1 GPU, 8 CPU, 90G memory, and 12 hours.

## Upload

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_audio_paperresnet50_af029_continuation_results.sh
```
