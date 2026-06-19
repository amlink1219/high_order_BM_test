# VGGSound AF012 Audio CNN-LSTM BM Continuation

This package continues the best current audio-only BM result:

```text
AF012 = CNN-LSTM4096 embedding, hidden 6x, label copies 5, epoch 180
full eval = 22.78%
```

The continuation keeps the exact AF012 geometry:

```text
audio input = 4096
label bits = 309 classes * 5 copies = 1545
hidden bits = 24576
total p-bits = 30217
```

New runs:

| ID | resume source | final epoch | purpose |
|---|---|---:|---|
| AF016 | AF012 `last.pt` | 260 | check whether the still-rising AF012 curve continues improving |
| AF017 | AF016 `last.pt` | 300 | extend the same trajectory further |

The script also writes/updates:

```text
vggsound_full_experiment_status.md
```

That status file records completed audio results, pending ResNet50/video-LSTM work, and this continuation plan.

## Server Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_lstm_af012_continuation_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_cnn_lstm_af012_continuation.sh
chmod +x push_vggsound_full_audio_cnn_lstm_af012_continuation_results.sh
sbatch sbatch_vggsound_full_audio_cnn_lstm_af012_continuation.sh
```

Monitor:

```bash
squeue
tail -f logs/vggsound_audio_cnn_lstm_af012_cont_*.out
```

Upload after completion:

```bash
./push_vggsound_full_audio_cnn_lstm_af012_continuation_results.sh
```

## Requirements

The following files/directories must already exist on the server:

```text
runs_vggsound_full_AF012_standard_audio_cnnlstm4096_h6_lc5_e180/last.pt
runs_vggsound_full_AF012_standard_audio_cnnlstm4096_h6_lc5_e180/best.pt
runs_vggsound_full_AF012_standard_audio_cnnlstm4096_h6_lc5_e180/history.json
data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz
```

Large `.pt` and `.npz` files should stay on the server and are not uploaded to GitHub.
