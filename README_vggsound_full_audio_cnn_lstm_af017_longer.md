# VGGSound AF017 Longer Audio CNN-LSTM BM

This package continues the current best audio-only BM:

```text
AF017 = CNN-LSTM4096 embedding, hidden 6x, epoch 300
full eval = 25.53%
```

The geometry is unchanged:

```text
audio input = 4096
label bits = 309 * 5 = 1545
hidden bits = 24576
total p-bits = 30217
```

New runs:

| ID | resume source | final epoch |
|---|---|---:|
| AF018 | AF017 `last.pt` | 500 |
| AF019 | AF018 `last.pt` | 700 |

This should fit roughly within a 10-hour budget on one GPU based on the previous AF016-AF017 runtime, with room for final full eval.

## Server Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_lstm_af017_longer_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_cnn_lstm_af017_longer.sh
chmod +x push_vggsound_full_audio_cnn_lstm_af017_longer_results.sh
sbatch sbatch_vggsound_full_audio_cnn_lstm_af017_longer.sh
```

Upload after completion:

```bash
./push_vggsound_full_audio_cnn_lstm_af017_longer_results.sh
```

Required existing files:

```text
runs_vggsound_full_AF017_standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016/last.pt
runs_vggsound_full_AF017_standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016/best.pt
runs_vggsound_full_AF017_standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016/history.json
data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz
```
