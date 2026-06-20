# VGGSound AF019 Longer Audio CNN-LSTM BM

This package continues the current best audio-only BM:

```text
AF019 = CNN-LSTM4096 embedding, hidden 6x, epoch 700
full eval = 29.78%
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
| AF022 | AF019 `last.pt` | 900 |
| AF023 | AF022 `last.pt` | 1000 |

## Server Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_lstm_af019_longer_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_cnn_lstm_af019_longer.sh
chmod +x push_vggsound_full_audio_cnn_lstm_af019_longer_results.sh
sbatch sbatch_vggsound_full_audio_cnn_lstm_af019_longer.sh
```

Upload after completion:

```bash
./push_vggsound_full_audio_cnn_lstm_af019_longer_results.sh
```

Required existing files:

```text
runs_vggsound_full_AF019_standard_audio_cnnlstm4096_h6_lc5_e700_resume_af018/last.pt
runs_vggsound_full_AF019_standard_audio_cnnlstm4096_h6_lc5_e700_resume_af018/best.pt
runs_vggsound_full_AF019_standard_audio_cnnlstm4096_h6_lc5_e700_resume_af018/history.json
data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz
```
