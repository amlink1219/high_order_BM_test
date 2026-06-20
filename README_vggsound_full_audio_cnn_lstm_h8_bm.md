# VGGSound Audio CNN-LSTM4096 Hidden 8x BM

This package tests whether increasing the BM hidden layer improves the current audio branch.

Current best audio:

```text
AF017 = CNN-LSTM4096, hidden 6x, epoch 300
full eval = 25.53%
```

New h8 geometry:

```text
audio input = 4096
label bits = 309 * 5 = 1545
hidden bits = 32768
total p-bits = 38409
```

Because hidden size changes from h6 to h8, this cannot warm-start from AF017. It trains h8 from scratch, then continues the same h8 run:

| ID | setup | final epoch |
|---|---|---:|
| AF020 | h8 from scratch | 300 |
| AF021 | continue AF020 | 500 |

The h8 model uses `batch_size=64` to reduce memory risk.

## Server Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_lstm_h8_bm_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_cnn_lstm_h8_bm.sh
chmod +x push_vggsound_full_audio_cnn_lstm_h8_bm_results.sh
sbatch sbatch_vggsound_full_audio_cnn_lstm_h8_bm.sh
```

Upload after completion:

```bash
./push_vggsound_full_audio_cnn_lstm_h8_bm_results.sh
```

Required existing feature file:

```text
data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz
```
