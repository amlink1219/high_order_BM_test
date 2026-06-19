# VGGSound Full Audio CNN-LSTM BM

This package tests a CNN-BiLSTM audio encoder before the standard BM.

## Motivation

Direct STFT-to-BM was weak:

- AF003: STFT128x96, hidden 3x, full eval = 4.05%.
- AF004: STFT128x96, hidden 4x, full eval = 4.47%.

The CNN-only encoder tests local time-frequency patterns:

```text
STFT -> CNN -> embedding -> BM
```

This package adds sequence modeling:

```text
STFT 128x96
-> CNN keeps the 96-frame time axis
-> BiLSTM models temporal evolution
-> pooled embedding
-> audio-only standard BM
```

## Experiments

Source STFT feature expected on the server:

```text
data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz
```

CNN-LSTM teacher features:

| ID | source | embedding | teacher epochs | model |
|---|---|---:|---:|---|
| ACL001 | STFT 128x96 | 2048 | 50 | CNN + BiLSTM hidden 512 |
| ACL002 | STFT 128x96 | 4096 | 50 | CNN + BiLSTM hidden 512 |

Audio BM experiments:

| ID | input | hidden factor | label copies | epochs |
|---|---:|---:|---:|---:|
| AF009 | CNN-LSTM 2048 | 4x | 5 | 180 |
| AF010 | CNN-LSTM 2048 | 6x | 5 | 180 |
| AF011 | CNN-LSTM 4096 | 4x | 5 | 180 |
| AF012 | CNN-LSTM 4096 | 6x | 5 | 180 |

For 309 classes, label dim is `309 * 5 = 1545`.

## Server Run

Copy these files to `/home/Hongjie_Zeng/high_order_BM`:

- `make_vggsound_full_audio_cnn_lstm_encoder_features.py`
- `run_vggsound_full_audio_cnn_lstm_bm.py`
- `sbatch_vggsound_full_audio_cnn_lstm_bm.sh`
- `push_vggsound_full_audio_cnn_lstm_bm_results.sh`
- `README_vggsound_full_audio_cnn_lstm_bm.md`
- `train_vggsound_mini20_bm.py` if the server copy is not already up to date.

Submit:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_lstm_bm_code_20260619.zip
chmod +x sbatch_vggsound_full_audio_cnn_lstm_bm.sh push_vggsound_full_audio_cnn_lstm_bm_results.sh
sbatch sbatch_vggsound_full_audio_cnn_lstm_bm.sh
```

Default resource request:

```text
GPU: 4
CPU: 16
Memory: 130G
Time: 3 days
```

The CNN-BiLSTM teacher stage uses PyTorch `DataParallel` across the allocated GPUs. The later BM stage still uses one GPU because the current BM training code is not multi-GPU.

Monitor:

```bash
squeue
tail -f logs/vggsound_full_audio_cnn_lstm_bm_*.out
```

After completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
bash push_vggsound_full_audio_cnn_lstm_bm_results.sh
```

## Faster Probe

To run only the 2048-d branch first:

```bash
/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python -u run_vggsound_full_audio_cnn_lstm_bm.py --root . --only_2048
```

To train the CNN-LSTM teacher only, without BM:

```bash
/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python -u run_vggsound_full_audio_cnn_lstm_bm.py --root . --skip_bm
```

## Notes

- The generated embedding `.npz` files and teacher `.pt` checkpoints are large and are intentionally not added by the push helper.
- The first metric to inspect is the teacher top-1/top-5. If teacher top-1 is not better than the CNN-only teacher, the LSTM is probably not worth the extra cost.
- This script requests four GPUs. Only the CNN-BiLSTM teacher is multi-GPU; BM training remains single-GPU.
