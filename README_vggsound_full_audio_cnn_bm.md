# VGGSound Full Audio CNN BM

This package runs the next audio-side experiment after AF001-AF004.

## Motivation

AF001-AF004 directly fed resized STFT spectrogram bins into a standard BM:

```text
mp4 audio -> 16 kHz mono -> 5 s crop -> STFT -> log + zscore + sigmoid -> BM visible input
```

Best direct-STFT result:

- AF004, STFT 128x96, input 12288, hidden 49152, full eval = 4.47%.

This package adds the missing CNN/audio-encoder step:

```text
STFT spectrogram
-> supervised audio CNN learns local time-frequency patterns
-> take penultimate embedding
-> normalize embedding to [0, 1]
-> audio-only standard BM
```

This makes audio processing more comparable to the video branch:

```text
video frames -> ResNet50 embedding -> BM
```

## Experiments

Source STFT feature expected on the server:

```text
data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz
```

Audio CNN teacher features:

| ID | source | embedding | CNN epochs | normalization |
|---|---|---:|---:|---|
| ACF001 | STFT 128x96 | 2048 | 60 | per-dim zscore + sigmoid |
| ACF002 | STFT 128x96 | 4096 | 60 | per-dim zscore + sigmoid |

Audio BM experiments:

| ID | input | hidden factor | label copies | epochs |
|---|---:|---:|---:|---:|
| AF005 | CNN 2048 | 4x | 5 | 180 |
| AF006 | CNN 2048 | 6x | 5 | 180 |
| AF007 | CNN 4096 | 4x | 5 | 180 |
| AF008 | CNN 4096 | 6x | 5 | 180 |

For 309 classes, label dim is `309 * 5 = 1545`.

## Server Run

Copy these files to `/home/Hongjie_Zeng/high_order_BM`:

- `make_vggsound_full_audio_cnn_encoder_features.py`
- `run_vggsound_full_audio_cnn_bm.py`
- `sbatch_vggsound_full_audio_cnn_bm.sh`
- `push_vggsound_full_audio_cnn_bm_results.sh`
- `README_vggsound_full_audio_cnn_bm.md`
- `train_vggsound_mini20_bm.py` if the server copy is not already up to date.

Submit:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_cnn_bm_code_20260619.zip
chmod +x sbatch_vggsound_full_audio_cnn_bm.sh push_vggsound_full_audio_cnn_bm_results.sh
sbatch sbatch_vggsound_full_audio_cnn_bm.sh
```

Monitor:

```bash
squeue
tail -f logs/vggsound_full_audio_cnn_bm_*.out
```

After completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
bash push_vggsound_full_audio_cnn_bm_results.sh
```

## Notes

- The generated audio embedding `.npz` files and teacher `.pt` checkpoints are large and are intentionally not added by the push helper.
- The first metric to inspect is the supervised audio CNN teacher top-1/top-5. If the teacher itself is weak, BM accuracy will also be limited.
- If runtime is too long, run only the 2048-d branch first:

```bash
/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python -u run_vggsound_full_audio_cnn_bm.py --root . --only_2048
```
