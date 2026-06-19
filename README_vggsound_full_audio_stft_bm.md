# VGGSound Full Audio STFT Standard BM

This batch tests pure-audio standard BM baselines on the available full VGGSound clips under:

```text
/home/Hongjie_Zeng/datasets/VGGSound_full
```

No visual input and no two-port fusion are used in this batch.

## Why These Input Sizes

The VGGSound paper uses an STFT spectrogram audio input. For a 5 second crop with the paper/official-style settings, the raw spectrogram is approximately:

```text
257 frequency bins x 500 time bins = 128500 samples
```

The original audio ResNet then uses convolution and global pooling, typically compressing the spectrogram to a 512-d or 2048-d representation before classification.

For a BM, feeding all 128k spectrogram pixels directly would make the dense visible-hidden matrix too large for the first full experiment. These experiments therefore use two physically interpretable time-frequency grids:

```text
64 x 64   = 4096 visible p-bits   roughly 31x smaller than 257 x 500
128 x 96  = 12288 visible p-bits  roughly 10.5x smaller than 257 x 500
```

The 4096-visible setting is the compact baseline. The 12288-visible setting keeps finer frequency and time structure while remaining much smaller than the raw STFT image.

## Audio Preprocessing

```text
mp4
-> ffmpeg audio decode
-> 16 kHz mono float waveform
-> 10 s decode, 5 s center crop
-> STFT spectrogram, nperseg=512, noverlap=353
-> log(spec + 1e-7)
-> per-clip zscore
-> resize to 64x64 or 128x96
-> sigmoid normalization to [0, 1]
-> standard BM visible layer
```

This follows the VGGSound paper/official code style more closely than the earlier mini20 raw-audio feature sweep.

## Experiments

The default run uses all eligible VGGSound classes available in the local clips directory.

| ID | Input | Hidden | Purpose |
|---|---:|---:|---|
| AF001 | 64x64 = 4096 | 4x = 16384 | Compact baseline |
| AF002 | 64x64 = 4096 | 6x = 24576 | Same input, larger BM capacity |
| AF003 | 128x96 = 12288 | 3x = 36864 | Finer spectrogram, controlled size |
| AF004 | 128x96 = 12288 | 4x = 49152 | Finer spectrogram, larger BM capacity |

For 309 classes with `label_copies=5`, the total p-bit counts are approximately:

```text
AF001: 22025
AF002: 30217
AF003: 50697
AF004: 62985
```

## Run On Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_stft_bm_code_20260618.zip
chmod +x sbatch_vggsound_full_audio_stft_bm.sh
sbatch sbatch_vggsound_full_audio_stft_bm.sh
```

The default sbatch requests:

```text
2 GPUs
24 CPU cores
96 GB RAM
48 hours
```

Audio feature extraction is mostly CPU/ffmpeg work. BM training uses CUDA if available.

## Faster First Pass

To run only the two 4096-visible experiments first, add this argument to the runner command inside the sbatch:

```bash
--only_4096
```

To run a smaller class subset, change:

```bash
--max_classes 100
```

## Monitor

```bash
squeue
tail -f logs/vggsound_full_audio_stft_bm_JOBID.out
tail -f logs/vggsound_full_audio_stft_bm_JOBID.err
```

Feature shard logs:

```bash
for s in 0 1; do
  tail -n 5 runs_vggsound_full_audio_stft64x64_feature_shard${s}_stdout.log
done
```

## Result Summary

```bash
cat vggsound_full_audio_stft_bm_log.md
```

## GitHub Upload

```bash
chmod +x push_vggsound_full_audio_stft_bm_results.sh
./push_vggsound_full_audio_stft_bm_results.sh
```

The push helper avoids large `.npz`, `.pt`, `best.pt`, and `last.pt` files.
