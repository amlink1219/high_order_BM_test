# VGGSound-mini20 Audio 4x, Aligned Audio, And Two-Port BM

This batch does three things:

1. Run an audio-only hidden 4x BM on the existing 1024-dim audio CNN feature.
2. Train/export a 4096-dim audio CNN feature so audio and video have the same port dimension.
3. Run same-pbit two-port BM using aligned video/audio features.

## Reference Results

Important existing references:

```text
V038 video-only standard BM:
  video = ResNet50 mean+std, 4096 dim
  label copies = 10
  hidden = 16384
  total_pbits = 20680
  full best = 57.54%

V042 audio-only standard BM:
  audio = audio CNN 1024 embedding
  label copies = 5
  hidden = 3072
  total_pbits = 4196
  full best = 43.60%
```

## New Experiments

| ID | Purpose | Config |
|---|---|---|
| V044 | audio hidden 4x | audio CNN1024, hidden=4096, label copies=5 |
| V045 | aligned audio-only same-pbit baseline | audio CNN4096, hidden=16384, label copies=10, total=20680 |
| V046 | same-pbit two-port | video4096 + audio4096, gamma=1.15, label copies=10, total=20680 |
| V047 | same-pbit additive ablation | video4096 + audio4096, gamma=0, label copies=10, total=20680 |
| V048 | same-pbit reduced-gamma two-port | video4096 + audio4096, gamma=0.5, label copies=10, total=20680 |

The `total_pbits=20680` two-port runs use the same physical p-bit count as V038:

```text
visible/video p-bits = 4096
label p-bits = 200
hidden p-bits = 16384
total p-bits = 20680
```

The audio port is the second input channel to the same p-bit array. Its 4096-dim feature vector is not counted as additional p-bit devices.

## Alignment

Video and audio features are aligned by clip path:

```text
video feature npz path_train/path_test
audio feature npz path_train/path_test
intersection by path
```

The aligned dataset is saved as:

```text
data_vggsound_mini/features/vggsound_mini20_aligned_video4096_audio4096.npz
```

The `.npz` itself is large and should not be committed to GitHub. Only its summary JSON is uploaded.

## Run On Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_audio4x_aligned_twoport_code_20260616.zip
chmod +x sbatch_vggsound_audio4x_aligned_twoport.sh
sbatch sbatch_vggsound_audio4x_aligned_twoport.sh
```

## Monitor

```bash
squeue
tail -f logs/vggsound_audio4x_aligned_twoport_JOBID.out
tail -f logs/vggsound_audio4x_aligned_twoport_JOBID.err
```

## Results

```bash
cat vggsound_audio4x_aligned_twoport_log.md
```

## Resume Behavior

If the job stops midway, submit the same sbatch file again. Existing features and BM runs with `summary.json` are skipped automatically.

## GitHub Upload

```bash
chmod +x push_vggsound_audio4x_aligned_twoport_results.sh
./push_vggsound_audio4x_aligned_twoport_results.sh
```

The push helper avoids large `.npz`, teacher `.pt`, `best.pt`, and `last.pt` files.
