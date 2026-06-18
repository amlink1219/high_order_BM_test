# VGGSound Full Visual/Motion Standard BM

This batch uses the available full VGGSound clips under:

```text
/home/Hongjie_Zeng/datasets/VGGSound_full
```

It runs pure visual standard BM experiments only. No audio and no two-port fusion are used in this batch.

## Feature Definitions

Video appearance:

```text
mp4 -> sampled RGB frames -> ImageNet pretrained ResNet50
-> per-frame 2048-d feature -> mean+std pooling -> 4096-d feature
-> per-dim minmax normalization
```

Motion:

```text
mp4 -> sampled RGB frames
-> adjacent absolute frame differences
-> ImageNet pretrained ResNet50
-> per-difference 2048-d feature -> mean+std pooling -> 4096-d feature
-> per-dim minmax normalization
```

This motion feature is not optical flow. It is a stronger frame-difference encoder baseline than the old raw low-resolution motion input.

## Experiments

All experiments are standard single-channel BM classifiers with the same input dimension and hidden factor:

```text
input_dim = 4096
hidden_dim = 4 * input_dim = 16384
label_copies = 5
num_classes = all eligible VGGSound classes by default
```

| ID | Input | Frames | Config |
|---|---|---:|---|
| VF001 | video appearance | 8 | ResNet50 mean+std, hidden 4x |
| VF002 | motion difference | 8 | ResNet50 mean+std, hidden 4x |
| VF003 | video appearance | 16 | ResNet50 mean+std, hidden 4x |
| VF004 | motion difference | 16 | ResNet50 mean+std, hidden 4x |

## Why 8 vs 16 Frames

Eight frames was enough to make the mini20 video appearance BM work, because static object/scene cues are strong in VGGSound.

For motion, eight frames is probably a weak lower bound: many actions and sound events are short or periodic. Therefore this batch includes 16-frame features as the main check for whether motion benefits from denser temporal sampling.

## Run On Server

The default sbatch uses 4 GPUs for feature extraction:

```text
4 parallel shards, 1 GPU per shard
BM training still uses a single GPU
```

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_visual_motion_bm_code_20260618.zip
chmod +x sbatch_vggsound_full_visual_motion_bm.sh
sbatch sbatch_vggsound_full_visual_motion_bm.sh
```

To run a faster 100-class smoke/first-pass experiment, edit the sbatch command:

```bash
--max_classes 100
```

To run only the 8-frame pair first:

```bash
--only_f8
```

## Monitor

```bash
squeue
tail -f logs/vggsound_full_visual_motion_bm_JOBID.out
tail -f logs/vggsound_full_visual_motion_bm_JOBID.err
```

Replace `JOBID` with the number printed by `sbatch`.

## Result Summary

```bash
cat vggsound_full_visual_motion_bm_log.md
```

## GitHub Upload

```bash
chmod +x push_vggsound_full_visual_motion_bm_results.sh
./push_vggsound_full_visual_motion_bm_results.sh
```

The push helper avoids large `.npz`, `.pt`, `best.pt`, and `last.pt` files.
