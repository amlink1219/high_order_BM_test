# VGGSound-mini20 Video Encoder Feature BM Sweep

This bundle runs video-only standard BM experiments with pretrained visual encoder features.

It does not run raw video pixels, audio, or video+audio two-port BM.

## Purpose

Previous raw-video BM tests used flattened resized frames and only reached about 12.62% full eval. This sweep asks whether semantic video encoder features make the video-only BM baseline meaningful before attempting multimodal two-port fusion.

Random chance for this 20-class task is 5%.

## Experiments

| ID | Feature | Pooling | BM input | Hidden | Notes |
|---|---|---|---:|---:|---|
| V017 | ImageNet ResNet18 | frame mean | 512 | 1024 | compact semantic visual feature |
| V018 | ImageNet ResNet18 | frame mean + std | 1024 | 2048 | adds temporal variation |
| V019 | ImageNet ResNet50 | frame mean | 2048 | 4096 | stronger visual encoder |
| V020 | ImageNet ResNet50 | frame mean + std | 4096 | 8192 | stronger feature + temporal variation |
| V021 | ImageNet ResNet18 | mean + std | 1024 | 2048 | threshold-binarized BM input |
| V022 | ImageNet ResNet50 | mean | 2048 | 4096 | threshold-binarized BM input |

All experiments use:

```text
video -> pretrained CNN frame features -> temporal pooling -> two-layer standard BM
```

The pretrained CNN is not used as a classifier. It only provides deployable/processed visible features for the BM.

## Expected Server Layout

Preferred:

```text
/home/Hongjie_Zeng/high_order_BM/
  data_vggsound_mini/
    clips/
    meta/
```

Also supported:

```text
/home/Hongjie_Zeng/high_order_BM/
  clips/
  meta/
```

## Dependencies

The server environment needs:

```text
torch
torchvision
imageio-ffmpeg
```

The first run downloads ImageNet weights through `torchvision` unless they are already cached under:

```text
/home/Hongjie_Zeng/.cache/torch
```

Do not use `--no_pretrained` for real conclusions. Random CNN weights would only test the plumbing, not useful visual features.

## Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_video_encoder_bm_sweep_code_20260616.zip
chmod +x sbatch_vggsound_video_encoder_bm_sweep.sh
sbatch sbatch_vggsound_video_encoder_bm_sweep.sh
```

## Monitor

```bash
squeue
tail -f logs/vggsound_video_encoder_bm_JOBID.out
tail -f logs/vggsound_video_encoder_bm_JOBID.err
```

Replace `JOBID` with the number printed by `sbatch`.

## Results

Main summary:

```bash
cat vggsound_video_encoder_bm_sweep_log.md
```

Each experiment saves:

```text
runs_vggsound_mini20_V0XX_*/config.json
runs_vggsound_mini20_V0XX_*/history.json
runs_vggsound_mini20_V0XX_*/summary.json
runs_vggsound_mini20_V0XX_*/full_eval_best_3000.json
runs_vggsound_mini20_V0XX_*/best.pt
runs_vggsound_mini20_V0XX_*/last.pt
```

For GitHub upload after all experiments complete:

```bash
chmod +x push_vggsound_video_encoder_bm_results.sh
./push_vggsound_video_encoder_bm_results.sh
```

The push helper avoids large checkpoints and feature `.npz` files.
