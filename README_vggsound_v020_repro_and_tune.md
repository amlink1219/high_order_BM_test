# VGGSound-mini20 V020 Reproduction And Tuning

This bundle continues from V020:

```text
V020 = ResNet50 frame features, mean+std temporal pooling, per-dim minmax, standard BM
full eval best = 47.38%
```

The goal is to check stability and see whether the video-only standard BM can be pushed beyond V020 before trying audio-video two-port fusion.

## Experiments

### Reproduction

| ID | Change from V020 |
|---|---|
| V023 | seed 124 |
| V024 | seed 125 |
| V025 | seed 126 |
| V026 | seed 127 |
| V027 | seed 128 |

These use the same feature and BM configuration as V020:

```text
encoder = ResNet50 pretrained on ImageNet
frames = 8
pool = mean + std
feature dim = 4096
hidden = 8192
label copies = 5
input = continuous minmax feature
```

### Focused Tuning

| ID | Change |
|---|---|
| V028 | hidden 1x |
| V029 | hidden 3x |
| V030 | label copies 10 |
| V031 | threshold input |
| V032 | Bernoulli sampled input |
| V033 | per-dim zscore sigmoid normalization |
| V034 | mean + max temporal pooling |
| V035 | 16 sampled frames, mean + std |

All experiments remain:

```text
video encoder features -> two-layer standard BM
```

No audio and no two-port fusion are used in this batch.

## Run On Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_v020_repro_and_tune_code_20260616.zip
chmod +x sbatch_vggsound_v020_repro_and_tune.sh
sbatch sbatch_vggsound_v020_repro_and_tune.sh
```

## Monitor

```bash
squeue
tail -f logs/vggsound_v020_repro_tune_JOBID.out
tail -f logs/vggsound_v020_repro_tune_JOBID.err
```

Replace `JOBID` with the number printed by `sbatch`.

## Result Summary

```bash
cat vggsound_v020_repro_and_tune_log.md
```

The log automatically reports:

```text
best full eval so far
V020 reproduction mean/std across V023-V027
```

## Resume Behavior

If the job stops midway, submit the same sbatch file again. Completed experiments with `summary.json` are skipped automatically.

## GitHub Upload

After the job finishes:

```bash
chmod +x push_vggsound_v020_repro_and_tune_results.sh
./push_vggsound_v020_repro_and_tune_results.sh
```

The push helper uploads scripts, compact logs, configs, histories, summaries, and full eval JSON files. It does not upload large `.npz`, `best.pt`, or `last.pt` files.
