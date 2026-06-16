# VGGSound-mini20 Video 4x/5x And Audio CNN BM Sweep

This bundle runs the next batch after V029.

It has two goals:

1. Test whether the best video-only BM improves further by increasing hidden size from 3x to 4x/5x.
2. Improve pure audio BM by replacing raw log-mel visible input with supervised audio CNN embeddings.

No video+audio two-port fusion is used in this batch.

## Video Experiments

Reference:

```text
V029 = ResNet50 mean+std video feature, input_dim=4096, hidden_dim=12288, full best=53.85%
```

New video-only standard BM experiments:

| ID | Config |
|---|---|
| V036 | ResNet50 mean+std, hidden 4x |
| V037 | ResNet50 mean+std, hidden 5x |
| V038 | ResNet50 mean+std, hidden 4x, label copies 10 |

Expected dimensions:

```text
V036: input=4096, label=100, hidden=16384, total=20580
V037: input=4096, label=100, hidden=20480, total=24676
V038: input=4096, label=200, hidden=16384, total=20680
```

## Audio Experiments

The previous best pure audio standard BM was:

```text
V016 = 96x64 log-mel threshold input, full best=25.00%
```

This batch trains supervised audio CNN encoders on the training split only, exports the penultimate-layer embeddings, then trains standard BM on those embeddings.

Audio CNN feature generation:

| Feature | Embedding | Input |
|---|---:|---|
| A001 | 512 | 96x64 per-mel log-mel |
| A002 | 1024 | 96x64 per-mel log-mel |

Audio-only BM experiments:

| ID | Config |
|---|---|
| V039 | audio CNN 512 embedding, hidden 2x |
| V040 | audio CNN 512 embedding, hidden 3x |
| V041 | audio CNN 1024 embedding, hidden 2x |
| V042 | audio CNN 1024 embedding, hidden 3x |
| V043 | audio CNN 1024 embedding, threshold input, hidden 2x |

Important interpretation note:

```text
The audio CNN is a supervised processed-input encoder.
The final reported BM accuracy is still measured by BM Gibbs label sampling.
Do not compare these audio CNN embedding results as raw-audio baselines.
```

## Run On Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_video4x5x_audio_cnn_bm_code_20260616.zip
chmod +x sbatch_vggsound_video4x5x_audio_cnn_bm.sh
sbatch sbatch_vggsound_video4x5x_audio_cnn_bm.sh
```

## Monitor

```bash
squeue
tail -f logs/vggsound_video4x5x_audio_cnn_bm_JOBID.out
tail -f logs/vggsound_video4x5x_audio_cnn_bm_JOBID.err
```

Replace `JOBID` with the number printed by `sbatch`.

## Result Summary

```bash
cat vggsound_video4x5x_audio_cnn_bm_log.md
```

The log includes:

```text
Video BM results
Audio CNN teacher test accuracy
Audio BM results
```

## Resume Behavior

If the job stops midway, submit the same sbatch file again. Existing BM runs with `summary.json` and existing audio CNN feature summaries are skipped automatically.

## GitHub Upload

```bash
chmod +x push_vggsound_video4x5x_audio_cnn_bm_results.sh
./push_vggsound_video4x5x_audio_cnn_bm_results.sh
```

The push helper avoids large `.npz`, `.pt`, `best.pt`, and `last.pt` files.
