# VGGSound-mini20 Audio-Only Standard BM Sweep

This bundle runs standard single-channel BM experiments only. It does not run video+audio two-port BM.

## Purpose

The current video-only and audio-only results are close to weak baselines, so this sweep checks whether audio alone contains learnable signal before trying multimodal two-port BM.

Random chance for this 20-class task is 5%.

## Experiments

| ID | Input | Normalization | Hidden | Notes |
|---|---:|---|---:|---|
| V010 | 64 x 32 = 2048 | per-clip zscore sigmoid | 1x input | confirms old audio feature style |
| V011 | 96 x 64 = 6144 | per-mel train zscore sigmoid | 1x input | higher time-frequency detail |
| V012 | 128 x 64 = 8192 | per-mel train zscore sigmoid | 1x input | more mel detail |
| V013 | 128 x 96 = 12288 | per-mel train zscore sigmoid | 1x input | largest audio map in this sweep |
| V014 | 96 x 64 = 6144 | per-mel train zscore sigmoid | 2x input | capacity check |
| V015 | 96 x 64 = 6144 | global train zscore sigmoid | 1x input | normalization check |
| V016 | 96 x 64 = 6144 | per-mel train zscore sigmoid | 1x input | threshold-binarized input |

All experiments use standard BM with `input_mode=audio`.

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

The feature extractor will search both layouts.

## Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_audio_standard_bm_sweep_code_20260616.zip
chmod +x sbatch_vggsound_audio_standard_bm_sweep.sh
sbatch sbatch_vggsound_audio_standard_bm_sweep.sh
```

## Monitor

```bash
squeue
tail -f logs/vggsound_audio_bm_JOBID.out
tail -f logs/vggsound_audio_bm_JOBID.err
```

Replace `JOBID` with the number printed by `sbatch`.

## Results

Main summary:

```bash
cat vggsound_audio_standard_bm_sweep_log.md
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

For GitHub upload, commit only scripts, logs, summaries, histories, and full eval JSONs. Do not commit `best.pt`, `last.pt`, or large feature npz files unless explicitly needed.
