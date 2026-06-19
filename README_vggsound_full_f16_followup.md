# VGGSound Full f16 Video BM Follow-up

This batch continues the best full VGGSound visual BM result and tests whether larger BM capacity improves the f16 video appearance baseline.

## Starting Point

Previous best visual-only BM:

```text
VF003 video ResNet50 mean+std, 16 frames
input_dim   = 4096
label_dim   = 309 * 5 = 1545
hidden_dim  = 4 * 4096 = 16384
total pbits = 22025
epoch       = 60
full eval   = 20.08%
```

The VF003 training curve was still improving at epoch 60, so continuing training is likely useful. Increasing hidden size is also plausible because 309 classes and full VGGSound visual structure may need more BM capacity than hidden 4x.

## Experiments

All experiments reuse the existing feature file:

```text
data_vggsound_full/features/vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz
```

No video decoding or ResNet feature extraction is performed in this batch.

| ID | Setup | Epochs | Hidden | Total p-bits |
|---|---|---:|---:|---:|
| VF005 | resume VF003 h4 | 60 -> 120 | 16384 | 22025 |
| VF006 | resume VF005 h4 | 120 -> 180 | 16384 | 22025 |
| VF007 | train from scratch h6 | 120 | 24576 | 30217 |
| VF008 | train from scratch h8 | 120 | 32768 | 38409 |

## What This Tests

1. Continued training:

```text
VF003 epoch 60 -> VF005 epoch 120 -> VF006 epoch 180
```

This checks whether the previous 20.08% was simply undertrained.

2. Larger BM capacity:

```text
hidden 4x vs 6x vs 8x
```

This checks whether the standard BM classifier is capacity-limited.

## Run On Server

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_f16_followup_code_20260619.zip
chmod +x sbatch_vggsound_full_f16_followup.sh
sbatch sbatch_vggsound_full_f16_followup.sh
```

Default sbatch:

```text
1 GPU
12 CPU cores
80 GB RAM
24 hours
```

## Faster Variants

Only run the resume/longer-training experiments:

```bash
python run_vggsound_full_f16_followup.py --only_resume
```

Only run the larger hidden-size experiments:

```bash
python run_vggsound_full_f16_followup.py --only_scale
```

## Monitor

```bash
squeue
tail -f logs/vggsound_full_f16_followup_JOBID.out
tail -f logs/vggsound_full_f16_followup_JOBID.err
```

## Upload Results

```bash
chmod +x push_vggsound_full_f16_followup_results.sh
./push_vggsound_full_f16_followup_results.sh
```

The push helper avoids `.pt` checkpoints and large feature files.
