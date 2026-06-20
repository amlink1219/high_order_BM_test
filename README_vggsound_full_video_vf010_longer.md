# VGGSound VF010 Longer Video BM

This package continues the current best video-only standard BM:

```text
VF010 = ResNet50 mean/std f16 video feature, hidden 8x, epoch 240
full eval = 37.66%
```

The geometry is unchanged:

```text
video input = 4096
label bits = 309 * 5 = 1545
hidden bits = 32768
total p-bits = 38409
```

New runs:

| ID | resume source | final epoch |
|---|---|---:|
| VF014 | VF010 `last.pt` | 360 |
| VF015 | VF014 `last.pt` | 480 |

This directly tests whether the best h8 video BM is still undertrained. It avoids the incomplete h12/h16 branch from JobID 289.

## Server Run

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_video_vf010_longer_code_20260620.zip
chmod +x sbatch_vggsound_full_video_vf010_longer.sh
chmod +x push_vggsound_full_video_vf010_longer_results.sh
sbatch sbatch_vggsound_full_video_vf010_longer.sh
```

Upload after completion:

```bash
./push_vggsound_full_video_vf010_longer_results.sh
```

Required existing files:

```text
runs_vggsound_full_VF010_video_resnet50_meanstd_f16_h8_lc5_resume240/last.pt
runs_vggsound_full_VF010_video_resnet50_meanstd_f16_h8_lc5_resume240/best.pt
runs_vggsound_full_VF010_video_resnet50_meanstd_f16_h8_lc5_resume240/history.json
data_vggsound_full/features/vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz
```
