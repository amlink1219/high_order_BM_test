# VGGSound Full Video BM Scale-up

This package runs the next video-side standard BM experiments after VF008.

## Context

Current best video-only full VGGSound standard BM:

- VF008: ResNet50 video f16 feature, input 4096, label copies 5, hidden 32768, total 38409 p-bits.
- Full eval: 32.91% at epoch 120.
- The VF008 curve was still increasing at the final epoch, so the next run tests both longer training and larger hidden layers.

## Experiments

| ID | purpose | input dim | label dim | hidden dim | total p-bits | epochs |
|---|---|---:|---:|---:|---:|---:|
| VF009 | continue VF008 h8 to epoch 180 | 4096 | 1545 | 32768 | 38409 | 180 |
| VF010 | continue h8 to epoch 240 | 4096 | 1545 | 32768 | 38409 | 240 |
| VF011 | train h10 from scratch | 4096 | 1545 | 40960 | 46601 | 160 |
| VF012 | train h12 from scratch | 4096 | 1545 | 49152 | 54793 | 160 |
| VF013 | aggressive h16 probe | 4096 | 1545 | 65536 | 71177 | 120 |

All runs use:

- feature file: `data_vggsound_full/features/vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz`
- input mode: `video`
- `cd_k=3`
- `lr=0.0002`
- `momentum=0.6`
- `weight_decay=0.0`
- quick eval: `400/100/thin=2`
- full eval on best: `3000/500/thin=2`

## Server Run

Copy these files to `/home/Hongjie_Zeng/high_order_BM`:

- `run_vggsound_full_video_scaleup.py`
- `sbatch_vggsound_full_video_scaleup.sh`
- `push_vggsound_full_video_scaleup_results.sh`
- `README_vggsound_full_video_scaleup.md`
- `train_vggsound_mini20_bm.py` if the server copy is not already up to date.

Submit:

```bash
cd /home/Hongjie_Zeng/high_order_BM
chmod +x sbatch_vggsound_full_video_scaleup.sh push_vggsound_full_video_scaleup_results.sh
sbatch sbatch_vggsound_full_video_scaleup.sh
```

Monitor:

```bash
squeue
tail -f logs/vggsound_full_video_scaleup_*.out
```

After completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
bash push_vggsound_full_video_scaleup_results.sh
```

## Notes

The BM training script currently uses one CUDA device. Requesting more GPUs will not automatically accelerate a single run unless the training code is rewritten for data/model parallelism.
