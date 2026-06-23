# VGGSound Full Best Audio-Video Two-Port BM

This package fuses the current best available unimodal features:

- video: `vggsound_full_video_lstm4096_resnet50_f16_seed123.npz`
- audio: `vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz`

It first aligns samples by `path_train/path_test`, then runs:

| ID | model | purpose |
|---|---|---|
| AV001 | standard BM on `(video + audio) / 2` | simple one-port fusion baseline |
| AV002 | two-port BM, `gamma=1.15`, h8 | main current-feature fusion |
| AV003 | two-port BM, `gamma=0.50`, h8 | weaker interaction test |
| AV004 | two-port BM, `gamma=0.00`, h8 | no cross-term ablation |
| AV005 | two-port BM, `gamma=1.15`, h6 | smaller capacity control |

Important accounting note:

```text
two-port: video 4096 + audio 4096 -> 4096 two-port visible p-bits
standard concat: video 4096 + audio 4096 -> 8192 single-port visible p-bits
```

Therefore the `AV00x` branch keeps the main two-port comparison at 4096 visible p-bits. The 8192 concat standard-BM diagnostic is in the waiting parameter sweep package, not in this main two-port package.

Run on server:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_best_av_twoport_bm_code_20260621.zip
chmod +x sbatch_vggsound_full_best_av_twoport_bm.sh push_vggsound_full_best_av_twoport_bm_results.sh
sbatch sbatch_vggsound_full_best_av_twoport_bm.sh
```

The sbatch requests 2 GPUs, 12 CPU, and 100G memory.
