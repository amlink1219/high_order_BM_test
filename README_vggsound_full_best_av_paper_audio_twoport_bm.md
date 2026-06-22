# VGGSound Full Paper-Audio Audio-Video Two-Port BM

This package is the current-best feature fusion branch. It differs from JobID 307:

```text
307 audio feature: older audio CNN-LSTM4096
this package audio feature: AF029 paper-STFT ResNet50 mean/std4096
```

Inputs:

```text
video = vggsound_full_video_lstm4096_resnet50_f16_seed123.npz
audio = vggsound_full_audio_paperresnet50_seqmeanstd4096_chunks4_w500_seed123.npz
```

Reference unimodal results:

```text
video-only BM VF024 = 42.74%
audio-only BM AF029 = 38.78%
```

Experiments:

| ID | model | purpose |
|---|---|---|
| AV006 | standard BM on `(video + audio) / 2` | one-port fusion baseline |
| AV007 | two-port BM, gamma=1.15, h8 | main current-best fusion |
| AV008 | two-port BM, gamma=0.50, h8 | weaker interaction test |
| AV009 | two-port BM, gamma=0.00, h8 | no cross-term ablation |
| AV010 | two-port BM, gamma=1.15, h6 | smaller capacity control |

Accounting:

```text
two-port: video 4096 + audio 4096 -> 4096 two-port visible p-bits
standard avg: video/audio are averaged to 4096 single-port visible p-bits
```

## Submit

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_best_av_paper_audio_twoport_bm_code_20260622.zip
chmod +x sbatch_vggsound_full_best_av_paper_audio_twoport_bm.sh push_vggsound_full_best_av_paper_audio_twoport_bm_results.sh
sbatch sbatch_vggsound_full_best_av_paper_audio_twoport_bm.sh
```

The sbatch requests 1 GPU, 8 CPU, 100G memory, and 1 day.

## Upload

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_best_av_paper_audio_twoport_bm_results.sh
```
