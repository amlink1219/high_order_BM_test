# VGGSound Full AF031 Audio-Video Two-Port BM

This package supersedes the older AV006-AV010 AF029 mean/std audio branch.

## Why This Pairing

- Best uploaded audio-only BM: AF031, paper-STFT ResNet50 LSTM4096, full accuracy 44.31%.
- Best directly compatible video-only BM: VF024, VLF002 video LSTM4096, full accuracy 42.74%.
- Best overall video-only BM: VF026, VLF003 video LSTM8192, full accuracy 42.84%.

AF031 is 4096-dimensional. The current two-port VGGSound BM trainer requires the two ports to have the same visible dimension, so this branch uses VF024/VLF002 4096-d video rather than VF026/VLF003 8192-d video. Using VF026 would require a projection or padding strategy and would no longer be the clean same-pbit two-port comparison.

## Experiments

| ID | model | purpose |
|---|---|---|
| AV011 | standard BM on averaged video/audio feature | one-port fusion baseline with the same 4096 visible p-bits |
| AV012 | two-port BM, h8, gamma=1.15 | main AF031 audio-video two-port run |
| AV013 | two-port BM, h8, gamma=0.50 | weaker cross-term test |
| AV014 | two-port BM, h8, gamma=0.00 | no cross-term ablation |
| AV015 | two-port BM, h6, gamma=1.15 | smaller hidden capacity control |

Two-port runs use batch size 32 to reduce the chance of the CUDA timeout seen in the older AV007 run.

## Server Commands

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_best_av_af031_twoport_bm_code_20260622.zip
chmod +x sbatch_vggsound_full_best_av_af031_twoport_bm.sh push_vggsound_full_best_av_af031_twoport_bm_results.sh
sbatch sbatch_vggsound_full_best_av_af031_twoport_bm.sh
```

After the job finishes:

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_best_av_af031_twoport_bm_results.sh
```

## Expected Inputs On Server

```text
data_vggsound_full/features/vggsound_full_video_lstm4096_resnet50_f16_seed123.npz
data_vggsound_full/features/vggsound_full_audio_paperresnet50_lstm4096_chunks4_w500_h1024_seed123.npz
train_vggsound_mini20_bm.py
```
