# VGGSound Phase 1 Continuation

This package continues the data/encoder improvement branch after AV016 became the current main two-port BM result.

## Current Main Result

AV016 is the current main VGGSound result:

```text
video = video_lstm4096
audio = paper-STFT ResNet50 audio_lstm4096
two-port BM hidden = h8 = 32768
total p-bit = 38409
full Gibbs best = 57.86%
```

The goal of this package is not to run ablations. It screens stronger unimodal features first. A new feature should only be promoted to two-port BM if its unimodal BM clearly improves over the current controls:

```text
video-only best: VF026 = 42.84%
audio-only best: AF036 = 44.98%
```

## Experiments

### Video

- `P1V002`: 32 frames, 320 center crop, standard video LSTM4096 encoder, standard BM h8/e320.
- `P1V003`: reuse the completed P1V001 24-frame 320 ResNet50 sequence feature, but train a stronger temporal encoder with `proj_dim=768` and `lstm_hidden=768`, then standard BM h8/e360.

### Audio

- `P1A001` retry: same dense `n_fft=1024`, `hop=160` STFT as the failed Job 323, but with much lower STFT worker pressure: `workers=4`, `worker_chunksize=2`, and `--resume`.
- `P1A002`: paper-STFT resolution, but denser 8-chunk temporal sequence before audio LSTM.

## Server Commands

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_phase1_continuation_code_20260624.zip

chmod +x sbatch_vggsound_phase1_video_continuation.sh
chmod +x sbatch_vggsound_phase1_audio_continuation.sh
chmod +x push_vggsound_phase1_continuation_results.sh

mkdir -p logs
sbatch sbatch_vggsound_phase1_video_continuation.sh
sbatch sbatch_vggsound_phase1_audio_continuation.sh
```

After completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_phase1_continuation_results.sh
```
