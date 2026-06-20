# VGGSound Full Experiment Status

Updated: 2026-06-20

This file is the running status ledger for the full VGGSound branch. It separates completed results, code already prepared but not yet analyzed, and discussed future directions.

## Dataset State

| item | status | note |
|---|---|---|
| Source dataset | available on server | HuggingFace VGGSound archive was mostly extracted |
| Missing/corrupt shard | `vggsound_08.tar.gz` | extraction failed; current experiments use available clips without this shard |
| Effective full feature split | `train=182615`, `test=15341`, `classes=309` | used by current full VGGSound experiments |
| Random top-1 | `1 / 309 = 0.32%` | all reported full eval values are top-1 Gibbs classification accuracy |

## Completed Results

### Video-Only BM

| branch | best experiment | full eval | note |
|---|---|---:|---|
| early full visual BM | VF003 | 20.08% | ResNet50 mean+std, 16 frames, h4, epoch 60 |
| f16 visual follow-up | VF008 | 32.91% | ResNet50 mean+std, 16 frames, h8, epoch 120 |
| video scale-up | VF010 | 37.66% | continued h8 to epoch 240; best visual-only result currently confirmed |
| h10 scale-up | VF011 | 37.16% | larger hidden from scratch, did not beat VF010 |

### Audio-Only BM

| branch | best experiment | full eval | delta vs previous audio best | note |
|---|---|---:|---:|---|
| direct STFT BM | AF004 | 4.47% | - | raw STFT128x96 directly into BM |
| CNN embedding BM | AF008 | 20.75% | +16.28 pp vs AF004 | small supervised audio CNN embedding, 4096 dim, h6 |
| CNN-LSTM embedding BM | AF012 | 22.78% | +2.03 pp vs AF008 | 4096 dim, h6, reached best at final epoch 180 |
| CNN-LSTM continuation | AF016 | 24.74% | +1.96 pp vs AF012 | continued AF012 to epoch 260 |
| CNN-LSTM continuation | AF017 | 25.53% | +0.79 pp vs AF016 | continued AF016 to epoch 300; current best audio-only BM |

AF017 is now the strongest audio-only BM result. Its quick best is also at the final epoch:

```text
AF017 quick best = 25.55% at epoch 300
AF017 full best  = 25.53% at epoch 300
```

This means AF012 was undertrained. The gain has slowed, but the curve has not clearly saturated.

## Code Prepared, Waiting For Results Or Analysis

| branch | IDs | code files | status | purpose |
|---|---|---|---|---|
| Audio ResNet50 embedding BM | ARF001 / AF013-AF015 | `make_vggsound_full_audio_resnet50_encoder_features.py`, `run_vggsound_full_audio_resnet50_bm.py`, `sbatch_vggsound_full_audio_resnet50_bm.sh` | submitted or waiting for server result | test whether stronger ResNet50 spectrogram teacher improves audio embedding quality |
| Video ResNet50 sequence + LSTM BM | VLF001-VLF002 / VF020-VF022 | `make_vggsound_full_video_resnet_sequence_features.py`, `make_vggsound_full_video_lstm_encoder_features.py`, `run_vggsound_full_video_lstm_bm.py`, `sbatch_vggsound_full_video_lstm_bm.sh` | submitted or waiting for server result | preserve frame order before video BM instead of mean+std pooling |
| Video large-hidden probes | VF012-VF013 | `run_vggsound_full_video_scaleup.py`, `sbatch_vggsound_full_video_scaleup.sh` | code exists; confirmed results not yet in current summary | test h12/h16 visual-only BM scale-up if resources allow |

## AF012 Continuation Result

| experiment | resume source | final epoch | input dim | label dim | hidden dim | total pbits | quick best | full best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AF016 standard_audio_cnnlstm4096_h6_lc5_e260_resume_af012 | AF012 `last.pt` | 260 | 4096 | 1545 | 24576 | 30217 | 24.71% | 24.74% |
| AF017 standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016 | AF016 `last.pt` | 300 | 4096 | 1545 | 24576 | 30217 | 25.55% | 25.53% |

Both kept the AF012 geometry fixed. The only change was longer training.

## Interpretation

The full VGGSound branch is no longer using raw video/audio directly for the main results. Effective BM performance only appeared after supervised feature extraction:

```text
video: frames -> ResNet50 embedding -> BM
audio: STFT -> CNN / CNN-LSTM / ResNet50 embedding -> BM
```

Current best unimodal results:

```text
video-only BM: VF010 = 37.66%
audio-only BM: AF017 = 25.53%
```

The audio branch is still weaker than video, but the improvement from STFT to learned audio embeddings is very large:

```text
AF004 direct STFT:        4.47%
AF008 CNN embedding:     20.75%
AF012 CNN-LSTM e180:     22.78%
AF017 CNN-LSTM e300:     25.53%
```

The main bottleneck is no longer whether BM can learn from audio at all. The bottleneck is the quality of the audio embedding and whether the BM needs longer training or more hidden capacity.

## Next Priority

| priority | next experiment | reason |
|---:|---|---|
| 1 | Wait for Audio ResNet50 ARF001 / AF013-AF015 | VGGSound paper-style ResNet on spectrograms may produce a stronger audio embedding than the current small CNN-LSTM |
| 2 | If AF017 remains best, continue AF017 to epoch 360 or 400 | best epoch is still final epoch 300, so the curve has not clearly saturated |
| 3 | Try CNN-LSTM4096 hidden 8x from scratch | AF017 is h6; larger hidden may help, but it cannot warm-start cleanly from h6 because hidden dimensionality changes |
| 4 | Wait for Video LSTM VLF/VF020-VF022 | if temporal video embedding beats VF010, use it for the future two-port fusion |
| 5 | Run first full VGGSound two-port BM | use strongest video feature and strongest audio feature after the pending branches settle |

## Discussed Future Directions

| direction | priority | rationale | trigger |
|---|---|---|---|
| Continue AF017 longer | high | AF017 best epoch was the final epoch 300 | if ResNet50 audio does not immediately beat AF017 |
| Audio ResNet50 teacher | high | VGGSound paper uses ResNet-style CNN on spectrograms; current audio CNN-LSTM teacher top1 is only about 35% | ARF001 teacher top1 or BM full eval beats AF017 |
| Stronger audio encoders | medium | ResNet50 may still be below paper baseline; possible later encoders include deeper ResNet or pretrained audio models | ResNet50 BM remains below video BM |
| Video temporal modeling | high | current best video feature uses mean+std over frames and discards temporal order | VLF/VF020-VF022 results beat or approach VF010 |
| Better motion representation | medium | current motion is frame-difference ResNet50, not optical flow; it underperforms appearance by about 5 pp in early runs | if visual temporal branch shows motion/order matters |
| Audio+video two-port BM | high but delayed | two-port fusion only makes sense after unimodal video/audio baselines are strong and stable | after audio ResNet50 and video LSTM results are known |
| Video+motion two-port BM | medium | pure visual two-port option without audio; tests whether appearance and motion interact usefully | after better motion/temporal features exist |
| Larger BM capacity for video | medium | VF010 h8 still improved to epoch 240; h10 did not clearly beat h8, but h12/h16 are not fully confirmed | if memory/runtime acceptable |
| Full-dataset integrity check | medium | shard 08 is missing/corrupt; current split is usable but not perfectly complete | before final paper-level claims |

## Decision Rule For Two-Port Fusion

Do not run the final two-port branch until these are known:

1. Audio ResNet50 result.
2. Video LSTM result.
3. Whether AF017 still improves at longer epoch if ResNet50 is weaker.

Then choose:

```text
best audio feature = max(AF017, AF013-AF015, later AF018/AF019)
best video feature = max(VF010, VF020-VF022)
```

The first full VGGSound two-port BM should compare:

```text
video-only BM
audio-only BM
video+audio two-port BM
```

using the same feature split and the same final Gibbs eval standard.

## Newly Prepared 2026-06-20

| branch | IDs | setup | status |
|---|---|---|---|
| AF017 longer continuation | AF018-AF019 | h6, 30217 p-bit, continue epoch 300 -> 500 -> 700 | code packaged |
| CNN-LSTM4096 hidden 8x BM | AF020-AF021 | h8, 38409 p-bit, train epoch 300 then continue to 500 | code packaged |

## AF017 Longer Continuation Pending/Result

| experiment | final epoch | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| AF018_standard_audio_cnnlstm4096_h6_lc5_e500_resume_af017 | 500 | 24576 | 30217 | 490 | 28.03% | 28.01% |
| AF019_standard_audio_cnnlstm4096_h6_lc5_e700_resume_af018 | 700 | 24576 | 30217 | 700 | 29.67% | 29.78% |

## CNN-LSTM4096 Hidden 8x Pending/Result

| experiment | final epoch | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| AF020_standard_audio_cnnlstm4096_h8_lc5_e300 | 300 | 32768 | 38409 | 300 | 28.03% | 27.76% |
| AF021_standard_audio_cnnlstm4096_h8_lc5_e500_resume_af020 | 500 | 32768 | 38409 | 475 | 29.37% | 27.42% |
