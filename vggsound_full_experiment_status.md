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
| video scale-up | VF010 | 37.66% | ResNet50 mean+std, 16 frames, h8, continued to epoch 240 |
| h10 scale-up | VF011 | 37.16% | larger hidden from scratch, did not beat VF010 |
| video LSTM | VF020 | 33.79% | ResNet50 frame sequence -> BiLSTM 2048 embedding, h6, epoch 220 |
| video LSTM | VF021 | 38.99% | ResNet50 frame sequence -> BiLSTM 4096 embedding, h6, epoch 220 |
| video LSTM | VF022 | 40.68% | ResNet50 frame sequence -> BiLSTM 4096 embedding, h8, epoch 220; current best video-only BM |

### Audio-Only BM

| branch | best experiment | full eval | delta vs previous audio best | note |
|---|---|---:|---:|---|
| direct STFT BM | AF004 | 4.47% | - | raw STFT128x96 directly into BM |
| CNN embedding BM | AF008 | 20.75% | +16.28 pp vs AF004 | small supervised audio CNN embedding, 4096 dim, h6 |
| CNN-LSTM embedding BM | AF012 | 22.78% | +2.03 pp vs AF008 | 4096 dim, h6, reached best at final epoch 180 |
| CNN-LSTM continuation | AF016 | 24.74% | +1.96 pp vs AF012 | continued AF012 to epoch 260 |
| CNN-LSTM continuation | AF017 | 25.53% | +0.79 pp vs AF016 | continued AF016 to epoch 300; current best audio-only BM |
| CNN-LSTM continuation | AF018 | 28.01% | +2.48 pp vs AF017 | continued AF017 to epoch 500 |
| CNN-LSTM continuation | AF019 | 29.78% | +1.77 pp vs AF018 | continued AF018 to epoch 700; current best audio-only BM |
| CNN-LSTM hidden 8x | AF020 | 27.76% | -2.02 pp vs AF019 | h8 from scratch to epoch 300 |
| CNN-LSTM hidden 8x | AF021 | 27.42% | -2.36 pp vs AF019 | continued h8 to epoch 500; did not beat h6 long training |

AF019 is now the strongest audio-only BM result. AF017/AF019 both show that the h6 audio BM benefited from much longer training:

```text
AF017 quick best = 25.55% at epoch 300
AF017 full best  = 25.53% at epoch 300
AF019 quick best = 29.67% at epoch 700
AF019 full best  = 29.78% at epoch 700
```

This means AF012/AF017 were undertrained. Hidden 8x did not beat h6 long training in the current run, so increasing BM size is not yet the preferred audio path.

## Code Prepared, Waiting For Results Or Analysis

| branch | IDs | code files | status | purpose |
|---|---|---|---|---|
| Audio ResNet50 embedding BM | ARF001 / AF013-AF015 | `make_vggsound_full_audio_resnet50_encoder_features.py`, `run_vggsound_full_audio_resnet50_bm.py`, `sbatch_vggsound_full_audio_resnet50_bm.sh` | submitted or waiting for server result | test whether stronger ResNet50 spectrogram teacher improves audio embedding quality |
| Video ResNet50 sequence + LSTM BM | VLF001-VLF002 / VF020-VF022 | `make_vggsound_full_video_resnet_sequence_features.py`, `make_vggsound_full_video_lstm_encoder_features.py`, `run_vggsound_full_video_lstm_bm.py`, `sbatch_vggsound_full_video_lstm_bm.sh` | completed and analyzed | preserve frame order before video BM instead of mean+std pooling |
| VF022 continuation and 8192 visible scale-up | VLF003 / VF023-VF026 | `run_vggsound_full_video_vf022_extend.py`, `sbatch_vggsound_full_video_vf022_extend.sh` | code packaged, not submitted yet | test whether current 4096 visible dimension and 220 epochs are limiting video BM |
| Video large-hidden probes | VF012-VF013 | `run_vggsound_full_video_scaleup.py`, `sbatch_vggsound_full_video_scaleup.sh` | JobID 289 did not complete full planned sweep | test h12/h16 visual-only BM scale-up if resources allow |

## Server Job Ledger

| JobID | log file | branch | state from available logs | result status |
|---:|---|---|---|---|
| 289 | `vggsound_full_video_scaleup_289.out` | video scale-up, VF009-VF013 | completed VF009-VF011, started VF012 and reached only epoch 10 in uploaded files | not a complete VF009-VF013 sweep; VF012 has no summary, VF013 did not appear in uploaded results |
| 296 | `vggsound_video_lstm_296.out` | Video ResNet50 sequence + LSTM, VLF001-VLF002 / VF020-VF022 | completed and uploaded | VF022 full = 40.68%, new best video-only BM |
| 297 | `vggsound_audio_cnn_lstm_af012_cont_297.out` | AF016-AF017 | completed | AF017 full = 25.53% |
| 298 | `vggsound_audio_cnn_lstm_af017_long_298.out` | AF018-AF019 | completed | AF019 full = 29.78% |
| 299 | `vggsound_audio_cnn_lstm_h8_299.out` | AF020-AF021 | completed | AF020 full = 27.76%, AF021 full = 27.42% |
| 300 | `vggsound_audio_cnn_lstm_af019_long_300.out` | AF022-AF023 | submitted on 2026-06-20 | continue AF019 h6 audio BM from epoch 700 to 900/1000 |
| 301 | `vggsound_video_vf010_long_301.out` | VF014-VF015 | submitted on 2026-06-20 | continue VF010 video ResNet50 mean/std h8 BM from epoch 240 to 360/480 |

Note on JobID 289: previous wording "partially uploaded" was imprecise. The GitHub upload did include files, but the experiment itself appears incomplete for the planned VF009-VF013 sweep. Specifically, VF010/VF011 summaries exist, while VF012 only has a short history through epoch 10 and no `summary.json`; VF013 has no uploaded result.

## Video LSTM Result

JobID 296 tested whether preserving temporal order helps video-only BM classification. The pipeline was:

```text
video clip -> 16 RGB frames at 224x224 -> per-frame ResNet50 feature [16, 2048]
          -> supervised BiLSTM video encoder -> embedding -> standard BM
```

Teacher feature summaries:

| feature | embedding dim | teacher best epoch | teacher top1 |
|---|---:|---:|---:|
| VLF001 | 2048 | 5 | 42.17% |
| VLF002 | 4096 | 5 | 42.34% |

BM results:

| experiment | feature | hidden | total pbits | best epoch | quick best | full best |
|---|---|---:|---:|---:|---:|---:|
| VF020 | VLF001 2048 | 12288 | 15881 | 220 | 33.79% | 33.79% |
| VF021 | VLF002 4096 | 24576 | 30217 | 220 | 38.92% | 38.99% |
| VF022 | VLF002 4096 | 32768 | 38409 | 220 | 40.67% | 40.68% |

Comparison to the previous best:

```text
VF010 ResNet50 mean/std f16 h8 full = 37.66%
VF022 ResNet50 sequence + LSTM4096 h8 full = 40.68%
absolute gain = +3.02 percentage points
```

Interpretation:

- Temporal order is useful, but only when the exported feature is wide enough. The 2048 LSTM embedding underperformed VF010, while the 4096 LSTM embedding surpassed it.
- BM capacity still matters on video: h8 improved over h6 by about 1.76 pp at the same 4096 LSTM input.
- VF022's best epoch is the final epoch 220, and the quick accuracy was still rising at the end. This branch is likely undertrained.
- The BM result is now close to the LSTM teacher top1, so further gains need both longer BM training and a stronger video teacher/encoder.

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
video-only BM: VF022 = 40.68%
audio-only BM: AF019 = 29.78%
```

The audio branch is still weaker than video, but the improvement from STFT to learned audio embeddings is very large:

```text
AF004 direct STFT:        4.47%
AF008 CNN embedding:     20.75%
AF012 CNN-LSTM e180:     22.78%
AF017 CNN-LSTM e300:     25.53%
AF019 CNN-LSTM e700:     29.78%
```

The main bottleneck is no longer whether BM can learn from audio at all. The bottleneck is the quality of the audio embedding and whether the BM needs longer training or more hidden capacity.

## Next Priority

| priority | next experiment | reason |
|---:|---|---|
| 1 | Continue VF022 beyond epoch 220 | VF022 is the new best video-only BM and was still improving at final epoch |
| 2 | Wait for AF022-AF023 / JobID 300 | AF019 was also still improving at epoch 700; this determines the audio-only best |
| 3 | Wait for VF014-VF015 / JobID 301 | useful control to see whether the older mean/std VF010 branch catches up with longer training |
| 4 | Wait for Audio ResNet50 ARF001 / AF013-AF015 if still pending | VGGSound paper-style ResNet on spectrograms may produce a stronger audio embedding than the current CNN-LSTM |
| 5 | Run first full VGGSound two-port BM | use strongest video feature and strongest audio feature after the pending branches settle |

## Discussed Future Directions

| direction | priority | rationale | trigger |
|---|---|---|---|
| Continue AF019 longer | medium | AF019 best epoch was the final epoch 700, but gains are slowing | if ResNet50 audio does not beat AF019 and runtime is acceptable |
| Audio ResNet50 teacher | high | VGGSound paper uses ResNet-style CNN on spectrograms; current audio CNN-LSTM teacher top1 is only about 35% | ARF001 teacher top1 or BM full eval beats AF017 |
| Stronger audio encoders | medium | ResNet50 may still be below paper baseline; possible later encoders include deeper ResNet or pretrained audio models | ResNet50 BM remains below video BM |
| Video temporal modeling | high | VF022 proves sequence/LSTM features beat mean+std by +3.02 pp | continue VF022, then use it as the video side of two-port BM |
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
best video feature = max(VF022, VF014-VF015, later VF022 continuation)
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
| AF019 longer continuation | AF022-AF023 | h6, 30217 p-bit, continue epoch 700 -> 900 -> 1000 | code packaged |
| VF010 longer continuation | VF014-VF015 | h8, 38409 p-bit, continue epoch 240 -> 360 -> 480 | code packaged |
| VF022 continuation and visible scale-up | VF023-VF026 / VLF003 | continue VF022 4096 h8 to epoch 360/500; train 8192-dim video LSTM feature and BM h4/h6 | code packaged |

## VF010 Longer Continuation Pending/Result

| experiment | final epoch | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| VF014_video_resnet50_meanstd_f16_h8_lc5_resume360 | 360 | 32768 | 38409 | 350 | 38.47% | 39.14% |
| VF015_video_resnet50_meanstd_f16_h8_lc5_resume480 | 480 | 32768 | 38409 | 390 | 38.67% | 39.45% |
