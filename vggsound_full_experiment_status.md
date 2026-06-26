# VGGSound Full Experiment Status

Updated: 2026-06-23

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
| mean/std continuation | VF014 | 39.14% | continued VF010 to epoch 360; best quick epoch 350 |
| mean/std continuation | VF015 | 39.45% | continued VF014 to epoch 480; best quick epoch 390, final epoch regressed |
| video LSTM | VF020 | 33.79% | ResNet50 frame sequence -> BiLSTM 2048 embedding, h6, epoch 220 |
| video LSTM | VF021 | 38.99% | ResNet50 frame sequence -> BiLSTM 4096 embedding, h6, epoch 220 |
| video LSTM | VF022 | 40.68% | ResNet50 frame sequence -> BiLSTM 4096 embedding, h8, epoch 220; former best video-only BM |
| video LSTM continuation | VF023 | 41.93% | continued VF022 to epoch 360; best epoch 335, final quick 42.03% |
| video LSTM continuation | VF024 | 42.74% | continued VF023 to epoch 500; best epoch 475, final quick 42.59%; current best video-only BM |
| video LSTM 8192 visible | VF025 | 42.54% | new VLF003 8192-d BiLSTM feature, h4, epoch 260; best epoch 255, final quick 42.48% |
| video LSTM 8192 visible | VF026 eval-only | 42.10% | VLF003 8192-d BiLSTM feature, h6, JobID 302 stopped at epoch 149; JobID 309 full-evaled best checkpoint at epoch 145; incomplete, not final |
| video LSTM 8192 visible | VF026 | 42.84% | continued incomplete VF026 to epoch 260; best epoch 250; current best video-only BM, but only +0.10 pp over VF024 |

### Audio-Only BM

| branch | best experiment | full eval | delta vs previous audio best | note |
|---|---|---:|---:|---|
| direct STFT BM | AF004 | 4.47% | - | raw STFT128x96 directly into BM |
| CNN embedding BM | AF008 | 20.75% | +16.28 pp vs AF004 | small supervised audio CNN embedding, 4096 dim, h6 |
| CNN-LSTM embedding BM | AF012 | 22.78% | +2.03 pp vs AF008 | 4096 dim, h6, reached best at final epoch 180 |
| CNN-LSTM continuation | AF016 | 24.74% | +1.96 pp vs AF012 | continued AF012 to epoch 260 |
| CNN-LSTM continuation | AF017 | 25.53% | +0.79 pp vs AF016 | continued AF016 to epoch 300; former intermediate best |
| CNN-LSTM continuation | AF018 | 28.01% | +2.48 pp vs AF017 | continued AF017 to epoch 500 |
| CNN-LSTM continuation | AF019 | 29.78% | +1.77 pp vs AF018 | continued AF018 to epoch 700; former intermediate best |
| CNN-LSTM hidden 8x | AF020 | 27.76% | -2.02 pp vs AF019 | h8 from scratch to epoch 300 |
| CNN-LSTM hidden 8x | AF021 | 27.42% | -2.36 pp vs AF019 | continued h8 to epoch 500; did not beat h6 long training |
| CNN-LSTM continuation | AF022 | 30.73% | +0.95 pp vs AF019 | continued AF019 to epoch 900; best quick epoch 890 |
| CNN-LSTM continuation | AF023 | 31.21% | +0.48 pp vs AF022 | continued AF022 to epoch 1000; former best audio-only BM |
| ResNet50 global embedding | AF013 | 22.32% | -8.89 pp vs AF023 | ARF001 teacher top1 = 32.84%; 2048-d global embedding, h4 |
| ResNet50 global embedding | AF014 | 20.98% | -10.23 pp vs AF023 | same ARF001 2048-d embedding, h6 |
| ResNet50 global embedding | AF015 | 19.10% | -12.11 pp vs AF023 | same ARF001 2048-d embedding, h8 |
| ResNet50 128x96 sequence mean/std | AF024 | 26.58% | -4.63 pp vs AF023 | ARF002 8 chunks x 32 frames, mean/std4096, h6 |
| ResNet50 128x96 sequence mean/std | AF025 | 26.14% | -5.07 pp vs AF023 | same feature, h8; quick acc overestimated relative to full eval |
| ResNet50 128x96 sequence LSTM | AF026 | 28.97% | -2.24 pp vs AF023 | ARF003 LSTM4096 teacher top1 = 32.42%; h6 |
| ResNet50 128x96 sequence LSTM | AF027 | 30.76% | -0.45 pp vs AF023 | same ARF003 LSTM4096 feature, h8; best epoch 320; still below AF023 and far below AF029 |
| paper-STFT ResNet50 global | AF028 | 20.11% | -11.10 pp vs AF023 | ARF004 global2048, h8; quick acc was unreliable for final conclusion |
| paper-STFT ResNet50 mean/std | AF029 | 38.78% | +7.57 pp vs AF023 | ARF004 teacher top1 = 51.74%, top5 = 77.04%; former best audio-only BM |
| paper-STFT ResNet50 mean/std | AF030 | 36.65% | -2.13 pp vs AF029 | h8, epoch 450; quick best 41.01% but full eval dropped, so h8 did not help mean/std |
| paper-STFT ResNet50 LSTM | AF031 | 44.31% | +5.53 pp vs AF029 | ARF005 paper-STFT ResNet50 sequence -> LSTM4096, h6; former best audio-only BM |
| paper-STFT ResNet50 mean/std continuation | AF032 | 39.30% | +0.52 pp vs AF029 | continued AF029 to epoch 650; improves mean/std but still far below AF031 |
| paper-STFT ResNet50 mean/std continuation | AF033 | 39.57% | +0.79 pp vs AF029 | continued AF032 to epoch 850; improves slowly, still far below AF031 |
| paper-STFT ResNet50 LSTM continuation | AF034 | 44.55% | +0.24 pp vs AF031 | continued AF031 to epoch 650; best epoch stayed early at 505 |
| paper-STFT ResNet50 LSTM continuation | AF035 | 44.55% | +0.24 pp vs AF031 | continued AF034 to epoch 850; no further full-eval gain and final quick regressed |
| paper-STFT ResNet50 LSTM hidden 8x | AF036 | 44.98% | +0.67 pp vs AF031 | h8 from scratch to epoch 500; current best audio-only BM |
| paper-STFT ResNet50 seqconcat8192 hidden 4x | AF038 | 43.80% | -0.51 pp vs AF031 | less-compressed 8192 visible sequence concat, h4; below LSTM4096 h8 |

AF036 is now the strongest uploaded audio-only BM result. AF017/AF019/AF023 showed that the h6 audio CNN-LSTM BM benefited from much longer training:

```text
AF017 quick best = 25.55% at epoch 300
AF017 full best  = 25.53% at epoch 300
AF019 quick best = 29.67% at epoch 700
AF019 full best  = 29.78% at epoch 700
AF023 quick best = 30.85% at epoch 990
AF023 full best  = 31.21% at epoch 990
```

This means AF012/AF017/AF019 were undertrained. Hidden 8x did not beat h6 long training in the current run, so increasing BM size is not yet the preferred audio path. However, AF023 is close to a plateau: the final quick eval at epoch 1000 fell back to 30.43%.

The later paper-STFT ResNet50 route changed the audio picture substantially:

```text
ARF004 paper-STFT ResNet50 teacher top1 = 51.74%, top5 = 77.04%
AF029 paper-STFT ResNet50 mean/std4096 h6 full = 38.78%
AF031 paper-STFT ResNet50 LSTM4096 h6 full     = 44.31%
```

This confirms that the earlier weak audio results were mainly caused by the 128x96 audio preprocessing/encoder bottleneck, not by audio being intrinsically weaker for VGGSound.

## Code Prepared, Waiting For Results Or Analysis

| branch | IDs | code files | status | purpose |
|---|---|---|---|---|
| Audio ResNet50 global embedding BM | ARF001 / AF013-AF015 | `make_vggsound_full_audio_resnet50_encoder_features.py`, `run_vggsound_full_audio_resnet50_bm.py`, `sbatch_vggsound_full_audio_resnet50_bm.sh` | completed and analyzed | global 2048-d ResNet50 embedding underperformed AF023; teacher top1 = 32.84% |
| Audio ResNet50 temporal sequence BM | ARF002-ARF003 / AF024-AF027 | `make_vggsound_full_audio_resnet50_sequence_features.py`, `make_vggsound_full_audio_resnet50_lstm_encoder_features.py`, `run_vggsound_full_audio_resnet50_sequence_bm.py`, `sbatch_vggsound_full_audio_resnet50_sequence_bm.sh` | JobID 304 completed, uploaded, and analyzed | use ResNet50 spectrogram teacher, then compare chunk mean/std and LSTM sequence audio embeddings |
| Video ResNet50 sequence + LSTM BM | VLF001-VLF002 / VF020-VF022 | `make_vggsound_full_video_resnet_sequence_features.py`, `make_vggsound_full_video_lstm_encoder_features.py`, `run_vggsound_full_video_lstm_bm.py`, `sbatch_vggsound_full_video_lstm_bm.sh` | completed and analyzed | preserve frame order before video BM instead of mean+std pooling |
| VF022 continuation and 8192 visible scale-up | VLF003 / VF023-VF026 | `run_vggsound_full_video_vf022_extend.py`, `sbatch_vggsound_full_video_vf022_extend.sh` | JobID 302 stopped by time limit; JobID 309 eval-only completed, uploaded, and analyzed | test whether current 4096 visible dimension and 220 epochs are limiting video BM |
| Video large-hidden probes | VF012-VF013 | `run_vggsound_full_video_scaleup.py`, `sbatch_vggsound_full_video_scaleup.sh` | JobID 289 did not complete full planned sweep | test h12/h16 visual-only BM scale-up if resources allow |

## Server Job Ledger

`resources` means requested Slurm resources from the corresponding sbatch file, not guaranteed simultaneous active usage. Check `squeue` for live allocation.

| JobID | log file | branch | resources requested | state from available logs | result status |
|---:|---|---|---|---|---|
| 289 | `vggsound_full_video_scaleup_289.out` | video scale-up, VF009-VF013 | 1 GPU, 16 CPU, 120G, 2d | completed VF009-VF011, started VF012 and reached only epoch 10 in uploaded files | not a complete VF009-VF013 sweep; VF012 has no summary, VF013 did not appear in uploaded results |
| 296 | `vggsound_video_lstm_296.out` | Video ResNet50 sequence + LSTM, VLF001-VLF002 / VF020-VF022 | 1 GPU, 8 CPU, 80G, 3d | completed and uploaded | VF022 full = 40.68%, new best video-only BM |
| 297 | `vggsound_audio_cnn_lstm_af012_cont_297.out` | AF016-AF017 | 1 GPU, 8 CPU, 80G, 2d | completed | AF017 full = 25.53% |
| 298 | `vggsound_audio_cnn_lstm_af017_long_298.out` | AF018-AF019 | 1 GPU, 8 CPU, 80G, 12h | completed | AF019 full = 29.78% |
| 299 | `vggsound_audio_cnn_lstm_h8_299.out` | AF020-AF021 | 1 GPU, 8 CPU, 90G, 12h | completed | AF020 full = 27.76%, AF021 full = 27.42% |
| 300 | `vggsound_audio_cnn_lstm_af019_long_300.out` | AF022-AF023 | 1 GPU, 8 CPU, 80G, 12h | completed and uploaded | AF023 full = 31.21%, former best audio-only BM |
| 301 | `vggsound_video_vf010_long_301.out` | VF014-VF015 | 1 GPU, 8 CPU, 90G, 12h | completed and uploaded | VF015 full = 39.45%; improves VF010 but remains below VF022 |
| 302 | `vggsound_video_vf022_extend_302.out` | VLF003 / VF023-VF026 | 1 GPU, 8 CPU, 120G, 1d | time-limit stopped on 2026-06-21; partial results uploaded: VF023 full = 41.93%, VF024 full = 42.74%, VLF003 teacher top1 = 42.40%, VF025 full = 42.54%; VF026 reached epoch 149 but has no `summary.json` | VF024 is current best uploaded video-only BM; VF026 needs eval-only checkpoint assessment before deciding whether to continue |
| 304 | `vggsound_audio_resnet50_sequence_304.out` | ARF002-ARF003 / AF024-AF027 | 4 GPU, 32 CPU, 110G, 2d | completed and uploaded; AF024 full = 26.58%, AF025 full = 26.14%, AF026 full = 28.97%, AF027 full = 30.76% | 128x96 ResNet50 sequence route does not beat AF023 or AF029; no longer a main audio route |
| 306 | `vggsound_audio_paper_resnet50_306.out` | ARF004-ARF005 / AF028-AF031 | 2 GPU, 24 CPU, 110G, 2d | completed and uploaded: AF028 full = 20.11%, AF029 full = 38.78%, AF030 full = 36.65%, AF031 full = 44.31% | AF031 paper-STFT ResNet50 LSTM4096 was the first strong audio-only BM; later AF036 improved it |
| 307 | `vggsound_best_av_twoport_307.out` | AV001-AV005 | 2 GPU, 12 CPU, 100G | completed and uploaded; AV001-AV005 all have summaries | old audio CNN-LSTM fusion branch: AV002 gamma=1.15 full = 40.63%, better than avg baseline but below current unimodal bests |
| 308 | `vggsound_waiting_param_sweep_308.out` | WP001-WP008 | 2 GPU, 12 CPU, 100G | partial results uploaded: WP001-WP005 completed; WP006 reached epoch235 quick 38.56% with no summary; WP007-WP008 not reached/uploaded | low-priority waiting sweep using old audio CNN-LSTM features; useful diagnostics are already captured, recommend cancelling if GPU resources are needed |
| 309 | `vggsound_vf026_eval_only_309.out` | VF026 eval-only checkpoint assessment | 1 GPU, 8 CPU, 120G, 4h | completed and uploaded; VF026 epoch145 full eval = 42.10% | VF026 is below VF024 now, but it was still improving when stopped; continue only if video capacity test is worth another 7-8 h |
| 310 | `vggsound_vf026_continue_to260_310.out` | VF026 continuation to epoch260 | 1 GPU, 8 CPU, 120G, 12h | completed and uploaded: VF026 full = 42.84% | VF026 is current best video-only BM, marginally above VF024 |
| 311 | `vggsound_best_av_paper_audio_311.out` | AV006-AV010 paper-audio AV fusion | 1 GPU, 8 CPU, 100G, 1d | partial/failed: AV006 full = 41.48%; AV007 started then wrapper failed; AV008-AV010 not run/uploaded | need inspect/upload AV007 child stderr to diagnose two-port failure |
| 312 | `vggsound_audio_paperresnet_af029_cont_312.out` | AF032-AF033 AF029 continuation | 1 GPU, 8 CPU, 90G, 12h | completed and uploaded: AF032 full = 39.30%, AF033 full = 39.57% | mean/std audio continues improving slowly but is far below AF031 |

Current submitted-but-not-yet-analyzed request total, assuming Job 308 is still active or queued:

```text
Job 308 = 2 GPU, 12 CPU cores, 100G memory requested
Total queued/running request = 2 GPU, 12 CPU cores, 100G memory
Recommendation = cancel Job 308 if it is still running, because the remaining WP006-WP008 runs are not on the current main feature branch
```

Resource planning note:

- Current BM training scripts mostly use a single GPU per job. Simply requesting more GPUs will not speed up the BM loop unless the runner is written for multi-GPU.
- Feature extraction and teacher/encoder training can benefit more from multi-GPU, either through shard parallelism or `DataParallel`.
- For future large video/audio feature extraction jobs, prefer planning for 2-4 GPUs and 16-32 CPU cores if the script supports sharding or data parallelism.
- For pure BM continuation jobs, higher CPU/GPU requests may waste resources unless we explicitly modify the training script.

Note on JobID 289: previous wording "partially uploaded" was imprecise. The GitHub upload did include files, but the experiment itself appears incomplete for the planned VF009-VF013 sweep. Specifically, VF010/VF011 summaries exist, while VF012 only has a short history through epoch 10 and no `summary.json`; VF013 has no uploaded result.

## Runtime Estimates From Available Logs

These estimates come from wrapper log timestamps, not Slurm accounting. They should be revised whenever child stdout logs with exact completion times are available.

### JobID 302 / VF026

The 24h time limit belonged to the whole JobID 302 wrapper, not VF026 alone. VF026 appears to have started at about 2026-06-21 06:15 and was stopped by the JobID 302 wall-time limit at 2026-06-21 16:12.

```text
VF026 completed epoch = 149 / 260
VF026 best quick = 42.12% at epoch 145
elapsed for VF026 ~= 10 h
estimated speed ~= 4.0 min / epoch
remaining to epoch 260 ~= 111 epochs ~= 7-8 h training + full eval
```

Observed eval-only result:

```text
JobID 309 full eval of VF026 best checkpoint:
best checkpoint epoch = 145
quick selection acc = 42.12%
full eval acc = 42.10%
current best video-only BM VF024 = 42.74%
gap to VF024 = -0.64 pp
gap to VF025 = -0.44 pp
```

Decision:

```text
VF026 does not beat VF024 at epoch 145.
However, VF026 was still improving when JobID 302 stopped, and it has only reached 149/260 epochs.
If the goal is to complete the 8192-visible h6 capacity test, continuing VF026 from last.pt to epoch 260 is still reasonable.
If GPU time is tight, prioritize newer fusion/audio experiments because VF026 has not yet shown a clear advantage over VF024/VF025.
```

### JobID 304 / Audio ResNet50 128x96 Sequence

Observed child starts from `vggsound_audio_resnet50_sequence_304.out`:

```text
22:05:56 start ARF002 sequence extraction
22:10:43 start ARF003 LSTM feature training      -> ARF002 took ~5 min
22:28:37 start AF024 h6 mean/std BM              -> ARF003 took ~18 min
02:18:33 start AF025 h8 mean/std BM              -> AF024 took ~3 h 50 min
07:53:13 start AF026 h6 LSTM BM                  -> AF025 took ~5 h 35 min
11:42:05 start AF027 h8 LSTM BM                  -> AF026 took ~3 h 49 min
```

AF027 has the same h8 BM size and batch setting as AF025, so its expected runtime is roughly 5.5-6 h. If uninterrupted, JobID 304 should have completed around 17:15-18:00 on 2026-06-21. If AF027 still has no `summary.json`, inspect its child stdout/stderr before treating it as failed.

### JobID 306 / Audio Paper-STFT ResNet50 Route

Observed child starts from `vggsound_audio_paper_resnet50_306.out`:

```text
22:30:26 start paper-STFT memmap
22:37:59 start ARF004 paper-STFT ResNet50 teacher -> memmap resume step took ~8 min
05:05:27 start ARF005 LSTM feature                -> teacher/features took ~6 h 27 min
05:26:09 start AF028 global2048 h8 BM             -> ARF005 took ~21 min
08:03:47 start AF029 mean/std4096 h6 BM           -> AF028 took ~2 h 38 min
13:23:32 start AF030 mean/std4096 h8 BM           -> AF029 took ~5 h 20 min
```

AF030 is h8 with batch size 64. Using AF025 and AF029 as references, a conservative runtime estimate is 7-8 h, so AF030 would be expected around 20:30-21:30 on 2026-06-21 if uninterrupted. AF031 is h6 LSTM4096 for 450 epochs; based on AF029, estimate another 5-6 h. Therefore, from AF030 start, the remaining AF030+AF031 total is roughly 12-14 h.

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

## Audio-Video Fusion Results

### Job 307: Video LSTM4096 + Old Audio CNN-LSTM4096

This branch used the then-current audio feature, `audio_cnnlstm4096_stft128x96`, paired with the VLF002 video LSTM4096 feature. It is now an older audio-feature ablation because AF031 later became much stronger on audio-only BM.

| experiment | model | hidden dim | gamma | best epoch | quick best | final quick | full best |
|---|---|---:|---:|---:|---:|---:|---:|
| AV001 | standard avg(video,audio) | 32768 | - | 320 | 37.26% | 37.26% | 37.29% |
| AV002 | two-port | 32768 | 1.15 | 270 | 41.06% | 40.47% | 40.63% |
| AV003 | two-port | 32768 | 0.50 | 320 | 37.04% | 37.04% | 37.13% |
| AV004 | two-port | 32768 | 0.00 | 320 | 36.84% | 36.84% | 36.93% |
| AV005 | two-port | 24576 | 1.15 | 285 | 40.48% | 40.27% | 40.32% |

Interpretation:

- In this older feature branch, the true two-port interaction matters: `gamma=1.15` improves over the standard avg baseline by +3.34 pp (`40.63%` vs `37.29%`).
- Reducing/removing the interaction term collapses the result back to baseline level: `gamma=0.50` gives `37.13%`, and `gamma=0.00` gives `36.93%`.
- Reducing hidden capacity from h8 to h6 with `gamma=1.15` slightly hurts (`40.32%` vs `40.63%`).
- Even the best old-audio two-port result is below the later unimodal bests: video-only VF026 = `42.84%`, audio-only AF034/AF035 = `44.55%`.

### Job 313: Video LSTM4096 + AF031 Paper-STFT ResNet50 LSTM4096

This is the current main AF031 audio-video branch. It aligns the strongest uploaded 4096-d video feature with the strongest uploaded 4096-d audio feature.

| experiment | model | hidden dim | gamma | best epoch | quick best | final quick | full best | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| AV011 | standard avg(video,audio) | 32768 | - | 320 | 43.60% | 43.60% | 43.69% | completed |
| AV012 | two-port | 32768 | 1.15 | 180 | 54.76% | - | 55.09% | eval-only full Gibbs completed from `best.pt`; training stopped at epoch 182 due to CUDA launch timeout |
| AV016 | two-port | 32768 | 1.15 | 310 | 57.83% | 57.77% | 57.86% | completed AV012 continuation; current main VGGSound result |
| AV013 | two-port | 32768 | 0.50 | 85 partial | 43.33% | - | - | failed at epoch86 by CUDA launch timeout; no full eval |
| AV014 | two-port | 32768 | 0.00 | - | - | - | - | not reached |
| AV015 | two-port | 24576 | 1.15 | - | - | - | - | not reached |

Interpretation:

- The simple avg one-port fusion baseline is already strong: AV011 full = `43.69%`.
- AV011 beats video-only VF026 (`42.84%`) by +0.85 pp, but it is below audio-only AF031 (`44.31%`) and the current audio-only best AF036 (`44.98%`).
- AV012 was the first valid AF031 true two-port full-eval result. It reached quick `54.76%` at epoch 180 and eval-only full Gibbs `55.09%` from `best.pt`.
- AV016 continued the interrupted AV012 branch to epoch320 and is now the main result: best epoch `310`, quick `57.83%`, full Gibbs `57.86%`, final quick `57.77%`.
- The uploaded child stderr shows `torch.AcceleratorError: CUDA error: the launch timed out and was terminated` at `loss.item()`. This is a CUDA kernel timeout/asynchronous CUDA error, not a dataset-missing error and not an ordinary Python exception in the BM logic.
- AV016 is far above AV011 (`43.69%` full), video-only VF026 (`42.84%`), and audio-only AF036 (`44.98%`). This validates the current two-port BM fusion route on VGGSound.
- AV013 gamma=0.50 only reached quick `43.33%` before failure, already near baseline level, but it is not a formal ablation because it has no full eval.
- Near-term priority is no longer ablation first. Continue improving upstream video/audio features and only promote a new feature to two-port if its unimodal BM beats the current controls.

### Job 308: Waiting Parameter Sweep With Old Audio CNN-LSTM Feature

This branch was submitted only to keep the server busy while stronger audio/video branches were pending. It uses the older `audio_cnnlstm4096_stft128x96` feature, so it is no longer the main VGGSound direction after AF031/AF034.

| experiment | input / model | hidden dim | label copies | status | full best / partial |
|---|---|---:|---:|---|---:|
| WP001 | video LSTM4096 standard BM | 40960 | 5 | completed | 42.21% |
| WP002 | video LSTM4096 standard BM | 32768 | 8 | completed | 42.38% |
| WP003 | old audio CNN-LSTM4096 standard BM | 40960 | 5 | completed | 23.99% |
| WP004 | old audio CNN-LSTM4096 standard BM | 24576 | 8 | completed | 28.32% |
| WP005 | avg(video LSTM4096, old audio CNN-LSTM4096) standard BM | 32768 | 8 | completed | 40.66% |
| WP006 | avg(video LSTM4096, old audio CNN-LSTM4096) standard BM | 40960 | 5 | uploaded partial | quick 38.56% at epoch235 |
| WP007-WP008 | 8192-visible concat diagnostics | - | - | not reached/uploaded | - |

Interpretation:

- WP001/WP002 are video-only controls and do not beat the current video-only best VF026 (`42.84%`).
- WP003/WP004 confirm that the old audio CNN-LSTM feature is much weaker than the current paper-STFT ResNet50 LSTM audio branch: WP004 `28.32%` vs AF034/AF035 `44.55%`.
- WP005 is a standard one-port average fusion control. It reaches `40.66%`, but it is below AV011 (`43.69%`) and far below the partial AV012 quick trajectory (`54.76%`).
- WP006 is still incomplete in the uploaded files and is unlikely to become a main result, because it uses the same old audio feature and is only at quick `38.56%` by epoch235.
- Recommendation: if Job 308 is still running, it is reasonable to stop it and free the GPUs.

## Interpretation

The full VGGSound branch is no longer using raw video/audio directly for the main results. Effective BM performance only appeared after supervised feature extraction:

```text
video: frames -> ResNet50 embedding -> BM
audio: STFT -> CNN / CNN-LSTM / ResNet50 embedding -> BM
```

Current best unimodal and fusion results:

```text
video-only BM: VF026 = 42.84%
audio-only BM: AF036 = 44.98%
audio+video two-port BM: AV016 = 57.86%
```

The audio branch is now stronger than video on the best uploaded unimodal BM. The improvement from STFT to learned audio embeddings is very large:

```text
AF004 direct STFT:        4.47%
AF008 CNN embedding:     20.75%
AF012 CNN-LSTM e180:     22.78%
AF017 CNN-LSTM e300:     25.53%
AF019 CNN-LSTM e700:     29.78%
AF023 CNN-LSTM e1000:    31.21%
AF029 paper-STFT ResNet: 38.78%
```

The main bottleneck is no longer whether BM can learn from audio at all. The bottleneck is the quality of the audio embedding and whether the BM needs longer training or more hidden capacity.

## Next Priority

| priority | next experiment | reason |
|---:|---|---|
| 1 | Continue VF022 beyond epoch 220 | VF022 is the new best video-only BM and was still improving at final epoch |
| 2 | Wait for VF022 extension / JobID 302 | this directly tests whether longer VF022 training and 8192 visible input beat 40.68% |
| 3 | Do not continue VF010 mean/std further for now | VF015 improves VF010 but still loses to VF022 and final epoch regressed |
| 4 | Wait for Audio ResNet50 ARF001 / AF013-AF015 if still pending | VGGSound paper-style ResNet on spectrograms may produce a stronger audio embedding than the current CNN-LSTM |
| 5 | Run provisional full VGGSound two-port BM | use current best available video/audio features while JobID 302/304/306 are still running |

## Discussed Future Directions

| direction | priority | rationale | trigger |
|---|---|---|---|
| Continue AF023 longer | low | AF029 has superseded AF023 by +7.57 pp | only as a low-priority control |
| Audio ResNet50 teacher | high | VGGSound paper uses ResNet-style CNN on spectrograms; current audio CNN-LSTM teacher top1 is only about 35% | ARF001 teacher top1 or BM full eval beats AF017 |
| Stronger audio encoders | medium | ResNet50 may still be below paper baseline; possible later encoders include deeper ResNet or pretrained audio models | ResNet50 BM remains below video BM |
| Video temporal modeling | high | VF022 proves sequence/LSTM features beat mean+std by +3.02 pp | continue VF022, then use it as the video side of two-port BM |
| Better motion representation | medium | current motion is frame-difference ResNet50, not optical flow; it underperforms appearance by about 5 pp in early runs | if visual temporal branch shows motion/order matters |
| Audio+video two-port BM | current main result | AV012 shows a large gain over both unimodal BM and standard avg fusion | continue/resume AV012 and run gamma/hidden ablations |
| Video+motion two-port BM | medium | pure visual two-port option without audio; tests whether appearance and motion interact usefully | after better motion/temporal features exist |
| Larger BM capacity for video | medium | VF010 h8 still improved to epoch 240; h10 did not clearly beat h8, but h12/h16 are not fully confirmed | if memory/runtime acceptable |
| Full-dataset integrity check | medium | shard 08 is missing/corrupt; current split is usable but not perfectly complete | before final paper-level claims |

## Decision Rule For Two-Port Fusion

The final two-port branch should wait until these are known:

1. Audio ResNet50 result.
2. Video LSTM result.
3. Whether AF017 still improves at longer epoch if ResNet50 is weaker.

Then choose:

```text
best audio feature = AF031 paper-STFT ResNet50 LSTM4096
best video feature = VF026 8192 visible video LSTM h6, or VF024 if same-resource 4096-visible is required
```

The first full VGGSound two-port BM should compare:

```text
video-only BM
audio-only BM
video+audio two-port BM
```

using the same feature split and the same final Gibbs eval standard.

Before those final choices are available, run a provisional two-port branch using:

```text
video = VLF002 video_lstm4096_resnet50_f16, strong 4096-visible video BM VF024 = 42.74%
audio = paper-STFT ResNet50 mean/std4096, strong audio BM AF029 = 38.78%
```

## Newly Prepared 2026-06-20

| branch | IDs | setup | status |
|---|---|---|---|
| AF017 longer continuation | AF018-AF019 | h6, 30217 p-bit, continue epoch 300 -> 500 -> 700 | code packaged |
| CNN-LSTM4096 hidden 8x BM | AF020-AF021 | h8, 38409 p-bit, train epoch 300 then continue to 500 | code packaged |
| AF019 longer continuation | AF022-AF023 | h6, 30217 p-bit, continue epoch 700 -> 900 -> 1000 | code packaged |
| VF010 longer continuation | VF014-VF015 | h8, 38409 p-bit, continue epoch 240 -> 360 -> 480 | code packaged |
| VF022 continuation and visible scale-up | VF023-VF026 / VLF003 | continue VF022 4096 h8 to epoch 360/500; train 8192-dim video LSTM feature and BM h4/h6 | code packaged |
| Audio ResNet50 temporal sequence | ARF002-ARF003 / AF024-AF027 | base ResNet50 spectrogram teacher, 8 audio chunks, mean/std4096 and LSTM4096 features, h6/h8 BM | submitted as JobID 304 after reducing memory to 110G; 4 GPU, 32 CPU |
| Audio paper-STFT ResNet50 route | ARF004-ARF005 / AF028-AF031 | no 128x96 teacher bottleneck; ResNet50 trains on random 257x500 STFT crops and exports full-10s global/sequence embeddings for BM | submitted as JobID 306 after reducing to 2 GPU; 24 CPU, 110G |

## Newly Prepared 2026-06-21

| branch | IDs | setup | status |
|---|---|---|---|
| Existing best audio-video two-port BM | AV001-AV005 | align then-current VLF002 video LSTM4096 with ACL002 audio CNN-LSTM4096 by path; compare avg one-port baseline, gamma=1.15/0.5/0 ablations, h6/h8 capacity; two 4096-d inputs still correspond to 4096 two-port visible p-bits | code packaged as a provisional branch; should be superseded by a new AF029/VF024 fusion branch |
| Waiting parameter sweep | WP001-WP008 | standard BM only; video h10, video label_copies=8, audio h10, audio label_copies=8, avg-fusion h8/lc8 and h10/lc5; WP007-WP008 add true 8192-visible concat diagnostics | code packaged; requests 2 GPU, 12 CPU, 100G |
| VF026 continuation | VF026-to260 | continue incomplete VF026 from epoch149 `last.pt` to epoch260, preserving JobID 309 eval-only summary before overwrite; 8192 visible h6 video BM | code packaged; requests 1 GPU, 8 CPU, 120G, 12h |

## Newly Prepared 2026-06-22

| branch | IDs | setup | status |
|---|---|---|---|
| Current-best paper-audio AV two-port BM | AV006-AV010 | align VLF002 video LSTM4096 with AF029 paper-STFT ResNet50 mean/std4096; run avg one-port baseline, gamma=1.15/0.5/0, and h6/h8 controls | code packaged; requests 1 GPU, 8 CPU, 100G, 1d |
| AF029 paper-ResNet audio continuation | AF032-AF033 | continue strongest uploaded audio-only BM AF029 from epoch450 to epoch650/850 | code packaged; requests 1 GPU, 8 CPU, 90G, 12h |
| AV007 resume and AV008-AV010 | AV007-AV010 | resume failed AV007 from `last.pt` with smaller batch; then run AV008 gamma=0.50, AV009 gamma=0, AV010 h6 gamma=1.15 | code packaged; requests 1 GPU, 8 CPU, 100G, 2d |
| AF031 best-audio AV two-port BM | AV011-AV015 | align AF031 paper-STFT ResNet50 LSTM4096 audio with the best same-dim VLF002/VF024 video LSTM4096 feature; run avg one-port baseline, gamma=1.15/0.5/0, and h6/h8 controls | AV011 completed; AV012 full eval completed at 55.09%; AV013-AV015 still not reached |
| AF031 audio BM improvement | AF034-AF040 | continue AF031, test h8/h10 capacity, and test less-compressed visible encodings: seqconcat8192 and global2048+lstm4096 concat6144 | code packaged; two sbatch files: capacity and encoding variants; each requests 1 GPU |

Note: AV006-AV010 used AF029 mean/std audio and should now be treated as an older audio-feature ablation. The current main audio-video two-port branch is AV011-AV015 because AF031/AF034-style paper-STFT ResNet50 LSTM audio is the strongest uploaded audio feature family. VF026 is the strongest uploaded video-only BM result, but it uses 8192-d video features; the current two-port trainer requires equal port dimensions, so AV011-AV015 uses the strongest 4096-d video branch instead.

## Submitted Jobs 2026-06-22

| JobID | squeue name | branch | IDs | requested resources | status / note |
|---:|---|---|---|---|---|
| 313 | `vgg-avaf` | AF031 best-audio AV two-port BM | AV011-AV015 | 1 GPU, 8 CPU, 100G, 2d | completed with failure after AV011: AV011 full = 43.69%; AV012 reached epoch 182, quick best = 54.76%, then CUDA launch timeout; AV013-AV015 not reached |
| 316 | `vgg-af31` | AF031 audio BM continuation/capacity | AF034-AF037 | 1 GPU, 8 CPU, 100G, 2d | completed and uploaded: AF034/AF035 h6 continuation full = 44.55%; AF036 h8 full = 44.98% and remains current best audio-only BM; AF037 h10 full = 44.31%, so h10 did not improve over h8 |
| 317 | `vgg-af31` | AF031 audio BM visible-encoding variants | AF038-AF040 | 1 GPU, 8 CPU, 110G, 2d | completed/partial upload analyzed: AF038 seqconcat8192 h4 full = 43.80%; AF039 seqconcat8192 h6 full = 44.81%; AF040 has config/history only and no summary |
| 318 | `vgg-av012-eval` | AV012 eval-only full Gibbs test | AV012 best checkpoint | 1 GPU, 8 CPU, 120G, 1d | completed and uploaded: epoch180 `best.pt`, full Gibbs `3000/500/thin=2`, eval batch 16, full acc = 55.09% |

## Newly Prepared 2026-06-23

| branch | IDs | setup | status |
|---|---|---|---|
| AV012 continuation | AV016 | resume interrupted AV012 from `last.pt` into a new output directory; keep copied epoch180 `best.pt` as baseline best; batch/eval batch reduced to 16; train to epoch320 and run full Gibbs eval | code packaged; requests 1 GPU, 8 CPU, 100G, 2d |
| AV012 ablations | AV013-AV015 | same AF031 audio LSTM4096 + VLF002 video LSTM4096 two-port setup; AV013 gamma=0.50 h8, AV014 gamma=0 h8, AV015 gamma=1.15 h6; batch/eval batch 16 | code packaged; requests 1 GPU, 8 CPU, 100G, 4d |

## Submitted Jobs 2026-06-23

| JobID | squeue name | branch | IDs | requested resources | status / note |
|---:|---|---|---|---|---|
| 319 | `vgg-av012c` | AV012 continuation | AV016 | 1 GPU, 8 CPU, 100G, 2d | completed and uploaded: AV016 full Gibbs = 57.86%, best epoch = 310; current main VGGSound result |
| 320 | `vgg-avabl` | AV012 ablations | AV013-AV015 | 1 GPU, 8 CPU, 100G, 4d | partial/failed and uploaded: AV013 gamma=0.50 reached epoch85 quick = 43.33%, then CUDA launch timeout at epoch86; AV014/AV015 not started |

## Next Ideas Recorded 2026-06-23

| direction | concrete idea | rationale | status |
|---|---|---|---|
| Stronger raw/early data processing | Try higher video frame resolution than 224, more frames, aspect-ratio-preserving crop/pad instead of direct square resize, and finer audio STFT/chunk processing | Current AV012 uses strong processed embeddings, but the upstream encoder may be bottlenecked by 224-frame visual input and fixed paper-STFT settings | recorded; design after AV016 / AV013-AV015 results |
| Deeper BM head | Extend final BM from the current visible-label-hidden two-layer style to a DBM-like architecture with an extra hidden layer | Current two-port gain is large, but final probabilistic model capacity may still be limited by a single hidden layer | recorded; needs new training/eval code and matched baselines |

## Training Conclusions And Plan 2026-06-23

The current main VGGSound direction is the AF031/VLF002-style two-port branch:

```text
video LSTM4096 + audio paper-STFT ResNet50 LSTM4096 -> two-port BM
AV016 full Gibbs acc = 57.86%
```

This result is far above the current unimodal and one-port fusion controls:

```text
video-only best: VF026 = 42.84%
audio-only best: AF036 = 44.98%
one-port avg fusion: AV011 = 43.69%
```

The previous AV012 checkpoint remains useful as the first proof of the branch, but AV016 supersedes it as the current main result:

```text
AV012 full Gibbs acc = 55.09%
AV016 full Gibbs acc = 57.86%
absolute gain = +2.77 pp
```

### Capacity And Epoch Lessons

| side | tested change | result | training conclusion |
|---|---|---|---|
| audio | Continue h6 AF031 from epoch450 to epoch650/850 | AF034/AF035 full = 44.55%; best stayed at epoch505; final quick regressed by epoch850 | Longer h6 training gives only small gain after ~500 epochs; do not spend more long jobs on h6 continuation unless used as a control |
| audio | Increase hidden from h6 to h8 | AF036 full = 44.98%; current audio-only best | h8 is useful for AF031 audio features and should be the default audio BM capacity |
| audio | Increase hidden from h8 to h10 | AF037 full = 44.31%, below AF036 | h10 does not help this audio feature; do not blindly scale hidden beyond h8 |
| audio | Less-compressed seqconcat8192 with h4 | AF038 full = 43.80%, below AF036 | Larger visible dimension alone is not enough; if using 8192 audio input, BM architecture/training must be redesigned |
| video | Mean/std ResNet50 h8 continuation | VF010 37.66% -> VF014 39.14% -> VF015 39.45%, but final epoch regressed | Longer training helps old mean/std video, but this branch is superseded by video LSTM features |
| video | Mean/std ResNet50 h10 from scratch | VF011 full = 37.16%, below VF010 | h10 did not help the old mean/std video feature |
| video | Video LSTM4096 h6 -> h8 | VF021 38.99% -> VF022 40.68% | h8 is useful for video LSTM4096 and should be the default two-layer video BM capacity |
| video | Continue video LSTM4096 h8 | VF022 40.68% -> VF023 41.93% -> VF024 42.74% | Longer training was valuable for the good video LSTM feature |
| video | Increase visible to 8192 | VF025 42.54%, VF026 42.84%; only +0.10 pp over VF024 | 8192 visible gives marginal gain relative to 4096 LSTM; use only when comparing high-capacity video features, not as default |

### Current Default Settings

Use these as the default starting point unless an experiment has a specific reason to differ:

```text
feature dim per port: 4096
label copies: 5
hidden factor: h8
gamma_h/gamma_l: 1.15 for main two-port, with gamma=0 and gamma=0.5 ablations
full eval: 3000 steps, 500 burn-in, thin=2
selection metric: test_label_gibbs_acc only
```

Avoid these unless they are explicit controls:

```text
audio h6 continuation beyond ~650 epochs
audio h10 with the current AF031 feature
old audio CNN-LSTM STFT128x96 feature
old video mean/std feature scale-up
8192 visible as a substitute for better encoder/input design
```

### Main Experiment Plan

| phase | goal | experiments | decision rule |
|---:|---|---|---|
| 0 | Current two-port main result | AV016 continuation of AV012 | AV016 is now the main checkpoint/result at full Gibbs `57.86%`; do not overwrite this conclusion with partial quick evals |
| 1 | Improve video input/encoder | Continue Phase 1 video: P1V002 32-frame/320 input and P1V003 stronger 24-frame/320 LSTM encoder | Promote to two-port only if video-only BM beats VF024/VF026 by a clear margin |
| 2 | Improve audio input/encoder | Continue Phase 1 audio: retry P1A001 with lower STFT workers and run P1A002 denser temporal chunks | Promote to two-port only if audio-only BM beats AF036 by a clear margin |
| 3 | Optimize current two-layer BM | h8 default; limited sweeps over label copies, gamma, label inhibit, training length | Do not repeat h10/long-epoch sweeps unless the feature changed substantially |
| 4 | Add DBM-like BM branch | Two-port visible layer with two hidden layers; matched standard DBM baseline | Compare under matched p-bit budget and same full Gibbs eval |
| 5 | Separate fused-head branch | Raw-aligned audio/video fusion CNN/LSTM -> split two port patterns -> BM | Deferred for now; keep as a recorded future AVF branch, not part of the near-term plan. If resumed later, do not mix with AV012; include classifier-only, standard BM, gamma=0 controls |

### Reporting Rules

- Record negative or low-yield conclusions, not only best accuracy.
- Do not compare quick eval to full eval as final accuracy.
- Do not mix AV012-style independent processed embeddings with future AVF fused-head inputs in the same result table.
- Every promoted feature should have unimodal BM controls before two-port fusion.
- Every new two-port result needs at least gamma=0 and one-port fusion controls.

## Newly Prepared 2026-06-23 Phase 1 Stronger Data

| branch | IDs | setup | status |
|---|---|---|---|
| Stronger video input | P1V001 | 24 RGB frames, 320 x 320 aspect-ratio-preserving center crop, ImageNet ResNet50 per frame, BiLSTM4096, standard BM h8/e360 | code packaged; first-round unimodal screening only |
| Stronger video optional | P1V002 | 32 RGB frames, 320 x 320 aspect-ratio-preserving center crop, BiLSTM4096, standard BM h8/e320 | defined in runner but not submitted by default |
| Stronger audio STFT | P1A001 | 16 kHz, n_fft=1024, hop=160, about 513 x 994 full-10s STFT, ResNet50, 4 chunks x 500, LSTM4096, standard BM h8/e500 | code packaged; first-round unimodal screening only |
| Stronger audio temporal chunks optional | P1A002 | original paper-STFT n_fft=512/hop159, but 8 chunks x 250 before LSTM4096, standard BM h8/e500 | defined in runner but not submitted by default |

Decision rule: promote a Phase 1 feature to two-port only if video-only clearly beats VF024/VF026 or audio-only clearly beats AF036. Otherwise record the result as a data-processing negative/low-yield conclusion.

## Uploaded/Analyzed 2026-06-24 Phase 1 And AV016

| job | IDs | result | conclusion |
|---:|---|---|---|
| 319 | AV016 | full Gibbs `57.86%`, best epoch 310 | current main VGGSound two-port BM result |
| 320 | AV013 partial | quick `43.33%` at epoch85, failed epoch86; AV014/AV015 not started | not a formal ablation; gamma=0.50 looks weak but needs full eval to claim |
| 321 | P1V001 | full Gibbs `42.56%` | 24 frames / 320 center crop did not beat video-only VF026 `42.84%` |
| 323 | P1A001 | failed during STFT extraction with exit code `-9` | likely resource/OOM kill; retry with lower STFT workers before judging the idea |

## Newly Prepared 2026-06-24 Phase 1 Continuation

| branch | IDs | setup | status |
|---|---|---|---|
| Video Phase1 continuation | P1V002 | 32 RGB frames, 320 center crop, ImageNet ResNet50 per frame, BiLSTM4096, standard BM h8/e320 | code packaged in `vggsound_phase1_continuation_code_20260624.zip` |
| Video encoder enhancement | P1V003 | reuse P1V001 24-frame/320 ResNet50 sequence feature; train stronger temporal encoder with `proj_dim=768`, `lstm_hidden=768`, embedding4096; standard BM h8/e360 | code packaged; isolates encoder strength from raw-frame setting |
| Audio Phase1 retry | P1A001 | dense STFT `n_fft=1024`, hop=160, but retry with workers=4 and chunksize=2 plus `--resume` | code packaged; designed to avoid Job 323 exit -9 |
| Audio chunk enhancement | P1A002 | paper-STFT resolution with denser 8-chunk temporal sequence before audio LSTM4096; standard BM h8/e500 | code packaged |

Decision rule remains unchanged: do not connect a Phase1 feature to AV016-style two-port BM unless its unimodal full Gibbs result clearly beats the current unimodal control (`VF026 = 42.84%` for video, `AF036 = 44.98%` for audio).

## Submitted Jobs 2026-06-24 Phase 1 Continuation

| JobID | squeue name | branch | IDs | requested resources | status / note |
|---:|---|---|---|---|---|
| 324 | `vgg-p1v2` | Phase1 video continuation | P1V002/P1V003 | 2 GPU, 16 CPU, 110G, 4d | submitted; tests 32-frame/320 input and stronger 24-frame/320 LSTM encoder |
| 325 | `vgg-p1a2` | Phase1 audio continuation | P1A001/P1A002 | 2 GPU, 20 CPU, 120G, 4d | running at user check: P1A001 completed 500 epochs, best quick `40.92%`; P1A002 reached epoch365/500, best quick `46.20%` at epoch275 and current `45.81%`; wait for final full eval before promoting |

## Newly Prepared 2026-06-24 Backbone Upgrade

| branch | IDs | setup | status |
|---|---|---|---|
| Video CNN backbone upgrade | P2V001 | 16 RGB frames, 300 direct resize, ImageNet EfficientNet-B3 per frame, LSTM4096 with `proj_dim=768` and `hidden=768`, standard BM h8/e360 | code packaged in `vggsound_backbone_upgrade_code_20260624.zip`; unimodal screening only |
| Audio CNN backbone upgrade | P2A001 | paper-STFT 257x1004, EfficientNet-B3 adapted to one-channel spectrogram input, 4 chunks x 500, LSTM4096, standard BM h8/e500 | code packaged in `vggsound_backbone_upgrade_code_20260624.zip`; unimodal screening only |

Rationale: AV016 uses ResNet50+LSTM processed features and reaches `57.86%`; the next encoder-level question is whether a stronger non-ResNet50 image backbone can produce better 4096-d video/audio evidence before BM fusion. EfficientNet-B3 is the first tested alternative because it is stronger than ResNet50 in many ImageNet settings while still feasible on the current 2-GPU jobs. If ImageNet weight download or memory becomes a problem, rerun the same scripts with `--backbone efficientnet_b0`, `--backbone resnet101`, or `--backbone wide_resnet50_2`.

Decision rule remains strict: do not replace AV016 inputs unless P2V001 beats video-only VF026 `42.84%` or P2A001 beats audio-only AF036 `44.98%` under full Gibbs evaluation.

## Submitted Jobs 2026-06-24 Backbone Upgrade

| JobID | squeue name | branch | IDs | requested resources | status / note |
|---:|---|---|---|---|---|
| 326 | `vgg-p2vid` | Video CNN backbone upgrade | P2V001 | 2 GPU, 16 CPU, 120G, 4d | failed immediately before training because video runner missed `--seed` argparse field; fixed in `vggsound_backbone_upgrade_code_20260624_fixed.zip`; resubmit needed |
| 327 | `vgg-p2aud` | Audio CNN backbone upgrade | P2A001 | 2 GPU, 20 CPU, 120G, 4d | active but impractically slow at user check: EfficientNet-B3 teacher reached epoch7 after about 2 days, epoch6 teacher top1 `50.16%`/top5 `77.37%`; process still using GPUs 4/5, but no LSTM feature or BM history yet; likely cancel/retry with shorter teacher or lighter export path |
| 328 | `vgg-p2vid` | Video CNN backbone upgrade | P2V001 fixed runner | 2 GPU, 16 CPU, 120G, 4d | fixed video runner resubmitted; queued behind running jobs at the time of submission |

## Newly Prepared 2026-06-26 P2A001 Resized Retry

| branch | ID | setup | status |
|---|---|---|---|
| Audio CNN backbone resized retry | P2A001R | paper-STFT `257x1004`; train crop `257x300`; bilinear resize to `300x300` before EfficientNet-B3; export `4 chunks x 300`; LSTM4096; standard BM h8/e260 | code packaged in `vggsound_backbone_p2a001_resized_code_20260626.zip`; first submission should use default `STOP_AFTER=teacher` to verify speed before downstream BM |

Rationale: original P2A001/Job 327 was not cost-effective because it only reached EfficientNet-B3 teacher epoch7 after about 48 hours. The retry keeps the same paper-STFT source but makes the CNN input close to EfficientNet's natural square image scale, and reduces the first teacher test to 12 epochs.

## Submitted Jobs 2026-06-26 P2A001 Resized Retry

| JobID | squeue name | branch | IDs | requested resources | status / note |
|---:|---|---|---|---|---|
| 330 | `vgg-p2aR` | Audio CNN backbone resized retry | P2A001R | 4 GPU, 40 CPU, 120G, 12h | submitted with `STOP_AFTER=teacher`; teacher batch 128, eval batch 192, export batch 128, workers 12/export workers 6; purpose is to verify whether 300x300 resized EfficientNet-B3 restores normal epoch speed |

## Newly Prepared 2026-06-26 AV016 Fixed-Feature Optimization

This branch keeps the current AV016 video/audio encoder features fixed and only tunes the two-port BM stage.

Reference remains:

```text
AV016 full Gibbs best = 57.86%
```

| ID | setup | status |
|---|---|---|
| AV017 | continue AV016 `last.pt` to epoch500, gamma=1.15, label_inhibit=0.30, cd_k=3 | code packaged in `vggsound_av016_fixed_feature_optimization_code_20260626.zip` |
| AV018 | warm-start AV016, gamma=1.10, label_inhibit=0.30, cd_k=3, epoch420 | code packaged |
| AV019 | warm-start AV016, gamma=1.20, label_inhibit=0.30, cd_k=3, epoch420 | code packaged |
| AV020 | warm-start AV016, gamma=1.15, label_inhibit=0.25, cd_k=3, epoch420 | code packaged |
| AV021 | warm-start AV016, gamma=1.15, label_inhibit=0.30, cd_k=5, epoch420 | code packaged |

Priority: AV017 first, then AV018/AV019 gamma bracket, then AV020 label inhibition, then AV021 cd_k=5. This is a low-risk attempt to push the current fixed-feature two-port BM from `57.86%` toward `58.5%-59%`; it should not be mixed with encoder-side P1/P2 experiments.

## Newly Prepared 2026-06-24 DBM-Like Branch

Input file verified locally:

```text
vggsound_full_aligned_videolstm4096_audioaf031_lstm4096_seed123.npz
video_train: (182615, 4096) float32
audio_train: (182615, 4096) float32
video_test:  (15341, 4096) float32
audio_test:  (15341, 4096) float32
```

This is the same processed AV016 evidence file: video端为 ResNet50-frame-sequence + LSTM 的 4096 维 embedding，audio端为 paper-STFT ResNet50 + LSTM 的 4096 维 embedding。

DBM-like structure:

```text
fixed video/audio evidence -> H1 <-> H2
label copies              -> H1
```

Training strategy is two-stage:

1. Pretrain the lower two-port BM with AV016-style `video/audio -> H1 -> label` structure.
2. Initialize the DBM-like model from the lower checkpoint, add `H2`, then jointly fine-tune `H1-H2-label` with CD updates.

The initial DBM-like sweep is packaged in `vggsound_dbmlike_code_20260624.zip`:

| ID | H1 | H2 | total p-bits | role |
|---|---:|---:|---:|---|
| D001 | 24576 | 4096 | 34313 | smaller DBM-like sanity run |
| D002 | 32768 | 4096 | 42505 | main DBM-like run, matching AV016 H1 plus one extra top layer |
| D003 | 32768 | 8192 | 46601 | optional larger top-layer run |

Decision rule: compare DBM-like full Gibbs accuracy against AV016 `57.86%`. If D002 does not improve clearly, do not expand DBM-like depth before improving the encoder inputs.

Server path note:

```text
old GPU server workdir: /home/Hongjie_Zeng/high_order_BM/
new server workdir:     /fs6/home/midfielder6/high_order_BM/
```

For DBM-like jobs on the new server, use `ROOT=/fs6/home/midfielder6/high_order_BM` and place the AV016 `.npz` under that directory unless `FEATURE_NPZ` is explicitly set.

## Prepared 2026-06-26 DBM-Like Multi-GPU Resume Patch

Current DBM-like jobs on the new server:

```text
745774 = D002, originally submitted with 1 GPU / 16 CPU / mem=0, currently single-GPU
753280 = D001, submitted with 1 GPU / 16 CPU / 180G, currently single-GPU
753281 = D003, queued; blocked by association CPU limit while D001/D002 are running
```

The DBM-like training script now supports `--data_parallel` and optional `--gpu_ids`. This uses `torch.nn.DataParallel` to split each mini-batch across the visible GPUs while preserving the same DBM-like CD update rule. Checkpoints still save the underlying model state, so existing single-GPU `last.pt` files can be resumed.

New resume helper:

```text
run_vggsound_dbmlike_multigpu_resume.py
sbatch_vggsound_dbmlike_multigpu_resume.sh
```

Default multi-GPU resume resources:

```text
partition 5090
gres gpu:2
cpus-per-task 16
mem 180G
global batch size 32
```

Recommended use: if a single-GPU DBM-like run is too slow, cancel the running job after confirming `last.pt/history.json` exist, then resume the same ID with the multi-GPU script. This avoids restarting pretraining or losing completed DBM-like epochs.

## Phase 1 Stronger Audio Input Results

| experiment | quick best | full best | note |
|---|---:|---:|---|
| P1A002 | 46.20% | 46.20% | Keep paper-STFT resolution but export denser 8-chunk temporal audio sequence before LSTM. |
