# VGGSound Full Experiment Status

Updated: 2026-06-20 03:31:45

## Completed Audio Results

| branch | best experiment | full eval | note |
|---|---|---:|---|
| direct STFT BM | AF004 | 4.47% | raw STFT128x96 directly into BM |
| CNN embedding BM | AF008 | 20.75% | small supervised audio CNN embedding, 4096 dim, h6 |
| CNN-LSTM embedding BM | AF012 | 22.78% | 4096 dim, h6, reached best at final epoch 180 |

## Running Or Waiting

| branch | IDs | status | purpose |
|---|---|---|---|
| Audio ResNet50 embedding BM | ARF001 / AF013-AF015 | waiting for server result | test whether stronger ResNet50 spectrogram teacher improves audio embedding quality |
| Video ResNet50 sequence + LSTM BM | VLF001-VLF002 / VF020-VF022 | waiting or pending | test temporal frame-order modeling before video BM |

## Planned / Newly Prepared

| experiment | final epoch | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| AF016_standard_audio_cnnlstm4096_h6_lc5_e260_resume_af012 | 260 | 24576 | 30217 | 260 | 24.71% | 24.74% |
| AF017_standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016 | 300 | 24576 | 30217 | 300 | 25.55% | 25.53% |

AF016 continues AF012 from epoch 180 to 260. AF017 continues AF016 from epoch 260 to 300.
Both keep the AF012 model geometry fixed: 4096 audio input, 1545 label bits, 24576 hidden bits, total 30217 p-bits.
