# VGGSound Full Audio STFT Standard BM

Updated: 2026-06-19 08:59:15

Purpose: pure audio standard BM baseline using VGGSound-paper-style log spectrogram inputs.

Reference scale: the VGGSound paper feeds an approximately 257x500 STFT spectrogram crop into an audio ResNet. ResNet-style CNNs then compress the spectrogram to a pooled 512/2048-d representation before the classifier. These BM inputs keep more explicit time-frequency bins than that pooled representation while avoiding the full 128k visible-pbit raw spectrogram.

Preprocessing: mp4 audio -> 16 kHz mono -> 5 s center crop -> STFT nperseg=512/noverlap=353 -> log(spec+1e-7) -> per-clip zscore -> resize -> sigmoid -> visible p-bits.

Planned input scales:

- 64x64 = 4096 visible p-bits, about 31x smaller than 257x500.
- 128x96 = 12288 visible p-bits, about 10.5x smaller than 257x500.

Best full eval in this batch: 4.47%

| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AF001_standard_audio_stft64x64_h4_lc5 | 309 | 4096 | 1545 | 16384 | 22025 | 60 | 1.79% | 1.84% |
| AF002_standard_audio_stft64x64_h6_lc5 | 309 | 4096 | 1545 | 24576 | 30217 | 60 | 2.04% | 2.07% |
| AF003_standard_audio_stft128x96_h3_lc5 | 309 | 12288 | 1545 | 36864 | 50697 | 60 | 4.04% | 4.05% |
| AF004_standard_audio_stft128x96_h4_lc5 | 309 | 12288 | 1545 | 49152 | 62985 | 60 | 4.45% | 4.47% |
