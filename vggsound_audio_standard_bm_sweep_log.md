# VGGSound-mini20 Standard BM Audio Sweep

Updated: 2026-06-16 16:13:19

Purpose: test whether audio-only BM carries enough signal before trying video+audio two-port BM.

Random chance for 20 classes is 5%. Final conclusions should use full best eval.

| experiment | audio input dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| V010_standard_audio_64x32_perclip_h1 | 2048 | 2048 | 4196 | 205 | 15.99% | 15.70% |
| V011_standard_audio_96x64_permel_h1 | 6144 | 6144 | 12388 | 130 | 22.09% | 22.67% |
| V012_standard_audio_128x64_permel_h1 | 8192 | 8192 | 16484 | 115 | 22.09% | 22.09% |
| V013_standard_audio_128x96_permel_h1 | 12288 | 12288 | 24676 | 105 | 24.13% | 24.13% |
| V014_standard_audio_96x64_permel_h2 | 6144 | 12288 | 18532 | 210 | 23.55% | 24.13% |
| V015_standard_audio_96x64_global_h1 | 6144 | 6144 | 12388 | 220 | 20.35% | 20.35% |
| V016_standard_audio_96x64_permel_threshold_h1 | 6144 | 6144 | 12388 | 150 | 25.00% | 25.00% |
