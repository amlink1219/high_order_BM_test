# VGGSound-mini20 V020 Reproduction And Tuning

Updated: 2026-06-16 20:11:52

Purpose: reproduce V020 and tune the best video-encoder-feature standard BM setting before trying two-port fusion.

Baseline reference: V020 ResNet50 mean+std, hidden 2x, seed 123, full best 47.38%.

Best full eval so far: 53.85%
V020 reproduction full eval: n=5, mean=47.75%, std=0.51%

| experiment | video feature dim | hidden dim | total pbits | best epoch | quick best | full best |
|---|---:|---:|---:|---:|---:|---:|
| V023_v020_repro_seed124 | 4096 | 8192 | 12388 | 210 | 48.00% | 48.00% |
| V024_v020_repro_seed125 | 4096 | 8192 | 12388 | 210 | 47.69% | 47.38% |
| V025_v020_repro_seed126 | 4096 | 8192 | 12388 | 200 | 47.69% | 47.08% |
| V026_v020_repro_seed127 | 4096 | 8192 | 12388 | 220 | 48.62% | 48.00% |
| V027_v020_repro_seed128 | 4096 | 8192 | 12388 | 215 | 48.62% | 48.31% |
| V028_v020_hidden1_seed123 | 4096 | 4096 | 8292 | 185 | 36.00% | 35.69% |
| V029_v020_hidden3_seed123 | 4096 | 12288 | 16484 | 205 | 53.85% | 53.85% |
| V030_v020_labelcopies10_seed123 | 4096 | 8192 | 12488 | 220 | 49.23% | 49.54% |
| V031_v020_threshold_seed123 | 4096 | 8192 | 12388 | 220 | 47.38% | 47.38% |
| V032_v020_sample_seed123 | 4096 | 8192 | 12388 | 195 | 47.69% | 44.00% |
| V033_resnet50_meanstd_zsig_seed123 | 4096 | 8192 | 12388 | 220 | 30.15% | 31.38% |
| V034_resnet50_meanmax_seed123 | 4096 | 8192 | 12388 | 215 | 48.92% | 49.23% |
| V035_resnet50_meanstd_f16_seed123 | 4096 | 8192 | 12388 | 215 | 48.31% | 48.62% |
