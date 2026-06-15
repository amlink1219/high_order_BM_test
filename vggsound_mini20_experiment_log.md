# VGGSound-mini20 4096 p-bit experiments

Updated: 2026-06-15 18:28:00

Dataset: VGGSound-mini20 processed features, 20 classes, train/test from clean downloaded clips.

Final accuracy should use `full_eval_best_acc`, not quick selection accuracy.

| experiment | model | best epoch | quick best | full best | output |
|---|---:|---:|---:|---:|---|
| V001_standard_video_only | standard | 145 | 8.43% | 8.14% | /home/Hongjie_Zeng/high_order_BM/runs_vggsound_mini20_V001_standard_video_only |
| V002_standard_audio_only | standard | 160 | 15.12% | 14.53% | /home/Hongjie_Zeng/high_order_BM/runs_vggsound_mini20_V002_standard_audio_only |
| V003_twoport_video_audio | twoport | 170 | 13.95% | 12.79% | /home/Hongjie_Zeng/high_order_BM/runs_vggsound_mini20_V003_twoport_video_audio |
| V004_twoport_motion_audio | twoport | 180 | 13.37% | 12.21% | /home/Hongjie_Zeng/high_order_BM/runs_vggsound_mini20_V004_twoport_motion_audio |
| V005_twoport_video_motion | twoport | 180 | 9.88% | 9.59% | /home/Hongjie_Zeng/high_order_BM/runs_vggsound_mini20_V005_twoport_video_motion |
