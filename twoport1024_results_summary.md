# Two-Port 1024 Result Summary

Final accuracy uses `final_full_test_label_gibbs_acc` when available.

## Top Results

| Rank | ID | Full Gibbs Acc | Purpose | Optical | Audio | Mix | Gamma | Seed |
|---:|---|---:|---|---|---|---:|---|---:|
| 1 | E018 | 98.64% | teacher_posterior_input_diagnostic | teacher_probs | raw | 0.5 | 1.15/1.15 | 123 |
| 2 | E021 | 97.61% | e017_multiseed_confirmation | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 124 |
| 3 | E022 | 97.52% | e017_multiseed_confirmation | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 125 |
| 4 | E025 | 97.46% | e017_multiseed_confirmation | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 128 |
| 5 | E032 | 97.39% | e017_gamma_refine | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.05/1.05 | 123 |
| 6 | E024 | 97.37% | e017_multiseed_confirmation | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 127 |
| 7 | E033 | 97.35% | e017_gamma_refine | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.1/1.1 | 123 |
| 8 | E017 | 97.31% | hybrid_processed_both_channels | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 123 |
| 9 | E028 | 97.31% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.3 | 1.15/1.15 | 123 |
| 10 | E036 | 97.29% | e017_split_gamma_refine | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.1/1.2 | 123 |
| 11 | E034 | 97.28% | e017_gamma_refine | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.2/1.2 | 123 |
| 12 | E027 | 97.27% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.25 | 1.15/1.15 | 123 |
| 13 | E026 | 97.24% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.2 | 1.15/1.15 | 123 |
| 14 | E035 | 97.23% | e017_gamma_refine | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.25/1.25 | 123 |
| 15 | E023 | 97.20% | e017_multiseed_confirmation | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 126 |
| 16 | E029 | 97.13% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.4 | 1.15/1.15 | 123 |
| 17 | E030 | 96.93% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.45 | 1.15/1.15 | 123 |
| 18 | E031 | 96.45% | e017_mix_sweep | raw_plus_image_rbm_probs | raw_plus_audio_mlp_probs | 0.5 | 1.15/1.15 | 123 |
| 19 | E037 | 95.93% | e017_ablation_optical_only | raw_plus_image_rbm_probs | raw | 0.35 | 1.15/1.15 | 123 |
| 20 | E038 | 95.69% | e017_ablation_audio_only | raw | raw_plus_audio_mlp_probs | 0.35 | 1.15/1.15 | 123 |

## All Runs

| ID | Full Gibbs Acc | Best Epoch | Final Epoch | Change |
|---|---:|---:|---:|---|
| E018 | 98.64% | 8 | 28 | optical=teacher_probs,audio=raw |
| E021 | 97.61% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=124 |
| E022 | 97.52% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=125 |
| E025 | 97.46% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=128 |
| E032 | 97.39% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.05,gamma_l=1.05,seed=123 |
| E024 | 97.37% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=127 |
| E033 | 97.35% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.1,seed=123 |
| E017 | 97.31% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35 |
| E028 | 97.31% | 8 | 28 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.3,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E036 | 97.29% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.2,seed=123 |
| E034 | 97.28% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.2,gamma_l=1.2,seed=123 |
| E027 | 97.27% | 12 | 32 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.25,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E026 | 97.24% | 18 | 38 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.2,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E035 | 97.23% | 6 | 26 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.25,gamma_l=1.25,seed=123 |
| E023 | 97.20% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=126 |
| E029 | 97.13% | 4 | 24 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.4,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E030 | 96.93% | 2 | 22 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.45,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E031 | 96.45% | 2 | 22 | optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.5,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E037 | 95.93% | 76 | 96 | optical=raw_plus_image_rbm_probs,audio=raw,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E038 | 95.69% | 6 | 26 | optical=raw,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 |
| E019 | 95.14% | 20 | 40 | optical=teacher_probs,audio=audio_mlp_probs |
| E010 | 95.11% | 86 | 100 | optical=image_rbm_probs,audio=raw |
| E003 | 94.22% | 34 | 50 | gamma_h=1.15,gamma_l=1.15 |
| E005 | 94.22% | 34 | 50 | gamma_h=1.10,gamma_l=1.20 |
| E001 | 94.20% | 34 | 50 | total_pbits_1024_only_from_best_twoport_config |
| E006 | 94.20% | 4 | 16 | warm_start=E003,distill_weight=0.02,lr=0.0001 |
| E007 | 94.19% | 4 | 16 | warm_start=E003,distill_weight=0.05,lr=0.0001 |
| E004 | 94.18% | 28 | 44 | gamma_h=1.20,gamma_l=1.20 |
| E008 | 94.16% | 4 | 16 | warm_start=E003,distill_weight=0.1,lr=0.0001 |
| E002 | 94.13% | 34 | 50 | gamma_h=1.05,gamma_l=1.05 |
| E015 | 92.56% | 2 | 22 | optical=image_rbm_probs,audio=audio_mlp_probs,pattern=blocks |
| E011 | 92.13% | 2 | 22 | optical=image_rbm_probs,audio=audio_mlp_probs |
| E014 | 89.07% | 2 | 22 | optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.75 |
| E013 | 88.94% | 100 | 100 | optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50 |
| E016 | 88.76% | 80 | 100 | optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50,pattern=blocks |
| E020 | 88.66% | 94 | 100 | optical=raw_plus_teacher_probs,audio=audio_mlp_probs,mix=0.35 |
| E012 | 87.43% | 88 | 100 | optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.25 |
| E009 | 86.54% | 2 | 22 | optical=raw,audio=audio_mlp_probs |
