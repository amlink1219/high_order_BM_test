# Two-Port 1024 Results Analysis

This note summarizes the current evidence for the 1024 p-bit two-port BM
optimization. Final accuracy claims use full test `test_label_gibbs_acc` only.

## Hardware/Inference Constraints

- Total physical p-bits: `1024`
- Input channels: two 400-dimensional physical input patterns
- Label p-bits: `50`
- Hidden p-bits: `574`
- Final inference: p-bit Gibbs sampling, not late-fusion post-processing
- Final full eval: `steps=3000`, `burn_in=500`, `thin=2`

## Baselines And Failed Directions

| Run group | Best full Gibbs acc | Main conclusion |
|---|---:|---|
| Previous 1000 p-bit two-port best | 94.59% | Old best reference |
| E001-E005 1024 gamma sweep | 94.22% | Gamma-only tuning did not improve the baseline |
| E006-E008 teacher distillation | 94.20% | KL/teacher loss did not transfer to Gibbs accuracy |
| E009 audio posterior input only | 86.54% | Replacing audio with posterior pattern alone damages the model |
| E011/E015 dual posterior only | 92.13% / 92.56% | Pure posterior patterns in both channels are not enough |

## Effective Processed-Input Results

| Run | Optical input | Audio input | Mix | Full Gibbs acc | Status |
|---|---|---|---:|---:|---|
| E010 | `image_rbm_probs` | raw audio | 0.50 | 95.11% | Useful conservative result |
| E017 | `raw_plus_image_rbm_probs` | `raw_plus_audio_mlp_probs` | 0.35 | **97.31%** | Current main-result candidate |
| E018 | `teacher_probs` | raw audio | 0.50 | 98.64% | Diagnostic upper bound, not the main physical claim |
| E019 | `teacher_probs` | `audio_mlp_probs` | 0.50 | 95.14% | Diagnostic; confirms audio posterior-only channel is brittle |
| E020 | `raw_plus_teacher_probs` | `audio_mlp_probs` | 0.35 | 88.66% | Diagnostic; not viable |

## Current Best Result

The strongest valid candidate is E017:

```text
final_full_test_label_gibbs_acc = 0.9731
best_checkpoint_epoch = 6
total_pbits = 1024
image_dim = 400
audio_dim = 400
label_dim = 50
hidden_dim = 574
gamma_h = 1.15
gamma_l = 1.15
distill_weight = 0.0
optical_feature_source = raw_plus_image_rbm_probs
audio_feature_source = raw_plus_audio_mlp_probs
processed_mix = 0.35
processed_feature_pattern = interleave
```

E017 remains a two-channel physical-input Gibbs model: both channels receive
400-dimensional input patterns, and the final recognition result is produced by
the BM Gibbs sampler.

## Interpretation

The successful pattern is not "teacher loss" and not "late-fusion inference".
The useful change is to encode deployable processed evidence into the physical
input channels before Gibbs inference. This preserves the two-port p-bit
hardware structure while making each channel more informative.

The results also show that replacing a channel entirely with a 10-class posterior
expanded to 400 dimensions is unstable. The best result keeps raw information in
both channels and mixes in processed posterior features:

```text
raw_plus_image_rbm_probs + raw_plus_audio_mlp_probs
```

## Remaining Confirmation Needed

E017 reaches the 96%-97% target, but it is currently a single seed/result. Before
using it as the main project conclusion, it should be confirmed with:

- Multi-seed replication: E021-E025
- Mix refinement around `0.35`: E026-E031
- Gamma refinement around `1.15`: E032-E036
- Ablation of optical-only and audio-only hybrid features: E037-E038

The prepared confirmation package is:

```text
twoport1024_e017_confirm_code_20260606.zip
```

The main result should be reported as stable only if the confirmation runs keep
full Gibbs accuracy in or near the 96%-97% target range.
