# EMNIST Letters + ISOLET 1024 p-bit Results Summary

This document summarizes the current 1024 p-bit EMNIST Letters + ISOLET experiments.
Final accuracy numbers use full Gibbs evaluation when available:

```text
eval_steps = 3000
eval_burn_in = 500
eval_thin = 2
label_init = random_onehot
label_update = binary for two-port BM
```

## 1. Task Definition

The current EMNIST experiment is not pure EMNIST image recognition.
It is a synthetic paired multimodal task:

```text
visual input: EMNIST Letters image
audio input: ISOLET spoken-letter feature sampled from the same class
target: 26-way letter label
```

Dataset construction:

```text
y = EMNIST image label
audio = random ISOLET sample from audio_by_class[y]
image = EMNIST image
return audio, image, y
```

This means the ISOLET audio channel carries real same-class evidence. The result should be reported as paired EMNIST+ISOLET multimodal recognition, not as EMNIST image-only recognition.

## 2. Data And Input Mapping

Full-data setting:

| Split | EMNIST rows | ISOLET rows |
|---|---:|---:|
| train | 124800 | 6237 |
| test | 20800 | 1560 |

Input preprocessing:

```text
EMNIST image: 28x28 -> mnist20_com_crop -> 400
ISOLET audio: 617 -> zscore_sigmoid -> linear interpolation -> 400
```

EMNIST 20x20 crop check:

| Split | mean retained mass | median retained mass | p05 retained mass | fraction < 0.80 |
|---|---:|---:|---:|---:|
| train | 90.97% | 91.86% | 80.59% | 4.44% |
| test | 91.01% | 91.91% | 80.60% | 4.45% |

Interpretation: the same MNIST-style `mnist20_com_crop` mapping is usable for EMNIST, but EMNIST letters lose more mass than MNIST, especially wide or descender-like letters such as `W` and `J`.

## 3. 1024 p-bit Hardware Allocation

For the two-port BM:

```text
400 visible/image p-bits
+ 130 label p-bits
+ 494 hidden p-bits
= 1024 p-bits
```

The 400 ISOLET dimensions do not consume additional p-bits. They enter as the second physical port on the visible p-bits.

Two-port p-bit field:

```text
field = X + Y + gamma * X * Y + feedback - c
p = sigmoid(2 * field)
```

For the standard BM baselines:

```text
400 input p-bits
+ 130 label p-bits
+ 494 hidden p-bits
= 1024 p-bits
```

The standard BM has only one input channel, so `image`, `audio`, or `avg(image,audio)` is supplied as a single 400-dimensional input.

## 4. Main 1024 p-bit EMNIST Results

| Run | Model | Input | Gamma | Best epoch | Full Gibbs acc |
|---|---|---|---:|---:|---:|
| L012 | two-port BM | EMNIST20 + ISOLET400 | 1.15 / 1.15 | 150 | 98.66% |
| L017 | two-port BM, gamma ablation | EMNIST20 + ISOLET400 | 0.00 / 0.00 | 110 | 94.69% |
| B008 | standard BM | EMNIST20 image only | N/A | 180 | 75.17% |
| B009 | standard BM | ISOLET400 audio only | N/A | 100 | 82.76% |
| B010 | standard BM | avg(EMNIST20, ISOLET400) | N/A | 175 | 77.44% |

Main comparison:

```text
L012 two-port BM:        98.66%
L017 gamma=0 ablation:   94.69%
standard BM image-only:  75.17%
standard BM audio-only:  82.76%
standard BM avg input:   77.44%
```

The gamma ablation shows that removing the two-port multiplicative interaction drops accuracy by:

```text
98.66% - 94.69% = 3.97 percentage points
```

## 5. Modality And Leakage Controls

L012 best checkpoint was evaluated with additional input controls.
These controls use quick Gibbs evaluation (`200/50/thin=2`) and are diagnostic rather than final headline numbers.

| Control | Accuracy | Interpretation |
|---|---:|---|
| normal paired image+audio | 98.65% | Reproduces L012 behavior |
| audio zeroed, image kept | 27.73% | The trained two-port model is weak as image-only at inference |
| image zeroed, audio kept | 79.12% | ISOLET audio alone is a strong cue |
| audio replaced by next class | 12.76% | Wrong audio severely hurts |
| audio replaced by random class | 13.97% | Random audio severely hurts |
| image replaced by next class, audio kept | 62.14% | Correct audio still partially dominates wrong image |
| both image and audio replaced by next class | 0.12% | No evidence of label-only evaluation leakage |
| scoring label shifted by next class | 0.22% | Eval does not directly use the true label to predict |

Conclusion:

- No direct label/eval leakage was found.
- The high L012 accuracy depends strongly on same-class ISOLET audio.
- L012 should be described as a paired multimodal result.

## 6. Classical Sanity Baselines

These are not p-bit BM results. They estimate how difficult the preprocessed inputs are for simple discriminative classifiers.

| Input | Classifier | Accuracy |
|---|---|---:|
| EMNIST20 image only | nearest centroid | 59.78% |
| EMNIST20 image only | ridge classifier | 60.27% |
| ISOLET400 unique audio only, current random split | nearest centroid | 87.88% |
| ISOLET400 unique audio only, current random split | ridge classifier | 92.44% |
| paired ISOLET400 audio reused to match EMNIST rows | ridge classifier | 92.45% |
| paired EMNIST20+ISOLET400 concatenation | nearest centroid | 93.65% |
| paired EMNIST20+ISOLET400 concatenation | ridge classifier | 96.67% |
| ISOLET400 approximate order split | nearest centroid | 86.53% |
| ISOLET400 approximate order split | ridge classifier | 91.47% |

Interpretation:

- EMNIST20 image-only is indeed hard.
- ISOLET400 is already highly class-informative.
- A simple concatenated ridge classifier reaches 96.67%, so the L012 two-port BM result at 98.66% is plausible for this synthetic paired task.

## 7. Comparison With MNIST/WSD 1024 p-bit Line

The MNIST/WSD 1024 experiments and the EMNIST+ISOLET 1024 experiments are related but not identical.

| Task | Main 1024 two-port result | Important caveat |
|---|---:|---|
| MNIST/WSD, processed physical input E017 family | 97.41% mean over 6 seeds | Uses processed physical input patterns |
| MNIST/WSD, raw/raw strict 1024 E001/E003 | about 94.2% | No processed input |
| EMNIST+ISOLET, raw/raw L012 | 98.66% | Strong same-class ISOLET audio modality |

Correct interpretation:

```text
EMNIST image-only is harder than MNIST image-only.
But EMNIST+ISOLET is a stronger paired multimodal task because ISOLET audio provides strong class evidence.
The 1024 p-bit two-port BM can exploit this complementary modality, so the paired multimodal accuracy can exceed the MNIST/WSD result.
```

Avoid saying:

```text
EMNIST image recognition reaches 98.66%.
```

Preferred statement:

```text
On a paired EMNIST Letters + ISOLET spoken-letter benchmark, the 1024 p-bit
two-port BM achieves 98.66% full-Gibbs accuracy. The result is primarily a
multimodal recognition result and benefits strongly from the ISOLET audio
channel.
```

## 8. Completeness Check

Core EMNIST 1024 experiments now completed:

| Item | Status | Notes |
|---|---|---|
| Full-data two-port main model | done | L012, 98.66% |
| Full-data standard BM image-only baseline | done | B008, 75.17% |
| Full-data standard BM audio-only baseline | done | B009, 82.76% |
| Full-data standard BM single-channel fusion baseline | done | B010 avg input, 77.44% |
| Gamma/high-order interaction ablation | done | L017 gamma=0, 94.69% |
| Eval-only modality ablations | done | audio/image zero and mismatch controls |
| Classical sanity baselines | done | image/audio/concat ridge and centroid |
| EMNIST 20x20 crop analysis | done | 91.01% mean test mass retained |

Remaining recommended experiments before making a strong paper-style claim:

| Priority | Missing experiment | Why it matters |
|---|---|---|
| High | L012 multi-seed runs | Current 98.66% is one seed; need robustness statistics |
| High | Use official/strict ISOLET speaker split if available | Current ISOLET split is random stratified; speaker leakage risk should be reduced |
| Medium | Retrained two-port image-only and audio-only ablations | Current image/audio ablations are mostly eval-only zeroing plus standard BM baselines |
| Medium | 1024 standard BM with concatenated 800-dim input and only 94 hidden p-bits | Alternative same-resource single-channel fusion baseline |
| Medium | EMNIST 1024 gamma sweep around 0.5, 1.0, 1.15, 1.25 | Only gamma=0 and gamma=1.15 are currently tested |
| Low | Per-class confusion matrix for L012 and B008/B009/B010 | Useful for presentation and error analysis |

Current judgment:

```text
The core EMNIST 1024 comparison is now complete enough for internal reporting.
For external/paper-style claims, the most important missing pieces are multi-seed
L012 replication and a stricter ISOLET split.
```

## 9. File Index

Main runs:

- `runs_letters_isolet_L012_twoport_rawraw_1024_20x20_linearisolet_full_e400`
- `runs_letters_isolet_L017_twoport_1024_gamma0_20x20_linearisolet_full_e220`
- `runs_letters_isolet_B008_standard_bm1024_emnist20_image_full_e180`
- `runs_letters_isolet_B009_standard_bm1024_isolet400_audio_full_e180`
- `runs_letters_isolet_B010_standard_bm1024_avg_emnist20_isolet400_full_e180`

Diagnostic scripts:

- `diagnose_twoport_letters_modalities.py`
- `diagnose_letters_isolet_classical_baselines.py`
- `run_letters_1024_missing_experiments.py`

Supporting analysis:

- `runs_letters_emnist20_crop_analysis/summary.json`
- `letters_isolet_4096_experiment_log.md`
