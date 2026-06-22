# VGGSound Full AF031 Audio BM Improvement

Goal: improve the current best audio BM result, AF031 full accuracy 44.31%.

Current reference:

- ARF004 paper-STFT ResNet50 teacher top1: 51.74%.
- ARF005 ResNet50-sequence LSTM encoder top1: 49.18%.
- AF031 BM on ARF005 LSTM4096 embedding: full 44.31%.

The gap suggests the STFT/ResNet50 teacher is good enough; the remaining loss is mostly in how teacher information is encoded into BM visible p-bits and then recovered by Gibbs label sampling.

## Experiments

### Capacity / Training Length

| ID | setup | purpose |
|---|---|---|
| AF034 | resume AF031 h6 from epoch450 to epoch650 | check if AF031 is still undertrained |
| AF035 | resume AF034 to epoch850 | longer continuation if AF034 still improves |
| AF036 | LSTM4096, h8, epoch500 | larger hidden capacity on the same AF031 feature |
| AF037 | LSTM4096, h10, epoch500 | stronger capacity test |

### Visible-Encoding Variants

| ID | setup | purpose |
|---|---|---|
| AF038 | flatten 4 x 2048 ResNet50 chunk embeddings to 8192 visible dims, h4 | less compressed sequence encoding |
| AF039 | same 8192 visible encoding, h6 | larger model on the less-compressed sequence |
| AF040 | concat full-10s global2048 and LSTM4096 to 6144 visible dims, h6 | combine global teacher information and temporal LSTM information |

These variants use ResNet50 embeddings, not teacher class logits/probabilities, so they are cleaner than directly feeding the teacher prediction distribution into BM.

## Submit

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_af031_improve_code_20260622.zip
chmod +x sbatch_vggsound_full_audio_af031_continue_capacity.sh sbatch_vggsound_full_audio_af031_encoding_variants.sh push_vggsound_full_audio_af031_improve_results.sh

sbatch sbatch_vggsound_full_audio_af031_continue_capacity.sh
sbatch sbatch_vggsound_full_audio_af031_encoding_variants.sh
```

If only one branch should run:

```bash
sbatch sbatch_vggsound_full_audio_af031_continue_capacity.sh
```

or:

```bash
sbatch sbatch_vggsound_full_audio_af031_encoding_variants.sh
```

## Upload Results

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_audio_af031_improve_results.sh
```

## Required Existing Files

```text
train_vggsound_mini20_bm.py
data_vggsound_full/features/vggsound_full_audio_paperresnet50_lstm4096_chunks4_w500_h1024_seed123.npz
data_vggsound_full/features/vggsound_full_audio_paperresnet50_seq_chunks4_w500_seed123.npz
data_vggsound_full/features/vggsound_full_audio_paperresnet50_global2048_seed123.npz
runs_vggsound_full_AF031_standard_audio_paperresnet50_lstm4096_h6_lc5_e450/last.pt
runs_vggsound_full_AF031_standard_audio_paperresnet50_lstm4096_h6_lc5_e450/history.json
```
