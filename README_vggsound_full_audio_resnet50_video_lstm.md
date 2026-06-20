# VGGSound Full Audio ResNet50 And Video LSTM BM Experiments

This package adds two follow-up experiment branches for the full VGGSound BM study.

## Audio ResNet50 branch

Goal: replace the small audio CNN feature extractor used in AF005-AF008 with a stronger ResNet50-style spectrogram encoder.

Pipeline:

1. Use the existing official-style STFT feature file:
   `data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz`.
2. Reshape each audio sample to `1 x 128 x 96`.
3. Train an ImageNet-initialized ResNet50 adapted to one spectrogram channel.
4. Export the 2048-dimensional penultimate embedding.
5. Train standard BM audio-only baselines:
   - AF013: 2048 input, hidden factor 4, 220 epochs.
   - AF014: 2048 input, hidden factor 6, 220 epochs.
   - AF015: 2048 input, hidden factor 8, 220 epochs.

The expected comparison point is AF008, where the smaller CNN embedding plus BM reached about 20.75% full eval.

## Video LSTM branch

Goal: test whether preserving frame order improves video-only BM performance beyond the previous mean/std ResNet50 aggregation.

Important interpretation:

- ResNet50 extracts spatial/object information from each video frame.
- BiLSTM processes the frame sequence over time.
- LSTM is not the spatial model; it is the temporal model for the video time dimension.

Pipeline:

1. Decode 16 RGB frames per clip at 224 x 224.
2. Extract a 2048-dimensional ResNet50 feature from each frame.
3. Store each video as a sequence `[16, 2048]`.
4. Train BiLSTM video teachers and export embeddings:
   - VLF001: 2048-dimensional LSTM embedding.
   - VLF002: 4096-dimensional LSTM embedding.
5. Train standard BM video-only baselines:
   - VF020: 2048 input, hidden factor 6, 220 epochs.
   - VF021: 4096 input, hidden factor 6, 220 epochs.
   - VF022: 4096 input, hidden factor 8, 220 epochs.

The expected comparison point is VF010, where the previous video ResNet50 mean/std feature BM reached about 37.66% full eval.

## Server commands

Unpack:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_resnet50_video_lstm_code_20260619.zip
chmod +x sbatch_vggsound_full_audio_resnet50_bm.sh
chmod +x sbatch_vggsound_full_video_lstm_bm.sh
chmod +x push_vggsound_full_audio_resnet50_video_lstm_results.sh
```

Submit audio ResNet50 branch:

```bash
sbatch sbatch_vggsound_full_audio_resnet50_bm.sh
```

Submit video LSTM branch:

```bash
sbatch sbatch_vggsound_full_video_lstm_bm.sh
```

Check jobs:

```bash
squeue
```

Upload results after jobs finish:

```bash
./push_vggsound_full_audio_resnet50_video_lstm_results.sh
```

## Notes

- The sbatch files request 4 GPUs, 16 CPU cores, and 110G memory.
- Large artifacts are intentionally not pushed to GitHub: `.npz` feature arrays and `.pt` checkpoints stay on the server.
- If ImageNet ResNet50 weights cannot be downloaded on the server, rerun the audio branch with `--no_pretrained` added inside `sbatch_vggsound_full_audio_resnet50_bm.sh`, but pretrained weights are preferred.
