# VGGSound Full Audio Paper-STFT ResNet50 BM

This package removes the previous audio teacher bottleneck where the STFT was resized to `128 x 96` before ResNet50.

Pipeline:

1. Decode mp4 audio to 16 kHz mono, fixed 10 s.
2. Compute paper-style STFT with `nperseg=512`, `noverlap=353`.
   - 5 s crop gives approximately `257 x 500`.
   - Full 10 s gives approximately `257 x 1004`.
3. Save full 10 s STFT as `float16` memmap, without resizing and without sigmoid.
4. Train ResNet50 teacher:
   - training: random `257 x 500` time crop.
   - eval/export: full 10 s STFT.
5. Export BM inputs:
   - `global2048`: full 10 s ResNet50 embedding.
   - `seqmeanstd4096`: mean/std over temporal chunk embeddings.
   - `lstm4096`: BiLSTM feature from temporal chunk embeddings.
6. Train standard BM on the exported audio embeddings.

Server run:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_audio_paper_resnet50_bm_code_20260620.zip
chmod +x sbatch_vggsound_full_audio_paper_resnet50_bm.sh push_vggsound_full_audio_paper_resnet50_bm_results.sh
sbatch sbatch_vggsound_full_audio_paper_resnet50_bm.sh
```

The sbatch requests 4 GPUs, 32 CPUs, and 110 GB memory on `gpu5090`.

Expected outputs:

- `data_vggsound_full/audio_paper_stft257x1004/`
- `data_vggsound_full/features/vggsound_full_audio_paperresnet50_*`
- `runs_vggsound_full_AF028_*`
- `runs_vggsound_full_AF029_*`
- `runs_vggsound_full_AF030_*`
- `runs_vggsound_full_AF031_*`
- `vggsound_full_audio_paper_resnet50_bm_log.md`
