# VGGSound Backbone Upgrade

This package tests whether the current AV016 feature pipeline is limited by the ResNet50 backbone.

The final BM-facing representation stays compatible with AV016-style experiments:

```text
CNN backbone -> temporal LSTM encoder -> 4096-d embedding -> standard BM screening
```

Only if a unimodal full Gibbs result clearly beats the current control should the feature be promoted to two-port BM:

```text
video-only control: VF026 = 42.84%
audio-only control: AF036 = 44.98%
current two-port main result: AV016 = 57.86%
```

## Experiments

- `P2V001`: video frames -> ImageNet EfficientNet-B3 -> LSTM4096 -> standard video BM.
- `P2A001`: paper-STFT audio -> ImageNet EfficientNet-B3 adapted to one-channel spectrograms -> LSTM4096 -> standard audio BM.

Both use EfficientNet-B3 as the first non-ResNet50 backbone test. The scripts also support `resnet101`, `wide_resnet50_2`, and `efficientnet_b0` if a lighter/fallback run is needed.

## Server Commands

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_backbone_upgrade_code_20260624.zip

chmod +x sbatch_vggsound_backbone_upgrade_video.sh
chmod +x sbatch_vggsound_backbone_upgrade_audio.sh
chmod +x push_vggsound_backbone_upgrade_results.sh

mkdir -p logs
sbatch sbatch_vggsound_backbone_upgrade_video.sh
sbatch sbatch_vggsound_backbone_upgrade_audio.sh
```

After completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_backbone_upgrade_results.sh
```
